"""The literature graph: relations, clusters, and explainable closest-prior-work ranking.

No embeddings and no network: similarity is token overlap with a stated score, so a researcher can
always ask *why* a paper was called "closest prior work" and get an answer they can check.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models.common import RelationType
from ..models.literature import (
    LiteraturePaper,
    LiteratureRelation,
    NodeKind,
)

#: Relation types that mean "this work is genuinely close", used to boost ranking.
CLOSENESS_BOOST: dict[RelationType, float] = {
    RelationType.SAME_MECHANISM: 0.30,
    RelationType.SAME_PROBLEM: 0.22,
    RelationType.EXTENDS: 0.12,
    RelationType.SIMILAR_MECHANISM: 0.12,
    RelationType.CONTRADICTS: 0.08,
    RelationType.SAME_METRIC: 0.04,
    RelationType.USES_SAME_DATASET: 0.04,
    RelationType.DIFFERENT_ASSUMPTION: 0.02,
}

_WORD = re.compile(r"[a-z0-9]+")

#: Words that carry no discriminative signal in paper titles/abstracts.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "without", "is",
        "are", "we", "our", "this", "that", "these", "those", "it", "its", "as", "at", "by",
        "from", "can", "be", "does", "do", "using", "use", "used", "via", "than", "then", "more",
        "less", "new", "novel", "paper", "method", "methods", "approach", "approaches", "results",
        "show", "shows", "study", "studies", "we",
    }
)


def tokens(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in STOPWORDS and len(word) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LiteratureGraph:
    """Edges between papers, methods, mechanisms, datasets, experiments and the current work."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ------------------------------------------------------------------ edges

    def add_relation(
        self,
        principal: Principal,
        *,
        from_kind: NodeKind | str,
        from_id: str,
        to_kind: NodeKind | str,
        to_id: str,
        relation: RelationType | str,
        confidence: float = 0.5,
        evidence: Sequence[str] = (),
        note: str = "",
    ) -> LiteratureRelation:
        principal.require(Cap.LITERATURE_WRITE, "literature.relation")
        edge = LiteratureRelation(
            from_kind=NodeKind(from_kind) if isinstance(from_kind, str) else from_kind,
            from_id=from_id,
            to_kind=NodeKind(to_kind) if isinstance(to_kind, str) else to_kind,
            to_id=to_id,
            relation=RelationType(relation) if isinstance(relation, str) else relation,
            confidence=confidence,
            evidence=list(evidence),
            note=note,
            created_by=principal.name,
        )
        self.kernel.relations.save(edge)
        self.kernel.events.append(
            "literature.relation",
            actor=principal.name,
            payload={
                "relation_id": edge.relation_id,
                "from": f"{edge.from_kind.value}:{from_id}",
                "to": f"{edge.to_kind.value}:{to_id}",
                "relation": edge.relation.value,
                "confidence": confidence,
            },
        )
        return edge

    def edges(self) -> list[LiteratureRelation]:
        return sorted(self.kernel.relations.all(), key=lambda e: e.relation_id)

    def neighbours(
        self,
        node_id: str,
        *,
        relation: RelationType | None = None,
        direction: str = "both",
    ) -> list[str]:
        out: list[str] = []
        for edge in self.edges():
            if relation is not None and edge.relation is not relation:
                continue
            if direction in ("out", "both") and edge.from_id == node_id:
                out.append(edge.to_id)
            if direction in ("in", "both") and edge.to_id == node_id:
                out.append(edge.from_id)
        return sorted(set(out))

    def clusters(self) -> list[list[str]]:
        """Connected components over all edges (union-find), deterministically ordered."""
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # deterministic union: the lexicographically smaller root wins
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        for edge in self.edges():
            union(edge.from_id, edge.to_id)
        groups: dict[str, list[str]] = {}
        for node in parent:
            groups.setdefault(find(node), []).append(node)
        return [sorted(members) for _root, members in sorted(groups.items())]

    # ------------------------------------------------------------------ prior work

    def _closeness_edges(self, paper_ids: Iterable[str]) -> dict[str, float]:
        """Boost for each paper that participates in a 'this work is close' relation."""
        boosts: dict[str, float] = {}
        wanted = set(paper_ids)
        for edge in self.edges():
            weight = CLOSENESS_BOOST.get(edge.relation, 0.0)
            if not weight:
                continue
            for endpoint in (edge.from_id, edge.to_id):
                if endpoint in wanted:
                    boosts[endpoint] = max(boosts.get(endpoint, 0.0), weight)
        return boosts

    def closest_prior_work(
        self, current_work_claim: str, *, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Rank papers by token overlap with the claim, boosted by declared relations.

        Returns ``(paper_id, score)`` sorted by score then id, so the answer is reproducible and the
        *reason* for a ranking can be shown.
        """
        claim_tokens = tokens(current_work_claim)
        papers = self.kernel.papers.all()
        boosts = self._closeness_edges(p.paper_id for p in papers)
        scored: list[tuple[str, float]] = []
        for paper in papers:
            text = " ".join(filter(None, [paper.title, paper.abstract or "", " ".join(paper.main_findings)]))
            base = jaccard(claim_tokens, tokens(text))
            title_only = jaccard(claim_tokens, tokens(paper.title))
            score = round(min(1.0, 0.6 * base + 0.4 * title_only + boosts.get(paper.paper_id, 0.0)), 4)
            scored.append((paper.paper_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]

    def explain_closeness(self, paper_id: str, current_work_claim: str) -> dict[str, object]:
        """Show the components of a closeness score, so the ranking is auditable."""
        paper = self.kernel.papers.get(paper_id)
        if paper is None:
            return {"paper_id": paper_id, "error": "not found"}
        claim_tokens = tokens(current_work_claim)
        text = " ".join(filter(None, [paper.title, paper.abstract or ""]))
        shared = sorted(claim_tokens & tokens(text))
        relations = [
            e.relation.value
            for e in self.edges()
            if paper_id in (e.from_id, e.to_id)
        ]
        return {
            "paper_id": paper_id,
            "title": paper.title,
            "shared_terms": shared,
            "jaccard": round(jaccard(claim_tokens, tokens(text)), 4),
            "relations": sorted(set(relations)),
            "boost": round(self._closeness_edges([paper_id]).get(paper_id, 0.0), 4),
        }

    # ------------------------------------------------------------------ reporting

    def coverage_gaps(self) -> list[str]:
        """Structural gaps in the map: papers with no relation, mechanisms with a single member."""
        gaps: list[str] = []
        connected = {e.from_id for e in self.edges()} | {e.to_id for e in self.edges()}
        isolated = [p.paper_id for p in self.kernel.papers.all() if p.paper_id not in connected]
        if isolated:
            gaps.append(f"{len(isolated)} paper(s) have no relation to anything else: {isolated[:5]}")
        mechanism_members: dict[str, set[str]] = {}
        for edge in self.edges():
            if edge.relation in (RelationType.SAME_MECHANISM, RelationType.SIMILAR_MECHANISM):
                mechanism_members.setdefault(edge.relation.value, set()).update(
                    {edge.from_id, edge.to_id}
                )
        for relation, members in sorted(mechanism_members.items()):
            if len(members) == 1:
                gaps.append(f"{relation} edge involves only one work: {sorted(members)}")
        if not self.edges():
            gaps.append("no literature relations recorded: the prior-art map is empty")
        return gaps

    def summary(self) -> dict[str, int]:
        by_relation: dict[str, int] = {}
        for edge in self.edges():
            by_relation[edge.relation.value] = by_relation.get(edge.relation.value, 0) + 1
        return {
            "papers": len(self.kernel.papers.all()),
            "relations": len(self.edges()),
            "clusters": len(self.clusters()),
            "literature_claims": len(self.kernel.literature_claims.all()),
            "fulltext_verified": sum(
                1 for p in self.kernel.papers.all() if p.supports_mechanism_claims()
            ),
            **{f"rel:{k}": v for k, v in sorted(by_relation.items())},
        }
