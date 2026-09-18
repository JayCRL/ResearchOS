"""Novelty audits — the one place allowed to say "we found no prior work", and only when earned.

The verdict is a function of *measured coverage*, not of how the search felt:

* insufficient coverage → ``NOVELTY_UNCERTAIN``, always, no matter how few matches were found;
* matches found → ``SIMILAR_PRIOR_WORK_FOUND``;
* no matches *and* coverage above every threshold → ``NO_MATCH_FOUND_IN_SEARCHED_COVERAGE``.

The phrase "nobody has done this" is never produced. Even the strongest verdict is scoped to the
coverage that was actually searched, and that scoping is part of the returned sentence.
"""

from __future__ import annotations

from typing import Sequence

from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models.common import GapKind, NoveltyVerdict, Severity
from ..models.literature import (
    FORBIDDEN_NOVELTY_PHRASES,
    LiteratureCoverage,
    LiteraturePaper,
    NoveltyAudit,
    NoveltyMatch,
    QueryPlan,
)
from ..models.timeline import TimelineEventKind
from .coverage import compute_coverage, coverage_report
from .graph import LiteratureGraph, STOPWORDS, jaccard, tokens

#: Similarity above which a paper counts as a match worth reporting to the author.
MATCH_THRESHOLD = 0.22
#: Similarity above which the match is treated as a serious threat to the novelty claim.
THREAT_THRESHOLD = 0.35

SIMILARITY_KINDS: tuple[str, ...] = (
    "exact_prior_art",
    "terminology_variant",
    "mechanistic_equivalent",
    "functional_equivalent",
    "historical_predecessor",
    "citation_neighbor",
    "recent_work",
)


def build_match(
    paper: LiteraturePaper,
    *,
    similarity_kind: str,
    overlap: str,
    difference: str,
    threat: Severity = Severity.MEDIUM,
    evidence_ids: Sequence[str] = (),
) -> NoveltyMatch:
    if similarity_kind not in SIMILARITY_KINDS:
        raise ValueError(f"unknown similarity kind {similarity_kind!r}; expected one of {SIMILARITY_KINDS}")
    return NoveltyMatch(
        paper_id=paper.paper_id,
        title=paper.title,
        similarity_kind=similarity_kind,
        overlap=overlap,
        difference=difference,
        threat=threat,
        evidence_ids=list(evidence_ids),
    )


class NoveltyAuditor:
    """Runs a novelty audit against the recorded literature, with coverage as a hard gate."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ------------------------------------------------------------------ phrases

    @staticmethod
    def scan_forbidden_phrases(text: str) -> list[str]:
        lowered = text.lower()
        return [phrase for phrase in FORBIDDEN_NOVELTY_PHRASES if phrase in lowered]

    # ------------------------------------------------------------------ verdict

    def verdict(
        self, coverage: LiteratureCoverage, matches: Sequence[NoveltyMatch]
    ) -> tuple[NoveltyVerdict, str]:
        deficits = coverage.deficits()
        if matches:
            threats = [m for m in matches if m.threat in (Severity.HIGH, Severity.BLOCKER)]
            return (
                NoveltyVerdict.SIMILAR_PRIOR_WORK_FOUND,
                f"{len(matches)} similar work(s) found ({len(threats)} high-threat): "
                + "; ".join(m.title[:60] for m in matches[:3]),
            )
        if deficits:
            return (
                NoveltyVerdict.NOVELTY_UNCERTAIN,
                "no match found, but the search coverage is insufficient to support that as a "
                "conclusion: " + "; ".join(deficits),
            )
        return (
            NoveltyVerdict.NO_MATCH_FOUND_IN_SEARCHED_COVERAGE,
            "no matching work found across the searched coverage: " + coverage.basis,
        )

    # ------------------------------------------------------------------ search

    def closest_prior_work(
        self, target_statement: str, *, top_k: int = 5
    ) -> list[NoveltyMatch]:
        """Deterministic, explainable matches from token overlap over title + abstract + findings."""
        target_tokens = tokens(target_statement)
        edges = LiteratureGraph(self.kernel)
        relation_of: dict[str, list[str]] = {}
        for edge in edges.edges():
            relation_of.setdefault(edge.from_id, []).append(edge.relation.value)
            relation_of.setdefault(edge.to_id, []).append(edge.relation.value)

        matches: list[NoveltyMatch] = []
        for paper in self.kernel.papers.all():
            text = " ".join(
                filter(None, [paper.title, paper.abstract or "", " ".join(paper.main_findings)])
            )
            score = jaccard(target_tokens, tokens(text))
            if score < MATCH_THRESHOLD:
                continue
            shared = sorted(target_tokens & tokens(text))[:8]
            relations = set(relation_of.get(paper.paper_id, []))
            if "same_mechanism" in relations or "similar_mechanism" in relations:
                kind = "mechanistic_equivalent"
            elif "same_problem" in relations:
                kind = "functional_equivalent"
            elif paper.year and paper.year < 2000:
                kind = "historical_predecessor"
            else:
                kind = "terminology_variant"
            matches.append(
                build_match(
                    paper,
                    similarity_kind=kind,
                    overlap="shared terms: " + ", ".join(shared),
                    difference="not established from the abstract alone" if not paper.supports_mechanism_claims() else "",
                    threat=Severity.HIGH if score >= THREAT_THRESHOLD else Severity.MEDIUM,
                )
            )
        matches.sort(key=lambda m: (-(len(tokens(m.overlap)) + (1 if m.threat is Severity.HIGH else 0)), m.paper_id))
        return matches[:top_k]

    def audit(
        self,
        principal: Principal,
        target_statement: str,
        *,
        plan: QueryPlan | None = None,
        task_id: str | None = None,
        provider=None,
    ) -> NoveltyAudit:
        principal.require(Cap.LITERATURE_NOVELTY_AUDIT, "novelty.audit")

        papers = self.kernel.papers.all()
        if plan is not None:
            coverage = compute_coverage(plan, papers)
        else:
            # No plan means no search happened in this audit: coverage is whatever is recorded.
            from ..models.literature import LiteratureState  # noqa: F401  (documentation anchor)

            state = self.kernel.research_state()
            record = state.literature_state
            coverage = LiteratureCoverage(
                queries_executed=record.queries_executed,
                families_covered=[],
                missing_required_families=list(self._missing_families(record.terminology_coverage)),
                providers_used=[],
                papers_screened=record.papers_screened,
                papers_fulltext_checked=record.fulltext_verified,
                score=record.coverage_score,
                basis=record.coverage_basis,
            )
            coverage.compute_score()

        matches = self.closest_prior_work(target_statement)
        verdict, reason = self.verdict(coverage, matches)
        forbidden = self.scan_forbidden_phrases(target_statement)

        gap_kind = GapKind.CANDIDATE_GAP
        if verdict is NoveltyVerdict.NO_MATCH_FOUND_IN_SEARCHED_COVERAGE and coverage.meets_threshold():
            gap_kind = GapKind.CANDIDATE_GAP  # still not VERIFIED_NOVELTY: that needs review
        if matches and any(m.threat in (Severity.HIGH, Severity.BLOCKER) for m in matches):
            gap_kind = GapKind.KNOWN

        audit = NoveltyAudit(
            target_statement=target_statement,
            query_plan_id=plan.query_plan_id if plan else None,
            coverage=coverage,
            verdict=verdict,
            verdict_reason=reason,
            matches=matches,
            closest_prior_work=[m.paper_id for m in matches],
            gap_kind=gap_kind,
            forbidden_phrases_detected=forbidden,
            created_by=principal.name,
            task_id=task_id,
            notes=coverage_report(coverage),
        )
        self.kernel.novelty_audits.save(audit)
        self.kernel.events.append(
            "novelty.audit",
            actor=principal.name,
            task_id=task_id,
            payload={
                "novelty_audit_id": audit.novelty_audit_id,
                "verdict": verdict.value,
                "gap_kind": gap_kind.value,
                "coverage_score": coverage.score,
                "matches": len(matches),
                "coverage_sufficient": coverage.meets_threshold(),
                "forbidden_phrases": forbidden,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.LITERATURE,
            f"Novelty audit: {verdict.value}",
            detail=reason,
            actor=principal.name,
            task_id=task_id,
            refs=[audit.novelty_audit_id],
            state_revision=self.kernel.state.revision(),
        )
        if forbidden:
            self.kernel.events.append(
                "novelty.forbidden_phrase",
                actor=principal.name,
                task_id=task_id,
                payload={"novelty_audit_id": audit.novelty_audit_id, "phrases": forbidden},
            )
        return audit

    @staticmethod
    def _missing_families(terminology_coverage: dict[str, int]) -> list[str]:
        from ..models.literature import REQUIRED_FAMILIES

        return [family.value for family in REQUIRED_FAMILIES if terminology_coverage.get(family.value, 0) == 0]


def safe_novelty_sentence(audit: NoveltyAudit) -> str:
    """The sentence a paper may print, with the coverage caveat baked in."""
    sentence = audit.verdict_sentence()
    forbidden = NoveltyAuditor.scan_forbidden_phrases(sentence)
    if forbidden:
        # Defensive: even our own phrasing must never contain an unqualified novelty claim.
        return (
            "Within the searched coverage we found no work implementing this exact combination; "
            "we treat novelty as an open question rather than a claim."
        )
    return sentence
