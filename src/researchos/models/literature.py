"""Literature objects: papers, relations, query plans, coverage, novelty audits, prior-art matrices.

Two rules run through this whole module:

1. **You cannot pretend to know a paper's full text.** Fields that require the paper (mechanism,
   components, memory state, selection mechanism, writeback, optimization, page references) are
   rejected unless ``fulltext_status == FULLTEXT_VERIFIED``.
2. **Absence of evidence is not evidence of absence.** Prior-art cells are three-valued and
   ``UNKNOWN`` is never silently collapsed into ``FALSE``; novelty verdicts are gated on measured
   search coverage.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from .common import (
    FulltextStatus,
    GapKind,
    NoveltyVerdict,
    RelationType,
    RosModel,
    Severity,
    SourceRef,
    StrEnum,
    Tri,
    new_id,
    utcnow,
)

# --------------------------------------------------------------------------------------
# Papers
# --------------------------------------------------------------------------------------


class ProviderKind(StrEnum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENREVIEW = "openreview"
    CROSSREF = "crossref"
    GITHUB = "github"
    WEB = "web"
    LOCAL_BIB = "local_bib"
    CACHE = "cache"
    MANUAL = "manual"


class LiteraturePaper(RosModel):
    """A paper we know something about — with an honest statement of *how much* we know."""

    paper_id: str = Field(default_factory=lambda: new_id("literature_paper"))
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1500, le=2200)
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    abstract: str | None = None
    bibtex_key: str | None = None
    citation_count: int | None = Field(default=None, ge=0)
    open_access_pdf: str | None = None
    license: str | None = None

    fulltext_status: FulltextStatus = FulltextStatus.UNKNOWN
    fulltext_path: str | None = Field(
        default=None, description="Local path to the retrieved full text, when we actually have it."
    )
    source_pages: list[str] = Field(
        default_factory=list, description="Which pages/sections were actually read."
    )
    page_offset_basis: str | None = Field(
        default=None, description="e.g. 'pdf page numbers', 'proceedings page numbers'."
    )

    # --- interpretation fields: require real full text ---------------------------------
    research_problem: str | None = None
    main_idea: str | None = None
    mechanism: str | None = None
    explicit_components: list[str] = Field(default_factory=list)
    implicit_components: list[str] = Field(default_factory=list)
    memory_state: str | None = None
    selection_mechanism: str | None = None
    writeback: str | None = None
    optimization: str | None = None

    # --- abstract-level fields: allowed from metadata ----------------------------------
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    scale: str | None = None
    main_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    relationship_to_current_work: str | None = None
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    screened: bool = False
    include: bool = Field(default=False, description="Human decision on inclusion in related work.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    providers: list[ProviderKind] = Field(default_factory=list)
    provider_ids: dict[str, str] = Field(default_factory=dict)
    retrieved_at: datetime | None = None
    raw_record: dict[str, object] = Field(
        default_factory=dict, description="Untouched provider payload, for auditing our parsing."
    )
    provenance_refs: list[SourceRef] = Field(default_factory=list)
    imported_from: str | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_fulltext_no_pretending(self) -> "LiteraturePaper":
        if self.fulltext_status is not FulltextStatus.FULLTEXT_VERIFIED:
            offenders: list[str] = []
            for name in (
                "mechanism",
                "memory_state",
                "selection_mechanism",
                "writeback",
                "optimization",
            ):
                if getattr(self, name):
                    offenders.append(name)
            for name in ("explicit_components", "implicit_components"):
                if getattr(self, name):
                    offenders.append(name)
            if self.source_pages:
                offenders.append("source_pages")
            if offenders:
                raise ValueError(
                    f"paper {self.paper_id}: fulltext_status={self.fulltext_status.value} but "
                    f"fulltext-only fields are populated: {sorted(offenders)}. "
                    "Either verify the full text or drop those fields — abstract-level inference is "
                    "not full-text knowledge."
                )
        if self.fulltext_status is FulltextStatus.FULLTEXT_VERIFIED and not (
            self.fulltext_path or self.source_pages
        ):
            raise ValueError(
                f"paper {self.paper_id}: FULLTEXT_VERIFIED requires fulltext_path or source_pages as proof"
            )
        return self

    def supports_mechanism_claims(self) -> bool:
        return self.fulltext_status is FulltextStatus.FULLTEXT_VERIFIED

    def citation_label(self) -> str:
        first = self.authors[0].split()[-1] if self.authors else "Anon"
        return f"{first} et al., {self.year}" if len(self.authors) > 1 else f"{first}, {self.year}"

    def identity_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        return f"title:{self.title.lower().strip()}"


class LiteratureClaimKind(StrEnum):
    PROBLEM = "PROBLEM"
    METHOD = "METHOD"
    MECHANISM = "MECHANISM"
    RESULT = "RESULT"
    LIMITATION = "LIMITATION"
    RELATION = "RELATION"
    DATASET = "DATASET"
    METRIC = "METRIC"


class LiteratureClaim(RosModel):
    """A claim *about a paper*. Invariant 5: every one of these has a source or it does not exist."""

    literature_claim_id: str = Field(default_factory=lambda: new_id("literature_claim"))
    paper_id: str
    statement: str = Field(min_length=1)
    kind: LiteratureClaimKind = LiteratureClaimKind.RESULT

    page: str | None = None
    section: str | None = None
    quote: str | None = Field(default=None, max_length=2000)
    figure_table: str | None = None
    fulltext_verified: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    supports_current_work: bool | None = None
    contradicts_current_work: bool | None = None
    threat_to_novelty: Severity = Severity.INFO
    verified_by: str | None = None
    verified_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "literature"

    @model_validator(mode="after")
    def _needs_source(self) -> "LiteratureClaim":
        if not (self.page or self.section or self.quote or self.figure_table):
            raise ValueError(
                f"literature claim {self.literature_claim_id}: no source. A literature claim needs a "
                "page, section, quote or figure/table locator (invariant 5)."
            )
        if self.kind is LiteratureClaimKind.MECHANISM and not self.fulltext_verified:
            raise ValueError(
                f"literature claim {self.literature_claim_id}: MECHANISM claims about another paper "
                "require fulltext_verified=True — abstracts do not describe mechanisms reliably."
            )
        if self.fulltext_verified and not (self.page or self.section):
            raise ValueError(
                f"literature claim {self.literature_claim_id}: fulltext_verified requires a page or "
                "section locator"
            )
        return self

    def locator(self) -> str:
        parts = [p for p in (self.section, f"p.{self.page}" if self.page else None) if p]
        return " ".join(parts) or "locator-missing"


class NodeKind(StrEnum):
    PAPER = "paper"
    METHOD = "method"
    MECHANISM = "mechanism"
    CLAIM = "claim"
    DATASET = "dataset"
    EXPERIMENT = "experiment"
    CURRENT_WORK = "current_work"


class LiteratureRelation(RosModel):
    relation_id: str = Field(default_factory=lambda: new_id("literature_relation"))
    from_kind: NodeKind
    from_id: str
    to_kind: NodeKind
    to_id: str
    relation: RelationType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, description="Literature-claim ids backing the edge.")
    note: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "literature"

    @model_validator(mode="after")
    def _no_self_edge(self) -> "LiteratureRelation":
        if self.from_kind is self.to_kind and self.from_id == self.to_id:
            raise ValueError(f"relation {self.relation_id}: self-edge")
        return self


# --------------------------------------------------------------------------------------
# Search planning
# --------------------------------------------------------------------------------------


class QueryFamily(StrEnum):
    """The query families a broad search must cover. One keyword is never enough."""

    EXACT_TERMINOLOGY = "exact_terminology"
    SYNONYMS = "synonyms"
    HISTORICAL_TERMINOLOGY = "historical_terminology"
    MECHANISTIC_EQUIVALENTS = "mechanistic_equivalents"
    FUNCTIONAL_EQUIVALENTS = "functional_equivalents"
    NEIGHBORING_COMMUNITIES = "neighboring_communities"
    RECENT_TERMINOLOGY = "recent_terminology"
    CITATION_EXPANSION = "citation_expansion"
    REFERENCE_EXPANSION = "reference_expansion"
    SEMANTIC_NEIGHBOR = "semantic_neighbor"
    AUTHOR_VENUE = "author_venue"


#: Families that must be exercised before a "no match found" verdict is even expressible.
REQUIRED_FAMILIES: tuple[QueryFamily, ...] = (
    QueryFamily.EXACT_TERMINOLOGY,
    QueryFamily.SYNONYMS,
    QueryFamily.MECHANISTIC_EQUIVALENTS,
    QueryFamily.FUNCTIONAL_EQUIVALENTS,
    QueryFamily.HISTORICAL_TERMINOLOGY,
    QueryFamily.RECENT_TERMINOLOGY,
)

#: Minimum coverage for ``NO_MATCH_FOUND_IN_SEARCHED_COVERAGE`` to be a legal verdict.
COVERAGE_THRESHOLDS: dict[str, int] = {
    "min_queries": 8,
    "min_families": 4,
    "min_providers": 2,
    "min_papers_screened": 15,
}

#: Phrases that are forbidden unless novelty has been verified with full coverage.
FORBIDDEN_NOVELTY_PHRASES: tuple[str, ...] = (
    "nobody has",
    "no one has",
    "no-one has",
    "nobody has done this",
    "no one has done this",
    "nobody has ever",
    "we are the first",
    "the first to",
    "first work to",
    "completely novel",
    "entirely novel",
    "unprecedented",
    "no prior work",
    "never been attempted",
    "never been studied",
    "no existing work",
)


class PlannedQuery(RosModel):
    query_id: str = Field(default_factory=lambda: new_id("query"))
    family: QueryFamily = QueryFamily.EXACT_TERMINOLOGY
    text: str = Field(min_length=1)
    rationale: str = ""
    providers: list[ProviderKind] = Field(default_factory=list)
    executed: bool = False
    executed_at: datetime | None = None
    result_count: int = Field(default=0, ge=0)
    new_paper_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class QueryPlan(RosModel):
    """A recorded, auditable search plan. The LLM may *expand* queries; it never grades coverage."""

    query_plan_id: str = Field(default_factory=lambda: new_id("query_plan"))
    target: str = Field(min_length=1, description="The question / mechanism / claim being searched for.")
    target_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "literature"
    seed_paper_ids: list[str] = Field(default_factory=list)
    queries: list[PlannedQuery] = Field(default_factory=list)
    providers: list[ProviderKind] = Field(default_factory=list)
    expansion_rounds: int = Field(default=0, ge=0)
    max_expansion_rounds: int = Field(default=2, ge=0)
    notes: list[str] = Field(default_factory=list)
    llm_used_for_expansion: bool = False

    def families_covered(self) -> list[QueryFamily]:
        return sorted({q.family for q in self.queries if q.executed}, key=lambda f: f.value)

    def executed_queries(self) -> list[PlannedQuery]:
        return [q for q in self.queries if q.executed]

    def unexecuted(self) -> list[PlannedQuery]:
        return [q for q in self.queries if not q.executed]

    def missing_required_families(self) -> list[QueryFamily]:
        covered = set(self.families_covered())
        return [f for f in REQUIRED_FAMILIES if f not in covered]


class ProviderRecord(RosModel):
    """Raw provider output, normalised but not interpreted."""

    provider: ProviderKind
    provider_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    references: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    open_access_pdf: str | None = None
    license: str | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    raw: dict[str, object] = Field(default_factory=dict)
    query: str | None = None
    score: float | None = None

    def identity_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        return f"title:{self.title.lower().strip()}"


class LiteratureCoverage(RosModel):
    """Measured search coverage — the number that licenses (or forbids) a novelty verdict."""

    queries_executed: int = Field(default=0, ge=0)
    queries_by_family: dict[str, int] = Field(default_factory=dict)
    families_covered: list[QueryFamily] = Field(default_factory=list)
    missing_required_families: list[QueryFamily] = Field(default_factory=list)
    providers_used: list[ProviderKind] = Field(default_factory=list)
    papers_screened: int = Field(default=0, ge=0)
    papers_fulltext_checked: int = Field(default=0, ge=0)
    seed_papers: int = Field(default=0, ge=0)
    expansion_rounds: int = Field(default=0, ge=0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    basis: str = ""
    search_log_refs: list[str] = Field(default_factory=list)

    def deficits(self) -> list[str]:
        t = COVERAGE_THRESHOLDS
        out: list[str] = []
        if self.queries_executed < t["min_queries"]:
            out.append(f"queries_executed {self.queries_executed} < {t['min_queries']}")
        if len(self.families_covered) < t["min_families"]:
            out.append(f"families_covered {len(self.families_covered)} < {t['min_families']}")
        if len(self.providers_used) < t["min_providers"]:
            out.append(f"providers_used {len(self.providers_used)} < {t['min_providers']}")
        if self.papers_screened < t["min_papers_screened"]:
            out.append(f"papers_screened {self.papers_screened} < {t['min_papers_screened']}")
        if self.missing_required_families:
            out.append(
                "missing required query families: "
                + ", ".join(f.value for f in self.missing_required_families)
            )
        return out

    def meets_threshold(self) -> bool:
        return not self.deficits()

    def compute_score(self) -> float:
        t = COVERAGE_THRESHOLDS
        parts = [
            min(1.0, self.queries_executed / t["min_queries"]),
            min(1.0, len(self.families_covered) / max(1, t["min_families"])),
            min(1.0, len(self.providers_used) / t["min_providers"]),
            min(1.0, self.papers_screened / t["min_papers_screened"]),
        ]
        self.score = round(sum(parts) / len(parts), 4)
        self.basis = (
            f"{self.queries_executed} queries across {len(self.families_covered)} families via "
            f"{len(self.providers_used)} providers; {self.papers_screened} papers screened; "
            + ("meets thresholds" if self.meets_threshold() else "below thresholds: " + "; ".join(self.deficits()))
        )
        return self.score


class NoveltyMatch(RosModel):
    paper_id: str
    title: str = ""
    similarity_kind: str = Field(
        default="same_problem",
        description="exact_prior_art | terminology_variant | mechanistic_equivalent | "
        "functional_equivalent | historical_predecessor | citation_neighbor | recent_work",
    )
    overlap: str = ""
    difference: str = ""
    threat: Severity = Severity.MEDIUM
    evidence_ids: list[str] = Field(default_factory=list)


class NoveltyAudit(RosModel):
    """The *only* object allowed to carry a novelty verdict, and it is coverage-gated."""

    novelty_audit_id: str = Field(default_factory=lambda: new_id("novelty_audit"))
    target_statement: str = Field(min_length=1)
    query_plan_id: str | None = None
    coverage: LiteratureCoverage = Field(default_factory=LiteratureCoverage)

    verdict: NoveltyVerdict = NoveltyVerdict.NOVELTY_UNCERTAIN
    verdict_reason: str = ""
    matches: list[NoveltyMatch] = Field(default_factory=list)
    closest_prior_work: list[str] = Field(default_factory=list)
    gap_kind: GapKind = GapKind.CANDIDATE_GAP

    forbidden_phrases_detected: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "novelty_auditor"
    task_id: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coverage_gate(self) -> "NoveltyAudit":
        if self.verdict is NoveltyVerdict.NO_MATCH_FOUND_IN_SEARCHED_COVERAGE:
            deficits = self.coverage.deficits()
            if deficits:
                raise ValueError(
                    "cannot return NO_MATCH_FOUND_IN_SEARCHED_COVERAGE with insufficient coverage: "
                    + "; ".join(deficits)
                    + " — the verdict must be NOVELTY_UNCERTAIN"
                )
        if self.verdict is NoveltyVerdict.SIMILAR_PRIOR_WORK_FOUND and not self.matches:
            raise ValueError("SIMILAR_PRIOR_WORK_FOUND requires at least one match")
        if self.gap_kind is GapKind.VERIFIED_NOVELTY:
            if self.verdict is not NoveltyVerdict.NO_MATCH_FOUND_IN_SEARCHED_COVERAGE:
                raise ValueError(
                    "VERIFIED_NOVELTY requires a NO_MATCH_FOUND_IN_SEARCHED_COVERAGE verdict with "
                    "satisfied coverage — CANDIDATE_GAP is not VERIFIED_NOVELTY"
                )
        return self

    def coverage_is_sufficient(self) -> bool:
        return self.coverage.meets_threshold()

    def verdict_sentence(self) -> str:
        """A sentence a paper may safely use, matched to the coverage we actually have."""
        if self.verdict is NoveltyVerdict.NO_MATCH_FOUND_IN_SEARCHED_COVERAGE:
            return (
                "Within the searched coverage ("
                + self.coverage.basis
                + "), we found no prior work implementing this exact combination."
            )
        if self.verdict is NoveltyVerdict.SIMILAR_PRIOR_WORK_FOUND:
            return (
                "Prior work addresses closely related problems; we detail the specific differences in "
                "Section~\\ref{sec:related}."
            )
        return (
            "Our search did not cover this space densely enough to support a novelty claim; we treat "
            "the distinction from prior work as an open question."
        )


# --------------------------------------------------------------------------------------
# Prior-art matrix
# --------------------------------------------------------------------------------------


class MatrixColumn(RosModel):
    key: str = Field(min_length=1, description="Machine key, e.g. 'fast_state'.")
    label: str = Field(min_length=1)
    question: str = Field(default="", description="The precise question a cell answers.")
    requires_fulltext: bool = Field(
        default=True,
        description="True when a TRUE/FALSE answer is only meaningful with the paper's full text.",
    )


class MatrixCell(RosModel):
    """One three-valued cell. ``UNKNOWN`` is a legitimate, reportable answer."""

    value: Tri = Tri.UNKNOWN
    paper_id: str | None = None
    page: str | None = None
    section: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    quote: str | None = None
    note: str | None = None
    literature_claim_id: str | None = None
    explicit_absence_evidence: bool = Field(
        default=False,
        description="Set only when the source explicitly states the feature is absent — the one "
        "legitimate path from UNKNOWN to FALSE.",
    )

    @model_validator(mode="after")
    def _check(self) -> "MatrixCell":
        if self.value in (Tri.TRUE, Tri.FALSE) and self.paper_id is None and not self.note:
            raise ValueError(
                "a TRUE/FALSE cell needs the paper it came from (or, for the current-work row, a note)"
            )
        if self.value is Tri.FALSE and not self.explicit_absence_evidence and not self.note:
            raise ValueError(
                "a FALSE cell needs either explicit absence evidence or an explanatory note "
                "(absence of description is weaker than documented absence)"
            )
        return self


class MatrixRow(RosModel):
    work_ref: str = Field(description="paper_id, or 'current_work'.")
    work_label: str
    cells: dict[str, MatrixCell] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    is_current_work: bool = False


class PriorArtMatrix(RosModel):
    """The comparison table, with honest coverage accounting."""

    matrix_id: str = Field(default_factory=lambda: new_id("prior_art_matrix"))
    title: str = Field(min_length=1)
    columns: list[MatrixColumn] = Field(min_length=1)
    rows: list[MatrixRow] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "literature"
    notes: list[str] = Field(default_factory=list)

    def cell(self, work_ref: str, column_key: str) -> MatrixCell:
        for row in self.rows:
            if row.work_ref == work_ref:
                return row.cells.get(column_key, MatrixCell())
        return MatrixCell()

    def counts(self) -> dict[str, int]:
        counts = {"TRUE": 0, "FALSE": 0, "UNKNOWN": 0}
        for row in self.rows:
            for column in self.columns:
                counts[row.cells.get(column.key, MatrixCell()).value.value] += 1
        return counts

    def coverage(self) -> dict[str, float | int]:
        total = len(self.rows) * len(self.columns)
        counts = self.counts()
        answered = counts["TRUE"] + counts["FALSE"]
        return {
            "total_cells": total,
            "answered": answered,
            "unknown": counts["UNKNOWN"],
            "answered_fraction": round(answered / total, 4) if total else 0.0,
            "fully_answered_rows": sum(
                1
                for row in self.rows
                if all(row.cells.get(c.key, MatrixCell()).value is not Tri.UNKNOWN for c in self.columns)
            ),
        }

    def unknown_cells(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for row in self.rows:
            for column in self.columns:
                cell = row.cells.get(column.key, MatrixCell())
                if cell.value is Tri.UNKNOWN:
                    out.append((row.work_ref, column.key))
        return out


class GapRecord(RosModel):
    """A characterisation of a gap. ``CANDIDATE_GAP`` is not a novelty claim."""

    gap_id: str = Field(default_factory=lambda: new_id("gap"))
    statement: str = Field(min_length=1)
    kind: GapKind = GapKind.OPEN_QUESTION
    novelty_audit_id: str | None = None
    supported_by: list[str] = Field(default_factory=list)
    addressed_by: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _check(self) -> "GapRecord":
        if self.kind is GapKind.VERIFIED_NOVELTY and not self.novelty_audit_id:
            raise ValueError(
                f"gap {self.gap_id}: VERIFIED_NOVELTY requires the novelty_audit_id that established it"
            )
        return self
