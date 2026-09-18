"""Paper compilation objects — the output of *compiling* research state, not of free generation.

A compiled sentence is a data structure, not just a string: it carries the claim, evidence and
number addresses that justify it. That is what makes a grounding violation mechanically detectable.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .common import (
    EvidenceLevel,
    ReadinessDimension,
    RosModel,
    Severity,
    SourceRef,
    StrEnum,
    clampf,
    new_id,
    utcnow,
)


class PaperSection(StrEnum):
    TITLE = "TITLE"
    ABSTRACT = "ABSTRACT"
    INTRODUCTION = "INTRODUCTION"
    RELATED_WORK = "RELATED_WORK"
    METHOD = "METHOD"
    EXPERIMENTS = "EXPERIMENTS"
    RESULTS = "RESULTS"
    ANALYSIS = "ANALYSIS"
    LIMITATIONS = "LIMITATIONS"
    DISCUSSION = "DISCUSSION"
    CONCLUSION = "CONCLUSION"
    REFERENCES = "REFERENCES"
    APPENDIX = "APPENDIX"


class NumberRef(RosModel):
    """The address of one numeral. A numeral without one of these cannot be compiled into a paper."""

    number_id: str = Field(default_factory=lambda: new_id("number"))
    value: float
    unit: str | None = None
    display: str | None = Field(default=None, description="How it should be rendered, e.g. '12.3%'.")
    analysis_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    result_field: str = Field(default="value", description="Which StatResult field, e.g. 'mean'.")
    artifact_path: str | None = None
    artifact_hash: str | None = None
    locator: str | None = Field(default=None, description="Line in the analysis artifact, if textual.")
    tolerance: float = Field(default=0.0, ge=0.0)
    rounding_note: str | None = None
    context: str = Field(default="", description="What this number means in the sentence.")

    def matches(self, candidate: float) -> bool:
        return abs(candidate - self.value) <= self.tolerance + 1e-12

    def render(self) -> str:
        if self.display:
            return self.display
        text = f"{self.value:g}"
        return f"{text} {self.unit}" if self.unit else text


class CitationRef(RosModel):
    """The address of a prior-work statement: sentence -> literature claim -> paper -> page."""

    literature_claim_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    page: str | None = None
    section: str | None = None
    quote: str | None = None
    fulltext_verified: bool = False
    display_key: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "CitationRef":
        if not (self.page or self.section or self.quote):
            raise ValueError(
                f"citation {self.literature_claim_id}: needs a page, section or quote locator"
            )
        return self

    def locator(self) -> str:
        return " ".join(p for p in (self.section, f"p.{self.page}" if self.page else None) if p)


class GroundedSentence(RosModel):
    """One sentence plus everything that licenses it."""

    sentence_id: str = Field(default_factory=lambda: new_id("finding"))
    text: str = Field(min_length=1)
    section: PaperSection = PaperSection.RESULTS
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    interpretation_ids: list[str] = Field(default_factory=list)
    numbers: list[NumberRef] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    author_note_ids: list[str] = Field(default_factory=list)
    language_level: EvidenceLevel = EvidenceLevel.L0_IDEA
    is_load_bearing: bool = Field(
        default=True, description="True when the sentence asserts a result, mechanism or novelty."
    )
    generated_by: str = Field(default="compiler", description="Rule id or template that produced it.")

    @property
    def is_grounded(self) -> bool:
        """A load-bearing sentence must point at something. Prose alone is not grounding."""
        if not self.is_load_bearing:
            return True
        return bool(self.claim_ids or self.evidence_ids or self.interpretation_ids or self.citations)


class GroundingViolationCode(StrEnum):
    UNGROUNDED_NUMBER = "UNGROUNDED_NUMBER"
    NUMBER_MISMATCH = "NUMBER_MISMATCH"
    UNRESOLVED_NUMBER = "UNRESOLVED_NUMBER"
    UNGROUNDED_CLAIM = "UNGROUNDED_CLAIM"
    CLAIM_NOT_APPROVED = "CLAIM_NOT_APPROVED"
    CLAIM_STATUS_INSUFFICIENT = "CLAIM_STATUS_INSUFFICIENT"
    CITATION_UNRESOLVED = "CITATION_UNRESOLVED"
    CITATION_SOURCE_MISSING = "CITATION_SOURCE_MISSING"
    CITATION_FULLTEXT_REQUIRED = "CITATION_FULLTEXT_REQUIRED"
    LANGUAGE_EXCEEDS_EVIDENCE = "LANGUAGE_EXCEEDS_EVIDENCE"
    NOVELTY_UNSUPPORTED = "NOVELTY_UNSUPPORTED"
    INTERPRETATION_NOT_APPROVED = "INTERPRETATION_NOT_APPROVED"
    MECHANISM_UNSUPPORTED = "MECHANISM_UNSUPPORTED"
    REJECTED_CLAIM_CITED = "REJECTED_CLAIM_CITED"


class GroundingViolation(RosModel):
    code: GroundingViolationCode
    severity: Severity = Severity.HIGH
    message: str
    section: PaperSection | None = None
    sentence_id: str | None = None
    quote: str | None = None
    expected: str | None = None
    found: str | None = None
    blocks_compilation: bool = True
    suggestion: str | None = None


class GroundingReport(RosModel):
    """The result of verifying a compiled paper against its sources. Compilation fails on violations."""

    grounding_report_id: str = Field(default_factory=lambda: new_id("audit"))
    numbers_checked: int = Field(default=0, ge=0)
    numbers_unmatched: int = Field(default=0, ge=0)
    citations_checked: int = Field(default=0, ge=0)
    citations_unresolved: int = Field(default=0, ge=0)
    sentences_checked: int = Field(default=0, ge=0)
    load_bearing_sentences: int = Field(default=0, ge=0)
    violations: list[GroundingViolation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    deterministic: bool = True

    @property
    def passed(self) -> bool:
        return not any(v.blocks_compilation for v in self.violations)

    def blockers(self) -> list[GroundingViolation]:
        return [v for v in self.violations if v.blocks_compilation]

    def summary(self) -> str:
        return (
            f"{self.numbers_checked} numbers checked ({self.numbers_unmatched} unmatched), "
            f"{self.citations_checked} citations checked ({self.citations_unresolved} unresolved), "
            f"{self.load_bearing_sentences}/{self.sentences_checked} load-bearing sentences, "
            f"{len(self.blockers())} blocking violations"
        )


class SectionDraft(RosModel):
    section: PaperSection
    heading: str = ""
    sentences: list[GroundedSentence] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    word_count: int = 0

    def render(self) -> str:
        body = " ".join(s.text.strip() for s in self.sentences)
        return f"## {self.heading}\n\n{body}\n" if self.heading else body + "\n"


class ReadinessDimensionResult(RosModel):
    """One readiness dimension. Deliberately reported separately — there is no single score."""

    dimension: ReadinessDimension
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    basis: str = ""
    blockers: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(
        default_factory=list, description="What we could not check, kept visible rather than scored away."
    )
    hard_blocked: bool = False

    def label(self) -> str:
        return self.dimension.value.replace("_", " ").title()


class PaperReadiness(RosModel):
    """Multi-dimensional readiness.

    There is intentionally **no** ``total``/``overall`` field: collapsing seven scientific
    dimensions into one number is exactly the kind of false precision this system exists to prevent.
    """

    readiness_id: str = Field(default_factory=lambda: new_id("audit"))
    dimensions: list[ReadinessDimensionResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    notes: list[str] = Field(default_factory=list)

    def get(self, dimension: ReadinessDimension) -> ReadinessDimensionResult | None:
        for item in self.dimensions:
            if item.dimension is dimension:
                return item
        return None

    def weakest(self, k: int = 3) -> list[ReadinessDimensionResult]:
        return sorted(self.dimensions, key=lambda d: d.score)[:k]

    def blockers(self) -> list[str]:
        out: list[str] = []
        for item in self.dimensions:
            out.extend(f"{item.dimension.value}: {b}" for b in item.blockers)
        return out

    def complete(self) -> bool:
        present = {d.dimension for d in self.dimensions}
        return present == set(ReadinessDimension)


class StyleCategory(StrEnum):
    EMPTY_BACKGROUND = "EMPTY_BACKGROUND"
    TEMPLATE_PHRASE = "TEMPLATE_PHRASE"
    REPETITION = "REPETITION"
    BUZZWORD_DENSITY = "BUZZWORD_DENSITY"
    UNSUPPORTED_CAUSAL = "UNSUPPORTED_CAUSAL"
    UNSUPPORTED_NOVELTY = "UNSUPPORTED_NOVELTY"
    OVERCLAIM = "OVERCLAIM"
    GENERIC_STATEMENT = "GENERIC_STATEMENT"
    AI_LIKE_STRUCTURE = "AI_LIKE_STRUCTURE"
    RHETORICAL_ADJECTIVE = "RHETORICAL_ADJECTIVE"
    MISSING_RESEARCHER_REASONING = "MISSING_RESEARCHER_REASONING"
    HISTORY_MISMATCH = "HISTORY_MISMATCH"
    CLAIM_STRENGTH_DRIFT = "CLAIM_STRENGTH_DRIFT"
    HEDGE_ABUSE = "HEDGE_ABUSE"
    SENTENCE_UNIFORMITY = "SENTENCE_UNIFORMITY"


class StyleFinding(RosModel):
    category: StyleCategory
    code: str
    severity: Severity = Severity.LOW
    message: str
    quote: str | None = None
    location: str | None = None
    section: PaperSection | None = None
    suggested_rewrite: str | None = None
    evidence_level: EvidenceLevel | None = None
    deterministic: bool = True


class PaperArtifact(RosModel):
    """The compiled paper. Its inputs are recorded so the compilation is replayable."""

    paper_id: str = Field(default_factory=lambda: new_id("paper"))
    title: str = ""
    abstract: str = ""
    sections: list[SectionDraft] = Field(default_factory=list)
    format: str = Field(default="markdown", description="markdown | latex")

    compiled_at: datetime = Field(default_factory=utcnow)
    compiled_by: str = "paper_compiler"
    state_revision: int = Field(default=0, ge=0)

    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    analysis_ids: list[str] = Field(default_factory=list)
    literature_claim_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    author_note_ids: list[str] = Field(default_factory=list)
    interpretation_ids: list[str] = Field(default_factory=list)

    grounding_report: GroundingReport | None = None
    style_audit_id: str | None = None
    readiness_id: str | None = None
    red_team_report_id: str | None = None

    #: Requests the writer made that were refused — kept as evidence of the guard working.
    refused_additions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)

    def all_sentences(self) -> list[GroundedSentence]:
        return [s for section in self.sections for s in section.sentences]

    def render(self) -> str:
        """Render markdown. The abstract is printed once, from the compiled section when present."""
        parts = [f"# {self.title}\n"]
        has_abstract_section = any(section.section is PaperSection.ABSTRACT for section in self.sections)
        if self.abstract and not has_abstract_section:
            parts.append(f"## Abstract\n\n{self.abstract}\n")
        parts.extend(section.render() for section in self.sections)
        return "\n".join(parts)

    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.all_sentences())

    def is_compilable(self) -> bool:
        """A paper is only a paper if grounding passed. Otherwise it is a draft, not an artifact."""
        return self.grounding_report is not None and self.grounding_report.passed


class RedTeamQuestion(RosModel):
    question: str = Field(min_length=1)
    category: str = Field(
        default="general",
        description="falsification | alternative_explanation | weakest_experiment | single_dependency | "
        "baseline_matching | metric_contamination | closest_prior_art | reviewer_reading",
    )
    target_ref: str | None = None
    concern: str = ""
    severity: Severity = Severity.MEDIUM
    what_would_falsify: str | None = None
    surviving_alternatives: list[str] = Field(default_factory=list)
    suggested_experiment: str | None = None
    addressed_by: list[str] = Field(default_factory=list)
    resolved: bool = False


class RedTeamReport(RosModel):
    """Adversarial review output. It reports; it never rewrites the research state."""

    red_team_report_id: str = Field(default_factory=lambda: new_id("audit"))
    subject: str = Field(min_length=1)
    subject_refs: list[str] = Field(default_factory=list)
    questions: list[RedTeamQuestion] = Field(default_factory=list)
    strongest_objection: str | None = None
    weakest_experiment_id: str | None = None
    single_dependencies: list[str] = Field(default_factory=list)
    meta_review: str = ""
    verdict: str = Field(default="NEEDS_WORK", description="READY | NEEDS_WORK | NOT_READY")
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "red_team"
    respects_state: bool = Field(default=True)

    @model_validator(mode="after")
    def _check(self) -> "RedTeamReport":
        if not self.respects_state:
            raise ValueError(
                f"red team report {self.red_team_report_id}: red team finds problems, it does not "
                "change official state"
            )
        return self

    def unaddressed(self) -> list[RedTeamQuestion]:
        return [q for q in self.questions if not q.resolved]

    def blocking(self) -> list[RedTeamQuestion]:
        return [q for q in self.unaddressed() if q.severity in (Severity.HIGH, Severity.BLOCKER)]
