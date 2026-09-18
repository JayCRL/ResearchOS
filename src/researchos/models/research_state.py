"""The Global Research State — the persistent source of truth that chat history is not.

``state/research_state.yaml`` is this model, dumped as YAML. It is intentionally
human-readable and git-diffable: a research direction change should be reviewable in a pull
request, exactly like a code change.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .common import (
    GapKind,
    RosModel,
    SourceRef,
    StrEnum,
    TaskPriority,
    new_id,
    utcnow,
)


class QuestionKind(StrEnum):
    CORE = "CORE"
    SECONDARY = "SECONDARY"
    OPEN = "OPEN"
    NON_GOAL = "NON_GOAL"


class CoreQuestion(RosModel):
    """The single question the project is trying to answer. Guarded: STR only."""

    question_id: str = Field(default_factory=lambda: new_id("finding"))
    statement: str = Field(min_length=1)
    motivation: str = ""
    scope: str = Field(default="", description="What is inside and outside this question.")
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    revised_at: datetime | None = None
    revision_note: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    is_placeholder: bool = Field(
        default=False,
        description="True when Research Import could not recover a confident core question and the "
        "human has not yet confirmed one.",
    )

    @property
    def is_confirmed(self) -> bool:
        return not self.is_placeholder


class OpenQuestion(RosModel):
    """An unresolved question, tagged with *how well* it is characterised.

    ``CANDIDATE_GAP`` is explicitly not ``VERIFIED_NOVELTY``: the taxonomy is enforced here so
    that a wording choice cannot upgrade a gap into a novelty claim.
    """

    question_id: str = Field(default_factory=lambda: new_id("gap"))
    statement: str = Field(min_length=1)
    kind: GapKind = GapKind.OPEN_QUESTION
    priority: TaskPriority = TaskPriority.SECONDARY
    related_claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    literature_ids: list[str] = Field(default_factory=list)
    blocker: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    resolved_at: datetime | None = None
    resolution: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "OpenQuestion":
        if self.kind is GapKind.VERIFIED_NOVELTY and not self.literature_ids:
            raise ValueError(
                f"open question {self.question_id}: VERIFIED_NOVELTY requires literature_ids from a "
                "completed novelty audit — CANDIDATE_GAP is not VERIFIED_NOVELTY"
            )
        if self.resolved_at is not None and not self.resolution:
            raise ValueError(f"open question {self.question_id}: a closed question needs a resolution")
        return self


class PriorityItem(RosModel):
    rank: int = Field(ge=1)
    statement: str = Field(min_length=1)
    rationale: str = ""
    claim_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)


class PriorWorkRef(RosModel):
    paper_id: str
    title: str = ""
    relation: str = "same_problem"
    threat_level: str = Field(
        default="UNKNOWN", description="NONE | LOW | MEDIUM | HIGH | UNKNOWN — how close this is."
    )
    note: str = ""
    page: str | None = None


class LiteratureState(RosModel):
    """Summary of what we know about the literature. Coverage is measured, never assumed."""

    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_basis: str = Field(
        default="no searches executed", description="Human-readable basis for coverage_score."
    )
    queries_executed: int = Field(default=0, ge=0)
    providers_used: list[str] = Field(default_factory=list)
    papers_screened: int = Field(default=0, ge=0)
    papers_retained: int = Field(default=0, ge=0)
    fulltext_verified: int = Field(default=0, ge=0)
    closest_prior_work: list[PriorWorkRef] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    last_search_at: datetime | None = None
    terminology_coverage: dict[str, int] = Field(
        default_factory=dict, description="query family -> number of queries executed."
    )


class SkillState(RosModel):
    active: list[str] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    deprecated: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    health: float = Field(default=0.0, ge=0.0, le=1.0)
    last_benchmark_at: datetime | None = None


class Frontier(RosModel):
    """Where the research actually stands right now."""

    statement: str = ""
    current_focus: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)


class ResearchState(RosModel):
    """★ The Global Research State. Persisted at ``.researchos/state/research_state.yaml``."""

    schema_version: str = "1.0"
    project_id: str = ""
    project_name: str = ""

    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utcnow)
    updated_by: str = "human"

    core_question: CoreQuestion | None = None
    core_claims: list[str] = Field(default_factory=list)
    secondary_questions: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    priorities: list[PriorityItem] = Field(default_factory=list)

    current_frontier: Frontier = Field(default_factory=Frontier)
    known_evidence: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    research_decisions: list[str] = Field(default_factory=list)

    literature_state: LiteratureState = Field(default_factory=LiteratureState)
    skill_state: SkillState = Field(default_factory=SkillState)

    active_task: str | None = None
    task_queue: list[str] = Field(default_factory=list)
    recent_tasks: list[str] = Field(default_factory=list)

    import_summary: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def guarded_roots(self) -> dict[str, object]:
        from .decision import GUARDED_PATHS

        return {path: getattr(self, path) for path in GUARDED_PATHS}
