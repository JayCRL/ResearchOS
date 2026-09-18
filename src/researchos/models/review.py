"""The human review queue — where imported material waits before it becomes research state.

Research Import is allowed to *reconstruct* state, not to *bless* it. Every uncertain recovery
lands here with its evidence and stays pending until a human accepts, edits or rejects it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import RosModel, ReviewDecision, Severity, SourceRef, StrEnum, new_id, utcnow


class ReviewKind(StrEnum):
    CORE_QUESTION_CANDIDATE = "CORE_QUESTION_CANDIDATE"
    CLAIM_CANDIDATE = "CLAIM_CANDIDATE"
    REJECTED_CLAIM_CANDIDATE = "REJECTED_CLAIM_CANDIDATE"
    EXPERIMENT_CANDIDATE = "EXPERIMENT_CANDIDATE"
    EVIDENCE_CANDIDATE = "EVIDENCE_CANDIDATE"
    ANALYSIS_CANDIDATE = "ANALYSIS_CANDIDATE"
    LITERATURE_CANDIDATE = "LITERATURE_CANDIDATE"
    DECISION_CANDIDATE = "DECISION_CANDIDATE"
    OPEN_QUESTION_CANDIDATE = "OPEN_QUESTION_CANDIDATE"
    NOTE_CANDIDATE = "NOTE_CANDIDATE"
    CONFLICT = "CONFLICT"
    UNPARSED_MATERIAL = "UNPARSED_MATERIAL"
    SKILL_GAP = "SKILL_GAP"


class ReviewItem(RosModel):
    review_item_id: str = Field(default_factory=lambda: new_id("review_item"))
    kind: ReviewKind
    title: str = Field(min_length=1)
    rationale: str = ""

    subject_ref: str | None = Field(
        default=None, description="Id of a created draft object, when one was materialised."
    )
    proposed_object: dict[str, object] = Field(
        default_factory=dict, description="The object as recovered, for the reviewer to inspect."
    )
    proposed_action: str = Field(default="CREATE")

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    severity: Severity = Severity.MEDIUM
    source_refs: list[SourceRef] = Field(default_factory=list)
    extraction_rule: str | None = Field(
        default=None, description="Which deterministic rule produced this; None means LLM-assisted."
    )
    deterministic: bool = True
    alternatives: list[str] = Field(default_factory=list)
    questions_for_human: list[str] = Field(default_factory=list)

    status: ReviewDecision = ReviewDecision.PENDING
    decided_by: str | None = None
    decided_at: datetime | None = None
    decision_note: str | None = None
    edits: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    batch_id: str | None = Field(default=None, description="Import batch that produced this item.")

    def is_pending(self) -> bool:
        return self.status is ReviewDecision.PENDING
