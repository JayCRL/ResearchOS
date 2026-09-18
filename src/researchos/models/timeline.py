"""The research timeline and the researcher's own voice.

The timeline exists so that a later reader (or the paper) can tell the true story:
observation → initial hypothesis → anomaly → control → hypothesis revision → remaining evidence →
final claim. It is not a reconstruction of the last chat session.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import RosModel, SourceRef, StrEnum, new_id, utcnow


class TimelineEventKind(StrEnum):
    PROJECT_CREATED = "PROJECT_CREATED"
    IMPORT = "IMPORT"
    IDEA = "IDEA"
    LITERATURE = "LITERATURE"
    DECISION = "DECISION"
    STATE_TRANSITION = "STATE_TRANSITION"
    TASK_CREATED = "TASK_CREATED"
    TASK_CLOSED = "TASK_CLOSED"
    FINDING = "FINDING"
    EXPERIMENT_REGISTERED = "EXPERIMENT_REGISTERED"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_RESULT = "EXPERIMENT_RESULT"
    EXPERIMENT_FAILED = "EXPERIMENT_FAILED"
    ANALYSIS = "ANALYSIS"
    EVIDENCE = "EVIDENCE"
    VERIFICATION = "VERIFICATION"
    AUDIT = "AUDIT"
    ANOMALY = "ANOMALY"
    CLAIM_CREATED = "CLAIM_CREATED"
    CLAIM_REVISION = "CLAIM_REVISION"
    CLAIM_REJECTED = "CLAIM_REJECTED"
    SKILL = "SKILL"
    PAPER = "PAPER"


class TimelineEvent(RosModel):
    event_id: str = Field(default_factory=lambda: new_id("timeline_event"))
    at: datetime = Field(default_factory=utcnow)
    kind: TimelineEventKind
    title: str = Field(min_length=1)
    detail: str = ""
    actor: str = "human"
    task_id: str | None = None
    refs: list[str] = Field(default_factory=list)
    state_revision: int | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    #: True when this event came from imported legacy material rather than live operation.
    imported: bool = False
    superseded_by: str | None = None


class NoteKind(StrEnum):
    """Why-this-experiment notes, diaries and post-mortems — the raw material of researcher voice."""

    DIARY = "DIARY"
    WHY_THIS_EXPERIMENT = "WHY_THIS_EXPERIMENT"
    FAILED_EXPERIMENT = "FAILED_EXPERIMENT"
    CLAIM_REVISION = "CLAIM_REVISION"
    AUDIT_NOTE = "AUDIT_NOTE"
    INTUITION = "INTUITION"
    ANOMALY = "ANOMALY"
    DECISION_NOTE = "DECISION_NOTE"
    TODO = "TODO"
    OTHER = "OTHER"


class AuthorNote(RosModel):
    """A human's own words, imported or live. Never paraphrased into a claim automatically."""

    note_id: str = Field(default_factory=lambda: new_id("finding"))
    kind: NoteKind = NoteKind.DIARY
    text: str = Field(min_length=1)
    author: str = "human"
    created_at: datetime = Field(default_factory=utcnow)
    source_refs: list[SourceRef] = Field(default_factory=list)
    related_claim_ids: list[str] = Field(default_factory=list)
    related_experiment_ids: list[str] = Field(default_factory=list)
    imported: bool = False
    used_in_paper: bool = False
    used_in_sections: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
