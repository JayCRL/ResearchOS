"""Decisions and state-transition requests — the mechanism that makes drift impossible and explainable."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from .common import (
    ChangeClass,
    RiskLevel,
    RosModel,
    StrEnum,
    TransitionStatus,
    new_id,
    sha256_json,
    utcnow,
)


class DecisionKind(StrEnum):
    RESEARCH_DIRECTION = "RESEARCH_DIRECTION"
    EXPERIMENT_ROUTE = "EXPERIMENT_ROUTE"
    CLAIM_REVISION = "CLAIM_REVISION"
    CLAIM_CANCELLED = "CLAIM_CANCELLED"
    CONTROL_ADDED = "CONTROL_ADDED"
    RESULT_EXCLUDED = "RESULT_EXCLUDED"
    SCOPE_CHANGE = "SCOPE_CHANGE"
    PRIORITY_CHANGE = "PRIORITY_CHANGE"
    TOOLING = "TOOLING"
    OTHER = "OTHER"


class Decision(RosModel):
    """A permanent, human-readable record of *why* something changed.

    This is what lets a later reader answer "why was the original claim dropped?",
    "why did the experiment route change?", "why was this control added?",
    "why is this result no longer in the paper?".
    """

    decision_id: str = Field(default_factory=lambda: new_id("decision"))
    kind: DecisionKind = DecisionKind.OTHER
    summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(default_factory=list)
    rejected_alternatives: list[str] = Field(default_factory=list)
    consequences: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    affected_claims: list[str] = Field(default_factory=list)
    affected_experiments: list[str] = Field(default_factory=list)
    made_by: str = "human"
    str_id: str | None = None
    task_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class StateOp(StrEnum):
    SET = "set"
    APPEND = "append"
    REMOVE = "remove"
    MERGE = "merge"


class StateOperation(RosModel):
    """One proposed mutation of the global research state, addressed by dotted path."""

    op: StateOp = StateOp.SET
    path: str = Field(
        min_length=1,
        description="Dotted path into ResearchState, e.g. 'core_question.statement', "
        "'priorities', 'core_claims'.",
    )
    value: object | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "StateOperation":
        if self.op is not StateOp.REMOVE and self.value is None:
            raise ValueError(f"operation {self.op.value} on {self.path!r} requires a value")
        return self


#: Fields whose modification is refused unless an approved STR covers them.
GUARDED_PATHS: tuple[str, ...] = (
    "core_question",
    "core_claims",
    "secondary_questions",
    "non_goals",
    "priorities",
)

#: Map a guarded path to its change class, for STR classification.
CHANGE_CLASS_OF: dict[str, ChangeClass] = {
    "core_question": ChangeClass.CORE_QUESTION,
    "core_claims": ChangeClass.CORE_CLAIM,
    "secondary_questions": ChangeClass.SCOPE,
    "non_goals": ChangeClass.NON_GOAL,
    "priorities": ChangeClass.PRIORITY,
}


def guarded_root(path: str) -> str | None:
    """Return the guarded root covering ``path`` (``"core_claims[0]"`` -> ``"core_claims"``)."""
    head = path.split(".", 1)[0].split("[", 1)[0]
    return head if head in GUARDED_PATHS else None


class StateTransitionRequest(RosModel):
    """The *only* legitimate way to change a guarded research-state field.

    Agents may create STRs. Only a principal holding ``state.transition.approve``
    (by default: the human) may approve one.
    """

    str_id: str = Field(default_factory=lambda: new_id("str"))
    requested_by: str
    created_at: datetime = Field(default_factory=utcnow)
    task_id: str | None = None

    current_revision: int = Field(ge=0)
    current_state_hash: str = Field(
        min_length=8,
        description="SHA-256 over canonical(guarded roots) at request time; detects staleness.",
    )
    proposed_operations: list[StateOperation] = Field(min_length=1)
    change_class: ChangeClass = ChangeClass.OTHER
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    affected_claims: list[str] = Field(default_factory=list)
    affected_experiments: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.MEDIUM

    status: TransitionStatus = TransitionStatus.PROPOSED
    decided_by: str | None = None
    decided_at: datetime | None = None
    decision_note: str | None = None
    applied_revision: int | None = None
    decision_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "StateTransitionRequest":
        if not self.proposed_operations:
            raise ValueError("an STR must propose at least one operation")
        classes = {CHANGE_CLASS_OF.get(guarded_root(op.path) or "", ChangeClass.OTHER) for op in self.proposed_operations}
        classes.discard(ChangeClass.OTHER)
        if self.change_class is ChangeClass.OTHER and classes:
            # keep the declaration honest: a core-touching STR cannot be filed as OTHER
            raise ValueError(
                "proposed operations touch guarded research state "
                f"({sorted(c.value for c in classes)}); declare the matching change_class"
            )
        if self.risk is not RiskLevel.LOW and not self.evidence_ids:
            raise ValueError(
                "an STR with risk != LOW must cite at least one evidence id "
                "(or be filed as LOW with an explicit reason)"
            )
        if self.status is TransitionStatus.APPROVED and not self.decided_by:
            raise ValueError("an approved STR requires decided_by")
        return self

    @property
    def is_open(self) -> bool:
        return self.status is TransitionStatus.PROPOSED

    def guarded_paths(self) -> list[str]:
        return [op.path for op in self.proposed_operations if guarded_root(op.path)]


def guarded_state_hash(state: object) -> str:
    """Hash of the guarded roots of a research state object.

    Deliberately *narrow*: unrelated edits (a new evidence link, a new open question) must not
    invalidate a pending STR, while any change to the core question/claims/priorities must.
    """
    payload: dict[str, object] = {}
    for path in GUARDED_PATHS:
        payload[path] = getattr(state, path, None)
    return sha256_json(payload)
