"""Local task state — the anti-drift container.

A task is a *bounded* piece of work with an explicit priority and stop condition. Its findings
are recorded on the task; promoting them to core research state requires an approved STR.
That is the whole point: solving a small problem can never silently redefine the project.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .common import (
    RiskLevel,
    RosModel,
    StrEnum,
    TaskPriority,
    TaskPurpose,
    TaskStatus,
    new_id,
    utcnow,
)


class FindingKind(StrEnum):
    OBSERVATION = "OBSERVATION"
    ANOMALY = "ANOMALY"
    HYPOTHESIS = "HYPOTHESIS"
    NEGATIVE_RESULT = "NEGATIVE_RESULT"
    DECISION = "DECISION"
    OPEN_QUESTION = "OPEN_QUESTION"
    TOOLING = "TOOLING"


class TaskFinding(RosModel):
    """What a bounded task discovered. Recorded on the task, *not* on the research state."""

    finding_id: str = Field(default_factory=lambda: new_id("finding"))
    kind: FindingKind = FindingKind.OBSERVATION
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggests_core_change: bool = Field(
        default=False,
        description="Agent's *opinion* that this should change core state. Opinion is not a transition.",
    )
    promotion_str_id: str | None = None
    promoted: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    notes: list[str] = Field(default_factory=list)


class Task(RosModel):
    """A bounded research task with a declared ceiling on what it may change."""

    task_id: str = Field(default_factory=lambda: new_id("task"))
    objective: str = Field(min_length=1)
    purpose: TaskPurpose = TaskPurpose.ANALYSE
    priority: TaskPriority = TaskPriority.EXPLORATORY

    parent_claim: str | None = None
    parent_experiment: str | None = None
    parent_question: str | None = None

    allowed_actions: list[str] = Field(
        default_factory=list, description="Capability names this task may exercise."
    )
    forbidden_actions: list[str] = Field(
        default_factory=list, description="Capabilities refused even if the principal holds them."
    )
    stop_condition: str = Field(
        default="",
        description="When to stop. Mandatory for any task that could touch core state.",
    )
    touches_core: bool = Field(
        default=False, description="Whether this task's findings are expected to bear on core state."
    )
    risk: RiskLevel = RiskLevel.LOW

    status: TaskStatus = TaskStatus.OPEN
    created_from_revision: int = Field(ge=0, default=0)
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    closed_at: datetime | None = None
    created_by: str = "human"

    findings: list[TaskFinding] = Field(default_factory=list)
    promotion_request_ids: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    result_summary: str | None = None
    notes: list[str] = Field(default_factory=list)
    context_refs: list[str] = Field(
        default_factory=list,
        description="Deliberately *narrow* context: only these ids may be loaded when running "
        "this task. Prevents whole-history context pollution.",
    )
    context_budget_files: int | None = Field(
        default=None,
        ge=0,
        description="Optional cap on how many source files this task may read.",
    )

    @model_validator(mode="after")
    def _check(self) -> "Task":
        overlap = set(self.allowed_actions) & set(self.forbidden_actions)
        if overlap:
            raise ValueError(f"task {self.task_id}: actions both allowed and forbidden: {sorted(overlap)}")
        if self.touches_core and not self.stop_condition:
            raise ValueError(
                f"task {self.task_id}: a task that touches core state must declare a stop_condition"
            )
        if self.touches_core and self.priority is TaskPriority.EXPLORATORY:
            raise ValueError(
                f"task {self.task_id}: an EXPLORATORY task may not be declared as touching core state; "
                "exploratory findings must be proposed via STR"
            )
        return self

    # ------------------------------------------------------------------ capabilities

    def may_request_core_change(self) -> bool:
        """The kernel's answer to "can this task propose changing the core question?".

        Debugging and explaining are *local* activities. They may produce findings and ask the
        human for a transition, but under no circumstances do they themselves constitute a
        research-direction change.
        """
        if self.priority is not TaskPriority.PRIMARY:
            return False
        if self.purpose in (TaskPurpose.DEBUG, TaskPurpose.EXPLAIN, TaskPurpose.IMPORT):
            return False
        return True

    def permits(self, capability: str) -> bool:
        if capability in self.forbidden_actions:
            return False
        if not self.allowed_actions:
            return True
        return capability in self.allowed_actions

    def is_open(self) -> bool:
        return self.status in (TaskStatus.OPEN, TaskStatus.RUNNING)

    def add_finding(self, finding: TaskFinding) -> TaskFinding:
        if not self.is_open():
            raise ValueError(f"task {self.task_id} is {self.status.value}; findings belong to open tasks")
        self.findings = [*self.findings, finding]
        return finding
