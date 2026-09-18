"""Task management — bounded work, explicit priority, no silent promotion.

The kernel records task findings and task lifecycle in the research state and the event log, but a
task's conclusions never become project conclusions on their own.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..models.common import TaskPriority, TaskPurpose, TaskStatus, utcnow
from ..models.task import FindingKind, Task, TaskFinding
from .errors import PermissionDenied, ResearchOSError
from .events import EventLog
from .permissions import Cap, Principal
from .store import EntityStore
from .state import StateManager

__all__ = ["TaskManager"]


class TaskManager:
    """Creates and closes bounded tasks; keeps the state's task queue honest."""

    def __init__(
        self,
        store: EntityStore[Task],
        state: StateManager,
        events: EventLog,
    ) -> None:
        self.store = store
        self.state = state
        self.events = events

    # ------------------------------------------------------------------ create

    def create(
        self,
        principal: Principal,
        *,
        objective: str,
        purpose: TaskPurpose = TaskPurpose.ANALYSE,
        priority: TaskPriority = TaskPriority.EXPLORATORY,
        parent_claim: str | None = None,
        parent_experiment: str | None = None,
        parent_question: str | None = None,
        allowed_actions: Sequence[str] = (),
        forbidden_actions: Sequence[str] = (),
        stop_condition: str = "",
        touches_core: bool = False,
        context_refs: Sequence[str] = (),
        context_budget_files: int | None = None,
        notes: Sequence[str] = (),
        task_id: str | None = None,
    ) -> Task:
        principal.require(Cap.STATE_WRITE_TASK, "task.create")

        # CLI and API callers pass these as strings; normalise once, here, so every downstream
        # consumer can rely on the enum.
        purpose = TaskPurpose(purpose) if isinstance(purpose, str) else purpose
        priority = TaskPriority(priority) if isinstance(priority, str) else priority

        task = Task(
            objective=objective,
            purpose=purpose,
            priority=priority,
            parent_claim=parent_claim,
            parent_experiment=parent_experiment,
            parent_question=parent_question,
            allowed_actions=list(allowed_actions),
            forbidden_actions=list(forbidden_actions),
            stop_condition=stop_condition,
            touches_core=touches_core,
            context_refs=list(context_refs),
            context_budget_files=context_budget_files,
            notes=list(notes),
            created_from_revision=self.state.revision(),
            created_by=principal.name,
        )
        if task_id:
            task.task_id = task_id
        self.store.save(task)
        self.state.update(
            principal,
            lambda s: setattr(s, "task_queue", [*s.task_queue, task.task_id]),
            event_kind="task.created",
            payload={
                "task_id": task.task_id,
                "objective": objective,
                "priority": priority.value,
                "purpose": purpose.value,
            },
            task_id=task.task_id,
        )
        return task

    # ------------------------------------------------------------------ read

    def get(self, task_id: str) -> Task | None:
        return self.store.get(task_id)

    def require(self, task_id: str) -> Task:
        task = self.store.get(task_id)
        if task is None:
            raise ResearchOSError(f"task {task_id!r} not found")
        return task

    def list(
        self,
        *,
        status: TaskStatus | None = None,
        priority: TaskPriority | None = None,
        open_only: bool = False,
    ) -> list[Task]:
        tasks = self.store.all()
        if status is not None:
            tasks = [t for t in tasks if t.status is status]
        if priority is not None:
            tasks = [t for t in tasks if t.priority is priority]
        if open_only:
            tasks = [t for t in tasks if t.is_open()]
        return tasks

    def active(self) -> Task | None:
        active_id = self.state.load().active_task
        return self.store.get(active_id) if active_id else None

    def context_scope(self, task_id: str) -> dict[str, Any]:
        """The *narrow* context a task is allowed to load.

        This is the concrete answer to "long context poisoned the research direction": a task
        declares what it may read, and the runner honours it.
        """
        task = self.require(task_id)
        scope: dict[str, Any] = {
            "task_id": task.task_id,
            "objective": task.objective,
            "purpose": task.purpose.value,
            "priority": task.priority.value,
            "context_refs": list(task.context_refs),
            "forbidden_actions": list(task.forbidden_actions),
            "may_request_core_change": task.may_request_core_change(),
            "stop_condition": task.stop_condition,
        }
        if task.parent_claim:
            scope["parent_claim"] = task.parent_claim
        if task.parent_experiment:
            scope["parent_experiment"] = task.parent_experiment
        if task.parent_question:
            scope["parent_question"] = task.parent_question
        return scope

    # ------------------------------------------------------------------ lifecycle

    def start(self, principal: Principal, task_id: str) -> Task:
        task = self.require(task_id)
        task.status = TaskStatus.RUNNING
        task.started_at = utcnow()
        self.store.save(task)
        self.state.update(
            principal,
            lambda s: setattr(s, "active_task", task_id),
            event_kind="task.started",
            payload={"task_id": task_id, "objective": task.objective},
            task_id=task_id,
        )
        return task

    def add_finding(
        self,
        principal: Principal,
        task_id: str,
        *,
        statement: str,
        kind: FindingKind = FindingKind.OBSERVATION,
        evidence_ids: Sequence[str] = (),
        confidence: float = 0.5,
        suggests_core_change: bool = False,
        notes: Sequence[str] = (),
    ) -> TaskFinding:
        task = self.require(task_id)
        finding = TaskFinding(
            kind=kind,
            statement=statement,
            evidence_ids=list(evidence_ids),
            confidence=confidence,
            suggests_core_change=suggests_core_change,
            notes=list(notes),
        )
        task.add_finding(finding)
        self.store.save(task)
        self.events.append(
            "task.finding",
            actor=principal.name,
            task_id=task_id,
            payload={
                "finding_id": finding.finding_id,
                "kind": kind.value,
                "suggests_core_change": suggests_core_change,
                "statement": statement[:400],
            },
        )
        return finding

    def close(
        self,
        principal: Principal,
        task_id: str,
        *,
        summary: str,
        status: TaskStatus = TaskStatus.DONE,
        note: str | None = None,
    ) -> Task:
        task = self.require(task_id)
        task.status = status
        task.result_summary = summary
        task.closed_at = utcnow()
        if note:
            task.notes = [*task.notes, note]
        self.store.save(task)

        def mutate(state) -> None:
            if state.active_task == task_id:
                state.active_task = None
            if task_id in state.task_queue:
                state.task_queue = [t for t in state.task_queue if t != task_id]
            if task_id not in state.recent_tasks:
                state.recent_tasks = [*state.recent_tasks, task_id]

        self.state.update(
            principal,
            mutate,
            event_kind="task.closed",
            payload={"task_id": task_id, "status": status.value, "summary": summary[:400]},
            task_id=task_id,
        )
        return task

    def record_promotion_request(self, task_id: str, str_id: str) -> None:
        task = self.require(task_id)
        task.promotion_request_ids = [*task.promotion_request_ids, str_id]
        self.store.save(task)

    # ------------------------------------------------------------------ helpers

    def guard_core_change_request(self, principal: Principal, task_id: str | None) -> None:
        """Refuse a core-change request coming from a task that has no business making one."""
        if principal.is_human:
            return
        if task_id is None:
            raise PermissionDenied(
                "a core-state transition request must be attached to a task; "
                "free-floating direction changes are exactly the drift the kernel prevents",
                principal=principal.name,
            )
        task = self.require(task_id)
        if not task.may_request_core_change():
            raise PermissionDenied(
                f"task {task_id} (purpose={task.purpose.value}, priority={task.priority.value}) may "
                "not request a core research-state change. Record a finding with "
                "suggests_core_change=True and let a human decide; solving a local problem does not "
                "redefine the project.",
                principal=principal.name,
                resource=task_id,
            )
