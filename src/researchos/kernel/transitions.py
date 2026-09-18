"""State Transition Requests — how research direction actually changes.

Agents detect, propose and justify. Humans approve. Nothing else changes the core question.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..models.common import (
    ChangeClass,
    RiskLevel,
    TransitionStatus,
    canonical_json,
    sha256_json,
    utcnow,
)
from ..models.decision import (
    CHANGE_CLASS_OF,
    Decision,
    DecisionKind,
    StateOperation,
    StateOp,
    StateTransitionRequest,
    guarded_root,
    guarded_state_hash,
)
from .errors import PermissionDenied, ResearchOSError, TransitionError
from .events import EventLog
from .permissions import Cap, Principal
from .store import EntityStore
from .state import StateManager
from .tasks import TaskManager

#: Which change classes a decision record should be filed under.
DECISION_KIND_BY_CHANGE: dict[ChangeClass, DecisionKind] = {
    ChangeClass.CORE_QUESTION: DecisionKind.RESEARCH_DIRECTION,
    ChangeClass.CORE_CLAIM: DecisionKind.CLAIM_REVISION,
    ChangeClass.PRIORITY: DecisionKind.PRIORITY_CHANGE,
    ChangeClass.NON_GOAL: DecisionKind.SCOPE_CHANGE,
    ChangeClass.SCOPE: DecisionKind.SCOPE_CHANGE,
    ChangeClass.OTHER: DecisionKind.OTHER,
}


class TransitionManager:
    """Create, approve, reject and apply STRs. Approval is human-only by default."""

    def __init__(
        self,
        store: EntityStore[StateTransitionRequest],
        decisions: EntityStore[Decision],
        state: StateManager,
        tasks: TaskManager,
        events: EventLog,
    ) -> None:
        self.store = store
        self.decisions = decisions
        self.state = state
        self.tasks = tasks
        self.events = events

    # ------------------------------------------------------------------ request

    def request(
        self,
        principal: Principal,
        *,
        operations: Sequence[StateOperation | dict[str, Any]],
        reason: str,
        evidence_ids: Sequence[str] = (),
        change_class: ChangeClass | None = None,
        task_id: str | None = None,
        affected_claims: Sequence[str] = (),
        affected_experiments: Sequence[str] = (),
        risk: RiskLevel = RiskLevel.MEDIUM,
    ) -> StateTransitionRequest:
        """File a request to change research state. This grants nothing by itself."""
        principal.require(Cap.STATE_TRANSITION_REQUEST, "str.create")

        risk = RiskLevel(risk) if isinstance(risk, str) else risk
        if change_class is not None and isinstance(change_class, str):
            change_class = ChangeClass(change_class)

        ops = [op if isinstance(op, StateOperation) else StateOperation.model_validate(op) for op in operations]
        if not ops:
            raise TransitionError("an STR must propose at least one operation")

        guarded = sorted({guarded_root(op.path) for op in ops if guarded_root(op.path)})
        if guarded:
            self.tasks.guard_core_change_request(principal, task_id)
            if change_class is None or change_class is ChangeClass.OTHER:
                change_class = CHANGE_CLASS_OF.get(guarded[0], ChangeClass.OTHER)
        elif change_class is None:
            change_class = ChangeClass.OTHER

        if risk is not RiskLevel.LOW and not evidence_ids:
            raise TransitionError(
                "a transition with risk != LOW needs evidence_ids: which observation justifies "
                "changing the project's direction?"
            )

        state = self.state.load(refresh=True)
        request = StateTransitionRequest(
            requested_by=principal.name,
            task_id=task_id,
            current_revision=state.revision,
            current_state_hash=guarded_state_hash(state),
            proposed_operations=ops,
            change_class=change_class,
            evidence_ids=list(evidence_ids),
            reason=reason,
            affected_claims=list(affected_claims),
            affected_experiments=list(affected_experiments),
            risk=risk,
        )
        self.store.save(request)
        if task_id:
            self.tasks.record_promotion_request(task_id, request.str_id)
        self.events.append(
            "str.requested",
            actor=principal.name,
            task_id=task_id,
            payload={
                "str_id": request.str_id,
                "change_class": request.change_class.value,
                "risk": request.risk.value,
                "guarded_paths": guarded,
                "operations": [op.model_dump(mode="json") for op in ops],
                "reason": reason[:600],
                "state_hash": request.current_state_hash,
            },
        )
        return request

    # ------------------------------------------------------------------ decision

    def approve(
        self,
        principal: Principal,
        str_id: str,
        *,
        note: str = "",
    ) -> tuple[StateTransitionRequest, Decision]:
        """Approve and apply an STR. Requires an explicit approval capability (human by default)."""
        principal.require(Cap.STATE_TRANSITION_APPROVE, f"str:{str_id}.approve")
        request = self.require(str_id)

        if not request.is_open:
            raise TransitionError(
                f"STR {str_id} is {request.status.value}; only PROPOSED requests can be approved"
            )

        state = self.state.load(refresh=True)
        if state.revision != request.current_revision:
            request = request.with_updates(
                status=TransitionStatus.STALE,
                decided_by=principal.name,
                decided_at=utcnow(),
                decision_note=(
                    f"state moved from revision {request.current_revision} to {state.revision} "
                    "before approval"
                ),
            )
            self.store.save(request)
            self.events.append(
                "str.stale",
                actor=principal.name,
                task_id=request.task_id,
                payload={"str_id": str_id, "reason": request.decision_note},
            )
            raise TransitionError(
                f"STR {str_id} is STALE: the research state changed since it was filed "
                f"(rev {request.current_revision} -> {state.revision}). Re-file it against the "
                "current state — approving a stale direction change would apply it to a project "
                "that no longer exists."
            )
        if guarded_state_hash(state) != request.current_state_hash:
            raise TransitionError(
                f"STR {str_id} is STALE: guarded state hash changed since the request was filed"
            )

        # 1. apply the proposed operations to the research state, guard explicitly authorised
        def mutate(target) -> None:
            for operation in request.proposed_operations:
                self.state.apply_operation(target, operation)

        new_state = self.state.update(
            principal,
            mutate,
            event_kind="str.applied",
            payload={
                "str_id": str_id,
                "change_class": request.change_class.value,
                "operations": [op.model_dump(mode="json") for op in request.proposed_operations],
            },
            task_id=request.task_id,
            expected_revision=request.current_revision,
            allow_guarded=True,
        )

        # 2. record *why* — a decision record, so the change is explainable years later
        decision = Decision(
            kind=DECISION_KIND_BY_CHANGE.get(request.change_class, DecisionKind.OTHER),
            summary=f"Applied transition {str_id}: {request.reason[:200]}",
            rationale=note or request.reason,
            evidence_ids=list(request.evidence_ids),
            affected_claims=list(request.affected_claims),
            affected_experiments=list(request.affected_experiments),
            made_by=principal.name,
            str_id=str_id,
            task_id=request.task_id,
            consequences=[
                f"{op.op.value} {op.path}" for op in request.proposed_operations
            ],
        )
        self.decisions.save(decision)

        # 3. close the request
        request = request.with_updates(
            status=TransitionStatus.APPROVED,
            decided_by=principal.name,
            decided_at=utcnow(),
            decision_note=note or None,
            applied_revision=new_state.revision,
            decision_id=decision.decision_id,
        )
        self.store.save(request)

        # 4. mirror the decision id into the state's decision index (unguarded field)
        self.state.update(
            principal,
            lambda s: setattr(s, "research_decisions", [*s.research_decisions, decision.decision_id]),
            event_kind="decision.recorded",
            payload={"decision_id": decision.decision_id, "str_id": str_id},
            task_id=request.task_id,
        )
        self.events.append(
            "str.approved",
            actor=principal.name,
            task_id=request.task_id,
            payload={
                "str_id": str_id,
                "decision_id": decision.decision_id,
                "applied_revision": new_state.revision,
            },
        )
        return request, decision

    def reject(self, principal: Principal, str_id: str, *, reason: str) -> StateTransitionRequest:
        principal.require(Cap.STATE_TRANSITION_APPROVE, f"str:{str_id}.reject")
        request = self.require(str_id)
        if not request.is_open:
            raise TransitionError(f"STR {str_id} is already {request.status.value}")
        request = request.with_updates(
            status=TransitionStatus.REJECTED,
            decided_by=principal.name,
            decided_at=utcnow(),
            decision_note=reason,
        )
        self.store.save(request)
        self.events.append(
            "str.rejected",
            actor=principal.name,
            task_id=request.task_id,
            payload={"str_id": str_id, "reason": reason},
        )
        return request

    def withdraw(self, principal: Principal, str_id: str, *, reason: str = "") -> StateTransitionRequest:
        request = self.require(str_id)
        if request.requested_by != principal.name and not principal.is_human:
            raise PermissionDenied(
                f"only {request.requested_by!r} or a human may withdraw STR {str_id}",
                principal=principal.name,
            )
        if not request.is_open:
            raise TransitionError(f"STR {str_id} is already {request.status.value}")
        request = request.with_updates(
            status=TransitionStatus.WITHDRAWN, decision_note=reason or None
        )
        self.store.save(request)
        self.events.append(
            "str.withdrawn",
            actor=principal.name,
            task_id=request.task_id,
            payload={"str_id": str_id, "reason": reason},
        )
        return request

    # ------------------------------------------------------------------ read

    def get(self, str_id: str) -> StateTransitionRequest | None:
        return self.store.get(str_id)

    def require(self, str_id: str) -> StateTransitionRequest:
        request = self.store.get(str_id)
        if request is None:
            raise ResearchOSError(f"state transition request {str_id!r} not found")
        return request

    def pending(self) -> list[StateTransitionRequest]:
        return [r for r in self.store.all() if r.is_open]

    def history(self) -> list[StateTransitionRequest]:
        return sorted(self.store.all(), key=lambda r: r.created_at)

    def diff_preview(self, str_id: str) -> list[dict[str, Any]]:
        """Show what approving this STR would change, without changing it."""
        request = self.require(str_id)
        state = self.state.load(refresh=True)
        previewed = self.state.apply_operations(state, request.proposed_operations)
        rows: list[dict[str, Any]] = []
        for operation in request.proposed_operations:
            before = _dig(state, operation.path)
            after = _dig(previewed, operation.path)
            rows.append(
                {
                    "op": operation.op.value,
                    "path": operation.path,
                    "before": before,
                    "after": after,
                    "guarded": bool(guarded_root(operation.path)),
                }
            )
        return rows

    def hash_state(self) -> str:
        return guarded_state_hash(self.state.load())


def _dig(state: Any, path: str) -> Any:
    current: Any = state
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    if hasattr(current, "model_dump"):
        return current.model_dump(mode="json")
    return current


def make_op(path: str, value: Any, op: StateOp = StateOp.SET) -> StateOperation:
    return StateOperation(op=op, path=path, value=value)
