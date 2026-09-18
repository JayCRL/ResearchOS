"""Research-state management: load, guard, apply, save.

The central idea: a mutation is applied to a *copy*, then the guarded roots are hashed before and
after. If a guarded root moved without an approved transition, the write is refused. Because the
check is a hash comparison rather than a review of calling code, **no code path can bypass it** —
including code written later by someone who has not read this file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from pydantic import ValidationError

from ..models.common import canonical_json, sha256_json, utcnow
from ..models.decision import GUARDED_PATHS, StateOperation, StateOp, guarded_root, guarded_state_hash
from ..models.research_state import ResearchState
from .errors import ResearchOSError, SilentStateChangeError, StateRevisionConflict
from .events import EventLog
from .paths import ProjectPaths
from .permissions import Principal
from .store import YamlIO


def _coerce(value: Any, current: Any) -> Any:
    """Coerce a raw YAML/JSON value into the shape the current field expects."""
    if hasattr(current, "model_validate") and isinstance(value, Mapping):
        return type(current).model_validate(dict(value))
    if isinstance(current, list) and isinstance(value, list):
        return value
    return value


class StateManager:
    """Owns ``state/research_state.yaml``."""

    def __init__(self, paths: ProjectPaths, events: EventLog) -> None:
        self.paths = paths
        self.events = events
        self._cache: ResearchState | None = None

    # ------------------------------------------------------------------ io

    @property
    def file(self) -> Path:
        return self.paths.research_state_file

    def load(self, *, refresh: bool = False) -> ResearchState:
        if self._cache is not None and not refresh:
            return self._cache
        if not self.file.is_file():
            state = ResearchState()
        else:
            try:
                state = ResearchState.model_validate(YamlIO.read(self.file))
            except ValidationError as exc:
                raise ResearchOSError(
                    f"{self.file} does not satisfy the research-state schema: {exc}"
                ) from exc
        self._cache = state
        return state

    def invalidate(self) -> None:
        self._cache = None

    def revision(self) -> int:
        return self.load().revision

    def state_hash(self) -> str:
        return guarded_state_hash(self.load())

    def guarded_snapshot(self) -> dict[str, str]:
        return {path: sha256_json(getattr(self.load(), path)) for path in GUARDED_PATHS}

    def save(
        self,
        state: ResearchState,
        *,
        actor: str,
        expected_revision: int | None = None,
        event_kind: str = "state.updated",
        payload: Mapping[str, Any] | None = None,
        task_id: str | None = None,
        bump_revision: bool = True,
        watch_guarded: bool = True,
    ) -> ResearchState:
        """Persist ``state``, with optional optimistic-concurrency and guard checks."""
        current = self.load(refresh=True)
        if expected_revision is not None and current.revision != expected_revision:
            raise StateRevisionConflict(
                f"research state is at revision {current.revision}, expected {expected_revision}; "
                "someone else changed it — reload and retry",
                principal=actor,
                resource=str(self.file),
            )

        if watch_guarded:
            before = guarded_state_hash(current)
            after = guarded_state_hash(state)
            if before != after:
                moved = [
                    path
                    for path in GUARDED_PATHS
                    if sha256_json(getattr(current, path)) != sha256_json(getattr(state, path))
                ]
                raise SilentStateChangeError(
                    "refusing a silent change to guarded research state: "
                    + ", ".join(moved)
                    + ". Changes to the core question, core claims, priorities, non-goals or scope "
                    "require an approved State Transition Request.",
                    path=", ".join(moved),
                    principal=actor,
                    task_id=task_id,
                )

        if bump_revision:
            state.revision = current.revision + 1
        state.updated_at = utcnow()
        state.updated_by = actor
        YamlIO.write_atomic(self.file, YamlIO.dump(state))
        self._cache = state
        self.events.append(
            event_kind,
            actor=actor,
            task_id=task_id,
            payload={
                "revision": state.revision,
                "state_hash": guarded_state_hash(state),
                **(dict(payload) if payload else {}),
            },
        )
        return state

    # ------------------------------------------------------------------ mutation

    def update(
        self,
        actor: Principal | str,
        mutate: Callable[[ResearchState], None],
        *,
        event_kind: str = "state.updated",
        payload: Mapping[str, Any] | None = None,
        task_id: str | None = None,
        expected_revision: int | None = None,
        allow_guarded: bool = False,
    ) -> ResearchState:
        """Apply ``mutate`` to a copy of the state and persist it if the guards allow.

        ``allow_guarded`` is reserved for :class:`~researchos.kernel.transitions.TransitionManager`,
        which has already verified an approved STR.
        """
        actor_name = actor.name if isinstance(actor, Principal) else actor
        state = self.load(refresh=True).model_copy(deep=True)
        mutate(state)
        return self.save(
            state,
            actor=actor_name,
            expected_revision=expected_revision,
            event_kind=event_kind,
            payload=payload,
            task_id=task_id,
            watch_guarded=not allow_guarded,
        )

    # ------------------------------------------------------------------ operations

    @staticmethod
    def _resolve(state: ResearchState, path: str) -> tuple[Any, str]:
        parts = path.split(".")
        target: Any = state
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ResearchOSError(f"unknown research-state path segment {part!r} in {path!r}")
            target = getattr(target, part)
            if target is None:
                raise ResearchOSError(
                    f"cannot address {path!r}: {part!r} is empty; set it first"
                )
        return target, parts[-1]

    def apply_operation(self, state: ResearchState, operation: StateOperation) -> None:
        """Apply one state operation in place (validation happens via pydantic assignment)."""
        target, attr = self._resolve(state, operation.path)
        current = getattr(target, attr, None)

        if operation.op is StateOp.SET:
            setattr(target, attr, _coerce(operation.value, current))
        elif operation.op is StateOp.APPEND:
            if not isinstance(current, list):
                raise ResearchOSError(f"append requires a list at {operation.path!r}")
            setattr(target, attr, [*current, *_as_list(operation.value)])
        elif operation.op is StateOp.REMOVE:
            if not isinstance(current, list):
                raise ResearchOSError(f"remove requires a list at {operation.path!r}")
            for item in _as_list(operation.value):
                if item in current:
                    current.remove(item)
            setattr(target, attr, list(current))
        elif operation.op is StateOp.MERGE:
            if not isinstance(current, Mapping):
                raise ResearchOSError(f"merge requires a mapping at {operation.path!r}")
            if not isinstance(operation.value, Mapping):
                raise ResearchOSError("merge value must be a mapping")
            merged = {**current, **dict(operation.value)}
            setattr(target, attr, merged)
        else:  # pragma: no cover - exhaustive enum
            raise ResearchOSError(f"unsupported state operation {operation.op!r}")

    def apply_operations(
        self, state: ResearchState, operations: Iterable[StateOperation]
    ) -> ResearchState:
        working = state.model_copy(deep=True)
        for operation in operations:
            self.apply_operation(working, operation)
        return working

    # ------------------------------------------------------------------ guards

    def touches_guarded(self, operations: Iterable[StateOperation]) -> list[str]:
        return sorted({guarded_root(op.path) for op in operations if guarded_root(op.path)})

    def describe_guarded(self) -> dict[str, Any]:
        state = self.load()
        return {
            "core_question": state.core_question.statement if state.core_question else None,
            "core_claims": list(state.core_claims),
            "priorities": [p.statement for p in state.priorities],
            "non_goals": list(state.non_goals),
            "hash": guarded_state_hash(state),
        }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def snapshot_line(state: ResearchState) -> str:
    """Compact one-line summary used in event payloads and CLI output."""
    question = state.core_question.statement if state.core_question else "(no core question)"
    return (
        f"rev {state.revision} | {question[:60]} | "
        f"{len(state.core_claims)} core claims | {len(state.open_questions)} open questions"
    )


def canonical_state_digest(state: ResearchState) -> str:
    return canonical_json(state.model_dump(mode="json"))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
