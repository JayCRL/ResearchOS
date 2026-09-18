"""Timeline recording — the append-only narrative of how the research actually happened.

Used to answer questions that a chat log cannot answer: *why was the original claim dropped?*,
*why did the experiment route change?*, *why was this control added?*, *why is this result no
longer in the paper?*
"""

from __future__ import annotations

from typing import Sequence

from ..models.common import utcnow
from ..models.timeline import TimelineEvent, TimelineEventKind
from .events import EventLog
from .store import EntityStore


class TimelineRecorder:
    def __init__(self, store: EntityStore[TimelineEvent], events: EventLog) -> None:
        self.store = store
        self.events = events

    def record(
        self,
        kind: TimelineEventKind | str,
        title: str,
        *,
        detail: str = "",
        actor: str = "human",
        task_id: str | None = None,
        refs: Sequence[str] = (),
        state_revision: int | None = None,
        imported: bool = False,
    ) -> TimelineEvent:
        event_kind = kind if isinstance(kind, TimelineEventKind) else TimelineEventKind(kind)
        event = TimelineEvent(
            kind=event_kind,
            title=title,
            detail=detail,
            actor=actor,
            task_id=task_id,
            refs=list(refs),
            state_revision=state_revision,
            imported=imported,
        )
        self.store.save(event)
        self.events.append(
            "timeline.recorded",
            actor=actor,
            task_id=task_id,
            payload={"event_id": event.event_id, "kind": event_kind.value, "title": title[:200]},
        )
        return event

    def all(self) -> list[TimelineEvent]:
        return sorted(self.store.all(), key=lambda e: e.at)

    def of_kind(self, *kinds: TimelineEventKind) -> list[TimelineEvent]:
        wanted = set(kinds)
        return [event for event in self.all() if event.kind in wanted]

    def for_ref(self, ref: str) -> list[TimelineEvent]:
        return [event for event in self.all() if ref in event.refs]

    def count(self) -> int:
        return self.store.count()

    def latest(self, n: int = 10) -> list[TimelineEvent]:
        return self.all()[-n:]

    def provenance_of_ref(self, ref: str) -> list[TimelineEvent]:
        """Everything that ever happened to one object, oldest first."""
        return sorted(self.for_ref(ref), key=lambda e: e.at)

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for event in self.all():
            out[event.kind.value] = out.get(event.kind.value, 0) + 1
        return out

    def now(self):
        return utcnow()
