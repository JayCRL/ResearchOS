"""The append-only, hash-chained event log.

Every kernel mutation writes one event. Each event's digest includes the previous event's digest,
so removing, editing or reordering a historical event breaks the chain and is detected by
``researchos audit log verify``.

Known limitation: a chain with no external anchor cannot detect *tail* truncation (deleting the
last N lines leaves a valid prefix). ``EventLog.head_digest()`` exists so an external witness —
a git commit, a signed tag, or a published digest — can pin the head.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from pydantic import Field

from ..models.common import RosModel, canonical_json, sha256_text, utcnow

GENESIS_HASH = "0" * 64


class EventRecord(RosModel):
    """One immutable history entry."""

    seq: int = Field(ge=1)
    at: datetime = Field(default_factory=utcnow)
    kind: str = Field(min_length=1)
    actor: str = "system"
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    event_hash: str = ""

    def hashable(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("event_hash", None)
        return data

    def compute_hash(self) -> str:
        return sha256_text(self.prev_hash + canonical_json(self.hashable()))


class EventLog:
    """Append-only JSONL log with a SHA-256 hash chain."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_seq, self._last_hash = self._tail_state()

    # ------------------------------------------------------------------ reading

    def _tail_state(self) -> tuple[int, str]:
        seq, digest = 0, GENESIS_HASH
        if not self.path.exists():
            return seq, digest
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                seq = int(record.get("seq", seq))
                digest = str(record.get("event_hash", digest))
        return seq, digest

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def __iter__(self) -> Iterator[EventRecord]:
        return self.iter_records()

    def iter_records(self) -> Iterator[EventRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield EventRecord.model_validate(json.loads(stripped))

    def head_digest(self) -> str:
        return self._last_hash

    def head_seq(self) -> int:
        return self._last_seq

    # ------------------------------------------------------------------ writing

    def append(
        self,
        kind: str,
        *,
        actor: str = "system",
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        record = EventRecord(
            seq=self._last_seq + 1,
            kind=kind,
            actor=actor,
            task_id=task_id,
            payload=payload or {},
            prev_hash=self._last_hash,
        )
        record.event_hash = record.compute_hash()
        line = canonical_json(record.model_dump(mode="json"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._last_seq = record.seq
        self._last_hash = record.event_hash
        return record

    # ------------------------------------------------------------------ verification

    def verify(self) -> "ChainVerification":
        """Recompute the chain. Returns a report rather than raising, so it can be displayed."""
        problems: list[str] = []
        expected_seq = 1
        expected_prev = GENESIS_HASH
        count = 0
        for record in self.iter_records():
            count += 1
            if record.seq != expected_seq:
                problems.append(
                    f"sequence break at line {count}: expected seq {expected_seq}, found {record.seq}"
                )
                expected_seq = record.seq
            if record.prev_hash != expected_prev:
                problems.append(
                    f"hash-chain break at seq {record.seq}: prev_hash {record.prev_hash[:12]}… "
                    f"does not match previous event_hash {expected_prev[:12]}…"
                )
            recomputed = record.compute_hash()
            if recomputed != record.event_hash:
                problems.append(
                    f"tampered content at seq {record.seq}: stored hash {record.event_hash[:12]}… "
                    f"recomputes to {recomputed[:12]}…"
                )
                expected_prev = recomputed
            else:
                expected_prev = record.event_hash
            expected_seq = record.seq + 1
        return ChainVerification(
            ok=not problems,
            events_checked=count,
            problems=problems,
            head_digest=self._last_hash,
            head_seq=self._last_seq,
        )

    def events_of_kind(self, *kinds: str) -> list[EventRecord]:
        wanted = set(kinds)
        return [r for r in self.iter_records() if r.kind in wanted]


class ChainVerification(RosModel):
    ok: bool
    events_checked: int = 0
    problems: list[str] = Field(default_factory=list)
    head_digest: str = ""
    head_seq: int = 0

    def summary(self) -> str:
        if self.ok:
            return f"event log intact: {self.events_checked} events, head {self.head_digest[:12]}…"
        return f"event log BROKEN: {len(self.problems)} problem(s) in {self.events_checked} events"
