"""The conflict ledger: ResearchOS's structured way of disagreeing with the user's own material.

When two sources disagree, ResearchOS does **not** pick one and move on quietly. It records a
:class:`Conflict`, keeps both sources, states the difference, and applies a deterministic trust
ordering as the *default* resolution:

    raw experiment (6) > verified analysis (5) > audit (4) > experiment report (3)
                      > paper prose (2) > AI summary (1)

An unresolved conflict blocks claim promotion to SUPPORTED/ROBUST, which is what makes the ledger
load-bearing rather than decorative.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..models.audit import Conflict, ConflictSource
from ..models.common import (
    ConflictKind,
    ConflictResolution,
    SOURCE_TRUST,
    SourceKind,
    utcnow,
)
from .errors import ConflictError, PermissionDenied, ResearchOSError
from .events import EventLog
from .permissions import Cap, Principal
from .store import EntityStore

#: Sources trusted for *numeric* facts, in descending order (mirrors SOURCE_TRUST).
TRUST_ORDER: tuple[SourceKind, ...] = tuple(
    sorted(SOURCE_TRUST, key=lambda kind: SOURCE_TRUST[kind], reverse=True)
)


class ConflictLedger:
    """Create, inspect and resolve conflicts without ever discarding a source."""

    def __init__(self, store: EntityStore[Conflict], events: EventLog) -> None:
        self.store = store
        self.events = events

    # ------------------------------------------------------------------ create

    def create(
        self,
        principal: Principal,
        *,
        kind: ConflictKind,
        subject: str,
        source_a: ConflictSource | Mapping[str, Any],
        source_b: ConflictSource | Mapping[str, Any],
        difference: str,
        claim_ids: Sequence[str] = (),
        experiment_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        auto_resolve_by_trust: bool = True,
        detected_by: str = "deterministic",
    ) -> Conflict:
        principal.require(Cap.CONFLICT_CREATE, "conflict.create")

        a = source_a if isinstance(source_a, ConflictSource) else ConflictSource.model_validate(source_a)
        b = source_b if isinstance(source_b, ConflictSource) else ConflictSource.model_validate(source_b)

        conflict = Conflict(
            kind=kind,
            subject=subject,
            source_a=a,
            source_b=b,
            difference=difference,
            created_by=principal.name,
            detected_by=detected_by,
            claim_ids=list(claim_ids),
            experiment_ids=list(experiment_ids),
            evidence_ids=list(evidence_ids),
        )
        if auto_resolve_by_trust:
            conflict = self._apply_trust_default(conflict)
        self.store.save(conflict)
        self.events.append(
            "conflict.created",
            actor=principal.name,
            payload={
                "conflict_id": conflict.conflict_id,
                "kind": kind.value,
                "subject": subject,
                "source_a": a.ref,
                "source_b": b.ref,
                "difference": difference[:400],
                "resolution": conflict.resolution.value,
                "auto": conflict.resolved_by == "kernel:trust-order",
            },
        )
        return conflict

    def _apply_trust_default(self, conflict: Conflict) -> Conflict:
        """Apply the deterministic trust ordering. Never deletes; marks the lower-trust source."""
        preferred = conflict.preferred_by_trust()
        if preferred is None:
            return conflict.with_updates(
                resolution=ConflictResolution.UNRESOLVED,
                resolution_reason=(
                    f"both sources have equal trust ({conflict.source_a.kind.value}); "
                    "a human must decide"
                ),
                blocks_claim_promotion=True,
            )
        other = conflict.source_b if preferred is conflict.source_a else conflict.source_a
        return conflict.with_updates(
            resolution=(
                ConflictResolution.A_WINS if preferred is conflict.source_a else ConflictResolution.B_WINS
            ),
            winning_ref=preferred.ref,
            superseded_ref=other.ref,
            resolved_by="kernel:trust-order",
            resolved_at=utcnow(),
            confidence=min(0.95, 0.5 + 0.05 * (preferred.trust - other.trust)),
            resolution_reason=(
                f"{preferred.kind.value} outranks {other.kind.value} in the provenance trust order "
                f"(trust {preferred.trust} vs {other.trust}); the lower-trust source is preserved and "
                "marked superseded"
            ),
            blocks_claim_promotion=False,
        )

    # ------------------------------------------------------------------ resolve

    def resolve(
        self,
        principal: Principal,
        conflict_id: str,
        *,
        resolution: ConflictResolution,
        reason: str,
        winning_ref: str | None = None,
        confidence: float = 0.8,
    ) -> Conflict:
        principal.require(Cap.CONFLICT_RESOLVE, f"conflict:{conflict_id}.resolve")
        conflict = self.require(conflict_id)

        if resolution is ConflictResolution.A_WINS:
            winning_ref = winning_ref or conflict.source_a.ref
        elif resolution is ConflictResolution.B_WINS:
            winning_ref = winning_ref or conflict.source_b.ref

        conflict = conflict.with_updates(
            resolution=resolution,
            resolution_reason=reason,
            resolved_by=principal.name,
            resolved_at=utcnow(),
            confidence=confidence,
            blocks_claim_promotion=resolution is ConflictResolution.UNRESOLVED,
        )

        if resolution in (ConflictResolution.A_WINS, ConflictResolution.B_WINS):
            if winning_ref not in (conflict.source_a.ref, conflict.source_b.ref):
                raise ConflictError(
                    f"winning_ref {winning_ref!r} is not one of the two sources; both are preserved "
                    "and the winner must be one of them"
                )
            conflict = conflict.with_updates(
                winning_ref=winning_ref,
                superseded_ref=(
                    conflict.source_b.ref if winning_ref == conflict.source_a.ref else conflict.source_a.ref
                ),
            )
        elif resolution in (ConflictResolution.BOTH_VALID, ConflictResolution.MERGED):
            conflict = conflict.with_updates(winning_ref=None, superseded_ref=None)

        self.store.save(conflict)
        self.events.append(
            "conflict.resolved",
            actor=principal.name,
            payload={
                "conflict_id": conflict_id,
                "resolution": resolution.value,
                "winning_ref": conflict.winning_ref,
                "reason": reason[:400],
            },
        )
        return conflict

    def escalate(self, principal: Principal, conflict_id: str, *, note: str = "") -> Conflict:
        """Mark a trusted-but-unresolved conflict as needing a human decision."""
        conflict = self.require(conflict_id)
        conflict = conflict.with_updates(
            blocks_claim_promotion=True,
            resolution_reason=(conflict.resolution_reason or "") + (f"\nESCALATED: {note}" if note else ""),
        )
        self.store.save(conflict)
        self.events.append(
            "conflict.escalated",
            actor=principal.name,
            payload={"conflict_id": conflict_id, "note": note},
        )
        return conflict

    # ------------------------------------------------------------------ read

    def require(self, conflict_id: str) -> Conflict:
        conflict = self.store.get(conflict_id)
        if conflict is None:
            raise ResearchOSError(f"conflict {conflict_id!r} not found")
        return conflict

    def unresolved(self) -> list[Conflict]:
        return [c for c in self.store.all() if c.is_unresolved()]

    def blocking(self) -> list[Conflict]:
        return [c for c in self.store.all() if c.blocks_claim_promotion]

    def for_claim(self, claim_id: str) -> list[Conflict]:
        return [c for c in self.store.all() if claim_id in c.claim_ids]

    def blocking_for_claim(self, claim_id: str) -> list[Conflict]:
        return [c for c in self.for_claim(claim_id) if c.blocks_claim_promotion]

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {r.value: 0 for r in ConflictResolution}
        for conflict in self.store.all():
            out[conflict.resolution.value] += 1
        out["blocking"] = len(self.blocking())
        return out

    def trust_table(self) -> list[tuple[str, int]]:
        return [(kind.value, SOURCE_TRUST[kind]) for kind in TRUST_ORDER]
