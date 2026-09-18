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


#: The phases a real research project moves through. Used to group the timeline into a narrative
#: instead of a log dump.
NARRATIVE_PHASES: tuple[tuple[str, tuple[TimelineEventKind, ...]], ...] = (
    (
        "observation",
        (
            TimelineEventKind.PROJECT_CREATED,
            TimelineEventKind.IMPORT,
            TimelineEventKind.IDEA,
            TimelineEventKind.LITERATURE,
        ),
    ),
    ("hypothesis", (TimelineEventKind.TASK_CREATED, TimelineEventKind.FINDING)),
    (
        "experiment",
        (
            TimelineEventKind.EXPERIMENT_REGISTERED,
            TimelineEventKind.EXPERIMENT_STARTED,
            TimelineEventKind.EXPERIMENT_RESULT,
            TimelineEventKind.EXPERIMENT_FAILED,
            TimelineEventKind.ANOMALY,
        ),
    ),
    ("analysis", (TimelineEventKind.ANALYSIS, TimelineEventKind.EVIDENCE, TimelineEventKind.VERIFICATION)),
    (
        "revision",
        (
            TimelineEventKind.AUDIT,
            TimelineEventKind.CLAIM_REVISION,
            TimelineEventKind.CLAIM_REJECTED,
            TimelineEventKind.STATE_TRANSITION,
            TimelineEventKind.DECISION,
        ),
    ),
    ("claim", (TimelineEventKind.CLAIM_CREATED, TimelineEventKind.SKILL)),
    ("paper", (TimelineEventKind.PAPER,)),
    ("housekeeping", (TimelineEventKind.TASK_CLOSED,)),
)


class TimelineView:
    """Read-only views over the research history.

    This is what lets a reader ask the four questions a chat log cannot answer:

    * why was the original claim cancelled?
    * why did the experiment route change?
    * why was this control added?
    * why is this result no longer in the paper?

    Everything here is assembled from *records* — decisions, claim history, transitions, events — never
    from a narrative the model produced afterwards.
    """

    def __init__(self, kernel: "ResearchKernel") -> None:
        self.kernel = kernel
        self.recorder = kernel.timeline

    # ------------------------------------------------------------------ views

    def all(self) -> list[TimelineEvent]:
        return self.recorder.all()

    def narrative(self) -> list[tuple[str, list[TimelineEvent]]]:
        """Timeline grouped into the phases a research project actually passes through."""
        events = self.all()
        assigned: set[str] = set()
        phases: list[tuple[str, list[TimelineEvent]]] = []
        for phase, kinds in NARRATIVE_PHASES:
            bucket = [event for event in events if event.kind in kinds and event.event_id not in assigned]
            for event in bucket:
                assigned.add(event.event_id)
            if bucket:
                phases.append((phase, bucket))
        leftovers = [event for event in events if event.event_id not in assigned]
        if leftovers:
            phases.append(("other", leftovers))
        return phases

    def provenance_of(self, ref: str) -> list[TimelineEvent]:
        return self.recorder.provenance_of_ref(ref)

    def claim_history(self, claim_id: str) -> dict[str, object]:
        """Everything that ever happened to one claim, in order, with the reasons."""
        claim = self.kernel.claims.get(claim_id)
        if claim is None:
            return {"claim_id": claim_id, "error": "claim not found"}
        steps: list[dict[str, object]] = []
        for entry in claim.history:
            steps.append(
                {
                    "at": entry.at.isoformat(),
                    "by": entry.by,
                    "from": entry.from_status.value if entry.from_status else None,
                    "to": entry.to_status.value,
                    "reason": entry.reason,
                    "evidence_ids": entry.evidence_ids,
                    "str_id": entry.str_id,
                }
            )
        return {
            "claim_id": claim_id,
            "statement": claim.statement,
            "status": claim.status.value,
            "rejection_reason": claim.rejection_reason,
            "superseded_by": claim.superseded_by,
            "steps": steps,
            "decisions": [
                {"decision_id": d.decision_id, "kind": d.kind.value, "summary": d.summary, "rationale": d.rationale}
                for d in self.kernel.decisions.all()
                if claim_id in d.affected_claims
            ],
            "conflicts": [
                {"conflict_id": c.conflict_id, "difference": c.difference, "resolution": c.resolution.value}
                for c in self.kernel.conflicts.all()
                if claim_id in c.claim_ids
            ],
            "events": [
                {"seq": e.seq, "kind": e.kind, "at": e.at.isoformat(), "payload": e.payload}
                for e in self.kernel.events.iter_records()
                if e.payload.get("claim_id") == claim_id
            ],
        }

    def why(self, *, ref: str | None = None, text: str | None = None) -> list[dict[str, object]]:
        """Explain a change: find the decision, transition or audit that caused it."""
        needle = (text or "").lower()
        found: list[dict[str, object]] = []
        for decision in self.kernel.decisions.all():
            haystack = f"{decision.summary} {decision.rationale} {' '.join(decision.consequences)}".lower()
            hits_ref = ref is not None and (ref in decision.affected_claims or ref in decision.affected_experiments or ref == decision.str_id)
            if hits_ref or (needle and needle in haystack):
                found.append(
                    {
                        "kind": "DECISION",
                        "at": decision.created_at.isoformat(),
                        "id": decision.decision_id,
                        "summary": decision.summary,
                        "rationale": decision.rationale,
                        "alternatives_considered": decision.alternatives_considered,
                        "rejected_alternatives": decision.rejected_alternatives,
                        "consequences": decision.consequences,
                        "evidence_ids": decision.evidence_ids,
                    }
                )
        for request in self.kernel.transitions.history():
            hits_ref = ref is not None and (ref == request.str_id or ref in request.affected_claims + request.affected_experiments)
            if hits_ref or (needle and needle in request.reason.lower()):
                found.append(
                    {
                        "kind": "STATE_TRANSITION",
                        "at": request.created_at.isoformat(),
                        "id": request.str_id,
                        "summary": f"{request.change_class.value} change ({request.status.value})",
                        "rationale": request.reason,
                        "evidence_ids": request.evidence_ids,
                        "operations": [op.model_dump(mode="json") for op in request.proposed_operations],
                        "decided_by": request.decided_by,
                        "decision_id": request.decision_id,
                    }
                )
        if ref:
            for event in self.provenance_of(ref):
                found.append(
                    {
                        "kind": "TIMELINE",
                        "at": event.at.isoformat(),
                        "id": event.event_id,
                        "summary": event.title,
                        "rationale": event.detail,
                    }
                )
        found.sort(key=lambda item: str(item.get("at")))
        return found

    def answer(self, question: str) -> dict[str, object]:
        """Route one of the four standing questions to the records that answer it."""
        lowered = question.lower()
        if "cancel" in lowered or "reject" in lowered or "drop" in lowered:
            rejected = [c for c in self.kernel.claims.all() if c.status.value in {"REJECTED", "SUPERSEDED"}]
            return {
                "question": question,
                "answer": (
                    "Rejected and superseded claims are retained as tombstones with their stated basis "
                    "and the decision record that retired them."
                ),
                "claims": [
                    {
                        "claim_id": c.claim_id,
                        "statement": c.statement,
                        "status": c.status.value,
                        "basis": c.rejection_basis.value if c.rejection_basis else None,
                        "reason": c.rejection_reason,
                        "history": self.claim_history(c.claim_id)["steps"],
                    }
                    for c in rejected
                ],
            }
        if "route" in lowered or "experiment" in lowered and "change" in lowered:
            return {
                "question": question,
                "answer": "Experiment-route changes are recorded as decisions with their alternatives.",
                "records": [
                    {
                        "decision_id": d.decision_id,
                        "summary": d.summary,
                        "rationale": d.rationale,
                        "alternatives": d.alternatives_considered,
                        "consequences": d.consequences,
                    }
                    for d in self.kernel.decisions.all()
                    if d.kind.value in {"EXPERIMENT_ROUTE", "CONTROL_ADDED", "OTHER"}
                ],
            }
        if "control" in lowered:
            return {
                "question": question,
                "answer": "Controls are visible as declared arms plus the decisions that added them.",
                "experiments": [
                    {
                        "experiment_id": e.experiment_id,
                        "title": e.title,
                        "control": e.control.name if e.control else None,
                        "matched_conditions": e.matched_conditions,
                        "intervention": e.design.has_intervention,
                        "level": e.evidence_level().value,
                    }
                    for e in self.kernel.experiments.all()
                ],
            }
        if "no longer" in lowered or "excluded" in lowered or "not in the paper" in lowered:
            papers = self.kernel.paper_artifacts.all()
            return {
                "question": question,
                "answer": (
                    "A claim only appears in a compiled paper if it is SUPPORTED or ROBUST; everything "
                    "else is visibly excluded."
                ),
                "included": sorted({cid for p in papers for cid in p.claim_ids}),
                "excluded": [
                    {"claim_id": c.claim_id, "status": c.status.value, "statement": c.statement[:120]}
                    for c in self.kernel.claims.all()
                    if c.status.value not in {"SUPPORTED", "ROBUST"}
                ],
            }
        return {
            "question": question,
            "answer": "Unrecognised question: showing the full timeline instead.",
            "narrative": [
                {"phase": phase, "events": [event.title for event in events]}
                for phase, events in self.narrative()
            ],
        }

    def summary(self) -> dict[str, object]:
        stats = self.recorder.stats()
        return {
            "events": sum(stats.values()),
            "by_kind": stats,
            "imported": sum(1 for event in self.all() if event.imported),
            "phases": [phase for phase, _events in self.narrative()],
            "first": self.all()[0].at.isoformat() if self.all() else None,
            "last": self.all()[-1].at.isoformat() if self.all() else None,
        }
