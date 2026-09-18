"""The claim lifecycle: every rung of the ladder has an obligation attached.

```
IDEA → HYPOTHESIS → TESTED → SUPPORTED → ROBUST
                 ↘          ↘          ↘   WEAKENED / REJECTED / SUPERSEDED
```

The two rules that matter most:

* **``HYPOTHESIS → SUPPORTED`` is refused.** Skipping the testing rung is the single most common
  way a research claim becomes false in a paper.
* **Nothing is promoted by an agent alone.** ``SUPPORTED`` and ``ROBUST`` require the human-held
  ``claim.approve`` capability, and an unresolved conflict blocks promotion outright.
"""

from __future__ import annotations

from typing import Sequence

from ..kernel.errors import ClaimTransitionError, PermissionDenied, ResearchOSError
from ..kernel.kernel import ResearchKernel
from ..kernel.levels import (
    aggregate_level,
    independent_experiments,
    level_of_experiment,
    replication_count,
)
from ..kernel.permissions import Cap, Principal
from ..models.claim import Claim, ClaimHistoryEntry, RejectionBasis
from ..models.common import ClaimStatus, EvidenceLevel, EvidenceType, ExperimentStatus, utcnow
from ..models.timeline import TimelineEventKind
from .language import permitted_verb

#: Legal transitions. Absence from this table is a refusal with a specific message.
LEGAL_TRANSITIONS: dict[ClaimStatus, frozenset[ClaimStatus]] = {
    ClaimStatus.IDEA: frozenset(
        {ClaimStatus.HYPOTHESIS, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED, ClaimStatus.WEAKENED}
    ),
    ClaimStatus.HYPOTHESIS: frozenset(
        {ClaimStatus.TESTED, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED, ClaimStatus.WEAKENED}
    ),
    ClaimStatus.TESTED: frozenset(
        {
            ClaimStatus.SUPPORTED,
            ClaimStatus.WEAKENED,
            ClaimStatus.REJECTED,
            ClaimStatus.SUPERSEDED,
        }
    ),
    ClaimStatus.SUPPORTED: frozenset(
        {ClaimStatus.ROBUST, ClaimStatus.WEAKENED, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED}
    ),
    ClaimStatus.ROBUST: frozenset(
        {ClaimStatus.WEAKENED, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED}
    ),
    # A weakened claim may be re-tested, superseded or finally rejected — never silently un-weakened.
    ClaimStatus.WEAKENED: frozenset(
        {ClaimStatus.TESTED, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED}
    ),
    # A rejected claim is a tombstone: it can only be explained by a successor.
    ClaimStatus.REJECTED: frozenset({ClaimStatus.SUPERSEDED}),
    ClaimStatus.SUPERSEDED: frozenset(),
}

#: Transitions that require the human-held approval capability.
HUMAN_GATED_TARGETS: frozenset[ClaimStatus] = frozenset({ClaimStatus.SUPPORTED, ClaimStatus.ROBUST})

#: Capability required to *attempt* a transition.
CAPABILITY_BY_TARGET: dict[ClaimStatus, Cap] = {
    ClaimStatus.REJECTED: Cap.CLAIM_REJECT,
    ClaimStatus.SUPERSEDED: Cap.CLAIM_SUPERSEDE,
    ClaimStatus.SUPPORTED: Cap.CLAIM_APPROVE,
    ClaimStatus.ROBUST: Cap.CLAIM_APPROVE,
}

#: Minimum independent experiments for ROBUST.
ROBUST_MIN_EXPERIMENTS = 2
#: Minimum evidence level for SUPPORTED.
SUPPORTED_MIN_LEVEL = EvidenceLevel.L2_REPRODUCED
#: Minimum evidence level for ROBUST.
ROBUST_MIN_LEVEL = EvidenceLevel.L3_CONTROLLED


class ClaimLifecycle:
    """Creates claims and moves them along the ladder, enforcing each rung's obligation."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ================================================================== create

    def create(
        self,
        principal: Principal,
        *,
        statement: str,
        scope: str,
        parent_question: str | None = None,
        evidence_ids: Sequence[str] = (),
        literature_ids: Sequence[str] = (),
        supporting_experiments: Sequence[str] = (),
        contradicting_experiments: Sequence[str] = (),
        assumptions: Sequence[str] = (),
        limitations: Sequence[str] = (),
        is_core: bool = False,
        task_id: str | None = None,
        status: ClaimStatus = ClaimStatus.IDEA,
    ) -> Claim:
        principal.require(Cap.CLAIM_PROPOSE, "claim.create")
        if status is not ClaimStatus.IDEA:
            raise ClaimTransitionError(
                "claims are created as IDEA; promote them through the lifecycle so each rung's "
                "obligation is checked and recorded"
            )
        claim = Claim(
            statement=statement,
            status=ClaimStatus.IDEA,
            scope=scope,
            parent_question=parent_question,
            evidence_ids=list(evidence_ids),
            literature_ids=list(literature_ids),
            supporting_experiments=list(supporting_experiments),
            contradicting_experiments=list(contradicting_experiments),
            assumptions=list(assumptions),
            limitations=list(limitations),
            is_core=is_core,
            created_by=principal.name,
            task_id=task_id,
            history=[
                ClaimHistoryEntry(
                    by=principal.name, from_status=None, to_status=ClaimStatus.IDEA,
                    reason="claim created", evidence_ids=list(evidence_ids),
                )
            ],
        )
        self.kernel.claims.save(claim)
        self.kernel.events.append(
            "claim.created",
            actor=principal.name,
            task_id=task_id,
            payload={"claim_id": claim.claim_id, "statement": statement[:300], "scope": scope[:200]},
        )
        self.kernel.timeline.record(
            TimelineEventKind.CLAIM_CREATED,
            f"Claim created: {statement[:120]}",
            actor=principal.name,
            task_id=task_id,
            refs=[claim.claim_id],
            state_revision=self.kernel.state.revision(),
        )
        return claim

    # ================================================================== gates

    def obligations(self, claim: Claim, to_status: ClaimStatus) -> list[str]:
        """Everything missing for this transition. An empty list means the rung is earned."""
        unmet: list[str] = []

        if to_status is ClaimStatus.HYPOTHESIS:
            if not claim.scope.strip():
                unmet.append("scope must be declared before a claim becomes a testable hypothesis")
            if not claim.statement.strip():
                unmet.append("statement must be non-empty")

        if to_status is ClaimStatus.TESTED:
            experiments = self._experiments(claim)
            attempted = [
                e
                for e in experiments
                if e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.ABORTED)
            ]
            if not attempted:
                unmet.append(
                    "TESTED requires at least one experiment that has actually run "
                    "(COMPLETED/FAILED/ABORTED); planned experiments are not tests"
                )

        if to_status is ClaimStatus.SUPPORTED:
            if claim.status is ClaimStatus.HYPOTHESIS:
                unmet.append(
                    "HYPOTHESIS -> SUPPORTED is not a legal transition: the claim must be TESTED first "
                    "(and its evidence assessed) before it can be supported"
                )
            evidence = self._evidence(claim)
            if not evidence:
                unmet.append("SUPPORTED requires at least one evidence record")
            level = self._level(claim)
            if level.rank < SUPPORTED_MIN_LEVEL.rank:
                unmet.append(
                    f"SUPPORTED requires evidence at {SUPPORTED_MIN_LEVEL.value} or above; "
                    f"this claim's evidence is {level.value}"
                )
            if not self._analyses(claim):
                unmet.append("SUPPORTED requires at least one analysis artifact (numbers, not prose)")
            blocking = self.kernel.ledger.blocking_for_claim(claim.claim_id)
            if blocking:
                unmet.append(
                    "unresolved conflicts block promotion: "
                    + "; ".join(c.difference[:120] for c in blocking)
                )
            contradicting = self._contradicting(claim)
            if contradicting:
                unmet.append(
                    f"contradicting experiment(s) present ({', '.join(contradicting)}); "
                    "the claim must be narrowed or rejected first"
                )

        if to_status is ClaimStatus.ROBUST:
            experiments = self._experiments(claim)
            independent = independent_experiments(experiments)
            if len(independent) < ROBUST_MIN_EXPERIMENTS:
                unmet.append(
                    f"ROBUST requires at least {ROBUST_MIN_EXPERIMENTS} *independent* experiments "
                    f"(different design/dataset/scale); found {len(independent)} "
                    f"across {replication_count(experiments)} run(s)"
                )
            level = self._level(claim)
            if level.rank < ROBUST_MIN_LEVEL.rank:
                unmet.append(
                    f"ROBUST requires evidence at {ROBUST_MIN_LEVEL.value} or above; "
                    f"this claim's evidence is {level.value}"
                )
            audits = [
                a
                for a in self.kernel.audits.all()
                if claim.claim_id in a.subjects
                and a.kind.value in {"MECHANISM", "STATISTICAL"}
            ]
            if not audits:
                unmet.append(
                    "ROBUST requires at least one mechanism or statistical audit attached to the claim"
                )
            blocking = self.kernel.ledger.blocking_for_claim(claim.claim_id)
            if blocking:
                unmet.append(
                    "unresolved conflicts block promotion: "
                    + "; ".join(c.difference[:120] for c in blocking)
                )

        if to_status is ClaimStatus.REJECTED:
            if not self._rejection_reason(claim):
                unmet.append("a rejection must state why (pass reason=...)")

        if to_status is ClaimStatus.SUPERSEDED and not claim.superseded_by:
            unmet.append("SUPERSEDED requires superseded_by to name the successor claim")

        return unmet

    # ================================================================== transition

    def transition(
        self,
        principal: Principal,
        claim_id: str,
        to_status: ClaimStatus,
        *,
        reason: str = "",
        evidence_ids: Sequence[str] = (),
        contradiction_ids: Sequence[str] = (),
        superseded_by: str | None = None,
        rejection_basis: RejectionBasis | None = None,
        str_id: str | None = None,
        language_strength: str | None = None,
    ) -> Claim:
        claim = self.kernel.claims.get(claim_id)
        if claim is None:
            raise ResearchOSError(f"claim {claim_id!r} not found")

        to_status = ClaimStatus(to_status) if isinstance(to_status, str) else to_status
        required = CAPABILITY_BY_TARGET.get(to_status, Cap.CLAIM_PROPOSE)
        principal.require(required, f"claim:{claim_id}:{to_status.value}")

        legal = LEGAL_TRANSITIONS.get(claim.status, frozenset())
        if to_status not in legal:
            raise ClaimTransitionError(
                f"{claim.status.value} -> {to_status.value} is not a legal claim transition. "
                f"From {claim.status.value} you may go to "
                f"{sorted(s.value for s in legal) or 'nowhere (terminal state)'}."
            )

        # materialise the fields the target status needs *before* validating obligations
        patch: dict[str, object] = {}
        if superseded_by:
            patch["superseded_by"] = superseded_by
        if to_status in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED):
            # These two gates do not depend on the claim's previous status, so the target status can
            # be part of the same validated update (it keeps the tombstone fields consistent).
            patch["status"] = to_status
        if to_status is ClaimStatus.REJECTED:
            patch["rejection_reason"] = reason or claim.rejection_reason
            patch["rejection_basis"] = rejection_basis or claim.rejection_basis or RejectionBasis.OTHER
            patch["rejected_at"] = utcnow()
            patch["tombstone"] = True
        if evidence_ids:
            patch["evidence_ids"] = [*claim.evidence_ids, *[e for e in evidence_ids if e not in claim.evidence_ids]]
        if contradiction_ids:
            patch["contradicting_experiments"] = [
                *claim.contradicting_experiments,
                *[e for e in contradiction_ids if e not in claim.contradicting_experiments],
            ]
        candidate = claim.with_updates(**patch) if patch else claim

        unmet = self.obligations(candidate, to_status)
        if unmet:
            raise ClaimTransitionError(
                f"claim {claim_id} cannot move to {to_status.value}: " + "; ".join(unmet)
            )

        level = self._level(candidate)
        updated = candidate.with_updates(
            status=to_status,
            # The evidence level is *derived and persisted* on every transition, so downstream readers
            # (paper compiler, style auditor, language calibration) never have to guess it.
            evidence_level=level,
            updated_at=utcnow(),
            language_strength=language_strength or permitted_verb(level),
            history=[
                *candidate.history,
                ClaimHistoryEntry(
                    by=principal.name,
                    from_status=claim.status,
                    to_status=to_status,
                    reason=reason,
                    evidence_ids=list(evidence_ids),
                    str_id=str_id,
                ),
            ],
        )
        self.kernel.claims.save(updated)

        if to_status in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED):
            state = self.kernel.research_state()
            if claim.claim_id not in state.rejected_claims:
                self.kernel.state.update(
                    principal,
                    lambda s: setattr(s, "rejected_claims", [*s.rejected_claims, claim.claim_id]),
                    event_kind="claim.retired",
                    payload={"claim_id": claim.claim_id, "status": to_status.value, "reason": reason[:300]},
                )

        self.kernel.events.append(
            "claim.transition",
            actor=principal.name,
            task_id=claim.task_id,
            payload={
                "claim_id": claim_id,
                "from": claim.status.value,
                "to": to_status.value,
                "reason": reason[:400],
                "evidence_level": level.value,
                "obligations_checked": ["scope", "experiments", "evidence", "level", "analyses", "conflicts"],
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.CLAIM_REJECTED if to_status is ClaimStatus.REJECTED else TimelineEventKind.CLAIM_REVISION,
            f"Claim {claim.status.value} -> {to_status.value}: {claim.statement[:90]}",
            detail=reason,
            actor=principal.name,
            task_id=claim.task_id,
            refs=[claim_id],
            state_revision=self.kernel.state.revision(),
        )
        return updated

    # ================================================================== helpers

    def _experiments(self, claim: Claim) -> list:
        ids = {*claim.supporting_experiments, *claim.contradicting_experiments}
        return [e for e in (self.kernel.experiments.get(i) for i in sorted(ids)) if e is not None]

    def _evidence(self, claim: Claim) -> list:
        return [e for e in (self.kernel.evidence.get(i) for i in claim.evidence_ids) if e is not None]

    def _analyses(self, claim: Claim) -> list:
        found = []
        for evidence in self._evidence(claim):
            if evidence.source_analysis:
                analysis = self.kernel.analyses.get(evidence.source_analysis)
                if analysis is not None:
                    found.append(analysis)
        for experiment in self._experiments(claim):
            for analysis_id in experiment.analysis_ids:
                analysis = self.kernel.analyses.get(analysis_id)
                if analysis is not None:
                    found.append(analysis)
        return found

    def _level(self, claim: Claim) -> EvidenceLevel:
        """The level this claim's *evidence* actually supports — never the level it wishes for."""
        evidence = self._evidence(claim)
        if not evidence:
            return EvidenceLevel.L0_IDEA
        experiments = {e.experiment_id: e for e in self.kernel.experiments.all()}
        analyses = {a.analysis_id: a for a in self.kernel.analyses.all()}
        return aggregate_level(evidence, experiments, analyses)

    def _contradicting(self, claim: Claim) -> list[str]:
        return [
            e.experiment_id
            for e in (self.kernel.experiments.get(i) for i in claim.contradicting_experiments)
            if e is not None and e.status is ExperimentStatus.COMPLETED
        ]

    @staticmethod
    def _rejection_reason(claim: Claim) -> str:
        return (claim.rejection_reason or "").strip()

    # ================================================================== reporting

    def explain(self, claim_id: str) -> dict[str, object]:
        """Why is this claim where it is, and what would move it forward?"""
        claim = self.kernel.claims.get(claim_id)
        if claim is None:
            raise ResearchOSError(f"claim {claim_id!r} not found")
        level = self._level(claim)
        next_steps: dict[str, list[str]] = {}
        for target in sorted(LEGAL_TRANSITIONS.get(claim.status, frozenset()), key=lambda s: s.value):
            if target in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED):
                continue
            unmet = self.obligations(claim, target)
            next_steps[target.value] = unmet or ["all obligations met"]
        return {
            "claim_id": claim_id,
            "status": claim.status.value,
            "evidence_level": level.value,
            "permitted_language": permitted_verb(level),
            "evidence_count": len(claim.evidence_ids),
            "experiment_count": len(claim.supporting_experiments),
            "blocking_conflicts": [c.conflict_id for c in self.kernel.ledger.blocking_for_claim(claim_id)],
            "next_transitions": next_steps,
            "history": [h.model_dump(mode="json") for h in claim.history],
        }
