"""The Research Cockpit: translating the research state machine into what a researcher needs to read.

The audit console answers *"is this system trustworthy?"* — hashes, provenance, event head, integrity.
This module answers the five questions a researcher actually opens the tool with:

1. What am I researching?
2. What do I already know?
3. What can I not trust yet?
4. Why can I not continue?
5. What do I do next?

Everything here is **derived deterministically from records**, never guessed and never narrated by a
model: the phase is the earliest unmet gate, the attention list is ordered by severity and carries the
exact command that clears it, and the research map only contains edges that exist in the stored links.
That is what makes the cockpit explainable — and testable, which is why it lives in Python rather than
in the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .claims.language import calibrate, permitted_verb
from .kernel.kernel import ResearchKernel
from .models.common import ClaimStatus, EvidenceType, Severity, VerificationStatus, StrEnum

#: Evidence maturity order, used to render the evidence funnel.
EVIDENCE_FUNNEL: tuple[EvidenceType, ...] = (
    EvidenceType.RAW,
    EvidenceType.ANALYZED,
    EvidenceType.VERIFIED,
    EvidenceType.INTERPRETED,
    EvidenceType.CLAIMED,
)

#: A review queue longer than this makes "decide the queue" the project's current work.
REVIEW_BACKLOG_THRESHOLD = 3

#: Literature coverage below this is treated as "no real search has happened yet".
LITERATURE_COVERAGE_FLOOR = 0.4


class ResearchPhase(StrEnum):
    """Where the project stands, as the earliest unmet gate rather than as a mood."""

    QUESTION_UNCONFIRMED = "QUESTION_UNCONFIRMED"
    IMPORT_REVIEW = "IMPORT_REVIEW"
    CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"
    EVIDENCE_VERIFICATION = "EVIDENCE_VERIFICATION"
    LITERATURE_AUDIT = "LITERATURE_AUDIT"
    EXPERIMENT_DESIGN = "EXPERIMENT_DESIGN"
    ANALYSIS_PENDING = "ANALYSIS_PENDING"
    CLAIM_VALIDATION = "CLAIM_VALIDATION"
    PAPER_COMPILATION = "PAPER_COMPILATION"
    PAPER_READY = "PAPER_READY"


#: One-line description of each phase, in researcher language (shown on the first screen).
PHASE_LABELS: dict[ResearchPhase, str] = {
    ResearchPhase.QUESTION_UNCONFIRMED: "Core question not confirmed",
    ResearchPhase.IMPORT_REVIEW: "Reviewing what was reconstructed",
    ResearchPhase.CONFLICT_RESOLUTION: "Resolving contradictions in the material",
    ResearchPhase.EVIDENCE_VERIFICATION: "Verifying evidence",
    ResearchPhase.LITERATURE_AUDIT: "Establishing prior art",
    ResearchPhase.EXPERIMENT_DESIGN: "Designing the next experiment",
    ResearchPhase.ANALYSIS_PENDING: "Turning raw runs into analysis",
    ResearchPhase.CLAIM_VALIDATION: "Deciding which claims hold",
    ResearchPhase.PAPER_COMPILATION: "Compiling the paper",
    ResearchPhase.PAPER_READY: "Paper is compiled and grounded",
}

#: The gates a project passes through, in order. Rendered as a checklist so the state machine is visible.
GATE_ORDER: tuple[str, ...] = (
    "reconstruct",
    "review",
    "resolve_conflicts",
    "verify_evidence",
    "literature",
    "analyse",
    "promote_claims",
    "compile_paper",
)

GATE_LABELS: dict[str, str] = {
    "reconstruct": "Research state reconstructed",
    "review": "Human review of imported material",
    "resolve_conflicts": "Contradictions resolved",
    "verify_evidence": "Evidence verified against artifacts",
    "literature": "Prior art established",
    "analyse": "Runs turned into analysis artifacts",
    "promote_claims": "Claims promoted on evidence",
    "compile_paper": "Paper compiled with grounding",
}


@dataclass
class NextAction:
    """One thing the researcher should do, with the reason and the exact command."""

    kind: str
    title: str
    why: str
    count: int
    severity: Severity
    command: str
    blocks: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    #: True when the dashboard can perform it (the human-in-the-loop actions).
    ui_action: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "title": self.title,
            "why": self.why,
            "count": self.count,
            "severity": self.severity.value,
            "command": self.command,
            "blocks": list(self.blocks),
            "refs": list(self.refs),
            "ui_action": self.ui_action,
        }


@dataclass
class MapNode:
    node_id: str
    kind: str
    label: str
    status: str = ""
    detail: str = ""
    depth: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "depth": self.depth,
        }


@dataclass
class MapEdge:
    source: str
    target: str
    relation: str

    def as_dict(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target, "relation": self.relation}


@dataclass
class GateStatus:
    gate: str
    label: str
    status: str  # done | current | pending
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"gate": self.gate, "label": self.label, "status": self.status, "detail": self.detail}


@dataclass
class CockpitView:
    phase: ResearchPhase
    phase_label: str
    phase_reason: str
    core_question: str | None
    core_question_confirmed: bool
    current_finding: str
    current_finding_kind: str
    knowledge: dict[str, object]
    gates: list[GateStatus]
    attention: list[NextAction]
    not_ready: list[str]
    readiness_digest: dict[str, object]
    research_map: dict[str, object]
    counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        """Serialise **only** research-facing fields.

        Deliberately absent: revision, event head, hashes, integrity counters. Those belong to the audit
        console; mixing them into the first screen is what makes a research state unreadable.
        """
        return {
            "phase": self.phase.value,
            "phase_label": self.phase_label,
            "phase_reason": self.phase_reason,
            "core_question": self.core_question,
            "core_question_confirmed": self.core_question_confirmed,
            "current_finding": self.current_finding,
            "current_finding_kind": self.current_finding_kind,
            "knowledge": self.knowledge,
            "gates": [gate.as_dict() for gate in self.gates],
            "attention": [action.as_dict() for action in self.attention],
            "not_ready": list(self.not_ready),
            "readiness_digest": self.readiness_digest,
            "research_map": self.research_map,
            "counts": dict(self.counts),
        }


class Cockpit:
    """Builds the cockpit view for one project."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ================================================================== build

    def build(self) -> CockpitView:
        state = self.kernel.research_state()
        question = state.core_question
        confirmed = bool(question and not question.is_placeholder)

        claims = self.kernel.claims.all()
        live = [c for c in claims if c.status not in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED)]
        supported = [c for c in claims if c.status in (ClaimStatus.SUPPORTED, ClaimStatus.ROBUST)]
        tested = [c for c in claims if c.status is ClaimStatus.TESTED]
        hypotheses = [c for c in claims if c.status is ClaimStatus.HYPOTHESIS]
        rejected = [c for c in claims if c.status in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED)]

        evidence = self.kernel.evidence.all()
        verified = [e for e in evidence if e.verification_status is VerificationStatus.VERIFIED]
        unverified = [e for e in evidence if e.verification_status is not VerificationStatus.VERIFIED]
        analyses = self.kernel.analyses.all()
        experiments = self.kernel.experiments.all()
        reviews = self.kernel.pending_reviews()
        blocking_conflicts = self.kernel.ledger.blocking()
        literature = state.literature_state

        knowledge = self._knowledge(evidence, analyses, experiments, claims, verified, unverified)
        phase, reason = self._phase(
            confirmed=confirmed,
            reviews=reviews,
            blocking_conflicts=blocking_conflicts,
            evidence=evidence,
            verified=verified,
            experiments=experiments,
            analyses=analyses,
            literature=literature,
            supported=supported,
            tested=tested,
            hypotheses=hypotheses,
        )
        attention = self._attention(
            reviews=reviews,
            blocking_conflicts=blocking_conflicts,
            unverified=unverified,
            experiments=experiments,
            analyses=analyses,
            literature=literature,
            claims=claims,
            rejected=rejected,
            supported=supported,
            confirmed=confirmed,
        )
        not_ready, digest = self._not_ready(
            supported=supported,
            verified=verified,
            evidence=evidence,
            literature=literature,
            blocking_conflicts=blocking_conflicts,
            confirmed=confirmed,
        )

        return CockpitView(
            phase=phase,
            phase_label=PHASE_LABELS[phase],
            phase_reason=reason,
            core_question=question.statement if question else None,
            core_question_confirmed=confirmed,
            current_finding=self._current_finding(supported, tested, hypotheses, verified, unverified),
            current_finding_kind=(
                "SUPPORTED" if supported else "TESTED" if tested else "HYPOTHESIS" if hypotheses else "NONE"
            ),
            knowledge=knowledge,
            gates=self._gates(
                phase=phase,
                evidence=evidence,
                verified=verified,
                experiments=experiments,
                analyses=analyses,
                reviews=reviews,
                blocking_conflicts=blocking_conflicts,
                literature=literature,
                supported=supported,
                confirmed=confirmed,
            ),
            attention=attention,
            not_ready=not_ready,
            readiness_digest=digest,
            research_map=self._research_map(question, claims, experiments, live),
            counts={
                "evidence": len(evidence),
                "evidence_verified": len(verified),
                "claims": len(claims),
                "claims_supported": len(supported),
                "experiments": len(experiments),
                "analyses": len(analyses),
                "open_questions": len([q for q in self.kernel.store("open_question").all() if q.resolved_at is None]),
                "reviews_pending": len(reviews),
                "conflicts_blocking": len(blocking_conflicts),
                "papers": len(self.kernel.paper_artifacts.all()),
            },
        )

    # ================================================================== phase

    def _phase(
        self,
        *,
        confirmed: bool,
        reviews: Sequence[object],
        blocking_conflicts: Sequence[object],
        evidence: Sequence[object],
        verified: Sequence[object],
        experiments: Sequence[object],
        analyses: Sequence[object],
        literature: object,
        supported: Sequence[object],
        tested: Sequence[object],
        hypotheses: Sequence[object],
    ) -> tuple[ResearchPhase, str]:
        """The earliest unmet gate. The reason is a sentence a researcher can act on."""
        if not confirmed:
            return (
                ResearchPhase.QUESTION_UNCONFIRMED,
                "this project has no confirmed core question, so nothing can be prioritised against it",
            )
        if len(reviews) > REVIEW_BACKLOG_THRESHOLD:
            return (
                ResearchPhase.IMPORT_REVIEW,
                f"{len(reviews)} reconstructed item(s) are waiting for a human decision; until they are "
                "decided they are candidates, not research facts",
            )
        if blocking_conflicts:
            return (
                ResearchPhase.CONFLICT_RESOLUTION,
                f"{len(blocking_conflicts)} contradiction(s) in the material are unresolved and block "
                "claim promotion",
            )
        if evidence and not verified:
            return (
                ResearchPhase.EVIDENCE_VERIFICATION,
                f"none of the {len(evidence)} evidence record(s) has been verified against its artifacts, "
                "so no claim can be promoted",
            )
        if not literature.queries_executed or literature.coverage_score < LITERATURE_COVERAGE_FLOOR:
            return (
                ResearchPhase.LITERATURE_AUDIT,
                "no literature search has been recorded, so novelty cannot be assessed and Related Work "
                "cannot be written",
            )
        if not experiments:
            return (
                ResearchPhase.EXPERIMENT_DESIGN,
                f"{len(hypotheses)} hypothes(es) have no experiment behind them yet",
            )
        if not analyses:
            return (
                ResearchPhase.ANALYSIS_PENDING,
                f"{len(experiments)} experiment(s) exist but no analysis artifact has been computed, so "
                "there are no numbers a claim could rest on",
            )
        if not supported:
            return (
                ResearchPhase.CLAIM_VALIDATION,
                f"{len(tested) + len(hypotheses)} claim(s) are not yet SUPPORTED: the evidence they need "
                "must be assessed and each promotion approved",
            )
        if not self.kernel.paper_artifacts.all():
            return (
                ResearchPhase.PAPER_COMPILATION,
                f"{len(supported)} claim(s) are supported and no paper artifact has been compiled yet",
            )
        compiled = self.kernel.paper_artifacts.all()[-1]
        if compiled.grounding_report and not compiled.grounding_report.passed:
            return (
                ResearchPhase.PAPER_COMPILATION,
                "the compiled paper still has blocking grounding violations",
            )
        return (ResearchPhase.PAPER_READY, "claims are supported and the compiled paper passes grounding")

    # ================================================================== knowledge

    def _knowledge(
        self,
        evidence: Sequence[object],
        analyses: Sequence[object],
        experiments: Sequence[object],
        claims: Sequence[object],
        verified: Sequence[object],
        unverified: Sequence[object],
    ) -> dict[str, object]:
        """The evidence → claim chain as a funnel, so "why can't I promote?" is visible in one look."""
        by_maturity: dict[str, int] = {maturity.value: 0 for maturity in EVIDENCE_FUNNEL}
        for item in evidence:
            by_maturity[item.evidence_type.value] = by_maturity.get(item.evidence_type.value, 0) + 1
        by_status: dict[str, int] = {status.value: 0 for status in ClaimStatus}
        for claim in claims:
            by_status[claim.status.value] = by_status.get(claim.status.value, 0) + 1
        # The funnel must reflect what is *usable*, not just what is stored: unverified evidence cannot
        # carry a claim, so it is reported separately rather than counted as verified.
        return {
            "evidence_funnel": [
                {"stage": maturity.value, "count": by_maturity.get(maturity.value, 0)}
                for maturity in EVIDENCE_FUNNEL
            ],
            "evidence_verified": len(verified),
            "evidence_unverified": len(unverified),
            "analyses": len(analyses),
            "experiments": len(experiments),
            "claims_by_status": by_status,
            "promotion_chain": [
                {
                    "step": "evidence recorded",
                    "count": len(evidence),
                    "blocked": False,
                    "note": "raw material and derived observations",
                },
                {
                    "step": "evidence verified",
                    "count": len(verified),
                    "blocked": bool(evidence) and not verified,
                    "note": (
                        "verified by re-hashing the artifacts behind each record"
                        if verified
                        else "nothing is verified yet, so no claim can be promoted"
                    ),
                },
                {
                    "step": "analysis artifacts",
                    "count": len(analyses),
                    "blocked": not analyses,
                    "note": "numbers a claim may cite come from here",
                },
                {
                    "step": "claims SUPPORTED",
                    "count": by_status.get(ClaimStatus.SUPPORTED.value, 0)
                    + by_status.get(ClaimStatus.ROBUST.value, 0),
                    "blocked": not (
                        by_status.get(ClaimStatus.SUPPORTED.value, 0)
                        + by_status.get(ClaimStatus.ROBUST.value, 0)
                    ),
                    "note": "only SUPPORTED or ROBUST claims may carry paper language",
                },
            ],
        }

    # ================================================================== finding

    def _current_finding(
        self,
        supported: Sequence[object],
        tested: Sequence[object],
        hypotheses: Sequence[object],
        verified: Sequence[object],
        unverified: Sequence[object],
    ) -> str:
        """The strongest honest statement the project can currently make."""
        if supported:
            claim = supported[0]
            level = claim.evidence_level
            return (
                f"{calibrate(claim.statement, level).rstrip('.')} "
                f"(status {claim.status.value}, evidence {level.value}, permitted wording: "
                f"'{permitted_verb(level)}')"
            )
        if tested:
            return (
                f"{len(tested)} claim(s) have been tested but none is supported yet; "
                "the evidence has been assessed, the promotion has not"
            )
        if hypotheses:
            claim = hypotheses[0]
            caveat = (
                f" — and the {len(unverified)} evidence record(s) behind it are not verified"
                if unverified and not verified
                else ""
            )
            return f"Working hypothesis (unproven): {claim.statement.rstrip('.')}{caveat}"
        return "No claim has been formulated from this material yet"

    # ================================================================== gates

    def _gates(
        self,
        *,
        phase: ResearchPhase,
        evidence: Sequence[object],
        verified: Sequence[object],
        experiments: Sequence[object],
        analyses: Sequence[object],
        reviews: Sequence[object],
        blocking_conflicts: Sequence[object],
        literature: object,
        supported: Sequence[object],
        confirmed: bool,
    ) -> list[GateStatus]:
        """The state machine as a checklist: done / current / pending, with the reason for each."""
        done = {
            # Reconstruction is done when the material produced *anything* — including review-queue
            # candidates, which are a reconstruction output even before a human confirms them.
            "reconstruct": bool(
                evidence or experiments or analyses or reviews or self.kernel.claims.all()
            ),
            "review": not reviews,
            "resolve_conflicts": not blocking_conflicts,
            "verify_evidence": bool(verified),
            "literature": bool(literature.queries_executed)
            and literature.coverage_score >= LITERATURE_COVERAGE_FLOOR,
            "analyse": bool(analyses),
            "promote_claims": bool(supported),
            "compile_paper": bool(self.kernel.paper_artifacts.all())
            and bool(
                (self.kernel.paper_artifacts.all()[-1].grounding_report or None)
                and self.kernel.paper_artifacts.all()[-1].grounding_report.passed
            ),
        }
        details = {
            "reconstruct": f"{len(evidence)} evidence, {len(experiments)} experiment(s) recovered"
            if (evidence or experiments)
            else "no material reconstructed yet",
            "review": f"{len(reviews)} decision(s) pending" if reviews else "queue empty",
            "resolve_conflicts": f"{len(blocking_conflicts)} blocking contradiction(s)"
            if blocking_conflicts
            else "no blocking contradictions",
            "verify_evidence": f"{len(verified)} of {len(evidence)} verified",
            "literature": f"coverage {literature.coverage_score:.2f} — {literature.coverage_basis or 'no search'}",
            "analyse": f"{len(analyses)} analysis artifact(s)",
            "promote_claims": f"{len(supported)} supported claim(s)",
            "compile_paper": (
                f"{len(self.kernel.paper_artifacts.all())} compiled paper(s)"
                + (
                    " · latest passes grounding"
                    if self.kernel.paper_artifacts.all()
                    and (self.kernel.paper_artifacts.all()[-1].grounding_report or None)
                    and self.kernel.paper_artifacts.all()[-1].grounding_report.passed
                    else " · latest has blocking violations"
                    if self.kernel.paper_artifacts.all()
                    else ""
                )
            ),
        }
        phase_gate = {
            ResearchPhase.QUESTION_UNCONFIRMED: "reconstruct",
            ResearchPhase.IMPORT_REVIEW: "review",
            ResearchPhase.CONFLICT_RESOLUTION: "resolve_conflicts",
            ResearchPhase.EVIDENCE_VERIFICATION: "verify_evidence",
            ResearchPhase.LITERATURE_AUDIT: "literature",
            ResearchPhase.EXPERIMENT_DESIGN: "analyse",
            ResearchPhase.ANALYSIS_PENDING: "analyse",
            ResearchPhase.CLAIM_VALIDATION: "promote_claims",
            ResearchPhase.PAPER_COMPILATION: "compile_paper",
            ResearchPhase.PAPER_READY: "compile_paper",
        }.get(phase, "reconstruct")
        _ = phase_gate  # gating is derived from `done`, see below

        # The first gate that is not done is the one the project is on; everything after it is pending.
        # (Rendering it this way means a checklist can never show a "current" gate above an unfinished
        # one, which is what made the earlier version confusing.)
        current_marked = False
        gates: list[GateStatus] = []
        for gate in GATE_ORDER:
            if done.get(gate):
                status = "done"
            elif not current_marked:
                status = "current"
                current_marked = True
            else:
                status = "pending"
            gates.append(GateStatus(gate=gate, label=GATE_LABELS[gate], status=status, detail=details.get(gate, "")))
        if not confirmed:
            gates.insert(
                0,
                GateStatus(
                    gate="core_question",
                    label="Core question confirmed",
                    status="current",
                    detail="the recovered question is still a placeholder",
                ),
            )
            for gate in gates[1:]:
                if gate.status == "current":
                    gate.status = "pending"
        return gates

    # ================================================================== attention

    def _attention(
        self,
        *,
        reviews: Sequence[object],
        blocking_conflicts: Sequence[object],
        unverified: Sequence[object],
        experiments: Sequence[object],
        analyses: Sequence[object],
        literature: object,
        claims: Sequence[object],
        rejected: Sequence[object],
        supported: Sequence[object],
        confirmed: bool,
    ) -> list[NextAction]:
        """The ordered answer to "what do I do next?" — severity first, then count."""
        from .claims import ClaimRegistry

        actions: list[NextAction] = []

        if not confirmed:
            actions.append(
                NextAction(
                    kind="CONFIRM_QUESTION",
                    title="Confirm the core research question",
                    why="guarded research state only changes through an approved transition, so the "
                    "recovered question is still a placeholder",
                    count=1,
                    severity=Severity.BLOCKER,
                    command="researchos task transition list && researchos task transition approve --str <id>",
                    blocks=["every claim's scope", "paper framing"],
                    ui_action="transition.approve",
                )
            )
        if blocking_conflicts:
            subjects = sorted({conflict.subject for conflict in blocking_conflicts})[:3]
            actions.append(
                NextAction(
                    kind="RESOLVE_CONFLICT",
                    title=f"Resolve {len(blocking_conflicts)} contradiction(s) in the material",
                    why="unresolved conflicts block claim promotion; both sources are kept, so this is a "
                    "decision rather than a deletion",
                    count=len(blocking_conflicts),
                    severity=Severity.BLOCKER,
                    command="researchos review list  # conflict items carry the two sources",
                    blocks=["claim promotion", "Results section"],
                    refs=[conflict.conflict_id for conflict in blocking_conflicts],
                    ui_action="review.decide",
                )
            )
            if subjects:
                actions[-1].why += " — " + "; ".join(subject[:70] for subject in subjects)
        if reviews:
            by_kind: dict[str, int] = {}
            for item in reviews:
                by_kind[item.kind.value] = by_kind.get(item.kind.value, 0) + 1
            breakdown = ", ".join(f"{count} {kind.lower().replace('_', ' ')}" for kind, count in sorted(by_kind.items()))
            actions.append(
                NextAction(
                    kind="REVIEW_QUEUE",
                    title=f"Decide {len(reviews)} reconstructed item(s)",
                    why=f"nothing imported is a research fact until a human decides it ({breakdown})",
                    count=len(reviews),
                    severity=Severity.HIGH,
                    command="researchos review list",
                    blocks=["claims becoming usable", "experiment confirmation"],
                    refs=[item.review_item_id for item in reviews[:30]],
                    ui_action="review.decide",
                )
            )
        if unverified and not supported:
            actions.append(
                NextAction(
                    kind="VERIFY_EVIDENCE",
                    title=f"Verify {len(unverified)} evidence record(s) by re-hashing their artifacts",
                    why="verification is what turns a recorded observation into something a claim may rest on",
                    count=len(unverified),
                    severity=Severity.HIGH,
                    command="researchos evidence list && researchos evidence verify <evidence_id>",
                    blocks=["claim promotion", "Results section"],
                    refs=[item.evidence_id for item in unverified[:30]],
                    ui_action="evidence.verify",
                )
            )
        if not literature.queries_executed:
            actions.append(
                NextAction(
                    kind="LITERATURE",
                    title="Run a literature search for the core question",
                    why="no search has been recorded, so prior work is unknown and novelty cannot be assessed",
                    count=0,
                    severity=Severity.HIGH,
                    command='researchos literature search "<core question terms>"',
                    blocks=["Related Work section", "any novelty statement"],
                )
            )
        for experiment in experiments:
            gaps: list[str] = []
            if experiment.control is None:
                gaps.append("no control arm")
            elif not experiment.matched_conditions:
                gaps.append("conditions not shown matched")
            if not experiment.seeds:
                gaps.append("no seeds recorded")
            if gaps:
                actions.append(
                    NextAction(
                        kind="EXPERIMENT_DESIGN",
                        title=f"Confirm the design of {experiment.title[:60]}",
                        why="a design was inferred from the material; " + ", ".join(gaps),
                        count=1,
                        severity=Severity.MEDIUM,
                        command="researchos experiment list && researchos review list",
                        blocks=["the evidence level this experiment can support"],
                        refs=[experiment.experiment_id],
                    )
                )
        if experiments and not analyses:
            actions.append(
                NextAction(
                    kind="ANALYSIS",
                    title="Compute analysis artifacts for the registered runs",
                    why="numbers in the paper may only come from analysis artifacts, never from prose",
                    count=0,
                    severity=Severity.MEDIUM,
                    command="researchos audit stats",
                    blocks=["Results section", "every number in the paper"],
                )
            )
        unlinked_claims = [
            claim
            for claim in claims
            if claim.status not in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED)
            and not claim.supporting_experiments
            and not claim.contradicting_experiments
        ]
        if unlinked_claims and experiments:
            actions.append(
                NextAction(
                    kind="LINK_CLAIMS",
                    title=f"Link {len(unlinked_claims)} claim(s) to the experiment(s) that test them",
                    why=(
                        "import recovered claims from prose and experiments from run artifacts, and it "
                        "refuses to guess which experiment tests which claim — but a claim with no "
                        "linked experiment can never leave HYPOTHESIS"
                    ),
                    count=len(unlinked_claims),
                    severity=Severity.HIGH,
                    command="researchos claim transition <claim_id> TESTED --reason '…'  # after linking",
                    blocks=["claim promotion (TESTED requires a supporting experiment)"],
                    refs=[claim.claim_id for claim in unlinked_claims],
                )
            )
        if unclaimed_experiments := [
            experiment
            for experiment in experiments
            if not experiment.claim_ids
            and not any(experiment.experiment_id in claim.supporting_experiments for claim in claims)
        ]:
            if not unlinked_claims:
                actions.append(
                    NextAction(
                        kind="LINK_EXPERIMENTS",
                        title=f"Record what {len(unclaimed_experiments)} experiment(s) were for",
                        why="an experiment that no claim points at is not evidence for anything yet",
                        count=len(unclaimed_experiments),
                        severity=Severity.MEDIUM,
                        command="researchos experiment list",
                        blocks=["the paper's method and results narrative"],
                        refs=[experiment.experiment_id for experiment in unclaimed_experiments],
                    )
                )
        overclaiming = ClaimRegistry(self.kernel).overclaiming()
        if overclaiming:
            actions.append(
                NextAction(
                    kind="CALIBRATE_LANGUAGE",
                    title=f"Calibrate {len(overclaiming)} claim statement(s)",
                    why="the wording demands more evidence than exists; the dashboard shows the permitted "
                    "rewrite for each",
                    count=len(overclaiming),
                    severity=Severity.MEDIUM,
                    command="researchos claim audit",
                    blocks=["accurate reporting of what was shown"],
                    refs=[claim.claim_id for claim, _safe in overclaiming],
                )
            )
        rejected_without_record = [
            claim for claim in rejected if not claim.rejection_reason or not claim.rejection_basis
        ]
        if rejected_without_record:
            actions.append(
                NextAction(
                    kind="RECORD_REJECTION",
                    title=f"Record why {len(rejected_without_record)} claim(s) were rejected",
                    why="a rejected claim is retained as history; without a stated basis it cannot be explained later",
                    count=len(rejected_without_record),
                    severity=Severity.LOW,
                    command="researchos timeline ask --question 'why was the original claim cancelled?'",
                    blocks=["the paper's honest account of the method"],
                    refs=[claim.claim_id for claim in rejected_without_record],
                )
            )
        for artifact in self.kernel.paper_artifacts.all():
            if artifact.grounding_report and not artifact.grounding_report.passed:
                actions.append(
                    NextAction(
                        kind="GROUNDING",
                        title=f"Fix {len(artifact.grounding_report.blockers())} grounding violation(s)",
                        why="the compiler refused to emit text that is not backed by an artifact",
                        count=len(artifact.grounding_report.blockers()),
                        severity=Severity.HIGH,
                        command="researchos paper compile",
                        blocks=["paper release"],
                        refs=[artifact.paper_id],
                    )
                )

        order = {Severity.BLOCKER: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}
        actions.sort(key=lambda action: (order.get(action.severity, 5), -action.count, action.kind))
        return self._merge_similar(actions)

    @staticmethod
    def _merge_similar(actions: Sequence[NextAction]) -> list[NextAction]:
        """One action per *kind*.

        Compiling five drafts produces five identical "fix the grounding" entries, and a list that
        repeats itself is a list nobody reads. The refs are unioned so nothing is lost.
        """
        merged: dict[str, NextAction] = {}
        for action in actions:
            existing = merged.get(action.kind)
            if existing is None:
                merged[action.kind] = action
                continue
            existing.refs = list(dict.fromkeys([*existing.refs, *action.refs]))
            if action.count > existing.count:
                existing.count = action.count
                existing.title = action.title
                existing.why = action.why
        return sorted(
            merged.values(),
            key=lambda action: (order_key(action.severity), -action.count, action.kind),
        )

    # ================================================================== not ready

    def _not_ready(
        self,
        *,
        supported: Sequence[object],
        verified: Sequence[object],
        evidence: Sequence[object],
        literature: object,
        blocking_conflicts: Sequence[object],
        confirmed: bool,
    ) -> tuple[list[str], dict[str, object]]:
        """Why a paper cannot be written yet, plus the seven dimensions behind a disclosure."""
        reasons: list[str] = []
        if not confirmed:
            reasons.append("the core question is still a placeholder")
        if blocking_conflicts:
            reasons.append(f"{len(blocking_conflicts)} unresolved contradiction(s) block claim promotion")
        if evidence and not verified:
            reasons.append(f"none of the {len(evidence)} evidence record(s) is verified")
        if not literature.queries_executed:
            reasons.append("no literature search has been executed, so Related Work has no sources")
        if not supported:
            reasons.append("no claim has reached SUPPORTED, so Results would have nothing to report")

        try:
            from .paper import ReadinessAssessor

            readiness = ReadinessAssessor(self.kernel).assess(self.kernel.human())
            dimensions = [
                {
                    "dimension": dimension.dimension.value,
                    "label": dimension.label(),
                    "score": dimension.score,
                    "basis": dimension.basis,
                    "blockers": dimension.blockers[:4],
                    "unknowns": dimension.unknowns[:3],
                }
                for dimension in readiness.dimensions
            ]
            digest = {
                "dimensions": dimensions,
                "blockers": readiness.blockers()[:10],
                "weakest": [dimension.label() for dimension in readiness.weakest(3)],
                "note": "seven independent dimensions; no total score exists by design",
            }
        except Exception as exc:  # noqa: BLE001 - readiness is a view; never break the cockpit
            digest = {"dimensions": [], "error": f"{type(exc).__name__}: {exc}"}
        return reasons, digest

    # ================================================================== research map

    def _research_map(
        self,
        question: object,
        claims: Sequence[object],
        experiments: Sequence[object],
        live: Sequence[object],
    ) -> dict[str, object]:
        """The reasoning structure, built only from links that exist in the records.

        Two relations are structural rather than scientific and are labelled as such: claims belong to
        the core question (they are this project's claims), and unchanged links are simply absent rather
        than invented.
        """
        nodes: list[MapNode] = []
        edges: list[MapEdge] = []
        experiment_by_id = {experiment.experiment_id: experiment for experiment in experiments}
        claim_by_id = {claim.claim_id: claim for claim in claims}

        root_id = "core_question"
        if question is not None:
            nodes.append(
                MapNode(
                    node_id=root_id,
                    kind="QUESTION",
                    label=question.statement,
                    status="CONFIRMED" if not question.is_placeholder else "PLACEHOLDER",
                    detail=question.scope or "",
                    depth=0,
                )
            )

        for claim in live:
            nodes.append(
                MapNode(
                    node_id=claim.claim_id,
                    kind="CLAIM",
                    label=claim.statement,
                    status=claim.status.value,
                    detail=f"evidence level {claim.evidence_level.value}"
                    + (f" · scope {claim.scope}" if claim.scope else ""),
                    depth=1,
                )
            )
            if question is not None:
                edges.append(MapEdge(source=root_id, target=claim.claim_id, relation="BELONGS_TO"))
            for experiment_id in claim.supporting_experiments:
                experiment = experiment_by_id.get(experiment_id)
                if experiment is None:
                    continue
                if not any(node.node_id == experiment_id for node in nodes):
                    nodes.append(self._experiment_node(experiment, depth=2))
                edges.append(MapEdge(source=experiment_id, target=claim.claim_id, relation="TESTS"))
            for experiment_id in claim.contradicting_experiments:
                if experiment_by_id.get(experiment_id) is None:
                    continue
                if not any(node.node_id == experiment_id for node in nodes):
                    nodes.append(self._experiment_node(experiment_by_id[experiment_id], depth=1))
                    if question is not None:
                        edges.append(
                            MapEdge(source=root_id, target=experiment_id, relation="BELONGS_TO")
                        )
                edges.append(MapEdge(source=experiment_id, target=claim.claim_id, relation="CONTRADICTS"))

        # experiments that declare a claim themselves (registered live, not imported)
        for experiment in experiments:
            if not any(node.node_id == experiment.experiment_id for node in nodes):
                nodes.append(self._experiment_node(experiment, depth=1))
            if question is not None and not any(
                edge.source == root_id and edge.target == experiment.experiment_id for edge in edges
            ):
                # structural, not scientific: this experiment is part of this project
                edges.append(MapEdge(source=root_id, target=experiment.experiment_id, relation="BELONGS_TO"))
            for claim_id in experiment.claim_ids:
                claim = claim_by_id.get(claim_id)
                if claim is None or claim.status in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED):
                    continue
                if not any(
                    edge.source == experiment.experiment_id and edge.target == claim_id for edge in edges
                ):
                    edges.append(MapEdge(source=experiment.experiment_id, target=claim_id, relation="TESTS"))
            if experiment.parent_experiment_id:
                edges.append(
                    MapEdge(
                        source=experiment.parent_experiment_id,
                        target=experiment.experiment_id,
                        relation="FOLLOWS_UP",
                    )
                )

        for claim in claims:
            if claim.status not in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED):
                continue
            nodes.append(
                MapNode(
                    node_id=claim.claim_id,
                    kind="REJECTED",
                    label=claim.statement,
                    status=claim.status.value,
                    detail=claim.rejection_reason or "no basis recorded",
                    depth=1,
                )
            )
            if question is not None:
                edges.append(MapEdge(source=root_id, target=claim.claim_id, relation="BELONGS_TO"))

        for open_question in self.kernel.store("open_question").all():
            if open_question.resolved_at is not None:
                continue
            nodes.append(
                MapNode(
                    node_id=open_question.question_id,
                    kind="OPEN_QUESTION",
                    label=open_question.statement,
                    status=open_question.kind.value,
                    detail=open_question.blocker or "",
                    depth=3,
                )
            )
            for claim_id in open_question.related_claim_ids:
                if any(node.node_id == claim_id for node in nodes):
                    edges.append(
                        MapEdge(source=claim_id, target=open_question.question_id, relation="OPENS")
                    )
            if not open_question.related_claim_ids and question is not None:
                # unlinked questions hang off the core question rather than off an invented claim
                edges.append(MapEdge(source=root_id, target=open_question.question_id, relation="OPENS"))

        linked = {edge.source for edge in edges} | {edge.target for edge in edges}
        node_ids = {node.node_id for node in nodes}

        # Shared provenance: a claim candidate and an experiment extracted from the *same* artifact are
        # related, but the relation is "these came from the same source", not "this experiment supports
        # that claim". Import never asserts the stronger relation, and neither does this map.
        claim_nodes = [
            node for node in nodes if node.kind in {"CLAIM", "REJECTED"} and node.node_id in claim_by_id
        ]
        for claim_node in claim_nodes:
            claim_sources = self._claim_source_paths(claim_by_id[claim_node.node_id])
            for experiment in experiments:
                if not claim_sources or not experiment.source_refs:
                    continue
                if claim_sources & set(experiment.source_refs):
                    edges.append(
                        MapEdge(
                            source=experiment.experiment_id,
                            target=claim.claim_id,
                            relation="SAME_SOURCE",
                        )
                    )

        return {
            "nodes": [node.as_dict() for node in nodes],
            "edges": [edge.as_dict() for edge in edges if edge.source in node_ids and edge.target in node_ids],
            "unlinked": sorted(node_ids - linked),
            "note": (
                "only stored or derived-from-records links are drawn. A claim with no experiment and a "
                "question with no parent claim appear unlinked rather than attached to something "
                "plausible, and SAME_SOURCE means shared provenance, not support."
            ),
        }

    def _claim_source_paths(self, claim: object) -> set[str]:
        """The files a claim traces back to, via its evidence.

        A claim has no ``source_refs`` of its own: its provenance is the evidence it rests on. Reading it
        that way keeps the chain honest — claim → evidence → file — instead of duplicating a path onto
        the claim and letting the two drift apart.
        """
        paths: set[str] = set()
        for evidence_id in claim.evidence_ids:
            evidence = self.kernel.evidence.get(evidence_id)
            if evidence is None:
                continue
            if evidence.source_path:
                paths.add(evidence.source_path)
            for ref in evidence.source_refs:
                if ref.path:
                    paths.add(ref.path)
        return paths

    @staticmethod
    def _experiment_node(experiment: object, *, depth: int) -> MapNode:
        """Label an experiment by the *role its declared design plays*, not by a guess from its title."""
        design = experiment.design
        if design.has_necessity_design or design.has_sufficiency_design or design.has_rescue:
            role = "component-removal test"
        elif design.has_control and design.has_matched_conditions:
            role = "matched comparison"
        elif design.has_control:
            role = "comparison"
        else:
            role = "observation"
        return MapNode(
            node_id=experiment.experiment_id,
            kind="EXPERIMENT",
            label=experiment.title,
            status=experiment.status.value,
            detail=f"{role} · ceiling {experiment.evidence_level().value}"
            + (f" · seeds {len(experiment.seeds)}" if experiment.seeds else " · no seeds"),
            depth=depth,
        )


def order_key(severity: Severity) -> int:
    """Sort key for severities: BLOCKER first."""
    return {Severity.BLOCKER: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}.get(
        severity, 5
    )


def cockpit_view(kernel: ResearchKernel) -> dict[str, object]:
    """Convenience: the cockpit payload for one project."""
    return Cockpit(kernel).build().as_dict()
