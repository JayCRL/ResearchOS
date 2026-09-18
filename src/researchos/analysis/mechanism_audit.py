"""Mechanism audit: place evidence on the causal ladder, then catch the language climbing it.

**Why this module exists.** The most common failure of an AI-assisted paper is not a wrong number;
it is a sentence that claims more than the experiment could ever show — "X causes Y" from an
observational run, "this establishes the mechanism" from a comparison of two conditions, "the
effect generalises across models" from a single model family. ARCHITECTURE.md §4.11 requires an
auditor that (a) positions every piece of evidence on the ladder

    OBSERVATION → CORRELATION → CONTROLLED_COMPARISON → INTERVENTION → NECESSITY → SUFFICIENCY →
    RESCUE → CROSS_SETTING_REPLICATION

and (b) flags every claim whose *language* sits higher on that ladder than its experiments do.

Three commitments make this deterministic rather than rhetorical:

* **The evidence rung comes from the declared design, never from the prose.** :meth:`rung_for_experiment`
  reads ``ExperimentDesign`` flags through ``assess_experiment_level_from_design``; writing more
  confidently cannot move an experiment up a rung.
* **The language rung comes from a fixed pattern table.** :data:`MechanismAuditor.RUNG_BY_LANGUAGE`
  is public and data-only, so a reviewer can read exactly which phrasing counts as a causal claim
  and argue with the table instead of with a black box. No LLM judgement enters the comparison.
* **Alternative explanations are derived from missing fields.** :meth:`alternative_explanations`
  emits an explanation only when the experiment's own record shows the gap that would produce it
  (no matched conditions, no budget, no optimizer/schedule match, one seed, one model, an
  undefined evaluation split). It never pads a list to look thorough.

The auditor **reports and never repairs**: it holds ``Cap.AUDIT_CREATE`` and writes an immutable
:class:`Audit`. Downgrading the claim is the claim owner's job, and it must be done in the record,
not in the audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from ..kernel.permissions import Cap
from ..models.audit import Audit, AuditVerdict, Finding
from ..models.common import AuditKind, MechanismRung, Severity
from ..models.experiment import Experiment, assess_experiment_level_from_design
from ..models.common import EvidenceLevel

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from ..kernel.kernel import ResearchKernel
    from ..kernel.permissions import Principal

#: Every code this auditor may emit. Pinned by tests so a rename cannot pass unnoticed.
MECHANISM_FINDING_CODES: tuple[str, ...] = (
    "CAUSAL_LANGUAGE_WITHOUT_INTERVENTION",
    "MECHANISM_CLAIM_OVERREACH",
    "NECESSITY_CLAIM_WITHOUT_REMOVAL",
    "SUFFICIENCY_CLAIM_WITHOUT_FORCING",
    "NO_RESCUE_EXPERIMENT",
    "SINGLE_SETTING_GENERALIZATION",
    "UNCONTROLLED_CONFOUND",
    "ALTERNATIVE_EXPLANATION_UNELIMINATED",
    "SINGLE_SEED_DEPENDENCY",
    "SINGLE_MODEL_DEPENDENCY",
    "NO_MATCHED_CONTROL",
    "PLACEMENT_EFFECT_OVERCLAIM",
)

#: Language that asserts a *mechanism* rather than a relation. Kept out of
#: :data:`MechanismAuditor.RUNG_BY_LANGUAGE` on purpose: "establishes the mechanism" is not a causal
#: verb, and routing it through the causal code would report the wrong defect. It is checked
#: separately against the NECESSITY rung, because isolating a component is what a mechanism claim
#: needs.
_MECHANISM_ASSERTIONS: tuple[str, ...] = (
    "establishes the mechanism",
    "establish the mechanism",
    "established the mechanism",
    "demonstrates the mechanism",
    "demonstrate the mechanism",
    "demonstrated the mechanism",
    "reveals the mechanism",
    "reveal the mechanism",
    "identifies the mechanism",
    "identify the mechanism",
    "proves the mechanism",
    "prove the mechanism",
    "the mechanism behind",
    "the mechanism by which",
    "the mechanism through which",
    "we identify a mechanism",
    "we have identified the mechanism",
)

#: Words that mark a comparison as being about *where* something is written / placed.
_PLACEMENT_WORDS: tuple[str, ...] = (
    "placement",
    "position",
    "writeback",
    "write back",
    "write-back",
    "write_pos",
    "write pos",
    "location",
    "slot",
)

#: Words that name an evaluation window (used by the contamination explanation).
_SPLIT_WORDS: tuple[str, ...] = (
    "held-out",
    "heldout",
    "held out",
    "test",
    "valid",
    "val ",
    "dev",
    "eval",
    "unseen",
    "holdout",
    "split",
)

#: Which code names the defect of speaking at a rung the experiments do not reach.
_OVERREACH_CODE_BY_RUNG: Mapping[MechanismRung, str] = {
    MechanismRung.CORRELATION: "UNCONTROLLED_CONFOUND",
    MechanismRung.CONTROLLED_COMPARISON: "NO_MATCHED_CONTROL",
    MechanismRung.INTERVENTION: "CAUSAL_LANGUAGE_WITHOUT_INTERVENTION",
    MechanismRung.NECESSITY: "NECESSITY_CLAIM_WITHOUT_REMOVAL",
    MechanismRung.SUFFICIENCY: "SUFFICIENCY_CLAIM_WITHOUT_FORCING",
    MechanismRung.RESCUE: "NO_RESCUE_EXPERIMENT",
    MechanismRung.CROSS_SETTING_REPLICATION: "SINGLE_SETTING_GENERALIZATION",
}

#: Language strength required to assert a *mechanism* at all: you must have removed or forced the
#: component, i.e. reached the NECESSITY rung, before "the mechanism" is a licensed phrase.
_MECHANISM_RUNG = MechanismRung.NECESSITY

_SUGGESTION_BY_CODE: Mapping[str, str] = {
    "CAUSAL_LANGUAGE_WITHOUT_INTERVENTION": (
        "either run an intervention (manipulate the hypothesised cause and hold everything else "
        "fixed) or rewrite the sentence as an association ('is associated with'); causal verbs are "
        "permitted only from L4 evidence upward"
    ),
    "MECHANISM_CLAIM_OVERREACH": (
        "replace 'establishes the mechanism' with what the experiment actually shows, and add a "
        "necessity or sufficiency experiment (remove or force the component) before claiming a "
        "mechanism"
    ),
    "NECESSITY_CLAIM_WITHOUT_REMOVAL": (
        "add the removal/ablation condition and show the effect disappears, or say 'contributes to' "
        "instead of 'is necessary for'; necessity is licensed by an experiment that removes the "
        "component, not by an argument"
    ),
    "SUFFICIENCY_CLAIM_WITHOUT_FORCING": (
        "add the forcing condition (introduce the component where it would otherwise be absent) and "
        "show the effect appears, or downgrade the sentence"
    ),
    "NO_RESCUE_EXPERIMENT": (
        "add the rescue condition (restore the component and show the effect returns), or drop the "
        "rescue language"
    ),
    "SINGLE_SETTING_GENERALIZATION": (
        "either repeat the experiment in at least one genuinely different setting (another model "
        "family, scale or dataset) or scope the sentence to the single setting that was run"
    ),
    "UNCONTROLLED_CONFOUND": (
        "control for the named confound (or measure it and include it in the model); until then the "
        "association is not interpretable"
    ),
    "ALTERNATIVE_EXPLANATION_UNELIMINATED": (
        "record each alternative explanation in the experiment's limitations and either eliminate "
        "it with a control or state plainly that it remains open"
    ),
    "SINGLE_SEED_DEPENDENCY": (
        "run at least two more seeds and report the spread; a single-seed effect cannot be "
        "distinguished from seed noise"
    ),
    "SINGLE_MODEL_DEPENDENCY": (
        "repeat in a second model family (or scope the claim to this model family); one model "
        "cannot support a claim about models in general"
    ),
    "NO_MATCHED_CONTROL": (
        "add a matched control arm (same data, same budget, same optimizer and schedule, one factor "
        "changed) before using comparative language"
    ),
    "PLACEMENT_EFFECT_OVERCLAIM": (
        "Two write placements compared under a matched control establish an association (a "
        "controlled comparison) between the placements; they do not by themselves establish the "
        "mechanism that produces the effect. Write it as a controlled comparison between "
        "placements, or add a necessity/sufficiency experiment to license a mechanism claim."
    ),
}


def _finding(
    code: str,
    severity: Severity,
    message: str,
    *,
    target_ref: str | None = None,
    blocks: bool = False,
    quote: str | None = None,
    claim_ids: Sequence[str] = (),
    experiment_ids: Sequence[str] = (),
    suggestion: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> Finding:
    """Build one finding with the code's standard suggestion unless one is supplied."""
    if code not in MECHANISM_FINDING_CODES:
        raise ValueError(f"unknown finding code {code!r}; add it to MECHANISM_FINDING_CODES")
    if blocks and severity not in (Severity.HIGH, Severity.BLOCKER):
        raise ValueError(f"{code}: a blocking finding must be HIGH or BLOCKER, got {severity.value}")
    text = suggestion or _SUGGESTION_BY_CODE.get(code)
    if not text:
        raise ValueError(f"{code}: a finding must carry a suggestion")
    return Finding(
        code=code,
        severity=severity,
        message=message,
        target_ref=target_ref,
        claim_ids=list(claim_ids),
        experiment_ids=list(experiment_ids),
        quote=quote[:2000] if quote else None,
        suggestion=text,
        blocks_publication=blocks,
        details=dict(details or {}),
    )


class MechanismAuditor:
    """Positions evidence on the mechanism ladder and reports overreach."""

    TOOL = "researchos.analysis.mechanism_audit/0.1"

    #: Language pattern table, one entry per rung. Public and data-only so a reviewer can audit the
    #: auditor: adding a phrase is a visible change to what counts as an intervention claim.
    RUNG_BY_LANGUAGE: ClassVar[dict[MechanismRung, tuple[str, ...]]] = {
        MechanismRung.OBSERVATION: (
            "we observe",
            "we observed",
            "we notice",
            "we noticed",
            "we report that",
            "we find that",
            "we found that",
            "it appears that",
            "we see that",
        ),
        MechanismRung.CORRELATION: (
            "correlates",
            "correlated with",
            "correlation between",
            "associated with",
            "association between",
            "co-occurs",
            "co-occur",
            "coincides with",
            "tracks with",
            "is linked to",
            "predicts",
        ),
        MechanismRung.CONTROLLED_COMPARISON: (
            "compared to",
            "compared with",
            "relative to a control",
            "relative to the control",
            "better than",
            "worse than",
            "outperforms",
            "improves over",
            "than the baseline",
            "than the control",
            "matched control",
        ),
        MechanismRung.INTERVENTION: (
            "causes",
            "caused by",
            "leads to",
            "lead to",
            "drives",
            "driven by",
            "we intervene",
            "intervening on",
            "we manipulate",
            "manipulating",
            "we perturb",
            "perturbing",
        ),
        MechanismRung.NECESSITY: (
            "is necessary for",
            "are necessary for",
            "necessary for",
            "is required for",
            "are required for",
            "required for",
            "removing",
            "removal of",
            "ablation removes",
            "ablating",
            "knockout",
            "abolishing",
        ),
        MechanismRung.SUFFICIENCY: (
            "is sufficient for",
            "are sufficient for",
            "sufficient for",
            "is enough to",
            "are enough to",
            "we force",
            "forcing",
            "suffices to",
        ),
        MechanismRung.RESCUE: (
            "rescue",
            "rescues",
            "rescued",
            "recovers the effect",
            "recovered the effect",
            "restores the effect",
            "restored the effect",
        ),
        MechanismRung.CROSS_SETTING_REPLICATION: (
            "across models",
            "across scales",
            "across datasets",
            "across settings",
            "across architectures",
            "across tasks",
            "generalises",
            "generalizes",
            "replicates at other scales",
            "replicated across",
            "holds across",
        ),
    }

    def __init__(self, kernel: "ResearchKernel") -> None:
        self.kernel = kernel

    # ================================================================== ladder placement

    def rung_for_experiment(self, experiment: Experiment) -> MechanismRung:
        """The highest rung this experiment's *declared design* licenses.

        The design → rung map is driven by the same ``assess_experiment_level_from_design`` the rest
        of ResearchOS uses, so an experiment cannot be an L5 for the claim registry and an
        observation here. The design flags then select *which* rung inside L5 was reached, because
        "we removed it", "we forced it" and "we put it back" are three different experiments.
        """
        design = experiment.design
        level = assess_experiment_level_from_design(design)

        # Settings are checked before the single-setting flags: two settings plus any manipulation
        # or control is what makes an effect portable, which is exactly what L6 means.
        if level is EvidenceLevel.L6_CROSS_SETTING_REPLICATION:
            return MechanismRung.CROSS_SETTING_REPLICATION
        if level is EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY:
            if design.has_rescue:
                return MechanismRung.RESCUE
            if design.has_necessity_design:
                return MechanismRung.NECESSITY
            if design.has_sufficiency_design:
                return MechanismRung.SUFFICIENCY
            # assess_* only reaches L5 through one of the three flags above.
            return MechanismRung.NECESSITY
        if level is EvidenceLevel.L4_INTERVENTION:
            return MechanismRung.INTERVENTION
        if level is EvidenceLevel.L3_CONTROLLED:
            return MechanismRung.CONTROLLED_COMPARISON
        if level is EvidenceLevel.L2_REPRODUCED:
            # Reproduced across runs but with no matched control: that is an association.
            return MechanismRung.CORRELATION
        if len(set(design.settings)) >= 2 and design.is_observational:
            # Two settings do not rescue an observational design: nothing was manipulated.
            return MechanismRung.OBSERVATION
        return MechanismRung.OBSERVATION

    def rung_for_text(self, text: str) -> MechanismRung:
        """The rung implied by the claim's language.

        The highest matching rung wins: a sentence containing both "associated with" and "causes" is
        a causal sentence, and the strongest verb is what a reader will take away.
        """
        lowered = " ".join(text.lower().split())
        best = MechanismRung.OBSERVATION
        for rung in MechanismRung:
            patterns = self.RUNG_BY_LANGUAGE.get(rung, ())
            if any(pattern in lowered for pattern in patterns):
                if rung.rank > best.rank:
                    best = rung
        return best

    # ================================================================== alternatives

    def alternative_explanations(self, experiment: Experiment) -> list[str]:
        """Deterministic checklist of explanations the design has not eliminated.

        Each entry is triggered by a *missing or insufficient field* in the experiment's own record,
        which is what keeps the checklist honest: an experiment that matched its arms, recorded its
        budget, fixed its optimizer and schedule, replicated over seeds and models, and named its
        evaluation split produces an empty list.
        """
        design = experiment.design
        budget = experiment.training_budget
        explanations: list[str] = []

        matched_conditions = " ".join(experiment.matched_conditions).lower()
        arm_matched = bool(experiment.matched_conditions) or bool(
            experiment.control and experiment.control.matched_on
        )
        if not design.has_control:
            explanations.append(
                "there is no control arm, so the difference may reflect any unrelated change "
                "between the two conditions"
            )
        elif not arm_matched:
            explanations.append(
                "token-position frequency differs between arms: the arms were not matched on the "
                "non-experimental factors that decide which tokens are written where"
            )
        if budget.steps is None and budget.tokens is None and budget.epochs is None:
            explanations.append(
                "control arm saw fewer update steps: no training budget is recorded, so the arms "
                "cannot be shown to have received the same amount of training"
            )
        schedule_matched = any(
            word in matched_conditions
            for word in ("optimizer", "learning rate", "lr", "schedule", "warmup")
        )
        if experiment.optimizer is None or experiment.learning_rate is None or not schedule_matched:
            explanations.append(
                "the effect could be a learning-rate artefact (arms not matched on optimizer/schedule)"
            )
        if len(set(experiment.seeds)) < 2:
            explanations.append(
                "single seed: the effect may be seed-dependent rather than systematic"
            )
        if experiment.model is None:
            explanations.append(
                "the model family is unrecorded, so the effect may hold only for the run that "
                "happened to be used"
            )
        elif len({experiment.model}) == 1:
            explanations.append(
                f"single model family ({experiment.model}): the effect may not hold for other "
                "architectures"
            )
        contaminated = [m.name for m in experiment.metrics if m.contamination_risks]
        undefined_split = [
            m.name
            for m in experiment.metrics
            if m.is_defined
            and not any(word in m.definition.lower() for word in _SPLIT_WORDS)
        ]
        if contaminated:
            explanations.append(
                f"metric measured on a contaminated split: {', '.join(sorted(contaminated))} "
                "declares contamination risk(s)"
            )
        elif undefined_split:
            explanations.append(
                f"metric measured on an unspecified split: {', '.join(sorted(undefined_split))} "
                "does not name the evaluation window"
            )
        for confound in design.confounds_uncontrolled:
            explanations.append(f"declared uncontrolled confound: {confound}")
        return explanations

    def _is_placement_comparison(self, experiment: Experiment) -> bool:
        """True when the two arms differ in *where* something is written rather than in *what*.

        Detected from the recorded arm descriptions, matched conditions and config, never from the
        claim's prose — a claim cannot make its own experiment count as a placement study.
        """
        design = experiment.design
        if not (design.has_control and design.has_matched_conditions):
            return False
        haystack = " ".join(
            [
                experiment.treatment.name if experiment.treatment else "",
                experiment.treatment.description if experiment.treatment else "",
                experiment.control.name if experiment.control else "",
                experiment.control.description if experiment.control else "",
                " ".join(experiment.matched_conditions),
                " ".join(f"{k} {v}" for k, v in experiment.config.items()),
            ]
        ).lower()
        return any(word in haystack for word in _PLACEMENT_WORDS)

    # ================================================================== entry point

    def audit(
        self,
        principal: "Principal",
        *,
        claim_ids: Sequence[str] = (),
        experiment_ids: Sequence[str] = (),
        task_id: str | None = None,
        title: str | None = None,
    ) -> Audit:
        """Audit claims against the experiments that support them, and persist the audit.

        The central check is the rung comparison: when a claim's language sits higher on the ladder
        than the best rung its experiments reach, the matching overreach code is emitted, and it
        blocks publication once the gap is two rungs or more — a single rung of slippage is usually
        wording, two rungs is a different claim.
        """
        principal.require(Cap.AUDIT_CREATE, "audit.create")

        claims = self._resolve_claims(claim_ids)
        experiments = self._resolve_experiments(experiment_ids, claims)
        experiment_by_id = {e.experiment_id: e for e in experiments}
        subjects = sorted({e.experiment_id for e in experiments} | {c.claim_id for c in claims})

        findings: list[Finding] = []
        for claim in claims:
            findings.extend(self._check_claim(claim, experiment_by_id))
        findings.extend(self._check_dependencies(experiments, claims))
        findings.sort(key=lambda f: (f.code, f.target_ref or "", f.message))

        blocking = [f for f in findings if f.blocks_publication]
        if not subjects:
            verdict = AuditVerdict.INCONCLUSIVE
        elif blocking:
            verdict = AuditVerdict.FAIL
        elif findings:
            verdict = AuditVerdict.WARN
        else:
            verdict = AuditVerdict.PASS
        summary = (
            f"{len(findings)} finding(s) across {len(claims)} claim(s) and "
            f"{len(experiments)} experiment(s); {len(blocking)} blocking."
            if subjects
            else "nothing was in scope, so no mechanism claim could be checked"
        )
        audit = Audit(
            kind=AuditKind.MECHANISM,
            title=title or "Mechanism audit",
            subjects=subjects,
            findings=findings,
            summary=summary,
            verdict=verdict,
            tool=self.TOOL,
            created_by=principal.name,
            task_id=task_id,
        )
        return self.kernel.save_audit(principal, audit)

    # ================================================================== scope resolution

    def _resolve_claims(self, claim_ids: Sequence[str]) -> list[Any]:
        store = self.kernel.claims
        if claim_ids:
            found = [store.get(cid) for cid in dict.fromkeys(claim_ids)]
            return sorted((c for c in found if c is not None), key=lambda c: c.claim_id)
        return sorted(store.all(), key=lambda c: c.claim_id)

    def _resolve_experiments(
        self, experiment_ids: Sequence[str], claims: Sequence[Any]
    ) -> list[Experiment]:
        wanted: set[str] = set(experiment_ids)
        for claim in claims:
            wanted.update(claim.supporting_experiments)
            wanted.update(claim.contradicting_experiments)
        store = self.kernel.experiments
        if not wanted:
            return sorted(store.all(), key=lambda e: e.experiment_id)
        found = [store.get(eid) for eid in sorted(wanted)]
        return [experiment for experiment in found if experiment is not None]

    # ================================================================== claim checks

    def _check_claim(
        self, claim: Any, experiment_by_id: Mapping[str, Experiment]
    ) -> list[Finding]:
        findings: list[Finding] = []
        target = claim.claim_id
        experiments = [
            experiment_by_id[eid]
            for eid in claim.supporting_experiments
            if eid in experiment_by_id
        ]
        language_rung = self.rung_for_text(claim.statement)
        supported_rung = max(
            (self.rung_for_experiment(e) for e in experiments),
            key=lambda rung: rung.rank,
            default=MechanismRung.OBSERVATION,
        )
        gap = language_rung.rank - supported_rung.rank
        base_details = {
            "claim_rung": language_rung.value,
            "supported_rung": supported_rung.value,
            "gap": gap,
            "supporting_experiments": [e.experiment_id for e in experiments],
        }

        # --- the language is stronger than the evidence ---------------------------------------
        if gap >= 1:
            code = _OVERREACH_CODE_BY_RUNG[language_rung]
            findings.append(
                _finding(
                    code,
                    Severity.BLOCKER if gap >= 2 else Severity.HIGH,
                    f"claim {target!r} speaks at {language_rung.value} "
                    f"(rank {language_rung.rank}) but its evidence reaches only "
                    f"{supported_rung.value} (rank {supported_rung.rank}); gap of {gap} rung(s)",
                    target_ref=target,
                    claim_ids=[target],
                    experiment_ids=[e.experiment_id for e in experiments],
                    quote=claim.statement,
                    blocks=gap >= 2,
                    details=base_details,
                )
            )

        # --- "the mechanism" is a phrase with a licence ----------------------------------------
        if self._asserts_mechanism(claim.statement) and supported_rung.rank < _MECHANISM_RUNG.rank:
            mechanism_gap = _MECHANISM_RUNG.rank - supported_rung.rank
            findings.append(
                _finding(
                    "MECHANISM_CLAIM_OVERREACH",
                    Severity.BLOCKER if mechanism_gap >= 2 else Severity.HIGH,
                    f"claim {target!r} states that the mechanism is (or has been) established, but "
                    f"its evidence reaches only {supported_rung.value}; establishing a mechanism "
                    "requires isolating the component (removal/forcing), not observing a difference",
                    target_ref=target,
                    claim_ids=[target],
                    experiment_ids=[e.experiment_id for e in experiments],
                    quote=claim.statement,
                    blocks=mechanism_gap >= 2,
                    details={**base_details, "asserts_mechanism": True},
                )
            )

        # --- comparative language without a matched control -------------------------------------
        if language_rung.rank >= MechanismRung.CONTROLLED_COMPARISON.rank and experiments:
            controlled = [
                e
                for e in experiments
                if e.design.has_control
                and (e.matched_conditions or (e.control and e.control.matched_on))
            ]
            if not controlled:
                findings.append(
                    _finding(
                        "NO_MATCHED_CONTROL",
                        Severity.HIGH,
                        f"claim {target!r} uses comparative language "
                        f"({language_rung.value}) but none of its experiments records a matched "
                        "control arm",
                        target_ref=target,
                        claim_ids=[target],
                        experiment_ids=[e.experiment_id for e in experiments],
                        quote=claim.statement,
                        details=base_details,
                    )
                )

        # --- placement comparisons are associations between placements --------------------------
        placements = [e for e in experiments if self._is_placement_comparison(e)]
        if placements and (
            self._asserts_mechanism(claim.statement)
            or language_rung.rank >= MechanismRung.INTERVENTION.rank
        ):
            findings.append(
                _finding(
                    "PLACEMENT_EFFECT_OVERCLAIM",
                    Severity.HIGH,
                    f"claim {target!r} attributes a mechanism-level conclusion to "
                    f"{', '.join(e.experiment_id for e in placements)}, which compares two write "
                    "placements under a matched control",
                    target_ref=target,
                    claim_ids=[target],
                    experiment_ids=[e.experiment_id for e in placements],
                    quote=claim.statement,
                    blocks=self._asserts_mechanism(claim.statement),
                    details={**base_details, "placement_comparison": True},
                )
            )

        findings.extend(self._check_experiment_confounds(experiments, target))
        return findings

    @staticmethod
    def _asserts_mechanism(text: str) -> bool:
        lowered = " ".join(text.lower().split())
        return any(phrase in lowered for phrase in _MECHANISM_ASSERTIONS)

    def _check_experiment_confounds(
        self, experiments: Sequence[Experiment], claim_id: str
    ) -> list[Finding]:
        """Per-experiment confound and un-eliminated-alternative checks for a claim's evidence."""
        findings: list[Finding] = []
        for experiment in experiments:
            target = experiment.experiment_id
            design = experiment.design
            if design.confounds_uncontrolled:
                findings.append(
                    _finding(
                        "UNCONTROLLED_CONFOUND",
                        Severity.HIGH,
                        f"experiment {target!r} declares {len(design.confounds_uncontrolled)} "
                        f"uncontrolled confound(s): {'; '.join(design.confounds_uncontrolled)}",
                        target_ref=target,
                        claim_ids=[claim_id],
                        experiment_ids=[target],
                        details={"confounds": list(design.confounds_uncontrolled)},
                    )
                )
            explanations = self.alternative_explanations(experiment)
            if explanations:
                findings.append(
                    _finding(
                        "ALTERNATIVE_EXPLANATION_UNELIMINATED",
                        Severity.MEDIUM,
                        f"experiment {target!r} leaves {len(explanations)} alternative "
                        f"explanation(s) un-eliminated by its design: {'; '.join(explanations)}",
                        target_ref=target,
                        claim_ids=[claim_id],
                        experiment_ids=[target],
                        details={"explanations": explanations},
                    )
                )
        return findings

    # ================================================================== dependency checks

    def _check_dependencies(
        self, experiments: Sequence[Experiment], claims: Sequence[Any]
    ) -> list[Finding]:
        """Single-seed and single-model dependencies across the audited evidence."""
        findings: list[Finding] = []
        for experiment in experiments:
            if len(set(experiment.seeds)) < 2:
                findings.append(
                    _finding(
                        "SINGLE_SEED_DEPENDENCY",
                        Severity.HIGH,
                        f"experiment {experiment.experiment_id!r} rests on {len(set(experiment.seeds))} "
                        f"seed(s) ({sorted(set(experiment.seeds)) or 'none recorded'})",
                        target_ref=experiment.experiment_id,
                        experiment_ids=[experiment.experiment_id],
                        details={"seeds": sorted(set(experiment.seeds))},
                    )
                )
        models = {e.model for e in experiments if e.model is not None}
        if experiments and len(models) == 1:
            only = next(iter(models))
            findings.append(
                _finding(
                    "SINGLE_MODEL_DEPENDENCY",
                    Severity.HIGH,
                    f"all {len(experiments)} experiment(s) in scope use the single model "
                    f"{only!r}; no effect can be attributed to models in general",
                    target_ref=None,
                    claim_ids=[c.claim_id for c in claims],
                    experiment_ids=[e.experiment_id for e in experiments],
                    details={"model": only, "experiments": len(experiments)},
                )
            )
        return findings


__all__ = ["MECHANISM_FINDING_CODES", "MechanismAuditor"]
