"""Statistical audit: recompute the numbers, and refuse to take anyone's word for them.

**Why this module exists.** ARCHITECTURE.md §4.11 asks for an auditor that recomputes
``n, mean, std, SE, CI, effect size, test statistic, p-value`` from artifacts and then checks the
things that make a result unreproducible even when the arithmetic is right: inconsistent seeds,
missing or duplicated runs, silently dropped failures, unmatched baselines, mismatched training
budgets, undefined or drifting metric definitions, contaminable metrics, uncorrected families of
comparisons, and numerals in prose that no artifact produced.

The auditor **reports and never repairs**. It holds ``Cap.AUDIT_CREATE`` and nothing else that
matters: it cannot edit an experiment, bump a claim or remove a number. That is the point — the
audit is a record of what was found at a moment in time, stored as an immutable :class:`Audit`
whose verdict is derived from its findings, so a PASS cannot coexist with a blocker.

Two design commitments are worth stating because they decide most of the code below:

* **A finding needs a code, a reason and a suggestion.** Every code this class emits is listed in
  :data:`FINDING_CODES`. ``blocks_publication`` is set only for the genuinely disqualifying four:
  no analysis artifact, a numeral with no artifact behind it, unrecorded exclusions, and missing
  ``n``/dispersion behind a supported claim. Everything else is a warning with a severity, because
  an auditor that blocks on style teaches researchers to ignore audits.
* **Numbers in prose are checked against the artifact's numbers, tolerantly but not loosely.**
  :meth:`StatisticalAuditor.compare_prose_numbers` is the audit-side twin of the paper compiler's
  number gate: the same numeral set, applied before the paper exists rather than after.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..kernel.permissions import Cap
from ..models.analysis import Analysis, StatMethod, StatResult
from ..models.audit import Audit, AuditVerdict, Finding
from ..models.common import AuditKind, ExperimentStatus, Severity, sha256_file
from ..models.experiment import Experiment
from . import statistics as stats

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from ..kernel.kernel import ResearchKernel
    from ..kernel.permissions import Principal

#: Every code this auditor may emit. Pinned by tests so a rename cannot pass unnoticed.
FINDING_CODES: tuple[str, ...] = (
    "NO_ANALYSIS_ARTIFACT",
    "INSUFFICIENT_N",
    "MISSING_DISPERSION",
    "SEED_INCONSISTENCY",
    "DUPLICATE_RUN",
    "MISSING_RUN",
    "FAILED_RUN_SILENTLY_EXCLUDED",
    "EXCLUSION_UNRECORDED",
    "BASELINE_NOT_MATCHED",
    "BUDGET_MISMATCH",
    "METRIC_DEFINITION_MISSING",
    "METRIC_DEFINITION_INCONSISTENT",
    "METRIC_CONTAMINATION_RISK",
    "START_POINT_CONTAMINATION",
    "MULTIPLE_COMPARISONS_UNCORRECTED",
    "PAIRED_UNPAIRED_AMBIGUOUS",
    "NUMBER_NOT_FOUND_IN_ARTIFACT",
    "NO_EFFECT_SIZE",
    "NO_CONFIDENCE_INTERVAL",
    "UNDECLARED_OBSERVATIONAL_DESIGN",
)

#: A comparison family needs at least this many inferential results before "uncorrected" matters.
MIN_FAMILY_SIZE = 3

#: Words that name an evaluation window. A metric definition without one cannot be contamination-checked.
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
    "probe",
    "split",
    "holdout",
)

#: Numerals in prose. The lookbehind rejects digits glued to an identifier ("GPT-2", "v1.2") so a
#: model name is never audited as a measurement.
_NUMBER_RE = re.compile(
    r"(?<![\w.+\-\u2212])(?P<num>[+\-\u2212]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)(?P<pct>%?)"
)

#: Labels that make a numeral structural rather than a measurement ("Figure 3", "Section 2").
_STRUCTURAL_RE = re.compile(
    r"(?i)\b(fig(?:ure)?s?|tab(?:le)?s?|eq(?:uation|n)?s?|sec(?:tion)?s?|alg(?:orithm)?s?|"
    r"appendix|step|ref(?:erence)?s?|item|row|col(?:umn)?s?|chapter|supp(?:lement)?|"
    r"prop(?:osition)?|lemma|thm|theorem|def(?:inition)?|remark)\s*[.\u00a7#]?\s*$"
)

#: Citation-looking brackets: "[12]", "[3, 4]", "(Smith et al., 2020)". Data brackets survive.
_CITATION_BRACKET_RE = re.compile(
    r"\[\s*\d+(?:\s*[,;\u2013\-]\s*\d+)*\s*\]|\([^()]*(?:et al|arXiv|doi|\b(?:19|20)\d{2}\b)[^()]*\)",
    re.IGNORECASE,
)


def _finding(
    code: str,
    severity: Severity,
    message: str,
    *,
    suggestion: str,
    target_ref: str | None = None,
    blocks: bool = False,
    quote: str | None = None,
    location: str | None = None,
    experiment_ids: Sequence[str] = (),
    claim_ids: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> Finding:
    """Build one finding. ``suggestion`` is required: a finding without a next step is noise."""
    if code not in FINDING_CODES:
        raise ValueError(f"unknown finding code {code!r}; add it to FINDING_CODES and a test")
    if blocks and severity not in (Severity.HIGH, Severity.BLOCKER):
        raise ValueError(f"{code}: a blocking finding must be HIGH or BLOCKER, got {severity.value}")
    return Finding(
        code=code,
        severity=severity,
        message=message,
        target_ref=target_ref,
        experiment_ids=list(experiment_ids),
        claim_ids=list(claim_ids),
        quote=quote[:2000] if quote else None,
        location=location,
        suggestion=suggestion,
        blocks_publication=blocks,
        details=dict(details or {}),
    )


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence containing ``[start, end)`` — findings quote context, not isolated digits."""
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    right_candidates = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right + 1].strip()


def _to_float(raw: str | None) -> float | None:
    """Parse one CSV cell; ``None`` for anything that is not a finite number.

    Thousands separators are tolerated (``"1,234.5"``) because exported CSVs contain them; anything
    else that does not parse is reported as a skipped row rather than coerced to 0.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if "," in text:
        candidate = text.replace(",", "")
        if candidate.replace(".", "").replace("-", "").isdigit():
            text = candidate
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _budget_is_unrecorded(budget: Any) -> bool:
    """True when a :class:`TrainingBudget` records no quantity at all."""
    return all(
        getattr(budget, name) is None
        for name in ("steps", "tokens", "epochs", "wall_clock_seconds", "flops")
    )


class StatisticalAuditor:
    """Recomputes and cross-checks the statistics behind experiments, analyses and claims."""

    #: Recorded on every audit so a re-run can be attributed to a version of this code.
    TOOL = "researchos.analysis.statistical_audit/0.1"

    def __init__(self, kernel: "ResearchKernel") -> None:
        self.kernel = kernel

    # ================================================================== scope resolution

    def _resolve_experiments(
        self,
        experiment_ids: Sequence[str],
        analyses: Sequence[Analysis],
        claims: Sequence[Any],
    ) -> list[Experiment]:
        """Experiments in scope: the named ones, those behind the named analyses, and those behind
        the claims in scope. With nothing named at all, the whole project is in scope."""
        wanted: set[str] = set(experiment_ids)
        for analysis in analyses:
            wanted.update(analysis.experiment_ids)
        for claim in claims:
            wanted.update(claim.supporting_experiments)
            wanted.update(claim.contradicting_experiments)
        store = self.kernel.experiments
        if not wanted:
            return sorted(store.all(), key=lambda e: e.experiment_id)
        found = [store.get(eid) for eid in sorted(wanted)]
        return [experiment for experiment in found if experiment is not None]

    def _resolve_analyses(self, analysis_ids: Sequence[str]) -> list[Analysis]:
        """Named analyses, or every analysis in the project when none are named."""
        store = self.kernel.analyses
        if analysis_ids:
            found = [store.get(aid) for aid in dict.fromkeys(analysis_ids)]
            return sorted((a for a in found if a is not None), key=lambda a: a.analysis_id)
        return sorted(store.all(), key=lambda a: a.analysis_id)

    def _resolve_claims(self, claim_ids: Sequence[str]) -> list[Any]:
        store = self.kernel.claims
        if claim_ids:
            found = [store.get(cid) for cid in dict.fromkeys(claim_ids)]
            return sorted((c for c in found if c is not None), key=lambda c: c.claim_id)
        return sorted(store.all(), key=lambda c: c.claim_id)

    def _relevant_claims(
        self, pool: Sequence[Any], experiments: Sequence[Experiment], *, explicit: bool
    ) -> list[Any]:
        """Claims worth auditing: the named ones, else those citing an experiment in scope.

        An unaudited claim is not evidence of anything, but auditing the whole claim registry on
        every narrow question would bury the findings that matter, so unrelated claims are dropped
        only when the caller narrowed the scope.
        """
        if explicit:
            return list(pool)
        ids = {e.experiment_id for e in experiments}
        return [
            claim
            for claim in pool
            if ids & (set(claim.supporting_experiments) | set(claim.contradicting_experiments))
        ]

    # ================================================================== entry point

    def audit(
        self,
        principal: "Principal",
        *,
        experiment_ids: Sequence[str] = (),
        analysis_ids: Sequence[str] = (),
        claim_ids: Sequence[str] = (),
        task_id: str | None = None,
        title: str | None = None,
    ) -> Audit:
        """Audit the named records (or everything, when nothing is named) and persist the result.

        The audit is stored with ``kernel.save_audit`` and is never mutated afterwards: a re-run
        produces a *new* audit, so the history of "what we knew about these numbers" survives.
        """
        principal.require(Cap.AUDIT_CREATE, "audit.create")

        analyses = self._resolve_analyses(analysis_ids)
        claim_pool = self._resolve_claims(claim_ids)
        # Resolve the *experiments* first from the named records, then keep only the claims that
        # actually cite them; otherwise a project-wide claim registry would drag unrelated
        # experiments into a narrowly scoped audit.
        if experiment_ids or analysis_ids:
            provisional = self._resolve_experiments(experiment_ids, analyses, [])
            claims = self._relevant_claims(claim_pool, provisional, explicit=bool(claim_ids))
        else:
            claims = list(claim_pool)
        experiments = self._resolve_experiments(experiment_ids, analyses, claims)
        if not (experiment_ids or analysis_ids or claim_ids):
            claims = self._relevant_claims(claim_pool, experiments, explicit=False)

        experiment_by_id = {e.experiment_id: e for e in experiments}
        supported_experiments = {
            eid
            for claim in claims
            if claim.status.value in ("SUPPORTED", "ROBUST")
            for eid in claim.supporting_experiments
        }
        supported_analyses = {
            analysis.analysis_id
            for analysis in analyses
            if set(analysis.experiment_ids) & supported_experiments
        }

        findings: list[Finding] = []
        findings.extend(self._check_experiments(experiments, supported_experiments))
        findings.extend(self._check_failed_runs(experiments, analyses))
        findings.extend(self._check_analyses(analyses, experiment_by_id, supported_analyses))
        findings.extend(self._check_claims(claims, experiment_by_id))
        findings.sort(key=lambda f: (f.code, f.target_ref or "", f.quote or "", f.message))

        subjects = sorted(
            {e.experiment_id for e in experiments}
            | {a.analysis_id for a in analyses}
            | {c.claim_id for c in claims}
        )
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
            f"{len(findings)} finding(s) across {len(experiments)} experiment(s), "
            f"{len(analyses)} analysis(es) and {len(claims)} claim(s); "
            f"{len(blocking)} blocking. Prose numerals are not checked here — call "
            "compare_prose_numbers against the analysis artifact's numeric_map()."
        )
        if not subjects:
            summary = "nothing was in scope, so no statistical claim could be checked"

        audit = Audit(
            kind=AuditKind.STATISTICAL,
            title=title or "Statistical audit",
            subjects=subjects,
            findings=findings,
            summary=summary,
            verdict=verdict,
            inputs_hashed=self._hash_inputs(experiments, analyses),
            tool=self.TOOL,
            created_by=principal.name,
            task_id=task_id,
        )
        return self.kernel.save_audit(principal, audit)

    def _hash_inputs(
        self, experiments: Sequence[Experiment], analyses: Sequence[Analysis]
    ) -> dict[str, str]:
        """Hash every artifact referenced by the audited records, so a re-run is comparable.

        Missing files are skipped rather than raising: the audit's job is to report, and a vanished
        artifact is a provenance problem for another auditor to record.
        """
        out: dict[str, str] = {}
        refs = [ref for e in experiments for ref in e.analysis_artifacts]
        refs += [ref for e in experiments for ref in e.raw_artifacts]
        refs += [a.output_artifact for a in analyses if a.output_artifact is not None]
        for ref in refs:
            path = Path(ref.path)
            if not path.is_absolute():
                path = self.kernel.root / ref.path
            if path.is_file():
                out[ref.path] = sha256_file(path)
        return out

    # ================================================================== experiment checks

    def _check_experiments(
        self, experiments: Sequence[Experiment], supported_experiments: set[str]
    ) -> list[Finding]:
        findings: list[Finding] = []
        completed = [e for e in experiments if e.status is ExperimentStatus.COMPLETED]

        for experiment in experiments:
            target = experiment.experiment_id

            # --- every number must have an artifact address -------------------------------
            if experiment.status is ExperimentStatus.COMPLETED and not (
                experiment.analysis_artifacts or experiment.analysis_ids
            ):
                findings.append(
                    _finding(
                        "NO_ANALYSIS_ARTIFACT",
                        Severity.BLOCKER,
                        f"experiment {target!r} ({experiment.title}) is COMPLETED but references no "
                        "analysis artifact and no analysis id",
                        target_ref=target,
                        experiment_ids=[target],
                        blocks=True,
                        suggestion=(
                            "run the analysis agent over the raw artifacts and attach the resulting "
                            "Analysis/artifact to the experiment; a headline value with no artifact "
                            "address cannot be compiled into a paper"
                        ),
                    )
                )

            # --- design declared at all ----------------------------------------------------
            design = experiment.design
            if (
                experiment.status is ExperimentStatus.COMPLETED
                and not design.has_control
                and not design.has_intervention
                and not design.is_observational
            ):
                findings.append(
                    _finding(
                        "UNDECLARED_OBSERVATIONAL_DESIGN",
                        Severity.HIGH,
                        f"experiment {target!r} declares neither a control arm, an intervention, nor "
                        "an observational design",
                        target_ref=target,
                        experiment_ids=[target],
                        suggestion=(
                            "declare the design honestly: set design.is_observational=True for a "
                            "purely observational study, or record the control arm; an undeclared "
                            "design cannot be placed on the evidence ladder"
                        ),
                    )
                )

            # --- baseline matching ----------------------------------------------------------
            if design.has_control and not (
                experiment.matched_conditions or (experiment.control and experiment.control.matched_on)
            ):
                findings.append(
                    _finding(
                        "BASELINE_NOT_MATCHED",
                        Severity.HIGH,
                        f"experiment {target!r} declares a control arm but records no matched "
                        "conditions between the arms",
                        target_ref=target,
                        experiment_ids=[target],
                        suggestion=(
                            "list the factors held equal across arms (tokenizer, data order, "
                            "optimizer, schedule, step count) in matched_conditions; an unmatched "
                            "control cannot separate the treatment from a nuisance difference"
                        ),
                    )
                )

            # --- seeds and runs -------------------------------------------------------------
            seeds = list(experiment.seeds)
            if len(seeds) != len(set(seeds)):
                findings.append(
                    _finding(
                        "DUPLICATE_RUN",
                        Severity.HIGH,
                        f"experiment {target!r} records the same seed more than once: {seeds}",
                        target_ref=target,
                        experiment_ids=[target],
                        suggestion=(
                            "drop the duplicated seed entry or record the runs separately; a repeated "
                            "seed is one run counted twice, which inflates apparent replication"
                        ),
                        details={"seeds": seeds},
                    )
                )
            if experiment.n is not None and seeds and len(set(seeds)) < experiment.n:
                findings.append(
                    _finding(
                        "MISSING_RUN",
                        Severity.HIGH,
                        f"experiment {target!r} declares n={experiment.n} but records only "
                        f"{len(set(seeds))} distinct seed(s)",
                        target_ref=target,
                        experiment_ids=[target],
                        suggestion=(
                            "record the missing runs or lower n to the number of runs actually "
                            "completed; the gap between declared and observed runs is where silent "
                            "exclusions hide"
                        ),
                        details={"n": experiment.n, "seeds": seeds},
                    )
                )
            provenance_seeds = set(experiment.provenance.random_seeds)
            if seeds and provenance_seeds and provenance_seeds != set(seeds):
                findings.append(
                    _finding(
                        "SEED_INCONSISTENCY",
                        Severity.HIGH,
                        f"experiment {target!r} declares seeds {sorted(set(seeds))} but its provenance "
                        f"records {sorted(provenance_seeds)}",
                        target_ref=target,
                        experiment_ids=[target],
                        suggestion=(
                            "reconcile the declared seeds with the provenance record; if the run used "
                            "different seeds than declared, the declared replication count is wrong"
                        ),
                    )
                )

            # --- budget ---------------------------------------------------------------------
            if experiment.status is ExperimentStatus.COMPLETED and _budget_is_unrecorded(
                experiment.training_budget
            ):
                findings.append(
                    _finding(
                        "BUDGET_MISMATCH",
                        Severity.MEDIUM,
                        f"experiment {target!r} records no training budget at all",
                        target_ref=target,
                        experiment_ids=[target],
                        suggestion=(
                            "record steps/tokens/epochs in training_budget; without a budget two arms "
                            "cannot be shown to have received the same amount of training"
                        ),
                    )
                )

            # --- n and dispersion -----------------------------------------------------------
            if experiment.status is ExperimentStatus.COMPLETED and (
                experiment.n is None or experiment.n < 2
            ):
                blocks = target in supported_experiments
                findings.append(
                    _finding(
                        "INSUFFICIENT_N",
                        Severity.HIGH if blocks else Severity.MEDIUM,
                        f"experiment {target!r} is COMPLETED with n={experiment.n}; a single run "
                        "cannot support a spread, an interval or a p-value",
                        target_ref=target,
                        experiment_ids=[target],
                        blocks=blocks,
                        suggestion=(
                            "run at least two seeds and record n, or downgrade the claim this "
                            "experiment supports to an observation"
                        ),
                        details={"n": experiment.n, "supported_claim": blocks},
                    )
                )

            findings.extend(self._check_metrics(experiment))

        findings.extend(self._check_metric_consistency(completed))
        findings.extend(self._check_cross_experiment_duplicates(experiments))
        return findings

    def _check_metrics(self, experiment: Experiment) -> list[Finding]:
        findings: list[Finding] = []
        target = experiment.experiment_id
        for metric in experiment.metrics:
            if not metric.is_defined:
                findings.append(
                    _finding(
                        "METRIC_DEFINITION_MISSING",
                        Severity.HIGH,
                        f"experiment {target!r} reports metric {metric.name!r} with no definition",
                        target_ref=f"{target}:{metric.name}",
                        experiment_ids=[target],
                        suggestion=(
                            "write the exact computation, the evaluation split and the averaging "
                            "order into MetricRef.definition; an undefined metric cannot be "
                            "reproduced or compared with prior work"
                        ),
                    )
                )
                continue
            if metric.contamination_risks:
                findings.append(
                    _finding(
                        "METRIC_CONTAMINATION_RISK",
                        Severity.MEDIUM,
                        f"metric {metric.name!r} in experiment {target!r} declares contamination "
                        f"risk(s): {'; '.join(metric.contamination_risks)}",
                        target_ref=f"{target}:{metric.name}",
                        experiment_ids=[target],
                        suggestion=(
                            "either remove the risk (evaluate on a split the model never saw) or "
                            "carry the risk into the claim's limitations and the paper's threats to "
                            "validity"
                        ),
                        details={"risks": list(metric.contamination_risks)},
                    )
                )
            lowered = metric.definition.lower()
            if not any(word in lowered for word in _SPLIT_WORDS):
                findings.append(
                    _finding(
                        "START_POINT_CONTAMINATION",
                        Severity.MEDIUM,
                        f"metric {metric.name!r} in experiment {target!r} does not say where "
                        "measurement starts or which split it is measured on",
                        target_ref=f"{target}:{metric.name}",
                        experiment_ids=[target],
                        suggestion=(
                            "state the evaluation window and split in the definition (e.g. 'measured "
                            "on the held-out split from step 0'); without it, a measurement that "
                            "begins inside the manipulated segment cannot be ruled out"
                        ),
                        details={"definition": metric.definition},
                    )
                )
        return findings

    def _check_metric_consistency(self, experiments: Sequence[Experiment]) -> list[Finding]:
        """The same metric *name* with two definitions makes two experiments incomparable."""
        definitions: dict[str, dict[str, list[str]]] = {}
        for experiment in experiments:
            for metric in experiment.metrics:
                if metric.is_defined:
                    definitions.setdefault(metric.name, {}).setdefault(
                        metric.definition.strip(), []
                    ).append(experiment.experiment_id)
        findings: list[Finding] = []
        for name in sorted(definitions):
            variants = definitions[name]
            if len(variants) < 2:
                continue
            findings.append(
                _finding(
                    "METRIC_DEFINITION_INCONSISTENT",
                    Severity.HIGH,
                    f"metric {name!r} has {len(variants)} different definitions across experiments "
                    f"({', '.join(sorted(eid for ids in variants.values() for eid in ids))})",
                    target_ref=name,
                    experiment_ids=sorted(eid for ids in variants.values() for eid in ids),
                    suggestion=(
                        "use one definition per metric name (or rename the variants); a table that "
                        "compares two differently-defined metrics compares nothing"
                    ),
                    details={"definitions": {k: sorted(v) for k, v in variants.items()}},
                )
            )
        return findings

    def _check_cross_experiment_duplicates(self, experiments: Sequence[Experiment]) -> list[Finding]:
        """Same design identity and overlapping seeds = the same run registered twice."""
        groups: dict[str, list[Experiment]] = {}
        for experiment in experiments:
            groups.setdefault(experiment.identity_signature(), []).append(experiment)
        findings: list[Finding] = []
        for signature in sorted(groups):
            group = groups[signature]
            if len(group) < 2:
                continue
            seen: dict[int, str] = {}
            for experiment in sorted(group, key=lambda e: e.experiment_id):
                for seed in experiment.seeds:
                    if seed in seen:
                        findings.append(
                            _finding(
                                "DUPLICATE_RUN",
                                Severity.HIGH,
                                f"seed {seed} appears in both {seen[seed]!r} and "
                                f"{experiment.experiment_id!r}, which share a design identity",
                                target_ref=experiment.experiment_id,
                                experiment_ids=[seen[seed], experiment.experiment_id],
                                suggestion=(
                                    "either the two records describe one run (merge them) or one of "
                                    "them used a different seed than recorded; as it stands the "
                                    "replication count is inflated"
                                ),
                                details={"seed": seed, "identity_signature": signature[:16]},
                            )
                        )
                    else:
                        seen[seed] = experiment.experiment_id
        return findings

    def _check_failed_runs(
        self, experiments: Sequence[Experiment], analyses: Sequence[Analysis]
    ) -> list[Finding]:
        """A failed run that vanished from the record is a biased sample of the runs."""
        groups: dict[str, list[Experiment]] = {}
        for experiment in experiments:
            groups.setdefault(experiment.identity_signature(), []).append(experiment)
        findings: list[Finding] = []
        for signature in sorted(groups):
            group = groups[signature]
            good = [e for e in group if e.status is ExperimentStatus.COMPLETED]
            bad = [
                e
                for e in group
                if e.status in (ExperimentStatus.FAILED, ExperimentStatus.ABORTED)
            ]
            if not good or not bad:
                continue
            for failed in sorted(bad, key=lambda e: e.experiment_id):
                recorded = self._failure_is_recorded(failed, good, analyses)
                if recorded:
                    continue
                findings.append(
                    _finding(
                        "FAILED_RUN_SILENTLY_EXCLUDED",
                        Severity.MEDIUM,
                        f"{failed.status.value} run {failed.experiment_id!r} shares a design with "
                        f"{', '.join(e.experiment_id for e in good)} but is referenced nowhere",
                        target_ref=failed.experiment_id,
                        experiment_ids=[failed.experiment_id, *[e.experiment_id for e in good]],
                        suggestion=(
                            "record the failed run in the analysis' excluded_inputs with its reason "
                            "(or in the experiment's limitations); failures that disappear from the "
                            "record turn a fair sample of runs into a favourable one"
                        ),
                        details={"failure_reason": failed.failure_reason},
                    )
                )
        return findings

    def _failure_is_recorded(
        self, failed: Experiment, siblings: Sequence[Experiment], analyses: Sequence[Analysis]
    ) -> bool:
        needles = {failed.experiment_id}
        if failed.title:
            needles.add(failed.title)
        for analysis in analyses:
            for item, reason in analysis.excluded_inputs:
                if any(needle in item or needle in reason for needle in needles):
                    return True
        for sibling in siblings:
            notes = list(sibling.limitations) + list(sibling.unexpected_observations)
            if sibling.result is not None:
                notes += list(sibling.result.notes)
            blob = " ".join(notes)
            if any(needle in blob for needle in needles):
                return True
        return False

    # ================================================================== analysis checks

    def _check_analyses(
        self,
        analyses: Sequence[Analysis],
        experiment_by_id: Mapping[str, Experiment],
        supported_analyses: set[str],
    ) -> list[Finding]:
        findings: list[Finding] = []
        for analysis in analyses:
            target = analysis.analysis_id
            exps = [
                experiment_by_id[eid] for eid in analysis.experiment_ids if eid in experiment_by_id
            ]

            # --- unrecorded exclusions ------------------------------------------------------
            expected = sum(len(set(e.seeds)) for e in exps)
            # The analysis model keeps per-result input counts; the analysis-level count is their sum
            # (an analysis with no recorded inputs is simply unknown, not zero).
            recorded_inputs = [r.n_inputs for r in analysis.results if r.n_inputs is not None]
            observed = sum(recorded_inputs) if recorded_inputs else None
            if observed is None:
                observed = max((r.n for r in analysis.results if r.n is not None), default=None)
            if expected > 0 and observed is not None and observed < expected and not analysis.excluded_inputs:
                findings.append(
                    _finding(
                        "EXCLUSION_UNRECORDED",
                        Severity.HIGH,
                        f"analysis {target!r} used {observed} of {expected} declared runs and records "
                        "no excluded_inputs",
                        target_ref=target,
                        experiment_ids=[e.experiment_id for e in exps],
                        blocks=True,
                        suggestion=(
                            "list every dropped run and the reason in excluded_inputs; an undisclosed "
                            "exclusion is indistinguishable from cherry-picking"
                        ),
                        details={"expected_runs": expected, "observed_runs": observed},
                    )
                )

            # --- baseline for a comparison --------------------------------------------------
            inferential = [r for r in analysis.results if r.is_inferential()]
            if inferential and exps and not any(e.design.has_control for e in exps):
                findings.append(
                    _finding(
                        "BASELINE_NOT_MATCHED",
                        Severity.HIGH,
                        f"analysis {target!r} reports {len(inferential)} inferential result(s) but "
                        "none of its experiments declares a control arm",
                        target_ref=target,
                        experiment_ids=[e.experiment_id for e in exps],
                        suggestion=(
                            "add the baseline/control condition, or report the comparison as an "
                            "association between observed conditions rather than as an effect"
                        ),
                    )
                )

            # --- training budget across compared arms ---------------------------------------
            findings.extend(self._check_budget(analysis, exps))

            # --- per-result checks ------------------------------------------------------------
            for result in analysis.results:
                if not result.is_inferential():
                    continue
                blocks = target in supported_analyses
                if result.n is None or result.n < 2:
                    findings.append(
                        _finding(
                            "INSUFFICIENT_N",
                            Severity.HIGH if blocks else Severity.MEDIUM,
                            f"result {result.name!r} in analysis {target!r} carries a p-value with "
                            f"n={result.n}",
                            target_ref=f"{target}:{result.name}",
                            experiment_ids=list(analysis.experiment_ids),
                            blocks=blocks,
                            suggestion=(
                                "report the per-arm n and re-run with at least two independent runs "
                                "per arm; a p-value without n is not checkable"
                            ),
                        )
                    )
                if result.std is None and result.se is None:
                    findings.append(
                        _finding(
                            "MISSING_DISPERSION",
                            Severity.HIGH if blocks else Severity.MEDIUM,
                            f"result {result.name!r} in analysis {target!r} reports no std and no se",
                            target_ref=f"{target}:{result.name}",
                            experiment_ids=list(analysis.experiment_ids),
                            blocks=blocks,
                            suggestion=(
                                "record the standard deviation or the standard error; a mean without "
                                "dispersion hides whether the difference is inside the noise"
                            ),
                        )
                    )
                if result.effect_size is None:
                    findings.append(
                        _finding(
                            "NO_EFFECT_SIZE",
                            Severity.MEDIUM,
                            f"result {result.name!r} in analysis {target!r} reports no effect size",
                            target_ref=f"{target}:{result.name}",
                            experiment_ids=list(analysis.experiment_ids),
                            suggestion=(
                                "report Cohen's d (or Hedges' g for small samples) alongside the "
                                "p-value; significance alone does not say whether the effect matters"
                            ),
                        )
                    )
                if result.ci_low is None or result.ci_high is None:
                    findings.append(
                        _finding(
                            "NO_CONFIDENCE_INTERVAL",
                            Severity.MEDIUM,
                            f"result {result.name!r} in analysis {target!r} reports no confidence "
                            "interval",
                            target_ref=f"{target}:{result.name}",
                            experiment_ids=list(analysis.experiment_ids),
                            suggestion=(
                                "report an interval (t-based or bootstrap) with its level; the "
                                "interval is what a reader uses to judge precision"
                            ),
                        )
                    )
                if result.paired is None:
                    findings.append(
                        _finding(
                            "PAIRED_UNPAIRED_AMBIGUOUS",
                            Severity.HIGH,
                            f"result {result.name!r} in analysis {target!r} does not declare whether "
                            "the test was paired",
                            target_ref=f"{target}:{result.name}",
                            experiment_ids=list(analysis.experiment_ids),
                            suggestion=(
                                "set StatResult.paired explicitly; paired and unpaired tests on the "
                                "same data give different p-values, and the reader cannot recover "
                                "which one produced this number"
                            ),
                        )
                    )

            findings.extend(self._check_multiple_comparisons(analysis))
        return findings

    def _check_budget(self, analysis: Analysis, exps: Sequence[Experiment]) -> list[Finding]:
        findings: list[Finding] = []
        if len(exps) < 2:
            return findings
        for field_name in ("steps", "tokens", "epochs"):
            declared = {
                getattr(e.training_budget, field_name): e.experiment_id
                for e in exps
                if getattr(e.training_budget, field_name) is not None
            }
            if len(declared) > 1:
                findings.append(
                    _finding(
                        "BUDGET_MISMATCH",
                        Severity.HIGH,
                        f"analysis {analysis.analysis_id!r} compares arms with different "
                        f"training_budget.{field_name}: "
                        + ", ".join(
                            f"{value} ({eid})" for value, eid in sorted(declared.items(), key=lambda kv: str(kv[1]))
                        ),
                        target_ref=analysis.analysis_id,
                        experiment_ids=[e.experiment_id for e in exps],
                        suggestion=(
                            "match the arms on training budget, or state the budget difference as a "
                            "confound; an arm that trained longer may win for that reason alone"
                        ),
                        details={"field": field_name},
                    )
                )
            elif declared and len(declared) != len(exps):
                findings.append(
                    _finding(
                        "BUDGET_MISMATCH",
                        Severity.MEDIUM,
                        f"analysis {analysis.analysis_id!r} compares arms where only some record "
                        f"training_budget.{field_name}",
                        target_ref=analysis.analysis_id,
                        experiment_ids=[e.experiment_id for e in exps],
                        suggestion=(
                            "record the budget for every arm; an unrecorded budget is where an "
                            "unmatched comparison hides"
                        ),
                        details={"field": field_name, "recorded": sorted(declared.values())},
                    )
                )
        return findings

    def _check_multiple_comparisons(self, analysis: Analysis) -> list[Finding]:
        """A family of comparisons without a correction inflates the false-positive rate.

        Families are formed by ``comparison_family``; results with no family form one implicit
        family, because "I did not label them" is not evidence that the comparisons are independent.
        """
        families: dict[str, list[StatResult]] = {}
        for result in analysis.results:
            if not result.is_inferential():
                continue
            families.setdefault(result.comparison_family or "<unlabelled>", []).append(result)
        findings: list[Finding] = []
        for family in sorted(families):
            members = families[family]
            if len(members) < MIN_FAMILY_SIZE:
                continue
            if all(member.corrected_p_value is not None for member in members):
                continue
            uncorrected = [m for m in members if m.corrected_p_value is None]
            findings.append(
                _finding(
                    "MULTIPLE_COMPARISONS_UNCORRECTED",
                    Severity.HIGH,
                    f"analysis {analysis.analysis_id!r} reports {len(members)} inferential results "
                    f"in comparison family {family!r}, of which {len(uncorrected)} carry no "
                    "corrected p-value",
                    target_ref=analysis.analysis_id,
                    experiment_ids=list(analysis.experiment_ids),
                    suggestion=(
                        "apply Holm–Bonferroni (family-wise error control; the right default when "
                        "any single false positive is costly) or Benjamini–Hochberg (false discovery "
                        "rate; the right default for a large exploratory family) and record the "
                        "adjusted values in corrected_p_value with correction_method set"
                    ),
                    details={
                        "family": family,
                        "size": len(members),
                        "uncorrected": [m.name for m in uncorrected],
                    },
                )
            )
        return findings

    # ================================================================== claim checks

    def _check_claims(
        self, claims: Sequence[Any], experiment_by_id: Mapping[str, Experiment]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for claim in claims:
            if claim.status.value not in ("SUPPORTED", "ROBUST"):
                continue
            support = [
                experiment_by_id[eid]
                for eid in claim.supporting_experiments
                if eid in experiment_by_id
            ]
            if not support:
                continue
            without_artifact = [
                e.experiment_id
                for e in support
                if not (e.analysis_artifacts or e.analysis_ids)
            ]
            if without_artifact:
                findings.append(
                    _finding(
                        "NO_ANALYSIS_ARTIFACT",
                        Severity.BLOCKER,
                        f"claim {claim.claim_id!r} is {claim.status.value} but its supporting "
                        f"experiment(s) {without_artifact} have no analysis artifact",
                        target_ref=claim.claim_id,
                        claim_ids=[claim.claim_id],
                        experiment_ids=without_artifact,
                        blocks=True,
                        quote=claim.statement,
                        suggestion=(
                            "attach the analysis artifact that produces the claim's numbers, or "
                            "move the claim back to TESTED; a supported claim whose numbers have no "
                            "artifact is unsupported by construction"
                        ),
                    )
                )
        return findings

    # ================================================================== artifact recomputation

    def recompute_from_artifact(
        self,
        artifact_path: str,
        *,
        value_column: str,
        group_column: str | None = None,
        name_prefix: str = "",
    ) -> list[StatResult]:
        """Recompute descriptive statistics (and a Welch comparison) straight from a CSV artifact.

        The point of this method is that the audited numbers are recomputed from the *bytes the
        experiment produced*, not from the summary someone typed into a YAML file. Rows whose value
        cell is not a finite number are skipped and **counted** — the count goes into the result's
        notes, because "we dropped 4 rows" is a finding even when the drop is legitimate.

        ``name_prefix`` is prepended verbatim (include your own separator, e.g. ``"results."``).
        Results are returned in a deterministic order; persisting them in an :class:`Analysis` is
        the caller's job, so this method stays a pure function of the file.
        """
        path = Path(artifact_path)
        if not path.is_absolute():
            path = self.kernel.root / artifact_path
        if not path.is_file():
            raise FileNotFoundError(f"analysis artifact not found: {path}")
        relative = self.kernel.paths.rel(path)

        groups: dict[str | None, list[float]] = {}
        skipped: dict[str | None, int] = {}
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if value_column not in columns:
                raise ValueError(
                    f"{relative} has no column {value_column!r}; available columns: {columns}. "
                    "Refusing to guess which column was measured."
                )
            if group_column is not None and group_column not in columns:
                raise ValueError(
                    f"{relative} has no column {group_column!r}; available columns: {columns}"
                )
            for row in reader:
                row_count += 1
                key = None if group_column is None else (row.get(group_column) or "").strip()
                value = _to_float(row.get(value_column))
                if value is None:
                    skipped[key] = skipped.get(key, 0) + 1
                    continue
                groups.setdefault(key, []).append(value)

        selector = f"{relative}#{value_column}"
        results: list[StatResult] = []

        if group_column is None:
            values = groups.get(None, [])
            results.append(
                self._descriptive_result(
                    name=f"{name_prefix}{value_column}",
                    values=values,
                    selector=selector,
                    skipped=skipped.get(None, 0),
                    rows=row_count,
                )
            )
            return results

        for key in sorted(groups, key=lambda k: (k or "")):
            values = groups[key]
            results.append(
                self._descriptive_result(
                    name=f"{name_prefix}{key or '<blank>'}",
                    values=values,
                    selector=f"{relative}#{group_column}={key}#{value_column}",
                    skipped=skipped.get(key, 0),
                    rows=len(values) + skipped.get(key, 0),
                    group=key or None,
                )
            )
        for key in sorted(k for k in skipped if k not in groups):
            results.append(
                self._descriptive_result(
                    name=f"{name_prefix}{key or '<blank>'}",
                    values=[],
                    selector=f"{relative}#{group_column}={key}#{value_column}",
                    skipped=skipped[key],
                    rows=skipped[key],
                    group=key or None,
                )
            )

        ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0] or ""))
        if len(ranked) < 2:
            results.append(
                StatResult(
                    name=f"{name_prefix}{value_column}.comparison",
                    method=StatMethod.DESCRIPTIVE,
                    inputs=[selector],
                    n_inputs=row_count,
                    notes=[
                        "no comparison produced: fewer than two groups had numeric values in "
                        f"{group_column!r}"
                    ],
                    warnings=["a grouped comparison needs at least two groups"],
                )
            )
            return results

        (name_a, values_a), (name_b, values_b) = ranked[0], ranked[1]
        comparison = stats.welch_t_test(values_a, values_b)
        results.append(
            stats.to_stat_result(
                f"{name_prefix}{name_a or '<blank>'}_minus_{name_b or '<blank>'}",
                comparison,
                group_a=name_a,
                group_b=name_b,
                inputs=[
                    f"{relative}#{group_column}={name_a}#{value_column}",
                    f"{relative}#{group_column}={name_b}#{value_column}",
                ],
                n_inputs=len(values_a) + len(values_b),
                notes=[
                    *comparison.notes,
                    f"Welch comparison of the two largest groups in {group_column!r}",
                ],
            )
        )
        return results

    def _descriptive_result(
        self,
        *,
        name: str,
        values: Sequence[float],
        selector: str,
        skipped: int,
        rows: int,
        group: str | None = None,
    ) -> StatResult:
        """One descriptive :class:`StatResult` for a column (or one group of a column)."""
        notes: list[str] = [f"{rows} row(s) read, {skipped} skipped as non-numeric"]
        warnings: list[str] = []
        if skipped:
            warnings.append(f"{skipped} non-numeric row(s) skipped in {selector}")
        if not values:
            warnings.append("no numeric values were available in this column")
            notes.append("no statistic computed: the column contained no finite numbers")
            return StatResult(
                name=name,
                method=StatMethod.DESCRIPTIVE,
                n=0,
                n_inputs=rows,
                inputs=[selector],
                group_a=group,
                notes=notes,
                warnings=warnings,
            )
        n = len(values)
        centre = stats.mean(values)
        spread = stats.std(values) if n >= 2 else None
        standard_error = stats.stderr(values) if n >= 2 else None
        low, high = stats.mean_ci(values)
        if n < 2:
            notes.append("n < 2: no standard deviation, standard error or interval was computed")
        return StatResult(
            name=name,
            value=centre,
            n=n,
            n_inputs=rows,
            mean=centre,
            std=spread,
            se=standard_error,
            median=stats.median(values),
            min=min(values),
            max=max(values),
            ci_low=None if math.isinf(low) else low,
            ci_high=None if math.isinf(high) else high,
            ci_level=0.95 if n >= 2 else None,
            method=StatMethod.DESCRIPTIVE,
            inputs=[selector],
            group_a=group,
            notes=notes,
            warnings=warnings,
        )

    # ================================================================== prose numbers

    def compare_prose_numbers(
        self, text: str, allowed: Mapping[str, float], *, tol: float = 0.0
    ) -> list[Finding]:
        """Flag every numeral in prose that no artifact value accounts for.

        This is the audit-side twin of the paper compiler's number gate, available *before* a paper
        exists. A numeral matches when it equals an allowed value, or — for a percentage — equals
        that value divided by 100, since "42.7%" and 0.427 are the same measurement. With ``tol=0``
        the comparison is exact up to float representation (``1e-9`` relative); a positive ``tol``
        accepts an absolute or relative difference of that size, which is what a rounding
        convention needs.

        Structural numerals are not audited: "Figure 3", "Section 2" and citation brackets are
        labels, not measurements, and treating them as ungrounded numbers would bury the real
        findings in noise.
        """
        if tol < 0.0:
            raise ValueError(f"tol must be >= 0, got {tol!r}")
        allowed_values = [float(v) for v in allowed.values()]
        skip_spans = [match.span() for match in _CITATION_BRACKET_RE.finditer(text)]
        findings: list[Finding] = []
        for match in _NUMBER_RE.finditer(text):
            start, end = match.span()
            if any(span_start <= start < span_end for span_start, span_end in skip_spans):
                continue
            if _STRUCTURAL_RE.search(text[:start]):
                continue
            raw = match.group("num").replace("\u2212", "-")
            try:
                value = float(raw)
            except ValueError:  # pragma: no cover - the regex only matches parseable numerals
                continue
            candidates = [value]
            if match.group("pct"):
                candidates.append(value / 100.0)
            if any(self._tolerance_match(candidate, allowed_values, tol) for candidate in candidates):
                continue
            findings.append(
                _finding(
                    "NUMBER_NOT_FOUND_IN_ARTIFACT",
                    Severity.HIGH,
                    f"numeral {match.group(0)!r} in the text matches no value in the analysis "
                    f"artifact ({len(allowed_values)} value(s) allowed)",
                    target_ref="prose",
                    blocks=True,
                    quote=_sentence_around(text, start, end),
                    location=f"char:{start}-{end}",
                    details={
                        "numeral": match.group(0),
                        "value": value,
                        "percent": bool(match.group("pct")),
                        "tolerance": tol,
                        "allowed_keys": sorted(allowed),
                    },
                    suggestion=(
                        "either delete the number from the prose or add the StatResult that produces "
                        "it from an analysis artifact; ResearchOS does not let prose introduce a "
                        "number that no computation supports"
                    ),
                )
            )
        return findings

    @staticmethod
    def _tolerance_match(value: float, allowed_values: Sequence[float], tol: float) -> bool:
        """Match within ``tol`` (absolute or relative); at ``tol=0``, within float representation."""
        for candidate in allowed_values:
            if tol == 0.0:
                if math.isclose(candidate, value, rel_tol=1e-9, abs_tol=1e-12):
                    return True
                continue
            if abs(candidate - value) <= tol:
                return True
            if value != 0.0 and abs(candidate - value) / abs(value) <= tol:
                return True
        return False


__all__ = ["FINDING_CODES", "MIN_FAMILY_SIZE", "StatisticalAuditor"]
