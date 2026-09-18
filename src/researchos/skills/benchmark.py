"""The benchmark harness: fixed tasks in, a deterministic, attributable verdict out.

Why this module exists
----------------------
"Rate this skill 8/10" is not a benchmark. This harness runs a skill — represented by an injected
``SkillRunner`` — against the fixed catalogue for a suite, and turns its *findings* into numbers by
comparing them with what the task fixture actually contains:

* ``score = weighted F1`` over required vs produced findings. Weights are per finding code and
  currently uniform (1.0), so the arithmetic degenerates to plain F1 and stays hand-checkable; the
  weighted form is the hook for per-code severity later, without changing the contract.
* A **forbidden** finding is a false alarm that would block a correct artifact. It hard-fails the
  task and zeroes its score. Over-flagging must be expensive, or the cheapest way to pass a
  benchmark is to report everything.
* ``passed`` means "produced every required finding and no forbidden one". Spurious findings do not
  block ``passed`` (generosity) but they do lower the score (honesty), and the activation gate reads
  the score.
* **Length is never scored.** ``output_text`` is recorded for auditability only; a long answer that
  names no required finding scores 0.

The regression gate lives on ``BenchmarkRun``: :meth:`BenchmarkHarness.run` computes the score and,
when an incumbent run is supplied, compares *task by task*, so an upgrade that gains overall while
breaking a previously-passing task is a regression (invariant 8).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, cast

from ..kernel.errors import SkillError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models import (
    BenchmarkRun,
    BenchmarkSuite,
    BenchmarkTask,
    SkillCard,
    SkillVersion,
    TaskOutcome,
)
from ..models.common import clampf
from ..models.timeline import TimelineEventKind
from .benchmark_tasks import ALL_FINDING_CODES, seed_default_tasks

#: A runner executes one task and returns its outcome. Injected on purpose: the harness never
#: imports or executes skill code itself, and tests inject a deterministic fake.
SkillRunner = Callable[[BenchmarkTask], TaskOutcome]

#: Per-finding weights. Uniform today; a data change here (e.g. BLOCKER findings weighing 2.0)
#: changes scoring without touching the outcome contract.
FINDING_WEIGHTS: Mapping[str, float] = {code: 1.0 for code in sorted(ALL_FINDING_CODES)}


def _normalise_findings(produced: Sequence[str]) -> list[str]:
    """Upper-case, strip and de-duplicate finding codes, order-preserving.

    Normalising is generous-but-honest: a skill that emits ``insufficient_n`` is reporting the right
    finding, and the benchmark measures judgement, not capitalisation.
    """
    out: list[str] = []
    seen: set[str] = set()
    for code in produced:
        key = str(code).strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _weight(code: str) -> float:
    return float(FINDING_WEIGHTS.get(code, 1.0))


def hard_fail(task: BenchmarkTask, produced_findings: Sequence[str]) -> str | None:
    """Return the hard-fail reason when a forbidden finding was produced, else ``None``."""
    produced = set(_normalise_findings(produced_findings))
    forbidden = set(_normalise_findings(task.forbidden_findings))
    hit = sorted(forbidden & produced)
    if not hit:
        return None
    return (
        f"produced forbidden finding(s) {', '.join(hit)}; on this fixture they are false alarms "
        "and would block a correct artifact"
    )


def score_outcome(
    task: BenchmarkTask, produced_findings: Sequence[str]
) -> tuple[float, list[str], list[str]]:
    """``(score, missed, spurious)`` as a weighted F1 over required vs produced findings.

    * a task with no required findings measures false positives only: score 1.0 when nothing
      spurious was produced, 0.0 otherwise;
    * no matched finding at all scores 0.0 (F1 would be undefined, and inventing credit for finding
      nothing is the opposite of what this layer is for).
    """
    required = _normalise_findings(task.required_findings)
    produced = _normalise_findings(produced_findings)
    required_set = set(required)
    produced_set = set(produced)
    matched = [code for code in required if code in produced_set]
    missed = [code for code in required if code not in produced_set]
    spurious = [code for code in produced if code not in required_set]

    if not required:
        return (0.0 if spurious else 1.0), missed, spurious

    weight_matched = sum(_weight(code) for code in matched)
    if weight_matched <= 0.0:
        return 0.0, missed, spurious
    weight_required = sum(_weight(code) for code in required)
    weight_spurious = sum(_weight(code) for code in spurious)
    recall = weight_matched / weight_required
    precision = weight_matched / (weight_matched + weight_spurious)
    f1 = 2 * precision * recall / (precision + recall)
    return round(clampf(f1), 4), missed, spurious


class BenchmarkHarness:
    """Runs a suite of fixed tasks against a skill and persists the attributable result."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ================================================================== reads

    def tasks_for(self, suite: BenchmarkSuite) -> list[BenchmarkTask]:
        """Persisted tasks of ``suite``, in a deterministic order.

        Reads only ``kernel.benchmark_tasks``: what is benchmarked must be a fixture that lives in
        the project, not something the harness invented for this run.
        """
        wanted = BenchmarkSuite(suite) if isinstance(suite, str) else suite
        tasks = [t for t in self.kernel.benchmark_tasks.all() if t.suite is wanted]
        return sorted(tasks, key=lambda t: (t.name, t.benchmark_task_id))

    def latest_run(self, skill_id: str) -> BenchmarkRun | None:
        runs = [r for r in self.kernel.benchmark_runs.all() if r.skill_id == skill_id]
        if not runs:
            return None
        return max(runs, key=lambda r: (r.created_at, r.benchmark_run_id))

    # ================================================================== writes

    def evaluate_outcome(
        self,
        task: BenchmarkTask,
        *,
        produced_findings: Sequence[str],
        output_text: str = "",
        extra: Mapping[str, float] | None = None,
    ) -> TaskOutcome:
        """Score one task outcome. Pure and deterministic: same inputs, same numbers.

        ``output_text`` and ``extra`` are recorded as notes and deliberately never enter the score —
        output length and unvalidated supplementary metrics would both reward verbosity over
        judgement.
        """
        produced = _normalise_findings(produced_findings)
        reason = hard_fail(task, produced)
        score, missed, spurious = score_outcome(task, produced)
        hard_failed = reason is not None
        if hard_failed:
            score = 0.0
        notes: list[str] = []
        if output_text:
            notes.append(
                f"output was {len(output_text)} characters; length is recorded for audit and never scored"
            )
        if extra:
            notes.extend(f"reported metric {key}={value} (not scored)" for key, value in sorted(extra.items()))
        if spurious:
            notes.append(f"spurious findings reduced the score: {', '.join(spurious)}")
        if missed:
            notes.append(f"required findings missed: {', '.join(missed)}")
        return TaskOutcome(
            benchmark_task_id=task.benchmark_task_id,
            passed=(not missed) and not hard_failed,
            score=score,
            produced_findings=produced,
            missed_findings=missed,
            spurious_findings=spurious,
            hard_failed=hard_failed,
            failure_mode=reason,
            notes=notes,
        )

    def run(
        self,
        principal: Principal,
        *,
        skill: SkillCard,
        suite: BenchmarkSuite,
        runner: SkillRunner,
        incumbent: BenchmarkRun | None = None,
        skill_version_id: str | None = None,
        deterministic: bool = True,
    ) -> BenchmarkRun:
        """Run ``suite`` for ``skill`` and persist the :class:`BenchmarkRun`.

        When ``incumbent`` is given, the regression gate is evaluated *task by task* against it — a
        skill that gains on aggregate while breaking a task the incumbent passed is a regression, and
        the run records exactly which tasks broke.
        """
        principal.require(Cap.SKILL_EVALUATE, f"benchmark.run:{skill.skill_id}")
        wanted = BenchmarkSuite(suite) if isinstance(suite, str) else suite

        notes: list[str] = []
        tasks = self.tasks_for(wanted)
        if not tasks:
            # Bootstrap convenience: installing the fixed fixture set is deterministic and
            # idempotent, and a benchmark with no tasks is not a benchmark.
            seed_default_tasks(self.kernel, principal=principal)
            tasks = self.tasks_for(wanted)
            notes.append("default catalogue seeded before this run")
        if not tasks:
            raise SkillError(
                f"no benchmark tasks for suite {wanted.value}; seed the catalogue before benchmarking",
                resource=skill.skill_id,
            )

        if skill_version_id is not None and self.kernel.skill_versions.get(skill_version_id) is None:
            raise SkillError(
                f"skill version {skill_version_id!r} does not exist", resource=skill.skill_id
            )

        outcomes: list[TaskOutcome] = []
        for task in tasks:
            outcome = runner(task)
            if not isinstance(outcome, TaskOutcome):
                raise SkillError(
                    f"runner returned {type(outcome).__name__} for task {task.benchmark_task_id}; "
                    "expected a TaskOutcome",
                    resource=skill.skill_id,
                )
            if outcome.benchmark_task_id != task.benchmark_task_id:
                outcome = outcome.with_updates(
                    benchmark_task_id=task.benchmark_task_id,
                    notes=[
                        *outcome.notes,
                        "runner returned a mismatched task id; re-stamped for attribution",
                    ],
                )
            outcomes.append(outcome)

        report = self._passing_sandbox_report_id(skill.skill_id)
        if report is None:
            notes.append(
                "no passing sandbox report for this skill: a benchmark result alone cannot make a "
                "skill ACTIVATE (the lifecycle requires the sandbox gate too)"
            )

        run = BenchmarkRun(
            skill_id=skill.skill_id,
            skill_version_id=skill_version_id,
            suite=wanted,
            task_outcomes=outcomes,
            sandbox_report_id=report,
            deterministic=deterministic,
            created_by=principal.name,
            notes=notes,
        )
        run.compute_score()
        run.suite_scores = {wanted.value: run.score}

        if incumbent is not None:
            run.incumbent_skill_id = incumbent.skill_id
            passed_task_ids = {
                outcome.benchmark_task_id
                for outcome in incumbent.task_outcomes
                if outcome.passed
            }
            run.evaluate_regression(incumbent.score, passed_task_ids)

        self.kernel.benchmark_runs.save(run)
        self._link_version(run, skill_version_id)

        self.kernel.events.append(
            "benchmark.run.completed",
            actor=principal.name,
            payload={
                "benchmark_run_id": run.benchmark_run_id,
                "skill_id": skill.skill_id,
                "suite": wanted.value,
                "score": run.score,
                "tasks": len(run.task_outcomes),
                "hard_failures": len([o for o in run.task_outcomes if o.hard_failed]),
                "regression_passed": run.regression_passed,
                "regression_delta": run.regression_delta,
                "regression_regressions": run.regression_regressions,
                "incumbent_skill_id": run.incumbent_skill_id,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Benchmark run: {skill.name} / {wanted.value}",
            detail=(
                f"score {run.score} over {len(run.task_outcomes)} task(s)"
                + (
                    f"; regression vs {run.incumbent_skill_id}: "
                    f"{'passed' if run.regression_passed else 'FAILED'}"
                    f" (delta {run.regression_delta})"
                    if run.regression_passed is not None
                    else ""
                )
            ),
            actor=principal.name,
            refs=[skill.skill_id, run.benchmark_run_id],
            state_revision=self.kernel.state.revision(),
        )
        return run

    # ================================================================== internals

    def _link_version(self, run: BenchmarkRun, skill_version_id: str | None) -> None:
        """Record the run on the version it measured, so a result is attributable to a revision."""
        if not skill_version_id:
            return
        version = self.kernel.skill_versions.get(skill_version_id)
        if version is None or run.benchmark_run_id in version.benchmark_run_ids:
            return
        updated = cast(
            SkillVersion,
            version.with_updates(
                benchmark_run_ids=[*version.benchmark_run_ids, run.benchmark_run_id]
            ),
        )
        self.kernel.skill_versions.save(updated)

    def _passing_sandbox_report_id(self, skill_id: str) -> str | None:
        reports = [
            r
            for r in self.kernel.sandbox_reports.all()
            if r.skill_id == skill_id and r.passed
        ]
        if not reports:
            return None
        latest = max(reports, key=lambda r: (r.created_at, r.sandbox_report_id))
        return latest.sandbox_report_id


__all__ = [
    "FINDING_WEIGHTS",
    "BenchmarkHarness",
    "SkillRunner",
    "hard_fail",
    "score_outcome",
]
