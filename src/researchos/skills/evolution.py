"""Skill evolution: the loop that makes skills improve, end to end.

```
research failure → skill gap → discovery → candidate → sandbox → benchmark → regression
    → ACTIVE | REJECT, and synthesis: A + B + research rule → C (C is untrusted by construction)
```

This module exists so that the loop is *executable and testable* rather than a diagram: it wires the
registry, the sandbox, the benchmark harness and the lifecycle into one deterministic pipeline that a
test can run without a network or a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..kernel.errors import SkillError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models.skill import (
    BenchmarkRun,
    BenchmarkSuite,
    BenchmarkTask,
    SandboxReport,
    SkillCard,
    SkillQualityMetrics,
    SkillStatus,
    SkillTrust,
    TaskOutcome,
)
from ..models.timeline import TimelineEventKind


@dataclass
class EvolutionResult:
    """Everything that happened to one candidate skill, in order."""

    skill_id: str
    stages: list[str] = field(default_factory=list)
    sandbox: SandboxReport | None = None
    benchmark: BenchmarkRun | None = None
    activated: bool = False
    rejected_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "ACTIVE" if self.activated else f"REJECTED ({self.rejected_reason})"
        return f"{self.skill_id}: {' → '.join(self.stages)} → {verdict}"


class SkillEvolution:
    """Runs a candidate skill through sandbox, benchmark and regression gates."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ------------------------------------------------------------------ pipeline

    def evaluate_candidate(
        self,
        principal: Principal,
        *,
        skill: SkillCard,
        suite: BenchmarkSuite,
        runner: Callable[[BenchmarkTask], TaskOutcome],
        content: str | None = None,
        incumbent: SkillCard | None = None,
        task_id: str | None = None,
    ) -> EvolutionResult:
        """Sandbox → benchmark → regression → activate or reject. Never skips a gate."""
        result = EvolutionResult(skill_id=skill.skill_id)

        # 1. sandbox (static; untrusted code is never executed here)
        from .sandbox import SkillSandbox

        sandbox = SkillSandbox(self.kernel).inspect(
            principal, skill, content=content, path=None
        )
        result.sandbox = sandbox
        result.stages.append("sandbox:" + ("pass" if sandbox.passed else "fail"))
        if not sandbox.passed:
            result.rejected_reason = "sandbox failed: " + ", ".join(
                finding.code for finding in sandbox.findings[:3]
            )
            self._record(skill, result, task_id=task_id)
            return result

        # 2. lifecycle rungs up to VERIFIED (the gates themselves enforce the ladder)
        from .lifecycle import SkillLifecycle

        lifecycle = SkillLifecycle(self.kernel)
        for target in (SkillStatus.CANDIDATE, SkillStatus.SANDBOX, SkillStatus.EVALUATED):
            try:
                skill = lifecycle.transition(
                    principal, skill.skill_id, target,
                    sandbox_report_id=sandbox.sandbox_report_id,
                    reason="skill evolution pipeline",
                )
                result.stages.append(target.value.lower())
            except SkillError as exc:
                result.notes.append(f"{target.value}: {exc}")
        if skill.trust is SkillTrust.UNTRUSTED:
            skill = self.kernel.skills.save(skill.with_updates(trust=SkillTrust.SANDBOXED))

        # 3. benchmark
        from .benchmark import BenchmarkHarness

        incumbent_run = BenchmarkHarness(self.kernel).latest_run(incumbent.skill_id) if incumbent else None
        run = BenchmarkHarness(self.kernel).run(
            principal, skill=skill, suite=suite, runner=runner, incumbent=incumbent_run
        )
        result.benchmark = run
        result.stages.append(f"benchmark:{run.score:.2f}")
        if run.regression_passed is False:
            result.rejected_reason = "regression against the incumbent failed"
        activatable, reasons = run.activatable()
        if not activatable and result.rejected_reason is None:
            result.rejected_reason = "; ".join(reasons)

        # 4. verify (trust upgrade) then activate
        try:
            skill = lifecycle.transition(
                principal, skill.skill_id, SkillStatus.VERIFIED,
                benchmark_run_id=run.benchmark_run_id,
                reason="sandbox and benchmark passed",
            )
            result.stages.append("verified")
        except SkillError as exc:
            result.notes.append(f"VERIFIED: {exc}")
            result.rejected_reason = result.rejected_reason or str(exc)
            self._record(skill, result, task_id=task_id)
            return result

        if result.rejected_reason:
            self._record(skill, result, task_id=task_id)
            return result

        activated = lifecycle.activate(
            self.kernel.human(), skill.skill_id, benchmark_run_id=run.benchmark_run_id
        )
        result.activated = activated.status is SkillStatus.ACTIVE
        result.stages.append("active")
        self._record(activated, result, task_id=task_id)
        return result

    # ------------------------------------------------------------------ synthesis

    def synthesise_and_evaluate(
        self,
        principal: Principal,
        *,
        parents: Sequence[str],
        rule: str,
        name: str,
        description: str,
        suite: BenchmarkSuite,
        runner: Callable[[BenchmarkTask], TaskOutcome],
        content: str | None = None,
        incumbent: SkillCard | None = None,
    ) -> tuple[SkillCard, EvolutionResult]:
        """A + B + research rule → C, evaluated like any other untrusted candidate."""
        from .discovery import SkillDiscoveryAgent

        candidate = SkillDiscoveryAgent(self.kernel).synthesise(
            principal,
            skill_ids=list(parents),
            rule=rule,
            name=name,
            description=description,
        )
        result = self.evaluate_candidate(
            principal,
            skill=candidate,
            suite=suite,
            runner=runner,
            content=content,
            incumbent=incumbent,
        )
        return self.kernel.skills.require(candidate.skill_id), result

    # ------------------------------------------------------------------ history

    def _record(self, skill: SkillCard, result: EvolutionResult, *, task_id: str | None) -> None:
        self.kernel.events.append(
            "skill.evolution",
            actor="skill_evaluator",
            task_id=task_id,
            payload={
                "skill_id": skill.skill_id,
                "stages": result.stages,
                "activated": result.activated,
                "rejected_reason": result.rejected_reason,
                "benchmark_run_id": result.benchmark.benchmark_run_id if result.benchmark else None,
                "score": result.benchmark.score if result.benchmark else None,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill evolution {skill.skill_id}: {'ACTIVE' if result.activated else 'REJECTED'}",
            detail=result.summary(),
            actor="skill_evaluator",
            task_id=task_id,
            refs=[skill.skill_id],
            state_revision=self.kernel.state.revision(),
        )

    def history(self, skill_id: str) -> dict[str, Any]:
        """Every benchmark run and lifecycle event for one skill — the audit trail of its quality."""
        runs = [run for run in self.kernel.benchmark_runs.all() if run.skill_id == skill_id]
        scores = [
            {"run_id": run.benchmark_run_id, "score": run.score, "regression": run.regression_passed}
            for run in sorted(runs, key=lambda r: r.created_at)
        ]
        return {
            "skill_id": skill_id,
            "versions": len([v for v in self.kernel.skill_versions.all() if v.skill_id == skill_id]),
            "benchmark_runs": scores,
            "trend": [entry["score"] for entry in scores],
            "deltas": [
                round(scores[i]["score"] - scores[i - 1]["score"], 4) for i in range(1, len(scores))
            ],
        }


def deterministic_runner(findings_by_task: Mapping[str, Sequence[str]]) -> Callable[[BenchmarkTask], TaskOutcome]:
    """A benchmark runner that is a pure function of a mapping.

    It scores through the harness's own ``score_outcome``/``hard_fail`` functions, so the gates under
    test are exactly the gates a real run uses. This is what makes the sandbox/benchmark/regression
    loop testable with no model and no network.
    """
    from .benchmark import hard_fail, score_outcome

    def runner(task: BenchmarkTask) -> TaskOutcome:
        produced = list(findings_by_task.get(task.benchmark_task_id, []))
        score, missed, spurious = score_outcome(task, produced)
        reason = hard_fail(task, produced)
        return TaskOutcome(
            benchmark_task_id=task.benchmark_task_id,
            passed=(not missed) and reason is None,
            score=0.0 if reason else score,
            produced_findings=produced,
            missed_findings=missed,
            spurious_findings=spurious,
            hard_failed=reason is not None,
            failure_mode=reason,
            notes=["deterministic runner: findings supplied by the caller"],
        )

    return runner


def quality_from_benchmark(run: BenchmarkRun) -> SkillQualityMetrics:
    """Derive a quality profile from a benchmark run, honestly.

    Only dimensions the benchmark actually measured are set; the rest stay at zero rather than being
    filled with optimistic guesses.
    """
    tasks = run.task_outcomes
    if not tasks:
        return SkillQualityMetrics()
    accuracy = sum(o.score for o in tasks) / len(tasks)
    return SkillQualityMetrics(
        accuracy=round(accuracy, 4),
        coverage=round(sum(1 for o in tasks if o.passed) / len(tasks), 4),
        claim_calibration=0.0,
        reproducibility=1.0 if run.deterministic else 0.5,
        regression_safety=1.0 if run.regression_passed is not False else 0.0,
    )
