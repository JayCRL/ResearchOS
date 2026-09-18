"""Research timeline, red-team loop and the skill evolution loop.

These three features close the second half of the spec: the project must be able to *explain itself*
(§20), attack itself (§19), and improve its own capabilities (§15) — all without a language model in the
loop.
"""

from __future__ import annotations

import pytest

from helpers import build_supported_chain

from researchos.agents import RedTeamAgent
from researchos.claims import ClaimLifecycle
from researchos.kernel import ResearchKernel
from researchos.models import (
    BenchmarkSuite,
    ClaimStatus,
    RejectionBasis,
    SkillCard,
    SkillQualityMetrics,
    SkillStatus,
    SkillTrust,
)
from researchos.skills.benchmark import BenchmarkHarness
from researchos.skills.benchmark_tasks import seed_default_tasks
from researchos.skills.evolution import SkillEvolution, deterministic_runner
from researchos.skills.registry import SkillRegistry


# ======================================================================================
# timeline: the project explains its own history
# ======================================================================================
@pytest.fixture()
def busy_kernel(kernel):
    """A project with a real history: a claim that was revised, a route decision, a control added."""
    chain = build_supported_chain(kernel)
    lifecycle = ClaimLifecycle(kernel)

    cancelled = lifecycle.create(
        kernel.principal("claim_manager"),
        statement="Direct writeback is a better optimiser",
        scope="gpt2-small, matched LR",
    )
    lifecycle.transition(
        kernel.principal("claim_manager"), cancelled.claim_id, ClaimStatus.HYPOTHESIS,
        reason="scope declared",
    )
    lifecycle.transition(
        kernel.principal("claim_manager"),
        cancelled.claim_id,
        ClaimStatus.REJECTED,
        reason="loss curves are indistinguishable once LR and schedule are matched",
        rejection_basis=RejectionBasis.CONTRADICTED,
        evidence_ids=[chain.evidence_id],
    )
    kernel.record_decision(
        kernel.human(),
        kind="CONTROL_ADDED",
        summary="add a matched-energy control",
        rationale="otherwise the gap reads as a capacity effect",
        alternatives_considered=["matched-FLOPs control", "argue from the literature"],
        rejected_alternatives=["argue from the literature"],
        affected_claims=[cancelled.claim_id],
        affected_experiments=[chain.experiment_id],
    )
    kernel.record_decision(
        kernel.human(),
        kind="EXPERIMENT_ROUTE",
        summary="drop the 1B run for this quarter",
        rationale="the compute budget is not available",
        consequences=["cross-setting generalisation stays untested"],
        affected_experiments=[chain.experiment_id],
    )
    return kernel, chain, cancelled


def test_timeline_groups_history_into_a_narrative(busy_kernel):
    kernel, _chain, _cancelled = busy_kernel
    phases = dict(kernel.history.narrative())
    assert phases, "a project with work in it has a timeline"
    for phase in ("experiment", "analysis", "revision", "claim"):
        assert phase in phases, f"missing timeline phase {phase!r}"
    summary = kernel.history.summary()
    assert summary["events"] > 0
    assert summary["phases"]


def test_timeline_answers_why_the_claim_was_cancelled(busy_kernel):
    kernel, _chain, cancelled = busy_kernel
    answer = kernel.history.answer("why was the original claim cancelled?")
    assert answer["claims"], "the cancelled claim must appear in the answer"
    entry = next(item for item in answer["claims"] if item["claim_id"] == cancelled.claim_id)
    assert entry["status"] == "REJECTED"
    assert entry["basis"] == "CONTRADICTED"
    assert "indistinguishable" in (entry["reason"] or "")
    assert entry["history"], "the claim's own history is part of the answer"

    history = kernel.history.claim_history(cancelled.claim_id)
    steps = [step["to"] for step in history["steps"]]
    assert steps == ["IDEA", "HYPOTHESIS", "REJECTED"]
    assert all(step["reason"] for step in history["steps"][1:])


def test_timeline_answers_why_the_route_changed_and_why_a_control_was_added(busy_kernel):
    kernel, chain, cancelled = busy_kernel

    route = kernel.history.answer("why did the experiment route change?")
    summaries = " ".join(record["summary"] for record in route["records"])
    assert "1B run" in summaries
    assert any(record["alternatives"] for record in route["records"])

    control = kernel.history.answer("why was this control added?")
    assert control["experiments"]
    assert any(exp["matched_conditions"] for exp in control["experiments"])

    explanation = kernel.history.why(ref=chain.experiment_id)
    assert explanation, "a decision referencing the experiment must be found"
    kinds = {record["kind"] for record in explanation}
    assert "DECISION" in kinds


def test_timeline_explains_what_is_not_in_the_paper(busy_kernel):
    kernel, _chain, cancelled = busy_kernel
    answer = kernel.history.answer("why is this result no longer in the paper?")
    excluded = {item["claim_id"] for item in answer["excluded"]}
    assert cancelled.claim_id in excluded
    assert answer["answer"]


def test_timeline_records_imported_material_as_imported(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    (source / "README.md").write_text("# Legacy\n\nWe observed something odd.\n", encoding="utf-8")
    root = tmp_path / "project"
    root.mkdir()
    kernel = ResearchKernel.create(root, name="legacy")

    from researchos.importer import ImportPipeline

    ImportPipeline(kernel).run(source)
    imported = [event for event in kernel.timeline.all() if event.imported]
    assert imported
    phases = dict(kernel.history.narrative())
    assert "observation" in phases


# ======================================================================================
# red team
# ======================================================================================
def test_red_team_reports_are_persisted_and_never_change_state(busy_kernel):
    kernel, chain, _cancelled = busy_kernel
    revision_before = kernel.state.revision()
    report = RedTeamAgent(kernel).review(claim_ids=[chain.claim_id])

    stored = kernel.red_team_reports.require(report.red_team_report_id)
    assert stored.questions
    assert stored.respects_state is True
    assert kernel.state.revision() == revision_before, "the red team must not touch research state"

    categories = {question.category for question in report.questions}
    assert "falsification" in categories
    assert all(question.what_would_falsify for question in report.questions if question.category == "falsification")

    kinds = [record.kind for record in kernel.events.iter_records()]
    assert "red_team.report" in kinds
    assert any(event.kind.value == "AUDIT" for event in kernel.timeline.all())


def test_red_team_flags_single_seed_and_missing_control(kernel, experiment_factory, register_experiment):
    register_experiment(kernel, experiment_factory(control=False, matched=False, seeds=[0], n=1))
    report = RedTeamAgent(kernel).review()
    categories = {question.category for question in report.questions}
    assert "baseline_matching" in categories
    assert "single_dependency" in categories
    assert report.blocking(), "uncontrolled, single-seed evidence is a high-severity objection"


# ======================================================================================
# skill evolution: sandbox → benchmark → regression → ACTIVE
# ======================================================================================
@pytest.fixture()
def skills_ready(kernel):
    kernel = kernel
    seed_default_tasks(kernel)
    registry = SkillRegistry(kernel)
    parent_a = registry.register(
        kernel.principal("skill_discovery"),
        SkillCard(name="literature-review", description="systematic literature review procedure",
                  quality_metrics=SkillQualityMetrics(accuracy=0.7)),
    )
    parent_b = registry.register(
        kernel.principal("skill_discovery"),
        SkillCard(name="citation-verification", description="check that a citation supports a claim",
                  quality_metrics=SkillQualityMetrics(accuracy=0.75)),
    )
    return kernel, registry, parent_a, parent_b


def _perfect_runner(kernel: ResearchKernel, suite: BenchmarkSuite = BenchmarkSuite.LITERATURE):
    """A runner that returns exactly the required findings (never the forbidden ones)."""
    harness = BenchmarkHarness(kernel)
    mapping = {
        task.benchmark_task_id: list(task.required_findings)
        for task in harness.tasks_for(suite)
    }
    return deterministic_runner(mapping), mapping


def test_skill_evolution_activates_a_candidate_that_passes_every_gate(skills_ready):
    kernel, registry, parent_a, parent_b = skills_ready
    runner, mapping = _perfect_runner(kernel)
    assert mapping, "the default catalogue must provide literature benchmark tasks"

    evolution = SkillEvolution(kernel)
    card, result = evolution.synthesise_and_evaluate(
        kernel.principal("skill_synthesizer"),
        parents=[parent_a.skill_id, parent_b.skill_id],
        rule="every literature claim must carry a page or section locator",
        name="locator-checking-review",
        description="literature review that refuses unsourced claims",
        suite=BenchmarkSuite.LITERATURE,
        runner=runner,
    )

    assert result.activated is True, result.summary()
    assert result.sandbox is not None and result.sandbox.passed
    assert result.stages[0].startswith("sandbox:pass")
    assert card.status is SkillStatus.ACTIVE
    assert card.trust in (SkillTrust.BENCHMARKED, SkillTrust.TRUSTED)
    # a generated skill is created as an untrusted candidate and must earn the rest
    assert card.parent_skill_ids == [parent_a.skill_id, parent_b.skill_id]
    assert card.last_benchmark_run_id

    history = evolution.history(card.skill_id)
    assert history["benchmark_runs"]
    assert history["trend"][-1] > 0


def test_skill_evolution_rejects_a_regression_even_when_the_score_rises(skills_ready):
    kernel, _registry, parent_a, parent_b = skills_ready
    evolution = SkillEvolution(kernel)
    runner, mapping = _perfect_runner(kernel)

    incumbent_card, first = evolution.synthesise_and_evaluate(
        kernel.principal("skill_synthesizer"),
        parents=[parent_a.skill_id, parent_b.skill_id],
        rule="baseline rule",
        name="incumbent-skill",
        description="the incumbent",
        suite=BenchmarkSuite.LITERATURE,
        runner=runner,
    )
    assert first.activated

    # the upgrade passes most tasks but breaks one the incumbent passed
    broken = dict(mapping)
    broken_task = sorted(broken)[0]
    broken[broken_task] = []
    regressing_runner = deterministic_runner(broken)

    upgraded, second = evolution.synthesise_and_evaluate(
        kernel.principal("skill_synthesizer"),
        parents=[parent_a.skill_id, parent_b.skill_id],
        rule="baseline rule plus a new shortcut",
        name="upgraded-skill",
        description="the upgrade that breaks a task",
        suite=BenchmarkSuite.LITERATURE,
        runner=regressing_runner,
        incumbent=incumbent_card,
    )

    assert second.activated is False
    assert "regression" in (second.rejected_reason or "")
    assert upgraded.status is not SkillStatus.ACTIVE
    assert second.benchmark is not None and second.benchmark.regression_regressions


def test_skill_evolution_rejects_a_candidate_that_fails_the_sandbox(skills_ready):
    kernel, registry, parent_a, parent_b = skills_ready
    evolution = SkillEvolution(kernel)
    runner, _mapping = _perfect_runner(kernel)

    candidate = registry.register(
        kernel.principal("skill_discovery"),
        SkillCard(name="destructive-skill", description="deletes raw evidence"),
    )
    result = evolution.evaluate_candidate(
        kernel.principal("skill_evaluator"),
        skill=candidate,
        suite=BenchmarkSuite.LITERATURE,
        runner=runner,
        content="import shutil\nshutil.rmtree('evidence/raw')\n",
    )
    assert result.activated is False
    assert result.stages == ["sandbox:fail"]
    assert "sandbox failed" in (result.rejected_reason or "")
    assert result.sandbox is not None and result.sandbox.protected_paths_touched
    # and it never even reached the benchmark
    assert result.benchmark is None
    assert kernel.skills.require(candidate.skill_id).status is SkillStatus.DISCOVERED


def test_skill_evolution_records_every_stage_in_the_event_log(skills_ready):
    kernel, _registry, parent_a, parent_b = skills_ready
    evolution = SkillEvolution(kernel)
    runner, _mapping = _perfect_runner(kernel)
    card, _result = evolution.synthesise_and_evaluate(
        kernel.principal("skill_synthesizer"),
        parents=[parent_a.skill_id, parent_b.skill_id],
        rule="rule",
        name="logged-skill",
        description="",
        suite=BenchmarkSuite.LITERATURE,
        runner=runner,
    )
    events = [record for record in kernel.events.iter_records() if record.kind == "skill.evolution"]
    assert events
    assert any(event.payload.get("skill_id") == card.skill_id for event in events)
    assert any(event.payload.get("activated") for event in events)
