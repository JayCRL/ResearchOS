"""The ten ResearchOS invariants.

Each test here corresponds to a specific way research software (and AI research assistance) usually
goes wrong. These are not stylistic preferences: if one of these fails, the system's central claim —
that research direction and scientific facts are structured, traceable state rather than chat — is
false.

| # | Invariant | Test |
|---|---|---|
| 1 | Agents cannot modify raw evidence | ``test_agent_cannot_modify_raw_evidence`` |
| 2 | A local task cannot silently change the core question | ``test_debug_task_cannot_promote_to_core`` |
| 3 | The writer cannot create new numbers | ``test_writer_cannot_invent_numbers`` |
| 4 | Every claim has evidence | ``test_claim_requires_evidence`` |
| 5 | Every literature claim has a source | ``test_literature_claim_requires_source`` |
| 6 | UNKNOWN never becomes FALSE | ``test_unknown_is_not_false`` |
| 7 | External skills cannot mutate research state | ``test_external_skill_cannot_mutate_state`` |
| 8 | A skill upgrade cannot break existing benchmarks | ``test_skill_upgrade_requires_regression_pass`` |
| 9 | Every experiment artifact traces to a source | ``test_artifact_traces_to_source`` |
| 10 | Rejected claims never vanish | ``test_rejected_claim_is_retained`` |
"""

from __future__ import annotations

import pytest

from helpers import build_supported_chain

from researchos.claims import ClaimLifecycle
from researchos.evidence import EvidenceRegistry
from researchos.kernel import (
    ImmutableResourceError,
    PermissionDenied,
    ProjectPaths,
    SilentStateChangeError,
)
from researchos.kernel.errors import (
    ClaimTransitionError,
    EvidenceError,
    LiteratureError,
    SkillError,
    StateRevisionConflict,
)
from researchos.kernel.permissions import Cap, PolicyEngine, ResourceClass, assert_mutable
from researchos.literature import PriorArtMatrixBuilder, render_markdown
from researchos.models import (
    BenchmarkRun,
    BenchmarkSuite,
    Claim,
    ClaimStatus,
    CoreQuestion,
    Evidence,
    EvidenceLevel,
    EvidenceType,
    FulltextStatus,
    LiteratureClaim,
    LiteratureClaimKind,
    LiteraturePaper,
    MatrixColumn,
    RejectionBasis,
    RiskLevel,
    SkillCapability,
    SkillCard,
    SkillQualityMetrics,
    SkillStatus,
    SkillTrust,
    StateOperation,
    TaskOutcome,
    TaskPriority,
    TaskPurpose,
    Tri,
    VerificationStatus,
)
from researchos.models.paper import GroundedSentence, NumberRef, PaperSection
from researchos.paper import PaperCompiler, verify_numbers, verify_paper
from researchos.paper.grounding import GroundingViolationCode
from researchos.skills.lifecycle import SkillLifecycle
from researchos.skills.registry import SkillRegistry

pytestmark = pytest.mark.invariant


# ======================================================================================
# 1. Agents cannot modify raw evidence
# ======================================================================================
def test_agent_cannot_modify_raw_evidence(kernel):
    registry = EvidenceRegistry(kernel)
    raw = Evidence(
        statement="direct beats shufwrite by 0.31 on the matched-energy control",
        evidence_type=EvidenceType.RAW,
        payload={"delta": 0.31},
        source_experiment="exp_x",
    )
    registry.create(kernel.principal("experiment"), raw)
    assert raw.immutable is True

    # the analysis agent holds evidence.create and evidence.verify, but not the right to write a raw
    # observation: nobody self-certifies raw data
    with pytest.raises(PermissionDenied):
        registry.create(
            kernel.principal("analysis"),
            Evidence(statement="x", evidence_type=EvidenceType.RAW, payload={"a": 1}),
        )

    # and the registry refuses the edit regardless of who asks
    edited = raw.with_updates(payload={"delta": 0.99})
    with pytest.raises(EvidenceError) as excinfo:
        registry.update(kernel.human(), edited, reason="fixing a typo in the number")
    assert "immutable" in str(excinfo.value)
    assert "supersedes" in str(excinfo.value)

    # the stored record still says 0.31
    assert kernel.evidence.require(raw.evidence_id).payload["delta"] == 0.31


def test_resource_guards_are_principal_independent():
    for resource in ResourceClass:
        with pytest.raises(ImmutableResourceError):
            assert_mutable(resource, "delete", PolicyEngine().get("human"))


# ======================================================================================
# 2. A local task cannot silently change the core question
# ======================================================================================
def test_debug_task_cannot_promote_to_core(kernel):
    task = kernel.task_manager.create(
        kernel.principal("planner"),
        objective="metric M drops on seed 3; find out why",
        purpose=TaskPurpose.DEBUG,
        priority=TaskPriority.EXPLORATORY,
        stop_condition="metric explained, or escalate to a human",
    )
    assert task.may_request_core_change() is False
    original = kernel.research_state().core_question

    # 2a. the guard refuses a direct write, whoever the principal is
    with pytest.raises(SilentStateChangeError) as excinfo:
        kernel.state.update(
            kernel.human(),
            lambda state: setattr(
                state, "core_question", CoreQuestion(statement="the metric is the project")
            ),
        )
    assert "State Transition Request" in str(excinfo.value)
    assert "core_question" in str(excinfo.value)

    # 2b. the debugging task cannot even file the request
    with pytest.raises(PermissionDenied):
        kernel.transitions.request(
            kernel.principal("planner"),
            operations=[StateOperation(op="set", path="core_question", value={"statement": "new"})],
            reason="a debugging finding suggested a new direction",
            task_id=task.task_id,
            evidence_ids=["evd_1"],
        )

    # 2c. the finding is recorded on the task, where it belongs
    finding = kernel.task_manager.add_finding(
        kernel.principal("planner"),
        task.task_id,
        statement="metric M is sensitive to data order, which may matter for the main claim",
        suggests_core_change=True,
    )
    assert finding.suggests_core_change is True
    assert kernel.research_state().core_question == original

    # 2d. a PRIMARY design task may request the change, and only a human may approve it
    design_task = kernel.task_manager.create(
        kernel.principal("planner"),
        objective="establish the placement effect under a matched control",
        purpose=TaskPurpose.DESIGN,
        priority=TaskPriority.PRIMARY,
        stop_condition="effect established or refuted",
    )
    evidence = Evidence(
        statement="data order confound found and fixed",
        evidence_type=EvidenceType.RAW,
        payload={"note": "both arms now see identical batches"},
    )
    EvidenceRegistry(kernel).create(kernel.principal("experiment"), evidence)
    request = kernel.transitions.request(
        kernel.principal("planner"),
        operations=[
            StateOperation(
                op="set",
                path="core_question",
                value={"statement": "Does write placement determine fast-weight retention?"},
            )
        ],
        reason="matched-control evidence makes this the project's real question",
        evidence_ids=[evidence.evidence_id],
        task_id=design_task.task_id,
        risk=RiskLevel.MEDIUM,
    )
    with pytest.raises(PermissionDenied):
        kernel.transitions.approve(kernel.principal("claim_manager"), request.str_id)
    with pytest.raises(PermissionDenied):
        kernel.transitions.approve(kernel.principal("planner"), request.str_id)

    approved, decision = kernel.transitions.approve(kernel.human(), request.str_id, note="approved")
    assert approved.applied_revision is not None
    assert kernel.research_state().core_question.statement.startswith("Does write placement")
    assert decision.rationale
    assert any(event.kind == "str.approved" for event in kernel.events.iter_records())


# ======================================================================================
# 3. The writer cannot create new numbers
# ======================================================================================
def test_writer_cannot_invent_numbers(kernel):
    invented = GroundedSentence(
        text="Direct writeback improves retention by 47.3% on the matched-energy control.",
        section=PaperSection.RESULTS,
        claim_ids=["clm_context"],
        is_load_bearing=True,
    )
    violations = verify_numbers([invented], numbers=[])
    codes = {violation.code for violation in violations}
    assert GroundingViolationCode.UNGROUNDED_NUMBER in codes
    violation = next(v for v in violations if v.code is GroundingViolationCode.UNGROUNDED_NUMBER)
    assert violation.blocks_compilation is True
    assert "47.3" in (violation.found or "")

    # the same sentence passes only when a NumberRef from an analysis artifact backs the numeral
    number = NumberRef(
        value=47.3,
        unit="%",
        analysis_id="ana_1",
        result_id="res_1",
        result_field="mean",
        context="retention delta",
    )
    grounded = invented.with_updates(numbers=[number])
    remaining = [
        v
        for v in verify_numbers([grounded], numbers=[number])
        if v.code is GroundingViolationCode.UNGROUNDED_NUMBER
    ]
    assert remaining == []


def test_compiler_grounding_catches_a_tampered_paper(kernel):
    build_supported_chain(kernel)
    artifact = PaperCompiler(kernel).compile(kernel.principal("paper_writer"))
    assert artifact.grounding_report is not None
    assert artifact.grounding_report.passed, artifact.grounding_report.summary()

    # someone "improves" the compiled prose after the fact
    tampered = artifact.model_copy(deep=True)
    tampered.sections[0].sentences[0].text += " This raises retention by 31.7% overall."
    report = verify_paper(tampered, kernel=kernel)
    assert not report.passed
    assert any(v.code is GroundingViolationCode.UNGROUNDED_NUMBER for v in report.violations)
    assert "31.7" in " ".join(v.found or "" for v in report.violations)


# ======================================================================================
# 4. Every claim has evidence
# ======================================================================================
def test_claim_requires_evidence(kernel):
    lifecycle = ClaimLifecycle(kernel)
    claim = lifecycle.create(
        kernel.principal("claim_manager"),
        statement="Fast-weight retention is driven by write position",
        scope="gpt2-small, 124M, wikitext-103",
    )
    assert claim.status is ClaimStatus.IDEA

    # schema level: a claim cannot even be constructed as SUPPORTED without evidence
    with pytest.raises(Exception) as excinfo:
        Claim(statement="x", status=ClaimStatus.SUPPORTED, scope="s")
    assert "evidence" in str(excinfo.value)

    # lifecycle level: obligations are checked with a message that says what is missing
    lifecycle.transition(
        kernel.principal("claim_manager"),
        claim.claim_id,
        ClaimStatus.HYPOTHESIS,
        reason="scope declared",
    )
    with pytest.raises(ClaimTransitionError) as excinfo:
        lifecycle.transition(
            kernel.principal("claim_manager"), claim.claim_id, ClaimStatus.TESTED, reason="we ran it"
        )
    assert "experiment" in str(excinfo.value)

    # the classic overclaim jump is refused explicitly
    with pytest.raises(ClaimTransitionError) as excinfo:
        lifecycle.transition(kernel.human(), claim.claim_id, ClaimStatus.SUPPORTED, reason="looks right")
    assert "is not a legal claim transition" in str(excinfo.value)
    assert "SUPPORTED" in str(excinfo.value)


# ======================================================================================
# 5. Every literature claim has a source
# ======================================================================================
def test_literature_claim_requires_source(kernel):
    paper = LiteraturePaper(
        title="Some related work", year=2021, fulltext_status=FulltextStatus.ABSTRACT_LEVEL_ONLY
    )
    kernel.papers.save(paper)

    with pytest.raises(Exception) as excinfo:
        LiteratureClaim(
            paper_id=paper.paper_id, statement="they use a fast state", kind=LiteratureClaimKind.RESULT
        )
    assert "no source" in str(excinfo.value)

    # a mechanism-level claim about someone else's paper requires the full text
    with pytest.raises(Exception) as excinfo:
        LiteratureClaim(
            paper_id=paper.paper_id,
            statement="the method writes back at the originating position",
            kind=LiteratureClaimKind.MECHANISM,
            page="4",
        )
    assert "fulltext_verified" in str(excinfo.value)

    ok = LiteratureClaim(
        paper_id=paper.paper_id,
        statement="the paper reports an associative memory mechanism",
        kind=LiteratureClaimKind.RESULT,
        section="Abstract",
    )
    kernel.literature_claims.save(ok)
    assert ok.locator() == "Abstract"


# ======================================================================================
# 6. UNKNOWN never becomes FALSE
# ======================================================================================
def test_unknown_is_not_false(kernel):
    builder = PriorArtMatrixBuilder(kernel)
    columns = (
        MatrixColumn(key="explicit_selector", label="Explicit selector"),
        MatrixColumn(key="replay", label="Replay"),
    )
    matrix = builder.build(
        kernel.human(),
        title="fast-weight writeback",
        columns=columns,
        rows=[("lit_a", "Paper A"), ("current_work", "This work")],
    )
    assert matrix.counts()["UNKNOWN"] == 4
    assert "?" in render_markdown(matrix)

    with pytest.raises(LiteratureError) as excinfo:
        builder.upgrade_unknown(
            matrix,
            "lit_a",
            "explicit_selector",
            value=Tri.FALSE,
            note="not mentioned in the abstract",
        )
    assert "explicit_absence_evidence" in str(excinfo.value)

    updated = builder.upgrade_unknown(
        matrix,
        "lit_a",
        "explicit_selector",
        value=Tri.FALSE,
        note="Section 3 states that no selection mechanism is used",
        paper_id="lit_a",
        section="3",
        explicit_absence_evidence=True,
    )
    cell = updated.cell("lit_a", "explicit_selector")
    assert cell.value is Tri.FALSE
    assert cell.section == "3"
    assert updated.cell("lit_a", "replay").value is Tri.UNKNOWN
    assert updated.counts()["UNKNOWN"] == 3


# ======================================================================================
# 7. External skills cannot mutate research state
# ======================================================================================
def test_external_skill_cannot_mutate_state(kernel):
    skill = kernel.policy.external_skill("some-github-skill")
    assert skill.is_external
    assert skill.has(Cap.PROPOSE_ANALYSIS)

    for capability in (
        Cap.STATE_WRITE_TASK,
        Cap.STATE_TRANSITION_REQUEST,
        Cap.STATE_CORE_WRITE,
        Cap.EVIDENCE_RAW_WRITE,
        Cap.EVIDENCE_CREATE,
        Cap.CLAIM_PROPOSE,
        Cap.CLAIM_APPROVE,
        Cap.LITERATURE_WRITE,
        Cap.SKILL_ACTIVATE,
    ):
        with pytest.raises(PermissionDenied):
            skill.require(capability)

    # the policy engine refuses to grant an external skill real authority even if asked
    with pytest.raises(PermissionDenied):
        kernel.policy.external_skill("bad-skill", extra=[Cap.STATE_CORE_WRITE])

    # and a skill card cannot even *declare* a write to a protected resource
    with pytest.raises(Exception) as excinfo:
        SkillCard(
            name="evil",
            capabilities=[
                SkillCapability(name="rewrite", writes=["evidence/raw"], is_advisory_only=False)
            ],
        )
    assert "protected resource" in str(excinfo.value)


# ======================================================================================
# 8. A skill upgrade cannot break existing benchmarks
# ======================================================================================
def test_skill_upgrade_requires_regression_pass(kernel):
    registry = SkillRegistry(kernel)
    incumbent = registry.register(
        kernel.principal("skill_discovery"),
        SkillCard(
            name="literature-review",
            quality_metrics=SkillQualityMetrics(accuracy=0.8, coverage=0.7),
        ),
    )
    # walk the skill ladder to VERIFIED so the test isolates the *upgrade* gate
    incumbent = kernel.skills.save(
        incumbent.with_updates(status=SkillStatus.VERIFIED, trust=SkillTrust.BENCHMARKED)
    )
    from researchos.models import SandboxReport

    kernel.sandbox_reports.save(SandboxReport(skill_id=incumbent.skill_id, passed=True))

    # a candidate version that gains on aggregate but breaks a previously passing task
    run = BenchmarkRun(
        skill_id=incumbent.skill_id,
        suite=BenchmarkSuite.LITERATURE,
        task_outcomes=[
            TaskOutcome(benchmark_task_id="t1", passed=False, score=0.0),
            TaskOutcome(benchmark_task_id="t2", passed=True, score=1.0),
            TaskOutcome(benchmark_task_id="t3", passed=True, score=1.0),
        ],
    )
    run.compute_score()
    passed = run.evaluate_regression(incumbent_score=0.60, incumbent_passed={"t1", "t2", "t3"})
    assert passed is False
    assert run.score > 0.60  # the aggregate improved…
    assert "t1" in run.regression_regressions  # …while a real task regressed
    kernel.benchmark_runs.save(run)

    with pytest.raises(SkillError) as excinfo:
        SkillLifecycle(kernel).activate(
            kernel.human(), incumbent.skill_id, benchmark_run_id=run.benchmark_run_id
        )
    assert "regression" in str(excinfo.value) or "activation" in str(excinfo.value)

    good = BenchmarkRun(
        skill_id=incumbent.skill_id,
        suite=BenchmarkSuite.LITERATURE,
        task_outcomes=[TaskOutcome(benchmark_task_id="t1", passed=True, score=0.9)],
    )
    good.compute_score()
    assert good.evaluate_regression(0.5, {"t1"}) is True
    kernel.benchmark_runs.save(good)
    activated = SkillLifecycle(kernel).activate(
        kernel.human(), incumbent.skill_id, benchmark_run_id=good.benchmark_run_id
    )
    assert activated.status is SkillStatus.ACTIVE


# ======================================================================================
# 9. Every experiment artifact traces to a source
# ======================================================================================
def test_artifact_traces_to_source(kernel, experiment_factory, register_experiment, artifact_writer):
    from researchos.kernel.provenance import register_artifact, verify_artifact

    artifact_writer("runs/metrics.csv", "arm,seed,retention_at_1\ndirect,0,0.421\n")
    experiment = register_experiment(kernel, experiment_factory())
    ref = register_artifact(kernel.root, "runs/metrics.csv", kind="raw")
    kernel.experiments.save(experiment.with_updates(raw_artifacts=[ref]))

    status, _detail = verify_artifact(kernel.root, ref)
    assert status is VerificationStatus.VERIFIED

    # …until the bytes change, which must be *detected*, not ignored
    artifact_writer("runs/metrics.csv", "arm,seed,retention_at_1\ndirect,0,0.499\n")
    status, detail = verify_artifact(kernel.root, ref)
    assert status is VerificationStatus.HASH_MISMATCH
    assert "changed" in (detail or "")

    raw = Evidence(
        statement="retention for the direct arm",
        evidence_type=EvidenceType.RAW,
        artifacts=[ref],
        payload={"metric": "retention_at_1"},
        evidence_level=EvidenceLevel.L1_OBSERVATION,
    )
    registry = EvidenceRegistry(kernel)
    registry.create(kernel.principal("experiment"), raw)
    verified = registry.verify(kernel.principal("analysis"), raw.evidence_id, method="re-hash")
    assert verified.verification_status is VerificationStatus.HASH_MISMATCH
    assert kernel.ledger.blocking(), "a changed artifact must produce a blocking conflict"

    assert all(a.path for a in kernel.experiments.require(experiment.experiment_id).raw_artifacts)


# ======================================================================================
# 10. Rejected claims never vanish
# ======================================================================================
def test_rejected_claim_is_retained(kernel):
    lifecycle = ClaimLifecycle(kernel)
    claim = lifecycle.create(
        kernel.principal("claim_manager"),
        statement="Direct writeback is a better optimiser than shuffled writeback",
        scope="gpt2-small, matched LR and schedule",
    )
    lifecycle.transition(
        kernel.principal("claim_manager"),
        claim.claim_id,
        ClaimStatus.HYPOTHESIS,
        reason="scope declared",
    )
    rejected = lifecycle.transition(
        kernel.principal("claim_manager"),
        claim.claim_id,
        ClaimStatus.REJECTED,
        reason="loss curves are indistinguishable once LR and schedule are matched",
        rejection_basis=RejectionBasis.CONTRADICTED,
        evidence_ids=["evd_rejection_basis"],
    )
    assert rejected.tombstone is True
    assert rejected.rejection_reason

    assert kernel.claims.get(claim.claim_id) is not None
    assert claim.claim_id in kernel.research_state().rejected_claims
    with pytest.raises(ImmutableResourceError):
        kernel.claims.delete(claim.claim_id, reason="cleanup")

    # a rejected claim cannot be resurrected — only superseded by a successor
    with pytest.raises(ClaimTransitionError):
        lifecycle.transition(
            kernel.principal("claim_manager"), claim.claim_id, ClaimStatus.TESTED, reason="let us retry"
        )

    # and a paper may not cite it
    from researchos.paper.grounding import verify_claims

    sentence = GroundedSentence(
        text="We show that direct writeback is a better optimiser.",
        section=PaperSection.RESULTS,
        claim_ids=[claim.claim_id],
    )
    violations = verify_claims([sentence], {claim.claim_id: rejected}, {})
    codes = {violation.code for violation in violations}
    assert GroundingViolationCode.REJECTED_CLAIM_CITED in codes
    assert all(v.blocks_compilation for v in violations)


# ======================================================================================
# Extra: the event log is tamper-evident
# ======================================================================================
def test_event_log_detects_tampering(kernel, project_root):
    kernel.events.append("test.event", actor="tester", payload={"n": 1})
    kernel.events.append("test.event", actor="tester", payload={"n": 2})
    assert kernel.events.verify().ok

    log_path = ProjectPaths.for_root(project_root).event_log
    lines = log_path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace('"n":2', '"n":3')
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verification = kernel.events.verify()
    assert verification.ok is False
    assert any("tampered" in problem for problem in verification.problems)


# ======================================================================================
# Extra: revisions cannot be silently overwritten
# ======================================================================================
def test_concurrent_state_write_is_refused(kernel):
    revision = kernel.state.revision()
    kernel.state.update(
        kernel.human(), lambda state: setattr(state, "notes", [*state.notes, "first writer"])
    )
    with pytest.raises(StateRevisionConflict):
        kernel.state.update(
            kernel.human(),
            lambda state: setattr(state, "notes", [*state.notes, "second writer"]),
            expected_revision=revision,
        )
