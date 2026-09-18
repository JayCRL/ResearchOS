"""Kernel tests: state, transitions, permissions, event log, conflicts, tasks, layout.

The invariant suite covers the guarantees; this suite covers the mechanics that make them usable
and the failure messages that make them debuggable.
"""

from __future__ import annotations

import pytest

from researchos.claims import ClaimLifecycle
from researchos.kernel import (
    ENTITY_DIRS,
    Cap,
    ChainVerification,
    EntityStore,
    EventLog,
    ProjectPaths,
    ResearchKernel,
    ResourceClass,
    assert_mutable,
)
from researchos.kernel.errors import (
    PermissionDenied,
    ProjectExists,
    ProjectNotFound,
    ResearchOSError,
    SilentStateChangeError,
    StateRevisionConflict,
    TaskBoundaryViolation,
    TransitionError,
)
from researchos.kernel.permissions import AGENT_CAPABILITIES, HUMAN_ONLY_CAPABILITIES
from researchos.models import (
    ChangeClass,
    ConflictKind,
    ConflictResolution,
    ConflictSource,
    Evidence,
    EvidenceType,
    OpenQuestion,
    RiskLevel,
    SourceKind,
    StateOperation,
    TaskPriority,
    TaskPurpose,
)


# ======================================================================================
# project lifecycle and layout
# ======================================================================================
def test_create_open_and_refuse_double_init(project_root):
    kernel = ResearchKernel.create(project_root, name="DLA", domain="AI/ML")
    assert ProjectPaths.for_root(project_root).exists()
    for relative in ("state", "state/tasks", "claims", "experiments", "evidence/raw",
                     "literature/papers", "literature/prior_art", "skills/registry",
                     "audits", "conflicts", "review", "paper/artifacts", "cache"):
        assert (kernel.ros_dir / relative).is_dir(), relative

    with pytest.raises(ProjectExists):
        ResearchKernel.create(project_root)

    reopened = ResearchKernel.open(project_root)
    assert reopened.project().name == "DLA"
    assert reopened.state.revision() == kernel.state.revision()


def test_discovery_walks_upward_and_fails_cleanly(tmp_path, tmp_path_factory):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    ResearchKernel.create(tmp_path, name="outer")
    found = ProjectPaths.discover(nested)
    assert found.root == tmp_path.resolve()

    # a directory tree with no project anywhere above it must fail loudly, not silently
    empty = tmp_path_factory.mktemp("no-project-here")
    with pytest.raises(ProjectNotFound):
        ProjectPaths.discover(empty)
    assert ProjectPaths.discover_or_none(empty) is None


def test_entity_files_are_yaml_and_human_readable(kernel):
    lifecycle = ClaimLifecycle(kernel)
    claim = lifecycle.create(
        kernel.principal("claim_manager"), statement="a claim", scope="scope"
    )
    path = kernel.claims.path_for(claim.claim_id)
    assert path.is_file() and path.suffix == ".yaml"
    text = path.read_text(encoding="utf-8")
    assert "statement: a claim" in text
    assert "claim_id:" in text


def test_store_refuses_deletes_and_unknown_keys(kernel):
    from researchos.models import SkillCard

    card = kernel.skills.save(SkillCard(name="x"))
    with pytest.raises(Exception):
        kernel.skills.delete(card.skill_id, reason="cleanup")
    # a tampered file with an unknown key must be rejected, not silently accepted
    path = kernel.skills.path_for(card.skill_id)
    path.write_text(path.read_text(encoding="utf-8") + "\nunexpected_key: 1\n", encoding="utf-8")
    with pytest.raises(ResearchOSError) as excinfo:
        kernel.skills.require(card.skill_id)
    assert "schema" in str(excinfo.value)


# ======================================================================================
# state and revisions
# ======================================================================================
def test_every_write_bumps_the_revision_and_is_logged(kernel):
    before = kernel.state.revision()
    kernel.state.update(kernel.human(), lambda state: setattr(state, "notes", [*state.notes, "a note"]))
    after = kernel.state.revision()
    assert after == before + 1
    assert kernel.events.head_seq() >= 1
    kinds = [record.kind for record in kernel.events.iter_records()]
    assert "state.updated" in kinds


def test_guarded_paths_cannot_move_without_an_approved_transition(kernel):
    with pytest.raises(SilentStateChangeError) as excinfo:
        kernel.state.update(
            kernel.human(),
            lambda state: setattr(
                state,
                "priorities",
                [{"rank": 1, "statement": "we only study interpretability now"}],
            ),
        )
    assert "priorities" in str(excinfo.value)
    assert "State Transition Request" in str(excinfo.value)


def test_transition_preview_shows_before_and_after(kernel):
    request = kernel.transitions.request(
        kernel.human(),
        operations=[StateOperation(op="set", path="non_goals", value=["we do not study vision"])],
        reason="scope discipline",
        risk=RiskLevel.LOW,
    )
    rows = kernel.transitions.diff_preview(request.str_id)
    assert rows[0]["path"] == "non_goals"
    assert rows[0]["guarded"] is True
    assert rows[0]["before"] == []
    assert rows[0]["after"] == ["we do not study vision"]


def test_stale_transition_is_refused_and_marked(kernel):
    request = kernel.transitions.request(
        kernel.human(),
        operations=[StateOperation(op="append", path="notes", value="a note")],
        reason="record something",
        risk=RiskLevel.LOW,
    )
    kernel.state.update(kernel.human(), lambda state: setattr(state, "notes", [*state.notes, "intervening edit"]))
    with pytest.raises(TransitionError) as excinfo:
        kernel.transitions.approve(kernel.human(), request.str_id)
    assert "STALE" in str(excinfo.value)
    assert kernel.transitions.get(request.str_id).status.value == "STALE"


def test_transition_requires_evidence_above_low_risk(kernel):
    with pytest.raises(TransitionError) as excinfo:
        kernel.transitions.request(
            kernel.human(),
            operations=[StateOperation(op="set", path="core_question", value={"statement": "x"})],
            reason="just feels right",
            risk=RiskLevel.HIGH,
        )
    assert "evidence_ids" in str(excinfo.value)


def test_decision_records_capture_alternatives_and_consequences(kernel):
    decision = kernel.record_decision(
        kernel.human(),
        kind="EXPERIMENT_ROUTE",
        summary="add a matched-energy control",
        rationale="reviewers will read the gap as a capacity effect otherwise",
        alternatives_considered=["a matched-FLOPs control", "no control, argue from the literature"],
        rejected_alternatives=["no control, argue from the literature"],
        consequences=["all baseline runs must be repeated"],
    )
    assert not decision.model_dump()["rejected_alternatives"] == []
    stored = kernel.decisions.require(decision.decision_id)
    assert stored.rationale.startswith("reviewers will")


def test_integrity_reports_referenced_but_missing_records(kernel):
    """The check exists for hand-edited state, so the test hand-edits the file."""
    import yaml

    from researchos.kernel.store import YamlIO

    path = kernel.paths.research_state_file
    data = YamlIO.read(path)
    data["core_claims"] = ["clm_does_not_exist"]
    YamlIO.write_atomic(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    kernel.state.load(refresh=True)

    integrity = kernel.integrity()
    assert integrity["ok"] is False
    assert any("does not exist" in issue for issue in integrity["issues"])


# ======================================================================================
# event log
# ======================================================================================
def test_event_log_is_a_chain(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    first = log.append("a", actor="t", payload={"x": 1})
    second = log.append("b", actor="t", payload={"x": 2})
    assert first.prev_hash == "0" * 64
    assert second.prev_hash == first.event_hash
    assert isinstance(log.verify(), ChainVerification)
    assert log.verify().ok

    # reordering events also breaks the chain (not just editing them)
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "events.jsonl").write_text("\n".join([lines[1], lines[0]]) + "\n", encoding="utf-8")
    verification = EventLog(tmp_path / "events.jsonl").verify()
    assert verification.ok is False
    assert verification.problems


def test_event_payloads_are_json_canonical(kernel):
    record = kernel.events.append("t", payload={"b": 1, "a": 2})
    assert record.event_hash
    assert '"a":2' in (kernel.ros_dir / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1]


# ======================================================================================
# permissions
# ======================================================================================
def test_human_only_capabilities_are_not_granted_to_agents(kernel):
    for agent_type in AGENT_CAPABILITIES:
        principal = kernel.principal(agent_type)
        for capability in HUMAN_ONLY_CAPABILITIES:
            assert not principal.has(capability), f"{agent_type} holds human-only {capability.value}"


def test_paper_writer_cannot_propose_claims(kernel):
    writer = kernel.principal("paper_writer")
    assert not writer.has(Cap.CLAIM_PROPOSE)
    with pytest.raises(PermissionDenied):
        writer.require(Cap.CLAIM_PROPOSE, "claim.create")


def test_task_allowed_actions_bound_the_principal(kernel):
    task = kernel.task_manager.create(
        kernel.principal("planner"),
        objective="only compute statistics",
        purpose=TaskPurpose.ANALYSE,
        priority=TaskPriority.SECONDARY,
        allowed_actions=[Cap.ANALYSIS_COMPUTE.value],
    )
    kernel.policy.require(
        kernel.principal("analysis"),
        Cap.ANALYSIS_COMPUTE,
        "analysis",
        task_allowed=task.allowed_actions,
        task_forbidden=task.forbidden_actions,
    )
    with pytest.raises(TaskBoundaryViolation):
        kernel.policy.require(
            kernel.principal("analysis"),
            Cap.EVIDENCE_VERIFY,
            "evidence",
            task_allowed=task.allowed_actions,
            task_forbidden=task.forbidden_actions,
        )


def test_unknown_agent_type_is_refused(kernel):
    with pytest.raises(PermissionDenied):
        kernel.agent("wizard")


# ======================================================================================
# tasks
# ======================================================================================
def test_task_findings_are_isolated_from_research_state(kernel):
    task = kernel.task_manager.create(
        kernel.principal("planner"),
        objective="explain the seed-3 anomaly",
        purpose=TaskPurpose.EXPLAIN,
        priority=TaskPriority.EXPLORATORY,
        stop_condition="explained",
    )
    kernel.task_manager.start(kernel.principal("planner"), task.task_id)
    kernel.task_manager.add_finding(
        kernel.principal("planner"),
        task.task_id,
        statement="the anomaly is a data-order artefact",
        suggests_core_change=True,
    )
    state = kernel.research_state()
    assert state.core_question is None
    assert state.notes == []
    stored = kernel.tasks_store.require(task.task_id)
    assert len(stored.findings) == 1

    kernel.task_manager.close(kernel.principal("planner"), task.task_id, summary="explained")
    state = kernel.research_state()
    assert state.active_task is None
    assert task.task_id in state.recent_tasks


def test_context_scope_is_narrow_and_explicit(kernel):
    task = kernel.task_manager.create(
        kernel.principal("planner"),
        objective="check the metric definition",
        purpose=TaskPurpose.AUDIT,
        priority=TaskPriority.SECONDARY,
        allowed_actions=["audit.create"],
        forbidden_actions=["claim.approve"],
        context_refs=["exp_1"],
    )
    scope = kernel.task_manager.context_scope(task.task_id)
    assert scope["forbidden_actions"] == ["claim.approve"]
    assert scope["context_refs"] == ["exp_1"]
    assert scope["may_request_core_change"] is False  # SECONDARY cannot


def test_closed_task_refuses_new_findings(kernel):
    task = kernel.task_manager.create(
        kernel.principal("planner"), objective="x", purpose=TaskPurpose.RUN, priority=TaskPriority.PRIMARY,
        stop_condition="done",
    )
    kernel.task_manager.close(kernel.principal("planner"), task.task_id, summary="done")
    with pytest.raises(Exception):
        kernel.task_manager.add_finding(kernel.principal("planner"), task.task_id, statement="late finding")


# ======================================================================================
# conflicts
# ======================================================================================
def test_trust_order_decides_the_default_resolution(kernel):
    conflict = kernel.ledger.create(
        kernel.principal("importer"),
        kind=ConflictKind.VALUE,
        subject="retention_at_1 for the direct arm",
        source_a=ConflictSource(kind=SourceKind.RAW_EXPERIMENT, ref="runs/metrics.csv:3", value=0.418),
        source_b=ConflictSource(kind=SourceKind.AUDIT, ref="audit/stats.md:14", value=0.402),
        difference="the audit reports a stale value",
    )
    assert conflict.resolution is ConflictResolution.A_WINS
    assert conflict.winning_ref == "runs/metrics.csv:3"
    assert conflict.superseded_ref == "audit/stats.md:14"
    assert conflict.blocks_claim_promotion is False
    assert "trust" in (conflict.resolution_reason or "")
    # both sources survive
    assert conflict.source_a.value == 0.418 and conflict.source_b.value == 0.402


def test_equal_trust_leaves_the_conflict_unresolved_and_blocking(kernel):
    conflict = kernel.ledger.create(
        kernel.principal("importer"),
        kind=ConflictKind.VALUE,
        subject="two notes disagree",
        source_a=ConflictSource(kind=SourceKind.AUTHOR_NOTE, ref="notes/a.md", value=1),
        source_b=ConflictSource(kind=SourceKind.AUTHOR_NOTE, ref="notes/b.md", value=2),
        difference="two human notes disagree",
    )
    assert conflict.resolution is ConflictResolution.UNRESOLVED
    assert conflict.blocks_claim_promotion is True
    assert kernel.ledger.blocking()


def test_unresolved_conflict_blocks_claim_promotion(kernel, experiment_factory, register_experiment):
    from helpers import build_supported_chain

    chain = build_supported_chain(kernel)
    kernel.ledger.create(
        kernel.principal("analysis"),
        kind=ConflictKind.VALUE,
        subject="retention_at_1",
        source_a=ConflictSource(kind=SourceKind.AUTHOR_NOTE, ref="notes/a.md", value=1),
        source_b=ConflictSource(kind=SourceKind.AUTHOR_NOTE, ref="notes/b.md", value=2),
        difference="notes disagree about the metric",
        claim_ids=[chain.claim_id],
    )
    lifecycle = ClaimLifecycle(kernel)
    claim = kernel.claims.require(chain.claim_id)
    blocked = kernel.claims.save(claim.with_updates(status=__import__("researchos.models", fromlist=["ClaimStatus"]).ClaimStatus.TESTED))
    assert blocked.status.value == "TESTED"
    with pytest.raises(Exception) as excinfo:
        lifecycle.transition(
            kernel.human(), chain.claim_id, __import__("researchos.models", fromlist=["ClaimStatus"]).ClaimStatus.SUPPORTED,
            reason="promote",
        )
    assert "conflict" in str(excinfo.value)


# ======================================================================================
# provenance
# ======================================================================================
def test_register_artifact_hashes_real_bytes(kernel, artifact_writer):
    from researchos.kernel.provenance import register_artifact, verify_artifact
    from researchos.models import VerificationStatus

    artifact_writer("runs/x.json", '{"a": 1}')
    ref = register_artifact(kernel.root, "runs/x.json")
    assert ref.bytes and ref.sha256
    assert verify_artifact(kernel.root, ref)[0] is VerificationStatus.VERIFIED
    artifact_writer("runs/x.json", '{"a": 2}')
    assert verify_artifact(kernel.root, ref)[0] is VerificationStatus.HASH_MISMATCH


def test_register_artifact_refuses_missing_files(kernel):
    from researchos.kernel.errors import VerificationError
    from researchos.kernel.provenance import register_artifact

    with pytest.raises(VerificationError):
        register_artifact(kernel.root, "does/not/exist.csv")


def test_entity_dirs_cover_every_registered_store(kernel):
    for kind in kernel.store_kinds():
        assert kind in ENTITY_DIRS, kind
    for kind in kernel.store_kinds():
        assert isinstance(kernel.store(kind), EntityStore)


def test_open_question_requires_a_source_for_verified_novelty(kernel):
    with pytest.raises(Exception) as excinfo:
        OpenQuestion(statement="nobody has studied X", kind=__import__("researchos.models", fromlist=["GapKind"]).GapKind.VERIFIED_NOVELTY)
    assert "literature_ids" in str(excinfo.value)
