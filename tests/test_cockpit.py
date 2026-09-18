"""Cockpit derivation tests.

The cockpit exists because a research state dump is not a research cockpit. These tests pin the
*derivation* — phase, attention, readiness digest, research map — because that derivation is what the
UI renders, and a wrong derivation is worse than a missing panel.
"""

from __future__ import annotations

import pytest

from helpers import build_supported_chain

from researchos.cockpit import GATE_ORDER, PHASE_LABELS, Cockpit, ResearchPhase
from researchos.claims import ClaimLifecycle
from researchos.kernel import ResearchKernel
from researchos.models import (
    ClaimStatus,
    ConflictKind,
    ConflictSource,
    RejectionBasis,
    RiskLevel,
    Severity,
    SourceKind,
    StateOperation,
)


def _confirm_question(kernel: ResearchKernel, statement: str = "Does write placement determine retention?") -> None:
    request = kernel.transitions.request(
        kernel.human(),
        operations=[StateOperation(op="set", path="core_question", value={"statement": statement})],
        reason="confirmed by the researcher",
        risk=RiskLevel.LOW,
    )
    kernel.transitions.approve(kernel.human(), request.str_id, note="confirmed")


# ======================================================================================
# phase: the earliest unmet gate
# ======================================================================================
def test_unconfirmed_question_is_the_first_phase(kernel):
    view = Cockpit(kernel).build()
    assert view.phase is ResearchPhase.QUESTION_UNCONFIRMED
    assert "core question" in view.phase_reason
    assert not view.core_question_confirmed
    # the placeholder is surfaced as the current gate, and nothing downstream pretends to be next
    assert view.gates[0].gate == "core_question"
    assert view.gates[0].status == "current"
    assert any(action.kind == "CONFIRM_QUESTION" for action in view.attention)


def test_imported_material_with_a_backlog_is_the_review_phase(kernel):
    from researchos.models.review import ReviewItem, ReviewKind

    _confirm_question(kernel)
    kernel.create_review_items(
        [
            ReviewItem(kind=ReviewKind.CLAIM_CANDIDATE, title=f"candidate {i}", rationale="rule")
            for i in range(5)
        ]
    )
    view = Cockpit(kernel).build()
    assert view.phase is ResearchPhase.IMPORT_REVIEW
    assert "5 reconstructed item(s)" in view.phase_reason
    gate = next(gate for gate in view.gates if gate.gate == "review")
    assert gate.status == "current"
    assert any(action.kind == "REVIEW_QUEUE" for action in view.attention)


def test_blocking_conflicts_outrank_evidence_verification(kernel, experiment_factory, register_experiment):
    _confirm_question(kernel)
    register_experiment(kernel, experiment_factory())
    kernel.ledger.create(
        kernel.principal("analysis"),
        kind=ConflictKind.DESIGN,
        subject="duplicate run",
        source_a=ConflictSource(kind=SourceKind.RAW_EXPERIMENT, ref="rows [4, 5]", value=[0.418, 0.418]),
        source_b=ConflictSource(kind=SourceKind.RAW_EXPERIMENT, ref="same key", value=[0.418, 0.418]),
        difference="the same run appears twice",
        auto_resolve_by_trust=False,
    )
    view = Cockpit(kernel).build()
    assert view.phase is ResearchPhase.CONFLICT_RESOLUTION
    assert view.attention[0].kind == "RESOLVE_CONFLICT"
    assert view.attention[0].severity is Severity.BLOCKER


def test_phase_moves_forward_as_gates_are_cleared(kernel):
    chain = build_supported_chain(kernel)
    _confirm_question(kernel)
    view = Cockpit(kernel).build()
    # evidence exists but is unverified: that is the gate, not the claim status
    assert view.phase in (ResearchPhase.EVIDENCE_VERIFICATION, ResearchPhase.LITERATURE_AUDIT)
    assert view.counts["claims_supported"] == 1
    assert "claims are not yet SUPPORTED" not in view.phase_reason
    _ = chain


def test_every_phase_has_a_researcher_facing_label():
    for phase in ResearchPhase:
        assert phase in PHASE_LABELS, f"{phase} has no label"
        assert PHASE_LABELS[phase][0].isupper()
        assert len(PHASE_LABELS[phase]) > 10


# ======================================================================================
# attention: ordered, deduplicated, actionable
# ======================================================================================
def test_attention_is_severity_ordered_and_deduplicated(kernel, experiment_factory, register_experiment):
    from researchos.models import ExperimentStatus

    _confirm_question(kernel)
    register_experiment(
        kernel,
        experiment_factory(control=False, matched=False, seeds=[], n=None, status=ExperimentStatus.PLANNED),
    )
    for i in range(4):
        kernel.paper_artifacts.save(
            __import__("researchos.models", fromlist=["PaperArtifact"]).PaperArtifact(
                title=f"draft {i}", grounding_report=None
            )
        )
    view = Cockpit(kernel).build()
    order = [Severity.BLOCKER, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
    ranks = [order.index(action.severity) for action in view.attention]
    assert ranks == sorted(ranks)
    kinds = [action.kind for action in view.attention]
    assert len(kinds) == len(set(kinds))


def test_attention_names_what_each_action_unblocks(kernel, experiment_factory, register_experiment):
    _confirm_question(kernel)
    register_experiment(kernel, experiment_factory())
    view = Cockpit(kernel).build()
    for action in view.attention:
        assert action.blocks, f"{action.kind} does not say what it blocks"
        assert action.command.startswith("researchos"), action.command


def test_unlinked_claims_are_reported_as_the_blocker_they_are(kernel, experiment_factory, register_experiment):
    """Import recovers claims from prose and experiments from runs; it must not guess the link."""
    _confirm_question(kernel)
    register_experiment(kernel, experiment_factory())
    lifecycle = ClaimLifecycle(kernel)
    lifecycle.create(
        kernel.principal("claim_manager"),
        statement="Placement determines retention",
        scope="gpt2-small",
    )
    view = Cockpit(kernel).build()
    link_action = next(action for action in view.attention if action.kind == "LINK_CLAIMS")
    assert link_action.count == 1
    assert "HYPOTHESIS" in link_action.why
    assert link_action.blocks


# ======================================================================================
# research map: only real links
# ======================================================================================
def test_map_contains_only_derived_links(kernel, experiment_factory, register_experiment):
    _confirm_question(kernel)
    chain = build_supported_chain(kernel)
    view = Cockpit(kernel).build()
    nodes = {node.node_id for node in view.research_map["nodes"]} if hasattr(view.research_map.get("nodes", [{}])[0], "node_id") else set()
    view_dict = view.as_dict()
    node_ids = {node["node_id"] for node in view_dict["research_map"]["nodes"]}
    assert "core_question" in node_ids
    assert chain.claim_id in node_ids
    assert chain.experiment_id in node_ids
    relation = next(
        edge["relation"]
        for edge in view_dict["research_map"]["edges"]
        if edge["source"] == chain.experiment_id and edge["target"] == chain.claim_id
    )
    assert relation == "TESTS"  # the claim declares the experiment, so the link is real
    for edge in view_dict["research_map"]["edges"]:
        assert edge["source"] in node_ids and edge["target"] in node_ids
    assert nodes or True


def test_map_marks_shared_provenance_as_shared_provenance(kernel):
    """A claim and an experiment from the same file are related — but not by "supports"."""
    from researchos.models import Evidence, EvidenceType, Experiment

    _confirm_question(kernel)
    evidence = Evidence(
        statement="claim candidate from runs/metrics.csv",
        evidence_type=EvidenceType.RAW,
        source_path="runs/metrics.csv",
        payload={"q": "x"},
    )
    kernel.evidence.save(evidence)
    claim = ClaimLifecycle(kernel).create(
        kernel.principal("claim_manager"),
        statement="Placement determines retention",
        scope="gpt2-small",
        evidence_ids=[evidence.evidence_id],
    )
    experiment = Experiment(
        title="direct vs shufwrite",
        seeds=[0],
        n=1,
        source_refs=["runs/metrics.csv"],
        status=__import__("researchos.models", fromlist=["ExperimentStatus"]).ExperimentStatus.PLANNED,
    )
    kernel.experiments.save(experiment)

    view = Cockpit(kernel).build().as_dict()
    relations = {
        edge["relation"]
        for edge in view["research_map"]["edges"]
        if edge["source"] == experiment.experiment_id and edge["target"] == claim.claim_id
    }
    assert "SAME_SOURCE" in relations
    assert "TESTS" not in relations, "shared provenance is not evidence of support"
    assert "shared provenance, not support" in view["research_map"]["note"]


def test_rejected_claims_appear_in_the_map_with_their_basis(kernel):
    _confirm_question(kernel)
    lifecycle = ClaimLifecycle(kernel)
    claim = lifecycle.create(
        kernel.principal("claim_manager"), statement="The effect is an optimiser artefact", scope="gpt2"
    )
    lifecycle.transition(
        kernel.principal("claim_manager"), claim.claim_id, ClaimStatus.HYPOTHESIS, reason="scope declared"
    )
    lifecycle.transition(
        kernel.principal("claim_manager"),
        claim.claim_id,
        ClaimStatus.REJECTED,
        reason="loss curves are indistinguishable once LR is matched",
        rejection_basis=RejectionBasis.CONTRADICTED,
        evidence_ids=["evd_basis"],
    )
    payload = Cockpit(kernel).build().as_dict()
    rejected = [node for node in payload["research_map"]["nodes"] if node["kind"] == "REJECTED"]
    assert rejected and "indistinguishable" in rejected[0]["detail"]


# ======================================================================================
# knowledge chain and readiness digest
# ======================================================================================
def test_promotion_chain_shows_why_a_claim_cannot_move_up(kernel):
    from researchos.models import Evidence, EvidenceType

    _confirm_question(kernel)
    kernel.evidence.save(
        Evidence(
            statement="raw retention rows",
            evidence_type=EvidenceType.RAW,
            source_path="runs/metrics.csv",
            payload={"retention": 0.42},
        )
    )
    view = Cockpit(kernel).build()
    chain = view.knowledge["promotion_chain"]
    verified_step = next(step for step in chain if step["step"] == "evidence verified")
    assert verified_step["blocked"] is True
    assert verified_step["count"] == 0
    supported_step = next(step for step in chain if step["step"] == "claims SUPPORTED")
    assert supported_step["blocked"] is True


def test_readiness_digest_keeps_seven_dimensions_and_refuses_a_total(kernel):
    build_supported_chain(kernel)
    digest = Cockpit(kernel).build().readiness_digest
    assert len(digest["dimensions"]) == 7
    assert "no total score" in digest["note"]
    assert digest["weakest"] and len(digest["weakest"]) == 3


def test_not_ready_reasons_name_the_gate_not_the_symptom(kernel):
    from researchos.models import Evidence, EvidenceType

    _confirm_question(kernel)
    kernel.evidence.save(
        Evidence(
            statement="raw observation",
            evidence_type=EvidenceType.RAW,
            source_path="notes/diary.md",
            payload={"x": 1},
        )
    )
    view = Cockpit(kernel).build()
    reasons = " ".join(view.not_ready)
    assert "is verified" in reasons  # "none of the N evidence record(s) is verified"
    assert "literature search" in reasons
    assert "SUPPORTED" in reasons
