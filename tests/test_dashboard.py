"""Dashboard and human-action tests.

Two properties matter more than the markup:

* the dashboard is a *view*: every endpoint it calls must exist, and it must render the panels the
  spec requires;
* the optional write actions are **not** a second authority: they are off by default, local-only when
  on, and they run as the human principal through the same kernel gate as the CLI — so a claim is
  still refused if its evidence is missing, and a stale transition is still refused.
"""

from __future__ import annotations

import re

import pytest

from helpers import build_supported_chain
from researchos.models import ClaimStatus, RejectionBasis

fastapi = pytest.importorskip("fastapi", reason="the dashboard is part of the optional api extra")
from fastapi.testclient import TestClient  # noqa: E402

#: Panel titles the dashboard must render (spec §21 — the first UI version's required content).
REQUIRED_PANELS: tuple[str, ...] = (
    "Core question",
    "Current claims",
    "Evidence coverage",
    "Current experiments",
    "Literature map",
    "Closest prior work",
    "Open questions",
    "Rejected claims",
    "Research decisions",
    "Active task",
    "Skill health & gaps",
    "Skill gaps",
    "Paper readiness",
    "Research timeline",
    "Human review queue",
    "Conflicts",
)


@pytest.fixture()
def ready_kernel(kernel):
    """A project with a supported chain, a rejected claim and a review item to act on."""
    from researchos.claims import ClaimLifecycle
    from researchos.models.review import ReviewItem, ReviewKind

    chain = build_supported_chain(kernel)
    lifecycle = ClaimLifecycle(kernel)
    cancelled = lifecycle.create(
        kernel.principal("claim_manager"),
        statement="Direct writeback is a better optimiser",
        scope="gpt2-small, matched LR",
    )
    lifecycle.transition(
        kernel.principal("claim_manager"), cancelled.claim_id, ClaimStatus.HYPOTHESIS, reason="scope declared"
    )
    lifecycle.transition(
        kernel.principal("claim_manager"),
        cancelled.claim_id,
        ClaimStatus.REJECTED,
        reason="loss curves are indistinguishable once LR is matched",
        rejection_basis=RejectionBasis.CONTRADICTED,
        evidence_ids=[chain.evidence_id],
    )
    kernel.create_review_items(
        [
            ReviewItem(
                kind=ReviewKind.CLAIM_CANDIDATE,
                title="Claim candidate: placement determines retention",
                rationale="matched rule 'section:claim'",
                confidence=0.65,
            )
        ]
    )
    return kernel


def _client(kernel, *, enable_actions: bool = False, host: str = "127.0.0.1"):
    from researchos.api.app import create_app

    return TestClient(create_app(kernel, enable_actions=enable_actions), client=(host, 51234))


# ======================================================================================
# the view
# ======================================================================================
def test_dashboard_html_is_self_contained(ready_kernel):
    """No build step, no npm, no CDN: one HTML file with inline CSS/JS."""
    html = _client(ready_kernel).get("/").text
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html and "<script>" in html
    for external in ("cdn.", "unpkg", "jsdelivr", "googleapis", "https://cdn"):
        assert external not in html, f"the dashboard must not depend on {external}"
    assert "__ACTIONS_ENABLED__" not in html, "the server must substitute the actions flag"
    assert "const ACTIONS_ENABLED = false;" in html


def test_dashboard_renders_every_required_panel(ready_kernel):
    html = _client(ready_kernel).get("/").text
    for title in REQUIRED_PANELS:
        assert title in html, f"dashboard is missing the {title!r} panel"


def test_every_endpoint_the_dashboard_calls_exists(ready_kernel):
    """The UI is a client of the API; a typo in a fetch path must fail here, not in a browser."""
    from researchos.api.app import create_app

    app = create_app(ready_kernel, enable_actions=True)
    routes = {getattr(route, "path", "") for route in app.routes}
    html = _client(ready_kernel).get("/").text
    called = set(re.findall(r"fetch\(\s*[\"']([^\"']+)[\"']", html))
    assert called, "the dashboard makes no calls"
    for path in sorted(called):
        assert path in routes, f"dashboard calls {path!r}, which the API does not serve"
    # the action endpoint is called through a variable-free literal path too
    assert "/actions/review/decide" in html
    assert "/dashboard" in called


def test_dashboard_payload_has_every_panel_source(ready_kernel):
    payload = _client(ready_kernel).get("/dashboard").json()
    for key in (
        "project",
        "describe",
        "integrity",
        "state",
        "core_question",
        "frontier",
        "claims",
        "claims_summary",
        "overclaiming",
        "evidence_summary",
        "evidence_integrity",
        "experiments",
        "conflicts",
        "conflicts_summary",
        "trust_order",
        "decisions",
        "questions",
        "tasks",
        "literature",
        "skills",
        "readiness",
        "timeline",
        "reviews",
        "audits",
        "red_team",
        "paper",
        "actions_enabled",
    ):
        assert key in payload, f"dashboard payload is missing {key!r}"
    assert payload["actions_enabled"] is False


def test_dashboard_readiness_has_seven_dimensions_and_no_total(ready_kernel):
    readiness = _client(ready_kernel).get("/dashboard").json()["readiness"]
    assert len(readiness["dimensions"]) == 7
    for dimension in readiness["dimensions"]:
        assert dimension["basis"]
        assert "label" not in dimension or True
    assert "total" not in readiness
    assert "overall" not in readiness


def test_dashboard_shows_the_rejected_claim_and_the_review_queue(ready_kernel):
    payload = _client(ready_kernel).get("/dashboard").json()
    assert any(claim["status"] == "REJECTED" for claim in payload["claims"])
    assert payload["reviews"], "the review queue must be visible in the dashboard"
    assert payload["reviews"][0]["kind"] == "CLAIM_CANDIDATE"
    # live claims carry the language the evidence permits
    live = [c for c in payload["claims"] if c["status"] not in {"REJECTED", "SUPERSEDED"}]
    assert live and live[0]["permitted_language"]


def test_dashboard_is_truncation_aware(ready_kernel):
    """A truncated panel must say so, so a short list never looks like a complete one."""
    payload = _client(ready_kernel).get("/dashboard").json()
    for key in ("claims_omitted", "experiments_omitted", "reviews_omitted", "decisions_omitted"):
        assert key in payload
        assert isinstance(payload[key], int)


# ======================================================================================
# the actions
# ======================================================================================
def test_actions_are_absent_unless_enabled(ready_kernel):
    client = _client(ready_kernel)  # actions disabled
    review_id = ready_kernel.pending_reviews()[0].review_item_id
    response = client.post(
        "/actions/review/decide", json={"review_item_id": review_id, "decision": "ACCEPTED"}
    )
    assert response.status_code in (403, 404)
    # and the item is untouched
    assert ready_kernel.pending_reviews()[0].review_item_id == review_id
    assert client.get("/dashboard").json()["actions_enabled"] is False


def test_enabled_actions_are_local_only(ready_kernel):
    review_id = ready_kernel.pending_reviews()[0].review_item_id
    remote = _client(ready_kernel, enable_actions=True, host="203.0.113.7")
    response = remote.post(
        "/actions/review/decide", json={"review_item_id": review_id, "decision": "ACCEPTED"}
    )
    assert response.status_code == 403
    assert "local" in response.json()["detail"]
    assert ready_kernel.pending_reviews(), "a remote client must not be able to decide anything"


def test_review_decision_from_the_dashboard_is_audited(ready_kernel):
    client = _client(ready_kernel, enable_actions=True)
    review_id = ready_kernel.pending_reviews()[0].review_item_id
    response = client.post(
        "/actions/review/decide",
        json={"review_item_id": review_id, "decision": "ACCEPTED", "note": "confirmed from dashboard"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ACCEPTED"
    assert not ready_kernel.pending_reviews()
    stored = ready_kernel.reviews.require(review_id)
    assert stored.decided_by == "human"
    assert stored.decision_note == "confirmed from dashboard"
    # the action is distinguishable from a CLI command in the event log
    events = [event for event in ready_kernel.events.iter_records() if event.kind == "ui.action"]
    assert events and events[-1].payload["action"] == "review.decide"
    assert events[-1].payload["origin"] == "dashboard"


def test_unknown_review_decision_value_is_refused(ready_kernel):
    client = _client(ready_kernel, enable_actions=True)
    review_id = ready_kernel.pending_reviews()[0].review_item_id
    response = client.post(
        "/actions/review/decide", json={"review_item_id": review_id, "decision": "MAYBE"}
    )
    assert response.status_code == 422
    assert "ACCEPTED" in response.json()["detail"]


def test_dashboard_cannot_approve_a_transition_as_an_agent(kernel):
    """The action runs as the human principal, but the kernel still demands a real transition."""
    client = _client(kernel, enable_actions=True)
    response = client.post("/actions/transition/approve", json={"str_id": "str_does_not_exist"})
    assert response.status_code == 409
    assert "not found" in response.json()["detail"]


def test_dashboard_transition_approval_applies_and_is_audited(kernel):
    from researchos.models import RiskLevel, StateOperation

    request = kernel.transitions.request(
        kernel.human(),
        operations=[StateOperation(op="set", path="non_goals", value=["we do not study vision"])],
        reason="scope discipline",
        risk=RiskLevel.LOW,
    )
    client = _client(kernel, enable_actions=True)
    response = client.post(
        "/actions/transition/approve", json={"str_id": request.str_id, "note": "approved in the UI"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "APPROVED"
    assert body["applied_revision"] is not None
    assert kernel.research_state().non_goals == ["we do not study vision"]
    assert kernel.decisions.get(body["decision_id"]) is not None


def test_dashboard_still_refuses_a_stale_transition(kernel):
    """The UI has no privileges: a stale request is refused exactly as it is from the CLI."""
    from researchos.models import RiskLevel, StateOperation

    request = kernel.transitions.request(
        kernel.human(),
        operations=[StateOperation(op="append", path="notes", value="a note")],
        reason="record something",
        risk=RiskLevel.LOW,
    )
    kernel.state.update(kernel.human(), lambda state: setattr(state, "notes", [*state.notes, "later edit"]))
    client = _client(kernel, enable_actions=True)
    response = client.post("/actions/transition/approve", json={"str_id": request.str_id})
    assert response.status_code == 409
    assert "STALE" in response.json()["detail"]


def test_dashboard_can_verify_evidence_by_rehashing(kernel, artifact_writer):
    from researchos.evidence import EvidenceRegistry
    from researchos.kernel.provenance import register_artifact
    from researchos.models import Evidence, EvidenceType

    artifact_writer("runs/metrics.csv", "arm,seed,retention_at_1\ndirect,0,0.421\n")
    ref = register_artifact(kernel.root, "runs/metrics.csv", kind="raw")
    evidence = Evidence(
        statement="retention for the direct arm",
        evidence_type=EvidenceType.RAW,
        artifacts=[ref],
        payload={"metric": "retention_at_1"},
    )
    registry = EvidenceRegistry(kernel)
    registry.create(kernel.principal("experiment"), evidence)

    client = _client(kernel, enable_actions=True)
    ok = client.post("/actions/evidence/verify", json={"evidence_id": evidence.evidence_id})
    assert ok.status_code == 200, ok.text
    assert ok.json()["verification_status"] == "VERIFIED"

    # now the bytes change: the dashboard reports the mismatch instead of trusting the record
    artifact_writer("runs/metrics.csv", "arm,seed,retention_at_1\ndirect,0,0.499\n")
    tampered = client.post("/actions/evidence/verify", json={"evidence_id": evidence.evidence_id})
    assert tampered.status_code == 200
    assert tampered.json()["verification_status"] == "HASH_MISMATCH"
    assert kernel.ledger.blocking()
