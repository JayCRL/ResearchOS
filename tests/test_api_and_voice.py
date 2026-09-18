"""Read-only API and researcher-voice tests.

The API test asserts the *absence* of write routes as firmly as the presence of read routes: a web
endpoint that mutates research state would reintroduce the ambient authority the kernel exists to remove.
"""

from __future__ import annotations

import pytest

from helpers import build_supported_chain

from researchos.models import NoteKind, SourceRef
from researchos.models.timeline import AuthorNote
from researchos.paper import ResearcherVoice, voice_context

fastapi = pytest.importorskip("fastapi", reason="the API surface is an optional extra")
from fastapi.testclient import TestClient  # noqa: E402


# ======================================================================================
# read-only API
# ======================================================================================
@pytest.fixture()
def client(kernel):
    from researchos.api.app import create_app

    build_supported_chain(kernel)
    return TestClient(create_app(kernel))


@pytest.mark.parametrize(
    "route",
    [
        "/health",
        "/project",
        "/state",
        "/claims",
        "/claims/summary",
        "/evidence",
        "/evidence/summary",
        "/experiments",
        "/analyses",
        "/conflicts",
        "/audits",
        "/reviews",
        "/literature/papers",
        "/literature/coverage",
        "/literature/novelty",
        "/literature/prior-art",
        "/timeline",
        "/timeline/summary",
        "/readiness",
        "/agents",
        "/skills",
        "/capabilities",
    ],
)
def test_read_routes_answer(client, route):
    response = client.get(route)
    assert response.status_code == 200, (route, response.text[:200])


def test_api_exposes_no_write_routes(client):
    """Deliberate: mutations need a principal and a capability check, which HTTP does not provide here."""
    for route in ("/claims", "/state", "/evidence", "/timeline", "/skills"):
        for method in ("post", "put", "patch"):
            response = getattr(client, method)(route, json={})
            assert response.status_code in (404, 405), (method, route, response.status_code)
        response = client.delete(route)
        assert response.status_code in (404, 405), ("delete", route, response.status_code)


def test_api_reports_unknown_ids_as_404(client):
    assert client.get("/claims/clm_does_not_exist").status_code == 404
    assert client.get("/evidence/evd_does_not_exist").status_code == 404


def test_api_claim_view_includes_obligations(client):
    claims = client.get("/claims").json()
    assert claims
    claim_id = claims[0]["claim_id"]
    detail = client.get(f"/claims/{claim_id}").json()
    assert detail["status"]
    assert "next_transitions" in detail
    assert "permitted_language" in detail


def test_api_capability_table_names_human_only_capabilities(client):
    payload = client.get("/capabilities").json()
    assert payload["read_only_http"] is True
    assert "claim.approve" in payload["human_only"]
    assert "state.core.write" in payload["human_only"]


def test_api_trace_route_returns_the_provenance_chain(client):
    claims = client.get("/claims").json()
    trace = client.get(f"/evidence/trace/{claims[0]['claim_id']}").json()
    assert trace["claim_id"] == claims[0]["claim_id"]
    assert trace["chains"]
    assert trace["complete"] is True


# ======================================================================================
# researcher voice
# ======================================================================================
@pytest.fixture()
def voiced_kernel(kernel):
    """A project whose history contains an anomaly, a control addition and a revision."""
    chain = build_supported_chain(kernel)
    for text, kind in (
        ("Anomaly: the gap vanished when I changed the data order, so both arms were not comparable.", NoteKind.ANOMALY),
        ("I expected capacity to explain the effect, but the matched-parameter run disagrees.", NoteKind.WHY_THIS_EXPERIMENT),
        ("Hypothesis revision: the relevant variable looks like alignment, not capacity.", NoteKind.CLAIM_REVISION),
    ):
        kernel.notes.save(AuthorNote(kind=kind, text=text, source_refs=[SourceRef(path="notes/diary.md")]))
    kernel.record_decision(
        kernel.human(),
        kind="CONTROL_ADDED",
        summary="add a matched-energy control",
        rationale="otherwise the gap reads as a capacity effect",
        affected_experiments=[chain.experiment_id],
    )
    return kernel, chain


def test_voice_reconstructs_the_real_arc(voiced_kernel):
    kernel, _chain = voiced_kernel
    voice = ResearcherVoice(kernel)
    phases = voice.arc_phases()
    assert "anomaly" in phases, "an anomaly note exists and must appear in the arc"
    assert "control" in phases
    assert "revision" in phases
    # the arc is ordered as research proceeds, not alphabetically
    assert phases.index("anomaly") < phases.index("control")
    assert voice.arc()[0].phase == "observation"


def test_voice_detects_a_faked_confirmation_arc(voiced_kernel):
    kernel, _chain = voiced_kernel
    voice = ResearcherVoice(kernel)
    problems = voice.distortion(
        "As we hypothesised, the control confirmed the effect. We report a clean confirmation."
    )
    assert problems
    assert any("anomaly" in problem or "revision" in problem for problem in problems)

    clean = voice.distortion(
        "We initially observed something odd, revised the hypothesis after the anomaly, and report "
        "the surviving claim."
    )
    assert not any("revised" in problem for problem in clean)


def test_voice_narrative_sentences_are_grounded_and_numeral_free(voiced_kernel):
    kernel, _chain = voiced_kernel
    voice = ResearcherVoice(kernel)
    sentences = voice.narrative_sentences()
    assert sentences
    for text, _section, note_ids, decision_ids, claim_ids in sentences:
        assert note_ids or decision_ids or claim_ids, (
            "a narrative sentence must point at the record it came from: " + text[:80]
        )
        assert not any(character.isdigit() for character in text), text


def test_voice_profile_is_taken_from_the_researchers_own_notes(voiced_kernel):
    kernel, _chain = voiced_kernel
    profile = ResearcherVoice(kernel).profile()
    assert profile.sample_size >= 3
    assert profile.mean_sentence_words > 0
    assert profile.first_person_ratio > 0, "research notes are written in the first person"
    assert profile.examples


def test_voice_context_is_structured_for_a_writer(voiced_kernel):
    kernel, _chain = voiced_kernel
    context = voice_context(kernel)
    assert context["arc"]
    assert context["profile"]["sample_size"] >= 1
    assert "source_refs" in context


def test_compiler_uses_the_real_arc_in_the_paper(voiced_kernel):
    from researchos.paper import PaperCompiler

    kernel, _chain = voiced_kernel
    artifact = PaperCompiler(kernel).compile(kernel.principal("paper_writer"))
    text = artifact.render().lower()
    assert "unexpected observation changed the course" in text or "revised our working hypothesis" in text
    # history that happened is described; the paper does not pretend the arc was clean
    assert artifact.grounding_report is not None
