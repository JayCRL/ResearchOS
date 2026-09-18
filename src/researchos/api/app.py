"""Read-only HTTP surface over the kernel.

The API is deliberately **read-only**. Every mutation in ResearchOS carries an explicit principal and a
capability check, and a web endpoint that writes would be the easiest place in the system for ambient
authority to reappear. Writes go through the CLI or through an agent with a declared principal; this
surface exists so a dashboard, a notebook or a reviewer can *see* the research state without holding the
keys to it.

Routes
------
``GET /health``                     project identity, revision, integrity, event head
``GET /state``                      the global research state
``GET /claims`` ``/claims/{id}``    the claim registry, and one claim's obligations and history
``GET /evidence`` ``/evidence/{id}`` ``/evidence/trace/{claim_id}``
``GET /experiments`` ``/analyses``
``GET /literature/papers`` ``/literature/coverage`` ``/audits`` ``/conflicts`` ``/reviews``
``GET /timeline`` ``/timeline/summary`` ``/timeline/claim/{id}`` ``/timeline/why``
``GET /readiness`` ``/agents`` ``/skills``
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query

from ..kernel.kernel import ResearchKernel


def create_app(kernel: ResearchKernel) -> FastAPI:
    """Build the read-only application for one project."""
    app = FastAPI(
        title="ResearchOS",
        version="0.1.0",
        description=(
            "Read-only view of a ResearchOS project: research state, evidence, claims, literature, "
            "audits and readiness. Mutations are not exposed over HTTP by design."
        ),
    )

    # ------------------------------------------------------------------ helpers

    def _require(identifier: str | None, store, kind: str):
        if identifier is None:
            raise HTTPException(status_code=404, detail=f"{kind} id is required")
        record = store.get(identifier)
        if record is None:
            raise HTTPException(status_code=404, detail=f"{kind} {identifier} not found")
        return record

    # ------------------------------------------------------------------ project

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **kernel.describe(), "integrity": kernel.integrity()}

    @app.get("/project")
    def project() -> dict[str, Any]:
        return kernel.project().model_dump(mode="json")

    @app.get("/state")
    def state() -> dict[str, Any]:
        return kernel.research_state().model_dump(mode="json")

    # ------------------------------------------------------------------ claims

    @app.get("/claims")
    def claims(
        status: Optional[str] = Query(None, description="Filter by claim status."),
        live_only: bool = Query(False),
    ) -> list[dict[str, Any]]:
        from ..claims import ClaimRegistry

        registry = ClaimRegistry(kernel)
        items = registry.by_status(status) if status else registry.all()
        if live_only:
            items = [c for c in items if c.is_live]
        return [claim.model_dump(mode="json") for claim in items]

    @app.get("/claims/summary")
    def claims_summary() -> dict[str, Any]:
        from ..claims import ClaimRegistry

        return ClaimRegistry(kernel).summary()

    @app.get("/claims/{claim_id}")
    def claim(claim_id: str) -> dict[str, Any]:
        from ..claims import ClaimLifecycle

        _require(claim_id, kernel.claims, "claim")
        return ClaimLifecycle(kernel).explain(claim_id)

    # ------------------------------------------------------------------ evidence

    @app.get("/evidence")
    def evidence() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in kernel.evidence.all()]

    @app.get("/evidence/summary")
    def evidence_summary() -> dict[str, Any]:
        from ..evidence import EvidenceGraph, EvidenceRegistry

        return {
            "registry": EvidenceRegistry(kernel).summary(),
            "integrity": EvidenceGraph(kernel).integrity(),
        }

    @app.get("/evidence/trace/{claim_id}")
    def evidence_trace(claim_id: str) -> dict[str, Any]:
        from ..evidence import EvidenceGraph

        return EvidenceGraph(kernel).trace_claim(claim_id)

    @app.get("/evidence/{evidence_id}")
    def evidence_item(evidence_id: str) -> dict[str, Any]:
        return _require(evidence_id, kernel.evidence, "evidence").model_dump(mode="json")

    # ------------------------------------------------------------------ experiments / analyses

    @app.get("/experiments")
    def experiments() -> list[dict[str, Any]]:
        return [
            {**item.model_dump(mode="json"), "evidence_level": item.evidence_level().value}
            for item in kernel.experiments.all()
        ]

    @app.get("/analyses")
    def analyses() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in kernel.analyses.all()]

    @app.get("/conflicts")
    def conflicts() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in kernel.conflicts.all()]

    @app.get("/audits")
    def audits() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in kernel.audits.all()]

    @app.get("/reviews")
    def reviews(pending_only: bool = Query(True)) -> list[dict[str, Any]]:
        items = kernel.pending_reviews() if pending_only else kernel.reviews.all()
        return [item.model_dump(mode="json") for item in items]

    # ------------------------------------------------------------------ literature

    @app.get("/literature/papers")
    def papers() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in kernel.papers.all()]

    @app.get("/literature/coverage")
    def literature_coverage() -> dict[str, Any]:
        state = kernel.research_state()
        literature = state.literature_state
        return {
            "coverage_score": literature.coverage_score,
            "basis": literature.coverage_basis,
            "queries_executed": literature.queries_executed,
            "providers_used": literature.providers_used,
            "papers_screened": literature.papers_screened,
            "fulltext_verified": literature.fulltext_verified,
            "closest_prior_work": [ref.model_dump(mode="json") for ref in literature.closest_prior_work],
        }

    @app.get("/literature/novelty")
    def novelty_audits() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in kernel.novelty_audits.all()]

    @app.get("/literature/prior-art")
    def prior_art() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in kernel.matrices.all()]

    # ------------------------------------------------------------------ timeline

    @app.get("/timeline")
    def timeline(ref: Optional[str] = Query(None, description="Only events touching this object.")) -> list[dict[str, Any]]:
        if ref:
            return [event.model_dump(mode="json") for event in kernel.history.provenance_of(ref)]
        return [event.model_dump(mode="json") for event in kernel.history.all()]

    @app.get("/timeline/summary")
    def timeline_summary() -> dict[str, Any]:
        return kernel.history.summary()

    @app.get("/timeline/claim/{claim_id}")
    def timeline_claim(claim_id: str) -> dict[str, Any]:
        return kernel.history.claim_history(claim_id)

    @app.get("/timeline/why")
    def timeline_why(
        ref: Optional[str] = None,
        question: Optional[str] = Query(None, description="One of the standing questions."),
    ) -> dict[str, Any]:
        if question:
            return kernel.history.answer(question)
        return {"records": kernel.history.why(ref=ref, text=None)}

    # ------------------------------------------------------------------ readiness / agents / skills

    @app.get("/readiness")
    def readiness() -> dict[str, Any]:
        from ..paper import ReadinessAssessor

        return ReadinessAssessor(kernel).assess(kernel.human()).model_dump(mode="json")

    @app.get("/agents")
    def agents() -> list[dict[str, Any]]:
        from ..agents import agent_table

        return [
            {"agent": name, "purpose": purpose, "capabilities": caps, "denied": denied}
            for name, purpose, caps, denied in agent_table(kernel)
        ]

    @app.get("/skills")
    def skills() -> dict[str, Any]:
        from ..skills.registry import SkillRegistry

        registry = SkillRegistry(kernel)
        return {
            "health": registry.health(),
            "skills": [skill.model_dump(mode="json") for skill in registry.list()],
        }

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        """The permission table itself, so a reviewer can see who may do what."""
        from ..kernel.permissions import ALL_CAPABILITIES, HUMAN_ONLY_CAPABILITIES

        return {
            "capabilities": sorted(capability.value for capability in ALL_CAPABILITIES),
            "human_only": sorted(capability.value for capability in HUMAN_ONLY_CAPABILITIES),
            "read_only_http": True,
        }

    return app


def app_for_project(root: Optional[str] = None) -> FastAPI:
    """Discover the project from ``root`` (or the working directory) and build the app."""
    return create_app(ResearchKernel.open(root))
