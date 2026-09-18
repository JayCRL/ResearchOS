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

import json
from importlib import resources
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from ..kernel.kernel import ResearchKernel

#: List sizes in the dashboard payload. Research-scale projects have hundreds of records; the UI needs
#: the shape of the state, and drilling into one record is a separate request.
DASHBOARD_LIMIT = 40


def _static_text(name: str) -> str:
    """Read a packaged static file, so the UI ships with the package and needs no build step."""
    return (resources.files("researchos.api") / "static" / name).read_text(encoding="utf-8")


def create_app(kernel: ResearchKernel, *, enable_actions: bool = False) -> FastAPI:
    """Build the application for one project.

    ``enable_actions`` also mounts the optional human-in-the-loop write endpoints
    (:mod:`researchos.api.actions`), which run as the human principal through the same kernel gate as
    the CLI and are restricted to local clients.
    """
    app = FastAPI(
        title="ResearchOS",
        version="0.1.0",
        description=(
            "Read-only view of a ResearchOS project: research state, evidence, claims, literature, "
            "audits and readiness, plus a zero-build dashboard at `/`. Research state is not stored "
            "here — it is the YAML/JSONL plane under `.researchos/`."
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

    # ------------------------------------------------------------------ cockpit

    @app.get("/cockpit", tags=["cockpit"])
    def cockpit() -> dict[str, Any]:
        """The research cockpit: what am I researching, what do I know, what can I not trust, why can I
        not continue, what do I do next.

        Deliberately contains no revision, hash, event head or integrity counter — those belong to the
        audit console at ``/dashboard``. Mixing the two is what makes a research state unreadable.
        """
        from ..cockpit import Cockpit

        return Cockpit(kernel).build().as_dict()

    # ------------------------------------------------------------------ dashboard

    @app.get("/dashboard", tags=["dashboard"])
    def dashboard() -> dict[str, Any]:
        """One payload with everything the dashboard shows.

        Bundled rather than fanned out so the first paint is a single request; every panel keeps its
        own endpoint for drill-down. Counts of *omitted* records are included, so a truncated panel
        never looks like a complete one.
        """
        from ..claims import ClaimLifecycle, ClaimRegistry
        from ..evidence import EvidenceGraph, EvidenceRegistry
        from ..paper import ReadinessAssessor

        state = kernel.research_state()
        claim_registry = ClaimRegistry(kernel)
        claims = kernel.claims.all()
        experiments = kernel.experiments.all()
        reviews = kernel.pending_reviews()
        timeline = kernel.history.all()
        literature = state.literature_state

        def _cap(items: list[Any]) -> tuple[list[Any], int]:
            return items[:DASHBOARD_LIMIT], max(0, len(items) - DASHBOARD_LIMIT)

        claim_rows = [
            {
                **claim.model_dump(mode="json"),
                "permitted_language": claim.language_strength,
                "hidden_evidence_count": len(claim.evidence_ids),
            }
            for claim in claims
        ]
        capped_claims, claims_omitted = _cap(sorted(claim_rows, key=lambda c: (c["status"], c["claim_id"])))
        capped_experiments, experiments_omitted = _cap(
            sorted(
                [
                    {**e.model_dump(mode="json"), "design_level": e.evidence_level().value}
                    for e in experiments
                ],
                key=lambda e: e["title"],
            )
        )
        capped_reviews, reviews_omitted = _cap(sorted(reviews, key=lambda r: (-r.confidence, r.review_item_id)))
        capped_timeline, timeline_omitted = _cap(list(reversed(timeline)))
        capped_decisions, decisions_omitted = _cap(
            sorted(kernel.decisions.all(), key=lambda d: d.created_at, reverse=True)
        )

        open_questions = [q for q in kernel.store("open_question").all() if q.resolved_at is None]
        capped_questions, questions_omitted = _cap(sorted(open_questions, key=lambda q: q.created_at))
        capped_conflicts, conflicts_omitted = _cap(
            sorted(kernel.conflicts.all(), key=lambda c: (not c.blocks_claim_promotion, c.created_at))
        )
        gaps = kernel.skill_gaps.all()
        capped_gaps, gaps_omitted = _cap(sorted(gaps, key=lambda g: (-g.occurrences, g.gap_id)))

        try:
            readiness = ReadinessAssessor(kernel).assess(kernel.human())
            readiness_payload: dict[str, Any] = {
                **readiness.model_dump(mode="json"),
                "blockers": readiness.blockers()[:12],
            }
        except Exception as exc:  # noqa: BLE001 - readiness is a view; never break the dashboard
            readiness_payload = {"dimensions": [], "error": f"{type(exc).__name__}: {exc}"}

        return {
            "project": kernel.project().model_dump(mode="json"),
            "describe": kernel.describe(),
            "integrity": kernel.integrity(),
            "state": state.model_dump(mode="json"),
            "core_question": (
                state.core_question.model_dump(mode="json") if state.core_question else None
            ),
            "frontier": state.current_frontier.model_dump(mode="json"),
            "claims": capped_claims,
            "claims_omitted": claims_omitted,
            "claims_summary": claim_registry.summary(),
            "overclaiming": [
                {"claim_id": claim.claim_id, "statement": claim.statement, "safe": safe}
                for claim, safe in claim_registry.overclaiming()
            ],
            "claims_explained": {
                claim.claim_id: ClaimLifecycle(kernel).explain(claim.claim_id)
                for claim in claims[:DASHBOARD_LIMIT]
                if claim.status.value not in {"REJECTED", "SUPERSEDED"}
            },
            "evidence_summary": EvidenceRegistry(kernel).summary(),
            "evidence_integrity": EvidenceGraph(kernel).integrity(),
            "experiments": capped_experiments,
            "experiments_omitted": experiments_omitted,
            "conflicts": capped_conflicts,
            "conflicts_omitted": conflicts_omitted,
            "conflicts_summary": kernel.ledger.summary(),
            "trust_order": kernel.ledger.trust_table(),
            "decisions": capped_decisions,
            "decisions_omitted": decisions_omitted,
            "questions": capped_questions,
            "questions_omitted": questions_omitted,
            "tasks": {
                "active": (
                    kernel.tasks_store.get(state.active_task).model_dump(mode="json")
                    if state.active_task and kernel.tasks_store.get(state.active_task)
                    else None
                ),
                "queue": [
                    task.model_dump(mode="json")
                    for task in kernel.task_manager.list(open_only=True)[:DASHBOARD_LIMIT]
                ],
                "recent": [
                    task.model_dump(mode="json")
                    for task in sorted(kernel.tasks_store.all(), key=lambda t: t.created_at, reverse=True)[
                        :DASHBOARD_LIMIT
                    ]
                ],
            },
            "literature": {
                "coverage": literature.model_dump(mode="json"),
                "papers": [
                    {
                        "paper_id": p.paper_id,
                        "title": p.title,
                        "year": p.year,
                        "venue": p.venue,
                        "fulltext_status": p.fulltext_status.value,
                        "include": p.include,
                        "providers": [x.value for x in p.providers],
                    }
                    for p in sorted(kernel.papers.all(), key=lambda p: (-(p.relevance or 0), p.title))[
                        :DASHBOARD_LIMIT
                    ]
                ],
                "paper_count": len(kernel.papers.all()),
                "novelty_audits": [
                    {
                        "novelty_audit_id": a.novelty_audit_id,
                        "target_statement": a.target_statement,
                        "verdict": a.verdict.value,
                        "gap_kind": a.gap_kind.value,
                        "coverage_sufficient": a.coverage_is_sufficient(),
                        "coverage_score": a.coverage.score,
                        "matches": len(a.matches),
                        "sentence": a.verdict_sentence(),
                    }
                    for a in sorted(kernel.novelty_audits.all(), key=lambda a: a.created_at, reverse=True)[:10]
                ],
            },
            "skills": _skills_payload(kernel, capped_gaps, gaps_omitted),
            "readiness": readiness_payload,
            "timeline": {
                "events": capped_timeline,
                "omitted": timeline_omitted,
                "summary": kernel.history.summary(),
            },
            "reviews": capped_reviews,
            "reviews_omitted": reviews_omitted,
            "audits": [
                {
                    "audit_id": audit.audit_id,
                    "kind": audit.kind.value,
                    "title": audit.title,
                    "verdict": audit.verdict.value,
                    "findings": len(audit.findings),
                    "blocking": len(audit.blockers()),
                    "created_at": audit.created_at.isoformat(),
                }
                for audit in sorted(kernel.audits.all(), key=lambda a: a.created_at, reverse=True)[:20]
            ],
            "red_team": [
                {
                    "red_team_report_id": r.red_team_report_id,
                    "subject": r.subject,
                    "verdict": r.verdict,
                    "questions": len(r.questions),
                    "blocking": len(r.blocking()),
                }
                for r in sorted(kernel.red_team_reports.all(), key=lambda r: r.created_at, reverse=True)[:5]
            ],
            "paper": [
                {
                    "paper_id": artifact.paper_id,
                    "title": artifact.title,
                    "compiled_at": artifact.compiled_at.isoformat(),
                    "words": artifact.word_count(),
                    "grounding_passed": (
                        artifact.grounding_report.passed if artifact.grounding_report else None
                    ),
                    "blocking": (
                        len(artifact.grounding_report.blockers()) if artifact.grounding_report else None
                    ),
                    "refusals": artifact.refused_additions[:5],
                }
                for artifact in sorted(
                    kernel.paper_artifacts.all(), key=lambda a: a.compiled_at, reverse=True
                )[:5]
            ],
            "actions_enabled": enable_actions,
        }

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard_html() -> HTMLResponse:
        """The zero-build dashboard: one HTML file, no bundler, no npm, no CDN."""
        try:
            html = _static_text("index.html")
        except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover - packaging error
            return HTMLResponse(
                "<h1>ResearchOS</h1><p>dashboard asset is missing from this installation; "
                "use the JSON API at <a href='/docs'>/docs</a>.</p>",
                status_code=500,
            )
        return HTMLResponse(html.replace("__ACTIONS_ENABLED__", "true" if enable_actions else "false"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> JSONResponse:
        return JSONResponse(status_code=204, content=None)

    if enable_actions:
        from .actions import build_actions_router

        app.include_router(build_actions_router(kernel, enabled=True))

    return app


def _skills_payload(kernel: ResearchKernel, gaps: list[Any], gaps_omitted: int) -> dict[str, Any]:
    """Skill health plus the gaps that drive the evolution loop."""
    try:
        from ..skills.registry import SkillRegistry

        registry = SkillRegistry(kernel)
        health = registry.health()
        skills = [
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "status": skill.status.value,
                "trust": skill.trust.value,
                "source": skill.source.value,
                "score": skill.quality_metrics.score(),
                "capabilities": [capability.name for capability in skill.capabilities],
                "last_benchmark_run_id": skill.last_benchmark_run_id,
                "known_failures": skill.known_failures[:3],
            }
            for skill in registry.list()
        ]
    except Exception as exc:  # noqa: BLE001 - the skill plane is optional for a state view
        health, skills = {"error": f"{type(exc).__name__}: {exc}"}, []

    return {
        "health": health,
        "skills": skills,
        "gaps": [gap.model_dump(mode="json") for gap in gaps],
        "gaps_omitted": gaps_omitted,
        "gap_count": len(kernel.skill_gaps.all()),
    }


def app_for_project(root: Optional[str] = None, *, enable_actions: bool = False) -> FastAPI:
    """Discover the project from ``root`` (or the working directory) and build the app."""
    return create_app(ResearchKernel.open(root), enable_actions=enable_actions)
