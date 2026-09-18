"""The ResearchOS Kernel: the single front door to research state.

Rules this module exists to enforce:

* **No ambient authority.** Every mutating method takes an explicit
  :class:`~researchos.kernel.permissions.Principal`.
* **No bypass.** Agents, auditors and skills all go through the same gate; the stores are not
  reachable from outside without a kernel.
* **No silent drift.** Guarded state (core question, core claims, priorities, non-goals, scope)
  only moves through an approved State Transition Request.
* **No lost history.** Deletes are refused; the event log is hash-chained.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from ..models import (
    Analysis,
    Audit,
    AuthorNote,
    BenchmarkRun,
    BenchmarkTask,
    Claim,
    Conflict,
    Decision,
    Evidence,
    Experiment,
    GapRecord,
    Interpretation,
    LiteratureClaim,
    LiteraturePaper,
    LiteratureRelation,
    NoveltyAudit,
    OpenQuestion,
    PaperArtifact,
    PriorArtMatrix,
    QueryPlan,
    RedTeamReport,
    ResearchProject,
    ResearchState,
    ReviewItem,
    SandboxReport,
    SkillCard,
    SkillGap,
    SkillVersion,
    StateTransitionRequest,
    Task,
    TimelineEvent,
)
from ..models.common import utcnow
from .conflicts import ConflictLedger
from .errors import ProjectExists, ProjectNotFound, ResearchOSError
from .events import EventLog
from .paths import MARKER, ProjectPaths
from .permissions import Cap, PolicyEngine, Principal, PrincipalKind
from .store import EntityStore, ProjectStore, YamlIO
from .state import StateManager
from .tasks import TaskManager
from .timeline import TimelineRecorder, TimelineView
from .transitions import TransitionManager


class ResearchKernel:
    """Owns the project layout, the stores, the policy engine and the guarded state."""

    def __init__(self, paths: ProjectPaths, *, auto_create: bool = False) -> None:
        self.paths = paths
        if auto_create and not paths.exists():
            paths.ensure()
        self.paths.assert_exists()

        self.events = EventLog(paths.event_log)
        self.state = StateManager(paths, self.events)
        self.policy = PolicyEngine()

        store = ProjectStore(paths)
        # --- state plane
        self.tasks_store = store.register("task", Task, "task_id")
        self.decisions = store.register("decision", Decision, "decision_id")
        self.strs = store.register("str", StateTransitionRequest, "str_id")
        # --- research plane
        self.claims = store.register("claim", Claim, "claim_id")
        self.experiments = store.register("experiment", Experiment, "experiment_id")
        self.evidence = store.register("evidence", Evidence, "evidence_id")
        self.analyses = store.register("analysis", Analysis, "analysis_id")
        self.interpretations = store.register("interpretation", Interpretation, "interpretation_id")
        # --- literature plane
        self.papers = store.register("literature_paper", LiteraturePaper, "paper_id")
        self.relations = store.register("literature_relation", LiteratureRelation, "relation_id")
        self.literature_claims = store.register("literature_claim", LiteratureClaim, "literature_claim_id")
        self.query_plans = store.register("query_plan", QueryPlan, "query_plan_id")
        self.novelty_audits = store.register("novelty_audit", NoveltyAudit, "novelty_audit_id")
        self.matrices = store.register("prior_art_matrix", PriorArtMatrix, "matrix_id")
        self.gaps = store.register("gap", GapRecord, "gap_id")
        self.open_questions = store.register("open_question", OpenQuestion, "question_id")
        # --- skill plane
        self.skills = store.register("skill", SkillCard, "skill_id")
        self.skill_versions = store.register("skill_version", SkillVersion, "skill_version_id")
        self.benchmark_tasks = store.register("benchmark_task", BenchmarkTask, "benchmark_task_id")
        self.benchmark_runs = store.register("benchmark_run", BenchmarkRun, "benchmark_run_id")
        self.sandbox_reports = store.register("sandbox_report", SandboxReport, "sandbox_report_id")
        self.skill_gaps = store.register("skill_gap", SkillGap, "gap_id")
        # --- audit plane
        self.audits = store.register("audit", Audit, "audit_id")
        self.conflicts = store.register("conflict", Conflict, "conflict_id")
        self.reviews = store.register("review_item", ReviewItem, "review_item_id")
        self.timeline_events = store.register("timeline_event", TimelineEvent, "event_id")
        self.notes = store.register("author_note", AuthorNote, "note_id")
        self.paper_artifacts = store.register("paper", PaperArtifact, "paper_id")
        self.red_team_reports = store.register("red_team_report", RedTeamReport, "red_team_report_id")
        self._store = store

        # --- managers
        self.task_manager = TaskManager(self.tasks_store, self.state, self.events)
        self.transitions = TransitionManager(
            self.strs, self.decisions, self.state, self.task_manager, self.events
        )
        self.ledger = ConflictLedger(self.conflicts, self.events)
        self.timeline = TimelineRecorder(self.timeline_events, self.events)
        self.history = TimelineView(self)

    # ================================================================== constructors

    @classmethod
    def create(
        cls,
        root: str | os.PathLike[str],
        *,
        name: str | None = None,
        description: str = "",
        domain: str = "",
        allow_existing: bool = False,
    ) -> "ResearchKernel":
        """``researchos init`` — create ``.researchos/`` and the initial research state."""
        paths = ProjectPaths.for_root(root)
        if paths.exists() and not allow_existing:
            raise ProjectExists(f"{paths.ros} already exists; refusing to overwrite research state")
        paths.ensure()

        project = ResearchProject(
            name=name or Path(root).resolve().name,
            description=description,
            domain=domain,
            root=str(paths.root),
        )
        YamlIO.write_atomic(paths.project_file, YamlIO.dump(project))

        state = ResearchState(
            project_id=project.project_id,
            project_name=project.name,
            revision=0,
        )
        YamlIO.write_atomic(paths.research_state_file, YamlIO.dump(state))

        kernel = cls(paths)
        kernel.events.append(
            "project.created",
            actor="human",
            payload={
                "project_id": project.project_id,
                "name": project.name,
                "root": str(paths.root),
                "schema_version": project.schema_version,
            },
        )
        kernel.timeline.record(
            "PROJECT_CREATED",
            f"Project {project.name} initialised",
            actor="human",
            state_revision=0,
        )
        return kernel

    @classmethod
    def open(cls, root: str | os.PathLike[str] | None = None) -> "ResearchKernel":
        return cls(ProjectPaths.discover(root))

    @classmethod
    def open_or_none(cls, root: str | os.PathLike[str] | None = None) -> "ResearchKernel | None":
        paths = ProjectPaths.discover_or_none(root)
        return cls(paths) if paths else None

    # ================================================================== accessors

    @property
    def root(self) -> Path:
        return self.paths.root

    @property
    def ros_dir(self) -> Path:
        return self.paths.ros

    def project(self) -> ResearchProject:
        if not self.paths.project_file.is_file():
            raise ProjectNotFound(f"{self.paths.project_file} is missing")
        return ResearchProject.model_validate(YamlIO.read(self.paths.project_file))

    def save_project(self, project: ResearchProject) -> ResearchProject:
        YamlIO.write_atomic(self.paths.project_file, YamlIO.dump(project))
        return project

    def research_state(self, *, refresh: bool = True) -> ResearchState:
        return self.state.load(refresh=refresh)

    def principal(self, name: str) -> Principal:
        """Resolve a principal by name, materialising ``external_skill:<id>`` on demand."""
        if name.startswith("external_skill:"):
            return self.policy.external_skill(name.split(":", 1)[1])
        return self.policy.get(name)

    def human(self) -> Principal:
        return self.policy.get("human")

    def agent(self, agent_type: str) -> Principal:
        return self.policy.agent(agent_type)

    def require_capability(self, principal: Principal, capability: Cap, resource: str = "") -> None:
        principal.require(capability, resource or None)

    def store(self, kind: str) -> EntityStore:
        return self._store[kind]

    def store_kinds(self) -> list[str]:
        return self._store.kinds()

    # ================================================================== status

    def counts(self) -> dict[str, int]:
        return {
            kind: self._store[kind].count() for kind in self._store.kinds()
        }

    def integrity(self) -> dict[str, Any]:
        """Cross-store consistency checks — cheap, deterministic, no LLM."""
        state = self.research_state()
        issues: list[str] = []
        for claim_id in state.core_claims:
            claim = self.claims.get(claim_id)
            if claim is None:
                issues.append(f"core claim {claim_id} is referenced by the state but does not exist")
            elif not claim.is_core:
                issues.append(f"claim {claim_id} is listed as core in state but not flagged is_core")
        for claim_id in state.rejected_claims:
            claim = self.claims.get(claim_id)
            if claim is None:
                issues.append(f"rejected claim {claim_id} is referenced by the state but missing")
            elif claim.status.value != "REJECTED":
                issues.append(f"claim {claim_id} is in rejected_claims but has status {claim.status.value}")
        for task_id in [state.active_task, *state.task_queue]:
            if task_id and self.tasks_store.get(task_id) is None:
                issues.append(f"task {task_id} is referenced by the state but missing")
        chain = self.events.verify()
        if not chain.ok:
            issues.extend(chain.problems)
        for conflict in self.ledger.unresolved():
            issues.append(f"unresolved conflict {conflict.conflict_id}: {conflict.difference[:80]}")
        return {
            "ok": not issues,
            "revision": state.revision,
            "events": chain.events_checked,
            "head": chain.head_digest[:12] if chain.ok else "BROKEN",
            "issues": issues,
        }

    def describe(self) -> dict[str, Any]:
        state = self.research_state()
        project = self.project()
        return {
            "project": project.name,
            "root": str(self.root),
            "revision": state.revision,
            "core_question": state.core_question.statement if state.core_question else None,
            "core_claims": len(state.core_claims),
            "counts": self.counts(),
            "timeline_events": self.timeline.count(),
            "head": self.events.head_digest()[:12],
        }

    # ================================================================== helpers

    def record_decision(
        self,
        principal: Principal,
        *,
        kind: str,
        summary: str,
        rationale: str,
        evidence_ids: list[str] | None = None,
        affected_claims: list[str] | None = None,
        affected_experiments: list[str] | None = None,
        alternatives_considered: list[str] | None = None,
        rejected_alternatives: list[str] | None = None,
        consequences: list[str] | None = None,
        task_id: str | None = None,
    ) -> Decision:
        """Record a decision. Decisions are append-only and explain changes years later."""
        principal.require(Cap.DECISION_CREATE, "decision.create")
        from ..models.decision import DecisionKind

        decision = Decision(
            kind=DecisionKind(kind) if isinstance(kind, str) else kind,
            summary=summary,
            rationale=rationale,
            evidence_ids=list(evidence_ids or []),
            affected_claims=list(affected_claims or []),
            affected_experiments=list(affected_experiments or []),
            alternatives_considered=list(alternatives_considered or []),
            rejected_alternatives=list(rejected_alternatives or []),
            consequences=list(consequences or []),
            made_by=principal.name,
            task_id=task_id,
        )
        self.decisions.save(decision)
        self.state.update(
            principal,
            lambda s: setattr(s, "research_decisions", [*s.research_decisions, decision.decision_id]),
            event_kind="decision.recorded",
            payload={"decision_id": decision.decision_id, "summary": summary[:300]},
            task_id=task_id,
        )
        return decision

    def save_audit(self, principal: Principal, audit: Audit) -> Audit:
        principal.require(Cap.AUDIT_CREATE, "audit.create")
        self.audits.save(audit)
        self.events.append(
            "audit.created",
            actor=principal.name,
            task_id=audit.task_id,
            payload={
                "audit_id": audit.audit_id,
                "kind": audit.kind.value,
                "verdict": audit.verdict.value,
                "findings": len(audit.findings),
                "blocking": len(audit.blockers()),
            },
        )
        self.timeline.record(
            "AUDIT",
            f"{audit.kind.value} audit: {audit.title}",
            detail=audit.summary,
            actor=principal.name,
            task_id=audit.task_id,
            refs=audit.subjects,
            state_revision=self.state.revision(),
        )
        return audit

    def create_review_items(self, items: list[ReviewItem], *, principal: Principal | None = None) -> int:
        actor = (principal or self.human()).name
        for item in items:
            self.reviews.save(item)
        self.events.append(
            "review.enqueued",
            actor=actor,
            payload={"count": len(items), "kinds": sorted({i.kind.value for i in items})},
        )
        return len(items)

    def pending_reviews(self) -> list[ReviewItem]:
        return [item for item in self.reviews.all() if item.is_pending()]

    def __iter__(self) -> Iterator[str]:
        return iter(self.store_kinds())

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<ResearchKernel {self.root.name!r} rev={self.state.revision()} at {MARKER}>"


__all__ = [
    "ResearchKernel",
    "ProjectPaths",
    "Principal",
    "PrincipalKind",
    "PolicyEngine",
    "Cap",
    "now",
]


def now():
    return utcnow()
