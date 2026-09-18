"""The fifteen agents, wired to the modules that do the real work.

Each agent class documents the boundary it may not cross. Those boundaries are enforced by the
kernel's capability table, not by convention — try to cross one and you get a
:class:`~researchos.kernel.errors.PermissionDenied` naming the missing capability.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..kernel.errors import ResearchOSError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap
from ..models.common import (
    ChangeClass,
    ClaimStatus,
    EvidenceLevel,
    RiskLevel,
    Severity,
    TaskPriority,
    TaskPurpose,
)
from ..models.paper import RedTeamQuestion, RedTeamReport
from ..models.research_state import OpenQuestion
from ..models.task import FindingKind
from ..models.timeline import TimelineEventKind
from .base import Agent, Proposal


# ======================================================================================
# 1. Research Planner
# ======================================================================================
class ResearchPlanner(Agent):
    """Turns the global research state into bounded tasks. May request transitions, never approve."""

    agent_type = "planner"
    purpose = "read global state, create bounded tasks, file state-transition requests"

    def plan(self, objective: str, *, purpose: TaskPurpose = TaskPurpose.ANALYSE,
             priority: TaskPriority = TaskPriority.SECONDARY, **kwargs) -> Any:
        self.require(Cap.STATE_WRITE_TASK, "task.create")
        return self.kernel.task_manager.create(
            self.principal, objective=objective, purpose=purpose, priority=priority, **kwargs
        )

    def propose_core_question_change(
        self,
        *,
        statement: str,
        reason: str,
        evidence_ids: Sequence[str],
        task_id: str,
        risk: RiskLevel = RiskLevel.MEDIUM,
    ) -> Any:
        """File an STR. This is the ONLY way an agent may move the research direction."""
        return self.kernel.transitions.request(
            self.principal,
            operations=[{"op": "set", "path": "core_question", "value": {"statement": statement}}],
            reason=reason,
            evidence_ids=list(evidence_ids),
            change_class=ChangeClass.CORE_QUESTION,
            task_id=task_id,
            risk=risk,
        )

    def next_actions(self) -> list[str]:
        state = self.kernel.research_state()
        actions = list(state.current_frontier.next_actions)
        if not actions:
            actions = [f"address open question: {q}" for q in state.open_questions[:3]]
        return actions


# ======================================================================================
# 2. Research Importer
# ======================================================================================
class ResearchImporter(Agent):
    """Reconstructs research state from existing material. Never approves a claim."""

    agent_type = "importer"
    purpose = "reconstruct research state from imported material; queue uncertain recoveries"

    def import_path(self, path: str, **kwargs) -> Any:
        from ..importer import ImportPipeline

        pipeline = ImportPipeline(self.kernel, principal=self.principal, **kwargs)
        return pipeline.run(path)

    def may_approve_claims(self) -> bool:
        return False  # documented, and also structurally true: the capability is absent


# ======================================================================================
# 3. Literature Researcher
# ======================================================================================
class LiteratureResearcher(Agent):
    """Broad, recorded literature search. May write literature state; never touches experiments."""

    agent_type = "literature_researcher"
    purpose = "plan and execute broad literature searches; build the literature map"

    def plan_search(self, target: str, **kwargs) -> Any:
        planner = self._planner()
        return planner.plan(self.principal, target, **kwargs)

    def execute_search(self, plan, providers, **kwargs) -> Any:
        return self._planner().execute(self.principal, plan, providers, **kwargs)

    def build_graph(self):
        from ..literature.graph import LiteratureGraph

        return LiteratureGraph(self.kernel)

    def _planner(self):
        try:
            from ..literature.query_planner import QueryPlanner
        except ImportError as exc:  # pragma: no cover - depends on optional module presence
            raise ResearchOSError(
                "the literature query planner is unavailable; the Literature OS core module is missing"
            ) from exc
        return QueryPlanner(self.kernel)


# ======================================================================================
# 4. Experiment Designer
# ======================================================================================
class ExperimentDesigner(Agent):
    """Registers experiments with declared designs. Design flags decide the evidence ceiling."""

    agent_type = "experiment_designer"
    purpose = "design experiments: variables, treatment, control, matched conditions, metrics"

    def register(self, experiment) -> Any:
        self.require(Cap.EXPERIMENT_REGISTER, "experiment.register")
        self.kernel.experiments.save(experiment)
        self.kernel.events.append(
            "experiment.registered",
            actor=self.principal.name,
            payload={
                "experiment_id": experiment.experiment_id,
                "title": experiment.title,
                "design_level": experiment.evidence_level().value,
                "matched_conditions": experiment.matched_conditions,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.EXPERIMENT_REGISTERED,
            f"Registered experiment: {experiment.title}",
            actor=self.principal.name,
            refs=[experiment.experiment_id],
            state_revision=self.kernel.state.revision(),
        )
        return experiment

    def design_gaps(self, experiment) -> list[str]:
        """What is missing for this design to support the claim it is meant to test."""
        gaps: list[str] = []
        if experiment.control is None:
            gaps.append("no control arm: this design can support an observation, not a comparison")
        if experiment.control is not None and not experiment.matched_conditions:
            gaps.append("no matched conditions recorded: the comparison may be confounded")
        if not experiment.seeds:
            gaps.append("no seeds recorded: single-run results are not reproducible evidence")
        if experiment.n is None:
            gaps.append("n is unrecorded")
        if experiment.undefined_metrics():
            gaps.append(f"metrics without definitions: {experiment.undefined_metrics()}")
        return gaps


# ======================================================================================
# 5. Experiment Agent
# ======================================================================================
class ExperimentAgent(Agent):
    """Runs experiments and records raw evidence. May never touch a core claim."""

    agent_type = "experiment"
    purpose = "execute registered experiments and register raw artifacts and evidence"

    def register_raw_artifact(self, experiment_id: str, relative_path: str):
        from ..kernel.provenance import register_artifact

        self.require(Cap.EVIDENCE_RAW_WRITE, "evidence.raw")
        artifact = register_artifact(self.kernel.root, relative_path, kind="raw")
        experiment = self.kernel.experiments.require(experiment_id)
        experiment = experiment.with_updates(
            raw_artifacts=[*experiment.raw_artifacts, artifact]
        )
        self.kernel.experiments.save(experiment)
        self.kernel.events.append(
            "experiment.artifact",
            actor=self.principal.name,
            payload={"experiment_id": experiment_id, "path": artifact.path, "sha256": artifact.sha256},
        )
        return artifact

    def record_failure(self, experiment_id: str, *, reason: str) -> Any:
        experiment = self.kernel.experiments.require(experiment_id)
        experiment = experiment.with_updates(
            status=__import__("researchos.models", fromlist=["ExperimentStatus"]).ExperimentStatus.FAILED,
            failure_reason=reason,
        )
        self.kernel.experiments.save(experiment)
        self.kernel.timeline.record(
            TimelineEventKind.EXPERIMENT_FAILED,
            f"Experiment failed: {experiment.title}",
            detail=reason,
            actor=self.principal.name,
            refs=[experiment_id],
        )
        return experiment

    def may_modify_core_claim(self) -> bool:
        return False


# ======================================================================================
# 6. Analysis Agent
# ======================================================================================
class AnalysisAgent(Agent):
    """Computes verified analysis from raw artifacts. Numbers come from here, not from prose."""

    agent_type = "analysis"
    purpose = "compute statistics from artifacts and verify evidence by re-hashing"

    def record_analysis(self, analysis) -> Any:
        self.require(Cap.ANALYSIS_COMPUTE, "analysis.compute")
        self.kernel.analyses.save(analysis)
        self.kernel.events.append(
            "analysis.computed",
            actor=self.principal.name,
            payload={
                "analysis_id": analysis.analysis_id,
                "results": len(analysis.results),
                "deterministic": analysis.deterministic,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.ANALYSIS,
            f"Analysis: {analysis.title}",
            actor=self.principal.name,
            refs=[analysis.analysis_id],
        )
        return analysis

    def verify_evidence(self, evidence_id: str, *, method: str, level: EvidenceLevel | None = None):
        from ..evidence import EvidenceRegistry

        return EvidenceRegistry(self.kernel).verify(
            self.principal, evidence_id, method=method, level=level
        )

    def statistical_audit(self, **kwargs):
        from ..analysis import StatisticalAuditor

        return StatisticalAuditor(self.kernel).audit(self.principal, **kwargs)


# ======================================================================================
# 7. Mechanism Auditor
# ======================================================================================
class MechanismAuditorAgent(Agent):
    """Checks that mechanism claims match the rung of the ladder the evidence reaches."""

    agent_type = "mechanism_auditor"
    purpose = "audit observation/correlation/control/intervention/necessity claims"

    def audit(self, **kwargs):
        from ..analysis import MechanismAuditor

        return MechanismAuditor(self.kernel).audit(self.principal, **kwargs)

    def may_modify_experiment(self) -> bool:
        return False


# ======================================================================================
# 8. Novelty Auditor
# ======================================================================================
class NoveltyAuditorAgent(Agent):
    """Answers "has this been done?" with a coverage-gated verdict, never with a slogan."""

    agent_type = "novelty_auditor"
    purpose = "run novelty audits and report coverage honestly"

    def audit(self, target_statement: str, **kwargs) -> Any:
        from ..literature.novelty import NoveltyAuditor

        return NoveltyAuditor(self.kernel).audit(
            self.principal, target_statement, task_id=kwargs.pop("task_id", None), **kwargs
        )


# ======================================================================================
# 9. Claim Manager
# ======================================================================================
class ClaimManager(Agent):
    """Proposes and performs *evidence-gated* claim transitions. Cannot grant SUPPORTED alone."""

    agent_type = "claim_manager"
    purpose = "manage the claim registry; propose transitions; never self-approve"

    def propose_claim(self, *, statement: str, scope: str, evidence_ids: Sequence[str] = (), **kwargs):
        from ..claims import ClaimLifecycle

        return ClaimLifecycle(self.kernel).create(
            self.principal, statement=statement, scope=scope, evidence_ids=list(evidence_ids), **kwargs
        )

    def transition(self, claim_id: str, to_status: ClaimStatus, **kwargs):
        from ..claims import ClaimLifecycle

        return ClaimLifecycle(self.kernel).transition(
            self.principal, claim_id, to_status, **kwargs
        )

    def request_approval(self, claim_id: str, *, reason: str) -> Proposal:
        """When a claim has earned SUPPORTED but the agent lacks approval authority, ask the human."""
        return self.propose(
            Proposal(
                kind="CLAIM_APPROVAL_REQUEST",
                summary=f"claim {claim_id} appears to meet the SUPPORTED obligations",
                rationale=reason,
                requires=("claim.approve",),
                payload={"claim_id": claim_id},
            )
        )

    def explain(self, claim_id: str) -> Mapping[str, Any]:
        from ..claims import ClaimLifecycle

        return ClaimLifecycle(self.kernel).explain(claim_id)


# ======================================================================================
# 10. Paper Writer
# ======================================================================================
class PaperWriter(Agent):
    """Compiles a paper from approved material. Cannot invent a number, citation or mechanism."""

    agent_type = "paper_writer"
    purpose = "compile sections from approved claims, verified evidence and literature claims"

    def compile(self, **kwargs):
        from ..paper import PaperCompiler

        return PaperCompiler(self.kernel).compile(self.principal, **kwargs)

    def available_numbers(self) -> int:
        """How many numbers the writer is even allowed to use."""
        return sum(len(a.results) for a in self.kernel.analyses.all())


# ======================================================================================
# 11. Paper Auditor
# ======================================================================================
class PaperAuditor(Agent):
    """Audits compiled prose: AI style, overclaiming, history mismatches."""

    agent_type = "paper_auditor"
    purpose = "audit a compiled paper for style, overclaim and grounding"

    def audit_style(self, paper, **kwargs):
        from ..paper import StyleAuditor

        return StyleAuditor().audit_paper(
            self.principal, paper, kernel=self.kernel, **kwargs
        )


# ======================================================================================
# 12. Red Team Agent
# ======================================================================================
class RedTeamAgent(Agent):
    """Asks what would falsify the claims, and which weakness a reviewer will find first."""

    agent_type = "red_team"
    purpose = "adversarially review claims and experiments; report, never rewrite"

    def review(self, *, subject: str = "current research state", claim_ids: Sequence[str] = ()) -> RedTeamReport:
        claims = [
            c
            for c in self.kernel.claims.all()
            if (not claim_ids or c.claim_id in claim_ids) and c.status is not ClaimStatus.REJECTED
        ]
        experiments = self.kernel.experiments.all()
        questions: list[RedTeamQuestion] = []

        for claim in claims:
            questions.append(
                RedTeamQuestion(
                    question=f"What observation would falsify: {claim.statement[:140]}?",
                    category="falsification",
                    target_ref=claim.claim_id,
                    concern="A claim without a stated falsifier is not a scientific claim.",
                    severity=Severity.HIGH,
                    what_would_falsify=(
                        "a matched-control run in which the effect reverses or disappears"
                    ),
                    suggested_experiment="Run the same comparison with a matched control and a fresh seed set.",
                )
            )
            if claim.status in (ClaimStatus.HYPOTHESIS, ClaimStatus.TESTED):
                questions.append(
                    RedTeamQuestion(
                        question=f"On what evidence would a reviewer reject {claim.claim_id}?",
                        category="reviewer_reading",
                        target_ref=claim.claim_id,
                        concern=f"claim is only {claim.status.value}",
                        severity=Severity.HIGH,
                        surviving_alternatives=["the effect is an optimisation artefact"],
                    )
                )

        settings: dict[tuple[Any, Any], list[str]] = {}
        for experiment in experiments:
            settings.setdefault((experiment.model, experiment.scale), []).append(experiment.experiment_id)
        for (model, scale), ids in sorted(settings.items(), key=lambda kv: str(kv[0])):
            if len(ids) >= 1 and len(settings) == 1:
                questions.append(
                    RedTeamQuestion(
                        question=f"Is the effect specific to {model or 'this model'} at {scale or 'this scale'}?",
                        category="single_dependency",
                        target_ref=ids[0],
                        concern="all completed experiments share one setting",
                        severity=Severity.MEDIUM,
                        suggested_experiment="Replicate at a second scale or on a second model family.",
                    )
                )

        for experiment in experiments:
            if experiment.control is None:
                questions.append(
                    RedTeamQuestion(
                        question=f"Which baseline is under-matched for {experiment.title}?",
                        category="baseline_matching",
                        target_ref=experiment.experiment_id,
                        concern="no control arm registered",
                        severity=Severity.HIGH,
                        suggested_experiment="Add a matched baseline arm and re-run.",
                    )
                )
            if len(experiment.seeds) < 2:
                questions.append(
                    RedTeamQuestion(
                        question=f"How much of {experiment.title} depends on one seed?",
                        category="single_dependency",
                        target_ref=experiment.experiment_id,
                        concern=f"only {len(experiment.seeds)} seed(s) recorded",
                        severity=Severity.MEDIUM,
                    )
                )

        for conflict in self.kernel.ledger.unresolved():
            questions.append(
                RedTeamQuestion(
                    question=f"Which number is right: {conflict.difference[:120]}?",
                    category="metric_contamination",
                    target_ref=conflict.conflict_id,
                    concern="unresolved source disagreement",
                    severity=Severity.HIGH,
                )
            )

        closed = [q for q in questions if q.category == "falsification"]
        blocking = [q for q in questions if q.severity in (Severity.HIGH, Severity.BLOCKER)]
        report = RedTeamReport(
            subject=subject,
            subject_refs=[c.claim_id for c in claims] + [e.experiment_id for e in experiments],
            questions=questions,
            strongest_objection=(
                questions[0].question if questions else "no claim is strong enough to object to yet"
            ),
            weakest_experiment_id=(experiments[0].experiment_id if experiments else None),
            single_dependencies=[
                q.target_ref for q in questions if q.category == "single_dependency" and q.target_ref
            ],
            meta_review=(
                f"{len(claims)} claim(s) and {len(experiments)} experiment(s) reviewed; "
                f"{len(blocking)} high-severity concern(s)"
            ),
            verdict="READY" if not questions else ("NEEDS_WORK" if claims else "NOT_READY"),
            created_by=self.principal.name,
        )
        self.kernel.red_team_reports.save(report)
        self.kernel.events.append(
            "red_team.report",
            actor=self.principal.name,
            payload={
                "report_id": report.red_team_report_id,
                "questions": len(questions),
                "verdict": report.verdict,
                "blocking": len(blocking),
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.AUDIT,
            f"Red-team review: {report.verdict}",
            detail=report.meta_review,
            actor=self.principal.name,
            refs=[report.red_team_report_id, *report.subject_refs[:5]],
            state_revision=self.kernel.state.revision(),
        )
        _ = closed
        return report


def report_blockers(questions: Sequence[RedTeamQuestion]) -> list[RedTeamQuestion]:
    return [q for q in questions if q.severity in (Severity.HIGH, Severity.BLOCKER)]


# ======================================================================================
# 13-15. Skill agents
# ======================================================================================
class SkillDiscoveryAgentWrapper(Agent):
    agent_type = "skill_discovery"
    purpose = "detect capability gaps and search the web/GitHub for candidate skills"

    def _agent(self):
        from ..skills.discovery import SkillDiscoveryAgent

        return SkillDiscoveryAgent(self.kernel)

    def detect_gaps(self, **kwargs):
        return self._agent().detect_gaps(self.principal, **kwargs)

    def search(self, gap, provider, **kwargs):
        return self._agent().search(self.principal, gap, provider=provider, **kwargs)


class SkillEvaluatorAgent(Agent):
    agent_type = "skill_evaluator"
    purpose = "sandbox, benchmark and regression-test candidate skills"

    def sandbox(self, skill, **kwargs):
        from ..skills.sandbox import SkillSandbox

        return SkillSandbox(self.kernel).inspect(self.principal, skill, **kwargs)

    def benchmark(self, **kwargs):
        from ..skills.benchmark import BenchmarkHarness

        return BenchmarkHarness(self.kernel).run(self.principal, **kwargs)


class SkillSynthesizerAgent(Agent):
    agent_type = "skill_synthesizer"
    purpose = "combine existing skills plus a research-specific rule into a new candidate"

    def synthesise(self, **kwargs):
        from ..skills.discovery import SkillDiscoveryAgent

        return SkillDiscoveryAgent(self.kernel).synthesise(self.principal, **kwargs)


# ======================================================================================
# Registry
# ======================================================================================

AGENT_CLASSES: tuple[type[Agent], ...] = (
    ResearchPlanner,
    ResearchImporter,
    LiteratureResearcher,
    ExperimentDesigner,
    ExperimentAgent,
    AnalysisAgent,
    MechanismAuditorAgent,
    NoveltyAuditorAgent,
    ClaimManager,
    PaperWriter,
    PaperAuditor,
    RedTeamAgent,
    SkillDiscoveryAgentWrapper,
    SkillEvaluatorAgent,
    SkillSynthesizerAgent,
)


def build_agents(kernel: ResearchKernel) -> dict[str, Agent]:
    """Instantiate all fifteen agents, keyed by agent type."""
    agents: dict[str, Agent] = {}
    for cls in AGENT_CLASSES:
        agent = cls(kernel)
        agents[agent.agent_type] = agent
    return agents


def agent_table(kernel: ResearchKernel) -> list[tuple[str, str, int, list[str]]]:
    """(agent, purpose, #capabilities, denied-by-default capabilities) for display."""
    denied = ["claim.approve", "state.transition.approve", "skill.activate", "state.core.write"]
    rows: list[tuple[str, str, int, list[str]]] = []
    for name, agent in sorted(build_agents(kernel).items()):
        rows.append(
            (
                name,
                agent.purpose,
                len(agent.principal.capabilities),
                [cap for cap in denied if not agent.can(cap)],
            )
        )
    return rows
