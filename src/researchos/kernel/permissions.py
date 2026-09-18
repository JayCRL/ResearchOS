"""The permission system: principals, capabilities and immutable-resource guards.

Two independent layers, both mandatory:

1. **Capabilities** — an agent may only do what its role grants. ``paper_writer`` has no
   ``claim.propose``; ``external_skill:*`` has no write capability at all.
2. **Resource guards** — some resources are immutable *regardless of capability*. Raw evidence,
   the event log, provenance, rejected claims and failed experiments are append-only. This is why
   "the agent was allowed to" can never justify destroying a raw result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, Mapping

from ..models.common import StrEnum
from .errors import ImmutableResourceError, PermissionDenied, TaskBoundaryViolation


class Cap(StrEnum):
    """Every capability in the system. Kernel mutations name one explicitly."""

    STATE_READ = "state.read"
    STATE_WRITE_TASK = "state.write.task"
    STATE_TRANSITION_REQUEST = "state.transition.request"
    STATE_TRANSITION_APPROVE = "state.transition.approve"
    STATE_CORE_WRITE = "state.core.write"
    STATE_ADMIN = "state.admin"

    CLAIM_READ = "claim.read"
    CLAIM_PROPOSE = "claim.propose"
    CLAIM_APPROVE = "claim.approve"
    CLAIM_REJECT = "claim.reject"
    CLAIM_SUPERSEDE = "claim.supersede"

    EVIDENCE_READ = "evidence.read"
    EVIDENCE_CREATE = "evidence.create"
    EVIDENCE_RAW_WRITE = "evidence.raw.write"
    EVIDENCE_VERIFY = "evidence.verify"

    EXPERIMENT_READ = "experiment.read"
    EXPERIMENT_REGISTER = "experiment.register"
    EXPERIMENT_RUN = "experiment.run"
    EXPERIMENT_UPDATE = "experiment.update"

    ANALYSIS_READ = "analysis.read"
    ANALYSIS_COMPUTE = "analysis.compute"

    LITERATURE_READ = "literature.read"
    LITERATURE_WRITE = "literature.write"
    LITERATURE_NOVELTY_AUDIT = "literature.novelty_audit"
    LITERATURE_PROVIDER_USE = "literature.provider_use"

    DECISION_CREATE = "decision.create"
    DECISION_READ = "decision.read"
    AUDIT_CREATE = "audit.create"
    AUDIT_READ = "audit.read"
    CONFLICT_CREATE = "conflict.create"
    CONFLICT_RESOLVE = "conflict.resolve"
    CONFLICT_READ = "conflict.read"

    PAPER_READ_APPROVED = "paper.read_approved"
    PAPER_COMPILE = "paper.compile"
    PAPER_STYLE_AUDIT = "paper.style_audit"
    PAPER_READ = "paper.read"

    SKILL_READ = "skill.read"
    SKILL_DISCOVER = "skill.discover"
    SKILL_PROPOSE = "skill.propose"
    SKILL_EVALUATE = "skill.evaluate"
    SKILL_INSTALL = "skill.install"
    SKILL_ACTIVATE = "skill.activate"
    SKILL_DEPRECATE = "skill.deprecate"

    TIMELINE_READ = "timeline.read"
    TIMELINE_WRITE = "timeline.write"
    EVENT_APPEND = "event.append"
    EVENT_VERIFY = "event.verify"

    REVIEW_READ = "review.read"
    REVIEW_DECIDE = "review.decide"

    # External skills may only *propose*; the kernel decides.
    PROPOSE_PROCEDURE = "propose.procedure"
    PROPOSE_RECOMMENDATION = "propose.recommendation"
    PROPOSE_ANALYSIS = "propose.analysis"
    PROPOSE_DRAFT = "propose.draft"


ALL_CAPABILITIES: FrozenSet[Cap] = frozenset(Cap)

READ_CAPS: FrozenSet[Cap] = frozenset(c for c in Cap if ".read" in c.value or c.value.endswith(".verify"))


class PrincipalKind(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    EXTERNAL_SKILL = "external_skill"
    SYSTEM = "system"


@dataclass(frozen=True)
class Principal:
    """An actor. There is no ambient authority anywhere in ResearchOS — every mutation names one."""

    name: str
    kind: PrincipalKind = PrincipalKind.AGENT
    capabilities: FrozenSet[Cap] = field(default_factory=frozenset)
    agent_type: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_human(self) -> bool:
        return self.kind is PrincipalKind.HUMAN

    @property
    def is_external(self) -> bool:
        return self.kind is PrincipalKind.EXTERNAL_SKILL

    def has(self, capability: Cap | str) -> bool:
        cap = Cap(capability) if isinstance(capability, str) else capability
        return cap in self.capabilities

    def require(self, capability: Cap | str, resource: str | None = None) -> None:
        cap = Cap(capability) if isinstance(capability, str) else capability
        if cap not in self.capabilities:
            raise PermissionDenied(
                f"principal {self.name!r} lacks capability {cap.value!r}"
                + (f" required for {resource!r}" if resource else ""),
                principal=self.name,
                resource=resource,
            )

    def describe(self) -> str:
        return f"{self.name} ({self.kind.value}, {len(self.capabilities)} capabilities)"


def _caps(*items: Cap) -> FrozenSet[Cap]:
    return frozenset(items)


#: The capability grant table. Read this as the formal statement of "what each agent may do".
AGENT_CAPABILITIES: Mapping[str, FrozenSet[Cap]] = {
    "planner": _caps(
        Cap.STATE_READ, Cap.STATE_WRITE_TASK, Cap.STATE_TRANSITION_REQUEST,
        Cap.CLAIM_READ, Cap.EVIDENCE_READ, Cap.EXPERIMENT_READ, Cap.ANALYSIS_READ,
        Cap.LITERATURE_READ, Cap.SKILL_READ, Cap.TIMELINE_READ, Cap.DECISION_CREATE,
        Cap.DECISION_READ, Cap.REVIEW_READ, Cap.AUDIT_READ, Cap.CONFLICT_READ, Cap.PAPER_READ,
    ),
    "importer": _caps(
        Cap.STATE_READ, Cap.STATE_WRITE_TASK, Cap.CLAIM_READ, Cap.CLAIM_PROPOSE,
        Cap.EVIDENCE_READ, Cap.EVIDENCE_CREATE, Cap.EXPERIMENT_READ, Cap.EXPERIMENT_REGISTER,
        Cap.ANALYSIS_READ, Cap.ANALYSIS_COMPUTE, Cap.LITERATURE_READ, Cap.LITERATURE_WRITE,
        Cap.DECISION_CREATE, Cap.DECISION_READ, Cap.CONFLICT_CREATE, Cap.CONFLICT_READ,
        Cap.TIMELINE_READ, Cap.TIMELINE_WRITE, Cap.REVIEW_READ, Cap.AUDIT_CREATE,
        Cap.AUDIT_READ, Cap.PAPER_READ, Cap.SKILL_READ,
    ),
    "literature_researcher": _caps(
        Cap.STATE_READ, Cap.LITERATURE_READ, Cap.LITERATURE_WRITE, Cap.LITERATURE_PROVIDER_USE,
        Cap.LITERATURE_NOVELTY_AUDIT, Cap.CLAIM_READ, Cap.EVIDENCE_READ, Cap.EVIDENCE_CREATE,
        Cap.AUDIT_CREATE, Cap.AUDIT_READ, Cap.TIMELINE_READ, Cap.TIMELINE_WRITE,
        Cap.DECISION_READ, Cap.CONFLICT_CREATE, Cap.REVIEW_READ,
    ),
    "novelty_auditor": _caps(
        Cap.STATE_READ, Cap.LITERATURE_READ, Cap.LITERATURE_NOVELTY_AUDIT,
        Cap.LITERATURE_PROVIDER_USE, Cap.AUDIT_CREATE, Cap.AUDIT_READ, Cap.EVIDENCE_READ,
        Cap.CLAIM_READ, Cap.TIMELINE_READ, Cap.TIMELINE_WRITE, Cap.CONFLICT_CREATE,
    ),
    "experiment_designer": _caps(
        Cap.STATE_READ, Cap.EXPERIMENT_READ, Cap.EXPERIMENT_REGISTER, Cap.CLAIM_READ,
        Cap.EVIDENCE_READ, Cap.ANALYSIS_READ, Cap.LITERATURE_READ, Cap.TIMELINE_READ,
        Cap.TIMELINE_WRITE, Cap.DECISION_CREATE, Cap.DECISION_READ, Cap.CONFLICT_READ,
    ),
    "experiment": _caps(
        Cap.STATE_READ, Cap.EXPERIMENT_READ, Cap.EXPERIMENT_REGISTER, Cap.EXPERIMENT_RUN,
        Cap.EXPERIMENT_UPDATE, Cap.EVIDENCE_READ, Cap.EVIDENCE_CREATE, Cap.EVIDENCE_RAW_WRITE,
        Cap.ANALYSIS_READ, Cap.CLAIM_READ, Cap.TIMELINE_READ, Cap.TIMELINE_WRITE,
        Cap.DECISION_READ, Cap.CONFLICT_CREATE,
    ),
    "analysis": _caps(
        Cap.STATE_READ, Cap.ANALYSIS_READ, Cap.ANALYSIS_COMPUTE, Cap.EVIDENCE_READ,
        Cap.EVIDENCE_CREATE, Cap.EVIDENCE_VERIFY, Cap.EXPERIMENT_READ, Cap.CLAIM_READ,
        Cap.TIMELINE_READ, Cap.TIMELINE_WRITE, Cap.CONFLICT_CREATE, Cap.CONFLICT_READ,
        Cap.AUDIT_CREATE, Cap.DECISION_READ,
    ),
    "mechanism_auditor": _caps(
        Cap.STATE_READ, Cap.AUDIT_CREATE, Cap.AUDIT_READ, Cap.EVIDENCE_READ,
        Cap.EXPERIMENT_READ, Cap.ANALYSIS_READ, Cap.CLAIM_READ, Cap.TIMELINE_READ,
        Cap.TIMELINE_WRITE, Cap.CONFLICT_CREATE, Cap.DECISION_READ, Cap.LITERATURE_READ,
    ),
    "claim_manager": _caps(
        Cap.STATE_READ, Cap.STATE_TRANSITION_REQUEST, Cap.CLAIM_READ, Cap.CLAIM_PROPOSE,
        Cap.CLAIM_REJECT, Cap.CLAIM_SUPERSEDE, Cap.EVIDENCE_READ, Cap.ANALYSIS_READ,
        Cap.EXPERIMENT_READ, Cap.AUDIT_CREATE, Cap.AUDIT_READ, Cap.TIMELINE_READ,
        Cap.TIMELINE_WRITE, Cap.CONFLICT_CREATE, Cap.CONFLICT_READ, Cap.DECISION_READ,
        Cap.LITERATURE_READ,
    ),
    "paper_writer": _caps(
        Cap.STATE_READ, Cap.CLAIM_READ, Cap.EVIDENCE_READ, Cap.ANALYSIS_READ,
        Cap.LITERATURE_READ, Cap.PAPER_READ_APPROVED, Cap.PAPER_READ, Cap.PAPER_COMPILE,
        Cap.TIMELINE_READ, Cap.DECISION_READ, Cap.AUDIT_READ,
    ),
    "paper_auditor": _caps(
        Cap.STATE_READ, Cap.PAPER_READ_APPROVED, Cap.PAPER_READ, Cap.PAPER_STYLE_AUDIT,
        Cap.AUDIT_CREATE, Cap.AUDIT_READ, Cap.CLAIM_READ, Cap.EVIDENCE_READ,
        Cap.LITERATURE_READ, Cap.TIMELINE_READ, Cap.ANALYSIS_READ,
    ),
    "red_team": _caps(
        Cap.STATE_READ, Cap.AUDIT_CREATE, Cap.AUDIT_READ, Cap.CLAIM_READ, Cap.EVIDENCE_READ,
        Cap.EXPERIMENT_READ, Cap.ANALYSIS_READ, Cap.LITERATURE_READ, Cap.PAPER_READ_APPROVED,
        Cap.PAPER_READ, Cap.TIMELINE_READ, Cap.CONFLICT_READ, Cap.DECISION_READ,
    ),
    "skill_discovery": _caps(
        Cap.STATE_READ, Cap.SKILL_READ, Cap.SKILL_DISCOVER, Cap.SKILL_PROPOSE,
        Cap.LITERATURE_READ, Cap.TIMELINE_READ, Cap.TIMELINE_WRITE, Cap.AUDIT_CREATE,
    ),
    "skill_evaluator": _caps(
        Cap.STATE_READ, Cap.SKILL_READ, Cap.SKILL_EVALUATE, Cap.AUDIT_CREATE, Cap.AUDIT_READ,
        Cap.TIMELINE_READ, Cap.TIMELINE_WRITE,
    ),
    "skill_synthesizer": _caps(
        Cap.STATE_READ, Cap.SKILL_READ, Cap.SKILL_PROPOSE, Cap.SKILL_EVALUATE,
        Cap.AUDIT_CREATE, Cap.TIMELINE_READ, Cap.TIMELINE_WRITE,
    ),
    "system": _caps(Cap.STATE_READ, Cap.EVENT_APPEND, Cap.EVENT_VERIFY, Cap.TIMELINE_READ),
}

#: Capabilities granted to an imported/external skill package. Proposals only.
EXTERNAL_SKILL_CAPABILITIES: FrozenSet[Cap] = _caps(
    Cap.PROPOSE_PROCEDURE, Cap.PROPOSE_RECOMMENDATION, Cap.PROPOSE_ANALYSIS, Cap.PROPOSE_DRAFT
)

#: Capabilities that only a human principal may hold.
HUMAN_ONLY_CAPABILITIES: FrozenSet[Cap] = frozenset(
    {
        Cap.STATE_TRANSITION_APPROVE,
        Cap.STATE_CORE_WRITE,
        Cap.CLAIM_APPROVE,
        Cap.SKILL_ACTIVATE,
        Cap.SKILL_INSTALL,
        Cap.SKILL_DEPRECATE,
        Cap.CONFLICT_RESOLVE,
        Cap.REVIEW_DECIDE,
        Cap.STATE_ADMIN,
    }
)


class ResourceClass(StrEnum):
    """Resources whose contents may never be overwritten or removed."""

    RAW_EVIDENCE = "raw_evidence"
    RAW_ARTIFACT = "raw_artifact"
    EVENT_LOG = "event_log"
    PROVENANCE = "provenance"
    REJECTED_CLAIM = "rejected_claim"
    FAILED_EXPERIMENT = "failed_experiment"
    AUDIT_RECORD = "audit_record"
    DECISION_RECORD = "decision_record"
    TIMELINE = "timeline"


#: Human-readable justification for each guard, surfaced in error messages.
GUARD_REASONS: Mapping[ResourceClass, str] = {
    ResourceClass.RAW_EVIDENCE: "raw observations are the ground truth of the project",
    ResourceClass.RAW_ARTIFACT: "artifact bytes are what every number traces back to",
    ResourceClass.EVENT_LOG: "the event log is the replayable history of the project",
    ResourceClass.PROVENANCE: "provenance written at record time cannot be rewritten later",
    ResourceClass.REJECTED_CLAIM: "a rejected claim must remain explainable (invariant 10)",
    ResourceClass.FAILED_EXPERIMENT: "failed experiments are results; deleting them is data loss",
    ResourceClass.AUDIT_RECORD: "an audit is a record of what was found at a point in time",
    ResourceClass.DECISION_RECORD: "decisions explain the past; they are never rewritten",
    ResourceClass.TIMELINE: "the research timeline is append-only",
}


def assert_mutable(resource: ResourceClass, operation: str, principal: Principal | None = None) -> None:
    """Raise for any attempt to mutate an append-only resource.

    Deliberately independent of the principal: capability checks happen first, and this guard
    then applies to *everyone*, including the human and the kernel itself.
    """
    reason = GUARD_REASONS.get(resource, "the resource is append-only")
    who = principal.name if principal else "kernel"
    raise ImmutableResourceError(
        f"{operation} refused on {resource.value!r} for {who!r}: {reason}. "
        "Append a new record instead, and link it with `supersedes`/`derived_from`.",
        principal=who,
        resource=resource.value,
    )


def is_immutable(resource: ResourceClass) -> bool:
    return True


class PolicyEngine:
    """The single place where permissions are decided."""

    def __init__(self) -> None:
        self._principals: dict[str, Principal] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._principals["human"] = Principal(
            name="human", kind=PrincipalKind.HUMAN, capabilities=ALL_CAPABILITIES
        )
        for name, caps in AGENT_CAPABILITIES.items():
            self._principals[name] = Principal(
                name=name, kind=PrincipalKind.AGENT, capabilities=caps, agent_type=name
            )

    def register(self, principal: Principal) -> Principal:
        self._principals[principal.name] = principal
        return principal

    def get(self, name: str) -> Principal:
        return self._principals.get(name) or Principal(name=name, kind=PrincipalKind.AGENT)

    def agent(self, agent_type: str, *, name: str | None = None) -> Principal:
        caps = AGENT_CAPABILITIES.get(agent_type)
        if caps is None:
            raise PermissionDenied(f"unknown agent type {agent_type!r}", principal=agent_type)
        return Principal(
            name=name or agent_type, kind=PrincipalKind.AGENT, capabilities=caps, agent_type=agent_type
        )

    def external_skill(self, skill_id: str, *, extra: Iterable[Cap] = ()) -> Principal:
        """External skills are untrusted by default: they may only propose."""
        caps = EXTERNAL_SKILL_CAPABILITIES | frozenset(extra)
        forbidden = caps & ALL_CAPABILITIES.difference(EXTERNAL_SKILL_CAPABILITIES)
        if forbidden:
            raise PermissionDenied(
                f"external skill {skill_id!r} may not hold capabilities "
                f"{sorted(c.value for c in forbidden)}",
                principal=skill_id,
            )
        return Principal(
            name=f"external_skill:{skill_id}",
            kind=PrincipalKind.EXTERNAL_SKILL,
            capabilities=caps,
            labels={"skill_id": skill_id},
        )

    def require(
        self,
        principal: Principal,
        capability: Cap | str,
        resource: str | None = None,
        *,
        task_allowed: Iterable[str] | None = None,
        task_forbidden: Iterable[str] = (),
    ) -> None:
        """Full check: capability, then task boundary."""
        cap = Cap(capability) if isinstance(capability, str) else capability
        principal.require(cap, resource)

        if task_forbidden and cap.value in set(task_forbidden):
            raise TaskBoundaryViolation(
                f"capability {cap.value!r} is forbidden in the active task",
                principal=principal.name,
                resource=resource,
            )
        if task_allowed:
            allowed = set(task_allowed)
            if allowed and cap.value not in allowed:
                raise TaskBoundaryViolation(
                    f"capability {cap.value!r} is outside the active task's allowed_actions "
                    f"{sorted(allowed)}; create a new task instead of widening this one",
                    principal=principal.name,
                    resource=resource,
                )

    def describe_table(self) -> list[tuple[str, int, list[str]]]:
        rows: list[tuple[str, int, list[str]]] = []
        for name, principal in sorted(self._principals.items()):
            rows.append(
                (name, len(principal.capabilities), sorted(c.value for c in principal.capabilities))
            )
        return rows
