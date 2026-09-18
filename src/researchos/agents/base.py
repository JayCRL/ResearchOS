"""Agent OS — fifteen capability-scoped agents on top of the kernel.

An agent in ResearchOS is **not** an autonomous loop with a system prompt. It is a named principal
with a fixed capability set (see ``kernel.permissions.AGENT_CAPABILITIES``) plus a small set of
*proposal* methods. Concretely:

* an agent can always *propose* — record a finding, draft a claim, suggest a transition;
* an agent can only *commit* what its capabilities allow;
* the kernel refuses the rest with an explicit error naming the missing capability.

This design is what makes "the agent went off and redefined the project" impossible rather than
merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal


@dataclass
class Proposal:
    """Something an agent wants to happen. Nothing happens until the kernel accepts it."""

    kind: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    requires: tuple[str, ...] = ()
    status: str = "PROPOSED"
    created_refs: list[str] = field(default_factory=list)

    def describe(self) -> str:
        needs = f" (requires {', '.join(self.requires)})" if self.requires else ""
        return f"[{self.kind}] {self.summary}{needs}"


class Agent:
    """Base class: a named principal plus helpers for proposing without committing."""

    #: Agent type, matching a key in ``AGENT_CAPABILITIES``.
    agent_type: str = "agent"
    #: One-line description of what this agent is for.
    purpose: str = ""

    def __init__(self, kernel: ResearchKernel, principal: Principal | None = None) -> None:
        self.kernel = kernel
        self.principal = principal or kernel.agent(self.agent_type)
        self.proposals: list[Proposal] = []

    # ------------------------------------------------------------------ capability surface

    def capabilities(self) -> list[str]:
        return sorted(c.value for c in self.principal.capabilities)

    def can(self, capability: Cap | str) -> bool:
        return self.principal.has(capability)

    def require(self, capability: Cap | str, resource: str = "") -> None:
        self.principal.require(capability, resource or None)

    def describe(self) -> Mapping[str, Any]:
        return {
            "agent": self.agent_type,
            "principal": self.principal.name,
            "purpose": self.purpose,
            "capabilities": self.capabilities(),
            "proposals": len(self.proposals),
        }

    # ------------------------------------------------------------------ proposal helpers

    def propose(self, proposal: Proposal) -> Proposal:
        """Record a proposal without acting on it. Proposals are how agents ask for permission."""
        self.proposals.append(proposal)
        self.kernel.events.append(
            "agent.proposal",
            actor=self.principal.name,
            payload={
                "agent": self.agent_type,
                "kind": proposal.kind,
                "summary": proposal.summary[:300],
                "requires": list(proposal.requires),
            },
        )
        return proposal

    def refused(self, action: str, reason: str) -> Proposal:
        return self.propose(
            Proposal(kind="REFUSED", summary=f"{action} refused: {reason}", rationale=reason)
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<{type(self).__name__} principal={self.principal.name!r}>"
