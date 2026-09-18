"""Agent OS — fifteen capability-scoped agents over the kernel.

See :mod:`researchos.agents.base` for the design rule: agents propose, the kernel commits.
"""

from .base import Agent, Proposal
from .registry import (
    AGENT_CLASSES,
    AnalysisAgent,
    ClaimManager,
    ExperimentAgent,
    ExperimentDesigner,
    LiteratureResearcher,
    MechanismAuditorAgent,
    NoveltyAuditorAgent,
    PaperAuditor,
    PaperWriter,
    RedTeamAgent,
    ResearchImporter,
    ResearchPlanner,
    SkillDiscoveryAgentWrapper,
    SkillEvaluatorAgent,
    SkillSynthesizerAgent,
    agent_table,
    build_agents,
)

__all__ = [
    "AGENT_CLASSES",
    "Agent",
    "AnalysisAgent",
    "ClaimManager",
    "ExperimentAgent",
    "ExperimentDesigner",
    "LiteratureResearcher",
    "MechanismAuditorAgent",
    "NoveltyAuditorAgent",
    "PaperAuditor",
    "PaperWriter",
    "Proposal",
    "RedTeamAgent",
    "ResearchImporter",
    "ResearchPlanner",
    "SkillDiscoveryAgentWrapper",
    "SkillEvaluatorAgent",
    "SkillSynthesizerAgent",
    "agent_table",
    "build_agents",
]
