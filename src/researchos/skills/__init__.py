"""The Skill Intelligence Layer: registry, sandbox, benchmark, lifecycle, discovery.

The five modules are one plane and are re-exported here so a caller never has to guess which file a
gate lives in:

* :class:`~researchos.skills.registry.SkillRegistry` — the single door in, and the deterministic
  router out. Registration is a claim; routing only ever returns an ``ACTIVE``, usable skill.
* :class:`~researchos.skills.sandbox.SkillSandbox` — static/behavioural inspection over *text* plus
  the proposal-only capability ceiling. Nothing in this plane ever executes skill code.
* :class:`~researchos.skills.benchmark.BenchmarkHarness` — fixed capability tasks in, a weighted-F1
  verdict out, with ``hard_failed``/regression gates on :class:`~researchos.models.BenchmarkRun`.
* :class:`~researchos.skills.lifecycle.SkillLifecycle` — every status change, one rung at a time,
  with activation and deprecation reserved to a human principal.
* :class:`~researchos.skills.discovery.SkillDiscoveryAgent` — evidence-driven capability gaps and
  candidate sourcing. Discovery can register ``DISCOVERED``/``UNTRUSTED`` cards and nothing more.

Layering: this package imports ``researchos.models`` and ``researchos.kernel`` only, and never
imports ``researchos.llm`` — the skill plane's verdicts are deterministic by construction.
"""

from .benchmark import (
    FINDING_WEIGHTS,
    BenchmarkHarness,
    SkillRunner,
    hard_fail,
    score_outcome,
)
from .benchmark_tasks import (
    ALL_FINDING_CODES,
    FINDING_CODES_BY_SUITE,
    default_catalogue,
    deterministic_task_id,
    seed_default_tasks,
)
from .discovery import (
    CAPABILITY_AREAS,
    CAPABILITY_QUERY_TEMPLATES,
    FINDING_CODE_CAPABILITIES,
    SkillDiscoveryAgent,
)
from .evolution import (
    EvolutionResult,
    SkillEvolution,
    deterministic_runner,
    quality_from_benchmark,
)
from .lifecycle import LEGAL_TRANSITIONS, SkillLifecycle
from .registry import SkillRegistry
from .sandbox import (
    DEFAULT_DENYLIST,
    PROTECTED_OPERATION_CODES,
    PROTECTED_PATH_PATTERNS,
    SkillSandbox,
)

__all__ = [
    "ALL_FINDING_CODES",
    "BenchmarkHarness",
    "CAPABILITY_AREAS",
    "CAPABILITY_QUERY_TEMPLATES",
    "DEFAULT_DENYLIST",
    "EvolutionResult",
    "FINDING_CODES_BY_SUITE",
    "FINDING_CODE_CAPABILITIES",
    "FINDING_WEIGHTS",
    "LEGAL_TRANSITIONS",
    "PROTECTED_OPERATION_CODES",
    "PROTECTED_PATH_PATTERNS",
    "SkillDiscoveryAgent",
    "SkillEvolution",
    "SkillLifecycle",
    "SkillRegistry",
    "SkillRunner",
    "SkillSandbox",
    "default_catalogue",
    "deterministic_runner",
    "deterministic_task_id",
    "hard_fail",
    "quality_from_benchmark",
    "score_outcome",
    "seed_default_tasks",
]
