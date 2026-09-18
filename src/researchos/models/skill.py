"""Skill objects — the Skill Meta-System's data model.

A skill is a *declared capability* with a measured quality profile, provenance to a commit, a
sandbox verdict and a benchmark history. "The code runs" is deliberately not a metric.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field, model_validator

from .common import RosModel, Severity, SkillStatus, StrEnum, clampf, new_id, sha256_json, utcnow


class SkillSource(StrEnum):
    WEB = "WEB"
    GITHUB = "GITHUB"
    ARXIV = "ARXIV"
    DOCS = "DOCS"
    COMMUNITY_REGISTRY = "COMMUNITY_REGISTRY"
    LOCAL = "LOCAL"
    SYNTHESIZED = "SYNTHESIZED"


class SkillTrust(StrEnum):
    UNTRUSTED = "UNTRUSTED"
    SANDBOXED = "SANDBOXED"
    BENCHMARKED = "BENCHMARKED"
    TRUSTED = "TRUSTED"


class SkillQualityMetrics(RosModel):
    """Weighted capability profile. ``extra='forbid'`` is inherited, so a skill cannot smuggle in
    a ``code_runs`` metric and call itself high quality."""

    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_calibration: float = Field(default=0.0, ge=0.0, le=1.0)
    reproducibility: float = Field(default=0.0, ge=0.0, le=1.0)
    research_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_handling: float = Field(default=0.0, ge=0.0, le=1.0)
    regression_safety: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Weights sum to 1.0. Nothing here rewards "it executed successfully".
    WEIGHTS: ClassVar[dict[str, float]] = {
        "accuracy": 0.20,
        "coverage": 0.15,
        "citation_correctness": 0.15,
        "claim_calibration": 0.15,
        "reproducibility": 0.10,
        "research_relevance": 0.10,
        "failure_handling": 0.075,
        "regression_safety": 0.075,
    }

    def score(self) -> float:
        total = sum(getattr(self, k) * w for k, w in self.WEIGHTS.items())
        return round(clampf(total), 4)

    def weakest(self, k: int = 3) -> list[tuple[str, float]]:
        pairs = [(name, getattr(self, name)) for name in self.WEIGHTS]
        return sorted(pairs, key=lambda kv: kv[1])[:k]


class SkillDependency(RosModel):
    name: str
    version: str | None = None
    kind: str = "python"
    optional: bool = False
    license: str | None = None


class SkillCapability(RosModel):
    """A capability the skill declares. Declared capabilities bound what the sandbox permits."""

    name: str = Field(min_length=1)
    description: str = ""
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(
        default_factory=list,
        description="Resources it may write. Research state and raw evidence are never writable by "
        "an external skill, regardless of what is declared here.",
    )
    requires_network: bool = False
    requires_execution: bool = False
    is_advisory_only: bool = Field(
        default=True, description="True when the capability only proposes, never mutates state."
    )


class SkillCard(RosModel):
    """The canonical skill description. Everything the router needs, plus the caveats."""

    skill_id: str = Field(default_factory=lambda: new_id("skill"))
    name: str = Field(min_length=1)
    description: str = ""
    version: str = "0.0.0"

    source: SkillSource = SkillSource.LOCAL
    source_url: str | None = None
    repository: str | None = None
    commit: str | None = None
    license: str | None = None
    entrypoint: str | None = Field(
        default=None, description="Path or command inside the skill package, e.g. 'skill.py:run'."
    )
    content_hash: str | None = Field(
        default=None, description="SHA-256 of the skill's content at its pinned commit."
    )

    dependencies: list[SkillDependency] = Field(default_factory=list)
    capabilities: list[SkillCapability] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    documentation_url: str | None = None
    benchmark: dict[str, object] = Field(default_factory=dict)
    compatibility: dict[str, str] = Field(default_factory=dict)
    quality_metrics: SkillQualityMetrics = Field(default_factory=SkillQualityMetrics)
    known_failures: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)

    status: SkillStatus = SkillStatus.DISCOVERED
    trust: SkillTrust = SkillTrust.UNTRUSTED
    discovered_at: datetime = Field(default_factory=utcnow)
    discovered_by: str = "skill_discovery"
    discovery_query: str | None = None
    discovery_rationale: str | None = None
    last_benchmark_run_id: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    parent_skill_ids: list[str] = Field(
        default_factory=list, description="Set when this skill was synthesized from others."
    )
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "SkillCard":
        if self.status is SkillStatus.ACTIVE and not self.last_benchmark_run_id:
            raise ValueError(
                f"skill {self.skill_id}: cannot be ACTIVE without a benchmark run "
                "(invariant 8: activation is earned, not declared)"
            )
        if self.status is SkillStatus.ACTIVE and self.trust is SkillTrust.UNTRUSTED:
            raise ValueError(
                f"skill {self.skill_id}: an ACTIVE skill cannot be UNTRUSTED — it must pass the sandbox"
            )
        for capability in self.capabilities:
            for target in capability.writes:
                if target.startswith(("state/", "evidence/raw", ".researchos/state", "claims/")):
                    raise ValueError(
                        f"skill {self.skill_id}: declared write to protected resource {target!r}; "
                        "external skills may only propose (invariant 7)"
                    )
        return self

    def card_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    def is_usable(self) -> bool:
        return self.status in (SkillStatus.ACTIVE, SkillStatus.VERIFIED) and (
            self.trust in (SkillTrust.BENCHMARKED, SkillTrust.TRUSTED)
        )


class SkillVersion(RosModel):
    """A pinned revision of a skill, so benchmark results remain attributable."""

    skill_version_id: str = Field(default_factory=lambda: new_id("skill_version"))
    skill_id: str
    version: str
    commit: str | None = None
    content_hash: str | None = None
    card_hash: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    parent_skill_version_ids: list[str] = Field(default_factory=list)
    synthesized: bool = False
    synthesis_inputs: list[str] = Field(
        default_factory=list, description="skill ids combined, when this version was synthesized."
    )
    synthesis_rule: str | None = Field(
        default=None, description="The research-specific rule injected during synthesis."
    )
    change_summary: str = ""
    benchmark_run_ids: list[str] = Field(default_factory=list)


class BenchmarkSuite(StrEnum):
    LITERATURE = "LITERATURE"
    EXPERIMENT = "EXPERIMENT"
    MECHANISM = "MECHANISM"
    WRITING = "WRITING"
    REPRODUCIBILITY = "REPRODUCIBILITY"


class BenchmarkTask(RosModel):
    """A fixed, deterministic capability task. Tasks are fixtures, not vibes."""

    benchmark_task_id: str = Field(default_factory=lambda: new_id("benchmark_task"))
    suite: BenchmarkSuite
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    fixture: str | None = Field(default=None, description="Path to the input fixture under benchmarks/.")
    expected_criteria: list[str] = Field(
        default_factory=list, description="Programmatically checkable success criteria."
    )
    required_findings: list[str] = Field(
        default_factory=list, description="Finding codes that must appear for a pass."
    )
    forbidden_findings: list[str] = Field(
        default_factory=list, description="Finding codes that force a hard fail if produced."
    )
    hard_fail_conditions: list[str] = Field(default_factory=list)
    max_minutes: float | None = None
    weight: float = Field(default=1.0, gt=0.0)


class TaskOutcome(RosModel):
    benchmark_task_id: str
    passed: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    produced_findings: list[str] = Field(default_factory=list)
    missed_findings: list[str] = Field(default_factory=list)
    spurious_findings: list[str] = Field(default_factory=list)
    hard_failed: bool = False
    failure_mode: str | None = None
    raw_output_ref: str | None = None
    notes: list[str] = Field(default_factory=list)


class BenchmarkRun(RosModel):
    """A benchmark result. The regression gate for skill activation lives on this object."""

    benchmark_run_id: str = Field(default_factory=lambda: new_id("benchmark_run"))
    skill_id: str
    skill_version_id: str | None = None
    suite: BenchmarkSuite
    task_outcomes: list[TaskOutcome] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    suite_scores: dict[str, float] = Field(default_factory=dict)

    incumbent_skill_id: str | None = None
    incumbent_score: float | None = Field(default=None, ge=0.0, le=1.0)
    regression_passed: bool | None = None
    regression_delta: float | None = None
    regression_regressions: list[str] = Field(
        default_factory=list, description="Tasks the incumbent passed and this version fails."
    )

    sandbox_report_id: str | None = None
    deterministic: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "skill_evaluator"
    duration_seconds: float | None = None
    notes: list[str] = Field(default_factory=list)

    #: A skill must clear this to be activatable.
    ACTIVATION_SCORE: ClassVar[float] = 0.60
    #: Maximum tolerated drop versus the incumbent (negative delta = regression).
    MAX_REGRESSION_DROP: ClassVar[float] = 0.02

    def compute_score(self) -> float:
        if not self.task_outcomes:
            self.score = 0.0
            return 0.0
        total_weight = 0.0
        acc = 0.0
        for outcome in self.task_outcomes:
            weight = 1.0
            total_weight += weight
            acc += outcome.score * weight
        self.score = round(clampf(acc / total_weight), 4)
        return self.score

    def evaluate_regression(self, incumbent_score: float, incumbent_passed: set[str]) -> bool:
        """Regression is *task-level*, not just score-level: a skill that gains overall while
        breaking a previously-passing task is a regression."""
        self.incumbent_score = incumbent_score
        self.regression_delta = round(self.score - incumbent_score, 4)
        self.regression_regressions = [
            o.benchmark_task_id
            for o in self.task_outcomes
            if o.benchmark_task_id in incumbent_passed and not o.passed
        ]
        self.regression_passed = (
            self.regression_delta >= -self.MAX_REGRESSION_DROP and not self.regression_regressions
        )
        return self.regression_passed

    def activatable(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.score < self.ACTIVATION_SCORE:
            reasons.append(f"score {self.score} < {self.ACTIVATION_SCORE}")
        if self.regression_passed is False:
            reasons.append("regression against incumbent failed")
        if any(o.hard_failed for o in self.task_outcomes):
            reasons.append("a benchmark task hard-failed")
        if not self.task_outcomes:
            reasons.append("no benchmark tasks were executed")
        return (not reasons), reasons


class SandboxFinding(RosModel):
    code: str
    severity: Severity
    message: str
    location: str | None = None
    snippet: str | None = None


class SandboxReport(RosModel):
    """Static and behavioural sandbox verdict for an external skill."""

    sandbox_report_id: str = Field(default_factory=lambda: new_id("audit"))
    skill_id: str
    passed: bool = False
    findings: list[SandboxFinding] = Field(default_factory=list)
    capability_ceiling: list[str] = Field(default_factory=list)
    denied_operations: list[str] = Field(default_factory=list)
    network_used: bool = False
    files_written: list[str] = Field(default_factory=list)
    protected_paths_touched: list[str] = Field(default_factory=list)
    static_analysis_only: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "skill_evaluator"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "SandboxReport":
        if self.protected_paths_touched and self.passed:
            raise ValueError(
                "a sandbox run that touched protected research paths cannot pass "
                "(raw evidence, research state, provenance, audits)"
            )
        return self


class SkillGap(RosModel):
    """A capability the system needed and did not have. The trigger of the evolution loop."""

    gap_id: str = Field(default_factory=lambda: new_id("gap"))
    description: str = Field(min_length=1)
    missing_capability: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list, description="Failure/audit ids that exposed it.")
    severity: Severity = Severity.MEDIUM
    occurrences: int = Field(default=1, ge=1)
    first_seen: datetime = Field(default_factory=utcnow)
    last_seen: datetime = Field(default_factory=utcnow)
    status: str = Field(default="OPEN", description="OPEN | SEARCHING | ADDRESSED | WONT_FIX")
    addressed_by_skill_id: str | None = None
    search_queries: list[str] = Field(default_factory=list)
