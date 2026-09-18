"""Experiment objects — the deterministic facts of the research.

An experiment record is the *only* legitimate origin of a number. Everything downstream
(analysis, evidence, claim, paper) cites one.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from .common import (
    ArtifactRef,
    EvidenceLevel,
    ExperimentStatus,
    Provenance,
    RosModel,
    StrEnum,
    new_id,
    sha256_json,
    utcnow,
)


class VariableKind(StrEnum):
    CATEGORICAL = "CATEGORICAL"
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    ORDINAL = "ORDINAL"


class Variable(RosModel):
    name: str = Field(min_length=1)
    kind: VariableKind = VariableKind.CATEGORICAL
    levels: list[str] = Field(default_factory=list)
    unit: str | None = None
    definition: str | None = None
    is_manipulated: bool = Field(
        default=False, description="True for treatment variables the experimenter sets."
    )


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"
    TWO_SIDED = "TWO_SIDED"


class MetricRef(RosModel):
    """A metric *with its definition*. Undefined metrics are a reproducibility hazard."""

    name: str = Field(min_length=1)
    definition: str = Field(
        default="",
        description="Exact computation, incl. split, averaging order and any truncation.",
    )
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    unit: str | None = None
    computed_from: list[str] = Field(
        default_factory=list, description="Artifacts or result ids this metric is derived from."
    )
    contamination_risks: list[str] = Field(default_factory=list)

    @property
    def is_defined(self) -> bool:
        return bool(self.definition.strip())


class Arm(RosModel):
    """One condition of the experiment (treatment or control)."""

    name: str = Field(min_length=1)
    description: str = ""
    config_patch: dict[str, object] = Field(default_factory=dict)
    matched_on: list[str] = Field(
        default_factory=list,
        description="Variables held identical to the comparison arm — the basis of a fair control.",
    )
    is_control: bool = False


class TrainingBudget(RosModel):
    steps: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    epochs: float | None = Field(default=None, ge=0)
    wall_clock_seconds: float | None = Field(default=None, ge=0)
    flops: float | None = Field(default=None, ge=0)
    hardware: str | None = None
    note: str | None = None

    def summary(self) -> str:
        parts = []
        for label, value in (
            ("steps", self.steps),
            ("tokens", self.tokens),
            ("epochs", self.epochs),
            ("wall", self.wall_clock_seconds),
            ("flops", self.flops),
        ):
            if value is not None:
                parts.append(f"{label}={value}")
        return ", ".join(parts) or "unspecified"


class ResultSummary(RosModel):
    """Point estimates attached to the experiment.

    ``headline`` is free text for humans; ``values`` must mirror an analysis artifact.
    The paper compiler never reads this directly for numerals — it reads the analysis artifact.
    """

    headline: str = ""
    values: dict[str, float] = Field(default_factory=dict)
    source_analysis_ids: list[str] = Field(default_factory=list)
    source_artifact: ArtifactRef | None = None
    notes: list[str] = Field(default_factory=list)
    unexpected: list[str] = Field(default_factory=list)


class ExperimentDesign(RosModel):
    """Declared design facts. Evidence level is *derived* from these, never from prose.

    Every flag here is a claim about what was actually done, and each one is what unlocks a
    rung of the evidence ladder (see :func:`assess_experiment_level`).
    """

    has_control: bool = False
    has_matched_conditions: bool = Field(
        default=False, description="Non-experimental factors held equal across arms."
    )
    has_intervention: bool = Field(
        default=False, description="The experimenter manipulates the hypothesised cause."
    )
    has_necessity_design: bool = Field(
        default=False, description="Removing/ablating the component and observing the outcome."
    )
    has_sufficiency_design: bool = Field(
        default=False, description="Forcing the component on where it would otherwise be absent."
    )
    has_rescue: bool = Field(
        default=False, description="Restoring the component recovers the effect."
    )
    settings: list[str] = Field(
        default_factory=list,
        description="Distinct settings (model families / scales / datasets) actually run.",
    )
    confounds_identified: list[str] = Field(default_factory=list)
    confounds_uncontrolled: list[str] = Field(default_factory=list)
    is_observational: bool = Field(
        default=False, description="No manipulation: associations only, never causation."
    )

    def ladder_position(self) -> str:
        return assess_experiment_level_from_design(self).value


def assess_experiment_level_from_design(design: ExperimentDesign) -> EvidenceLevel:
    """Deterministic mapping design -> *maximum* evidence level.

    This is deliberately conservative and monotone: you cannot claim intervention evidence
    by describing an observational study in strong words, because the prose is not an input.
    """
    if design.has_rescue:
        return EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY
    if design.has_necessity_design or design.has_sufficiency_design:
        return EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY
    if len(set(design.settings)) >= 2 and (design.has_intervention or design.has_control):
        return EvidenceLevel.L6_CROSS_SETTING_REPLICATION
    if design.has_intervention:
        return EvidenceLevel.L4_INTERVENTION
    if design.has_control and design.has_matched_conditions:
        return EvidenceLevel.L3_CONTROLLED
    if design.has_control:
        return EvidenceLevel.L2_REPRODUCED
    if design.is_observational:
        return EvidenceLevel.L1_OBSERVATION
    return EvidenceLevel.L1_OBSERVATION


class Experiment(RosModel):
    """A registered experiment. Failed experiments are retained forever (status changes only)."""

    experiment_id: str = Field(default_factory=lambda: new_id("experiment"))
    title: str = Field(min_length=1)
    research_question: str = ""
    hypothesis: str = ""
    claim_ids: list[str] = Field(default_factory=list)
    parent_experiment_id: str | None = None

    independent_variables: list[Variable] = Field(default_factory=list)
    dependent_variables: list[Variable] = Field(default_factory=list)
    treatment: Arm | None = None
    control: Arm | None = None
    matched_conditions: list[str] = Field(default_factory=list)
    design: ExperimentDesign = Field(default_factory=ExperimentDesign)

    model: str | None = None
    dataset: str | None = None
    dataset_hash: str | None = None
    scale: str | None = None
    optimizer: str | None = None
    learning_rate: float | None = Field(default=None)
    seeds: list[int] = Field(default_factory=list)
    n: int | None = Field(default=None, ge=0, description="Number of runs/units per arm.")
    training_budget: TrainingBudget = Field(default_factory=TrainingBudget)

    metrics: list[MetricRef] = Field(default_factory=list)

    code_commit: str | None = None
    code_dirty: bool | None = None
    config: dict[str, object] = Field(default_factory=dict)
    config_hash: str | None = None

    raw_artifacts: list[ArtifactRef] = Field(default_factory=list)
    analysis_artifacts: list[ArtifactRef] = Field(default_factory=list)
    result: ResultSummary | None = None
    statistics: dict[str, float] = Field(default_factory=dict)
    analysis_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    limitations: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    unexpected_observations: list[str] = Field(default_factory=list)

    status: ExperimentStatus = ExperimentStatus.PLANNED
    provenance: Provenance = Field(default_factory=Provenance)
    task_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: str = "human"
    source_refs: list[str] = Field(
        default_factory=list,
        description="Imported origin paths (set by Research Import so nothing appears from nowhere).",
    )
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Experiment":
        if self.status is ExperimentStatus.FAILED and not self.failure_reason:
            raise ValueError(
                f"experiment {self.experiment_id}: FAILED requires failure_reason — failed runs "
                "are first-class results, not gaps"
            )
        if self.status in (ExperimentStatus.COMPLETED,) and not self.seeds:
            raise ValueError(
                f"experiment {self.experiment_id}: a completed experiment must record at least one seed"
            )
        if self.status is ExperimentStatus.COMPLETED and self.n is None:
            raise ValueError(
                f"experiment {self.experiment_id}: a completed experiment must record n"
            )
        return self

    # ------------------------------------------------------------------ derivations

    def evidence_level(self) -> EvidenceLevel:
        """Derived, deterministic evidence ceiling for this experiment."""
        return assess_experiment_level_from_design(self.design)

    def metric(self, name: str) -> MetricRef | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def undefined_metrics(self) -> list[str]:
        return [m.name for m in self.metrics if not m.is_defined]

    def identity_signature(self) -> str:
        """Hash of the *design* identity, used to decide whether two experiments are independent.

        Two experiments with the same signature differ only by seed, so they cannot by
        themselves justify ``ROBUST``.
        """
        return sha256_json(
            {
                "design": self.design.model_dump(mode="json"),
                "model": self.model,
                "dataset": self.dataset,
                "scale": self.scale,
                "optimizer": self.optimizer,
                "config": self.config,
                "budget": self.training_budget.model_dump(mode="json"),
            }
        )

    def is_independent_of(self, other: "Experiment") -> bool:
        """Independence means a *different design*, not merely a different seed.

        Same design + different seed is a **replication** (:meth:`is_replication_of`): useful
        evidence of reproducibility, but not on its own a second independent result.
        """
        if self.experiment_id == other.experiment_id:
            return False
        return self.identity_signature() != other.identity_signature()

    def is_replication_of(self, other: "Experiment") -> bool:
        """Same design identity with disjoint seeds: a genuine replication attempt."""
        if self.experiment_id == other.experiment_id:
            return False
        if self.identity_signature() != other.identity_signature():
            return False
        if not self.seeds or not other.seeds:
            return False
        return set(self.seeds).isdisjoint(set(other.seeds))

    def settings(self) -> tuple[str | None, str | None, str | None]:
        """(model, scale, dataset) — the tuple that distinguishes "another setting"."""
        return (self.model, self.scale, self.dataset)
