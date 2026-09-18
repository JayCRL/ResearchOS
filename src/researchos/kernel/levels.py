"""Deterministic evidence-level derivation.

Prose never sets an evidence level. The level comes from the *declared design* of the experiments
that produced the evidence, so writing more confidently cannot raise it. Everything in this module
is a pure function over models, which is what makes it testable and auditable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..models.analysis import Analysis, StatResult
from ..models.common import EvidenceLevel, ExperimentStatus
from ..models.evidence import Evidence
from ..models.experiment import Experiment, assess_experiment_level_from_design

#: Ordered ladder, index == rank.
LADDER: tuple[EvidenceLevel, ...] = (
    EvidenceLevel.L0_IDEA,
    EvidenceLevel.L1_OBSERVATION,
    EvidenceLevel.L2_REPRODUCED,
    EvidenceLevel.L3_CONTROLLED,
    EvidenceLevel.L4_INTERVENTION,
    EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY,
    EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
)

#: Evidence-level ceiling implied by each evidence maturity type.
CEILING_BY_EVIDENCE_TYPE: dict[str, EvidenceLevel] = {
    "RAW": EvidenceLevel.L1_OBSERVATION,
    "ANALYZED": EvidenceLevel.L2_REPRODUCED,
    "VERIFIED": EvidenceLevel.L3_CONTROLLED,
    "INTERPRETED": EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
    "CLAIMED": EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
}


def level_at(rank: int) -> EvidenceLevel:
    return LADDER[max(0, min(rank, len(LADDER) - 1))]


def lower(a: EvidenceLevel, b: EvidenceLevel) -> EvidenceLevel:
    """The weaker of two levels — used for clamping."""
    return a if a.rank <= b.rank else b


def max_level(levels: Iterable[EvidenceLevel]) -> EvidenceLevel:
    materialised = list(levels)
    if not materialised:
        return EvidenceLevel.L0_IDEA
    return max(materialised, key=lambda level: level.rank)


def level_of_experiment(experiment: Experiment) -> EvidenceLevel:
    """The highest level an experiment's *design* can support, regardless of outcome."""
    return assess_experiment_level_from_design(experiment.design)


def completed_level_of_experiment(experiment: Experiment) -> EvidenceLevel:
    """Design level, capped by what actually happened."""
    if experiment.status is ExperimentStatus.PLANNED:
        return EvidenceLevel.L0_IDEA
    if experiment.status in (ExperimentStatus.FAILED, ExperimentStatus.ABORTED):
        # A failed experiment still teaches us something (an observation), never more.
        return lower(EvidenceLevel.L1_OBSERVATION, level_of_experiment(experiment))
    return level_of_experiment(experiment)


def level_of_evidence(
    evidence: Evidence,
    *,
    experiment: Experiment | None = None,
    analysis: Analysis | None = None,
) -> EvidenceLevel:
    """Derive the level for one evidence record.

    The evidence's *declared* level is capped by what its sources can actually support. This is
    where "we observed a correlation" cannot be written up as an intervention.
    """
    ceiling = CEILING_BY_EVIDENCE_TYPE.get(evidence.evidence_type.value, EvidenceLevel.L0_IDEA)
    if experiment is not None:
        ceiling = lower(ceiling, completed_level_of_experiment(experiment))
    if evidence.evidence_type.value == "VERIFIED" and analysis is None:
        # "verified" is meaningless without the analysis artifact it verifies
        ceiling = lower(ceiling, EvidenceLevel.L2_REPRODUCED)
    if evidence.evidence_type.value == "ANALYZED" and analysis is not None:
        ceiling = lower(ceiling, level_of_analysis(analysis))
    return level_at(min(evidence.evidence_level.rank, ceiling.rank))


def is_statistically_informative(result: StatResult) -> bool:
    """A result counts towards REPRODUCED-or-better only with n and a dispersion estimate."""
    return bool(result.n and result.n > 1 and (result.std is not None or result.se is not None))


def level_of_analysis(analysis: Analysis) -> EvidenceLevel:
    if not analysis.results:
        return EvidenceLevel.L0_IDEA
    informative = any(is_statistically_informative(r) for r in analysis.results)
    return EvidenceLevel.L2_REPRODUCED if informative else EvidenceLevel.L1_OBSERVATION


def aggregate_level(
    evidences: Sequence[Evidence],
    experiments: dict[str, Experiment] | None = None,
    analyses: dict[str, Analysis] | None = None,
) -> EvidenceLevel:
    """Highest level jointly supported by a set of evidence records."""
    experiments = experiments or {}
    analyses = analyses or {}
    levels: list[EvidenceLevel] = []
    for evidence in evidences:
        experiment = experiments.get(evidence.source_experiment) if evidence.source_experiment else None
        analysis = analyses.get(evidence.source_analysis) if evidence.source_analysis else None
        levels.append(level_of_evidence(evidence, experiment=experiment, analysis=analysis))
    return max_level(levels)


def cross_setting_supported(experiments: Sequence[Experiment]) -> bool:
    """True when at least two *distinct settings* produced evidence (model/scale/dataset differ)."""
    settings: set[tuple[str | None, str | None, str | None]] = set()
    for experiment in experiments:
        settings.add((experiment.model, experiment.scale, experiment.dataset))
    return len(settings) >= 2


def independent_experiments(experiments: Sequence[Experiment]) -> list[Experiment]:
    """A maximal set of experiments with pairwise *different* designs.

    Two runs that share a design identity are replications of each other, not independent
    evidence; they are counted by :func:`replication_count` instead.
    """
    chosen: list[Experiment] = []
    seen: set[str] = set()
    for candidate in experiments:
        signature = candidate.identity_signature()
        if signature not in seen:
            seen.add(signature)
            chosen.append(candidate)
    return chosen


def replication_count(experiments: Sequence[Experiment]) -> int:
    """How many distinct (design identity, seed) runs exist — the honest "how many times" number."""
    pairs: set[tuple[str, int]] = set()
    for experiment in experiments:
        for seed in experiment.seeds or [-1]:
            pairs.add((experiment.identity_signature(), seed))
    return len(pairs)
