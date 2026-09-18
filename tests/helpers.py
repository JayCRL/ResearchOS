"""Test helpers shared across suites.

The most important one is :func:`build_supported_chain`: it fabricates a *complete, legitimate*
evidence chain (experiment → analysis → evidence → claim → SUPPORTED) so that tests which need a
paper-ready claim do not have to fake one. Faking it would defeat the purpose of the gate tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchos.claims import ClaimLifecycle
from researchos.evidence import EvidenceRegistry
from researchos.kernel import ResearchKernel
from researchos.models import (
    Analysis,
    Arm,
    ClaimStatus,
    Evidence,
    EvidenceLevel,
    EvidenceType,
    Experiment,
    ExperimentDesign,
    ExperimentStatus,
    MetricRef,
    Provenance,
    StatMethod,
    StatResult,
    TrainingBudget,
)

DIRECT_VALUES = [0.421, 0.416, 0.418]
SHUF_VALUES = [0.274, 0.269, 0.271]


@dataclass
class Chain:
    """Ids of a fully-formed evidence chain."""

    experiment_id: str
    analysis_id: str
    evidence_id: str
    claim_id: str
    raw_evidence_id: str | None = None
    result_id: str | None = None


def build_supported_chain(
    kernel: ResearchKernel,
    *,
    statement: str = "Direct writeback retains more than shuffled writeback at the 124M scale",
    scope: str = "gpt2-small, 124M, wikitext-103, seeds 0-2",
    metric: str = "retention_at_1",
) -> Chain:
    """Create a valid experiment → analysis → evidence → SUPPORTED claim chain."""
    experiment = Experiment(
        title="direct vs shufwrite",
        research_question="Does write placement affect retention?",
        hypothesis="Direct writeback retains more.",
        treatment=Arm(name="direct", is_control=False),
        control=Arm(name="shufwrite", is_control=True),
        matched_conditions=["model", "dataset", "optimizer", "learning_rate"],
        design=ExperimentDesign(
            has_control=True,
            has_matched_conditions=True,
            has_intervention=True,
            settings=["gpt2-small/124M/wikitext-103"],
        ),
        model="gpt2-small",
        dataset="wikitext-103",
        scale="124M",
        optimizer="adamw",
        learning_rate=3e-4,
        seeds=[0, 1, 2],
        n=3,
        training_budget=TrainingBudget(steps=10000),
        metrics=[MetricRef(name=metric, definition="fraction of fast weights readable one window later")],
        provenance=Provenance(code_commit="deadbeef", random_seeds=[0, 1, 2], config_hash="c0ffee"),
        status=ExperimentStatus.COMPLETED,
    )
    kernel.experiments.save(experiment)
    kernel.events.append("experiment.registered", actor="experiment", payload={"experiment_id": experiment.experiment_id})

    mean_direct = sum(DIRECT_VALUES) / len(DIRECT_VALUES)
    mean_shuf = sum(SHUF_VALUES) / len(SHUF_VALUES)
    result = StatResult(
        name=f"direct_vs_shufwrite.{metric}",
        value=round(mean_direct - mean_shuf, 6),
        mean=round(mean_direct - mean_shuf, 6),
        n=6,
        std=0.0032,
        se=0.0013,
        ci_low=0.144,
        ci_high=0.153,
        ci_level=0.95,
        effect_size=1.87,
        effect_size_kind="cohens_d",
        test=StatMethod.WELCH_TTEST,
        statistic=11.3,
        df=4.1,
        p_value=0.0004,
        alpha=0.05,
        significant=True,
        paired=False,
        group_a="direct",
        group_b="shufwrite",
        comparison_family=metric,
    )
    analysis = Analysis(
        title=f"Welch comparison of {metric}",
        question="Do the two write placements differ on retention?",
        experiment_ids=[experiment.experiment_id],
        method=StatMethod.WELCH_TTEST,
        method_description="Welch t-test over three seeds per arm",
        deterministic=True,
        results=[result],
    )
    kernel.analyses.save(analysis)
    kernel.events.append("analysis.computed", actor="analysis", payload={"analysis_id": analysis.analysis_id})

    registry = EvidenceRegistry(kernel)
    raw = Evidence(
        statement=f"raw {metric} rows for both arms across seeds 0-2",
        evidence_type=EvidenceType.RAW,
        payload={f"{metric}.direct": mean_direct, f"{metric}.shufwrite": mean_shuf},
        source_experiment=experiment.experiment_id,
        experiment_ids=[experiment.experiment_id],
        evidence_level=EvidenceLevel.L1_OBSERVATION,
    )
    registry.create(kernel.principal("experiment"), raw)

    analyzed = Evidence(
        statement=(
            f"Welch comparison of {metric} between direct and shuffled writeback: "
            f"mean difference {mean_direct - mean_shuf:.4f}"
        ),
        evidence_type=EvidenceType.ANALYZED,
        source_kind=__import__("researchos.models", fromlist=["SourceKind"]).SourceKind.VERIFIED_ANALYSIS,
        source_analysis=analysis.analysis_id,
        source_experiment=experiment.experiment_id,
        experiment_ids=[experiment.experiment_id],
        evidence_level=EvidenceLevel.L2_REPRODUCED,
        payload={
            "metric": metric,
            "mean_difference": round(mean_direct - mean_shuf, 6),
            "n": 6,
            "p_value": 0.0004,
        },
    )
    registry.create(kernel.principal("analysis"), analyzed)
    registry.verify(kernel.principal("analysis"), analyzed.evidence_id, method="recomputed from runs/metrics.csv")

    lifecycle = ClaimLifecycle(kernel)
    claim = lifecycle.create(
        kernel.principal("claim_manager"),
        statement=statement,
        scope=scope,
        evidence_ids=[analyzed.evidence_id, raw.evidence_id],
        supporting_experiments=[experiment.experiment_id],
        limitations=["three seeds only", "single model family"],
    )
    lifecycle.transition(kernel.principal("claim_manager"), claim.claim_id, ClaimStatus.HYPOTHESIS, reason="scope declared")
    lifecycle.transition(kernel.principal("claim_manager"), claim.claim_id, ClaimStatus.TESTED, reason="experiment completed")
    lifecycle.transition(kernel.human(), claim.claim_id, ClaimStatus.SUPPORTED, reason="evidence at L2 with analysis")

    return Chain(
        experiment_id=experiment.experiment_id,
        analysis_id=analysis.analysis_id,
        evidence_id=analyzed.evidence_id,
        claim_id=claim.claim_id,
        raw_evidence_id=raw.evidence_id,
        result_id=result.result_id,
    )


def add_literature_claim(
    kernel: ResearchKernel,
    *,
    title: str = "Learning to Control Fast-Weight Memories",
    statement: str = "the method maintains a fast-weight memory written at the active position",
    year: int = 1992,
    fulltext: bool = True,
    page: str = "4",
    section: str = "2",
):
    """Create a paper plus one locator-bearing literature claim (full text verified by default)."""
    from researchos.models import FulltextStatus, LiteratureClaim, LiteratureClaimKind, LiteraturePaper

    paper = LiteraturePaper(
        title=title,
        authors=["Schmidhuber, Jürgen"],
        year=year,
        venue="Neural Computation",
        bibtex_key="schmidhuber1992",
        fulltext_status=FulltextStatus.FULLTEXT_VERIFIED if fulltext else FulltextStatus.ABSTRACT_LEVEL_ONLY,
        fulltext_path="lit/papers/schmidhuber1992.pdf" if fulltext else None,
        source_pages=["4", "5"] if fulltext else [],
    )
    kernel.papers.save(paper)
    claim = LiteratureClaim(
        paper_id=paper.paper_id,
        statement=statement,
        kind=LiteratureClaimKind.MECHANISM if fulltext else LiteratureClaimKind.RESULT,
        page=page,
        section=section,
        quote="we write the fast weight at the position that produced it",
        fulltext_verified=fulltext,
        threat_to_novelty=__import__("researchos.models", fromlist=["Severity"]).Severity.MEDIUM,
    )
    kernel.literature_claims.save(claim)
    return paper, claim
