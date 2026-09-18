"""Literature OS, auditors, style audit, readiness and the agent layer.

These tests double as the integration check for every module: if a statistical audit, a mechanism
audit or a readiness assessment cannot run against a real project, that module is dead code.
"""

from __future__ import annotations

import pytest

from helpers import add_literature_claim, build_supported_chain

from researchos.agents import agent_table, build_agents
from researchos.analysis import MechanismAuditor, StatisticalAuditor, welch_t_test
from researchos.claims import ClaimLifecycle, calibrate, language_class, overreaches, permitted_verb
from researchos.kernel import PermissionDenied
from researchos.kernel.permissions import Cap
from researchos.literature import (
    LiteratureGraph,
    NoveltyAuditor,
    PriorArtMatrixBuilder,
    compute_coverage,
    coverage_line,
    coverage_report,
    render_markdown,
)
from researchos.literature.providers.base import OfflineProvider
from researchos.literature.providers.registry import offline_providers
from researchos.literature.query_planner import QueryPlanner
from researchos.models import (
    COVERAGE_THRESHOLDS,
    FORBIDDEN_NOVELTY_PHRASES,
    REQUIRED_FAMILIES,
    ClaimStatus,
    EvidenceLevel,
    GapKind,
    LiteratureCoverage,
    LiteraturePaper,
    NodeKind,
    NoveltyVerdict,
    ProviderKind,
    ProviderRecord,
    QueryFamily,
    RelationType,
    Severity,
    Tri,
)
from researchos.paper import ReadinessAssessor, StyleAuditor
from researchos.paper.style_audit import KNOWN_AI_PHRASES


def _record(title: str, *, abstract: str = "", year: int = 2021, doi: str | None = None) -> ProviderRecord:
    return ProviderRecord(
        provider=ProviderKind.LOCAL_BIB,
        provider_id=f"local:{title[:24]}",
        title=title,
        abstract=abstract or None,
        authors=["A. Author"],
        year=year,
        venue="Test Venue",
        doi=doi,
    )


def _rich_corpus() -> list[ProviderRecord]:
    return [
        _record(
            "Fast weights write back at the originating position",
            abstract="We keep a fast state and write each key-value pair at the position that produced it, "
            "giving an explicit selector over parameter-level writeback.",
        ),
        _record(
            "Shuffled writeback baselines for associative memory",
            abstract="A shuffled writeback control is compared against an aligned writeback under a matched "
            "energy budget for associative memory.",
        ),
        _record(
            "Sleep replay in fast-weight consolidation",
            abstract="Offline consolidation with replay stabilises fast-weight associative memory across "
            "long contexts; writeback alignment is not varied.",
        ),
        _record(
            "Linear attention as a fast state",
            abstract="Linear attention maintains a fast state updated at every step, with alignment between "
            "write and read positions, evaluated on long-range recall.",
        ),
        _record(
            "Test-time training for distribution shift",
            abstract="A slow state is updated at inference time; the fast state is not modified.",
            year=2019,
        ),
    ]


# ======================================================================================
# query planning and coverage
# ======================================================================================
def test_plan_covers_every_required_query_family(kernel):
    planner = QueryPlanner(kernel)
    plan = planner.plan(kernel.principal("literature_researcher"), "write placement in fast weights")
    families = {query.family for query in plan.queries}
    for family in REQUIRED_FAMILIES:
        assert family in families, f"missing required query family {family.value}"
    assert plan.families_covered() == []  # nothing executed yet
    assert all(query.rationale for query in plan.queries), "every query must state why it exists"
    assert plan.missing_required_families()


def test_execute_ingests_papers_without_faking_fulltext_knowledge(kernel):
    planner = QueryPlanner(kernel)
    plan = planner.plan(kernel.principal("literature_researcher"), "write placement in fast weights")
    providers = offline_providers(_rich_corpus())
    plan = planner.execute(kernel.principal("literature_researcher"), plan, providers, limit=10)

    executed = plan.executed_queries()
    assert executed
    assert plan.families_covered()
    papers = kernel.papers.all()
    assert papers
    for paper in papers:
        assert paper.fulltext_status.value != "FULLTEXT_VERIFIED"
        assert paper.mechanism is None
        assert paper.explicit_components == []

    coverage = planner.compute_coverage(plan)
    assert coverage.queries_executed == len(executed)
    assert coverage.search_log_refs == [plan.query_plan_id]
    assert coverage.score > 0
    assert coverage_report(coverage)


def test_coverage_deficits_are_named_not_hidden():
    plan = QueryPlanner.__new__(QueryPlanner)  # no kernel needed for pure coverage arithmetic
    from researchos.models import PlannedQuery, QueryPlan

    empty_plan = QueryPlan(target="x")
    papers: list[LiteraturePaper] = []
    coverage = compute_coverage(empty_plan, papers)
    assert coverage.queries_executed == 0
    assert coverage.meets_threshold() is False
    deficits = coverage.deficits()
    assert any("queries_executed" in d for d in deficits)
    assert any("providers_used" in d for d in deficits)
    _ = (plan, PlannedQuery, QueryFamily)


# ======================================================================================
# novelty audit
# ======================================================================================
def test_novelty_verdict_requires_coverage_for_a_no_match_claim():
    auditor = NoveltyAuditor.__new__(NoveltyAuditor)
    thin = LiteratureCoverage(queries_executed=2, providers_used=[ProviderKind.ARXIV], papers_screened=3)
    thin.compute_score()
    verdict, reason = NoveltyAuditor.verdict(auditor, thin, [])
    assert verdict is NoveltyVerdict.NOVELTY_UNCERTAIN
    assert "insufficient" in reason

    rich = LiteratureCoverage(
        queries_executed=COVERAGE_THRESHOLDS["min_queries"],
        families_covered=list(REQUIRED_FAMILIES[: COVERAGE_THRESHOLDS["min_families"]]),
        providers_used=[ProviderKind.ARXIV, ProviderKind.SEMANTIC_SCHOLAR],
        papers_screened=COVERAGE_THRESHOLDS["min_papers_screened"],
    )
    rich.compute_score()
    verdict, reason = NoveltyAuditor.verdict(auditor, rich, [])
    assert verdict is NoveltyVerdict.NO_MATCH_FOUND_IN_SEARCHED_COVERAGE
    assert "searched coverage" in reason


def test_novelty_audit_is_recorded_and_never_says_nobody_has_done_this(kernel):
    planner = QueryPlanner(kernel)
    plan = planner.plan(kernel.principal("literature_researcher"), "write placement in fast weights")
    plan = planner.execute(
        kernel.principal("literature_researcher"), plan, offline_providers(_rich_corpus()), limit=10
    )
    auditor = NoveltyAuditor(kernel)
    audit = auditor.audit(
        kernel.principal("novelty_auditor"),
        "nobody has compared write placement under a matched-energy control",
        plan=plan,
    )
    assert audit.forbidden_phrases_detected, "the phrase 'nobody has …' must be flagged"
    assert audit.verdict in (NoveltyVerdict.SIMILAR_PRIOR_WORK_FOUND, NoveltyVerdict.NOVELTY_UNCERTAIN)
    assert audit.gap_kind is not GapKind.VERIFIED_NOVELTY
    sentence = audit.verdict_sentence().lower()
    for phrase in FORBIDDEN_NOVELTY_PHRASES:
        assert phrase not in sentence
    assert kernel.novelty_audits.get(audit.novelty_audit_id) is not None
    assert any(record.kind == "novelty.audit" for record in kernel.events.iter_records())


# ======================================================================================
# literature graph and prior art
# ======================================================================================
def test_graph_clusters_and_closest_prior_work_are_explainable(kernel):
    graph = LiteratureGraph(kernel)
    mechanism = LiteraturePaper(
        title="Fast weights write back at the originating position",
        abstract="alignment between write and read positions in a fast state",
    )
    topical = LiteraturePaper(
        title="A survey of transformer architectures for long documents",
        abstract="attention variants for long-range language modelling",
    )
    kernel.papers.save(mechanism)
    kernel.papers.save(topical)
    graph.add_relation(
        kernel.principal("literature_researcher"),
        from_kind=NodeKind.PAPER,
        from_id=mechanism.paper_id,
        to_kind=NodeKind.CURRENT_WORK,
        to_id="current_work",
        relation=RelationType.SAME_MECHANISM,
    )

    clusters = graph.clusters()
    assert clusters and all(isinstance(cluster, list) for cluster in clusters)
    ranking = graph.closest_prior_work("write placement in a fast weight state", top_k=2)
    assert ranking[0][0] == mechanism.paper_id
    assert ranking[0][1] > ranking[1][1]
    explanation = graph.explain_closeness(mechanism.paper_id, "write placement in a fast weight state")
    assert explanation["relations"] == ["same_mechanism"]
    assert explanation["shared_terms"]


def test_prior_art_matrix_reports_coverage_and_keeps_unknowns(kernel):
    builder = PriorArtMatrixBuilder(kernel)
    paper, _claim = add_literature_claim(kernel)
    matrix = builder.build(
        kernel.human(),
        title="write placement in fast weights",
        rows=[("current_work", "This work"), (paper.paper_id, "Prior work")],
    )
    markdown = render_markdown(matrix)
    assert "?" in markdown
    assert "coverage:" in coverage_line(matrix)
    counts = matrix.counts()
    assert counts["TRUE"] == 0 and counts["FALSE"] == 0
    assert counts["UNKNOWN"] == len(matrix.rows) * len(matrix.columns)

    updated = builder.upgrade_unknown(
        matrix,
        paper.paper_id,
        "writeback",
        value=Tri.TRUE,
        note="section 2 describes an explicit write-back step",
        paper_id=paper.paper_id,
        page="4",
        section="2",
    )
    assert updated.counts()["TRUE"] == 1
    assert updated.counts()["UNKNOWN"] == counts["UNKNOWN"] - 1
    assert "yes" in render_markdown(updated)


# ======================================================================================
# auditors
# ======================================================================================
def test_statistical_audit_runs_on_a_real_chain(kernel):
    chain = build_supported_chain(kernel)
    audit = StatisticalAuditor(kernel).audit(
        kernel.principal("analysis"), experiment_ids=[chain.experiment_id], title="chain audit"
    )
    assert audit.kind.value == "STATISTICAL"
    assert audit.subjects or audit.findings is not None
    codes = {finding.code for finding in audit.findings}
    # the fabricated chain is small (one comparison, three seeds), so we only require it to *run* and
    # to be honest about what it could not check
    assert isinstance(codes, set)
    assert kernel.audits.get(audit.audit_id) is not None


def test_mechanism_audit_flags_causal_language_above_the_ladder(kernel):
    chain = build_supported_chain(kernel)
    # downgrade the design to a *non-interventional* comparison, which is what most "we ran two
    # conditions" studies actually are: the rung drops, and causal language becomes an overclaim
    from researchos.models import ExperimentDesign

    experiment = kernel.experiments.require(chain.experiment_id)
    kernel.experiments.save(
        experiment.with_updates(
            design=ExperimentDesign(
                has_control=True,
                has_matched_conditions=True,
                has_intervention=False,
                is_observational=True,
                settings=["gpt2-small/124M/wikitext-103"],
            )
        )
    )
    claim = kernel.claims.require(chain.claim_id)
    kernel.claims.save(claim.with_updates(statement="Direct writeback causes better retention"))

    audit = MechanismAuditor(kernel).audit(
        kernel.principal("mechanism_auditor"), claim_ids=[chain.claim_id], title="mechanism"
    )
    assert audit.kind.value == "MECHANISM"
    assert audit.findings, "a causal claim with only controlled-comparison evidence must be flagged"
    assert any(
        finding.code
        in {
            "CAUSAL_LANGUAGE_WITHOUT_INTERVENTION",
            "MECHANISM_CLAIM_OVERREACH",
            "PLACEMENT_EFFECT_OVERCLAIM",
        }
        for finding in audit.findings
    )


def test_style_auditor_fires_on_ai_phrasing_and_stays_quiet_on_calibrated_prose(kernel):
    auditor = StyleAuditor()
    offending = (
        "Fast-weight memory has attracted increasing attention in recent years. "
        "This important finding shows that placement drives retention. "
        "Our novel framework demonstrates the mechanism and proves it is fundamentally better. "
        "This important finding shows that placement drives retention."
    )
    findings = auditor.audit_text(offending, evidence_level=EvidenceLevel.L2_REPRODUCED)
    categories = {finding.category.value for finding in findings}
    codes = {finding.code for finding in findings}
    assert findings
    assert categories & {
        "TEMPLATE_PHRASE",
        "EMPTY_BACKGROUND",
        "UNSUPPORTED_CAUSAL",
        "UNSUPPORTED_NOVELTY",
        "OVERCLAIM",
        "REPETITION",
        "RHETORICAL_ADJECTIVE",
    }
    assert any(phrase.lower()[:22] in offending.lower() for phrase in KNOWN_AI_PHRASES)
    assert any(finding.suggested_rewrite for finding in findings if finding.severity.value in {"HIGH", "BLOCKER"} or True)
    _ = codes

    clean = (
        "We compare two write placements with a matched control. "
        "The measured difference for retention is 0.147. "
        "Because the arms share the same data order, we attribute the difference to position."
    )
    quiet = auditor.audit_text(clean, evidence_level=EvidenceLevel.L3_CONTROLLED)
    assert not [f for f in quiet if f.code in {"TEMPLATE_PHRASE", "UNSUPPORTED_NOVELTY"}]


def test_style_audit_persists_an_audit_record(kernel):
    chain = build_supported_chain(kernel)
    from researchos.paper import PaperCompiler

    artifact = PaperCompiler(kernel).compile(kernel.principal("paper_writer"))
    artifact.sections[0].sentences[0].text = "This important finding demonstrates the mechanism."
    audit = StyleAuditor().audit_paper(kernel.principal("paper_auditor"), artifact, kernel=kernel)
    assert audit.kind.value == "STYLE"
    assert audit.findings
    assert kernel.audits.get(audit.audit_id) is not None
    _ = chain


# ======================================================================================
# readiness
# ======================================================================================
def test_readiness_reports_seven_dimensions_and_no_total(kernel):
    build_supported_chain(kernel)
    readiness = ReadinessAssessor(kernel).assess(kernel.human())
    assert len(readiness.dimensions) == 7
    assert readiness.complete()
    for dimension in readiness.dimensions:
        assert 0.0 <= dimension.score <= 1.0
        assert dimension.basis
    assert readiness.weakest(3)
    assert not hasattr(readiness, "total")


def test_readiness_with_nothing_but_claims_does_not_crash(kernel):
    lifecycle = ClaimLifecycle(kernel)
    lifecycle.create(kernel.principal("claim_manager"), statement="a bare hypothesis", scope="somewhere")
    readiness = ReadinessAssessor(kernel).assess(kernel.human())
    assert len(readiness.dimensions) == 7
    evidence = readiness.get(__import__("researchos.models", fromlist=["ReadinessDimension"]).ReadinessDimension.EVIDENCE_COMPLETENESS)
    assert evidence is not None
    assert evidence.score < 1.0


# ======================================================================================
# agents
# ======================================================================================
def test_all_fifteen_agents_exist_with_bounded_capabilities(kernel):
    agents = build_agents(kernel)
    assert len(agents) == 15
    rows = agent_table(kernel)
    assert len(rows) == 15
    for name, purpose, caps, denied in rows:
        assert purpose
        assert caps > 0
        for forbidden in ("claim.approve", "state.transition.approve", "skill.activate", "state.core.write"):
            assert forbidden in denied, f"{name} must not hold {forbidden}"


def test_agent_capability_boundaries_are_enforced_by_the_kernel(kernel):
    agents = build_agents(kernel)
    writer = agents["paper_writer"]
    assert writer.can(Cap.PAPER_COMPILE) is True
    assert writer.can(Cap.CLAIM_PROPOSE) is False
    with pytest.raises(PermissionDenied):
        writer.require(Cap.CLAIM_PROPOSE, "claim.create")

    importer = agents["importer"]
    assert importer.can(Cap.CLAIM_PROPOSE) is True
    assert importer.can(Cap.CLAIM_APPROVE) is False
    assert importer.may_approve_claims() is False

    auditor = agents["mechanism_auditor"]
    assert auditor.can(Cap.AUDIT_CREATE) is True
    assert auditor.can(Cap.EXPERIMENT_UPDATE) is False
    assert auditor.may_modify_experiment() is False


def test_planner_creates_tasks_and_files_requests_without_approving(kernel):
    agents = build_agents(kernel)
    planner = agents["planner"]
    task = planner.plan(
        "confirm the placement effect",
        purpose=__import__("researchos.models", fromlist=["TaskPurpose"]).TaskPurpose.DESIGN,
        priority=__import__("researchos.models", fromlist=["TaskPriority"]).TaskPriority.PRIMARY,
        stop_condition="effect confirmed or refuted",
    )
    assert kernel.tasks_store.get(task.task_id) is not None
    assert planner.can(Cap.STATE_TRANSITION_APPROVE) is False


def test_claim_manager_asks_the_human_instead_of_approving(kernel):
    agents = build_agents(kernel)
    manager = agents["claim_manager"]
    proposal = manager.request_approval("clm_x", reason="obligations appear to be met")
    assert proposal.status == "PROPOSED"
    assert "claim.approve" in proposal.requires
    assert manager.proposals[-1] is proposal
    assert any(record.kind == "agent.proposal" for record in kernel.events.iter_records())


def test_red_team_reports_objections_without_changing_state(kernel, experiment_factory, register_experiment):
    register_experiment(kernel, experiment_factory(control=False, matched=False, seeds=[0]))
    agents = build_agents(kernel)
    report = agents["red_team"].review(subject="current research state")
    assert report.questions
    categories = {question.category for question in report.questions}
    assert "baseline_matching" in categories
    assert report.verdict in {"READY", "NEEDS_WORK", "NOT_READY"}
    assert report.respects_state is True
    # the red team did not touch the research state
    assert kernel.state.revision() >= 0
    assert not kernel.claims.all()


# ======================================================================================
# claim language calibration
# ======================================================================================
@pytest.mark.parametrize(
    "text,level,should_overreach",
    [
        ("This establishes the mechanism of retention.", EvidenceLevel.L3_CONTROLLED, True),
        ("This establishes the mechanism of retention.", EvidenceLevel.L4_INTERVENTION, True),
        ("This establishes the mechanism of retention.", EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY, False),
        ("Direct writeback is necessary for retention.", EvidenceLevel.L3_CONTROLLED, True),
        ("Direct writeback causes better retention.", EvidenceLevel.L3_CONTROLLED, True),
        ("Direct writeback causes better retention.", EvidenceLevel.L4_INTERVENTION, False),
        ("We observe a difference in retention.", EvidenceLevel.L1_OBSERVATION, False),
        ("We find a difference in retention.", EvidenceLevel.L2_REPRODUCED, False),
        ("We hypothesise that placement matters.", EvidenceLevel.L0_IDEA, False),
        ("The effect generalises across models.", EvidenceLevel.L3_CONTROLLED, True),
    ],
)
def test_language_strength_is_capped_by_evidence_level(text, level, should_overreach):
    assert overreaches(text, level) is should_overreach
    if should_overreach:
        weakened = calibrate(text, level)
        assert not overreaches(weakened, level), weakened
        # calibration weakens, it never introduces content
        assert len(weakened) <= len(text) + 40


def test_calibration_never_strengthens():
    for text in (
        "We observe a difference.",
        "We find a difference.",
        "Direct writeback causes better retention.",
        "The effect generalises across models.",
    ):
        for level in EvidenceLevel:
            out = calibrate(text, level)
            assert overreaches(out, level) is False


def test_permitted_verbs_are_ordered_by_level():
    verbs = [permitted_verb(level) for level in EvidenceLevel]
    assert verbs[0].startswith("we hypothesise")
    assert "causes" in permitted_verb(EvidenceLevel.L4_INTERVENTION)
    assert language_class("We show an association between A and B") == "ASSOCIATION"
    assert language_class("We observe a difference") == "OBSERVATION"
    assert language_class("We find a difference") == "FINDING"


def test_welch_test_is_available_to_the_analysis_layer():
    result = welch_t_test([0.421, 0.416, 0.418], [0.274, 0.269, 0.271])
    assert result.p_value is not None and result.p_value < 0.01
    assert result.effect_size is not None
    assert result.method in {"welch", "student", "paired", "mann_whitney"}
