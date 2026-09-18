"""Model schema and validation tests.

These tests exist because every object in ResearchOS is a *contract*: a hand-edited YAML file that
violates the contract must fail loudly at load time rather than silently degrade the research state.
"""

from __future__ import annotations

import pytest

from researchos.models import (
    Analysis,
    ArtifactRef,
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceLevel,
    EvidenceType,
    Experiment,
    ExperimentDesign,
    FulltextStatus,
    LiteratureCoverage,
    LiteraturePaper,
    NoveltyAudit,
    NoveltyVerdict,
    PaperReadiness,
    Provenance,
    QueryPlan,
    ReadinessDimension,
    ReadinessDimensionResult,
    ResearchState,
    StatResult,
    Task,
    TaskPriority,
    Tri,
    VerificationStatus,
    assess_experiment_level_from_design,
    canonical_json,
    guarded_root,
    sha256_json,
)


# ---------------------------------------------------------------------------- evidence level
def test_evidence_level_ordering_is_monotone():
    levels = list(EvidenceLevel)
    for earlier, later in zip(levels, levels[1:]):
        assert earlier.rank < later.rank
        assert earlier < later


@pytest.mark.parametrize(
    "design,expected",
    [
        (ExperimentDesign(is_observational=True), EvidenceLevel.L1_OBSERVATION),
        (ExperimentDesign(has_control=True), EvidenceLevel.L2_REPRODUCED),
        (
            ExperimentDesign(has_control=True, has_matched_conditions=True),
            EvidenceLevel.L3_CONTROLLED,
        ),
        (ExperimentDesign(has_intervention=True), EvidenceLevel.L4_INTERVENTION),
        (ExperimentDesign(has_necessity_design=True), EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY),
        (ExperimentDesign(has_rescue=True), EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY),
        (
            ExperimentDesign(has_control=True, has_matched_conditions=True, has_intervention=True,
                             settings=["a", "b"]),
            EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
        ),
        (
            # two "settings" that are really one setting must NOT unlock cross-setting replication
            ExperimentDesign(has_control=True, has_intervention=True, settings=["only"]),
            EvidenceLevel.L4_INTERVENTION,
        ),
    ],
)
def test_design_flags_determine_the_evidence_ceiling(design, expected):
    assert assess_experiment_level_from_design(design) == expected


def test_prose_cannot_raise_the_level():
    """A paper sentence is not an input to the level calculation, by construction."""
    design = ExperimentDesign(has_control=False, is_observational=True)
    assert assess_experiment_level_from_design(design) is EvidenceLevel.L1_OBSERVATION
    # and there is simply no parameter through which a string could influence it
    with pytest.raises(TypeError):
        assess_experiment_level_from_design(design, description="this establishes the mechanism")


# ---------------------------------------------------------------------------- claim
def test_claim_rejects_status_jumps_and_missing_tombstone_fields():
    with pytest.raises(Exception) as supported:
        Claim(statement="x", status=ClaimStatus.SUPPORTED, scope="s")
    assert "evidence" in str(supported.value)

    with pytest.raises(Exception) as tested:
        Claim(statement="x", status=ClaimStatus.TESTED, scope="s")
    assert "experiment" in str(tested.value)

    with pytest.raises(Exception) as rejected:
        Claim(statement="x", status=ClaimStatus.REJECTED, scope="s")
    assert "rejection_reason" in str(rejected.value)

    with pytest.raises(Exception) as superseded:
        Claim(statement="x", status=ClaimStatus.SUPERSEDED, scope="s")
    assert "superseded_by" in str(superseded.value)


def test_claim_with_history_is_append_only_in_practice():
    claim = Claim(statement="a claim", scope="scope")
    claim.record(ClaimStatus.HYPOTHESIS, by="tester", reason="scope declared")
    moved = claim.with_updates(status=ClaimStatus.HYPOTHESIS)
    assert [entry.to_status for entry in moved.history] == [ClaimStatus.HYPOTHESIS]
    assert moved.history[0].from_status is ClaimStatus.IDEA


# ---------------------------------------------------------------------------- evidence
def test_raw_evidence_is_immutable_and_needs_a_source():
    with pytest.raises(Exception) as no_source:
        Evidence(statement="x", evidence_type=EvidenceType.RAW)
    assert "artifact, payload or source_path" in str(no_source.value)

    raw = Evidence(statement="x", evidence_type=EvidenceType.RAW, payload={"v": 1})
    assert raw.immutable is True

    with pytest.raises(Exception) as verified:
        Evidence(statement="x", evidence_type=EvidenceType.VERIFIED, payload={"v": 1})
    assert "verified_by" in str(verified.value)

    with pytest.raises(Exception) as analyzed:
        Evidence(statement="x", evidence_type=EvidenceType.ANALYZED, payload={"v": 1})
    assert "source_analysis" in str(analyzed.value)


def test_evidence_content_hash_ignores_metadata_but_tracks_content():
    base = Evidence(statement="delta 0.31", evidence_type=EvidenceType.RAW, payload={"d": 0.31})
    renamed = base.with_updates(confidence_note="a note")
    assert renamed.content_hash() == base.content_hash()
    changed = base.with_updates(payload={"d": 0.99})
    assert changed.content_hash() != base.content_hash()


def test_artifact_ref_verification_reports_missing_and_mismatch(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    ref = ArtifactRef(path="a.txt", sha256="deadbeef")
    assert ref.verify(tmp_path) is VerificationStatus.HASH_MISMATCH
    assert ArtifactRef(path="nope.txt", sha256="x").verify(tmp_path) is VerificationStatus.MISSING_ARTIFACT
    good = ArtifactRef(path="a.txt", sha256=__import__("researchos.models", fromlist=["sha256_file"]).sha256_file(target))
    assert good.verify(tmp_path) is VerificationStatus.VERIFIED


# ---------------------------------------------------------------------------- experiment
def test_completed_experiment_requires_seeds_and_n():
    with pytest.raises(Exception) as seeds:
        Experiment(title="t", status=__import__("researchos.models", fromlist=["ExperimentStatus"]).ExperimentStatus.COMPLETED, n=3)
    assert "seed" in str(seeds.value)

    with pytest.raises(Exception) as failed:
        Experiment(title="t", status=__import__("researchos.models", fromlist=["ExperimentStatus"]).ExperimentStatus.FAILED)
    assert "failure_reason" in str(failed.value)


def test_independence_and_replication_are_different_relations(experiment_factory):
    base = experiment_factory()
    same_design_other_seeds = experiment_factory(seeds=[7, 8, 9])
    other_setting = experiment_factory(model="gpt2-medium", scale="355M")

    assert base.is_replication_of(same_design_other_seeds) is True
    assert base.is_independent_of(same_design_other_seeds) is False
    assert base.is_independent_of(other_setting) is True


# ---------------------------------------------------------------------------- literature
def test_paper_may_not_pretend_to_know_full_text_details():
    with pytest.raises(Exception) as excinfo:
        LiteraturePaper(
            title="p", fulltext_status=FulltextStatus.ABSTRACT_LEVEL_ONLY, mechanism="a two-stage write"
        )
    assert "fulltext-only fields" in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        LiteraturePaper(title="p", fulltext_status=FulltextStatus.FULLTEXT_VERIFIED)
    assert "proof" in str(excinfo.value)


def test_novelty_audit_is_coverage_gated():
    thin = LiteratureCoverage(queries_executed=2, families_covered=[], providers_used=[])
    with pytest.raises(Exception) as excinfo:
        NoveltyAudit(
            target_statement="nobody has done this",
            coverage=thin,
            verdict=NoveltyVerdict.NO_MATCH_FOUND_IN_SEARCHED_COVERAGE,
        )
    assert "insufficient coverage" in str(excinfo.value)

    uncertain = NoveltyAudit(target_statement="x", coverage=thin)
    assert uncertain.verdict is NoveltyVerdict.NOVELTY_UNCERTAIN
    assert uncertain.coverage_is_sufficient() is False
    assert "nobody has" not in uncertain.verdict_sentence()


def test_coverage_threshold_arithmetic():
    coverage = LiteratureCoverage(
        queries_executed=8,
        families_covered=[],
        providers_used=[],
        papers_screened=0,
    )
    coverage.compute_score()
    assert coverage.queries_executed == 8
    assert coverage.meets_threshold() is False
    assert any("families" in deficit for deficit in coverage.deficits())


def test_query_plan_reports_missing_required_families():
    plan = QueryPlan(target="write placement")
    assert set(plan.missing_required_families())  # nothing executed yet -> everything is missing


# ---------------------------------------------------------------------------- task / state guards
def test_task_boundaries_are_validated():
    with pytest.raises(Exception) as overlap:
        Task(
            objective="x",
            allowed_actions=["analysis.compute"],
            forbidden_actions=["analysis.compute"],
        )
    assert "both allowed and forbidden" in str(overlap.value)

    with pytest.raises(Exception) as core:
        Task(objective="x", touches_core=True, stop_condition="")
    assert "stop_condition" in str(core.value)

    exploratory = Task(objective="x", priority=TaskPriority.EXPLORATORY, touches_core=False)
    assert exploratory.may_request_core_change() is False
    primary = Task(objective="x", priority=TaskPriority.PRIORITY if False else TaskPriority.PRIMARY, stop_condition="done")
    assert primary.may_request_core_change() is True


def test_guarded_root_resolution():
    assert guarded_root("core_question.statement") == "core_question"
    assert guarded_root("priorities[0]") == "priorities"
    assert guarded_root("open_questions") is None
    assert guarded_root("literature_state.coverage_score") is None


def test_research_state_guarded_roots_hash_is_stable_and_narrow():
    state = ResearchState(project_name="p")
    before = sha256_json(state.guarded_roots())
    state.notes = ["an unrelated note"]
    assert sha256_json(state.guarded_roots()) == before
    state.core_claims = ["clm_1"]
    assert sha256_json(state.guarded_roots()) != before


# ---------------------------------------------------------------------------- paper readiness
def test_readiness_has_no_single_total_score():
    readiness = PaperReadiness(
        dimensions=[
            ReadinessDimensionResult(dimension=d, score=0.5, basis="test")
            for d in ReadinessDimension
        ]
    )
    assert readiness.complete() is True
    assert len(readiness.dimensions) == 7
    for forbidden in ("total", "overall", "score", "summary_score"):
        assert not hasattr(readiness, forbidden)


# ---------------------------------------------------------------------------- canonical forms
def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"a": [1, 2]}) != canonical_json({"a": [2, 1]})


def test_analysis_requires_results_or_an_explanation():
    with pytest.raises(Exception) as excinfo:
        Analysis(title="empty")
    assert "must produce results" in str(excinfo.value)

    ok = Analysis(title="explained", warnings=["no comparable runs were available"])
    assert ok.results == []
    assert ok.numeric_map() == {}


def test_stat_result_keeps_paired_semantics_honest():
    with pytest.raises(Exception) as excinfo:
        StatResult(name="x", method=__import__("researchos.models", fromlist=["StatMethod"]).StatMethod.PAIRED_TTEST, paired=False)
    assert "paired" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo:
        StatResult(name="x", method=__import__("researchos.models", fromlist=["StatMethod"]).StatMethod.WELCH_TTEST, paired=True)
    assert "unpaired" in str(excinfo.value)


def test_provenance_reports_what_is_missing():
    provenance = Provenance(code_commit="abc")
    assert "config_hash" in provenance.missing()
    assert 0.0 <= provenance.completeness() <= 1.0
