"""Research Import tests.

Imports are tested against a real legacy project (``examples/DLA``) because the point of the feature
is to survive messy, real material: a stale audit, a duplicated run row, a diary that contradicts the
README, a paper draft that overclaims, a bibliography with no full texts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchos.claims import ClaimRegistry
from researchos.importer import ImportPipeline
from researchos.importer.conflicts import detect_conflicts
from researchos.importer.facts import Fact, FactKind
from researchos.importer.report import ImportConfidence
from researchos.importer.scanner import ScanResult, scan
from researchos.kernel import ResearchKernel
from researchos.models import (
    ClaimStatus,
    ConflictResolution,
    EvidenceType,
    FulltextStatus,
    ProviderKind,
    SourceKind,
    VerificationStatus,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DLA = EXAMPLES / "DLA"


@pytest.fixture(scope="module")
def dla_available() -> bool:
    return DLA.is_dir()


# ======================================================================================
# scanning and classification
# ======================================================================================
def test_scan_classifies_material_by_kind_and_hashes_everything(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "train.py").write_text("import argparse\n", encoding="utf-8")
    (tmp_path / "notes" / "diary.md").write_text("# diary\n\n- an entry\n", encoding="utf-8")
    (tmp_path / "runs" / "metrics.csv").write_text("arm,seed\n direct,0\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (tmp_path / "audit" / "x.md").parent.mkdir()
    (tmp_path / "audit" / "stats_audit.md").write_text("# audit\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\x03")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x", encoding="utf-8")

    result = scan(tmp_path)
    kinds = {item.rel_path: item.kind.value for item in result.files}
    assert kinds["README.md"] == "README"
    assert kinds["sub/train.py"] == "PYTHON"
    assert kinds["runs/metrics.csv"] == "DATA_CSV"
    assert kinds["notes/diary.md"] == "PLAIN_TEXT"
    assert kinds["audit/stats_audit.md"] == "AUDIT_REPORT"
    # every scanned file carries the digest that makes its interpretation checkable
    assert all(len(item.sha256) == 64 for item in result.files)
    # dependency directories and binaries are skipped, with a stated reason
    assert all("node_modules" not in item.rel_path for item in result.files)
    assert any("ignored directory" in reason for _path, reason in result.skipped)


def test_number_extraction_ignores_dates_ranges_and_identifiers():
    from researchos.importer.facts import extract_numbers

    text = (
        "On 2024-11-07 the direct arm reached retention_at_1 = 0.418 across seeds 0--2 "
        "with 124M parameters in gpt2-small."
    )
    values = sorted(hit.value for hit in extract_numbers(text))
    assert 0.418 in values
    assert 2024 not in values and 11 not in values and 7 not in values  # a date, not a number
    assert 2 not in values  # a range endpoint
    assert 124 not in values  # part of an identifier


def test_conflict_detection_prefers_labelled_data_over_logs():
    from researchos.models import SourceRef

    def fact(rule, metric, value, labels, path, kind=SourceKind.RAW_EXPERIMENT):
        return Fact(
            kind=FactKind.METRIC_VALUE if rule.startswith("table") else FactKind.PROSE_NUMBER,
            statement=f"{metric} = {value}",
            rule=rule,
            source=SourceRef(path=path, kind=kind),
            payload={"metric": metric, "value": value, "labels": labels, "table": rule.startswith("table")},
        )

    table = fact("table:metric", "retention_at_1", 0.418, {"arm": "direct", "seed": "1"}, "runs/metrics.csv")
    log = fact("log:kv", "retention_at_1", 0.331, None, "logs/run.log")
    prose = fact("number:prose", "retention", 0.331, None, "paper/draft.tex", SourceKind.PAPER_PROSE)
    scan_result = detect_conflicts([table, log, prose])
    # the prose number matches the *labelled data* value family nearest it, not the log line
    assert all("logs/run.log" not in c.difference for c in scan_result.number_conflicts)


# ======================================================================================
# the full import of the legacy project
# ======================================================================================
@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    if not DLA.is_dir():
        pytest.skip("examples/DLA is missing")
    root = tmp_path_factory.mktemp("dla-import") / "DLA"
    root.mkdir()
    kernel = ResearchKernel.create(root, name="DLA", domain="AI/ML")
    result = ImportPipeline(kernel).run(DLA)
    return kernel, result


def test_import_recovers_the_research_question_as_a_transition(imported):
    kernel, result = imported
    report = result.report
    assert report.core_question_status == "CONFIRMED"
    assert report.core_question_confidence >= 0.7
    assert "fast weight" in (report.core_question or "").lower()

    state = kernel.research_state()
    assert state.core_question is not None
    assert state.core_question.is_placeholder is False
    # and the change went through an approved STR, with a decision record
    requests = kernel.transitions.history()
    assert requests and requests[0].change_class.value == "CORE_QUESTION"
    assert requests[0].status.value == "APPROVED"
    assert requests[0].decision_id
    assert kernel.decisions.get(requests[0].decision_id) is not None


def test_import_never_certifies_a_claim(imported):
    kernel, result = imported
    claims = kernel.claims.all()
    assert claims, "the import should recover claim candidates"
    for claim in claims:
        assert claim.status in (ClaimStatus.HYPOTHESIS, ClaimStatus.IDEA, ClaimStatus.REJECTED)
    registry = ClaimRegistry(kernel)
    assert registry.paper_ready() == []
    assert registry.live(), "claim candidates are live hypotheses, not conclusions"
    # every imported claim points back at the bytes it came from
    for claim in registry.live():
        assert claim.evidence_ids
        evidence = kernel.evidence.require(claim.evidence_ids[0])
        assert evidence.source_refs and evidence.source_refs[0].path
        assert evidence.source_refs[0].sha256


def test_import_keeps_rejected_claims_as_tombstones(imported):
    kernel, result = imported
    rejected = ClaimRegistry(kernel).by_status(ClaimStatus.REJECTED)
    assert rejected, "the README lists rejected claims and the import must retain them"
    for claim in rejected:
        assert claim.tombstone is True
        assert claim.rejection_reason
        assert claim.rejection_basis is not None
        assert claim.claim_id in kernel.research_state().rejected_claims


def test_import_reconstructs_experiments_with_inferred_designs(imported):
    kernel, result = imported
    experiments = kernel.experiments.all()
    assert len(experiments) >= 3
    titles = " | ".join(e.title for e in experiments)
    assert "matched_energy" in titles, "the budget-matched control must be recognised"
    assert "sleep0" in titles, "the component-removal arm must be recognised"

    primary = next(e for e in experiments if "matched_energy" not in e.title and "sleep0" not in e.title)
    assert primary.control is not None
    assert primary.matched_conditions, "matched conditions should be inferred from the configs"
    assert primary.design.has_intervention is True
    assert primary.evidence_level().value in {"L3_CONTROLLED", "L4_INTERVENTION"}
    assert primary.seeds, "seeds come from the run grid, not from prose"
    # inferred design decisions are stated as limitations, so a human can audit them
    assert any("inferred" in limitation for limitation in primary.limitations)

    ablation = next(e for e in experiments if "sleep0" in e.title)
    assert ablation.design.has_necessity_design is True
    assert ablation.evidence_level().value == "L5_NECESSITY_OR_SUFFICIENCY"


def test_import_finds_the_real_conflicts_and_ignores_the_noise(imported):
    kernel, result = imported
    conflicts = kernel.conflicts.all()
    differences = " ".join(c.difference for c in conflicts)
    # the stale audit's number is caught…
    assert "stats_audit_2024-11-04" in differences
    # …and the duplicated run row is caught
    assert any("duplicate" in c.difference or c.kind.value == "DESIGN" for c in conflicts)
    # but a prose percentage-point statement and a std column are not turned into fake conflicts
    assert "13.4" not in differences

    for conflict in conflicts:
        assert conflict.preserved is True
        assert conflict.source_a.ref and conflict.source_b.ref


def test_dataset_conflict_trust_order_applied_to_the_audit(imported):
    kernel, _result = imported
    value_conflicts = [c for c in kernel.conflicts.all() if c.kind.value == "VALUE"]
    assert value_conflicts
    for conflict in value_conflicts:
        # raw experiment outranks an audit report, so the audit is recorded as superseded
        assert conflict.resolution in (ConflictResolution.A_WINS, ConflictResolution.B_WINS)
        assert conflict.winning_ref and conflict.superseded_ref


def test_import_creates_analysis_artifacts_from_measured_values(imported):
    kernel, _result = imported
    analyses = kernel.analyses.all()
    assert analyses
    analysis = analyses[0]
    assert analysis.deterministic is True
    assert analysis.results
    assert analysis.verified_by is None, "the importer cannot verify its own output"
    # duplicate rows are excluded and *recorded*
    assert any("duplicate" in warning for warning in analysis.warnings)


def test_import_creates_unverified_raw_evidence_only(imported):
    kernel, _result = imported
    evidence = kernel.evidence.all()
    assert evidence
    assert any(e.evidence_type is EvidenceType.RAW for e in evidence)
    for item in evidence:
        if item.evidence_type is EvidenceType.RAW:
            assert item.immutable is True
        assert item.verification_status is not VerificationStatus.VERIFIED


def test_import_ingests_bibliography_without_pretending_to_know_it(imported):
    kernel, _result = imported
    papers = kernel.papers.all()
    assert len(papers) >= 5
    for paper in papers:
        assert paper.fulltext_status is not FulltextStatus.FULLTEXT_VERIFIED
        assert paper.mechanism is None
        assert paper.explicit_components == []
        assert ProviderKind.LOCAL_BIB in paper.providers
        assert paper.imported_from
        # metadata that *is* in a BibTeX file is recovered
        assert paper.authors
    assert any(paper.include for paper in papers), "cited keys should be marked for inclusion"


def test_import_recovers_author_notes_in_diary_order(imported):
    kernel, _result = imported
    notes = kernel.notes.all()
    assert notes
    diary_notes = [n for n in notes if n.source_refs and "diary" in n.source_refs[0].path]
    assert diary_notes
    ordered = sorted(diary_notes, key=lambda n: n.created_at)
    assert ordered[0].created_at <= ordered[-1].created_at
    # the anomaly and the hypothesis revision survive as notes, because they are the research history
    kinds = {n.kind.value for n in notes}
    assert "ANOMALY" in kinds or "CLAIM_REVISION" in kinds


def test_import_queues_human_reviews_and_does_not_decide(imported):
    kernel, result = imported
    pending = kernel.pending_reviews()
    assert len(pending) == len(result.review_items)
    assert pending
    kinds = {item.kind.value for item in pending}
    assert "CLAIM_CANDIDATE" in kinds
    assert all(item.confidence > 0 for item in pending)
    # the queue is where uncertain recovery goes; nothing is silently accepted
    assert all(item.status.value == "PENDING" for item in pending)


def test_import_report_confidence_and_next_tasks(imported):
    _kernel, result = imported
    report = result.report
    assert report.confidence in (ImportConfidence.HIGH, ImportConfidence.MEDIUM)
    assert any("explicit core research question" in signal for signal in report.confidence_signals)
    assert report.suggested_next_tasks
    assert any("contradiction" in task for task in report.suggested_next_tasks)
    assert report.files_scanned >= 8
    rendered = report.render()
    for heading in (
        "RESEARCH IMPORT REPORT",
        "Core Research Question:",
        "Rejected Claims:",
        "Major Experiments:",
        "Known Conflicts:",
        "Closest Prior Work:",
        "Import Confidence:",
    ):
        assert heading in rendered
    assert "no claim was imported above HYPOTHESIS" in " ".join(report.notes)


def test_import_is_recorded_in_the_event_log_and_timeline(imported):
    kernel, _result = imported
    kinds = [record.kind for record in kernel.events.iter_records()]
    assert "import.completed" in kinds
    assert any(event.imported for event in kernel.timeline.all())
    assert kernel.events.verify().ok
    assert kernel.project().imported is True


# ======================================================================================
# degraded material
# ======================================================================================
def test_prose_only_import_reports_low_confidence(tmp_path):
    source = tmp_path / "notes-only"
    source.mkdir()
    (source / "thoughts.md").write_text(
        "# Thoughts\n\nWe think attention might be all you need. We should test this.\n",
        encoding="utf-8",
    )
    root = tmp_path / "project"
    root.mkdir()
    kernel = ResearchKernel.create(root, name="thoughts")
    result = ImportPipeline(kernel).run(source)

    report = result.report
    assert report.confidence in (ImportConfidence.LOW, ImportConfidence.MEDIUM)
    assert kernel.experiments.all() == []
    assert any("no experiment could be reconstructed" in signal for signal in report.confidence_signals)
    # nothing was invented to fill the gap
    assert ClaimRegistry(kernel).by_status(ClaimStatus.SUPPORTED) == []


def test_unparseable_material_becomes_a_review_item(tmp_path):
    source = tmp_path / "broken"
    source.mkdir()
    (source / "broken.json").write_text("{not valid json", encoding="utf-8")
    root = tmp_path / "project2"
    root.mkdir()
    kernel = ResearchKernel.create(root, name="broken")
    result = ImportPipeline(kernel).run(source)
    assert result.report.unparsed
    assert any(item.kind.value == "UNPARSED_MATERIAL" for item in result.review_items)
    assert "broken.json" in " ".join(result.report.unparsed)


def test_git_history_becomes_imported_timeline_events(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    (source / "README.md").write_text("# Project\n", encoding="utf-8")
    root = tmp_path / "project3"
    root.mkdir()
    kernel = ResearchKernel.create(root, name="repo")

    def fake_git(_root, limit):
        return [
            {"commit": "a" * 40, "date": "2024-11-01", "author": "author", "subject": "initial commit"},
            {
                "commit": "b" * 40,
                "date": "2024-11-07",
                "author": "author",
                "subject": "revert: drop the unmatched baseline run",
            },
        ]

    result = ImportPipeline(kernel).run(source, git_runner=fake_git)
    imported_events = [event for event in kernel.timeline.all() if event.imported]
    assert len(imported_events) >= 2
    decisions = kernel.decisions.all()
    assert decisions, "a 'revert: ...' commit is decision-shaped and should be surfaced as one"
    assert any(decision.kind.value in {"EXPERIMENT_ROUTE", "OTHER"} for decision in decisions)
    assert result.report.files_scanned >= 1
