"""The Research Import Report — the first thing a researcher sees after handing over a project.

The report is the product experience described in the spec: it must show the *recovered research
state* (question, claims, experiments, evidence, conflicts, prior work, frontier, open questions,
skill gaps, next tasks) and an honest confidence, rather than a summary of the repository.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from ..models.common import RosModel, StrEnum, new_id, utcnow
from .conflicts import ConflictScan


class ImportConfidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ImportReport(RosModel):
    report_id: str = Field(default_factory=lambda: new_id("audit"))
    project_name: str = ""
    source_path: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    task_id: str | None = None
    batch_id: str = ""

    files_scanned: int = 0
    files_read: int = 0
    bytes_scanned: int = 0
    file_kinds: dict[str, int] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)

    core_question: str | None = None
    core_question_confidence: float = 0.0
    core_question_status: str = "NOT_FOUND"
    current_core_claims: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    major_experiments: list[str] = Field(default_factory=list)
    strongest_evidence: list[str] = Field(default_factory=list)
    weakest_evidence: list[str] = Field(default_factory=list)
    known_conflicts: list[str] = Field(default_factory=list)
    closest_prior_work: list[str] = Field(default_factory=list)
    literature_coverage: str = "no literature state recovered"
    current_frontier: str = ""
    open_questions: list[str] = Field(default_factory=list)
    possible_skill_gaps: list[str] = Field(default_factory=list)
    suggested_next_tasks: list[str] = Field(default_factory=list)

    confidence: ImportConfidence = ImportConfidence.LOW
    confidence_signals: list[str] = Field(default_factory=list)
    created_objects: dict[str, int] = Field(default_factory=dict)
    unparsed: list[str] = Field(default_factory=list)
    review_queue_size: int = 0
    evidence_levels: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Render the report in the canonical layout."""
        separator = "=" * 40
        lines = [
            separator,
            "RESEARCH IMPORT REPORT",
            separator,
            "",
            f"Project: {self.project_name}",
            f"Source:  {self.source_path}",
            f"Scanned: {self.files_scanned} files ({self.files_read} read, "
            f"{self.bytes_scanned / 1024:.0f} KiB) at {self.created_at:%Y-%m-%d %H:%M}",
            "",
            "Core Research Question:",
            f"  {self.core_question or '(not recovered — see review queue)'}",
            f"  [{self.core_question_status}, confidence {self.core_question_confidence:.2f}]",
            "",
            "Current Core Claims:",
        ]
        lines += _bullets(self.current_core_claims) or ["  (none recovered)"]
        lines += ["", "Rejected Claims:"]
        lines += _bullets(self.rejected_claims) or ["  (none recovered)"]
        lines += ["", "Major Experiments:"]
        lines += _bullets(self.major_experiments) or ["  (none recovered)"]
        lines += ["", "Strongest Evidence:"]
        lines += _bullets(self.strongest_evidence) or ["  (none recovered)"]
        lines += ["", "Weakest Evidence:"]
        lines += _bullets(self.weakest_evidence) or ["  (none recovered)"]
        lines += ["", "Known Conflicts:"]
        lines += _bullets(self.known_conflicts) or ["  (none detected)"]
        lines += ["", "Closest Prior Work:"]
        lines += _bullets(self.closest_prior_work) or ["  (none recovered)"]
        lines += [
            "",
            f"Literature Coverage: {self.literature_coverage}",
            "",
            "Current Research Frontier:",
            f"  {self.current_frontier or '(not established)'}",
        ]
        lines += ["", "Open Questions:"]
        lines += _bullets(self.open_questions) or ["  (none recovered)"]
        lines += ["", "Possible Skill Gaps:"]
        lines += _bullets(self.possible_skill_gaps) or ["  (none detected)"]
        lines += ["", "Suggested Next Tasks:"]
        lines += _bullets(self.suggested_next_tasks) or ["  (none)"]
        lines += [
            "",
            f"Import Confidence: {self.confidence.value}",
        ]
        lines += [f"  - {signal}" for signal in self.confidence_signals]
        lines += [
            "",
            "Created objects: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.created_objects.items()) if v),
            f"Human review queue: {self.review_queue_size} item(s) awaiting a decision",
        ]
        if self.unparsed:
            lines.append("")
            lines.append(f"Unparsed material ({len(self.unparsed)}):")
            lines += _bullets(self.unparsed[:10])
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines += _bullets(self.notes)
        lines += ["", separator]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        return "```\n" + self.render() + "\n```\n"


def _bullets(items: list[str]) -> list[str]:
    return [f"  - {item}" for item in items if item]


def assess_confidence(
    *,
    has_core_question: bool,
    question_confidence: float,
    experiment_count: int,
    data_artifact_count: int,
    config_count: int,
    claims_in_data: int,
    claims_in_prose_only: int,
    git_commits: int,
    conflict_count: int,
    unparsed_count: int,
    raw_evidence_count: int,
) -> tuple[ImportConfidence, list[str]]:
    """Deterministic import confidence.

    Confidence is about *how well the material pins down the research state*, not about how good
    the research is. Every point awarded is explained in the returned signals, so a LOW rating can
    be acted on.
    """
    score = 0
    signals: list[str] = []

    if has_core_question:
        points = 3 if question_confidence >= 0.7 else 2
        score += points
        signals.append(f"+{points} an explicit core research question was found "
                       f"(confidence {question_confidence:.2f})")
    else:
        signals.append("+0 no explicit core research question; inferred or left empty")
    if experiment_count:
        points = min(4, 2 + experiment_count)
        score += points
        signals.append(f"+{points} {experiment_count} experiment(s) reconstructed")
    else:
        signals.append("+0 no experiment could be reconstructed (prose-only import)")
    if data_artifact_count:
        score += 3
        signals.append(f"+3 {data_artifact_count} data artifact(s) give measured numbers")
    if config_count:
        score += 2
        signals.append(f"+2 {config_count} config file(s) pin the conditions")
    if raw_evidence_count:
        score += 1
        signals.append(f"+1 {raw_evidence_count} raw evidence record(s) created with hashes")
    if git_commits:
        score += 1
        signals.append(f"+1 git history available ({git_commits} commits)")
    if claims_in_data:
        score += 2
        signals.append(f"+2 {claims_in_data} claim candidate(s) backed by measured numbers")
    if claims_in_prose_only:
        signals.append(f"+0 {claims_in_prose_only} claim candidate(s) exist only in prose")
    if conflict_count:
        penalty = min(3, conflict_count)
        score -= penalty
        signals.append(f"-{penalty} {conflict_count} contradiction(s) need human resolution")
    if unparsed_count:
        penalty = min(4, unparsed_count)
        score -= penalty
        signals.append(f"-{penalty} {unparsed_count} file(s) could not be parsed")

    maximum = 16
    ratio = max(0.0, score / maximum)
    if ratio >= 0.72:
        confidence = ImportConfidence.HIGH
    elif ratio >= 0.4:
        confidence = ImportConfidence.MEDIUM
    else:
        confidence = ImportConfidence.LOW
    signals.append(f"score {score}/{maximum} -> {confidence.value}")
    return confidence, signals


def suggest_next_tasks(report: ImportReport, scan: ConflictScan) -> list[str]:
    """Turn the imported state into concrete, bounded next tasks (never into a paper)."""
    tasks: list[str] = []
    if report.core_question_status != "CONFIRMED":
        tasks.append("Review and confirm the core research question (queued as CORE_QUESTION_CANDIDATE)")
    if scan.number_conflicts:
        tasks.append(
            f"Resolve {len(scan.number_conflicts)} number contradiction(s) between prose and data"
        )
    if scan.duplicate_runs:
        tasks.append(f"De-duplicate {len(scan.duplicate_runs)} repeated run(s) before any analysis")
    if scan.missing_values:
        tasks.append(f"Decide how to treat {len(scan.missing_values)} missing measurement(s)")
    if any("no measured value found" in line for line in scan.metric_disagreements):
        tasks.append("Locate the artifacts behind numbers that appear only in prose")
    if report.weakest_evidence:
        tasks.append("Strengthen or retire the claims whose evidence is prose-only")
    if report.confidence is ImportConfidence.LOW:
        tasks.append("Attach raw artifacts (configs, metrics, logs) and re-run the import")
    tasks.append("Run a literature coverage search for the recovered core question")
    tasks.append("Run statistical and mechanism audits over the reconstructed experiments")
    return tasks
