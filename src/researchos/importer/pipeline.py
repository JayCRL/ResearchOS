"""The Research Import pipeline (Research Archaeology).

```
INPUT FILES → PARSER → CLASSIFICATION → FACT EXTRACTION → EXPERIMENT EXTRACTION →
CLAIM EXTRACTION → LITERATURE EXTRACTION → DECISION EXTRACTION → CONFLICT DETECTION →
PROVENANCE LINKING → RESEARCH STATE RECONSTRUCTION → HUMAN REVIEW QUEUE
```

Two principles govern every line of this module:

1. **Do not "summarise into a paper".** Reconstruct the research state, and keep the material's
   own numbers and words attached to it.
2. **Reconstruct, do not bless.** Imported claims enter at ``IDEA``/``HYPOTHESIS``; nothing is ever
   imported as ``SUPPORTED``. Anything uncertain goes to the human review queue with its evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..models.analysis import Analysis, StatMethod, StatResult
from ..models.common import (
    ClaimStatus,
    ConflictKind,
    EvidenceLevel,
    EvidenceType,
    FulltextStatus,
    GapKind,
    RiskLevel,
    Severity,
    SourceKind,
    SourceRef,
    TaskPriority,
    TaskPurpose,
    VerificationStatus,
    canonical_json,
    new_id,
    sha256_json,
    utcnow,
)
from ..models.audit import Conflict, ConflictSource
from ..models.claim import Claim, ClaimHistoryEntry, RejectionBasis
from ..models.decision import Decision, DecisionKind, StateOp, StateOperation
from ..models.evidence import Evidence
from ..models.experiment import Arm, Experiment, ExperimentDesign, ExperimentStatus, MetricRef, TrainingBudget
from ..models.literature import LiteraturePaper, ProviderKind
from ..models.research_state import Frontier, OpenQuestion, PriorWorkRef
from ..models.review import ReviewItem, ReviewKind
from ..models.skill import SkillGap
from ..models.timeline import AuthorNote, NoteKind, TimelineEventKind
from ..kernel.errors import ResearchOSError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..kernel.provenance import capture_provenance, git_info
from . import extractors_artifacts as artifacts
from . import extractors_text as text_extractors
from .conflicts import ConflictScan, detect_conflicts, summarise
from .facts import (
    ExtractionOutput,
    Fact,
    FactKind,
    FileKind,
    SourceFile,
    claim_status_ceiling,
    implied_level,
    normalise_text,
)
from .report import ImportConfidence, ImportReport, assess_confidence, suggest_next_tasks
from .scanner import ScanResult, scan

#: Claim text that is boilerplate rather than an assertion.
CLAIM_STOPWORDS: tuple[str, ...] = (
    "we thank",
    "we refer the reader",
    "the remainder of this paper",
    "this paper is organized",
    "in this section we describe",
    "we use the following notation",
)

#: Arm-name tokens that identify a control condition.
CONTROL_TOKENS: tuple[str, ...] = ("shuf", "rand", "control", "baseline", "naive", "placebo", "none", "noop")
#: Arm-name tokens that identify a matched-budget control family.
ENERGY_TOKENS: tuple[str, ...] = ("matched_energy", "matched_param", "matched_flops", "energy_matched", "matched")
#: Arm-name tokens that identify a component-removal (necessity) experiment.
REMOVAL_TOKENS: tuple[str, ...] = (
    "sleep0", "no_sleep", "sleep_off", "ablation", "ablated", "removed", "without", "minus", "nowrite",
)
REMOVAL_PATTERN = re.compile(r"(?:^|_)0(?:$|_)")

NECESSITY_LANGUAGE: tuple[str, ...] = (
    r"\bremoving\b", r"\bremoval\b", r"\bablat", r"\bis necessary\b", r"\brequired for\b",
    r"\bwithout (?:it|the|this)\b", r"\bcollapsed\b", r"\bno longer\b.*\b(?:work|retain)",
)
SUFFICIENCY_LANGUAGE: tuple[str, ...] = (r"\bis sufficient\b", r"\bforcing\b", r"\badding .* (?:is enough|restores)\b")
RESCUE_LANGUAGE: tuple[str, ...] = (r"\brescue", r"\brecovered the effect\b", r"\brestor")

#: Config keys that count towards "conditions are matched".
MATCH_KEYS: tuple[str, ...] = (
    "model", "dataset", "scale", "optimizer", "learning_rate", "lr", "steps", "batch_size",
    "training_budget", "sleep_steps", "seeds", "seed", "tokenizer",
)

#: Config keys that name the run rather than describing an experimental condition.
CONFIG_META_KEYS: frozenset[str] = frozenset(
    {
        "arm", "variant", "condition", "name", "run", "run_id", "id", "out", "output", "output_dir",
        "log", "log_dir", "tag", "tags", "comment", "comments", "note", "notes", "description",
    }
)

#: Config keys that describe how the *control* was built (budget accounting), not the treatment.
BUDGET_KEYS: frozenset[str] = frozenset(
    {
        "matched_energy", "matched_params", "matched_flops", "matched_compute", "budget",
        "steps", "tokens", "flops", "wall_clock", "epochs", "batch_size",
    }
)

MAX_REPORTED = 12


@dataclass
class ImportResult:
    report: ImportReport
    facts: list[Fact] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    review_items: list[ReviewItem] = field(default_factory=list)
    created: dict[str, list[str]] = field(default_factory=dict)
    scan: ScanResult | None = None

    def summary_line(self) -> str:
        counts = ", ".join(f"{k}={len(v)}" for k, v in sorted(self.created.items()) if v)
        return f"import: {counts}; {len(self.review_items)} review item(s); confidence {self.report.confidence.value}"


class ImportPipeline:
    """Reconstruct a research state from an existing research directory or repository."""

    def __init__(
        self,
        kernel: ResearchKernel,
        *,
        principal: Principal | None = None,
        max_claims: int = 60,
        max_notes: int = 120,
        approve_core_question: bool = True,
        verbose: bool = False,
    ) -> None:
        self.kernel = kernel
        self.principal = principal or kernel.human()
        self.max_claims = max_claims
        self.max_notes = max_notes
        self.approve_core_question = approve_core_question
        self.verbose = verbose
        self.created: dict[str, list[str]] = {}
        self.review_items: list[ReviewItem] = []
        self.batch_id = new_id("finding")
        self.notes: list[str] = []

    # ================================================================== entry point

    def run(
        self,
        source: str | Path,
        *,
        task_id: str | None = None,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        git_runner=None,
    ) -> ImportResult:
        source_path = Path(source)
        if not source_path.is_absolute():
            # CLI semantics: a relative path is relative to the working directory. Only fall back to
            # the project root when the caller clearly meant a path inside the project.
            from_cwd = (Path.cwd() / source_path).resolve()
            from_root = (self.kernel.root / source_path).resolve()
            source_path = from_cwd if from_cwd.exists() else from_root
        if not source_path.exists():
            raise ResearchOSError(f"nothing to import at {source_path}")

        task_id = task_id or self._ensure_task(source_path)
        root = source_path if source_path.is_dir() else source_path.parent

        # 1-3. scan, classify, parse
        scanned = scan(root, include=include, exclude=exclude)
        raw_facts, extraction = self._extract_all(scanned)
        raw_facts.extend(artifacts.extract_git(root, runner=git_runner).facts)

        # 4. conflict detection over the *raw* fact set: deduplication would hide duplicate runs,
        #    and duplicate runs are exactly the silent double-count we want to surface.
        conflict_scan = detect_conflicts(raw_facts)
        facts = self._dedupe(raw_facts)

        # 5-11. reconstruction
        report = self._reconstruct(
            source_path=source_path,
            root=root,
            scanned=scanned,
            facts=facts,
            extraction=extraction,
            conflict_scan=conflict_scan,
            task_id=task_id,
        )
        return ImportResult(
            report=report,
            facts=facts,
            conflicts=[self.kernel.ledger.require(cid) for cid in self.created.get("conflict", [])],
            review_items=self.review_items,
            created=self.created,
            scan=scanned,
        )

    # ================================================================== task boundary

    def _ensure_task(self, source_path: Path) -> str:
        existing = self.kernel.task_manager.active()
        if existing is not None and existing.purpose is TaskPurpose.IMPORT:
            return existing.task_id
        task = self.kernel.task_manager.create(
            self.principal,
            objective=f"reconstruct research state from {source_path.name}",
            purpose=TaskPurpose.IMPORT,
            priority=TaskPriority.PRIMARY,
            stop_condition="all material classified, research state reconstructed, review queue emitted",
            touches_core=True,
            notes=["Import never approves claims; uncertain recoveries go to the review queue."],
        )
        self.kernel.task_manager.start(self.principal, task.task_id)
        return task.task_id

    # ================================================================== extraction

    def _extract_all(self, scanned: ScanResult) -> tuple[list[Fact], ExtractionOutput]:
        combined = ExtractionOutput()
        for file in scanned.files:
            try:
                combined.extend(self._extract_one(file))
            except Exception as exc:  # a single bad file must never abort an import
                combined.unparsed.append(
                    (file.rel_path, f"extractor crashed: {type(exc).__name__}: {exc}")
                )
        return list(combined.facts), combined

    def _extract_one(self, file: SourceFile) -> ExtractionOutput:
        kind = file.kind
        if kind in {
            FileKind.README, FileKind.MARKDOWN, FileKind.LATEX, FileKind.PAPER_DRAFT,
            FileKind.AUDIT_REPORT, FileKind.PLAIN_TEXT,
        }:
            return text_extractors.extract_document(file)
        if kind is FileKind.PYTHON:
            return artifacts.extract_python(file)
        if kind is FileKind.CONFIG:
            return artifacts.extract_config(file)
        if kind is FileKind.DATA_CSV:
            return artifacts.extract_table(file)
        if kind in {FileKind.DATA_JSON, FileKind.NOTEBOOK}:
            return artifacts.extract_json_data(file)
        if kind is FileKind.LOG:
            return artifacts.extract_log(file)
        if kind is FileKind.BIB:
            return artifacts.extract_bib(file)
        if kind is FileKind.PDF:
            return artifacts.extract_pdf(file)
        if kind is FileKind.FIGURE:
            out = ExtractionOutput()
            out.facts.append(
                Fact(
                    kind=FactKind.EVIDENCE,
                    statement=f"figure {file.rel_path} ({file.size} bytes)",
                    rule="figure:artifact",
                    source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT, locator="whole file"),
                    confidence=0.7,
                    payload={"source_file": file.rel_path, "figure": True},
                    files=[file.rel_path],
                )
            )
            return out
        if kind in {FileKind.CODE_OTHER, FileKind.DATA_OTHER, FileKind.BINARY_UNSUPPORTED, FileKind.UNKNOWN}:
            out = ExtractionOutput()
            if file.text:
                out = text_extractors.extract_document(file)
                out.notes.append(f"{file.rel_path}: read as text despite kind={kind.value}")
            else:
                out.unparsed.append((file.rel_path, f"unsupported material ({kind.value})"))
            return out
        return ExtractionOutput()

    @staticmethod
    def _dedupe(facts: Iterable[Fact]) -> list[Fact]:
        """Merge duplicate facts, keeping the highest confidence and unioning the source files."""
        merged: dict[str, Fact] = {}
        for fact in facts:
            key = fact.key()
            if key in merged:
                kept = merged[key]
                if fact.confidence > kept.confidence:
                    kept.confidence = fact.confidence
                    kept.source = fact.source
                    kept.payload = {**kept.payload, **fact.payload}
                kept.files = sorted(set(kept.files) | set(fact.files))
                continue
            merged[key] = fact
        return list(merged.values())

    # ================================================================== reconstruction

    def _reconstruct(
        self,
        *,
        source_path: Path,
        root: Path,
        scanned: ScanResult,
        facts: list[Fact],
        extraction: ExtractionOutput,
        conflict_scan: ConflictScan,
        task_id: str,
    ) -> ImportReport:
        by_kind: dict[FactKind, list[Fact]] = {}
        for fact in facts:
            by_kind.setdefault(fact.kind, []).append(fact)

        question = self._core_question(by_kind.get(FactKind.CORE_QUESTION, []), task_id)
        claims = self._claims(by_kind.get(FactKind.CLAIM, []), task_id)
        rejected = self._rejected_claims(by_kind.get(FactKind.REJECTED_CLAIM, []), task_id)
        experiments = self._experiments(by_kind, root, task_id)
        analyses = self._analyses(by_kind, experiments, root, task_id)
        evidence = self._evidence(by_kind, experiments, analyses, scanned, task_id)
        papers = self._literature(by_kind.get(FactKind.LITERATURE, []), scanned, root, task_id)
        decisions = self._decisions(by_kind.get(FactKind.DECISION, []), task_id)
        questions = self._open_questions(by_kind.get(FactKind.OPEN_QUESTION, []), task_id)
        notes = self._notes(by_kind, scanned, task_id)
        gaps = self._skill_gaps(by_kind.get(FactKind.SKILL_GAP, []), task_id)
        conflicts = self._conflicts(conflict_scan, experiments, task_id)
        timeline_count = self._timeline(by_kind.get(FactKind.GIT_COMMIT, []), source_path, task_id)
        self._unparsed_reviews(extraction, scanned)
        # Persist the review queue: import *proposes*, it never blesses. Everything uncertain is a
        # pending item a human must decide on.
        self.kernel.create_review_items(self.review_items, principal=self.principal)

        # --- state update: everything unguarded in a single revision ---
        self._update_state(
            source_path=source_path,
            question=question,
            claims=claims,
            rejected=rejected,
            experiments=experiments,
            evidence=evidence,
            papers=papers,
            questions=questions,
            gaps=gaps,
            conflicts=conflicts,
            scanned=scanned,
        )

        report = self._build_report(
            source_path=source_path,
            scanned=scanned,
            facts=facts,
            extraction=extraction,
            conflict_scan=conflict_scan,
            question=question,
            claims=claims,
            rejected=rejected,
            experiments=experiments,
            evidence=evidence,
            papers=papers,
            questions=questions,
            gaps=gaps,
            task_id=task_id,
            timeline_count=timeline_count,
        )
        self.kernel.events.append(
            "import.completed",
            actor=self.principal.name,
            task_id=task_id,
            payload={
                "batch_id": self.batch_id,
                "source": str(source_path),
                "confidence": report.confidence.value,
                "created": {k: len(v) for k, v in sorted(self.created.items())},
                "conflicts": len(conflicts),
                "review_items": len(self.review_items),
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.IMPORT,
            f"Imported research material from {source_path.name}",
            detail=(
                f"{scanned.files.__len__()} files scanned; confidence {report.confidence.value}; "
                f"{len(self.review_items)} review item(s)"
            ),
            actor=self.principal.name,
            task_id=task_id,
            state_revision=self.kernel.state.revision(),
            imported=True,
        )
        return report

    # ------------------------------------------------------------------ questions

    def _core_question(self, facts: list[Fact], task_id: str) -> dict[str, Any]:
        """Recover the core question, then change guarded state *through an STR*.

        The STR is the point: even the human running an import cannot silently redefine the
        project — the change is recorded as a reviewable transition with a decision record.
        """
        best = max(facts, key=lambda f: (f.confidence, len(f.statement)), default=None)
        if best is None:
            return {"status": "NOT_FOUND", "statement": None, "confidence": 0.0, "str_id": None}

        confidence = best.confidence
        if best.statement.rstrip().endswith("?"):
            confidence = min(0.95, confidence + 0.12)
        if Path(best.source.path).name.lower().startswith("readme"):
            confidence = min(0.95, confidence + 0.05)
        if best.source.path.endswith((".tex",)) or "/paper/" in best.source.path:
            confidence = max(0.3, confidence - 0.15)
        if len(best.statement.split()) > 45:
            confidence = max(0.3, confidence - 0.1)

        evidence = self._evidence_from_fact(
            best,
            task_id=task_id,
            statement=f"core research question recovered from {best.source.path}",
            kind=SourceKind.AUTHOR_NOTE,
        )
        confirmed = confidence >= 0.7
        status = "CONFIRMED" if confirmed else "PLACEHOLDER"

        statement = best.statement if confirmed else f"[unconfirmed] {best.statement}"
        operations = [
            StateOperation(
                op=StateOp.SET,
                path="core_question",
                value={
                    "statement": statement,
                    "motivation": f"recovered by Research Import from {best.source.path}",
                    "scope": "unknown — confirm or narrow during review",
                    "is_placeholder": not confirmed,
                    "source_refs": [best.source.model_dump(mode="json")],
                },
                note=f"import confidence {confidence:.2f} (rule {best.rule})",
            )
        ]
        strq = self.kernel.transitions.request(
            self.principal,
            operations=operations,
            reason=(
                f"Research Import recovered a core research question from {best.source.path} "
                f"({best.source.locator or 'locator unknown'}); confidence {confidence:.2f}"
            ),
            evidence_ids=[evidence.evidence_id],
            change_class=None,
            task_id=task_id,
            risk=RiskLevel.MEDIUM,
        )
        approved = False
        if confirmed and self.approve_core_question and self.principal.is_human:
            self.kernel.transitions.approve(
                self.principal,
                strq.str_id,
                note="Approved as part of a human-initiated import (explicit human action).",
            )
            approved = True
        else:
            self._review(
                ReviewKind.CORE_QUESTION_CANDIDATE,
                title=f"Confirm the core research question (confidence {confidence:.2f})",
                rationale=(
                    "Import recovered a candidate core question. Guarded research state only changes "
                    "through an approved state transition request."
                ),
                proposed={"statement": best.statement, "str_id": strq.str_id},
                source_refs=[best.source],
                confidence=confidence,
                severity=Severity.HIGH,
                rule=best.rule,
                questions_for_human=[
                    "Is this the question the project is actually trying to answer?",
                    "What is in scope and out of scope for it?",
                ],
            )
        self._record("str", strq.str_id)
        self._record("evidence", evidence.evidence_id)
        return {
            "status": status if approved or confirmed else "PLACEHOLDER",
            "statement": statement,
            "confidence": confidence,
            "str_id": strq.str_id,
            "source": best.source.path,
        }

    # ------------------------------------------------------------------ claims

    def _claims(self, facts: list[Fact], task_id: str) -> list[Claim]:
        candidates = [f for f in facts if not self._is_boilerplate(f)]
        candidates.sort(key=lambda f: (-f.confidence, f.statement))
        created: list[Claim] = []
        for fact in candidates[: self.max_claims]:
            evidence = self._evidence_from_fact(
                fact,
                task_id=task_id,
                statement=f"claim candidate recovered from {fact.source.path}",
                kind=SourceKind.PAPER_PROSE if fact.source.kind.value == "PAPER_PROSE" else SourceKind.AUTHOR_NOTE,
            )
            implied = fact.implied_level or EvidenceLevel.L1_OBSERVATION
            status = claim_status_ceiling(fact)
            claim = Claim(
                statement=fact.statement,
                status=status,
                scope="imported: scope not yet declared — confirm during review",
                evidence_level=EvidenceLevel.L1_OBSERVATION,
                evidence_ids=[evidence.evidence_id],
                limitations=[
                    f"imported from {fact.source.path} ({fact.source.locator or 'locator unknown'})",
                    f"source language implies {implied.value}; available evidence level is L1_OBSERVATION",
                    "scope, assumptions and limitations were not stated in the source material",
                ],
                parent_question=None,
                created_by=self.principal.name,
                task_id=task_id,
                history=[
                    ClaimHistoryEntry(
                        by=self.principal.name,
                        from_status=ClaimStatus.IDEA,
                        to_status=status,
                        reason="created by Research Import; never imported above HYPOTHESIS",
                        evidence_ids=[evidence.evidence_id],
                    )
                ],
                language_strength=implied.value,
            )
            self.kernel.claims.save(claim)
            self._record("claim", claim.claim_id)
            self._record("evidence", evidence.evidence_id)
            created.append(claim)
            self._review(
                ReviewKind.CLAIM_CANDIDATE,
                title=f"Claim candidate: {fact.statement[:90]}",
                rationale=(
                    f"Matched rule {fact.rule!r} in {fact.source.path}. "
                    "Imported at HYPOTHESIS: promotion requires evidence and a lifecycle transition."
                ),
                proposed={"statement": fact.statement, "status": claim.status.value,
                          "implied_language_level": implied.value},
                source_refs=[fact.source],
                confidence=fact.confidence,
                severity=Severity.MEDIUM,
                rule=fact.rule,
                subject_ref=claim.claim_id,
                questions_for_human=[
                    "Is this what you meant, in your own words?",
                    "What is the scope where it is claimed to hold?",
                ],
            )
        return created

    @staticmethod
    def _is_boilerplate(fact: Fact) -> bool:
        lowered = normalise_text(fact.statement)
        return any(token in lowered for token in CLAIM_STOPWORDS)

    # ------------------------------------------------------------------ rejected claims

    def _rejected_claims(self, facts: list[Fact], task_id: str) -> list[Claim]:
        created: list[Claim] = []
        for fact in facts[: self.max_claims]:
            evidence = self._evidence_from_fact(
                fact,
                task_id=task_id,
                statement=f"rejected-claim record recovered from {fact.source.path}",
                kind=SourceKind.AUTHOR_NOTE,
            )
            reason = str(fact.payload.get("reason") or "listed as rejected in the source material")
            basis = (
                RejectionBasis.CONTRADICTED
                if "contradict" in reason.lower() or "not comparable" in reason.lower()
                else RejectionBasis.OTHER
            )
            claim = Claim(
                statement=fact.statement,
                status=ClaimStatus.REJECTED,
                scope="imported: rejected claim retained for research history",
                evidence_level=EvidenceLevel.L0_IDEA,
                evidence_ids=[evidence.evidence_id],
                rejection_reason=reason,
                rejection_basis=basis,
                rejected_at=utcnow(),
                tombstone=True,
                limitations=["rejected before import; retained so the research history is explainable"],
                created_by=self.principal.name,
                task_id=task_id,
                history=[
                    ClaimHistoryEntry(
                        by=self.principal.name,
                        from_status=None,
                        to_status=ClaimStatus.REJECTED,
                        reason=f"recovered as rejected: {reason}",
                        evidence_ids=[evidence.evidence_id],
                    )
                ],
            )
            self.kernel.claims.save(claim)
            self._record("claim", claim.claim_id)
            created.append(claim)
            self._review(
                ReviewKind.REJECTED_CLAIM_CANDIDATE,
                title=f"Rejected claim recovered: {fact.statement[:80]}",
                rationale=(
                    "Import found material describing a rejected/abandoned claim. It is retained as a "
                    "tombstone (never deleted) and needs confirmation of *why* it was rejected."
                ),
                proposed={"statement": fact.statement, "reason": reason, "basis": basis.value},
                source_refs=[fact.source],
                confidence=fact.confidence,
                severity=Severity.MEDIUM,
                rule=fact.rule,
                subject_ref=claim.claim_id,
            )
        return created

    # ------------------------------------------------------------------ experiments

    def _experiments(
        self, by_kind: Mapping[FactKind, list[Fact]], root: Path, task_id: str
    ) -> list[Experiment]:
        metric_facts = [f for f in by_kind.get(FactKind.METRIC_VALUE, []) if f.payload.get("table")]
        config_facts = by_kind.get(FactKind.CONFIG, [])
        code_facts = by_kind.get(FactKind.CODE_SURFACE, [])
        failure_facts = by_kind.get(FactKind.FAILED_RUN, [])

        if not metric_facts:
            # No measured numbers: still reconstruct the *declared* experiments so the researcher
            # can see what the code and configs say, marked PLANNED rather than COMPLETED.
            return self._planned_experiments(config_facts, code_facts, root, task_id)

        # --- group measured values by arm and metric
        arms: dict[str, dict[str, list[Fact]]] = {}
        for fact in metric_facts:
            label = str((fact.payload.get("labels") or {}).get("arm")
                        or (fact.payload.get("labels") or {}).get("variant")
                        or (fact.payload.get("labels") or {}).get("condition")
                        or "default")
            arms.setdefault(label, {}).setdefault(fact.metric() or "metric", []).append(fact)

        config_facts = by_kind.get(FactKind.CONFIG, [])
        necessity_language = self._language_present(by_kind, NECESSITY_LANGUAGE)
        sufficiency_language = self._language_present(by_kind, SUFFICIENCY_LANGUAGE)
        rescue_language = self._language_present(by_kind, RESCUE_LANGUAGE)

        groups = self._arm_groups(sorted(arms))
        created: list[Experiment] = []
        for role, members in groups:
            experiment = self._build_experiment(
                role=role,
                members=members,
                arms=arms,
                config_facts=config_facts,
                code_facts=code_facts,
                failure_facts=failure_facts,
                necessity=necessity_language,
                sufficiency=sufficiency_language,
                rescue=rescue_language,
                root=root,
                task_id=task_id,
                all_arms=sorted(arms),
            )
            if experiment is None:
                continue
            self.kernel.experiments.save(experiment)
            self._record("experiment", experiment.experiment_id)
            created.append(experiment)
            self._review(
                ReviewKind.EXPERIMENT_CANDIDATE,
                title=f"Reconstructed experiment: {experiment.title}",
                rationale=(
                    f"Rebuilt from measured values ({', '.join(members)}) and "
                    f"{len(config_facts)} config file(s). Design flags inferred during import are "
                    "listed in the experiment's limitations."
                ),
                proposed={
                    "title": experiment.title,
                    "status": experiment.status.value,
                    "evidence_level": experiment.evidence_level().value,
                    "design": experiment.design.model_dump(mode="json"),
                },
                source_refs=[
                    f.source for m in members for metrics in arms.get(m, {}).values() for f in metrics[:1]
                ][:1],
                confidence=0.7 if experiment.design.has_matched_conditions else 0.5,
                severity=Severity.HIGH,
                rule="table:metric+config",
                subject_ref=experiment.experiment_id,
                questions_for_human=[
                    "Was the control arm matched on the conditions listed in matched_conditions?",
                    "Any run that should be excluded from this comparison?",
                ],
            )
        return created

    @staticmethod
    def _arm_groups(arm_names: Sequence[str]) -> list[tuple[str, list[str]]]:
        """Group arms into comparison families: primary, budget-matched, and component-removal."""
        primary: list[str] = []
        energy: list[str] = []
        removal: list[str] = []
        for arm in arm_names:
            lowered = arm.lower()
            if any(token in lowered for token in ENERGY_TOKENS):
                energy.append(arm)
            elif any(token in lowered for token in REMOVAL_TOKENS) or REMOVAL_PATTERN.search(lowered):
                removal.append(arm)
            else:
                primary.append(arm)

        groups: list[tuple[str, list[str]]] = []
        if primary:
            groups.append(("primary", primary))
        if energy:
            groups.append(("budget_matched", energy))
        for arm in removal:
            # pair a removal arm with the primary arm it most resembles
            counterpart = max(
                primary,
                key=lambda candidate: (len(set(_tokens(arm)) & set(_tokens(candidate))), -len(candidate)),
                default=None,
            )
            groups.append(("removal", ([counterpart, arm] if counterpart else [arm])))
        return groups

    def _build_experiment(
        self,
        *,
        role: str,
        members: list[str],
        arms: Mapping[str, Mapping[str, list[Fact]]],
        config_facts: Sequence[Fact],
        code_facts: Sequence[Fact],
        failure_facts: Sequence[Fact],
        necessity: list[Fact],
        sufficiency: list[Fact],
        rescue: list[Fact],
        root: Path,
        task_id: str,
        all_arms: Sequence[str],
    ) -> Experiment | None:
        member_metrics = {m: arms[m] for m in members if m in arms}
        if not member_metrics:
            return None

        numeric_metrics = sorted({metric for metrics in member_metrics.values() for metric in metrics})
        seeds = sorted(
            {
                int(str(labels["seed"]))
                for metrics in member_metrics.values()
                for facts in metrics.values()
                for f in facts
                for labels in [f.payload.get("labels") or {}]
                if str(labels.get("seed", "")).isdigit()
            }
        )
        row_counts = {m: max(len(f) for f in metrics.values()) for m, metrics in member_metrics.items()}

        control_name = self._pick_control(members, member_metrics)
        treatment_names = [m for m in members if m != control_name]
        treatment_name = treatment_names[0] if treatment_names else members[0]

        configs = self._configs_for(members, config_facts)
        matched_keys, differing_keys = _compare_configs(configs)
        conflicting_keys: list[str] = []
        manifest = next((f for f in code_facts if f.payload.get("argparse")), None)
        defaults: dict[str, Any] = dict(manifest.payload.get("defaults") or {}) if manifest else {}
        merged_config: dict[str, Any] = {}
        for config in configs.values():
            merged_config.update({k: v for k, v in config.items() if v is not None})

        limitations = [
            "design flags were inferred deterministically during import; verify them in review",
            f"arm grouping rule: role={role}, members={members}",
        ]
        has_control = control_name is not None and len(members) >= 2
        has_matched = bool(has_control and matched_keys and not conflicting_keys)
        manipulated = differing_keys
        # An intervention is "the experimenter set this variable while everything else was held
        # equal". That is exactly what a matched two-arm comparison of one manipulated key gives.
        has_intervention = bool(has_control and has_matched and manipulated)

        necessity_evidence: list[Fact] = []
        sufficiency_evidence: list[Fact] = []
        rescue_evidence: list[Fact] = []
        if role == "removal":
            necessity_evidence = necessity
        if necessity_evidence and len(members) >= 2:
            limitations.append(
                "necessity design inferred: a component was removed and the outcome compared "
                f"(evidence: {necessity_evidence[0].source.path})"
            )
        if sufficiency and len(members) >= 2:
            sufficiency_evidence = sufficiency
            limitations.append(
                f"sufficiency language found in {sufficiency[0].source.path}; no forcing experiment detected"
            )
        if rescue:
            rescue_evidence = rescue
            limitations.append(f"rescue language found in {rescue[0].source.path}")

        setting = "/".join(
            str(value)
            for value in (
                merged_config.get("model") or defaults.get("model"),
                merged_config.get("scale"),
                merged_config.get("dataset") or defaults.get("dataset"),
            )
            if value
        )
        settings = [setting] if setting else []
        design = ExperimentDesign(
            has_control=has_control,
            has_matched_conditions=has_matched,
            has_intervention=has_intervention,
            has_necessity_design=bool(necessity_evidence and len(members) >= 2),
            has_sufficiency_design=bool(sufficiency_evidence and len(members) >= 2),
            has_rescue=bool(rescue_evidence and len(members) >= 2),
            settings=settings,
            is_observational=not has_intervention,
            confounds_identified=[],
            confounds_uncontrolled=[] if has_matched else ["conditions could not be shown matched — no config artifacts"],
        )

        failed_rows = sum(
            1
            for m in members
            for metrics in arms.get(m, {}).values()
            for f in metrics
            if str(f.payload.get("status", "")).startswith("failed")
        )
        has_values = any(
            f.payload.get("value") is not None and not str(f.payload.get("status", "")).startswith("failed")
            for m in members
            for metrics in arms.get(m, {}).values()
            for f in metrics
        )
        status = ExperimentStatus.COMPLETED if has_values else ExperimentStatus.FAILED
        failure_reason = None
        if status is ExperimentStatus.FAILED:
            failure_reason = "no successful measurement row found for this arm"
        elif failed_rows:
            limitations.append(f"{failed_rows} row(s) marked as failed; excluded from descriptions")

        title = (
            f"{' vs '.join(members)} (imported)" if len(members) > 1 else f"{members[0]} (imported)"
        )
        provenance = capture_provenance(
            root,
            config=merged_config or None,
            seeds=seeds,
            command=f"imported from {root.name}",
            data_files=[str(Path(root) / f.payload["table"]) for f in
                        [x for metrics in arms.get(members[0], {}).values() for x in metrics[:1]]],
        )
        git = git_info(root)
        if git.get("commit"):
            provenance.code_commit = str(git["commit"])

        experiment = Experiment(
            title=title,
            research_question="imported: no explicit question found in the config",
            hypothesis="",
            treatment=Arm(
                name=treatment_name,
                description=f"imported arm {treatment_name!r}",
                config_patch={k: v for k, v in merged_config.items() if k in manipulated},
                matched_on=matched_keys,
                is_control=False,
            ),
            control=(
                Arm(
                    name=control_name or "none",
                    description=f"imported comparator arm {control_name!r}",
                    config_patch={},
                    matched_on=matched_keys,
                    is_control=True,
                )
                if control_name
                else None
            ),
            matched_conditions=matched_keys,
            design=design,
            model=str(merged_config.get("model")) if merged_config.get("model") else None,
            dataset=str(merged_config.get("dataset")) if merged_config.get("dataset") else None,
            scale=str(merged_config.get("scale")) if merged_config.get("scale") else None,
            optimizer=str(merged_config.get("optimizer")) if merged_config.get("optimizer") else None,
            learning_rate=_as_float(merged_config.get("learning_rate") or merged_config.get("lr")),
            seeds=seeds or ([int(defaults["seed"])] if str(defaults.get("seed", "")).isdigit() else []),
            n=max(row_counts.values()) if row_counts else None,
            training_budget=TrainingBudget(
                steps=_as_int(merged_config.get("steps") or defaults.get("steps")),
                note=f"imported; default steps={defaults.get('steps')}" if defaults.get("steps") else "imported",
            ),
            metrics=[MetricRef(name=metric, definition="") for metric in numeric_metrics],
            config=merged_config,
            config_hash=sha256_json(merged_config) if merged_config else None,
            limitations=limitations,
            status=status,
            failure_reason=failure_reason,
            provenance=provenance,
            task_id=task_id,
            created_by=self.principal.name,
            source_refs=sorted(
                {f.source.path for m in members for metrics in arms.get(m, {}).values() for f in metrics}
            ),
            tags=["imported", role],
        )
        # a completed experiment must carry a seed and n (schema rule) — record the gap explicitly
        if experiment.status is ExperimentStatus.COMPLETED and (not experiment.seeds or experiment.n is None):
            experiment = experiment.with_updates(
                status=ExperimentStatus.ABORTED,
                failure_reason=(
                    "imported run has results but no complete seed/n bookkeeping; "
                    "treated as incomplete until a human confirms the run grid"
                ),
            )
            limitations.append("marked ABORTED by import: seed/n bookkeeping incomplete")
        if not all(f.is_defined for f in experiment.metrics):
            experiment.metrics = [
                m if m.is_defined else m.with_updates(
                    definition=f"imported from {Path(experiment.source_refs[0]).name if experiment.source_refs else 'data'} column; definition unknown"
                )
                for m in experiment.metrics
            ]
        return experiment

    def _planned_experiments(
        self,
        config_facts: Sequence[Fact],
        code_facts: Sequence[Fact],
        root: Path,
        task_id: str,
    ) -> list[Experiment]:
        """Reconstruct *declared* experiments when no measured values exist at all.

        A legacy project with configs and code but no results is still a research state: it tells us
        what was intended. These experiments are PLANNED (level L0), never inferred as evidence.
        """
        created: list[Experiment] = []
        manifest = next((f for f in code_facts if f.payload.get("argparse")), None)
        defaults = dict(manifest.payload.get("defaults") or {}) if manifest else {}
        metric_names = list(manifest.payload.get("metric_names") or []) if manifest else []

        configs = self._configs_for(
            [str((f.payload.get("config") or {}).get("arm") or Path(f.source.path).stem) for f in config_facts],
            config_facts,
        )
        for fact in config_facts:
            config = dict(fact.payload.get("config") or {})
            arm_name = str(config.get("arm") or Path(fact.source.path).stem)
            experiment = Experiment(
                title=f"{arm_name} (declared, no results found)",
                research_question="imported: no results artifact was found",
                hypothesis="",
                treatment=Arm(name=arm_name, description="declared arm", is_control=False),
                design=ExperimentDesign(
                    has_control=False,
                    has_matched_conditions=False,
                    is_observational=True,
                    confounds_uncontrolled=["no results artifact found; design cannot be checked"],
                ),
                model=str(config.get("model")) if config.get("model") else None,
                dataset=str(config.get("dataset")) if config.get("dataset") else None,
                scale=str(config.get("scale")) if config.get("scale") else None,
                optimizer=str(config.get("optimizer")) if config.get("optimizer") else None,
                learning_rate=_as_float(config.get("learning_rate")),
                seeds=[int(s) for s in (config.get("seeds") or []) if str(s).isdigit()],
                training_budget=TrainingBudget(steps=_as_int(config.get("steps"))),
                metrics=[
                    MetricRef(name=name, definition="declared in the training script")
                    for name in metric_names
                ],
                config=config,
                config_hash=sha256_json(config) if config else None,
                status=ExperimentStatus.PLANNED,
                provenance=capture_provenance(
                    root, config=config or None, command=f"imported from {fact.source.path}"
                ),
                limitations=[
                    "imported as PLANNED: configuration found, no measured results accompanied it",
                    f"declared in {fact.source.path}",
                ],
                task_id=task_id,
                created_by=self.principal.name,
                source_refs=[fact.source.path],
                tags=["imported", "planned"],
            )
            self.kernel.experiments.save(experiment)
            self._record("experiment", experiment.experiment_id)
            created.append(experiment)
            self._review(
                ReviewKind.EXPERIMENT_CANDIDATE,
                title=f"Declared experiment without results: {experiment.title}",
                rationale=(
                    "A configuration file describes this condition but no results artifact was found. "
                    "It is recorded as PLANNED so the research state is complete, not as evidence."
                ),
                proposed={"config": config, "status": experiment.status.value},
                source_refs=[fact.source],
                confidence=0.5,
                severity=Severity.MEDIUM,
                rule="config:declared_experiment",
                subject_ref=experiment.experiment_id,
                questions_for_human=["Were results produced for this configuration? Where are they?"],
            )
        _ = configs
        return created

    @staticmethod
    def _pick_control(members: Sequence[str], metrics: Mapping[str, Mapping[str, list[Fact]]]) -> str | None:
        if len(members) < 2:
            return None
        for member in members:
            if any(token in member.lower() for token in CONTROL_TOKENS):
                return member
        # deterministic fallback: the arm with the smallest mean on the first shared metric loses
        first_metric = sorted({m for metrics_map in metrics.values() for m in metrics_map})[0]
        means: dict[str, float] = {}
        for member, metric_map in metrics.items():
            values = [f.numeric() for f in metric_map.get(first_metric, []) if f.numeric() is not None]
            if values:
                means[member] = sum(values) / len(values)
        if len(means) < 2:
            return sorted(members)[-1]
        return min(means, key=lambda m: (means[m], m))

    @staticmethod
    def _configs_for(members: Sequence[str], config_facts: Sequence[Fact]) -> dict[str, dict[str, Any]]:
        """Attach config files to arms by name proximity (``configs/direct.yaml`` -> ``direct``)."""
        out: dict[str, dict[str, Any]] = {}
        for fact in config_facts:
            config = dict(fact.payload.get("config") or {})
            arm_name = str(config.get("arm") or config.get("variant") or "").strip()
            if not arm_name:
                stem = Path(fact.source.path).stem.lower()
                arm_name = next((m for m in members if m.lower() == stem or m.lower() in stem), "")
            if not arm_name:
                for member in members:
                    if any(_token_overlap(_tokens(member), _tokens(Path(fact.source.path).stem))):
                        arm_name = member
                        break
            if arm_name:
                out.setdefault(arm_name, {}).update(config)
        return out

    @staticmethod
    def _language_present(
        by_kind: Mapping[FactKind, list[Fact]], patterns: Sequence[str]
    ) -> list[Fact]:
        hits: list[Fact] = []
        for kind in (FactKind.CLAIM, FactKind.ANOMALY, FactKind.NOTE, FactKind.DECISION,
                     FactKind.FAILED_RUN, FactKind.REJECTED_CLAIM):
            for fact in by_kind.get(kind, []):
                lowered = fact.statement.lower()
                if any(re.search(pattern, lowered) for pattern in patterns):
                    hits.append(fact)
        return hits

    # ------------------------------------------------------------------ analyses

    def _analyses(
        self,
        by_kind: Mapping[FactKind, list[Fact]],
        experiments: Sequence[Experiment],
        root: Path,
        task_id: str,
    ) -> list[Analysis]:
        """Descriptive statistics per (arm, metric), computed from measured values only."""
        metric_facts = [f for f in by_kind.get(FactKind.METRIC_VALUE, []) if f.payload.get("table")]
        if not metric_facts:
            return []
        by_table: dict[str, list[Fact]] = {}
        for fact in metric_facts:
            by_table.setdefault(str(fact.payload.get("table")), []).append(fact)

        created: list[Analysis] = []
        for table, facts in sorted(by_table.items()):
            grouped: dict[tuple[str, str], list[float]] = {}
            duplicate_notes: list[str] = []
            for fact in facts:
                labels = fact.payload.get("labels") or {}
                arm = str(labels.get("arm") or labels.get("variant") or "default")
                metric = fact.metric() or "metric"
                value = fact.numeric()
                if value is None:
                    continue
                key = (str(metric), arm)
                if key in grouped and any(abs(value - existing) < 1e-12 for existing in grouped[key]):
                    duplicate_notes.append(
                        f"duplicate row {fact.payload.get('row')} for {metric}/{arm} excluded"
                    )
                    continue
                grouped.setdefault(key, []).append(value)

            results: list[StatResult] = []
            for (metric, arm), values in sorted(grouped.items()):
                results.append(_descriptive(f"{arm}.{metric}", values, group=arm))
            # Comparisons are driven by the *reconstructed experiment design*, not by whatever
            # pairs happen to exist in the table: treatment vs control, for each metric they share.
            for experiment in experiments:
                if table not in experiment.source_refs:
                    continue
                treatment = experiment.treatment.name if experiment.treatment else None
                control = experiment.control.name if experiment.control else None
                if not treatment or not control:
                    continue
                for metric in sorted({m for m, _ in grouped}):
                    a_values = grouped.get((metric, treatment))
                    b_values = grouped.get((metric, control))
                    if not a_values or not b_values:
                        continue
                    results.append(
                        _comparison(
                            f"{treatment}_vs_{control}.{metric}", a_values, b_values,
                            treatment, control, metric,
                        )
                    )

            if not results:
                continue
            analysis = Analysis(
                title=f"Imported descriptive statistics over {table}",
                question="What do the measured artifacts say, before any interpretation?",
                experiment_ids=[e.experiment_id for e in experiments if table in e.source_refs],
                method=StatMethod.DESCRIPTIVE,
                method_description=(
                    "Grouped descriptive statistics computed by Research Import directly from the "
                    "tabular artifact; inferential tests only where a two-arm comparison exists."
                ),
                parameters={"table": table, "groups": sorted({arm for _, arm in grouped})},
                deterministic=True,
                results=results,
                warnings=duplicate_notes,
                limitations=[
                    "computed by the importer, not verified by the Analysis Agent",
                    "assumes rows are independent runs; check the run grid",
                ],
                created_by=self.principal.name,
                task_id=task_id,
            )
            self.kernel.analyses.save(analysis)
            self._record("analysis", analysis.analysis_id)
            created.append(analysis)
            self._review(
                ReviewKind.ANALYSIS_CANDIDATE,
                title=f"Verify imported analysis of {table}",
                rationale=(
                    f"{len(results)} descriptive result(s) computed from {len(facts)} measured values. "
                    "Imported analyses are unverified: the Analysis Agent must recompute and verify "
                    "before any claim may use them."
                ),
                proposed={"analysis_id": analysis.analysis_id, "results": len(results),
                          "warnings": duplicate_notes},
                source_refs=[facts[0].source],
                confidence=0.75,
                severity=Severity.MEDIUM,
                rule="table:descriptive",
                subject_ref=analysis.analysis_id,
            )
        return created

    # ------------------------------------------------------------------ evidence

    def _evidence(
        self,
        by_kind: Mapping[FactKind, list[Fact]],
        experiments: Sequence[Experiment],
        analyses: Sequence[Analysis],
        scanned: ScanResult,
        task_id: str,
    ) -> list[Evidence]:
        created: list[Evidence] = []
        # 1. raw data artifacts
        for fact in by_kind.get(FactKind.EVIDENCE, []):
            source_file = fact.payload.get("source_file")
            level = EvidenceLevel.L1_OBSERVATION
            statement = fact.statement
            if fact.payload.get("table"):
                statement = (
                    f"{source_file}: {fact.payload.get('row_count')} rows, "
                    f"numeric columns {', '.join(fact.payload.get('numeric_columns') or []) or 'none'}"
                )
                level = EvidenceLevel.L2_REPRODUCED if (fact.payload.get("row_count") or 0) > 1 else level
            elif fact.payload.get("figure"):
                level = EvidenceLevel.L0_IDEA
            evidence = Evidence(
                statement=statement,
                evidence_type=EvidenceType.RAW,
                source_kind=SourceKind.RAW_EXPERIMENT,
                source_path=source_file,
                artifact_hash=fact.source.sha256,
                source_refs=[fact.source],
                payload={
                    "summary": fact.payload.get("summary"),
                    "columns": fact.payload.get("columns"),
                    "rows": fact.payload.get("row_count"),
                    "imported": True,
                },
                evidence_level=level,
                provenance=capture_provenance(
                    self.kernel.root,
                    data_files=[str(self.kernel.root / source_file)] if source_file else [],
                    command=f"researchos import {self.kernel.root.name}",
                ),
                task_id=task_id,
                created_by=self.principal.name,
                confidence_note="created by Research Import from a measured artifact; not yet verified",
            )
            self.kernel.evidence.save(evidence)
            self._record("evidence", evidence.evidence_id)
            created.append(evidence)

        # 2. analyses become ANALYZED evidence (still unverified)
        for analysis in analyses:
            evidence = Evidence(
                statement=(
                    f"analysis {analysis.analysis_id} computed {len(analysis.results)} descriptive "
                    f"result(s) from {len(analysis.parameters.get('groups', []))} group(s)"
                ),
                evidence_type=EvidenceType.ANALYZED,
                source_kind=SourceKind.VERIFIED_ANALYSIS,
                source_analysis=analysis.analysis_id,
                source_refs=[
                    SourceRef(
                        path=f".researchos/analysis/{analysis.analysis_id}.yaml",
                        kind=SourceKind.VERIFIED_ANALYSIS,
                        locator=analysis.analysis_id,
                    )
                ],
                payload={f"{r.name}.{field}": value for r in analysis.results for field, value in
                         (("mean", r.mean), ("n", r.n), ("std", r.std)) if value is not None},
                evidence_level=EvidenceLevel.L2_REPRODUCED,
                verification_status=VerificationStatus.UNVERIFIED,
                task_id=task_id,
                created_by=self.principal.name,
                confidence_note="ANALYZED but NOT VERIFIED: the importer cannot verify its own output",
            )
            self.kernel.evidence.save(evidence)
            self._record("evidence", evidence.evidence_id)
            created.append(evidence)
        return created

    def _evidence_from_fact(
        self,
        fact: Fact,
        *,
        task_id: str,
        statement: str,
        kind: SourceKind,
    ) -> Evidence:
        """One raw-evidence record per imported statement: the pointer back to the source bytes."""
        evidence = Evidence(
            statement=statement,
            evidence_type=EvidenceType.RAW,
            source_kind=kind,
            source_path=fact.source.path,
            artifact_hash=fact.source.sha256,
            source_refs=[fact.source],
            payload={
                "quote": fact.source.quote,
                "locator": fact.source.locator,
                "rule": fact.rule,
                "extracted_statement": fact.statement,
                "confidence": fact.confidence,
            },
            evidence_level=EvidenceLevel.L1_OBSERVATION,
            verification_status=VerificationStatus.UNVERIFIED,
            task_id=task_id,
            created_by=self.principal.name,
            confidence_note=(
                f"extracted by deterministic rule {fact.rule!r}; a human must confirm the reading"
            ),
        )
        self.kernel.evidence.save(evidence)
        self._record("evidence", evidence.evidence_id)
        return evidence

    # ------------------------------------------------------------------ literature

    def _literature(
        self, facts: list[Fact], scanned: ScanResult, root: Path, task_id: str
    ) -> list[LiteraturePaper]:
        entries = [f for f in facts if f.payload.get("title") and f.payload.get("key")]
        cited = {
            str(f.payload.get("bibtex_key")) for f in facts if f.rule == "latex:cite"
        }
        created: list[LiteraturePaper] = []
        for fact in entries:
            payload = fact.payload
            existing = self.kernel.papers.find(
                lambda p: (p.bibtex_key and p.bibtex_key == payload.get("key"))
                or p.title.lower() == str(payload.get("title", "")).lower()
            )
            if existing is not None:
                continue
            paper = LiteraturePaper(
                title=str(payload.get("title")),
                authors=list(payload.get("authors") or []),
                year=payload.get("year") if isinstance(payload.get("year"), int) else None,
                venue=payload.get("venue"),
                doi=payload.get("doi"),
                arxiv_id=payload.get("arxiv_id"),
                url=payload.get("url"),
                bibtex_key=str(payload.get("key")),
                abstract=None,
                # No full text was read, so nothing mechanism-level may be recorded.
                fulltext_status=FulltextStatus.UNKNOWN,
                providers=[ProviderKind.LOCAL_BIB],
                provider_ids={"local_bib": str(payload.get("key"))},
                raw_record=dict(payload.get("raw_fields") or {}),
                provenance_refs=[fact.source],
                imported_from=fact.source.path,
                screened=False,
                include=str(payload.get("key")) in cited,
                confidence=0.8,
                relevance=0.5 if str(payload.get("key")) in cited else 0.3,
                tags=["imported-bibliography"] + (["cited"] if str(payload.get("key")) in cited else []),
            )
            self.kernel.papers.save(paper)
            self._record("literature_paper", paper.paper_id)
            created.append(paper)

        if created:
            self._review(
                ReviewKind.LITERATURE_CANDIDATE,
                title=f"{len(created)} bibliography entr(y/ies) imported without full text",
                rationale=(
                    "BibTeX metadata only: these papers are ABSTRACT_LEVEL_ONLY/UNKNOWN. No mechanism "
                    "or component-level statements about them may be written until the full text is read."
                ),
                proposed={"papers": [p.title for p in created[:10]]},
                source_refs=[created[0].provenance_refs[0]] if created[0].provenance_refs else [],
                confidence=0.8,
                severity=Severity.MEDIUM,
                rule="bib:entry",
                questions_for_human=[
                    "Which of these did you actually read in full?",
                    "Any key prior work missing from this bibliography?",
                ],
            )
        return created

    # ------------------------------------------------------------------ decisions / questions / notes

    def _decisions(self, facts: list[Fact], task_id: str) -> list[Decision]:
        created: list[Decision] = []
        for fact in facts[: self.max_notes]:
            decision = Decision(
                kind=DecisionKind.OTHER if fact.rule.startswith("pattern") else DecisionKind.EXPERIMENT_ROUTE,
                summary=fact.statement[:300],
                rationale=f"recovered by Research Import (rule {fact.rule}) from {fact.source.path}",
                made_by="unknown (pre-import)",
                task_id=task_id,
                evidence_ids=[],
            )
            self.kernel.decisions.save(decision)
            self._record("decision", decision.decision_id)
            created.append(decision)
            self._review(
                ReviewKind.DECISION_CANDIDATE,
                title=f"Recovered decision: {fact.statement[:80]}",
                rationale="Import recovered a decision-shaped statement; confirm the rationale.",
                proposed=fact.model_dump(mode="json") if hasattr(fact, "model_dump") else {
                    "statement": fact.statement, "rule": fact.rule},
                source_refs=[fact.source],
                confidence=fact.confidence,
                severity=Severity.LOW,
                rule=fact.rule,
                subject_ref=decision.decision_id,
            )
        return created

    def _open_questions(self, facts: list[Fact], task_id: str) -> list[OpenQuestion]:
        created: list[OpenQuestion] = []
        for fact in facts[: self.max_notes]:
            kind = GapKind.OPEN_QUESTION
            lowered = fact.statement.lower()
            if "gap" in lowered or "no prior" in lowered or "has not been" in lowered:
                kind = GapKind.CANDIDATE_GAP
            question = OpenQuestion(
                statement=fact.statement,
                kind=kind,
                priority=TaskPriority.SECONDARY,
                evidence_ids=[],
            )
            self.kernel.store("open_question").save(question)
            self._record("open_question", question.question_id)
            created.append(question)
            self._review(
                ReviewKind.OPEN_QUESTION_CANDIDATE,
                title=f"Recovered open question: {fact.statement[:80]}",
                rationale=(
                    "Recovered from the material. A CANDIDATE_GAP is never a verified novelty claim — "
                    "that requires a completed novelty audit."
                ),
                proposed={"statement": fact.statement, "kind": kind.value},
                source_refs=[fact.source],
                confidence=fact.confidence,
                severity=Severity.LOW,
                rule=fact.rule,
                subject_ref=question.question_id,
            )
        return created

    def _notes(self, by_kind: Mapping[FactKind, list[Fact]], scanned: ScanResult, task_id: str) -> list[AuthorNote]:
        created: list[AuthorNote] = []
        candidates: list[Fact] = []
        for kind in (FactKind.NOTE, FactKind.ANOMALY, FactKind.FAILED_RUN, FactKind.REJECTED_CLAIM):
            candidates.extend(by_kind.get(kind, []))
        diaries = {f.rel_path: (f.text or "") for f in scanned.files if f.kind is FileKind.PLAIN_TEXT}
        diary_paths = {path for path, text in diaries.items() if "diary" in path.lower() or "note" in path.lower()}
        diary_facts = [f for f in candidates if f.source.path in diary_paths]
        ordered = sorted(diary_facts, key=lambda f: (_locator_line(f.source.locator), f.statement))
        for fact in (ordered or candidates)[: self.max_notes]:
            note_kind = NoteKind.DIARY
            lowered = fact.statement.lower()
            if fact.kind is FactKind.ANOMALY:
                note_kind = NoteKind.ANOMALY
            elif fact.kind is FactKind.FAILED_RUN:
                note_kind = NoteKind.FAILED_EXPERIMENT
            elif fact.kind is FactKind.REJECTED_CLAIM:
                note_kind = NoteKind.CLAIM_REVISION
            elif "we decided" in lowered or "we chose" in lowered:
                note_kind = NoteKind.DECISION_NOTE
            elif "because" in lowered or "expected" in lowered:
                note_kind = NoteKind.WHY_THIS_EXPERIMENT
            note = AuthorNote(
                kind=note_kind,
                text=fact.statement,
                author="unknown (pre-import)",
                created_at=_note_timestamp(fact, diaries) or utcnow(),
                source_refs=[fact.source],
                imported=True,
                tags=["imported", fact.rule],
            )
            self.kernel.notes.save(note)
            self._record("author_note", note.note_id)
            created.append(note)
        return created

    def _skill_gaps(self, facts: list[Fact], task_id: str) -> list[SkillGap]:
        created: list[SkillGap] = []
        for fact in facts[:40]:
            capability = _missing_capability(fact.statement)
            gap = SkillGap(
                description=fact.statement[:400],
                missing_capability=capability,
                evidence=[f"{fact.source.path}:{fact.source.locator}"],
                severity=Severity.LOW,
                search_queries=[f"{capability} skill", f"{capability} tool open source"],
            )
            self.kernel.skill_gaps.save(gap)
            self._record("skill_gap", gap.gap_id)
            created.append(gap)
        if created:
            self._review(
                ReviewKind.SKILL_GAP,
                title=f"{len(created)} possible capability gap(s) found in the material",
                rationale=(
                    "The material contains 'this was manual / needs tooling' markers. A gap seen once is "
                    "not a gap; the Skill Critic confirms repeats before any discovery search runs."
                ),
                proposed={"gaps": [g.missing_capability for g in created]},
                source_refs=[f.source for f in facts[:3]],
                confidence=0.4,
                severity=Severity.LOW,
                rule="pattern:todo",
            )
        return created

    # ------------------------------------------------------------------ conflicts

    def _conflicts(
        self, scan_result: ConflictScan, experiments: Sequence[Experiment], task_id: str
    ) -> list[Conflict]:
        created: list[Conflict] = []
        experiment_ids = [e.experiment_id for e in experiments]
        for item in scan_result.number_conflicts:
            measured, narrative = item.measured, item.narrative
            conflict = self.kernel.ledger.create(
                self.principal,
                kind=ConflictKind.VALUE,
                subject=item.metric,
                source_a=ConflictSource(
                    kind=SourceKind.RAW_EXPERIMENT,
                    ref=f"{measured.source.path} ({measured.source.locator or 'row'})",
                    value=measured.numeric(),
                    artifact_hash=measured.source.sha256,
                    locator=measured.source.locator,
                ),
                source_b=ConflictSource(
                    kind=narrative.source.kind,
                    ref=f"{narrative.source.path} ({narrative.source.locator or 'line'})",
                    value=narrative.numeric(),
                    artifact_hash=narrative.source.sha256,
                    locator=narrative.source.locator,
                ),
                difference=item.difference,
                experiment_ids=experiment_ids,
                detected_by="deterministic",
            )
            self._record("conflict", conflict.conflict_id)
            created.append(conflict)
            self._review(
                ReviewKind.CONFLICT,
                title=f"Number disagreement on {item.metric}: {item.difference[:80]}",
                rationale=(
                    "The measured artifact and the written material disagree. The trust order "
                    "(raw experiment > verified analysis > audit > report > prose > AI summary) picks a "
                    "default, but both values are preserved and a human should confirm which is current."
                ),
                proposed={"metric": item.metric, "measured": measured.numeric(),
                          "narrative": narrative.numeric(), "resolution": conflict.resolution.value},
                source_refs=[measured.source, narrative.source],
                confidence=0.8,
                severity=Severity.HIGH if item.is_relative_mismatch else Severity.MEDIUM,
                rule="conflict:number",
                subject_ref=conflict.conflict_id,
                questions_for_human=[
                    "Which value is current?",
                    "Was the written number superseded by a later run?",
                ],
            )
        for duplicate in scan_result.duplicate_runs:
            conflict = self.kernel.ledger.create(
                self.principal,
                kind=ConflictKind.DESIGN,
                subject=f"duplicate run {duplicate.metric} {duplicate.labels}",
                source_a=ConflictSource(kind=SourceKind.RAW_EXPERIMENT,
                                        ref=f"rows {duplicate.rows}", value=duplicate.values),
                source_b=ConflictSource(kind=SourceKind.RAW_EXPERIMENT,
                                        ref="same (metric, arm, seed) key", value=duplicate.values),
                difference=(
                    f"the same run appears {len(duplicate.rows)} times at rows {duplicate.rows}; "
                    + ("values are identical (double count)" if duplicate.identical else "values differ")
                ),
                experiment_ids=experiment_ids,
                auto_resolve_by_trust=False,
            )
            self._record("conflict", conflict.conflict_id)
            created.append(conflict)
        for fact in scan_result.missing_values:
            self._review(
                ReviewKind.EVIDENCE_CANDIDATE,
                title=f"Missing measurement: {fact.metric()} for {fact.payload.get('labels')}",
                rationale="A run row exists but the metric is empty. Decide whether the run failed or the metric was not recorded.",
                proposed=fact.payload,
                source_refs=[fact.source],
                confidence=0.7,
                severity=Severity.MEDIUM,
                rule=fact.rule,
            )
        for message in scan_result.metric_disagreements:
            metric = message.split("metric ")[-1].split(" mentioned")[0].strip("'\"")
            self._review(
                ReviewKind.EVIDENCE_CANDIDATE,
                title=f"Ungrounded number in prose: {metric}",
                rationale=(
                    "A number appears in written material with no matching measured value in any "
                    "artifact. Either the artifact is missing from this import, or the number was never "
                    "produced by an experiment."
                ),
                proposed={"note": message},
                source_refs=[],
                confidence=0.5,
                severity=Severity.HIGH,
                rule="conflict:ungrounded_number",
                questions_for_human=["Where did this number come from?"],
            )
        return created

    def _unparsed_reviews(self, extraction: ExtractionOutput, scanned: ScanResult) -> None:
        for path, reason in extraction.unparsed:
            self._review(
                ReviewKind.UNPARSED_MATERIAL,
                title=f"Could not parse {path}",
                rationale=reason,
                proposed={"path": path, "reason": reason},
                source_refs=[],
                confidence=0.9,
                severity=Severity.MEDIUM,
                rule="scanner:unparsed",
                questions_for_human=["Can you export this material as text/CSV so it can be imported?"],
            )

    # ------------------------------------------------------------------ state / timeline

    def _timeline(self, commits: list[Fact], source_path: Path, task_id: str) -> int:
        count = 0
        for fact in sorted(commits, key=lambda f: str(f.payload.get("date", ""))):
            self.kernel.timeline.record(
                TimelineEventKind.EXPERIMENT_RESULT if "result" in fact.statement.lower()
                else TimelineEventKind.IMPORT,
                fact.statement[:200],
                detail=f"imported git history: {fact.payload.get('commit', '')[:10]}",
                actor=str(fact.payload.get("author") or "unknown"),
                task_id=task_id,
                refs=[str(fact.payload.get("commit", ""))[:12]],
                imported=True,
            )
            count += 1
        self.kernel.timeline.record(
            TimelineEventKind.IMPORT,
            f"Import of {source_path.name} started",
            detail=f"batch {self.batch_id}",
            actor=self.principal.name,
            task_id=task_id,
            imported=True,
        )
        return count + 1

    def _update_state(
        self,
        *,
        source_path: Path,
        question: Mapping[str, Any],
        claims: Sequence[Claim],
        rejected: Sequence[Claim],
        experiments: Sequence[Experiment],
        evidence: Sequence[Evidence],
        papers: Sequence[LiteraturePaper],
        questions: Sequence[OpenQuestion],
        gaps: Sequence[SkillGap],
        conflicts: Sequence[Conflict],
        scanned: ScanResult,
    ) -> None:
        frontier = (
            f"{len(claims)} claim candidate(s), {len(experiments)} reconstructed experiment(s), "
            f"{len(evidence)} evidence record(s); "
            f"{sum(1 for c in conflicts if c.is_unresolved())} unresolved conflict(s) pending review"
        )

        def mutate(state) -> None:
            state.known_evidence = [*state.known_evidence, *(e.evidence_id for e in evidence)]
            state.rejected_claims = [*state.rejected_claims, *(c.claim_id for c in rejected)]
            state.open_questions = [*state.open_questions, *(q.question_id for q in questions)]
            state.current_frontier = Frontier(
                statement=frontier,
                current_focus=[
                    "resolve imported conflicts",
                    "confirm reconstructed experiments",
                    "declare scope for imported claims",
                ],
                blocking_issues=[c.difference[:200] for c in conflicts if c.is_unresolved()][:5],
                next_actions=suggest_next_tasks(self._report_stub(question, claims, conflicts), ConflictScan()),
            )
            state.skill_state = state.skill_state.model_copy(
                update={"gaps": [*state.skill_state.gaps, *(g.gap_id for g in gaps)]}
            )
            state.literature_state = state.literature_state.model_copy(
                update={
                    "papers_retained": state.literature_state.papers_retained + len(papers),
                    "papers_screened": state.literature_state.papers_screened + len(papers),
                    "providers_used": sorted(
                        {*state.literature_state.providers_used, "local_bib"}
                    ) if papers else state.literature_state.providers_used,
                    "coverage_basis": (
                        "imported bibliography only — no literature search has been executed yet, "
                        "so coverage is 0 and no novelty statement is supportable"
                    ),
                    "coverage_score": 0.0,
                    "closest_prior_work": [
                        PriorWorkRef(paper_id=p.paper_id, title=p.title,
                                     note="imported from bibliography; full text not read")
                        for p in list(papers)[:8]
                    ],
                }
            )
            state.import_summary = {
                "source": str(source_path),
                "batch_id": self.batch_id,
                "files_scanned": len(scanned.files),
                "claims": len(claims),
                "experiments": len(experiments),
                "evidence": len(evidence),
                "papers": len(papers),
                "conflicts": len(conflicts),
                "at": utcnow().isoformat(),
            }
            state.notes = [
                *state.notes,
                f"imported from {source_path} (batch {self.batch_id}); review queue holds "
                f"{len(self.review_items)} item(s)",
            ]

        self.kernel.state.update(
            self.principal,
            mutate,
            event_kind="state.import_summary",
            payload={"batch_id": self.batch_id, "source": str(source_path)},
        )

        project = self.kernel.project()
        self.kernel.save_project(
            project.model_copy(
                update={
                    "imported": True,
                    "source_roots": sorted({*project.source_roots, str(source_path)}),
                }
            )
        )

    @staticmethod
    def _report_stub(question: Mapping[str, Any], claims: Sequence[Claim], conflicts: Sequence[Conflict]) -> ImportReport:
        stub = ImportReport(project_name="", source_path="")
        stub.core_question_status = str(question.get("status", "NOT_FOUND"))
        stub.current_core_claims = [c.statement for c in claims[:5]]
        stub.known_conflicts = [c.difference for c in conflicts[:5]]
        return stub

    # ------------------------------------------------------------------ report

    def _build_report(
        self,
        *,
        source_path: Path,
        scanned: ScanResult,
        facts: list[Fact],
        extraction: ExtractionOutput,
        conflict_scan: ConflictScan,
        question: Mapping[str, Any],
        claims: Sequence[Claim],
        rejected: Sequence[Claim],
        experiments: Sequence[Experiment],
        evidence: Sequence[Evidence],
        papers: Sequence[LiteraturePaper],
        questions: Sequence[OpenQuestion],
        gaps: Sequence[SkillGap],
        task_id: str,
        timeline_count: int,
    ) -> ImportReport:
        data_artifacts = [f for f in facts if f.payload.get("table") or f.payload.get("records")]
        configs = [f for f in facts if f.kind is FactKind.CONFIG]
        measured_metrics = {f.metric() for f in facts if f.rule == "table:metric"}
        prose_claims = [c for c in claims if all("PAPER_PROSE" in (lim or "") for lim in c.limitations)]
        claims_in_data = len(claims) - len(prose_claims)
        commits = [f for f in facts if f.kind is FactKind.GIT_COMMIT]

        confidence, signals = assess_confidence(
            has_core_question=bool(question.get("statement")),
            question_confidence=float(question.get("confidence") or 0.0),
            experiment_count=len(experiments),
            data_artifact_count=len(data_artifacts),
            config_count=len(configs),
            claims_in_data=claims_in_data,
            claims_in_prose_only=len(prose_claims),
            git_commits=len(commits),
            conflict_count=len([c for c in conflict_scan.number_conflicts]),
            unparsed_count=len(extraction.unparsed),
            raw_evidence_count=len([e for e in evidence if e.evidence_type is EvidenceType.RAW]),
        )

        strongest = []
        for experiment in sorted(experiments, key=lambda e: (-e.evidence_level().rank, e.title)):
            strongest.append(
                f"{experiment.title} — {experiment.evidence_level().value}, "
                f"{experiment.status.value}, seeds {experiment.seeds or 'unknown'}"
            )
        weakest = [c.statement[:110] for c in claims if "source language implies" in " ".join(c.limitations)][:6]
        if not weakest:
            weakest = [f"prose-only number: {f.statement[:100]}" for f in facts if f.rule == "number:unattributed"][:6]

        report = ImportReport(
            project_name=self.kernel.project().name,
            source_path=str(source_path),
            task_id=task_id,
            batch_id=self.batch_id,
            files_scanned=len(scanned.files),
            files_read=len(scanned.text_files()),
            bytes_scanned=scanned.total_bytes,
            file_kinds=scanned.by_kind(),
            skipped=[f"{path}: {reason}" for path, reason in scanned.skipped[:8]],
            core_question=str(question.get("statement") or "") or None,
            core_question_confidence=float(question.get("confidence") or 0.0),
            core_question_status=str(question.get("status", "NOT_FOUND")),
            current_core_claims=[f"{c.statement[:120]} [{c.status.value}, {c.evidence_level.value}]" for c in claims[:MAX_REPORTED]],
            rejected_claims=[c.statement[:120] for c in rejected[:MAX_REPORTED]],
            major_experiments=[f"{e.title} [{e.status.value}]" for e in experiments[:MAX_REPORTED]],
            strongest_evidence=strongest[:MAX_REPORTED],
            weakest_evidence=weakest,
            known_conflicts=summarise(conflict_scan)[:MAX_REPORTED],
            closest_prior_work=[f"{p.citation_label()}: {p.title[:80]}" for p in papers[:MAX_REPORTED]],
            literature_coverage=(
                f"{len(papers)} bibliography entr(y/ies) imported, 0 full texts verified; "
                "no literature search executed (coverage 0.0)"
                if papers
                else "no literature material found; coverage 0.0"
            ),
            current_frontier=(
                f"{len(claims)} claim candidate(s), {len(experiments)} experiment(s), "
                f"{len(evidence)} evidence record(s), "
                f"{len([c for c in conflict_scan.number_conflicts])} number contradiction(s) found"
            ),
            open_questions=[q.statement[:120] for q in questions[:MAX_REPORTED]],
            possible_skill_gaps=[f"{g.missing_capability}: {g.description[:80]}" for g in gaps[:MAX_REPORTED]],
            confidence=confidence,
            confidence_signals=signals,
            created_objects={k: len(v) for k, v in self.created.items() if v},
            unparsed=[f"{path}: {reason}" for path, reason in extraction.unparsed[:10]],
            review_queue_size=len(self.review_items),
            evidence_levels=_level_histogram(experiments),
            notes=[
                f"{timeline_count} timeline event(s) recorded from imported history",
                f"{len(measured_metrics)} distinct measured metric(s) found in data artifacts",
                "no claim was imported above HYPOTHESIS; nothing was imported as SUPPORTED",
            ]
            + extraction.notes[:5],
        )
        report.suggested_next_tasks = suggest_next_tasks(report, conflict_scan)
        return report

    # ------------------------------------------------------------------ helpers

    def _review(
        self,
        kind: ReviewKind,
        *,
        title: str,
        rationale: str,
        proposed: Mapping[str, Any],
        source_refs: Sequence[Any],
        confidence: float,
        severity: Severity,
        rule: str | None,
        subject_ref: str | None = None,
        questions_for_human: Sequence[str] = (),
    ) -> ReviewItem:
        item = ReviewItem(
            kind=kind,
            title=title,
            rationale=rationale,
            subject_ref=subject_ref,
            proposed_object=dict(proposed),
            confidence=confidence,
            severity=severity,
            source_refs=list(source_refs),
            extraction_rule=rule,
            deterministic=True,
            questions_for_human=list(questions_for_human),
            batch_id=self.batch_id,
        )
        self.review_items.append(item)
        return item

    def _record(self, kind: str, identifier: str) -> None:
        if not identifier:
            return
        self.created.setdefault(kind, []).append(identifier)


# --------------------------------------------------------------------------------------
# module-level helpers
# --------------------------------------------------------------------------------------


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _token_overlap(a: Sequence[str], b: Sequence[str]) -> set[str]:
    return set(a) & set(b)


def _compare_configs(configs: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """Which conditions are matched across arms, and which one was manipulated.

    Bookkeeping keys (``arm``, ``name``, output paths) are not conditions, and budget-accounting keys
    (``matched_energy``, ``steps``…) describe the *control's* construction rather than the treatment.
    Counting either as "the manipulated variable" would make a properly controlled comparison look
    uncontrolled — the exact opposite of what a design analysis should conclude.
    """
    if len(configs) < 2:
        return [], []
    keys = {key for config in configs.values() for key in config}
    matched: list[str] = []
    manipulated: list[str] = []
    for key in sorted(keys):
        if key in CONFIG_META_KEYS:
            continue
        present = [key in config for config in configs.values()]
        if not all(present):
            continue
        values = {canonical_json(config.get(key)) for config in configs.values()}
        if len(values) == 1:
            matched.append(key)
        elif key not in BUDGET_KEYS:
            manipulated.append(key)
    return matched, manipulated


def _descriptive(name: str, values: Sequence[float], *, group: str) -> StatResult:
    n = len(values)
    mean = sum(values) / n if n else None
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 and mean is not None else None
    se = (std / math.sqrt(n)) if std is not None and n else None
    ci_low = ci_high = None
    if mean is not None and se is not None:
        ci_low, ci_high = mean - 1.96 * se, mean + 1.96 * se
    return StatResult(
        name=name,
        value=mean,
        n=n,
        mean=mean,
        std=std,
        se=se,
        median=sorted(values)[n // 2] if n else None,
        min=min(values) if n else None,
        max=max(values) if n else None,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=0.95 if ci_low is not None else None,
        method=StatMethod.DESCRIPTIVE,
        group_a=group,
        inputs=[f"imported:{group}"],
        n_inputs=n,
        warnings=[] if n > 1 else ["single run: no dispersion estimate available"],
        notes=["computed by Research Import from a tabular artifact"] if n > 1 else [],
    )


def _comparison(name: str, a: Sequence[float], b: Sequence[float], arm_a: str, arm_b: str, metric: str) -> StatResult:
    """Welch comparison when a statistics backend is available; otherwise say so explicitly."""
    try:  # the analysis package owns the statistics; the importer must not fork them
        from ..analysis.statistics import welch_t_test  # type: ignore

        result = welch_t_test(list(a), list(b))
        return StatResult(
            name=name,
            value=(result.mean_a - result.mean_b),
            n=len(a) + len(b),
            mean=result.mean_a - result.mean_b,
            std=result.se_diff,
            se=result.se_diff,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            ci_level=result.ci_level,
            effect_size=result.effect_size,
            effect_size_kind=result.effect_size_kind,
            test=StatMethod.WELCH_TTEST,
            statistic=result.t,
            df=result.df,
            p_value=result.p_value,
            paired=False,
            group_a=arm_a,
            group_b=arm_b,
            comparison_family=metric,
            method=StatMethod.WELCH_TTEST,
            n_inputs=len(a) + len(b),
            warnings=list(result.notes),
            notes=[f"Welch comparison between {arm_a} and {arm_b} on {metric}"],
        )
    except Exception as exc:  # noqa: BLE001 - missing backend must degrade honestly
        return StatResult(
            name=name,
            value=(sum(a) / len(a) - sum(b) / len(b)) if a and b else None,
            n=len(a) + len(b),
            mean=(sum(a) / len(a) - sum(b) / len(b)) if a and b else None,
            group_a=arm_a,
            group_b=arm_b,
            comparison_family=metric,
            method=StatMethod.DESCRIPTIVE,
            paired=None,
            n_inputs=len(a) + len(b),
            warnings=[f"no inferential test computed: statistics backend unavailable ({type(exc).__name__})"],
            notes=["a p-value is deliberately not reported rather than approximated"],
        )


def _level_histogram(experiments: Sequence[Experiment]) -> dict[str, int]:
    out: dict[str, int] = {}
    for experiment in experiments:
        level = experiment.evidence_level().value
        out[level] = out.get(level, 0) + 1
    return out


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _locator_line(locator: str | None) -> int:
    if not locator:
        return 0
    match = re.search(r"(\d+)", locator)
    return int(match.group(1)) if match else 0


def _note_timestamp(fact: Fact, diaries: Mapping[str, str]) -> datetime | None:
    """Timestamp a diary sentence with the date heading that precedes it.

    Using the real date keeps the timeline truthful: the import must not collapse months of work
    into "now".
    """
    text = diaries.get(fact.source.path)
    if not text:
        return None
    line = _locator_line(fact.source.locator)
    prefix = "\n".join(text.splitlines()[:line]) if line else text[:400]
    matches = list(re.finditer(r"(\d{4})-(\d{2})-(\d{2})", prefix))
    if not matches:
        return None
    last = matches[-1]
    try:
        return datetime(int(last.group(1)), int(last.group(2)), int(last.group(3)))
    except ValueError:
        return None


def _missing_capability(statement: str) -> str:
    lowered = statement.lower()
    for keyword, capability in (
        ("manually", "automation of a manual research step"),
        ("todo", "unspecified missing capability"),
        ("implement", "tooling implementation"),
        ("automat", "automation"),
        ("check", "verification tooling"),
    ):
        if keyword in lowered:
            return capability
    return "unspecified missing capability"
