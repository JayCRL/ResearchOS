"""The Paper Compiler: research state in, grounded artifact out.

The compiler is deliberately *not* a language model wrapper. It:

1. gathers only approved material (SUPPORTED/ROBUST claims, verified evidence, analysis numbers,
   literature claims with locators, approved interpretations, decisions, author notes),
2. materialises a **number pool** — one :class:`NumberRef` per analysis result field, carrying the
   analysis id and, where available, the artifact hash,
3. builds sentences as *data structures* with those references attached,
4. runs the grounding gates and refuses to call the result a paper if any gate fails.

A writer can be plugged in (``proposer=``) to rephrase sentences, but it can never see a number it
is allowed to change: the numbers live in the sentence metadata and are re-verified afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from ..claims.language import calibrate, overreaches, permitted_verb
from ..kernel.errors import GroundingError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models.analysis import Analysis, StatResult
from ..models.common import ClaimStatus, EvidenceLevel, EvidenceType, Severity, VerificationStatus
from ..models.literature import LiteratureClaim, LiteraturePaper
from ..models.paper import (
    CitationRef,
    GroundedSentence,
    GroundingReport,
    GroundingViolation,
    GroundingViolationCode,
    NumberRef,
    PaperArtifact,
    PaperSection,
    SectionDraft,
)
from ..models.timeline import TimelineEventKind
from .grounding import verify_paper

#: Analysis result fields that may be printed, with how they are rendered.
PRINTABLE_FIELDS: tuple[tuple[str, str], ...] = (
    ("mean", "mean"),
    ("std", "std"),
    ("ci_low", "ci_low"),
    ("ci_high", "ci_high"),
    ("effect_size", "effect_size"),
    ("p_value", "p_value"),
    ("n", "n"),
)

#: Sections the compiler emits, in order.
DEFAULT_SECTIONS: tuple[PaperSection, ...] = (
    PaperSection.ABSTRACT,
    PaperSection.INTRODUCTION,
    PaperSection.RELATED_WORK,
    PaperSection.METHOD,
    PaperSection.EXPERIMENTS,
    PaperSection.RESULTS,
    PaperSection.ANALYSIS,
    PaperSection.LIMITATIONS,
    PaperSection.DISCUSSION,
    PaperSection.CONCLUSION,
)


class TextProposer(Protocol):
    """Optional writer hook. Gets structured facts; returns prose or ``None`` to use the template."""

    def propose(self, section: PaperSection, facts: Mapping[str, object]) -> str | None: ...


@dataclass(frozen=True)
class CompilationContext:
    """Everything the compiler is allowed to read. Assembled once, then treated as read-only."""

    core_question: str
    frontier: str
    claims: tuple[object, ...]
    evidence_by_claim: Mapping[str, tuple[object, ...]]
    analyses: Mapping[str, Analysis]
    numbers: Mapping[str, tuple[NumberRef, ...]]
    experiments: tuple[object, ...]
    literature_claims: Mapping[str, LiteratureClaim]
    papers: Mapping[str, LiteraturePaper]
    interpretations: tuple[object, ...]
    decisions: tuple[object, ...]
    notes: tuple[object, ...]
    unresolved_conflicts: tuple[str, ...]
    warnings: tuple[str, ...]

    def all_numbers(self) -> list[NumberRef]:
        return [number for group in self.numbers.values() for number in group]


class PaperCompiler:
    """Compiles research state into a :class:`PaperArtifact` and verifies it."""

    def __init__(self, kernel: ResearchKernel, *, proposer: TextProposer | None = None) -> None:
        self.kernel = kernel
        self.proposer = proposer

    # ================================================================== context

    def build_context(self, principal: Principal) -> CompilationContext:
        principal.require(Cap.PAPER_READ_APPROVED, "paper.read_approved")
        state = self.kernel.research_state()
        warnings: list[str] = []

        claims = [
            c
            for c in self.kernel.claims.all()
            if c.status in (ClaimStatus.SUPPORTED, ClaimStatus.ROBUST)
        ]
        if not claims:
            warnings.append(
                "no SUPPORTED or ROBUST claim exists yet: the compiler can only describe the "
                "project, not report results"
            )

        analyses = {a.analysis_id: a for a in self.kernel.analyses.all()}
        numbers: dict[str, list[NumberRef]] = {}
        for analysis in analyses.values():
            numbers[analysis.analysis_id] = list(self._number_pool(analysis))

        evidence_by_claim: dict[str, tuple[object, ...]] = {}
        for claim in claims:
            evidence_by_claim[claim.claim_id] = tuple(
                e for e in (self.kernel.evidence.get(i) for i in claim.evidence_ids) if e is not None
            )

        experiments = tuple(self.kernel.experiments.all())
        literature_claims = {c.literature_claim_id: c for c in self.kernel.literature_claims.all()}
        papers = {p.paper_id: p for p in self.kernel.papers.all()}
        interpretations = tuple(i for i in self.kernel.interpretations.all() if i.is_approved)
        decisions = tuple(self.kernel.decisions.all())
        notes = tuple(self.kernel.notes.all())
        unresolved = tuple(c.difference for c in self.kernel.ledger.unresolved())

        if not literature_claims:
            warnings.append(
                "no literature claim with a page/section locator exists, so Related Work cannot cite "
                "anything: run a literature search and record claims with locators"
            )
        unverified = [
            e.evidence_id
            for c in claims
            for e in evidence_by_claim.get(c.claim_id, ())
            if e.verification_status is not VerificationStatus.VERIFIED
        ]
        if unverified:
            warnings.append(
                f"{len(unverified)} evidence record(s) behind reported claims are not verified: "
                f"{unverified[:5]}"
            )

        return CompilationContext(
            core_question=state.core_question.statement if state.core_question else "(no core question)",
            frontier=state.current_frontier.statement,
            claims=tuple(claims),
            evidence_by_claim=evidence_by_claim,
            analyses=analyses,
            numbers={k: tuple(v) for k, v in numbers.items()},
            experiments=experiments,
            literature_claims=literature_claims,
            papers=papers,
            interpretations=interpretations,
            decisions=decisions,
            notes=notes,
            unresolved_conflicts=unresolved,
            warnings=tuple(warnings),
        )

    def _number_pool(self, analysis: Analysis) -> Iterable[NumberRef]:
        """One NumberRef per printable field of every result — the paper's entire numeric vocabulary."""
        for result in analysis.results:
            for field, _label in PRINTABLE_FIELDS:
                value = getattr(result, field, None)
                if value is None or not isinstance(value, (int, float)):
                    continue
                yield NumberRef(
                    value=float(value),
                    unit=result.unit if field in {"value", "mean"} else None,
                    analysis_id=analysis.analysis_id,
                    result_id=result.result_id,
                    result_field=field,
                    artifact_path=analysis.output_artifact.path if analysis.output_artifact else None,
                    artifact_hash=analysis.output_artifact.sha256 if analysis.output_artifact else None,
                    tolerance=max(1e-9, abs(float(value)) * 1e-6),
                    context=f"{result.name} [{field}]",
                )

    def number_for(
        self, context: CompilationContext, analysis_id: str, result_name: str, field: str = "mean"
    ) -> NumberRef | None:
        for number in context.numbers.get(analysis_id, ()):
            if field != number.result_field:
                continue
            if number.context.startswith(f"{result_name} "):
                return number
        return None

    # ================================================================== compile

    def compile(
        self,
        principal: Principal,
        *,
        title: str | None = None,
        sections: Sequence[PaperSection] = DEFAULT_SECTIONS,
        format: str = "markdown",
        task_id: str | None = None,
        strict: bool = True,
    ) -> PaperArtifact:
        principal.require(Cap.PAPER_COMPILE, "paper.compile")
        context = self.build_context(principal)

        artifact = PaperArtifact(
            title=title or self._title(context),
            format=format,
            state_revision=self.kernel.state.revision(),
            compiled_by=principal.name,
            warnings=list(context.warnings),
        )

        builders = {
            PaperSection.ABSTRACT: self._abstract,
            PaperSection.INTRODUCTION: self._introduction,
            PaperSection.RELATED_WORK: self._related_work,
            PaperSection.METHOD: self._method,
            PaperSection.EXPERIMENTS: self._experiments,
            PaperSection.RESULTS: self._results,
            PaperSection.ANALYSIS: self._analysis,
            PaperSection.LIMITATIONS: self._limitations,
            PaperSection.DISCUSSION: self._discussion,
            PaperSection.CONCLUSION: self._conclusion,
            PaperSection.REFERENCES: self._references,
            PaperSection.APPENDIX: self._appendix,
        }
        for section in sections:
            builder = builders.get(section)
            if builder is None:
                continue
            draft = builder(context)
            if draft.sentences:
                artifact.sections.append(draft)
                artifact.claim_ids.extend(
                    cid for s in draft.sentences for cid in s.claim_ids if cid not in artifact.claim_ids
                )
                artifact.evidence_ids.extend(
                    eid for s in draft.sentences for eid in s.evidence_ids if eid not in artifact.evidence_ids
                )
                artifact.literature_claim_ids.extend(
                    c.literature_claim_id
                    for s in draft.sentences
                    for c in s.citations
                    if c.literature_claim_id not in artifact.literature_claim_ids
                )

        if section_abstract := next((s for s in artifact.sections if s.section is PaperSection.ABSTRACT), None):
            artifact.abstract = " ".join(s.text for s in section_abstract.sentences)

        report = verify_paper(artifact, kernel=self.kernel, numbers=context.all_numbers())
        artifact.grounding_report = report
        artifact.readiness_id = None
        self._record_refusals(artifact, report, context)

        self.kernel.paper_artifacts.save(artifact)
        self.kernel.events.append(
            "paper.compiled",
            actor=principal.name,
            task_id=task_id,
            payload={
                "paper_id": artifact.paper_id,
                "sections": len(artifact.sections),
                "words": artifact.word_count(),
                "grounding_passed": report.passed,
                "blocking_violations": len(report.blockers()),
                "claims_used": len(artifact.claim_ids),
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.PAPER,
            f"Paper compiled: {artifact.title}",
            detail=report.summary(),
            actor=principal.name,
            task_id=task_id,
            refs=[artifact.paper_id],
            state_revision=artifact.state_revision,
        )
        if strict and not report.passed:
            # The artifact is stored (the researcher should see what failed), but it is not presented
            # as a paper. Callers decide whether to raise; the CLI prints the violations.
            return artifact
        return artifact

    def _record_refusals(
        self, artifact: PaperArtifact, report: GroundingReport, context: CompilationContext
    ) -> None:
        for violation in report.violations:
            artifact.refused_additions.append(f"{violation.code.value}: {violation.message}")
        if not context.claims:
            artifact.refused_additions.append(
                "RESULTS: refused to write a results section because no claim is SUPPORTED yet"
            )
        if not context.literature_claims:
            artifact.refused_additions.append(
                "RELATED_WORK: refused to name prior work because no literature claim has a locator"
            )

    # ================================================================== prose helpers

    def _sentence(
        self,
        text: str,
        *,
        section: PaperSection,
        level: EvidenceLevel = EvidenceLevel.L1_OBSERVATION,
        claim_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        numbers: Sequence[NumberRef] = (),
        citations: Sequence[CitationRef] = (),
        interpretation_ids: Sequence[str] = (),
        decision_ids: Sequence[str] = (),
        note_ids: Sequence[str] = (),
        load_bearing: bool = True,
    ) -> GroundedSentence:
        """Build one grounded sentence, calibrating the text to the evidence level it rests on."""
        if load_bearing and overreaches(text, level):
            text = calibrate(text, level)
        return GroundedSentence(
            text=text,
            section=section,
            claim_ids=list(claim_ids),
            evidence_ids=list(evidence_ids),
            numbers=list(numbers),
            citations=list(citations),
            interpretation_ids=list(interpretation_ids),
            decision_ids=list(decision_ids),
            author_note_ids=list(note_ids),
            language_level=level,
            is_load_bearing=load_bearing,
        )

    def _format_number(self, number: NumberRef) -> str:
        if number.result_field == "p_value":
            return f"p = {number.value:.3g}" if number.value >= 1e-4 else "p < 0.0001"
        if number.result_field == "ci_low":
            return f"{number.value:.3g}"
        if number.result_field == "ci_high":
            return f"{number.value:.3g}"
        if number.unit == "%":
            return f"{number.value:.1f}%"
        return f"{number.value:.3g}"

    def _claim_level(self, claim) -> EvidenceLevel:
        level = claim.evidence_level
        return level if isinstance(level, EvidenceLevel) else EvidenceLevel(str(level))

    def _values_for(self, context: CompilationContext, claim) -> tuple[str, list[NumberRef]]:
        """The one reported comparison for a claim, rendered from analysis artifacts only."""
        parts: list[str] = []
        refs: list[NumberRef] = []
        for evidence in context.evidence_by_claim.get(claim.claim_id, ()):
            analysis_id = getattr(evidence, "source_analysis", None)
            if not analysis_id or analysis_id not in context.analyses:
                continue
            analysis = context.analyses[analysis_id]
            for result in analysis.results:
                if result.p_value is None:
                    continue
                mean = self.number_for(context, analysis_id, result.name, "mean")
                if mean is None:
                    continue
                p_value = self.number_for(context, analysis_id, result.name, "p_value")
                effect = self.number_for(context, analysis_id, result.name, "effect_size")
                parts.append(
                    f"the measured difference for {result.name} is {self._format_number(mean)}"
                    + (f" ({self._format_number(p_value)})" if p_value else "")
                    + (f" with effect size {self._format_number(effect)}" if effect else "")
                )
                refs.extend([n for n in (mean, p_value, effect) if n is not None])
                break
            if parts:
                break
        return ("; ".join(parts), refs)

    # ================================================================== sections

    def _title(self, context: CompilationContext) -> str:
        question = context.core_question.rstrip("?").strip()
        return question[:180] if question and question != "(no core question)" else "Untitled research artifact"

    def _abstract(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        has_question = context.core_question and context.core_question != "(no core question)"
        sentences.append(
            self._sentence(
                (
                    f"We study whether {context.core_question[0].lower() + context.core_question[1:]}"
                    if context.core_question[:1].isupper()
                    else f"We study whether {context.core_question}"
                )
                if has_question
                else "This artifact reports the current state of a research project whose core "
                "question has not been confirmed yet.",
                section=PaperSection.ABSTRACT,
                level=EvidenceLevel.L1_OBSERVATION,
                load_bearing=False,
            )
        )
        for claim in context.claims[:3]:
            level = self._claim_level(claim)
            statement = calibrate(claim.statement, level)
            value_text, refs = self._values_for(context, claim)
            text = f"{statement.rstrip('.')}"
            if value_text:
                text += f"; {value_text}"
            sentences.append(
                self._sentence(
                    text + ".",
                    section=PaperSection.ABSTRACT,
                    level=level,
                    claim_ids=[claim.claim_id],
                    evidence_ids=list(claim.evidence_ids),
                    numbers=refs,
                )
            )
        if not context.claims:
            sentences.append(
                self._sentence(
                    "This artifact records the current research state; it presents no supported "
                    "result because no claim has reached SUPPORTED status.",
                    section=PaperSection.ABSTRACT,
                    load_bearing=False,
                )
            )
        return SectionDraft(section=PaperSection.ABSTRACT, heading="Abstract", sentences=sentences)

    def _introduction(self, context: CompilationContext) -> SectionDraft:
        has_question = context.core_question and context.core_question != "(no core question)"
        sentences = [
            self._sentence(
                f"This project investigates: {context.core_question}"
                if has_question
                else "This artifact records a research project whose core question is still unconfirmed.",
                section=PaperSection.INTRODUCTION,
                load_bearing=False,
            )
        ]
        if context.frontier:
            sentences.append(
                self._sentence(
                    f"Current frontier: {context.frontier}",
                    section=PaperSection.INTRODUCTION,
                    load_bearing=False,
                )
            )
        for claim in context.claims:
            level = self._claim_level(claim)
            sentences.append(
                self._sentence(
                    f"We test the claim that {calibrate(claim.statement, level).rstrip('.').lower()}."
                    if not claim.statement[:1].isupper()
                    else f"We test the claim: {calibrate(claim.statement, level)}",
                    section=PaperSection.INTRODUCTION,
                    level=level,
                    claim_ids=[claim.claim_id],
                )
            )
        for note in context.notes[:2]:
            sentences.append(
                self._sentence(
                    f"Research notes record: {note.text[:200]}",
                    section=PaperSection.INTRODUCTION,
                    load_bearing=False,
                    note_ids=[note.note_id],
                )
            )
        # The narrative follows the research that actually happened, not a template arc.
        from .voice import ResearcherVoice

        for text, section, note_ids, decision_ids, claim_ids in ResearcherVoice(self.kernel).narrative_sentences(limit=3):
            if section is not PaperSection.INTRODUCTION:
                continue
            sentences.append(
                self._sentence(
                    text,
                    section=section,
                    load_bearing=False,
                    note_ids=note_ids,
                    decision_ids=decision_ids,
                    claim_ids=claim_ids,
                )
            )
        return SectionDraft(section=PaperSection.INTRODUCTION, heading="Introduction", sentences=sentences)

    def _related_work(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        by_paper: dict[str, list[LiteratureClaim]] = {}
        for claim in context.literature_claims.values():
            by_paper.setdefault(claim.paper_id, []).append(claim)
        for paper_id in sorted(by_paper):
            paper = context.papers.get(paper_id)
            if paper is None:
                continue
            for claim in by_paper[paper_id]:
                citation = CitationRef(
                    literature_claim_id=claim.literature_claim_id,
                    paper_id=paper_id,
                    page=claim.page,
                    section=claim.section,
                    quote=claim.quote,
                    fulltext_verified=claim.fulltext_verified,
                    display_key=paper.bibtex_key or paper.citation_label(),
                )
                level = EvidenceLevel.L3_CONTROLLED if claim.fulltext_verified else EvidenceLevel.L1_OBSERVATION
                sentences.append(
                    self._sentence(
                        f"{paper.citation_label()} reports that {claim.statement.rstrip('.')}.",
                        section=PaperSection.RELATED_WORK,
                        level=level,
                        citations=[citation],
                    )
                )
        if not sentences:
            for paper in list(context.papers.values())[:8]:
                sentences.append(
                    self._sentence(
                        f"{paper.title} is related work whose details have not yet been verified "
                        "against its full text.",
                        section=PaperSection.RELATED_WORK,
                        load_bearing=False,
                    )
                )
        return SectionDraft(section=PaperSection.RELATED_WORK, heading="Related Work", sentences=sentences)

    def _method(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for experiment in context.experiments:
            design_bits = []
            if experiment.treatment:
                design_bits.append(f"treatment {experiment.treatment.name!r}")
            if experiment.control:
                design_bits.append(f"control {experiment.control.name!r}")
            matched = ", ".join(experiment.matched_conditions[:6])
            text = f"{experiment.title}: {', '.join(design_bits) or 'single condition'}"
            if matched:
                text += f", matched on {matched}"
            if experiment.design.has_intervention:
                text += "; the manipulated variable was set by the experimenter"
            sentences.append(
                self._sentence(
                    text + ".",
                    section=PaperSection.METHOD,
                    level=EvidenceLevel.L1_OBSERVATION,
                    load_bearing=False,
                )
            )
            if not context.numbers:
                sentences.append(
                    self._sentence(
                        "Hyperparameters are recorded in the run configuration artifact; the compiler "
                        "prints a value only when an analysis artifact produced it.",
                        section=PaperSection.METHOD,
                        load_bearing=False,
                    )
                )
        return SectionDraft(section=PaperSection.METHOD, heading="Method", sentences=sentences)

    def _experiments(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for experiment in context.experiments:
            text = (
                f"{experiment.title} ran with status {experiment.status.value}"
                f" on {experiment.model or 'an unspecified model'}"
                f" over {experiment.dataset or 'an unspecified dataset'}"
            )
            if experiment.seeds:
                text += f" across {len(experiment.seeds)} seeds"
            sentences.append(
                self._sentence(
                    text + ".",
                    section=PaperSection.EXPERIMENTS,
                    level=EvidenceLevel.L1_OBSERVATION,
                    evidence_ids=list(experiment.evidence_ids),
                    load_bearing=False,
                )
            )
            if experiment.design.confounds_uncontrolled:
                sentences.append(
                    self._sentence(
                        f"Uncontrolled conditions for {experiment.title}: "
                        + "; ".join(experiment.design.confounds_uncontrolled[:3])
                        + ".",
                        section=PaperSection.EXPERIMENTS,
                        load_bearing=False,
                    )
                )
        return SectionDraft(section=PaperSection.EXPERIMENTS, heading="Experiments", sentences=sentences)

    def _results(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for claim in context.claims:
            level = self._claim_level(claim)
            statement = calibrate(claim.statement, level)
            value_text, refs = self._values_for(context, claim)
            text = statement if statement.endswith(".") else statement + "."
            if value_text:
                text = f"{text.rstrip('.')} — {value_text}."
            sentences.append(
                self._sentence(
                    text,
                    section=PaperSection.RESULTS,
                    level=level,
                    claim_ids=[claim.claim_id],
                    evidence_ids=list(claim.evidence_ids),
                    numbers=refs,
                )
            )
        if not sentences:
            sentences.append(
                self._sentence(
                    "No claim has reached SUPPORTED status, so this artifact reports no results.",
                    section=PaperSection.RESULTS,
                    load_bearing=False,
                )
            )
        return SectionDraft(section=PaperSection.RESULTS, heading="Results", sentences=sentences)

    def _analysis(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for interpretation in context.interpretations:
            level = EvidenceLevel.L3_CONTROLLED
            sentences.append(
                self._sentence(
                    interpretation.statement,
                    section=PaperSection.ANALYSIS,
                    level=level,
                    evidence_ids=list(interpretation.evidence_ids),
                    claim_ids=list(interpretation.claim_ids),
                    interpretation_ids=[interpretation.interpretation_id],
                )
            )
            for alternative in interpretation.alternative_explanations[:3]:
                sentences.append(
                    self._sentence(
                        f"An alternative explanation we considered: {alternative}.",
                        section=PaperSection.ANALYSIS,
                        load_bearing=False,
                    )
                )
        if not sentences:
            sentences.append(
                self._sentence(
                    "No interpretation has been approved, so this artifact offers none. "
                    "Interpretations become citable only after explicit approval.",
                    section=PaperSection.ANALYSIS,
                    load_bearing=False,
                )
            )
        return SectionDraft(section=PaperSection.ANALYSIS, heading="Analysis", sentences=sentences)

    def _limitations(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for claim in context.claims:
            for limitation in claim.limitations[:4]:
                sentences.append(
                    self._sentence(
                        f"Regarding the claim that {claim.statement.rstrip('.').lower()}: {limitation}.",
                        section=PaperSection.LIMITATIONS,
                        load_bearing=False,
                        claim_ids=[claim.claim_id],
                    )
                )
        for experiment in context.experiments:
            settings = {experiment.model, experiment.scale, experiment.dataset} - {None}
            if len(settings) <= 1 and experiment.status.value == "COMPLETED":
                sentences.append(
                    self._sentence(
                        f"{experiment.title} was run in a single setting, so cross-setting "
                        "generalisation is untested.",
                        section=PaperSection.LIMITATIONS,
                        load_bearing=False,
                    )
                )
        for conflict in context.unresolved_conflicts[:5]:
            sentences.append(
                self._sentence(
                    f"Unresolved source disagreement: {conflict}",
                    section=PaperSection.LIMITATIONS,
                    load_bearing=False,
                )
            )
        # Compiler *warnings* (unverified evidence, missing literature locators, unconfirmed core
        # question) are deliberately NOT rendered into the paper: they are metadata about the
        # compilation, they carry numerals the paper cannot ground, and a reader of the paper should
        # not have to parse the tool's own bookkeeping. They live on ``artifact.warnings``.
        return SectionDraft(section=PaperSection.LIMITATIONS, heading="Limitations", sentences=sentences)

    def _discussion(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for decision in context.decisions[:5]:
            sentences.append(
                self._sentence(
                    f"Design decision ({decision.kind.value}): {decision.summary} — {decision.rationale}",
                    section=PaperSection.DISCUSSION,
                    load_bearing=False,
                    decision_ids=[decision.decision_id],
                )
            )
        open_questions = [
            q for q in self.kernel.store("open_question").all() if q.resolved_at is None
        ]
        for question in open_questions[:6]:
            sentences.append(
                self._sentence(
                    f"Open question ({question.kind.value}): {question.statement}",
                    section=PaperSection.DISCUSSION,
                    load_bearing=False,
                )
            )
        from .voice import ResearcherVoice

        for text, section, note_ids, decision_ids, claim_ids in ResearcherVoice(self.kernel).narrative_sentences():
            if section is not PaperSection.DISCUSSION:
                continue
            sentences.append(
                self._sentence(
                    text,
                    section=section,
                    load_bearing=False,
                    note_ids=note_ids,
                    decision_ids=decision_ids,
                    claim_ids=claim_ids,
                )
            )
        return SectionDraft(section=PaperSection.DISCUSSION, heading="Discussion", sentences=sentences)

    def _conclusion(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for claim in context.claims:
            level = self._claim_level(claim)
            sentences.append(
                self._sentence(
                    f"Within the settings studied, {calibrate(claim.statement, level).rstrip('.').lower()} "
                    f"({permitted_verb(level)}).",
                    section=PaperSection.CONCLUSION,
                    level=level,
                    claim_ids=[claim.claim_id],
                    evidence_ids=list(claim.evidence_ids),
                )
            )
        if not sentences:
            sentences.append(
                self._sentence(
                    "This artifact states no conclusion, because no claim has earned one.",
                    section=PaperSection.CONCLUSION,
                    load_bearing=False,
                )
            )
        return SectionDraft(section=PaperSection.CONCLUSION, heading="Conclusion", sentences=sentences)

    def _references(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for paper in sorted(context.papers.values(), key=lambda p: (p.year or 0, p.title)):
            if not paper.include and paper.bibtex_key is None:
                continue
            authors = ", ".join(paper.authors[:4]) or "unknown authors"
            sentence = (
                f"{paper.bibtex_key or paper.title}: {authors} ({paper.year or 'n.d.'}), "
                f"{paper.venue or 'venue unknown'}"
            )
            if paper.doi or paper.arxiv_id:
                sentence += f", {paper.doi or paper.arxiv_id}"
            sentences.append(
                self._sentence(
                    sentence + ". Full text read: "
                    + ("yes" if paper.fulltext_status.value == "FULLTEXT_VERIFIED" else "no")
                    + ".",
                    section=PaperSection.REFERENCES,
                    load_bearing=False,
                )
            )
        return SectionDraft(section=PaperSection.REFERENCES, heading="References", sentences=sentences)

    def _appendix(self, context: CompilationContext) -> SectionDraft:
        sentences: list[GroundedSentence] = []
        for evidence in (e for group in context.evidence_by_claim.values() for e in group):
            sentences.append(
                self._sentence(
                    f"{evidence.evidence_id} ({evidence.evidence_type.value}, "
                    f"{evidence.verification_status.value}): {evidence.statement}",
                    section=PaperSection.APPENDIX,
                    load_bearing=False,
                    evidence_ids=[evidence.evidence_id],
                )
            )
        return SectionDraft(section=PaperSection.APPENDIX, heading="Appendix: evidence ledger", sentences=sentences)


def compile_paper(
    kernel: ResearchKernel,
    principal: Principal | None = None,
    **kwargs,
) -> PaperArtifact:
    """Convenience wrapper: compile as the human principal unless told otherwise."""
    compiler = PaperCompiler(kernel, proposer=kwargs.pop("proposer", None))
    return compiler.compile(principal or kernel.human(), **kwargs)
