"""Paper readiness: seven *separate* dimensions, computed from stored data, never collapsed.

Why there is no single score
----------------------------
A paper is not "82% ready". Readiness is a set of independent scientific obligations — does every
claim have evidence, are the statistics complete, does the language match the mechanism evidence,
was the literature actually searched, does every claim trace to an experiment, can the runs be
reproduced, and is the writing permitted by the claim lifecycle. Each of those can be blocked while
the others are fine, so collapsing them into one number is exactly the false precision this system
exists to prevent (:class:`~researchos.models.paper.PaperReadiness` has no total field, by design).

Conventions (both are asserted in the tests)
--------------------------------------------
1. **Not evaluated ≠ failed.** When a dimension has no evaluable input at all (for example
   reproducibility with zero stored experiments, or statistical completeness when no claim has an
   analysis), we report ``score = NOT_EVALUATED_SCORE`` (1.0), a ``basis`` beginning with
   ``"NOT EVALUATED"``, and a specific ``unknowns`` entry naming what could not be looked at.
   ``ReadinessDimensionResult.score`` is a required float in ``[0, 1]``, so a neutral value plus an
   explicit unknown is the only honest encoding: writing 0.0 would read as "we looked, and it
   failed". A score below 1.0 therefore always means *an evaluated deficit*.
2. **Unknowns are never scored away.** Anything we cannot check (unverifiable artifacts, claims
   with no analysis, experiments with no stored design) is listed in ``unknowns`` and left visible.

Nothing here is inferred from prose: evidence levels come from ``kernel.levels.aggregate_level`` and
``assess_experiment_level_from_design``, coverage comes from ``ResearchState.literature_state``, and
artifact verification re-hashes the bytes on disk. Standard library only, no LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ..kernel.levels import aggregate_level
from ..kernel.permissions import Cap, Principal
from ..kernel.provenance import verify_artifact
from ..models import (
    Analysis,
    ArtifactRef,
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceLevel,
    EvidenceType,
    Experiment,
    ExperimentStatus,
    PaperReadiness,
    ReadinessDimension,
    ReadinessDimensionResult,
    StatResult,
    VerificationStatus,
)
from ..models.experiment import assess_experiment_level_from_design
from .style_audit import NOVELTY_PHRASES, required_evidence_level

#: Score reported for a dimension that could not be evaluated. See convention (1) in the docstring.
NOT_EVALUATED_SCORE: float = 1.0

#: Prefix of the ``basis`` of a not-evaluated dimension, so callers can detect the convention.
NOT_EVALUATED_PREFIX: str = "NOT EVALUATED"

#: Levels at or above this one assert a mechanism/causal relation and therefore require a
#: full-text-verified source before they may be written into a related-work or mechanism sentence.
MECHANISM_LEVEL: EvidenceLevel = EvidenceLevel.L4_INTERVENTION

#: Number of readiness lines kept in ``research_state.notes`` when note recording is enabled.
MAX_READINESS_NOTES: int = 25

_NOTE_PREFIX = "readiness:"


def _truncate(text: str, limit: int = 110) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "\u2026"


def _max_level(levels: Sequence[EvidenceLevel]) -> EvidenceLevel:
    if not levels:
        return EvidenceLevel.L0_IDEA
    return max(levels, key=lambda level: level.rank)


def _not_evaluated(
    dimension: ReadinessDimension, basis: str, *unknowns: str
) -> ReadinessDimensionResult:
    """Build the documented "could not be evaluated" result (convention 1)."""
    return ReadinessDimensionResult(
        dimension=dimension,
        score=NOT_EVALUATED_SCORE,
        basis=f"{NOT_EVALUATED_PREFIX}: {basis}",
        blockers=[],
        unknowns=[u for u in unknowns if u],
        hard_blocked=False,
    )


class ReadinessAssessor:
    """Assesses whether the stored research state can support a paper, one dimension at a time."""

    def __init__(self, kernel) -> None:
        self.kernel = kernel

    # ================================================================== entry point

    def assess(
        self,
        principal: Principal,
        *,
        claim_ids: Sequence[str] = (),
        task_id: str | None = None,
        record_note: bool = False,
    ) -> PaperReadiness:
        """Assess all seven dimensions and return them *separately*.

        ``claim_ids`` defaults to the core claims in research state; when no core claims are
        declared, every stored claim is used and the substitution is recorded in ``notes`` (a silent
        substitution would let a project look readier than the state says it is).

        ``record_note`` appends one summary line to ``research_state.notes``. It is off by default
        because an assessment is a *read*: state mutation should be asked for explicitly.
        """
        claims, notes = self._resolve_claims(claim_ids)
        dimensions = [
            self.evidence_completeness(claims),
            self.statistical_completeness(claims),
            self.mechanism_evidence(claims),
            self.literature_coverage(),
            self.claim_grounding(claims),
            self.reproducibility(),
            self.writing_readiness(claims),
        ]
        order = {dimension: index for index, dimension in enumerate(ReadinessDimension)}
        dimensions.sort(key=lambda item: order[item.dimension])

        readiness = PaperReadiness(dimensions=dimensions, notes=notes)
        if record_note:
            self._record_note(principal, readiness, task_id=task_id)
        return readiness

    # ================================================================== dimension 1

    def evidence_completeness(self, claims: Sequence[Claim]) -> ReadinessDimensionResult:
        """Fraction of claims that may be *written* with at least one resolvable evidence record.

        A claim with no evidence is not a weak claim, it is not a claim (invariant 4), so the
        blockers name the claims that have nothing behind them.
        """
        claims = list(claims)
        dimension = ReadinessDimension.EVIDENCE_COMPLETENESS
        if not claims:
            return _not_evaluated(
                dimension,
                "no claims are available, so evidence completeness cannot be evaluated",
                "no claim was supplied and research state declares no core claim: there is nothing "
                "to check evidence against",
            )

        promotable: list[Claim] = []
        blockers: list[str] = []
        unknowns: list[str] = []
        for claim in claims:
            evidence, missing = self._claim_evidence(claim)
            if claim.status in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED):
                blockers.append(
                    f"claim {claim.claim_id} is {claim.status.value} (retained tombstone) and cannot "
                    f"appear in a paper: {_truncate(claim.statement)}"
                )
                continue
            if not claim.evidence_ids:
                blockers.append(
                    f"claim {claim.claim_id} ({claim.status.value}) has no evidence id: "
                    f"{_truncate(claim.statement)}"
                )
                continue
            if not evidence:
                blockers.append(
                    f"claim {claim.claim_id} references {len(missing)} evidence id(s) that are not "
                    f"stored: {', '.join(missing)}"
                )
                continue
            if not claim.promotable_to_paper:
                blockers.append(
                    f"claim {claim.claim_id} is only {claim.status.value}: evidence exists, but the "
                    "claim has not been promoted to paper language"
                )
                continue
            promotable.append(claim)
            if max(e.evidence_type.rank for e in evidence) < EvidenceType.VERIFIED.rank:
                unknowns.append(
                    f"claim {claim.claim_id}: the strongest supporting record is "
                    f"{_max_evidence_type(evidence).value}, not VERIFIED — the numbers behind this "
                    "claim have not been independently re-checked"
                )

        score = len(promotable) / len(claims)
        return ReadinessDimensionResult(
            dimension=dimension,
            score=round(score, 4),
            basis=(
                f"{len(promotable)}/{len(claims)} claims are promotable to paper language with at "
                "least one evidence id that resolves to a stored Evidence record"
            ),
            blockers=blockers,
            unknowns=unknowns,
            hard_blocked=bool(blockers),
        )

    # ================================================================== dimension 2

    def statistical_completeness(self, claims: Sequence[Claim]) -> ReadinessDimensionResult:
        """Of the claims with analyses: how many have n>1, a CI, an effect size, no blocking conflict.

        A claim with no analysis at all is *not* counted as a statistical failure; it is recorded as
        an unknown, because "no analysis exists" and "an analysis exists but is incomplete" are
        different problems with different fixes.
        """
        claims = list(claims)
        dimension = ReadinessDimension.STATISTICAL_COMPLETENESS
        if not claims:
            return _not_evaluated(
                dimension,
                "no claims are available, so statistical completeness cannot be evaluated",
                "no claim was supplied and research state declares no core claim",
            )

        evaluated = 0
        complete = 0
        blockers: list[str] = []
        unknowns: list[str] = []
        for claim in claims:
            analyses = self._claim_analyses(claim)
            if not analyses:
                unknowns.append(
                    f"claim {claim.claim_id}: no analysis artifact is linked, so n, confidence "
                    "interval and effect size could not be checked"
                )
                continue
            evaluated += 1
            results = [result for analysis in analyses for result in analysis.results]
            missing_pieces: list[str] = []
            if not any(result.n is not None and result.n > 1 for result in results):
                missing_pieces.append("n>1")
            if not any(
                result.ci_low is not None and result.ci_high is not None for result in results
            ):
                missing_pieces.append("confidence interval")
            if not any(result.effect_size is not None for result in results):
                missing_pieces.append("effect size")
            conflicts = self.kernel.ledger.blocking_for_claim(claim.claim_id)
            if conflicts:
                missing_pieces.append(
                    f"{len(conflicts)} blocking conflict(s): "
                    + "; ".join(_truncate(item.difference, 60) for item in conflicts)
                )
            if any(not analysis.is_verified for analysis in analyses):
                unknowns.append(
                    f"claim {claim.claim_id}: analysis "
                    f"{', '.join(a.analysis_id for a in analyses if not a.is_verified)} is not "
                    "verified, so its numbers are provisional"
                )
            if missing_pieces:
                blockers.append(
                    f"claim {claim.claim_id}: missing {', '.join(missing_pieces)} "
                    f"(analyses {', '.join(a.analysis_id for a in analyses)})"
                )
            else:
                complete += 1

        if evaluated == 0:
            return _not_evaluated(
                dimension,
                "no claim has a linked analysis artifact, so statistical completeness cannot be "
                "evaluated",
                *unknowns,
                "no analyses are stored for any claim: this is an unknown, not a measured zero",
            )

        return ReadinessDimensionResult(
            dimension=dimension,
            score=round(complete / evaluated, 4),
            basis=(
                f"{complete}/{evaluated} claims that have analyses carry n>1, a confidence interval "
                "and an effect size with no blocking unresolved conflict"
            ),
            blockers=blockers,
            unknowns=unknowns,
            hard_blocked=bool(blockers),
        )

    # ================================================================== dimension 3

    def mechanism_evidence(self, claims: Sequence[Claim]) -> ReadinessDimensionResult:
        """The evidence rung actually reached, versus the strength of the language in the claims.

        The rung comes from ``aggregate_level`` (which caps declared levels by experiment design and
        evidence maturity) and the design ceiling from
        :func:`~researchos.models.experiment.assess_experiment_level_from_design`. A design that
        *could* support an intervention but whose evidence only reaches a correlation is reported as
        an unknown — the experiment happened, the analysis has not caught up.
        """
        claims = list(claims)
        dimension = ReadinessDimension.MECHANISM_EVIDENCE
        if not claims:
            return _not_evaluated(
                dimension,
                "no claims are available, so mechanism evidence cannot be compared with language",
                "no claim was supplied and research state declares no core claim",
            )

        evaluated = 0
        within = 0
        blockers: list[str] = []
        unknowns: list[str] = []
        achieved_levels: list[EvidenceLevel] = []
        design_levels: list[EvidenceLevel] = []
        for claim in claims:
            evidence, _missing_evidence = self._claim_evidence(claim)
            experiments, missing_experiments = self._claim_experiments(claim)
            designs = [assess_experiment_level_from_design(item.design) for item in experiments]
            evidence_rung = (
                aggregate_level(
                    evidence,
                    {item.experiment_id: item for item in experiments},
                    {item.analysis_id: item for item in self.kernel.analyses.all()},
                )
                if evidence
                else EvidenceLevel.L0_IDEA
            )
            design_ceiling = _max_level(designs)
            achieved_levels.append(evidence_rung)
            design_levels.append(design_ceiling)

            for experiment_id in missing_experiments:
                unknowns.append(
                    f"claim {claim.claim_id} references experiment {experiment_id} which is not "
                    "stored, so its design — and therefore the evidence ceiling it licenses — is "
                    "unknown"
                )
            if not evidence and not experiments:
                unknowns.append(
                    f"claim {claim.claim_id}: no evidence or experiment is linked, so no mechanism "
                    "rung can be assessed"
                )
                continue

            evaluated += 1
            required = required_evidence_level(claim.statement)
            if required.rank > evidence_rung.rank:
                blockers.append(
                    f"claim {claim.claim_id} is written at {required.value} but its evidence reaches "
                    f"only {evidence_rung.value}: {_truncate(claim.statement)}"
                )
            else:
                within += 1
            if design_ceiling.rank > evidence_rung.rank:
                unknowns.append(
                    f"claim {claim.claim_id}: the registered design could support "
                    f"{design_ceiling.value} but the stored evidence reaches {evidence_rung.value} — "
                    "the design is stronger than the analysis so far"
                )

        if evaluated == 0:
            return _not_evaluated(
                dimension,
                "no claim has linked evidence or experiments, so no mechanism rung can be compared "
                "with the claim language",
                *unknowns,
                "no evidence or experiment is linked to any claim: this is an unknown, not a "
                "measured zero",
            )

        return ReadinessDimensionResult(
            dimension=dimension,
            score=round(within / evaluated, 4),
            basis=(
                f"highest evidence rung reached across the assessed claims is "
                f"{_max_level(achieved_levels).value} (highest design ceiling "
                f"{_max_level(design_levels).value}); {within}/{evaluated} assessed claims use "
                "language within their achieved rung"
            ),
            blockers=blockers,
            unknowns=unknowns,
            hard_blocked=bool(blockers),
        )

    # ================================================================== dimension 4

    def literature_coverage(self) -> ReadinessDimensionResult:
        """Coverage measured by the literature layer, reported with its own basis string.

        Coverage is never assumed: the score is whatever ``literature_state`` says, and a project
        that has run no query at all is blocked from writing related work or any novelty sentence.
        """
        dimension = ReadinessDimension.LITERATURE_COVERAGE
        state = self.kernel.research_state()
        literature = state.literature_state
        claims, _notes = self._resolve_claims(())

        blockers: list[str] = []
        unknowns: list[str] = []
        if literature.queries_executed == 0:
            blockers.append(
                "no literature query has been executed (queries_executed == 0): related work cannot "
                "be written and no novelty statement is licensed by the recorded coverage"
            )
        mechanism_claims = [
            claim
            for claim in claims
            if required_evidence_level(claim.statement).rank >= MECHANISM_LEVEL.rank
            or "mechanism" in claim.statement.lower()
        ]
        if mechanism_claims and literature.fulltext_verified == 0:
            blockers.append(
                f"{len(mechanism_claims)} claim(s) assert a mechanism or causal relation but no "
                "source has fulltext_status FULLTEXT_VERIFIED: abstracts cannot back a "
                "mechanism-level citation (ARCHITECTURE §4.12)"
            )
        if literature.queries_executed > 0 and literature.coverage_score == 0.0:
            unknowns.append(
                "queries ran but coverage_score is still 0.0: it has not been computed from a "
                "QueryPlan, so coverage is unknown rather than zero"
            )
        if literature.fulltext_verified > 0 and literature.papers_screened == 0:
            unknowns.append(
                "fulltext_verified > 0 while papers_screened == 0: the screening counters and the "
                "full-text counters disagree"
            )
        if literature.last_search_at is None and literature.queries_executed > 0:
            unknowns.append("queries executed but no search timestamp is recorded")

        return ReadinessDimensionResult(
            dimension=dimension,
            score=round(literature.coverage_score, 4),
            basis=(
                f"literature coverage_score {literature.coverage_score:.2f} from stored state "
                f"({literature.queries_executed} queries, {literature.papers_screened} papers "
                f"screened, {literature.fulltext_verified} full texts verified); basis: "
                f"{_truncate(literature.coverage_basis, 160)}"
            ),
            blockers=blockers,
            unknowns=unknowns,
            hard_blocked=bool(blockers),
        )

    # ================================================================== dimension 5

    def claim_grounding(self, claims: Sequence[Claim]) -> ReadinessDimensionResult:
        """Fraction of claims whose full chain resolves: evidence + experiment (+ literature if novel).

        A claim with no experiment link is a blocker by name: without it the sentence cannot be
        traced to an artifact, which is the whole point of the compilation model.
        """
        claims = list(claims)
        dimension = ReadinessDimension.CLAIM_GROUNDING
        if not claims:
            return _not_evaluated(
                dimension,
                "no claims are available, so claim grounding cannot be evaluated",
                "no claim was supplied and research state declares no core claim",
            )

        grounded = 0
        blockers: list[str] = []
        unknowns: list[str] = []
        for claim in claims:
            evidence, _missing_evidence = self._claim_evidence(claim)
            experiments, missing_experiments = self._claim_experiments(claim)
            asserts_novelty = self._asserts_novelty(claim)
            literature, missing_literature = self._claim_literature(claim)

            if not experiments:
                blockers.append(
                    f"claim {claim.claim_id}: no experiment link "
                    f"(supporting={claim.supporting_experiments}, "
                    f"contradicting={claim.contradicting_experiments}) — the claim cannot be traced "
                    "to an artifact"
                )
                continue
            if not evidence:
                blockers.append(f"claim {claim.claim_id}: no resolvable evidence record")
                continue
            for experiment_id in missing_experiments:
                unknowns.append(
                    f"claim {claim.claim_id}: referenced experiment {experiment_id} is not stored"
                )
            for literature_id in missing_literature:
                unknowns.append(
                    f"claim {claim.claim_id}: referenced literature record {literature_id} is not "
                    "stored, so its citation cannot be resolved"
                )
            if asserts_novelty and not literature:
                blockers.append(
                    f"claim {claim.claim_id} asserts novelty but has no resolvable literature "
                    "reference: novelty needs a searched coverage, not an assertion"
                )
                continue
            grounded += 1

        return ReadinessDimensionResult(
            dimension=dimension,
            score=round(grounded / len(claims), 4),
            basis=(
                f"{grounded}/{len(claims)} claims resolve evidence + experiment"
                " (+ literature where novelty is asserted)"
            ),
            blockers=blockers,
            unknowns=unknowns,
            hard_blocked=bool(blockers),
        )

    # ================================================================== dimension 6

    def reproducibility(self) -> ReadinessDimensionResult:
        """Fraction of experiments whose provenance, seeds, config hash and artifacts all check out.

        Artifacts that cannot be *looked at* (missing file, no recorded hash) are reported as
        unknowns; an artifact whose bytes no longer match the recorded hash is a blocker, because
        that is a detected change rather than an unverifiable one.
        """
        dimension = ReadinessDimension.REPRODUCIBILITY
        experiments = self.kernel.experiments.all()
        if not experiments:
            return _not_evaluated(
                dimension,
                "no experiment is stored, so reproducibility cannot be evaluated",
                "zero experiments are registered: reproducibility is unknown, not zero. Register "
                "and run an experiment before reading this dimension",
            )

        evaluable = [item for item in experiments if item.status is not ExperimentStatus.PLANNED]
        unknowns: list[str] = []
        for item in experiments:
            if item.status is ExperimentStatus.PLANNED:
                unknowns.append(
                    f"experiment {item.experiment_id} is still PLANNED: it has no run to reproduce"
                )
        if not evaluable:
            return _not_evaluated(
                dimension,
                "every registered experiment is still PLANNED, so nothing has a run to reproduce",
                *unknowns,
            )

        blockers: list[str] = []
        reproducible = 0
        for experiment in evaluable:
            provenance = experiment.provenance
            missing = provenance.missing()
            problems: list[str] = []
            if missing:
                problems.append(f"provenance incomplete (missing {', '.join(missing)})")
            if not experiment.seeds:
                problems.append("no seeds recorded")
            if not (experiment.config_hash or provenance.config_hash):
                problems.append("no config hash")
            for reference in [*experiment.raw_artifacts, *experiment.analysis_artifacts]:
                status, detail = self._verify_artifact(reference)
                if status is VerificationStatus.VERIFIED:
                    continue
                if status is VerificationStatus.HASH_MISMATCH:
                    problems.append(
                        f"artifact {reference.path} no longer matches its recorded hash ({detail})"
                    )
                else:
                    unknowns.append(
                        f"experiment {experiment.experiment_id} artifact {reference.path} cannot be "
                        f"verified ({status.value}: {detail}) — the artifact is unverifiable, not "
                        "proven changed"
                    )
            if experiment.status in (ExperimentStatus.FAILED, ExperimentStatus.ABORTED):
                unknowns.append(
                    f"experiment {experiment.experiment_id} is {experiment.status.value}: its record "
                    "is retained and its provenance is still checked"
                )
            if problems:
                blockers.append(
                    f"experiment {experiment.experiment_id}: " + "; ".join(problems)
                )
            else:
                reproducible += 1

        return ReadinessDimensionResult(
            dimension=dimension,
            score=round(reproducible / len(evaluable), 4),
            basis=(
                f"{reproducible}/{len(evaluable)} non-planned experiments have complete provenance "
                "(commit, config hash, seeds, timestamp, command) with every recorded artifact "
                "re-hashing to the same digest"
            ),
            blockers=blockers,
            unknowns=unknowns,
            hard_blocked=bool(blockers),
        )

    # ================================================================== dimension 7

    def writing_readiness(self, claims: Sequence[Claim]) -> ReadinessDimensionResult:
        """Fraction of claims the claim lifecycle permits to appear as paper language.

        Blocking audits are subtracted here rather than in their own dimension because a style or
        statistical audit that blocks publication blocks *writing*, not the underlying experiment.
        """
        claims = list(claims)
        dimension = ReadinessDimension.WRITING_READINESS
        if not claims:
            return _not_evaluated(
                dimension,
                "no claims are available, so writing readiness cannot be evaluated",
                "no claim was supplied and research state declares no core claim",
            )

        blocked_by_audit = self._blocking_audits_by_claim()
        ready = 0
        blockers: list[str] = []
        unknowns: list[str] = []
        for claim in claims:
            audit_hits = blocked_by_audit.get(claim.claim_id, [])
            if audit_hits:
                blockers.append(
                    f"claim {claim.claim_id} is blocked by "
                    + "; ".join(
                        f"audit {audit.audit_id} ({audit.kind.value}, {audit.verdict.value}) "
                        f"finding {finding.code}"
                        for audit, finding in audit_hits
                    )
                )
                continue
            if not claim.promotable_to_paper:
                blockers.append(
                    f"claim {claim.claim_id} is {claim.status.value}: the lifecycle does not permit "
                    "it to be stated as a paper finding yet"
                )
                continue
            if not claim.audit_ids:
                unknowns.append(
                    f"claim {claim.claim_id}: no audit id is recorded, so its language has not been "
                    "independently checked"
                )
            ready += 1

        if not self.kernel.paper_artifacts.all():
            unknowns.append(
                "no compiled paper exists yet, so style-audit findings against the rendered text are "
                "not part of this dimension"
            )

        return ReadinessDimensionResult(
            dimension=dimension,
            score=round(ready / len(claims), 4),
            basis=(
                f"{ready}/{len(claims)} claims may be written as paper language and are not named by "
                "any blocking audit finding"
            ),
            blockers=blockers,
            unknowns=unknowns,
            hard_blocked=bool(blockers),
        )

    # ================================================================== lookups

    def _resolve_claims(
        self, claim_ids: Sequence[str]
    ) -> tuple[list[Claim], list[str]]:
        """Resolve the claim set for an assessment, recording any substitution in ``notes``."""
        notes: list[str] = []
        wanted = [cid for cid in claim_ids if cid]
        if wanted:
            found: list[Claim] = []
            for claim_id in wanted:
                claim = self.kernel.claims.get(claim_id)
                if claim is None:
                    notes.append(f"claim {claim_id} was requested but is not stored; skipped")
                else:
                    found.append(claim)
            return found, notes
        state = self.kernel.research_state()
        if state.core_claims:
            found = []
            for claim_id in state.core_claims:
                claim = self.kernel.claims.get(claim_id)
                if claim is None:
                    notes.append(f"core claim {claim_id} is referenced by research state but missing")
                else:
                    found.append(claim)
            return found, notes
        every = self.kernel.claims.all()
        if every:
            notes.append(
                "research state declares no core claim, so all "
                f"{len(every)} stored claims were assessed"
            )
        return every, notes

    def _claim_evidence(self, claim: Claim) -> tuple[list[Evidence], list[str]]:
        evidence: list[Evidence] = []
        missing: list[str] = []
        for evidence_id in claim.evidence_ids:
            record = self.kernel.evidence.get(evidence_id)
            if record is None:
                missing.append(evidence_id)
            else:
                evidence.append(record)
        return evidence, missing

    def _claim_experiments(self, claim: Claim) -> tuple[list[Experiment], list[str]]:
        ids: list[str] = list(
            dict.fromkeys([*claim.supporting_experiments, *claim.contradicting_experiments])
        )
        for evidence_id in claim.evidence_ids:
            record = self.kernel.evidence.get(evidence_id)
            if record is None:
                continue
            for experiment_id in [*record.experiment_ids, *([record.source_experiment] if record.source_experiment else [])]:
                if experiment_id and experiment_id not in ids:
                    ids.append(experiment_id)
        experiments: list[Experiment] = []
        missing: list[str] = []
        for experiment_id in ids:
            record = self.kernel.experiments.get(experiment_id)
            if record is None:
                missing.append(experiment_id)
            else:
                experiments.append(record)
        return experiments, missing

    def _claim_analyses(self, claim: Claim) -> list[Analysis]:
        """Analyses that belong to a claim: explicit links plus those its evidence came from."""
        found: dict[str, Analysis] = {}
        for analysis in self.kernel.analyses.all():
            if claim.claim_id in analysis.claim_ids:
                found[analysis.analysis_id] = analysis
        for evidence_id in claim.evidence_ids:
            record = self.kernel.evidence.get(evidence_id)
            if record is None or not record.source_analysis:
                continue
            analysis = self.kernel.analyses.get(record.source_analysis)
            if analysis is not None:
                found[analysis.analysis_id] = analysis
        for experiment in self._claim_experiments(claim)[0]:
            for analysis_id in experiment.analysis_ids:
                analysis = self.kernel.analyses.get(analysis_id)
                if analysis is not None:
                    found[analysis.analysis_id] = analysis
        return [found[key] for key in sorted(found)]

    def _claim_literature(self, claim: Claim) -> tuple[list[str], list[str]]:
        found: list[str] = []
        missing: list[str] = []
        for literature_id in claim.literature_ids:
            if (
                self.kernel.literature_claims.get(literature_id) is not None
                or self.kernel.papers.get(literature_id) is not None
            ):
                found.append(literature_id)
            else:
                missing.append(literature_id)
        return found, missing

    @staticmethod
    def _asserts_novelty(claim: Claim) -> bool:
        lowered = claim.statement.lower()
        if any(phrase in lowered for phrase in NOVELTY_PHRASES):
            return True
        if "novel" in lowered or "novelty" in lowered:
            return True
        return any("novel" in tag.lower() for tag in claim.tags)

    def _blocking_audits_by_claim(self) -> Mapping[str, list[tuple[object, object]]]:
        blocked: dict[str, list[tuple[object, object]]] = {}
        for audit in self.kernel.audits.all():
            for finding in audit.findings:
                if not finding.blocks_publication:
                    continue
                for claim_id in finding.claim_ids:
                    blocked.setdefault(claim_id, []).append((audit, finding))
        return blocked

    def _verify_artifact(self, reference: ArtifactRef) -> tuple[VerificationStatus, str | None]:
        """Re-hash one artifact through the kernel's provenance engine when it exposes one.

        The kernel currently exports ``verify_artifact`` as a module function taking a root, so both
        call shapes are supported; whichever is used, the status comes from re-reading the bytes.
        """
        engine = getattr(self.kernel, "provenance", None)
        candidate = getattr(engine, "verify_artifact", None) if engine is not None else None
        if callable(candidate):
            try:
                outcome = candidate(reference)
            except TypeError:
                outcome = None
            if isinstance(outcome, tuple) and len(outcome) == 2:
                return outcome  # type: ignore[return-value]
        root = getattr(self.kernel, "root", None)
        return verify_artifact(Path(root) if root is not None else Path("."), reference)

    # ================================================================== notes

    def _record_note(
        self, principal: Principal, readiness: PaperReadiness, *, task_id: str | None
    ) -> None:
        """Append one readiness line to ``research_state.notes`` (bounded, never the whole report)."""
        line = _NOTE_PREFIX + " " + ", ".join(
            f"{item.dimension.value}={item.score:.2f}" for item in readiness.dimensions
        ) + f" ({len(readiness.blockers())} blocker(s))"
        actor = getattr(principal, "name", None) or "system"

        def mutate(state) -> None:
            kept = [note for note in state.notes if not note.startswith(_NOTE_PREFIX)]
            state.notes = [*kept, line][-MAX_READINESS_NOTES:]

        self.kernel.state.update(
            actor,
            mutate,
            event_kind="paper.readiness_assessed",
            payload={
                "blockers": len(readiness.blockers()),
                "dimensions": len(readiness.dimensions),
            },
            task_id=task_id,
        )


def _max_evidence_type(evidence: Sequence[Evidence]) -> EvidenceType:
    return max(evidence, key=lambda item: item.evidence_type.rank).evidence_type


__all__ = [
    "MECHANISM_LEVEL",
    "NOT_EVALUATED_PREFIX",
    "NOT_EVALUATED_SCORE",
    "ReadinessAssessor",
    "StatResult",
]
