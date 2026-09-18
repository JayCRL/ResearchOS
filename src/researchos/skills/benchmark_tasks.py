"""The default benchmark catalogue: the fixed capability tasks a skill must pass to earn ACTIVE.

Why this module exists
----------------------
ResearchOS refuses to judge a skill by "the code runs". A skill earns a status by producing the
*finding codes the rest of ResearchOS already emits* on a task whose correct answer is known in
advance. That makes evaluation a deterministic comparison — required findings present, forbidden
(false-positive) findings absent — instead of a vibe check.

Two design decisions worth stating:

* **Task ids are content-derived, not random.** ``btk_`` + 26 Crockford characters derived from the
  task name, so the catalogue is byte-identical on every machine. A benchmark run recorded last
  month is still comparable with one recorded today, and re-seeding is idempotent.
* **A task declares both what must be found and what must not be found.** ``required_findings``
  are the problems actually present in the fixture; ``forbidden_findings`` are the plausible
  *false alarms* a poorly calibrated skill emits (flagging an adequate n, "correcting" a single
  comparison, blocking a correctly cited sentence). Over-flagging is a capability failure, and the
  score must show it.

The finding codes below are the ones the statistical, mechanism, literature and writing auditors
elsewhere in ResearchOS emit; a skill that cannot speak that vocabulary cannot be benchmarked and
therefore cannot be activated.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models import BenchmarkSuite, BenchmarkTask

#: Same Crockford base32 alphabet as ``models.common.new_id`` (no I, L, O, U).
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

#: The finding vocabulary of the four capability suites, keyed by ``BenchmarkSuite`` value.
FINDING_CODES_BY_SUITE: Mapping[str, tuple[str, ...]] = {
    "LITERATURE": (
        "CITATION_UNRESOLVED",
        "COVERAGE_INSUFFICIENT",
        "MISSED_TERMINOLOGY",
        "PRIOR_ART_MATRIX_INCOMPLETE",
    ),
    "EXPERIMENT": (
        "NO_ANALYSIS_ARTIFACT",
        "INSUFFICIENT_N",
        "MULTIPLE_COMPARISONS_UNCORRECTED",
        "BASELINE_NOT_MATCHED",
        "SEED_INCONSISTENCY",
        "EXCLUSION_UNRECORDED",
    ),
    "MECHANISM": (
        "CAUSAL_LANGUAGE_WITHOUT_INTERVENTION",
        "NECESSITY_CLAIM_WITHOUT_REMOVAL",
        "SINGLE_SETTING_GENERALIZATION",
        "ALTERNATIVE_EXPLANATION_UNELIMINATED",
    ),
    "WRITING": (
        "UNGROUNDED_NUMBER",
        "LANGUAGE_EXCEEDS_EVIDENCE",
        "NOVELTY_UNSUPPORTED",
        "TEMPLATE_PHRASE",
    ),
}

#: Every code the default catalogue can require or forbid. Used by discovery to recognise a
#: capability gap hiding inside a free-text finding statement.
ALL_FINDING_CODES: frozenset[str] = frozenset(
    code for codes in FINDING_CODES_BY_SUITE.values() for code in codes
)


def deterministic_task_id(name: str) -> str:
    """Derive a stable ``btk_`` id from a task name.

    Random ids would make two catalogues seeded on two machines incomparable, and would make
    ``seed_default_tasks`` non-idempotent. A content hash keeps the fixture set reproducible.
    """
    digest = hashlib.sha256(name.strip().encode("utf-8")).digest()
    value = int.from_bytes(digest[:17], "big")
    chars: list[str] = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "btk_" + "".join(reversed(chars))


def _task(
    *,
    suite: BenchmarkSuite,
    name: str,
    prompt: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
    criteria: tuple[str, ...],
    hard_fail: str,
    minutes: float = 5.0,
) -> BenchmarkTask:
    """Build one catalogue task. ``weight`` stays 1.0 for every task on purpose.

    ``BenchmarkTask.weight`` exists as a field, but ``BenchmarkRun.compute_score`` currently
    averages every outcome with weight 1.0 (see REQUESTED CHANGES). Declaring per-task weights the
    model ignores would be a claim the code does not honour.
    """
    for code in (*required, *forbidden):
        if code not in ALL_FINDING_CODES:
            raise ValueError(f"benchmark task {name!r} uses unknown finding code {code!r}")
    return BenchmarkTask(
        benchmark_task_id=deterministic_task_id(name),
        suite=suite,
        name=name,
        prompt=prompt,
        fixture=None,
        expected_criteria=list(criteria),
        required_findings=list(required),
        forbidden_findings=list(forbidden),
        hard_fail_conditions=[hard_fail],
        max_minutes=minutes,
        weight=1.0,
    )


def default_catalogue() -> list[BenchmarkTask]:
    """The fixed, deterministic capability suite: 19 tasks over literature/experiment/mechanism/writing.

    Every task scores a *capability*, never an execution. Each declares the findings that must be
    produced on its fixture, and the false alarms that hard-fail it.
    """
    lit, exp, mech, wr = (
        BenchmarkSuite.LITERATURE,
        BenchmarkSuite.EXPERIMENT,
        BenchmarkSuite.MECHANISM,
        BenchmarkSuite.WRITING,
    )
    return [
        # ---------------------------------------------------------------- literature
        _task(
            suite=lit,
            name="find the earliest prior work using historical terminology",
            prompt=(
                "Given a prior-art fixture whose papers describe retrieval interference using the "
                "term 'catastrophic forgetting' and one 1998 paper using 'retroactive interference', "
                "report the terminology the modern query set misses."
            ),
            required=("MISSED_TERMINOLOGY",),
            forbidden=("COVERAGE_INSUFFICIENT",),
            criteria=(
                "the output names the historical term 'retroactive interference'",
                "the output names at least one paper that uses it, with a resolvable identifier",
                "the output does not claim the search coverage itself was insufficient",
            ),
            hard_fail=(
                "Reporting COVERAGE_INSUFFICIENT on this fixture is a false alarm: every required "
                "query family returned results, so the finding would block a correct review."
            ),
        ),
        _task(
            suite=lit,
            name="resolve every citation to a verified identifier before it enters the matrix",
            prompt=(
                "A prior-art matrix was assembled from search snippets. One row cites a paper by "
                "title only, with no DOI or arXiv id, and one row cites a DOI that does not resolve. "
                "Report the citation defect rather than silently dropping the row."
            ),
            required=("CITATION_UNRESOLVED",),
            forbidden=("PRIOR_ART_MATRIX_INCOMPLETE",),
            criteria=(
                "the unresolved row is named",
                "the row is retained in the matrix and marked, not deleted",
                "no claim of completeness is made for the matrix",
            ),
            hard_fail=(
                "Reporting PRIOR_ART_MATRIX_INCOMPLETE here is a false alarm: the matrix has the "
                "required rows and columns; only one citation is unresolvable."
            ),
        ),
        _task(
            suite=lit,
            name="report literature coverage honestly when a query family returned nothing",
            prompt=(
                "A query plan declares four families (terminology, mechanism, benchmark, negative "
                "results). The negative-results family returned zero records. State the coverage "
                "shortfall and its consequence for any novelty statement."
            ),
            required=("COVERAGE_INSUFFICIENT",),
            forbidden=("MISSED_TERMINOLOGY",),
            criteria=(
                "the empty family is named",
                "any novelty statement is marked as unsupported rather than dropped",
                "the report does not invent a terminology defect",
            ),
            hard_fail=(
                "Reporting MISSED_TERMINOLOGY here is a false alarm: the terminology sweep was the "
                "family that did return results."
            ),
        ),
        _task(
            suite=lit,
            name="complete a prior-art matrix without turning UNKNOWN into FALSE",
            prompt=(
                "A prior-art matrix has three cells with no evidence either way. Fill the matrix and "
                "report what is missing, keeping three-valued logic intact."
            ),
            required=("PRIOR_ART_MATRIX_INCOMPLETE",),
            forbidden=("CITATION_UNRESOLVED",),
            criteria=(
                "every unevidenced cell stays UNKNOWN",
                "the missing evidence is reported as a matrix defect",
                "no UNKNOWN cell is reported as FALSE",
            ),
            hard_fail=(
                "Reporting CITATION_UNRESOLVED is a false alarm: every cited identifier in this "
                "fixture resolves."
            ),
        ),
        _task(
            suite=lit,
            name="expand a capability query set before any novelty verdict is issued",
            prompt=(
                "A novelty audit is about to conclude 'no match found' after two queries. Expand the "
                "query set across the declared families, re-run, and report both the coverage gap and "
                "the historical terminology the expanded set reveals."
            ),
            required=("COVERAGE_INSUFFICIENT", "MISSED_TERMINOLOGY"),
            forbidden=("CITATION_UNRESOLVED",),
            criteria=(
                "at least two additional query families are recorded",
                "the coverage shortfall is reported",
                "the historical terminology found by the expanded set is reported",
            ),
            hard_fail=(
                "Reporting CITATION_UNRESOLVED is a false alarm: the fixture's identifiers resolve."
            ),
        ),
        # ---------------------------------------------------------------- experiment
        _task(
            suite=exp,
            name="detect a confound in an unmatched baseline",
            prompt=(
                "Two runs are compared: the treatment ran with the matched tokenizer and data order, "
                "the baseline differs in tokenizer, data order and learning rate. Report the design "
                "defect rather than the win."
            ),
            required=("BASELINE_NOT_MATCHED",),
            forbidden=("NO_ANALYSIS_ARTIFACT",),
            criteria=(
                "the unmatched conditions are named individually",
                "the comparison is marked as confounded",
                "the reported delta is not promoted to a result",
            ),
            hard_fail=(
                "Reporting NO_ANALYSIS_ARTIFACT is a false alarm: the analysis artifact exists and "
                "is hash-verified in this fixture."
            ),
        ),
        _task(
            suite=exp,
            name="refuse to accept a result with no analysis artifact",
            prompt=(
                "A completed experiment reports a metric delta in prose, but no analysis artifact is "
                "registered and no provenance config hash is recorded. Report the missing artifact "
                "and withhold the result."
            ),
            required=("NO_ANALYSIS_ARTIFACT",),
            forbidden=("BASELINE_NOT_MATCHED",),
            criteria=(
                "the absence of an analysis artifact is reported",
                "the prose delta is marked as unverified",
                "no baseline defect is invented",
            ),
            hard_fail=(
                "Reporting BASELINE_NOT_MATCHED is a false alarm: this fixture's arms are matched on "
                "every declared condition."
            ),
        ),
        _task(
            suite=exp,
            name="flag a comparison that was run once with a single seed",
            prompt=(
                "An experiment reports a 0.4-point improvement from one seed, with no variance "
                "estimate. Report the statistical weakness and what would be needed to support the "
                "claim."
            ),
            required=("INSUFFICIENT_N",),
            forbidden=("MULTIPLE_COMPARISONS_UNCORRECTED",),
            criteria=(
                "the seed count is reported as insufficient for a variance estimate",
                "the number of additional seeds needed is stated",
                "no multiple-comparison finding is invented",
            ),
            hard_fail=(
                "Reporting MULTIPLE_COMPARISONS_UNCORRECTED is a false alarm: this fixture compares "
                "exactly one pair, so there is nothing to correct."
            ),
        ),
        _task(
            suite=exp,
            name="catch an uncorrected multiple-comparison sweep",
            prompt=(
                "A sweep evaluates twelve configurations against one baseline and reports the best "
                "p-value. Report the selection defect and the correction that is missing."
            ),
            required=("MULTIPLE_COMPARISONS_UNCORRECTED",),
            forbidden=("INSUFFICIENT_N",),
            criteria=(
                "the number of comparisons is reported",
                "a family-wise correction is named",
                "the best-of-twelve p-value is marked as uncorrected",
            ),
            hard_fail=(
                "Reporting INSUFFICIENT_N is a false alarm: every configuration in this fixture was "
                "run with the full seed set."
            ),
        ),
        _task(
            suite=exp,
            name="trace a seed mismatch between the recorded config and the reported runs",
            prompt=(
                "The experiment config declares seeds [0, 1, 2]; the three raw run artifacts record "
                "seeds 0, 1 and 7. Report the provenance defect and do not repair it silently."
            ),
            required=("SEED_INCONSISTENCY",),
            forbidden=("EXCLUSION_UNRECORDED",),
            criteria=(
                "the mismatching seed is named",
                "the affected run artifact is named",
                "the record is marked for human resolution instead of being rewritten",
            ),
            hard_fail=(
                "Reporting EXCLUSION_UNRECORDED is a false alarm: this fixture excludes no run."
            ),
        ),
        _task(
            suite=exp,
            name="record every excluded run instead of quietly dropping it",
            prompt=(
                "Three of nine runs diverged and were excluded from the reported mean. The exclusion "
                "exists only in a chat log. Report the unrecorded exclusion and its effect on the "
                "reported number."
            ),
            required=("EXCLUSION_UNRECORDED",),
            forbidden=("SEED_INCONSISTENCY",),
            criteria=(
                "the excluded runs are named",
                "the reported mean is marked as computed over a filtered subset",
                "no seed defect is invented",
            ),
            hard_fail=(
                "Reporting SEED_INCONSISTENCY is a false alarm: every recorded seed matches the "
                "declared config in this fixture."
            ),
        ),
        # ---------------------------------------------------------------- mechanism
        _task(
            suite=mech,
            name="refuse to strengthen a correlation into causation",
            prompt=(
                "A draft sentence reads 'X drives Y' for a study that measured a correlation across "
                "observational runs with no intervention. Report the language defect and the "
                "strongest wording the evidence supports."
            ),
            required=("CAUSAL_LANGUAGE_WITHOUT_INTERVENTION",),
            forbidden=("ALTERNATIVE_EXPLANATION_UNELIMINATED",),
            criteria=(
                "the offending sentence is quoted",
                "the calibration rung actually reached is named",
                "replacement wording stays at the observed rung",
            ),
            hard_fail=(
                "Reporting ALTERNATIVE_EXPLANATION_UNELIMINATED is a false alarm: this fixture's "
                "rival explanation was already tested and excluded."
            ),
        ),
        _task(
            suite=mech,
            name="reject a necessity claim that has no removal experiment",
            prompt=(
                "A claim states that component C is necessary for the effect, but no run removes C "
                "while holding everything else fixed. Report the missing experiment."
            ),
            required=("NECESSITY_CLAIM_WITHOUT_REMOVAL",),
            forbidden=("CAUSAL_LANGUAGE_WITHOUT_INTERVENTION",),
            criteria=(
                "the absence of a removal/ablation run is reported",
                "the experiment that would test necessity is specified",
                "the claim is held below the necessity rung",
            ),
            hard_fail=(
                "Reporting CAUSAL_LANGUAGE_WITHOUT_INTERVENTION is a false alarm: an intervention "
                "run is present and the draft language already reflects it."
            ),
        ),
        _task(
            suite=mech,
            name="stop a single-setting result from generalising",
            prompt=(
                "One mechanism result was obtained at one scale on one dataset, and the draft "
                "generalises it across settings. Report the generalisation defect."
            ),
            required=("SINGLE_SETTING_GENERALIZATION",),
            forbidden=("NECESSITY_CLAIM_WITHOUT_REMOVAL",),
            criteria=(
                "the single setting is named",
                "the generalising sentence is quoted",
                "the replication that would license generalisation is described",
            ),
            hard_fail=(
                "Reporting NECESSITY_CLAIM_WITHOUT_REMOVAL is a false alarm: this fixture has a "
                "removal run and claims no necessity."
            ),
        ),
        _task(
            suite=mech,
            name="require an alternative explanation to be eliminated before the mechanism claim",
            prompt=(
                "A candidate mechanism is supported, but a competing explanation (distribution shift "
                "in the evaluation split) was never tested. Report the uneliminated alternative and "
                "the discriminating experiment."
            ),
            required=("ALTERNATIVE_EXPLANATION_UNELIMINATED",),
            forbidden=("SINGLE_SETTING_GENERALIZATION",),
            criteria=(
                "the untested alternative is stated explicitly",
                "a discriminating experiment is proposed",
                "the mechanism claim is held at its current rung",
            ),
            hard_fail=(
                "Reporting SINGLE_SETTING_GENERALIZATION is a false alarm: this fixture replicates "
                "across two settings and generalises nowhere."
            ),
        ),
        _task(
            suite=mech,
            name="hold language to the intervention rung the experiment actually reached",
            prompt=(
                "An experiment intervened on one component, and the draft claims both causation and "
                "necessity. Report every claim that outruns the design."
            ),
            required=("CAUSAL_LANGUAGE_WITHOUT_INTERVENTION", "NECESSITY_CLAIM_WITHOUT_REMOVAL"),
            forbidden=("ALTERNATIVE_EXPLANATION_UNELIMINATED",),
            criteria=(
                "each overreaching sentence is quoted separately",
                "the intervention rung and the necessity rung are distinguished",
                "an alternative explanation already excluded is not reported as open",
            ),
            hard_fail=(
                "Reporting ALTERNATIVE_EXPLANATION_UNELIMINATED is a false alarm: the fixture's "
                "alternative was tested and excluded."
            ),
        ),
        # ---------------------------------------------------------------- writing
        _task(
            suite=wr,
            name="remove AI tone without adding a number",
            prompt=(
                "Rewrite a paragraph that contains 'It is important to note that', 'delve' and a "
                "three-item parallel close. No new quantitative content may appear."
            ),
            required=("TEMPLATE_PHRASE",),
            forbidden=("UNGROUNDED_NUMBER",),
            criteria=(
                "every template phrase in the input is reported",
                "the rewrite contains no number absent from the input",
                "the rewrite preserves every original claim",
            ),
            hard_fail=(
                "Producing UNGROUNDED_NUMBER is fatal here: adding a number that the evidence does "
                "not contain is the exact failure this task exists to catch."
            ),
        ),
        _task(
            suite=wr,
            name="never introduce a number that is not in the evidence",
            prompt=(
                "A draft sentence states 'accuracy improved by 12 percent' while the evidence "
                "artifact records 0.031 absolute. Report the ungrounded number."
            ),
            required=("UNGROUNDED_NUMBER",),
            forbidden=("TEMPLATE_PHRASE",),
            criteria=(
                "the unsupported number is quoted with its sentence",
                "the closest grounded value is named",
                "no style finding is invented",
            ),
            hard_fail=(
                "Reporting TEMPLATE_PHRASE is a false alarm: this fixture contains no template "
                "scaffolding."
            ),
        ),
        _task(
            suite=wr,
            name="lower claim language to the evidence level",
            prompt=(
                "A draft says 'this proves that the method generalises', while the evidence reaches "
                "one controlled comparison at one setting. Report the language defect and the "
                "evidence level that licenses the wording."
            ),
            required=("LANGUAGE_EXCEEDS_EVIDENCE",),
            forbidden=("NOVELTY_UNSUPPORTED",),
            criteria=(
                "the over-strong sentence is quoted",
                "the evidence level actually reached is named",
                "replacement wording matches that level",
            ),
            hard_fail=(
                "Reporting NOVELTY_UNSUPPORTED is a false alarm: this fixture already carries a "
                "supported novelty statement with its prior-art references."
            ),
        ),
        _task(
            suite=wr,
            name="stop an unsupported novelty sentence from reaching the draft",
            prompt=(
                "A draft claims 'this is the first work to do X', while the prior-art matrix is "
                "incomplete for that claim. Report the unsupported novelty claim."
            ),
            required=("NOVELTY_UNSUPPORTED",),
            forbidden=("LANGUAGE_EXCEEDS_EVIDENCE",),
            criteria=(
                "the novelty sentence is quoted",
                "the incomplete prior-art evidence is named",
                "the sentence is removed or weakened rather than published",
            ),
            hard_fail=(
                "Reporting LANGUAGE_EXCEEDS_EVIDENCE is a false alarm: the remaining sentences in "
                "this fixture stay within their evidence level."
            ),
        ),
    ]


def seed_default_tasks(
    kernel: ResearchKernel, *, principal: Principal | None = None
) -> list[BenchmarkTask]:
    """Persist the default catalogue into ``kernel.benchmark_tasks``; idempotent.

    ``principal=None`` means "install the fixture set at kernel bootstrap"; it resolves to
    ``kernel.human()`` rather than inventing an ambient actor, and the ``skill.evaluate``
    capability is required either way.
    """
    actor = principal if principal is not None else kernel.human()
    actor.require(Cap.SKILL_EVALUATE, "benchmark.catalogue.seed")
    tasks = default_catalogue()
    created = 0
    for task in tasks:
        if kernel.benchmark_tasks.get(task.benchmark_task_id) is None:
            kernel.benchmark_tasks.save(task)
            created += 1
    if created:
        kernel.events.append(
            "benchmark.catalogue.seeded",
            actor=actor.name,
            payload={
                "tasks": len(tasks),
                "created": created,
                "suites": sorted(FINDING_CODES_BY_SUITE),
            },
        )
    return tasks


__all__ = [
    "ALL_FINDING_CODES",
    "FINDING_CODES_BY_SUITE",
    "default_catalogue",
    "deterministic_task_id",
    "seed_default_tasks",
]
