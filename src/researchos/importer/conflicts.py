"""Conflict detection during import.

The rule that matters: when two sources disagree, ResearchOS does **not** pick one silently. It
creates a :class:`Conflict`, keeps both sources with their digests, states the difference, and
applies the deterministic provenance trust order only as a *default*.

Concretely, the most valuable conflict this module finds is "the number in the prose does not match
the number in the data" — precisely the class of error that makes AI-written papers unscientific.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..models.common import ConflictKind, SourceKind
from .facts import Fact, FactKind

#: How close two numbers must be to count as "the same number".
DEFAULT_ABS_TOLERANCE = 0.005
DEFAULT_REL_TOLERANCE = 0.02

#: Facts that carry a *measured* number, strongest first.
MEASURED_RULES: tuple[str, ...] = ("table:metric", "json:scalar", "log:kv")

#: Sources whose numbers are narrative rather than measured.
NARRATIVE_SOURCE_KINDS: frozenset[SourceKind] = frozenset(
    {SourceKind.PAPER_PROSE, SourceKind.AUDIT, SourceKind.AUTHOR_NOTE, SourceKind.AI_SUMMARY}
)

#: Statistical quantities that describe a measurement rather than being one. A prose "std = 0.004"
#: cannot contradict a measured "retention = 0.42"; treating it as if it could manufactures noise.
DISPERSION_TOKENS: tuple[str, ...] = (
    "std", "stdev", "sd", "se", "sem", "ci", "ci_low", "ci_high", "variance", "var",
    "median", "iqr", "min", "max", "range", "quartile", "percentile", "df", "n",
)

#: Reporting modifiers: they say how a value is summarised, not which quantity it is.
STAT_MODIFIERS: frozenset[str] = frozenset(
    {"mean", "avg", "average", "value", "level", "score", "est", "estimate", "overall", "total"}
)


@dataclass
class NumberConflict:
    """A disagreement about one number, with both sides preserved."""

    metric: str
    labels: dict[str, str]
    measured: Fact
    narrative: Fact
    difference: str
    delta: float
    is_relative_mismatch: bool
    severity: str = "HIGH"


@dataclass
class DuplicateRun:
    metric: str
    labels: dict[str, str]
    rows: list[int]
    values: list[float]
    identical: bool


@dataclass
class ConflictScan:
    number_conflicts: list[NumberConflict] = field(default_factory=list)
    duplicate_runs: list[DuplicateRun] = field(default_factory=list)
    missing_values: list[Fact] = field(default_factory=list)
    failed_runs: list[Fact] = field(default_factory=list)
    metric_disagreements: list[str] = field(default_factory=list)


def _close(a: float, b: float, *, abs_tol: float, rel_tol: float) -> bool:
    if abs(a - b) <= abs_tol:
        return True
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale <= rel_tol


def _labels_match(a: dict, b: dict) -> bool:
    """Two numbers are comparable only when their labels agree on the shared keys."""
    shared = set(a) & set(b)
    if not shared:
        return True
    return all(str(a[key]).strip().lower() == str(b[key]).strip().lower() for key in shared)


def _normalise_metric(metric: str) -> str:
    return metric.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_metric(metric: str) -> str:
    """Canonical metric name: ``retention@1 mean`` and ``retention_at_1`` mean the same thing."""
    base = _normalise_metric(metric).replace("@", "_at_")
    while "__" in base:
        base = base.replace("__", "_")
    return base.strip("_")


def base_metric(metric: str) -> str:
    """Metric with reporting modifiers removed: ``retention_at_1_mean`` -> ``retention_at_1``.

    Without this, a prose "retention@1 mean" has no exact counterpart in the data and would be
    compared against whatever retention-family number happened to be nearest — including the
    ``retention@4`` column.
    """
    return "_".join(
        token for token in canonical_metric(metric).split("_") if token and token not in STAT_MODIFIERS
    )


def _metric_family(metric: str) -> str:
    """Group ``retention``, ``retention_at_1``, ``retention@1`` into one family for comparison."""
    base = base_metric(metric)
    for marker in ("_at_", "_"):
        if marker in base:
            head = base.split(marker)[0]
            if head:
                return head
    return base


def is_dispersion_metric(metric: str | None) -> bool:
    """True when the name describes a spread, not a value. Such numbers are never compared."""
    if not metric:
        return False
    tokens = set(re.split(r"[^a-z0-9]+", _normalise_metric(metric)))
    if tokens & set(DISPERSION_TOKENS):
        return True
    return any(token in _normalise_metric(metric) for token in ("_std", "_se", "_ci", "_sem", "_min", "_max"))


def detect_conflicts(facts: Sequence[Fact]) -> ConflictScan:
    """Compare every narrative number against the measured numbers for the same metric."""
    scan = ConflictScan()
    measured: list[Fact] = [f for f in facts if f.rule in MEASURED_RULES and f.numeric() is not None]
    narrative: list[Fact] = [
        f
        for f in facts
        if f.numeric() is not None
        and f.rule not in MEASURED_RULES
        and (f.kind in (FactKind.PROSE_NUMBER, FactKind.METRIC_VALUE))
    ]

    for fact in narrative:
        metric = fact.metric()
        if not metric:
            continue
        if is_dispersion_metric(metric):
            scan.metric_disagreements.append(
                f"{fact.source.path} reports a spread statistic ({metric!r}); it is not compared "
                "against measured levels"
            )
            continue
        canonical = base_metric(metric)
        family = _metric_family(metric)
        value = fact.numeric()
        if value is None:
            continue
        labels = {k: str(v) for k, v in (fact.payload.get("labels") or {}).items()}

        def tier(candidate: Fact) -> int:
            """Rank a candidate comparison source. Deterministic and explainable.

            4 = same metric, from a labelled data artifact (the ideal comparison)
            3 = same metric, from an unlabelled source (e.g. a log line)
            2 = same metric family, labelled data
            1 = same metric family, unlabelled
            0 = not comparable

            Labelled data outranks unlabelled logs deliberately: a checkpoint line in a training log
            is not a result, and letting it satisfy a prose number would hide real contradictions.
            """
            candidate_labels = {k: str(v) for k, v in (candidate.payload.get("labels") or {}).items()}
            if labels and candidate_labels and not _labels_match(labels, candidate_labels):
                return 0
            same_metric = base_metric(candidate.metric() or "") == canonical
            same_family = _metric_family(candidate.metric() or "") == family
            cand_labelled = bool(candidate_labels)
            if same_metric and cand_labelled:
                return 4
            if same_metric:
                return 3
            if same_family and cand_labelled:
                return 2
            if same_family:
                return 1
            return 0

        ranked = [(tier(m), m) for m in measured]
        best_tier = max((rank for rank, _ in ranked), default=0)
        candidates = [m for rank, m in ranked if rank == best_tier and rank > 0]
        if not candidates:
            scan.metric_disagreements.append(
                f"no measured value found for metric {metric!r} mentioned in {fact.source.path}"
            )
            continue
        best = min(candidates, key=lambda m: abs((m.numeric() or 0.0) - value))
        best_value = best.numeric() or 0.0

        # Unit-scale guard: a prose statement of "14.7 points" is a *difference* on a metric whose
        # data lives in [0, 1]. Comparing them numerically would manufacture a fake contradiction.
        family_values = [m.numeric() for m in measured if _metric_family(m.metric() or "") == family]
        family_max = max((abs(v) for v in family_values if v is not None), default=0.0)
        if family_max <= 1.0 and abs(value) > 1.5 and not fact.payload.get("table"):
            scan.metric_disagreements.append(
                f"{fact.source.path} expresses {metric!r} as {value:g} while measured values for this "
                f"metric are fractions (max {family_max:g}); treated as a difference/percentage "
                "statement rather than a contradiction"
            )
            continue

        if _close(value, best_value, abs_tol=DEFAULT_ABS_TOLERANCE, rel_tol=DEFAULT_REL_TOLERANCE):
            continue
        delta = value - best_value
        rel = abs(delta) / max(abs(best_value), 1e-12)
        scan.number_conflicts.append(
            NumberConflict(
                metric=metric,
                labels=labels,
                measured=best,
                narrative=fact,
                difference=(
                    f"{fact.source.path} states {metric} = {value:g}"
                    + (f" ({', '.join(f'{k}={v}' for k, v in sorted(labels.items()))})" if labels else "")
                    + f", but {best.source.path} records {best_value:g}"
                    f" (difference {delta:+.4g}, {rel:.1%} relative)"
                ),
                delta=delta,
                is_relative_mismatch=rel > 0.10,
            )
        )

    scan.duplicate_runs = _find_duplicate_runs(facts)
    scan.missing_values = [
        f for f in facts if f.kind is FactKind.METRIC_VALUE and f.payload.get("missing")
    ]
    scan.failed_runs = [f for f in facts if f.kind is FactKind.FAILED_RUN]
    return scan


def _find_duplicate_runs(facts: Sequence[Fact]) -> list[DuplicateRun]:
    """Detect the same (metric, seed, arm) reported twice — a classic silent double-count."""
    grouped: dict[tuple[str, str], list[Fact]] = {}
    for fact in facts:
        if fact.rule != "table:metric" or fact.numeric() is None:
            continue
        labels = fact.payload.get("labels") or {}
        seed = str(labels.get("seed", labels.get("run", "")))
        arm = str(labels.get("arm", labels.get("variant", labels.get("condition", ""))))
        if not seed and not arm:
            continue
        grouped.setdefault((f"{fact.metric()}|{arm}|{seed}", fact.source.path), []).append(fact)

    duplicates: list[DuplicateRun] = []
    for key, group in grouped.items():
        if len(group) < 2:
            continue
        values = [f.numeric() or 0.0 for f in group]
        metric_and_labels = key[0].split("|")
        duplicates.append(
            DuplicateRun(
                metric=metric_and_labels[0],
                labels={"arm": metric_and_labels[1], "seed": metric_and_labels[2]},
                rows=[int(f.payload.get("row", 0)) for f in group],
                values=values,
                identical=all(abs(v - values[0]) < 1e-12 for v in values),
            )
        )
    return duplicates


def conflict_severity(conflict: NumberConflict) -> str:
    """Measured-vs-prose mismatches are the high-severity class; audit-vs-raw is medium."""
    if conflict.narrative.source.kind is SourceKind.PAPER_PROSE:
        return "HIGH"
    if conflict.narrative.source.kind is SourceKind.AUDIT:
        return "MEDIUM"
    return "LOW"


def summarise(scan: ConflictScan) -> list[str]:
    lines: list[str] = []
    for conflict in scan.number_conflicts:
        lines.append(f"[{conflict_severity(conflict)}] {conflict.difference}")
    for duplicate in scan.duplicate_runs:
        lines.append(
            f"[MEDIUM] duplicate run: {duplicate.metric} {duplicate.labels} appears at rows "
            f"{duplicate.rows} (values {duplicate.values})"
        )
    for fact in scan.missing_values:
        lines.append(
            f"[MEDIUM] missing value: {fact.metric()} for {fact.payload.get('labels')} in {fact.source.path}"
        )
    for fact in scan.failed_runs:
        lines.append(f"[INFO] failed run recorded in {fact.source.path}: {fact.statement[:120]}")
    for message in scan.metric_disagreements:
        lines.append(f"[INFO] {message}")
    return lines


def unresolved_metrics(scan: ConflictScan) -> list[str]:
    return sorted({c.metric for c in scan.number_conflicts})
