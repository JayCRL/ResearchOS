"""Literature coverage accounting.

Coverage is the number that licenses (or forbids) a novelty claim, so it is *measured* from what the
search actually did — queries executed per family, providers exercised, papers screened — and never
estimated from how confident the summary sounds.

This module is import-light on purpose: :mod:`researchos.literature.query_planner` and
:mod:`researchos.literature.novelty` both depend on it, and a coverage calculation must never be the
reason a search cannot run.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..models.literature import (
    COVERAGE_THRESHOLDS,
    REQUIRED_FAMILIES,
    LiteratureCoverage,
    LiteraturePaper,
    QueryFamily,
    QueryPlan,
    ProviderKind,
)


def compute_coverage(plan: QueryPlan, papers: Sequence[LiteraturePaper]) -> LiteratureCoverage:
    """Turn a query plan plus its screened papers into a measured coverage record."""
    executed = plan.executed_queries()
    by_family: dict[str, int] = {}
    for query in executed:
        by_family[query.family.value] = by_family.get(query.family.value, 0) + 1

    families = sorted({q.family for q in executed}, key=lambda f: f.value)
    providers = sorted(
        {p for q in executed for p in q.providers} or set(plan.providers), key=lambda p: p.value
    )
    fulltext = sum(1 for p in papers if p.fulltext_status.value == "FULLTEXT_VERIFIED")
    screened = sum(1 for p in papers if p.screened) or len(papers)

    coverage = LiteratureCoverage(
        queries_executed=len(executed),
        queries_by_family=by_family,
        families_covered=families,
        missing_required_families=plan.missing_required_families(),
        providers_used=providers,
        papers_screened=screened,
        papers_fulltext_checked=fulltext,
        seed_papers=len(plan.seed_paper_ids),
        expansion_rounds=plan.expansion_rounds,
        search_log_refs=[plan.query_plan_id],
    )
    coverage.compute_score()
    return coverage


def coverage_report(coverage: LiteratureCoverage) -> list[str]:
    """Human-readable coverage lines, deficits first. Never hides an unmet threshold."""
    lines: list[str] = []
    for deficit in coverage.deficits():
        lines.append(f"DEFICIT: {deficit}")
    lines.append(
        f"executed {coverage.queries_executed} queries across "
        f"{len(coverage.families_covered)} families via "
        f"{len(coverage.providers_used)} provider(s)"
    )
    if coverage.queries_by_family:
        lines.append(
            "per family: "
            + ", ".join(f"{family}={count}" for family, count in sorted(coverage.queries_by_family.items()))
        )
    lines.append(
        f"screened {coverage.papers_screened} paper(s); {coverage.papers_fulltext_checked} full text(s) verified"
    )
    lines.append(f"coverage score {coverage.score:.2f} — {coverage.basis}")
    return lines


def thresholds() -> dict[str, int]:
    return dict(COVERAGE_THRESHOLDS)


def required_families() -> tuple[QueryFamily, ...]:
    return REQUIRED_FAMILIES


def provider_names(providers: Iterable[object]) -> list[ProviderKind]:
    """Extract ``ProviderKind``s from provider objects, ignoring anything that does not declare one."""
    out: list[ProviderKind] = []
    for provider in providers:
        kind = getattr(provider, "kind", None)
        if isinstance(kind, ProviderKind):
            out.append(kind)
        elif isinstance(kind, str):
            try:
                out.append(ProviderKind(kind))
            except ValueError:
                continue
    return sorted(set(out), key=lambda k: k.value)


def is_sufficient(coverage: LiteratureCoverage) -> bool:
    return coverage.meets_threshold()
