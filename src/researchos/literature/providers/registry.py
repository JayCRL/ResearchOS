"""Provider registry (no provider is hard-wired into the planner).

Why a registry and not a fixed list: coverage is only meaningful if the *set of sources actually
searched* is explicit and recorded. The planner therefore receives provider instances from here and
writes their kinds into the :class:`~researchos.models.literature.QueryPlan`; nothing downstream may
assume a source exists that was not in that list.

Two construction paths:

* :func:`default_providers` — the real world. Network providers are wrapped in
  :class:`~researchos.literature.providers.cache.CachedProvider` under
  ``.researchos/cache/literature/`` so a run is replayable and rate limits are respected.
* :func:`offline_providers` — a closed fixture corpus, for tests, demos and for re-running a plan
  when the network is unavailable. Nothing here ever performs I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from ...kernel.kernel import ResearchKernel
from ...models.literature import ProviderKind, ProviderRecord
from .arxiv import ArxivProvider
from .base import Fetcher, LiteratureProvider, OfflineProvider
from .cache import CachedProvider, cache_providers
from .crossref import CrossrefProvider
from .github import GitHubProvider
from .local_bib import LocalBibProvider
from .openreview import OpenReviewProvider
from .semantic_scholar import SemanticScholarProvider
from .web import WebProvider

#: Kinds reachable by construction. Used by the planner to sanity-check a requested provider list.
NETWORK_PROVIDER_KINDS: tuple[ProviderKind, ...] = (
    ProviderKind.ARXIV,
    ProviderKind.SEMANTIC_SCHOLAR,
    ProviderKind.CROSSREF,
    ProviderKind.OPENREVIEW,
    ProviderKind.GITHUB,
)


def default_providers(
    kernel: ResearchKernel,
    *,
    offline: bool = False,
    bib_paths: Sequence[Path] = (),
    fetcher: Fetcher | None = None,
    web_search_callable: Callable[[str], bytes] | None = None,
    web_search_url: str | None = None,
    api_keys: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> list[LiteratureProvider]:
    """Build the provider list for a research project.

    ``offline=True`` returns only providers that cannot touch the network (a local bibliography).
    It deliberately returns *fewer* providers rather than mock ones: a run that searched one local
    file must show up as thin coverage, not as a full search.

    ``fetcher`` is forwarded to every network provider, which is how recorded-response tests drive
    the real parsers. ``web_search_callable``/``web_search_url`` configure
    :class:`WebProvider`; without them that provider would be unavailable and is left out entirely
    instead of being added as a source that can never answer.
    """
    keys = api_keys or {}
    paths = [Path(p) for p in bib_paths]
    providers: list[LiteratureProvider] = []

    if paths:
        providers.append(LocalBibProvider(paths))
    if not offline:
        providers.append(ArxivProvider(fetcher))
        providers.append(SemanticScholarProvider(fetcher, api_key=keys.get("semantic_scholar")))
        providers.append(CrossrefProvider(fetcher))
        providers.append(OpenReviewProvider(fetcher))
        providers.append(GitHubProvider(fetcher, token=keys.get("github")))
        if web_search_callable is not None:
            providers.append(WebProvider(search_callable=web_search_callable, base_url=web_search_url))

    if offline or not use_cache:
        return providers
    target_dir = cache_dir or (kernel.paths.ros / "cache" / "literature")
    return cache_providers(providers, target_dir)


def offline_providers(
    records: Sequence[ProviderRecord],
    *,
    kind: ProviderKind = ProviderKind.LOCAL_BIB,
    require_overlap: bool = False,
) -> list[LiteratureProvider]:
    """A single fixture provider over ``records`` — the deterministic provider used by tests."""
    return [OfflineProvider(records, kind=kind, require_overlap=require_overlap)]


def provider_kinds(providers: Sequence[LiteratureProvider]) -> list[ProviderKind]:
    """Distinct kinds in a provider list, sorted — what a plan would record as its sources."""
    return sorted({p.kind for p in providers}, key=lambda k: k.value)


def describe_providers(providers: Sequence[LiteratureProvider]) -> list[dict[str, object]]:
    """One row per provider for CLI/report output, including cache statistics when present."""
    rows: list[dict[str, object]] = []
    for provider in providers:
        row = provider.describe()
        if isinstance(provider, CachedProvider):
            row["cache"] = provider.stats()
        rows.append(row)
    return rows


__all__ = [
    "NETWORK_PROVIDER_KINDS",
    "default_providers",
    "describe_providers",
    "offline_providers",
    "provider_kinds",
]
