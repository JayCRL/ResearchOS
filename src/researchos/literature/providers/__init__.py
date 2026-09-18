"""Literature provider adapters.

Package layout:

    base.py         — the provider contract, the injected fetcher, text normalisation helpers
    arxiv.py        — arXiv Atom export API
    semantic_scholar.py — Semantic Scholar Graph API (citations + references)
    crossref.py     — Crossref DOI registry (publication of record, reference DOIs)
    openreview.py   — OpenReview notes (submissions, reviews, unindexed mechanism work)
    github.py       — repository search (implementations are prior art too)
    web.py          — generic search over an injected endpoint (unavailable unless configured)
    local_bib.py    — the researcher's own BibTeX files, parsed by hand
    cache.py        — read-through disk cache with replayable raw payloads
    registry.py     — construction of the provider list, with no provider hard-wired

Import order matters for exactly one reason: ``providers`` must be importable before the rest of
``researchos.literature`` because the graph and the planner reuse the lexical-similarity helpers
defined in :mod:`researchos.literature.providers.base`.
"""

from .arxiv import ArxivProvider
from .base import (
    Fetcher,
    LiteratureProvider,
    OfflineProvider,
    ProviderError,
    build_record,
    default_fetcher,
    jaccard,
    lexical_similarity,
    normalise_title,
    tokenise,
)
from .cache import CachedProvider, cache_providers, read_cached_records
from .crossref import CrossrefProvider
from .github import GitHubProvider
from .local_bib import BibEntry, BibParseError, LocalBibProvider, load_bib_file, parse_bibtex
from .openreview import OpenReviewProvider
from .registry import default_providers, describe_providers, offline_providers, provider_kinds
from .semantic_scholar import SemanticScholarProvider
from .web import WebProvider, parse_web_payload

__all__ = [
    "ArxivProvider",
    "BibEntry",
    "BibParseError",
    "CachedProvider",
    "CrossrefProvider",
    "Fetcher",
    "GitHubProvider",
    "LiteratureProvider",
    "LocalBibProvider",
    "OfflineProvider",
    "OpenReviewProvider",
    "ProviderError",
    "SemanticScholarProvider",
    "WebProvider",
    "build_record",
    "cache_providers",
    "default_fetcher",
    "default_providers",
    "describe_providers",
    "jaccard",
    "lexical_similarity",
    "load_bib_file",
    "normalise_title",
    "offline_providers",
    "parse_bibtex",
    "parse_web_payload",
    "provider_kinds",
    "read_cached_records",
    "tokenise",
]
