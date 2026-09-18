"""Semantic Scholar provider: JSON envelope in, :class:`ProviderRecord` out.

Why include it at all when arXiv exists: Semantic Scholar is the only free source in this package
that supplies *both* directions of the citation graph plus resolved external ids, which is what
reference/citation expansion and cross-provider de-duplication need. It is also the source most
likely to be rate-limited, which is why an API key is a constructor argument rather than something
read from the environment behind the caller's back — an unkeyed run is slower and the caller
should be able to see that in the plan.

Auth handling is explicit: the key travels as the ``x-api-key`` header on the production
``urllib`` path. When a custom ``fetcher`` is injected, the fetcher owns the request, so it must
carry credentials itself (a test stub does not need them).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Mapping
from urllib.parse import quote

from ...models.literature import ProviderKind, ProviderRecord
from .base import Fetcher, LiteratureProvider, ProviderError, build_record, clean_text, decode_text, default_fetcher

DEFAULT_S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"

#: Fields requested on every call. Asking for fewer fields would save bandwidth and cost accuracy;
#: ``externalIds`` is what makes de-duplication against Crossref and arXiv possible at all.
SEARCH_FIELDS = (
    "paperId,title,abstract,year,venue,publicationVenue,authors,externalIds,"
    "citationCount,influentialCitationCount,openAccessPdf,url,fieldsOfStudy,referenceCount"
)
GRAPH_FIELDS = "paperId,title,abstract,year,venue,authors,externalIds,citationCount,openAccessPdf,url"


class SemanticScholarProvider(LiteratureProvider):
    """Search, fetch and citation-expand through the Semantic Scholar Graph API."""

    kind: ClassVar[ProviderKind] = ProviderKind.SEMANTIC_SCHOLAR

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_S2_BASE_URL,
        timeout: float | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    @property
    def name(self) -> str:
        suffix = "" if self._api_key else " (unauthenticated: shared rate limit)"
        return f"Semantic Scholar (Graph API){suffix}"

    @property
    def authenticated(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------ request plumbing

    def _get(self, url: str) -> bytes:
        if self._fetcher is not None:
            return self._fetcher(url)
        headers = {"x-api-key": self._api_key} if self._api_key else None
        return default_fetcher(url, headers=headers, timeout=self._timeout or 30.0)

    def _json(self, url: str) -> Any:
        payload = self._get(url)
        try:
            data = json.loads(decode_text(payload))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Semantic Scholar returned non-JSON for {url}: {exc}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise ProviderError(f"Semantic Scholar error for {url}: {data.get('error')}")
        return data

    def search_url(self, query: str, *, limit: int = 20) -> str:
        return (
            f"{self._base_url}/paper/search?query={quote(query)}"
            f"&limit={max(1, limit)}&fields={SEARCH_FIELDS}"
        )

    def fetch_url(self, identifier: str) -> str:
        return f"{self._base_url}/paper/{quote(identifier.strip())}?fields={GRAPH_FIELDS}"

    # ------------------------------------------------------------------ API

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        url = self.search_url(query, limit=limit)
        data = self._json(url)
        items = data.get("data") if isinstance(data, dict) else None
        if items is None:
            raise ProviderError(f"Semantic Scholar search response has no 'data' list: keys={sorted(data) if isinstance(data, dict) else type(data).__name__}")
        return [self._record(item, url=url, query=query) for item in items if isinstance(item, dict)][: max(0, limit)]

    def fetch(self, identifier: str) -> ProviderRecord | None:
        url = self.fetch_url(identifier)
        data = self._json(url)
        if not isinstance(data, dict) or not data.get("paperId"):
            return None
        return self._record(data, url=url, query=None)

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """References and citations, from the record's own payload when present, else the graph API.

        Reading ``raw["references"]`` first means an expansion can be replayed from a cached payload
        with no network at all — the property that makes citation expansion auditable.
        """
        cached_references = record.raw.get("references")
        cached_citations = record.raw.get("citations")
        if isinstance(cached_references, list) or isinstance(cached_citations, list):
            return (
                [self._record(item, url="", query=None) for item in (cached_references or []) if isinstance(item, dict)],
                [self._record(item, url="", query=None) for item in (cached_citations or []) if isinstance(item, dict)],
            )

        seed = record.raw.get("paperId") or record.provider_id
        references = self._graph(seed, "references", record)
        citations = self._graph(seed, "citations", record)
        return references, citations

    def _graph(self, seed: str, direction: str, record: ProviderRecord) -> list[ProviderRecord]:
        url = f"{self._base_url}/paper/{quote(str(seed))}/{direction}?limit=100&fields={GRAPH_FIELDS}"
        data = self._json(url)
        items = data.get("data") if isinstance(data, dict) else None
        out: list[ProviderRecord] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            inner = item.get("citedPaper") or item.get("citingPaper") or item
            if isinstance(inner, dict) and inner.get("paperId"):
                out.append(self._record(inner, url=url, query=None))
        return out

    # ------------------------------------------------------------------ parsing

    def _record(self, item: Mapping[str, Any], *, url: str, query: str | None) -> ProviderRecord:
        external = item.get("externalIds") or {}
        if not isinstance(external, Mapping):
            external = {}
        pdf = item.get("openAccessPdf") or {}
        pdf_url = pdf.get("url") if isinstance(pdf, Mapping) else None
        venue = item.get("venue")
        if not venue and isinstance(item.get("publicationVenue"), Mapping):
            venue = item["publicationVenue"].get("name")
        authors = [
            author.get("name") if isinstance(author, Mapping) else author
            for author in (item.get("authors") or [])
        ]
        return build_record(
            ProviderKind.SEMANTIC_SCHOLAR,
            str(item.get("paperId") or ""),
            clean_text(item.get("title")),
            abstract=item.get("abstract"),
            authors=authors,
            year=item.get("year"),
            venue=venue,
            doi=external.get("DOI"),
            arxiv_id=external.get("ArXiv"),
            url=item.get("url") or (f"https://www.semanticscholar.org/paper/{item.get('paperId')}" if item.get("paperId") else None),
            citation_count=item.get("citationCount"),
            open_access_pdf=pdf_url,
            references=[i for i in (item.get("referenceIds") or []) if isinstance(i, str)],
            query=query,
            raw={
                "source": "semantic_scholar_graph",
                "request_url": url,
                "payload": dict(item),
                "external_ids": dict(external),
                "fields_of_study": list(item.get("fieldsOfStudy") or []),
            },
        )
