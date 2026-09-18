"""Generic web search provider — deliberately the *hardest* one to configure.

Why it is written this way: a "web search" adapter is the easiest place in a literature system to
start inventing results (fake hit lists, hallucinated snippets, an LLM summarising "the web"). So
this provider has no default endpoint at all. With nothing configured it reports
``available() == False`` and :meth:`WebProvider.search` raises :class:`ProviderError` — an
unavailable source is recorded as a coverage gap, whereas a fabricated source is a silent
corruption of the novelty gate.

The endpoint is injected as a callable ``Callable[[str], bytes]`` that receives a request target and
returns the response body:

* ``base_url=None`` (default): the callable receives the *query text itself*, which suits a caller
  with its own search backend (a local index, a vendor SDK wrapper, a recorded fixture).
* ``base_url="https://host/search?q={query}&n={limit}"``: the template is filled and the callable
  receives a real URL, so a plain HTTP fetcher can be used.

Response envelopes vary wildly, so parsing is tolerant (a list, or one of the usual list keys) and
``parse`` may be supplied for a backend whose shape is genuinely different. Nothing is inferred: a
hit without a title and without a URL is dropped rather than given a generated name.
"""

from __future__ import annotations

import json
from typing import Any, Callable, ClassVar, Mapping, Sequence
from urllib.parse import quote

from ...models.common import sha256_text
from ...models.literature import ProviderKind, ProviderRecord
from .base import Fetcher, LiteratureProvider, ProviderError, build_record, clean_text, decode_text

#: Response-envelope keys tried, in order, when a backend returns an object instead of a list.
HIT_KEYS: tuple[str, ...] = (
    "results", "items", "organic_results", "data", "hits", "webPages", "RelatedTopics", "entries",
)
#: Field aliases seen across search backends.
TITLE_KEYS: tuple[str, ...] = ("title", "name", "Text", "heading")
URL_KEYS: tuple[str, ...] = ("url", "link", "href", "FirstURL", "uri")
SNIPPET_KEYS: tuple[str, ...] = ("snippet", "description", "abstract", "Text", "summary", "content")
DATE_KEYS: tuple[str, ...] = ("published", "publishedDate", "date", "first_seen", "age")

def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return the first present, non-empty value among ``keys`` (field aliases across backends)."""
    for key in keys:
        if key in mapping and mapping[key] not in (None, "", [], {}):
            return mapping[key]
    return None


class WebProvider(LiteratureProvider):
    """Generic search over an injected endpoint. Unavailable unless one is configured."""

    kind: ClassVar[ProviderKind] = ProviderKind.WEB

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        search_callable: Callable[[str], bytes] | None = None,
        base_url: str | None = None,
        parse: Callable[[bytes], list[dict[str, Any]]] | None = None,
        timeout: float | None = None,
        label: str = "generic web search",
    ) -> None:
        self._search_callable = search_callable or fetcher
        self._base_url = base_url
        self._parse = parse
        self._label = label
        self._timeout = timeout

    @property
    def name(self) -> str:
        if not self.available():
            return f"{self._label} (unavailable: no endpoint configured)"
        return f"{self._label} ({self._base_url or 'callable'})"

    def available(self) -> bool:
        """True only when a search callable is configured. A configuration check, not a probe."""
        return self._search_callable is not None

    def request_target(self, query: str, *, limit: int = 20) -> str:
        """What the injected callable receives — a URL when a template is configured, else the query."""
        if not self._base_url:
            return query
        if "{query}" in self._base_url or "{limit}" in self._base_url:
            return self._base_url.format(query=quote(query), limit=max(1, limit))
        separator = "&" if "?" in self._base_url else "?"
        return f"{self._base_url}{separator}q={quote(query)}&limit={max(1, limit)}"

    def search_url(self, query: str, *, limit: int = 20) -> str:
        return self.request_target(query, limit=limit)

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        if self._search_callable is None:
            raise ProviderError(
                f"{self._label}: no search endpoint configured. Configure one explicitly "
                "(search_callable/base_url) — this provider will not invent results."
            )
        target = self.request_target(query, limit=limit)
        payload = self._search_callable(target)
        hits = self._parse(payload) if self._parse is not None else parse_web_payload(payload)
        records: list[ProviderRecord] = []
        for hit in hits[: max(0, limit)]:
            record = self._record(hit, target=target, query=query)
            if record is not None:
                records.append(record)
        return records

    def _record(self, hit: Mapping[str, Any], *, target: str, query: str) -> ProviderRecord | None:
        title = clean_text(_first(hit, TITLE_KEYS))
        url = clean_text(_first(hit, URL_KEYS))
        snippet = clean_text(_first(hit, SNIPPET_KEYS))
        if not title and not url:
            return None
        if not title and snippet:  # DuckDuckGo-style "Title - description" blobs
            title, _, rest = snippet.partition(" - ")
            snippet = rest or snippet
        identifier = url or sha256_text(f"{title}|{snippet}")[:32]
        return build_record(
            ProviderKind.WEB,
            identifier,
            title or url,
            abstract=snippet,
            year=_first(hit, DATE_KEYS),
            venue=clean_text(hit.get("source") or hit.get("engine")) or None,
            url=url,
            query=query,
            raw={
                "source": "generic_web",
                "request_target": target,
                "payload": dict(hit),
            },
        )


def parse_web_payload(payload: bytes) -> list[dict[str, Any]]:
    """Best-effort extraction of hits from an unknown search envelope.

    Raising here is intentional: a backend whose shape nobody anticipated must be configured via
    ``parse`` rather than silently yielding zero results (which would look like "nothing exists").
    """
    text = decode_text(payload)
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data: Any = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            "web search payload is not JSON; supply a `parse` callable for this backend "
            f"({exc})"
        ) from exc

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, Mapping):
        for key in HIT_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, Mapping) and isinstance(value.get("value"), list):
                return [item for item in value["value"] if isinstance(item, dict)]
        raise ProviderError(f"web search envelope has no recognisable hit list: keys={sorted(data)}")
    raise ProviderError(f"unsupported web search payload type {type(data).__name__}")


__all__ = ["HIT_KEYS", "WebProvider", "parse_web_payload"]
