"""Crossref provider: the DOI registry, used as the *publication of record* for venue/year metadata.

Why it is here next to arXiv: arXiv preprints frequently appear later as a conference or journal
paper with a different title and a DOI. Crossref is how the literature plane learns that two
records are the same work at a later stage of its life, and it is the only source in this package
whose identifiers (DOIs) are stable enough to serve as the primary identity key.

Crossref has no outgoing citation list — it publishes a citation *count* only. Citation ids are
therefore never fabricated here: ``citations`` is always empty and ``citation_count`` is the
reported number, kept in the record precisely so a report can say "cited 412 times, citing works
not enumerated by this source".
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Mapping
from urllib.parse import quote

from ...models.literature import ProviderKind, ProviderRecord
from .base import Fetcher, LiteratureProvider, ProviderError, build_record, clean_text, decode_text, default_fetcher

DEFAULT_CROSSREF_BASE_URL = "https://api.crossref.org/works"

#: Crossref asks clients to identify themselves ("polite pool") for better service and stability.
DEFAULT_MAILTO = "researchos@example.invalid"


class CrossrefProvider(LiteratureProvider):
    """Search and fetch DOI metadata through the Crossref REST API."""

    kind: ClassVar[ProviderKind] = ProviderKind.CROSSREF

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        base_url: str = DEFAULT_CROSSREF_BASE_URL,
        mailto: str = DEFAULT_MAILTO,
        timeout: float | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._base_url = base_url
        self._mailto = mailto
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "Crossref (REST API)"

    # ------------------------------------------------------------------ request plumbing

    def _get(self, url: str) -> bytes:
        if self._fetcher is not None:
            return self._fetcher(url)
        return default_fetcher(url, timeout=self._timeout or 30.0)

    def _json(self, url: str) -> Any:
        payload = self._get(url)
        try:
            data = json.loads(decode_text(payload))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Crossref returned non-JSON for {url}: {exc}") from exc
        if isinstance(data, dict) and data.get("status") not in (None, "ok"):
            raise ProviderError(f"Crossref reported status {data.get('status')!r} for {url}")
        return data

    def search_url(self, query: str, *, limit: int = 20) -> str:
        return (
            f"{self._base_url}?query={quote(query)}&rows={max(1, limit)}"
            f"&select=DOI,title,author,issued,container-title,abstract,URL,license,type,"
            f"reference,is-referenced-by-count&mailto={quote(self._mailto)}"
        )

    def fetch_url(self, identifier: str) -> str:
        return f"{self._base_url}/{quote(identifier.strip())}?mailto={quote(self._mailto)}"

    # ------------------------------------------------------------------ API

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        url = self.search_url(query, limit=limit)
        data = self._json(url)
        items = _message_items(data)
        return [self._record(item, url=url, query=query) for item in items][: max(0, limit)]

    def fetch(self, identifier: str) -> ProviderRecord | None:
        url = self.fetch_url(identifier)
        data = self._json(url)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, Mapping):
            return None
        return self._record(message, url=url, query=None)

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """Resolve the DOIs Crossref listed under ``reference``; citations are not enumerable."""
        resolved: list[ProviderRecord] = []
        for reference in record.references[:20]:
            if not reference.lower().startswith("10."):
                continue
            try:
                found = self.fetch(reference)
            except ProviderError:
                continue  # a missing DOI reference must not abort expansion
            if found is not None:
                resolved.append(found)
        return resolved, []

    # ------------------------------------------------------------------ parsing

    def _record(self, item: Mapping[str, Any], *, url: str, query: str | None) -> ProviderRecord:
        titles = item.get("title") or []
        title = titles[0] if isinstance(titles, list) and titles else str(titles or "")
        containers = item.get("container-title") or []
        venue = containers[0] if isinstance(containers, list) and containers else str(containers or "")
        authors = [
            author.get("name")
            or " ".join(part for part in (author.get("given"), author.get("family")) if part)
            for author in (item.get("author") or [])
            if isinstance(author, Mapping)
        ]
        licenses = item.get("license") or []
        license_url = licenses[0].get("URL") if licenses and isinstance(licenses[0], Mapping) else None
        references: list[str] = []
        for reference in item.get("reference") or []:
            if not isinstance(reference, Mapping):
                continue
            identifier = reference.get("DOI") or reference.get("article-title") or reference.get("unstructured")
            if identifier:
                references.append(clean_text(identifier))
        return build_record(
            ProviderKind.CROSSREF,
            str(item.get("DOI") or ""),
            title,
            abstract=item.get("abstract"),
            authors=authors,
            year=_issued_year(item.get("issued")),
            venue=venue,
            doi=item.get("DOI"),
            url=item.get("URL"),
            citation_count=item.get("is-referenced-by-count"),
            license=license_url,
            references=references,
            query=query,
            raw={
                "source": "crossref_works",
                "request_url": url,
                "payload": dict(item),
                "type": item.get("type"),
                "reference_count": len(references),
            },
        )


def _message_items(data: Any) -> list[Mapping[str, Any]]:
    """Pull ``message.items`` out of a Crossref search envelope, tolerating both wrapper shapes."""
    message = data.get("message") if isinstance(data, Mapping) else None
    if isinstance(message, Mapping):
        items = message.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, Mapping)]
        if "DOI" in message:  # a single-work response, reused as a one-item list
            return [message]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, Mapping)]
    raise ProviderError("Crossref response has no message.items list")


def _issued_year(issued: Any) -> int | None:
    """Crossref dates are ``{"date-parts": [[2021, 5, 3]]}`` and are frequently partial."""
    if not isinstance(issued, Mapping):
        return None
    parts = issued.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        return parts[0][0] if isinstance(parts[0][0], int) else None
    return None


__all__ = ["DEFAULT_CROSSREF_BASE_URL", "CrossrefProvider"]
