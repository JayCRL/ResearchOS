"""arXiv provider: Atom feed in, :class:`ProviderRecord` out.

Why the Atom feed rather than a scrape: the export API is stable, paginated and returns structured
author/abstract/category fields, and it is the one source that covers preprints *before* they have
a DOI or appear in Crossref — which is exactly the window where a novelty question is usually asked.

Notes that matter for research integrity:

* arXiv has **no citation graph**. :meth:`ArxivProvider.expand` therefore returns reference ids only
  when the entry text itself names them (``arXiv:2301.00001`` inside a comment), and never invents
  citations. An empty list here means "not knowable from this source", which is recorded, not
  guessed.
* The per-entry Atom XML is preserved verbatim in ``ProviderRecord.raw["entry_xml"]`` so a parsed
  record can always be audited against the bytes it came from.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Callable, ClassVar
from urllib.parse import quote

from ...models.literature import ProviderKind, ProviderRecord
from .base import Fetcher, LiteratureProvider, ProviderError, build_record, clean_text, decode_text, default_fetcher

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

DEFAULT_ARXIV_BASE_URL = "https://export.arxiv.org/api/query"

#: An expansion that resolves every id in a survey's comment field would flood a search run.
MAX_EXPANSION_IDS = 20

#: Matches ``arXiv:2301.00001``, ``arXiv:2301.00001v2`` and old-style ``arXiv:cs/0701001``.
_ARXIV_ID_RE = re.compile(r"arxiv[:\s/]*((?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7})|(?:\d{4}\.\d{4,5}))(?:v\d+)?", re.IGNORECASE)
_VERSION_RE = re.compile(r"v\d+$")


class ArxivProvider(LiteratureProvider):
    """Search and fetch arXiv preprints through the public Atom export API."""

    kind: ClassVar[ProviderKind] = ProviderKind.ARXIV

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        base_url: str = DEFAULT_ARXIV_BASE_URL,
        timeout: float | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._base_url = base_url
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "arXiv (Atom export API)"

    # ------------------------------------------------------------------ request plumbing

    def _get(self, url: str) -> bytes:
        if self._fetcher is not None:
            return self._fetcher(url)
        return default_fetcher(url, timeout=self._timeout or 30.0)

    def search_url(self, query: str, *, limit: int = 20) -> str:
        """The exact URL a search would request — exposed so a run can be replayed by hand."""
        return (
            f"{self._base_url}?search_query=all:{quote(query)}"
            f"&start=0&max_results={max(1, limit)}&sortBy=relevance&sortOrder=descending"
        )

    def fetch_url(self, identifier: str) -> str:
        return f"{self._base_url}?id_list={quote(identifier.strip())}&max_results=1"

    # ------------------------------------------------------------------ API

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        url = self.search_url(query, limit=limit)
        entries = self._entries(self._get(url))
        return [self._record(entry, url=url, query=query) for entry in entries[: max(0, limit)]]

    def fetch(self, identifier: str) -> ProviderRecord | None:
        url = self.fetch_url(identifier)
        entries = self._entries(self._get(url))
        return self._record(entries[0], url=url, query=None) if entries else None

    def expand(
        self, record: ProviderRecord, *, resolve: bool = True
    ) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """References named in the entry text, resolved to records when ``resolve`` is true.

        Citations stay empty: arXiv publishes no citation graph, so anything returned there would be
        an invention. Reference ids are only taken from text the entry itself contains
        (``arXiv:2301.00001`` in a comment, journal ref or abstract).
        """
        haystack = " ".join(
            str(part)
            for part in (
                record.raw.get("comment", ""),
                record.raw.get("journal_ref", ""),
                record.abstract or "",
                record.raw.get("summary", ""),
            )
            if part
        )
        references: list[str] = list(record.references)
        for candidate in arxiv_ids_in_text(haystack):
            if candidate not in references and candidate.lower() != (record.arxiv_id or "").lower():
                references.append(candidate)
        if not resolve:
            return [], []
        resolved: list[ProviderRecord] = []
        for identifier in references[:MAX_EXPANSION_IDS]:
            try:
                found = self.fetch(identifier)
            except ProviderError:
                continue  # one unresolvable reference must not abort the whole expansion
            if found is not None:
                resolved.append(found)
        return resolved, []

    # ------------------------------------------------------------------ parsing

    def _entries(self, payload: bytes) -> list[ET.Element]:
        text = decode_text(payload)
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise ProviderError(f"arXiv returned malformed Atom: {exc}") from exc
        if root.tag != f"{ATOM}feed":
            raise ProviderError(f"arXiv response is not an Atom feed (root tag {root.tag!r})")
        return list(root.findall(f"{ATOM}entry"))

    def _record(self, entry: ET.Element, *, url: str, query: str | None) -> ProviderRecord:
        entry_id = clean_text(_text(entry, f"{ATOM}id"))
        arxiv_id = _arxiv_id_from_url(entry_id)
        authors = [
            clean_text(_text(author, f"{ATOM}name"))
            for author in entry.findall(f"{ATOM}author")
        ]
        published = _text(entry, f"{ATOM}published") or _text(entry, f"{ATOM}updated")
        pdf_url = ""
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        primary_category = entry.find(f"{ARXIV}primary_category")
        categories = [c.get("term", "") for c in entry.findall(f"{ATOM}category")]
        journal_ref = clean_text(_text(entry, f"{ARXIV}journal_ref"))
        comment = clean_text(_text(entry, f"{ARXIV}comment"))
        entry_xml = ET.tostring(entry, encoding="unicode")
        return build_record(
            ProviderKind.ARXIV,
            arxiv_id or entry_id,
            _text(entry, f"{ATOM}title"),
            abstract=_text(entry, f"{ATOM}summary"),
            authors=authors,
            year=published,
            venue=journal_ref or "arXiv preprint",
            doi=clean_text(_text(entry, f"{ARXIV}doi")) or None,
            arxiv_id=arxiv_id,
            url=entry_id,
            open_access_pdf=pdf_url,
            license=clean_text(_text(entry, f"{ARXIV}license")) or None,
            query=query,
            raw={
                "source": "arxiv_atom",
                "request_url": url,
                "entry_xml": entry_xml,
                "summary": clean_text(_text(entry, f"{ATOM}summary")),
                "comment": comment,
                "journal_ref": journal_ref,
                "published": published,
                "updated": _text(entry, f"{ATOM}updated"),
                "primary_category": primary_category.get("term") if primary_category is not None else None,
                "categories": [c for c in categories if c],
            },
        )


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path)
    return found.text if found is not None and found.text else ""


def _arxiv_id_from_url(entry_id: str) -> str:
    """``http://arxiv.org/abs/2301.00001v3`` -> ``2301.00001`` (version stripped for identity)."""
    if not entry_id:
        return ""
    tail = entry_id.rstrip("/").split("/abs/")[-1]
    return _VERSION_RE.sub("", tail)


def arxiv_ids_in_text(text: str) -> list[str]:
    """Every distinct arXiv id mentioned in ``text``, in order of appearance.

    Public because the reference-expansion path and the graph both need the same id extraction.
    """
    seen: list[str] = []
    for match in _ARXIV_ID_RE.finditer(text or ""):
        candidate = _VERSION_RE.sub("", match.group(1))
        if candidate not in seen:
            seen.append(candidate)
    return seen


__all__ = ["DEFAULT_ARXIV_BASE_URL", "ArxivProvider", "arxiv_ids_in_text"]
