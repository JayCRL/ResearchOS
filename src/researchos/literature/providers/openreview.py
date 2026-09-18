"""OpenReview provider: conference submissions and reviews, including work that never appeared elsewhere.

Why it matters for novelty: a large fraction of ML mechanism work is first visible as an OpenReview
submission — possibly rejected, possibly withdrawn, and almost never indexed by Crossref. A search
that ignores this venue systematically over-states novelty for exactly the kind of work ResearchOS
is built to audit.

The API has two content shapes in the wild (v1 flat fields, v2 ``{"value": ...}`` wrappers) and both
are parsed here rather than being pinned to one, because a re-submitted venue keeps serving the
older shape for years.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping
from urllib.parse import quote

from ...models.literature import ProviderKind, ProviderRecord
from .base import Fetcher, LiteratureProvider, ProviderError, build_record, clean_text, decode_text, default_fetcher

DEFAULT_OPENREVIEW_BASE_URL = "https://api2.openreview.net"


class OpenReviewProvider(LiteratureProvider):
    """Search and fetch OpenReview notes."""

    kind: ClassVar[ProviderKind] = ProviderKind.OPENREVIEW

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        base_url: str = DEFAULT_OPENREVIEW_BASE_URL,
        timeout: float | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._base_url = base_url
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "OpenReview (API v2, v1 payloads tolerated)"

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
            raise ProviderError(f"OpenReview returned non-JSON for {url}: {exc}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise ProviderError(f"OpenReview error for {url}: {data.get('error')}")
        return data

    def search_url(self, query: str, *, limit: int = 20) -> str:
        return (
            f"{self._base_url}/notes/search?term={quote(query)}"
            f"&limit={max(1, limit)}&type=terms&content=all&group=all&source=all"
        )

    def fetch_url(self, identifier: str) -> str:
        return f"{self._base_url}/notes?id={quote(identifier.strip())}"

    # ------------------------------------------------------------------ API

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        url = self.search_url(query, limit=limit)
        data = self._json(url)
        notes = _notes(data)
        records = [self._record(note, url=url, query=query) for note in notes]
        records = [r for r in records if r.title]
        return records[: max(0, limit)]

    def fetch(self, identifier: str) -> ProviderRecord | None:
        url = self.fetch_url(identifier)
        data = self._json(url)
        notes = _notes(data)
        if not notes and isinstance(data, Mapping) and "id" in data:
            notes = [data]
        return self._record(notes[0], url=url, query=None) if notes else None

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """OpenReview publishes no citation graph; references are returned only if the payload has them."""
        cached = record.raw.get("references")
        if isinstance(cached, list):
            return [self._record(item, url="", query=None) for item in cached if isinstance(item, Mapping)], []
        return [], []

    # ------------------------------------------------------------------ parsing

    def _record(self, note: Mapping[str, Any], *, url: str, query: str | None) -> ProviderRecord:
        content = note.get("content")
        if not isinstance(content, Mapping):
            content = {}
        authors = _value(content, "authors") or []
        if isinstance(authors, str):
            authors = [part.strip() for part in authors.split(",") if part.strip()]
        pdf = _value(content, "pdf")
        venue = _value(content, "venue") or _value(content, "conference") or note.get("invitation")
        return build_record(
            ProviderKind.OPENREVIEW,
            str(note.get("id") or note.get("forum") or ""),
            _value(content, "title") or "",
            abstract=_value(content, "abstract"),
            authors=list(authors) if isinstance(authors, (list, tuple)) else [],
            year=_note_year(note),
            venue=venue,
            url=f"https://openreview.net/forum?id={note.get('forum') or note.get('id')}",
            open_access_pdf=f"https://openreview.net{pdf}" if isinstance(pdf, str) and pdf.startswith("/") else pdf,
            query=query,
            raw={
                "source": "openreview_notes",
                "request_url": url,
                "payload": dict(note),
                "keywords": _value(content, "keywords"),
                "venueid": _value(content, "venueid"),
            },
        )


def _notes(data: Any) -> list[Mapping[str, Any]]:
    """Extract the note list from either ``{"notes": [...]}`` or a bare list response."""
    if isinstance(data, Mapping):
        notes = data.get("notes")
        if isinstance(notes, list):
            return [note for note in notes if isinstance(note, Mapping)]
        return []
    if isinstance(data, list):
        return [note for note in data if isinstance(note, Mapping)]
    return []


def _value(content: Mapping[str, Any], key: str) -> Any:
    """Read a content field, unwrapping the API-v2 ``{"value": x}`` shape when present."""
    raw = content.get(key)
    if isinstance(raw, Mapping) and "value" in raw:
        return raw["value"]
    return raw


def _note_year(note: Mapping[str, Any]) -> int | None:
    """``cdate``/``tcdate`` are epoch milliseconds; ``odate``/``pdate`` are ISO strings."""
    for key in ("pdate", "odate", "cdate", "tcdate", "mdate"):
        value = note.get(key)
        if isinstance(value, (int, float)) and value > 10_000_000_000:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).year
        if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


__all__ = ["DEFAULT_OPENREVIEW_BASE_URL", "OpenReviewProvider"]
