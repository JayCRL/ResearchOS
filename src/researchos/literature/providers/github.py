"""GitHub provider: search *repositories*, not papers.

Why a code host belongs in a literature OS: for AI/ML mechanism work, the implementation frequently
precedes or replaces the paper. A repository with 4k stars that implements the same mechanism is a
prior-art fact that no bibliographic database contains. Excluding it would make "we found no prior
work" an artefact of where we looked.

Honesty rules encoded here:

* ``ProviderRecord.venue`` is always the literal ``"repository"`` — a repository must never be
  rendered in a bibliography as if it were a published paper.
* ``citation_count`` is left ``None``. Stars are *popularity*, not citations; conflating them would
  corrupt every later ranking or coverage statistic that reads that field. Stars and topics live in
  ``raw`` where they cannot be mistaken for bibliometrics.
* ``title`` is the ``owner/name`` slug, because a repository description is often empty or a sentence.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Mapping
from urllib.parse import quote

from ...models.literature import ProviderKind, ProviderRecord
from .base import Fetcher, LiteratureProvider, ProviderError, build_record, clean_authors, decode_text, default_fetcher

DEFAULT_GITHUB_BASE_URL = "https://api.github.com/search/repositories"

#: The literal venue label for every record this provider produces.
REPOSITORY_VENUE = "repository"


class GitHubProvider(LiteratureProvider):
    """Search repositories through the GitHub REST search API."""

    kind: ClassVar[ProviderKind] = ProviderKind.GITHUB

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        base_url: str = DEFAULT_GITHUB_BASE_URL,
        token: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._base_url = base_url
        self._token = token
        self._timeout = timeout

    @property
    def name(self) -> str:
        suffix = "" if self._token else " (unauthenticated: 10 requests/minute)"
        return f"GitHub (repository search){suffix}"

    # ------------------------------------------------------------------ request plumbing

    def _get(self, url: str) -> bytes:
        if self._fetcher is not None:
            return self._fetcher(url)
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return default_fetcher(url, headers=headers, timeout=self._timeout or 30.0)

    def _json(self, url: str) -> Any:
        payload = self._get(url)
        try:
            data = json.loads(decode_text(payload))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"GitHub returned non-JSON for {url}: {exc}") from exc
        if isinstance(data, Mapping) and data.get("message") and not data.get("items"):
            raise ProviderError(f"GitHub error for {url}: {data.get('message')}")
        return data

    def search_url(self, query: str, *, limit: int = 20) -> str:
        return (
            f"{self._base_url}?q={quote(query)}&per_page={max(1, limit)}"
            "&sort=stars&order=desc"
        )

    def fetch_url(self, identifier: str) -> str:
        return f"https://api.github.com/repos/{identifier.strip()}"

    # ------------------------------------------------------------------ API

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        url = self.search_url(query, limit=limit)
        data = self._json(url)
        items = data.get("items") if isinstance(data, Mapping) else None
        if items is None:
            raise ProviderError("GitHub search response has no 'items' list")
        return [self._record(item, url=url, query=query) for item in items if isinstance(item, Mapping)][
            : max(0, limit)
        ]

    def fetch(self, identifier: str) -> ProviderRecord | None:
        url = self.fetch_url(identifier)
        data = self._json(url)
        if not isinstance(data, Mapping) or not data.get("full_name"):
            return None
        return self._record(data, url=url, query=None)

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """Repositories have no citation graph; forks/related repos are not prior *work* evidence."""
        return [], []

    # ------------------------------------------------------------------ parsing

    def _record(self, item: Mapping[str, Any], *, url: str, query: str | None) -> ProviderRecord:
        full_name = str(item.get("full_name") or "")
        owner = item.get("owner") if isinstance(item.get("owner"), Mapping) else {}
        topics = [str(t) for t in (item.get("topics") or [])]
        stars = int(item.get("stargazers_count") or 0)
        license_info = item.get("license") if isinstance(item.get("license"), Mapping) else {}
        return build_record(
            ProviderKind.GITHUB,
            full_name,
            full_name,
            abstract=item.get("description"),
            authors=clean_authors([owner.get("login")]) if owner else [],
            year=item.get("created_at"),
            venue=REPOSITORY_VENUE,
            url=item.get("html_url"),
            license=license_info.get("spdx_id"),
            query=query,
            raw={
                "source": "github_repositories",
                "request_url": url,
                "payload": dict(item),
                "stars": stars,
                "forks": int(item.get("forks_count") or 0),
                "topics": topics,
                "language": item.get("language"),
                "pushed_at": item.get("pushed_at"),
                "archived": bool(item.get("archived")),
                "is_fork": bool(item.get("fork")),
            },
        )


__all__ = ["DEFAULT_GITHUB_BASE_URL", "REPOSITORY_VENUE", "GitHubProvider"]
