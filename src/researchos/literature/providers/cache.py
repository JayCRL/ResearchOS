"""Response caching: make a search run auditable, replayable and cheap — without hiding its source.

Why a cache belongs in a *research* system rather than being an optimisation detail:

* **Auditability.** A search result set that changes every time it is fetched cannot be reasoned about
  later. The cache file stores the untouched provider payload, so "the plan found 14 papers on
  Tuesday" can be re-checked on Friday against exactly the bytes that produced it.
* **Offline replay.** A reviewer with no network can re-run the pipeline against recorded payloads.
* **Rate limits.** Providers such as Semantic Scholar and GitHub punish repeated identical queries,
  and a planner deliberately issues many queries per target.

What the cache must never do: change the answer. It stores whole provider responses keyed by
``sha256_text(f"{kind}|{query}|{limit}")`` and revalidates nothing — a corrupt, truncated or
unreadable entry is treated as a *miss* and the live provider is called, because returning a partial
result set would under-report prior work silently. The wrapped provider's ``kind`` is reported
unchanged (a cache is not a source), so provider accounting in coverage can never be laundered
through the cache plane.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from ...models.common import sha256_text, utcnow
from ...models.literature import ProviderKind, ProviderRecord
from .base import LiteratureProvider, ProviderError

#: Bumped when the on-disk envelope changes shape; older entries are ignored rather than guessed at.
CACHE_VERSION = 1

#: Default freshness window. Long enough to make a re-run free, short enough that a citation count
#: is not quoted from a stale snapshot a month later.
DEFAULT_TTL_SECONDS: int = 7 * 24 * 3600


class CachedProvider(LiteratureProvider):
    """Read-through disk cache around another provider."""

    def __init__(
        self,
        provider: LiteratureProvider,
        cache_dir: Path | str,
        *,
        ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        # Counters exist so a test (or a report) can prove a second call was served from disk.
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.corrupt_entries = 0

    # ------------------------------------------------------------------ identity

    @property
    def kind(self) -> ProviderKind:  # type: ignore[override]
        """The wrapped provider's kind: a cache is a plane, not a source of literature."""
        return self.provider.kind

    @property
    def requires_network(self) -> bool:  # type: ignore[override]
        return self.provider.requires_network

    @property
    def name(self) -> str:
        return f"cache({self.provider.name})"

    def available(self) -> bool:
        return self.provider.available()

    # ------------------------------------------------------------------ keys and paths

    def cache_key(self, query: str, *, limit: int = 20) -> str:
        """``sha256_text(f"{kind}|{query}|{limit}")`` — the documented cache identity."""
        return sha256_text(f"{self.provider.kind.value}|{query}|{limit}")

    def fetch_key(self, identifier: str) -> str:
        return sha256_text(f"{self.provider.kind.value}|fetch|{identifier.strip()}")

    def cache_path(self, query: str, *, limit: int = 20) -> Path:
        return self.cache_dir / f"{self.cache_key(query, limit=limit)}.json"

    # ------------------------------------------------------------------ API

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        path = self.cache_path(query, limit=limit)
        cached = self._read(path, query=query, limit=limit)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        records = self.provider.search(query, limit=limit)
        self._write(path, records, query=query, limit=limit)
        return records

    def fetch(self, identifier: str) -> ProviderRecord | None:
        """Cached fetch. A miss (``None``) is stored as an empty record list, so repeating a lookup
        that found nothing does not hit the network again inside the TTL."""
        path = self.cache_dir / f"{self.fetch_key(identifier)}.json"
        cached = self._read(path, query=f"fetch:{identifier}", limit=1)
        if cached is not None:
            self.hits += 1
            return cached[0] if cached else None
        self.misses += 1
        record = self.provider.fetch(identifier)
        self._write(path, [record] if record is not None else [], query=f"fetch:{identifier}", limit=1)
        return record

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """Delegated, not cached: expansion is keyed by the seed, and seeds are already cached."""
        return self.provider.expand(record)

    # ------------------------------------------------------------------ replay

    def replay(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        """Return a recorded result set from disk without touching the provider.

        Raises :class:`ProviderError` when nothing usable is recorded — a replay that quietly
        returned ``[]`` would be indistinguishable from "the search found nothing".
        """
        path = self.cache_path(query, limit=limit)
        cached = self._read(path, query=query, limit=limit, ignore_ttl=True)
        if cached is None:
            raise ProviderError(f"no usable cache entry for {query!r} at {path}")
        return cached

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider.kind.value,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "corrupt_entries": self.corrupt_entries,
            "ttl_seconds": self.ttl_seconds,
            "cache_dir": str(self.cache_dir),
        }

    # ------------------------------------------------------------------ disk

    def _read(
        self,
        path: Path,
        *,
        query: str,
        limit: int,
        ignore_ttl: bool = False,
    ) -> list[ProviderRecord] | None:
        """Return the cached records, or ``None`` for miss/expired/corrupt.

        Corruption is counted, never raised: one bad file must not break a search run, but it must
        also not be silently treated as valid data.
        """
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or payload.get("cache_version") != CACHE_VERSION:
                self.corrupt_entries += 1
                return None
            if not ignore_ttl and self._expired(payload.get("stored_at")):
                return None
            raw_records = payload.get("records")
            if not isinstance(raw_records, list):
                self.corrupt_entries += 1
                return None
            return [ProviderRecord.model_validate(item) for item in raw_records]
        except (json.JSONDecodeError, ValidationError, OSError, TypeError, ValueError):
            self.corrupt_entries += 1
            return None

    def _expired(self, stored_at: object) -> bool:
        if self.ttl_seconds is None:
            return False
        if not isinstance(stored_at, str):
            return True
        try:
            stamped = datetime.fromisoformat(stored_at)
        except ValueError:
            return True
        if stamped.tzinfo is None:
            stamped = stamped.replace(tzinfo=timezone.utc)
        return utcnow() - stamped > timedelta(seconds=self.ttl_seconds)

    def _write(self, path: Path, records: Sequence[ProviderRecord], *, query: str, limit: int) -> None:
        """Write the envelope atomically. A cache write failure is not a search failure."""
        envelope = {
            "cache_version": CACHE_VERSION,
            "note": (
                "records[].raw holds the untouched provider payload for this call; this file is the "
                "audit trail of one provider request and may be replayed offline"
            ),
            "provider_kind": self.provider.kind.value,
            "provider_name": self.provider.name,
            "query": query,
            "limit": limit,
            "stored_at": utcnow().isoformat(),
            "record_count": len(records),
            "records": [record.model_dump(mode="json") for record in records],
        }
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(envelope, ensure_ascii=False, indent=1), encoding="utf-8")
            temporary.replace(path)
            self.writes += 1
        except OSError:  # pragma: no cover - unwritable cache directory
            pass


def cache_providers(
    providers: Sequence[LiteratureProvider],
    cache_dir: Path | str,
    *,
    ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
) -> list[LiteratureProvider]:
    """Wrap every network-touching provider in a :class:`CachedProvider`.

    Local providers (``requires_network == False``) and already-cached ones are returned unchanged:
    caching a local bibliography adds a layer that can only go stale, and double-wrapping would
    double the files for one logical call.
    """
    out: list[LiteratureProvider] = []
    for provider in providers:
        if isinstance(provider, CachedProvider) or not provider.requires_network:
            out.append(provider)
        else:
            out.append(CachedProvider(provider, cache_dir, ttl_seconds=ttl_seconds))
    return out


def read_cached_records(cache_dir: Path | str, key: str) -> list[ProviderRecord]:
    """Read a raw cache file by key. Used by audit tooling that has the key but not the provider."""
    path = Path(cache_dir) / f"{key}.json"
    if not path.is_file():
        raise ProviderError(f"no cache file at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ProviderRecord.model_validate(item) for item in payload.get("records", [])]


__all__ = [
    "CACHE_VERSION",
    "DEFAULT_TTL_SECONDS",
    "CachedProvider",
    "cache_providers",
    "read_cached_records",
]
