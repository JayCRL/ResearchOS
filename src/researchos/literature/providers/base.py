"""The provider contract: what "searching the literature" concretely means here.

Why this module exists
----------------------
ResearchOS refuses to let literature knowledge live in a model's memory. Everything we claim
about prior work must arrive as a *record*: a payload fetched from a named source, parsed
deterministically, and stored with the untouched bytes next to the parsed fields. This module
defines the five things every source must be able to do (``available``, ``search``, ``fetch``,
``expand``, ``name``) so that the query planner, the coverage accountant and the novelty auditor
never need to know which source they are talking to.

Two design rules are load-bearing:

1. **Network access is injected, never implicit.** Every HTTP provider takes
   ``fetcher: Callable[[str], bytes]``. Tests pass a stub that returns a recorded payload, so the
   parsers are exercised without touching the network and a recording can be replayed forever.
   The production default (:func:`default_fetcher`) is a timeout-bounded ``urllib`` call.
2. **A provider that cannot answer says so.** :meth:`LiteratureProvider.available` is a promise,
   and a provider with no configured endpoint (see :class:`~researchos.literature.providers.web.WebProvider`)
   returns ``False`` rather than inventing plausible-looking results. Fabricated hits would poison
   coverage accounting, which is the one number a novelty verdict is allowed to lean on.

Lexical matching helpers (``tokenise``, ``jaccard``, ``lexical_similarity``) also live here, because
the offline fixture provider and the literature graph must score a claim the same way — two
different definitions of "similar" in one system is exactly how unexplainable rankings appear.
"""

from __future__ import annotations

import html
import re
import unicodedata
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Iterable, Mapping, Sequence

from ...models.literature import ProviderKind, ProviderRecord

# --------------------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------------------

#: A fetcher takes a request URL and returns the response body. Nothing else is assumed:
#: no sessions, no cookies, no redirect policy, no retry semantics.
Fetcher = Callable[[str], bytes]

#: Requests are bounded so a hanging provider cannot stall a research session forever.
DEFAULT_TIMEOUT_SECONDS: float = 30.0

#: Identifies ResearchOS to providers that police anonymous traffic (Crossref, GitHub).
USER_AGENT: str = "ResearchOS/0.1 (literature provider; +https://github.com/researchos)"


class ProviderError(Exception):
    """A provider could not answer: transport failure, malformed payload, or misconfiguration.

    Deliberately *not* a :class:`researchos.kernel.errors.ResearchOSError`: a provider failure is a
    fact about the outside world, not a violation of research-state rules, and the query planner
    records it on the :class:`~researchos.models.literature.PlannedQuery` as ``error`` instead of
    aborting the whole search.
    """


def default_fetcher(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bytes:
    """Fetch ``url`` with ``urllib``. Usable directly as a :data:`Fetcher` (``default_fetcher(url)``).

    ``headers`` is keyword-only so the function still satisfies ``Callable[[str], bytes]`` — which
    is what providers accept. Providers pass auth headers only on this production path; when a
    custom fetcher is injected, it is expected to carry whatever credentials it needs, because the
    injected callable is the only thing that knows how the request is really made.
    """
    request_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        request_headers.update({str(k): str(v) for k, v in headers.items()})
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - scheme-checked below
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        detail = ""
        try:
            detail = exc.read()[:400].decode("utf-8", "replace")
        except Exception:  # pragma: no cover - best effort diagnostics only
            detail = ""
        raise ProviderError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - network path
        raise ProviderError(f"transport failure for {url}: {exc}") from exc


def decode_text(payload: bytes, *, encoding: str = "utf-8") -> str:
    """Decode a provider payload defensively.

    Providers occasionally serve latin-1 or a truncated multi-byte sequence. Refusing to parse an
    otherwise usable response because of one bad byte would silently shrink coverage, so decoding
    is replacement-tolerant and the substitution characters stay visible in ``raw``.
    """
    if isinstance(payload, str):  # a stub fetcher may hand back text
        return payload
    return payload.decode(encoding, "replace")


# --------------------------------------------------------------------------------------
# Text normalisation (shared by providers, the graph and the novelty auditor)
# --------------------------------------------------------------------------------------

#: Words too common to carry topical meaning in a title/abstract/claim comparison.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been by for from has have how in into is it its of on or that the their
    then there these this those to using was were what when where which who why will with without
    we our us they them he she his her you your not no do does did can could should would may
    """.split()
)

_WORD_RE = re.compile(r"[0-9a-z]+(?:[-_][0-9a-z]+)*", re.IGNORECASE)
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")
_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]{0,200}>")
_MONTH_ALIASES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def normalise_title(title: str) -> str:
    """Aggressive title normalisation for cross-provider identity.

    ``"Fast Weights: A <Emph>{New}</Emph> Mechanism!"`` and ``"fast weights a new mechanism"`` must
    be recognised as the same paper, because providers disagree about case, punctuation, LaTeX and
    HTML escaping. Digits are kept (``gpt-3``), LaTeX commands are dropped while their arguments
    survive, and CJK characters are preserved rather than stripped to nothing.
    """
    text = html.unescape(title or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _LATEX_CMD_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return _WS_RE.sub(" ", text).strip()


def clean_text(value: object, *, max_length: int | None = None) -> str:
    """Collapse whitespace and strip markup from a provider-supplied string.

    Crossref ships JATS XML inside ``abstract`` and some web backends ship HTML snippets; storing
    those tags would put markup in front of a human reader and break token comparison later.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "\u2026"
    return text


def parse_year(value: object) -> int | None:
    """Best-effort year extraction that refuses implausible values.

    ``LiteraturePaper.year`` is constrained to 1500..2200; silently keeping ``"in press"`` or a
    four-digit page number would create a paper that cannot be stored at all.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        year = value
    else:
        match = re.search(r"(1[5-9]\d{2}|2\d{3})", str(value))
        if not match:
            return None
        year = int(match.group(1))
    return year if 1500 <= year <= 2200 else None


def parse_month(value: object) -> int | None:
    """Map a BibTeX month macro, month name or number onto 1..12 (``None`` when unknowable)."""
    if value is None:
        return None
    text = str(value).strip().strip("{}\"").lower()
    if not text:
        return None
    if text in _MONTH_ALIASES:
        return _MONTH_ALIASES[text]
    for token in re.split(r"[^a-z0-9]+", text):
        if token in _MONTH_ALIASES:
            return _MONTH_ALIASES[token]
    match = re.search(r"\b(1[0-2]|0?[1-9])\b", text)
    return int(match.group(1)) if match else None


def clean_authors(values: Iterable[object]) -> list[str]:
    """Normalise an author list: strings, ``{"name": ...}`` dicts or ``"Family, Given"`` forms."""
    out: list[str] = []
    for raw in values or []:
        if isinstance(raw, Mapping):
            name = raw.get("name") or raw.get("literal")
            if not name and (raw.get("given") or raw.get("family")):
                name = f"{raw.get('given', '')} {raw.get('family', '')}"
            text = clean_text(name)
        else:
            text = clean_text(raw)
        text = text.strip(" ,")
        if text and text not in out:
            out.append(text)
    return out


#: Accent commands as they appear in real bibliographies (``J{\"u}rgen``, ``Fran\c{c}ois``).
_ACCENT_MARKS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303",
    "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b",
    "r": "\u030a", "c": "\u0327", "k": "\u0328", "b": "\u0331", "d": "\u0323",
}
_ACCENT_RE = re.compile(r"\{?\\([a-zA-Z\"'`^~=.])\s*\{?([A-Za-z])\}?\}?")


def decode_latex_accents(text: str) -> str:
    """Turn ``J{\\"u}rgen`` into ``Jürgen`` for display fields (author names), never for raw payloads.

    Names are compared with human eyes and appear in citations; keeping the LaTeX source there would
    make ``citation_label()`` produce ``J{"u}rgen, 1992``. Titles are deliberately *not* passed
    through this, because their braces carry capitalisation information.
    """
    if not text or "\\" not in text:
        return text
    decoded = _ACCENT_RE.sub(
        lambda m: unicodedata.normalize("NFC", m.group(2) + _ACCENT_MARKS.get(m.group(1), "")),
        text,
    )
    return decoded.replace("{", "").replace("}", "")


def split_bibtex_authors(value: str) -> list[str]:
    """Split a BibTeX ``author`` field on the literal `` and `` separator.

    BibTeX's separator is case-insensitive and must be surrounded by whitespace, which is why
    ``Sandoval`` or ``Andersen`` never split. Braces are removed only from the outside: corporate
    authors such as ``{Google Brain}`` survive as one author.
    """
    parts = re.split(r"\s+and\s+", value.strip(), flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        name = part.strip().strip(",").strip()
        name = decode_latex_accents(name)
        name = re.sub(r"^\{+|\}+$", "", name).strip()
        name = _WS_RE.sub(" ", name)
        if name and name.lower() != "others" and name not in out:
            out.append(name)
    return out


def tokenise(text: str) -> list[str]:
    """Lowercase word tokens, LaTeX- and markup-tolerant, stopwords removed.

    Used for every lexical comparison in the literature plane. Deliberately not stemmed: a stemmer
    is another dependency and another source of unexplainable matches, and the score returned to
    the caller must be reconstructible by hand.
    """
    if not text:
        return []
    prepared = _LATEX_CMD_RE.sub(" ", html.unescape(text.lower()))
    prepared = prepared.replace("{", " ").replace("}", " ")
    return [t for t in _WORD_RE.findall(prepared) if t not in STOPWORDS and len(t) > 1]


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """Jaccard similarity of two token collections. ``0.0`` when either side is empty."""
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_similarity(left: str, right: str) -> float:
    """Token-overlap similarity of two free-text strings — the explainable similarity metric."""
    return jaccard(tokenise(left), tokenise(right))


def build_record(
    provider: ProviderKind,
    provider_id: str,
    title: str,
    *,
    abstract: object = None,
    authors: object = None,
    year: object = None,
    venue: object = None,
    doi: object = None,
    arxiv_id: object = None,
    url: object = None,
    citation_count: object = None,
    open_access_pdf: object = None,
    license: object = None,
    references: Sequence[str] = (),
    citations: Sequence[str] = (),
    query: str | None = None,
    score: float | None = None,
    raw: Mapping[str, object] | None = None,
    abstract_max_length: int | None = None,
) -> ProviderRecord:
    """Assemble a :class:`ProviderRecord` without letting a missing field abort the parse.

    Every provider maps its own envelope onto this one constructor, so "what a record is" is
    defined in exactly one place.

    ``title`` falls back to the provider identifier because ``ProviderRecord.title`` has
    ``min_length=1``. That fallback is a last resort for a malformed envelope, *not* a licence to
    create a paper: the record is marked ``raw["_title_missing"] = True`` so the ingest path can
    refuse it. A title invented from an id would poison the literature map and the closest-prior-work
    ranking, which is the opposite of what this system is for.
    """
    clean_title = clean_text(title)
    title_missing = not clean_title
    if title_missing:
        clean_title = clean_text(provider_id) or "untitled"
    raw_payload = dict(raw or {})
    if title_missing:
        raw_payload["_title_missing"] = True
    return ProviderRecord(
        provider=provider,
        provider_id=str(provider_id),
        title=clean_title,
        authors=clean_authors(authors or []),
        year=parse_year(year),
        venue=clean_text(venue) or None,
        doi=clean_text(doi) or None,
        arxiv_id=clean_text(arxiv_id) or None,
        url=clean_text(url) or None,
        abstract=clean_text(abstract, max_length=abstract_max_length) or None,
        citation_count=int(citation_count) if isinstance(citation_count, (int, float)) and not isinstance(citation_count, bool) else None,
        references=[str(r) for r in references if r],
        citations=[str(c) for c in citations if c],
        open_access_pdf=clean_text(open_access_pdf) or None,
        license=clean_text(license) or None,
        query=query,
        score=score,
        raw=raw_payload,
    )


# --------------------------------------------------------------------------------------
# The provider contract
# --------------------------------------------------------------------------------------


class LiteratureProvider(ABC):
    """A source of literature records.

    Subclasses declare :attr:`kind` and :attr:`requires_network`, implement :meth:`search`, and
    override :meth:`fetch` / :meth:`expand` when the source can support them. The defaults are
    honest abstentions — returning an empty list is a *recorded* answer, whereas guessing is not.
    """

    kind: ClassVar[ProviderKind] = ProviderKind.MANUAL
    requires_network: ClassVar[bool] = True

    @property
    def name(self) -> str:
        """Human label used in reports and error messages."""
        return self.kind.value

    def available(self) -> bool:
        """Whether this provider can currently answer at all.

        Must not perform I/O: the query planner calls it before every search and a probe that
        costs a request would double the traffic of a search run.
        """
        return True

    @abstractmethod
    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        """Return up to ``limit`` records for ``query``. Raises :class:`ProviderError` on failure."""

    def fetch(self, identifier: str) -> ProviderRecord | None:
        """Resolve one identifier (DOI, arXiv id, repository name, BibTeX key). ``None`` if absent."""
        return None

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """Return ``(references, citations)`` for a record.

        The default abstains: a provider with no citation graph must say "I do not know" rather
        than return the paper's own bibliography as if it were data it retrieved.
        """
        return [], []

    # ------------------------------------------------------------------ helpers

    def _require_available(self) -> None:
        if not self.available():
            raise ProviderError(
                f"provider {self.name!r} is not available (no endpoint configured or no data)"
            )

    def describe(self) -> dict[str, object]:
        """Machine-readable description used by ``researchos lit providers`` style reports."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "requires_network": self.requires_network,
            "available": self.available(),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<{type(self).__name__} kind={self.kind.value} available={self.available()}>"


class OfflineProvider(LiteratureProvider):
    """A deterministic provider over a fixed corpus — the fixture that makes tests meaningful.

    Why it returns its whole corpus by default: a fixture is a *closed* world. Keyword filtering
    would make coverage arithmetic depend on how lucky the query templates happened to be, which
    turns "did the planner cover the field?" into "did the test author guess good synonyms?".
    With ``require_overlap=True`` the provider behaves like :class:`LocalBibProvider` and returns
    only records sharing tokens with the query, which is what the filtering tests exercise.
    """

    kind: ProviderKind = ProviderKind.LOCAL_BIB
    requires_network: bool = False

    def __init__(
        self,
        records: Sequence[ProviderRecord],
        *,
        kind: ProviderKind = ProviderKind.LOCAL_BIB,
        require_overlap: bool = False,
        label: str | None = None,
    ) -> None:
        self._records: list[ProviderRecord] = list(records)
        self.kind = kind  # type: ignore[assignment] - instance-level so fixtures can impersonate any source
        self.require_overlap = require_overlap
        self._label = label
        self.search_calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return self._label or f"offline-fixture({self.kind.value}, {len(self._records)} records)"

    def available(self) -> bool:
        return True

    def _ranked(self, query: str) -> list[ProviderRecord]:
        tokens = tokenise(query)
        scored: list[tuple[float, str, ProviderRecord]] = []
        for record in self._records:
            haystack = f"{record.title} {record.abstract or ''}"
            score = jaccard(tokens, tokenise(haystack))
            scored.append((score, record.provider_id, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if self.require_overlap:
            scored = [item for item in scored if item[0] > 0.0]
        return [
            record.model_copy(update={"query": query, "score": round(score, 4)})
            for score, _, record in scored
        ]

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        self.search_calls.append((query, limit))
        return self._ranked(query)[: max(0, limit)]

    def fetch(self, identifier: str) -> ProviderRecord | None:
        wanted = identifier.strip()
        normalised = normalise_title(wanted)
        for record in self._records:
            if wanted in {record.provider_id, record.doi, record.arxiv_id, record.url}:
                return record
            if normalised and normalise_title(record.title) == normalised:
                return record
        return None

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """Resolve the record's own reference/citation id lists against the fixture corpus.

        This makes citation expansion testable offline: the ids in ``record.references`` are looked
        up in the same closed world instead of being invented by a network call.
        """
        references = [r for r in (self.fetch(ref) for ref in record.references) if r is not None]
        citations = [r for r in (self.fetch(cite) for cite in record.citations) if r is not None]
        return references, citations


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "Fetcher",
    "LiteratureProvider",
    "OfflineProvider",
    "ProviderError",
    "STOPWORDS",
    "USER_AGENT",
    "build_record",
    "clean_authors",
    "clean_text",
    "decode_text",
    "default_fetcher",
    "jaccard",
    "lexical_similarity",
    "normalise_title",
    "parse_month",
    "parse_year",
    "split_bibtex_authors",
    "tokenise",
]
