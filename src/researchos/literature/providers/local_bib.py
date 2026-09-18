"""Local BibTeX provider: the researcher's own bibliography, parsed by hand.

Why hand-written rather than a library: this package must run on the standard library alone, and a
BibTeX subset that is *explicitly specified* is more trustworthy than a dependency whose edge-case
behaviour is unknown. The subset implemented here covers what real ``.bib`` files actually use:

* ``@type{key, field = {value}, ...}``, ``field = "value"`` and bare values (``year = 2021``);
* nested braces inside a value, which is how titles keep capitalisation (``{GPT}-2``);
* ``author = {A and B and C}`` split on the BibTeX separator (whitespace-delimited, case-insensitive,
  so ``Sandoval`` never splits);
* ``@string{abbr = "..."}`` abbreviations, resolved when a field value is exactly an abbreviation
  (the untouched text is kept in ``raw`` either way);
* ``@comment``, ``@preamble`` and ``%`` line comments ignored — they are not publications.

Two invariants this provider must not break: ``requires_network`` is ``False`` (a bibliography is
local data), and a parsed entry is *metadata only*. The resulting
:class:`~researchos.models.literature.LiteraturePaper` therefore lands at
``FulltextStatus.ABSTRACT_LEVEL_ONLY`` (or ``UNKNOWN`` when there is no abstract): holding a BibTeX
record is not the same as having read the paper, and the model's validators exist to make that
distinction unforgeable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from ...models.literature import ProviderKind, ProviderRecord
from .base import (
    LiteratureProvider,
    ProviderError,
    build_record,
    jaccard,
    normalise_title,
    parse_month,
    parse_year,
    split_bibtex_authors,
    tokenise,
)

#: Entry types that never describe a publication.
NON_PUBLICATION_TYPES: frozenset[str] = frozenset({"comment", "preamble", "string"})

#: ``key = value`` field names are case-insensitive in BibTeX; the canonical case is used for output.
_FIELD_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
_WS_RE = re.compile(r"\s+")
_ARXIV_EPRINT_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")


@dataclass
class BibEntry:
    """One parsed BibTeX entry, with the untouched text it was parsed from."""

    entry_type: str
    key: str
    fields: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""
    resolved_abbreviations: dict[str, str] = field(default_factory=dict)

    def get(self, name: str, default: str = "") -> str:
        return self.fields.get(name.lower(), default)

    def summary(self) -> str:
        return f"@{self.entry_type}{{{self.key}, ...}} ({len(self.fields)} fields)"


class BibParseError(ProviderError):
    """The file is not parseable as the BibTeX subset this provider supports."""


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse BibTeX ``text`` into entries. Raises :class:`BibParseError` on unbalanced braces.

    Refusing to guess matters: a silently half-parsed bibliography would under-report prior work,
    and under-reported prior work is exactly what makes a novelty claim false.
    """
    entries: list[BibEntry] = []
    abbreviations: dict[str, str] = {}
    index = 0
    length = len(text)

    while True:
        at = text.find("@", index)
        if at < 0:
            break
        cursor = at + 1
        match = _FIELD_NAME_RE.match(text, cursor)
        if match is None:
            index = cursor
            continue
        entry_type = match.group(0).lower()
        cursor = match.end()
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length or text[cursor] not in "{(":
            index = cursor
            continue
        opener = text[cursor]
        closer = "}" if opener == "{" else ")"
        body_start = cursor + 1
        body_end = _find_closing(text, body_start, opener, closer)
        if body_end < 0:
            raise BibParseError(
                f"unbalanced '{opener}' in @{entry_type} entry starting at character {at}: "
                "the file ends before the entry does"
            )
        body = text[body_start:body_end]
        raw_text = text[at : body_end + 1]
        index = body_end + 1

        if entry_type in NON_PUBLICATION_TYPES and entry_type != "string":
            continue  # @comment/@preamble are not publications and never become records
        if entry_type == "string":
            name, value = _split_key_and_value(body)
            if name:
                resolved = _resolve_value(value, abbreviations)
                abbreviations[name.lower()] = resolved
            continue

        key, field_body = _split_comma(body)
        fields = _parse_fields(field_body, abbreviations)
        entries.append(
            BibEntry(
                entry_type=entry_type,
                key=key or f"entry{len(entries) + 1}",
                fields=fields,
                raw_text=raw_text,
                resolved_abbreviations=dict(abbreviations),
            )
        )
    return entries


def load_bib_file(path: Path) -> list[BibEntry]:
    """Read and parse one ``.bib`` file. Missing files raise, because silence would look like "empty"."""
    if not path.is_file():
        raise BibParseError(f"{path} does not exist or is not a file")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise BibParseError(f"cannot read {path}: {exc}") from exc
    return parse_bibtex(text)


def _find_closing(text: str, start: int, opener: str, closer: str) -> int:
    """Index of the ``closer`` that balances the ``opener`` just before ``start``, or ``-1``.

    A ``"`` only delimits a value when it immediately follows ``=``; everywhere else it is a
    literal character. That is what makes accent escapes such as ``J{\\"u}rgen`` parse correctly
    while ``title = "{GPT} is a big model"`` still protects its braces from being counted.
    """
    depth = 1
    in_quotes = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            if in_quotes:
                in_quotes = False
            elif _previous_nonspace(text, position) == "=":
                in_quotes = True
            continue
        if in_quotes:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return position
    return -1


def _previous_nonspace(text: str, position: int) -> str:
    """Last non-whitespace character before ``position`` (``""`` at the start of the text)."""
    index = position - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return text[index] if index >= 0 else ""


def _split_comma(body: str) -> tuple[str, str]:
    """Split ``key, rest`` at the first top-level comma. A trailing comma is not an error."""
    depth = 0
    in_quotes = False
    for position, char in enumerate(body):
        if char == '"':
            in_quotes = not in_quotes
        elif not in_quotes:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == "," and depth == 0:
                return body[:position].strip(), body[position + 1 :]
    return body.strip(), ""


def _split_key_and_value(body: str) -> tuple[str, str]:
    """Split ``name = value`` (used by ``@string`` and by every field)."""
    depth = 0
    in_quotes = False
    for position, char in enumerate(body):
        if char == '"':
            in_quotes = not in_quotes
        elif not in_quotes:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == "=" and depth == 0:
                return body[:position].strip().lower(), body[position + 1 :].strip()
    return body.strip().lower(), ""


def _parse_fields(body: str, abbreviations: dict[str, str]) -> dict[str, str]:
    """Parse ``name = value`` pairs, keeping values verbatim apart from outer-brace stripping."""
    fields: dict[str, str] = {}
    for name, raw_value in _iter_field_chunks(body):
        if not name:
            continue
        fields[name] = _resolve_value(raw_value, abbreviations)
    return fields


def _iter_field_chunks(body: str) -> Iterator[tuple[str, str]]:
    """Yield ``(field_name, raw_value_text)`` for each top-level field in an entry body."""
    position = 0
    length = len(body)
    while position < length:
        while position < length and (body[position].isspace() or body[position] == ","):
            position += 1
        if position >= length:
            return
        match = _FIELD_NAME_RE.match(body, position)
        if match is None:
            # Unrecognised junk: skip to the next comma so one bad field cannot desynchronise the rest.
            next_comma = body.find(",", position)
            if next_comma < 0:
                return
            position = next_comma + 1
            continue
        name = match.group(0).lower()
        cursor = match.end()
        while cursor < length and body[cursor].isspace():
            cursor += 1
        if cursor >= length or body[cursor] != "=":
            next_comma = body.find(",", cursor)
            if next_comma < 0:
                return
            position = next_comma + 1
            continue
        cursor += 1
        while cursor < length and body[cursor].isspace():
            cursor += 1
        value, cursor = _read_value(body, cursor)
        yield name, value
        position = cursor


def _read_value(body: str, start: int) -> tuple[str, int]:
    """Read one BibTeX value: ``{...}`` (nested), ``"..."``, or a bare token. Returns (text, next)."""
    length = len(body)
    if start >= length:
        return "", start
    char = body[start]
    if char == "{":
        end = _find_closing(body, start + 1, "{", "}")
        if end < 0:
            raise BibParseError(f"unbalanced braces in field value starting at {start}")
        return body[start + 1 : end], end + 1
    if char == '"':
        cursor = start + 1
        escaped = False
        out: list[str] = []
        while cursor < length:
            current = body[cursor]
            if escaped:
                out.append(current)
                escaped = False
            elif current == "\\":
                out.append(current)
                escaped = True
            elif current == '"':
                return "".join(out), cursor + 1
            else:
                out.append(current)
            cursor += 1
        raise BibParseError(f"unterminated quoted value starting at {start}")
    end = start
    depth = 0
    while end < length:
        current = body[end]
        if current == "{":
            depth += 1
        elif current == "}":
            depth -= 1
        elif current == "," and depth == 0:
            break
        end += 1
    return body[start:end].strip(), end


def _resolve_value(raw_value: str, abbreviations: dict[str, str]) -> str:
    """Resolve a bare identifier that names a ``@string`` abbreviation; otherwise keep text as-is.

    Whitespace is collapsed (BibTeX values wrap freely across lines) but braces and punctuation are
    preserved verbatim, because ``{GPT}-2`` and ``p.~3--7`` are content, not formatting noise.
    """
    text = raw_value.strip()
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]*", text) and text.lower() in abbreviations:
        return abbreviations[text.lower()]
    return _WS_RE.sub(" ", text)


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


def bib_entry_to_record(entry: BibEntry) -> ProviderRecord:
    """Map one :class:`BibEntry` onto a :class:`ProviderRecord`, keeping the entry text in ``raw``."""
    title = entry.get("title") or entry.key
    authors = split_bibtex_authors(entry.get("author")) if entry.get("author") else []
    if not authors and entry.get("editor"):
        authors = split_bibtex_authors(entry.get("editor"))
    venue = entry.get("journal") or entry.get("booktitle") or entry.get("publisher") or entry.get("school")
    month = entry.get("month")
    pages = entry.get("pages")
    year = parse_year(entry.get("year"))
    eprint = entry.get("eprint")
    archive = entry.get("archiveprefix")
    arxiv_id = None
    if eprint and ((archive or "").lower().startswith("arxiv") or _ARXIV_EPRINT_RE.match(eprint)):
        arxiv_id = eprint
    return build_record(
        ProviderKind.LOCAL_BIB,
        entry.key,
        title,
        abstract=entry.get("abstract"),
        authors=authors,
        year=year,
        venue=venue,
        doi=entry.get("doi"),
        arxiv_id=arxiv_id,
        url=entry.get("url"),
        license=entry.get("license"),
        raw={
            "source": "local_bibtex",
            "entry_type": entry.entry_type,
            "bibtex_key": entry.key,
            "fields": dict(entry.fields),
            "month": month,
            "month_number": parse_month(month),
            "pages": pages,
            "entry_text": entry.raw_text,
            "resolved_abbreviations": dict(entry.resolved_abbreviations),
        },
    )


class LocalBibProvider(LiteratureProvider):
    """Search the researcher's own ``.bib`` files. Never touches the network."""

    kind = ProviderKind.LOCAL_BIB
    requires_network = False

    def __init__(self, paths: Sequence[Path | str]) -> None:
        self.paths: list[Path] = [Path(p) for p in paths]
        self._entries: list[BibEntry] | None = None
        self._errors: dict[str, str] = {}

    # ------------------------------------------------------------------ loading

    @property
    def missing_paths(self) -> list[Path]:
        """Configured paths that do not exist yet. Surfaced rather than raised at construction."""
        return [p for p in self.paths if not p.is_file()]

    @property
    def name(self) -> str:
        existing = len(self.paths) - len(self.missing_paths)
        return f"local BibTeX ({existing}/{len(self.paths)} file(s) present)"

    def available(self) -> bool:
        """Available when at least one configured file exists — existence, verified without parsing."""
        return any(p.is_file() for p in self.paths)

    def entries(self) -> list[BibEntry]:
        """Parse (once) and return every entry.

        A parse failure raises rather than being skipped: a bibliography we cannot read would
        silently shrink the search space, and a shrunken search space is how "no prior work" becomes
        false. Per-path problems that are *not* parse failures (missing files) are recorded in
        :meth:`parse_errors` and reported by :attr:`missing_paths`.
        """
        if self._entries is None:
            parsed: list[BibEntry] = []
            for path in self.paths:
                if not path.is_file():
                    self._errors[str(path)] = "missing"
                    continue
                try:
                    parsed.extend(load_bib_file(path))
                except BibParseError as exc:
                    self._errors[str(path)] = str(exc)
                    raise
            self._entries = parsed
        return self._entries

    def parse_errors(self) -> dict[str, str]:
        """Per-path problems from the last load: a bibliography we could not read is a coverage gap."""
        return dict(self._errors)

    def records(self) -> list[ProviderRecord]:
        return [bib_entry_to_record(entry) for entry in self.entries()]

    # ------------------------------------------------------------------ API

    def search(self, query: str, *, limit: int = 20) -> list[ProviderRecord]:
        self._require_available()
        tokens = tokenise(query)
        scored: list[tuple[float, str, ProviderRecord]] = []
        for record in self.records():
            haystack = " ".join(
                part
                for part in (record.title, record.abstract or "", record.venue or "", " ".join(record.authors))
                if part
            )
            score = jaccard(tokens, tokenise(haystack)) if tokens else 0.0
            if tokens and score == 0.0:
                continue  # a bibliography is searched, not dumped: unmatched entries are not hits
            scored.append((score, record.provider_id, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        out: list[ProviderRecord] = []
        for score, _, record in scored[: max(0, limit)]:
            out.append(record.model_copy(update={"query": query, "score": round(score, 4)}))
        return out

    def fetch(self, identifier: str) -> ProviderRecord | None:
        if not self.available():
            return None
        wanted = identifier.strip().lower()
        normalised = normalise_title(identifier)
        for record in self.records():
            if wanted in {
                record.provider_id.lower(),
                (record.doi or "").lower(),
                (record.arxiv_id or "").lower(),
                (record.url or "").lower(),
            }:
                return record
            if normalised and normalise_title(record.title) == normalised:
                return record
        return None

    def expand(self, record: ProviderRecord) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
        """Resolve ``crossref``/``related`` style fields against the same bibliography when present."""
        keys: list[str] = []
        source = record.raw.get("fields") if isinstance(record.raw.get("fields"), dict) else {}
        for name in ("crossref", "related", "xdata"):
            value = str(source.get(name, "") or "")
            keys.extend(part.strip() for part in re.split(r"[,\s]+", value) if part.strip())
        resolved = [found for found in (self.fetch(key) for key in keys) if found is not None]
        return resolved, []


__all__ = [
    "NON_PUBLICATION_TYPES",
    "BibEntry",
    "BibParseError",
    "LocalBibProvider",
    "bib_entry_to_record",
    "load_bib_file",
    "parse_bibtex",
]
