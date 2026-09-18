"""Extractors for human-written material: READMEs, notes, diaries, audits, drafts, LaTeX.

Everything here is a deterministic rule over text. There is no summarisation and no model: a
sentence becomes a claim candidate only if it matches an explicit assertion pattern, and each
candidate keeps its file, line number and quote so a human can check it in one click.
"""

from __future__ import annotations

import re

from ..models.common import EvidenceLevel, SourceKind
from .facts import (
    ANOMALY_PATTERNS,
    CLAIM_PATTERNS,
    DECISION_PATTERNS,
    FAILURE_PATTERNS,
    Fact,
    FactKind,
    ExtractionOutput,
    OPEN_QUESTION_PATTERNS,
    REJECTION_PATTERNS,
    SourceFile,
    TODO_PATTERNS,
    extract_numbers,
    extract_p_values,
    extract_pm_pairs,
    implied_level,
    iso_or_none,
    line_number,
    matches_any,
    normalise_text,
    parse_markdown_table,
    sentences,
)

#: Heading keywords -> the fact kind that section produces.
SECTION_ROUTES: tuple[tuple[tuple[str, ...], FactKind], ...] = (
    (("research question", "core question", "problem statement", "aim", "objective", "goal"), FactKind.CORE_QUESTION),
    (("secondary question", "sub-question", "subquestion"), FactKind.SECONDARY_QUESTION),
    (("rejected", "abandoned", "negative result", "what did not work", "ruled out", "dead end"), FactKind.REJECTED_CLAIM),
    (("open question", "open questions", "future work", "next steps", "unresolved"), FactKind.OPEN_QUESTION),
    (("decision", "decisions", "design choice", "rationale"), FactKind.DECISION),
    (("claim", "claims", "findings", "contributions", "conclusions", "headline"), FactKind.CLAIM),
    (("result", "results", "metrics", "numbers", "experiments"), FactKind.CLAIM),
    (("limitation", "limitations", "caveat", "threats to validity"), FactKind.NOTE),
    (("note", "notes", "diary", "log"), FactKind.NOTE),
)

#: Marks a claim as *not* an assertion but an intention.
HYPOTHESIS_MARKERS: tuple[str, ...] = (
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bhypothes", r"\bspeculat", r"\bwe suspect",
    r"\bwe believe", r"\bpossibly\b", r"\bperhaps\b", r"\bwe expect",
)

_MD_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
_TEX_SECTION = re.compile(r"\\(?P<kind>section|subsection|subsubsection|chapter)\*?\{(?P<title>[^}]*)\}")
_TEX_COMMENT = re.compile(r"(?<!\\)%.*$", flags=re.MULTILINE)
_TEX_CITE = re.compile(r"\\(?:cite|citep|citet|parencite|textcite|autocite)\*?(?:\[[^\]]*\])*\{([^}]+)\}")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<body>.+?)\s*$")

#: Table columns that are bookkeeping, not measurements. Extracting numbers from them produces
#: exactly the kind of bogus "ungrounded number" noise that makes an importer useless.
NON_METRIC_COLUMNS: frozenset[str] = frozenset(
    {
        "arm", "arms", "seed", "seeds", "run", "runs", "status", "id", "index", "row", "n",
        "params", "parameters", "notes", "note", "method", "model", "dataset", "scale", "steps",
        "time", "date", "commit", "count", "gpu", "gpus", "epoch", "epochs", "budget",
    }
)


def strip_latex(text: str) -> str:
    """Crude but predictable LaTeX flattening: enough to read prose, not a TeX implementation."""
    text = _TEX_COMMENT.sub("", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ").replace("\\\\", "\n")
    return text


def document_sections(file: SourceFile) -> list[tuple[str, str]]:
    """Split a document into ``(heading, body)`` pairs, marking the preamble as its own section."""
    text = file.text or ""
    if file.kind.name in {"LATEX", "PAPER_DRAFT"} or file.rel_path.endswith(".tex"):
        pattern = _TEX_SECTION
        lines_for_body = None
    else:
        pattern = _MD_HEADING
        lines_for_body = text.splitlines()

    sections: list[tuple[str, str]] = []
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    if matches[0].start() > 0:
        sections.append(("<preamble>", text[: matches[0].start()]))
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((title, text[start:end]))
    if lines_for_body is not None:
        # markdown tables survive heading splits fine; nothing extra to do
        pass
    return sections


def route_heading(heading: str) -> FactKind | None:
    """Route a heading to a fact kind using *word-boundary* keyword matching.

    Substring matching is a trap here: "Current cl**aim**s" contains "aim", and a naive
    implementation files every claims section as the core research question.
    """
    lowered = heading.lower()
    for keywords, kind in SECTION_ROUTES:
        for keyword in keywords:
            if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", lowered):
                return kind
    return None


def _fact(
    file: SourceFile,
    *,
    kind: FactKind,
    statement: str,
    rule: str,
    confidence: float,
    locator: str | None = None,
    payload: dict | None = None,
    level: EvidenceLevel | None = None,
) -> Fact:
    return Fact(
        kind=kind,
        statement=statement.strip(),
        rule=rule,
        source=file.source_ref(locator=locator, quote=statement.strip(), kind=_source_kind(file)),
        confidence=confidence,
        payload=payload or {},
        implied_level=level,
        files=[file.rel_path],
    )


def _source_kind(file: SourceFile) -> SourceKind:
    if file.kind.value in {"PAPER_DRAFT", "LATEX"}:
        return SourceKind.PAPER_PROSE
    if file.kind.value == "AUDIT_REPORT":
        return SourceKind.AUDIT
    return SourceKind.AUTHOR_NOTE


def extract_document(file: SourceFile) -> ExtractionOutput:
    """Extract question/claim/rejection/decision/question/note facts from one document."""
    out = ExtractionOutput()
    text = file.text or ""
    if not text.strip():
        out.unparsed.append((file.rel_path, "empty file"))
        return out

    is_tex = file.rel_path.endswith((".tex", ".bbl"))
    prose = strip_latex(text) if is_tex else text
    seen: set[str] = set()

    sections = document_sections(file) if not is_tex else _tex_sections(text)
    for heading, body in sections:
        body_prose = strip_latex(body) if is_tex else body
        routed = route_heading(heading)
        if routed is None:
            routed = FactKind.NOTE if file.kind.value in {"PLAIN_TEXT", "AUDIT_REPORT"} else None

        items = _section_items(body_prose)
        for item in items:
            if _is_table_fragment(item):
                continue  # the table extractor owns these rows
            line = _find_line(prose, item)
            locator = f"line {line}" if line else None
            lowered = item.lower()

            if routed is FactKind.CORE_QUESTION and len(item.split()) > 4:
                _add(out, seen, _fact(file, kind=FactKind.CORE_QUESTION, statement=_clean(item),
                                      rule="section:question", confidence=0.72, locator=locator,
                                      level=EvidenceLevel.L0_IDEA))
                continue
            if routed is FactKind.SECONDARY_QUESTION and len(item.split()) > 4:
                _add(out, seen, _fact(file, kind=FactKind.SECONDARY_QUESTION, statement=_clean(item),
                                      rule="section:secondary_question", confidence=0.6, locator=locator,
                                      level=EvidenceLevel.L0_IDEA))
                continue
            if routed is FactKind.REJECTED_CLAIM:
                _add(out, seen, _fact(file, kind=FactKind.REJECTED_CLAIM, statement=_clean(item),
                                      rule="section:rejected", confidence=0.75, locator=locator,
                                      payload={"reason": _rejection_reason(item)},
                                      level=EvidenceLevel.L0_IDEA))
                continue
            if routed is FactKind.OPEN_QUESTION:
                _add(out, seen, _fact(file, kind=FactKind.OPEN_QUESTION, statement=_clean(item),
                                      rule="section:open_question", confidence=0.6, locator=locator,
                                      level=EvidenceLevel.L0_IDEA))
                continue
            if routed is FactKind.DECISION and len(item.split()) > 4:
                _add(out, seen, _fact(file, kind=FactKind.DECISION, statement=_clean(item),
                                      rule="section:decision", confidence=0.6, locator=locator,
                                      level=EvidenceLevel.L0_IDEA))
                continue
            if routed is FactKind.CLAIM:
                # A bullet (or sentence) inside a claims/results section *is* a claim candidate;
                # the language level is recorded so the calibrator can weaken it later.
                level = implied_level(item)
                _add(out, seen, _fact(file, kind=FactKind.CLAIM, statement=_clean(item),
                                      rule="section:claim", confidence=0.65, locator=locator,
                                      payload={"language_level": level.value},
                                      level=level))
                continue
            if routed is FactKind.NOTE and len(item.split()) > 3:
                _add(out, seen, _fact(file, kind=FactKind.NOTE, statement=_clean(item),
                                      rule="section:note", confidence=0.5, locator=locator))
                continue

            # No route: fall through to pattern-based classification on every sentence.
            _classify_sentence(out, seen, file, item, locator, source_kind=_source_kind(file))

    # Pattern scan over the whole document regardless of routing (catches inline remarks).
    # Table rows are blanked out (same length, so offsets and line numbers stay valid) because the
    # table extractor already handles them with correct column attribution.
    plain = _blank_tables(prose)
    for sentence in sentences(plain):
        line = _find_line(prose, sentence)
        _classify_sentence(
            out, seen, file, sentence, f"line {line}" if line else None, source_kind=_source_kind(file)
        )

    _extract_numbers(out, seen, file, plain, source_kind=_source_kind(file))
    _extract_tables(out, seen, file, prose)
    _extract_todos(out, seen, file, plain)
    if is_tex:
        _extract_citations(out, seen, file, text)
    return out


def _tex_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_TEX_SECTION.finditer(text))
    if not matches:
        return [("", strip_latex(text))]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        body = text[match.end() : (matches[index + 1].start() if index + 1 < len(matches) else len(text))]
        sections.append((match.group("title"), strip_latex(body)))
    return sections


def _section_items(body: str) -> list[str]:
    """Bullets if the section uses them, otherwise sentences.

    Struck-through bullets are *kept*: in a "Rejected claims" section a struck-through line is
    precisely the rejected claim we must reconstruct (and the tombstone must survive import).
    """
    bullets = [_BULLET.match(line).group("body") for line in body.splitlines() if _BULLET.match(line)]
    long_enough = [b for b in bullets if len(b.split()) >= 3]
    if len(long_enough) >= 2:
        return [_clean(b) for b in long_enough]
    return [s for s in sentences(body) if len(s.split()) >= 4]


def _clean(text: str) -> str:
    cleaned = text.strip().strip("*_`#>")
    cleaned = re.sub(r"~~(.*?)~~", r"\1", cleaned)
    cleaned = cleaned.replace("*", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .") + "." if cleaned and not cleaned.endswith((".", "?", "!")) else cleaned


def _rejection_reason(text: str) -> str:
    for pattern, code in REJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return f"matched rejection pattern {code!r}"
    return "listed in a rejected/abandoned section"


def _find_line(text: str, needle: str) -> int | None:
    probe = needle.strip()[:60]
    if not probe:
        return None
    index = text.find(probe)
    return line_number(text, index) if index >= 0 else None


def _add(out: ExtractionOutput, seen: set[str], fact: Fact) -> None:
    key = fact.key()
    if key in seen:
        return
    seen.add(key)
    out.facts.append(fact)


def _classify_sentence(
    out: ExtractionOutput,
    seen: set[str],
    file: SourceFile,
    sentence: str,
    locator: str | None,
    *,
    source_kind: SourceKind,
) -> None:
    text = sentence.strip()
    if len(text.split()) < 4 or len(text) > 600:
        return
    lowered = text.lower()

    rejection = matches_any(lowered, [pattern for pattern, _ in REJECTION_PATTERNS])
    if rejection and matches_any(lowered, [pattern for pattern, _, _ in CLAIM_PATTERNS]):
        _add(out, seen, _fact(file, kind=FactKind.REJECTED_CLAIM, statement=_clean(text),
                              rule=f"pattern:reject:{rejection}", confidence=0.7, locator=locator,
                              payload={"reason": f"language pattern {rejection!r}"},
                              level=EvidenceLevel.L0_IDEA))
        return

    decision = matches_any(lowered, [pattern for pattern, _ in DECISION_PATTERNS])
    if decision and len(text.split()) > 5:
        _add(out, seen, _fact(file, kind=FactKind.DECISION, statement=_clean(text),
                              rule=f"pattern:decision:{decision}", confidence=0.6, locator=locator))
        return

    anomaly = matches_any(lowered, ANOMALY_PATTERNS)
    if anomaly:
        _add(out, seen, _fact(file, kind=FactKind.ANOMALY, statement=_clean(text),
                              rule=f"pattern:anomaly:{anomaly}", confidence=0.55, locator=locator))

    failure = matches_any(lowered, FAILURE_PATTERNS)
    if failure:
        _add(out, seen, _fact(file, kind=FactKind.FAILED_RUN, statement=_clean(text),
                              rule=f"pattern:failure:{failure}", confidence=0.6, locator=locator))

    question = matches_any(lowered, OPEN_QUESTION_PATTERNS)
    if question:
        _add(out, seen, _fact(file, kind=FactKind.OPEN_QUESTION, statement=_clean(text),
                              rule=f"pattern:open_question:{question}", confidence=0.55, locator=locator,
                              level=EvidenceLevel.L0_IDEA))

    for pattern, code, level in CLAIM_PATTERNS:
        if re.search(pattern, lowered):
            hypothesised = bool(matches_any(lowered, HYPOTHESIS_MARKERS))
            effective = EvidenceLevel.L0_IDEA if hypothesised and level.rank > 1 else level
            if source_kind is SourceKind.PAPER_PROSE:
                # Prose in a draft is a *report of* a claim, not the claim itself.
                confidence = 0.5
            else:
                confidence = 0.7
            _add(out, seen, _fact(file, kind=FactKind.CLAIM, statement=_clean(text),
                                  rule=f"pattern:claim:{code}", confidence=confidence, locator=locator,
                                  payload={"language_level": effective.value, "hypothesised": hypothesised},
                                  level=effective))
            return

    _add(out, seen, _fact(file, kind=FactKind.NOTE, statement=_clean(text),
                          rule="sentence:unclassified", confidence=0.3, locator=locator))


def _extract_numbers(
    out: ExtractionOutput,
    seen: set[str],
    file: SourceFile,
    prose: str,
    *,
    source_kind: SourceKind,
) -> None:
    for hit in [*extract_numbers(prose), *extract_p_values(prose), *extract_pm_pairs(prose)]:
        payload = {
            "value": hit.value,
            "unit": hit.unit,
            "metric": hit.metric,
            "raw": hit.raw,
            "context": hit.context,
            "source_kind": source_kind.value,
        }
        _add(
            out,
            seen,
            _fact(
                file,
                kind=FactKind.PROSE_NUMBER,
                statement=f"{hit.raw} — {hit.context}",
                rule="number:prose" if hit.metric else "number:unattributed",
                confidence=0.6 if hit.metric else 0.35,
                locator=f"line {line_number(prose, hit.offset)}",
                payload=payload,
            ),
        )


def _blank_tables(text: str) -> str:
    """Replace markdown table lines with same-length blanks, preserving offsets and line numbers."""
    return re.sub(
        r"^[ \t]*\|.*\|[ \t]*$",
        lambda m: " " * len(m.group(0)),
        text,
        flags=re.MULTILINE,
    )


def _is_table_fragment(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("|") or stripped.count("|") >= 2


def _extract_tables(out: ExtractionOutput, seen: set[str], file: SourceFile, prose: str) -> None:
    for header, rows in parse_markdown_table(prose):
        for row in rows:
            for column, cell in zip(header, row):
                if column.strip().lower() in NON_METRIC_COLUMNS:
                    continue
                for hit in extract_numbers(cell, metric_names=[column]):
                    _add(
                        out,
                        seen,
                        _fact(
                            file,
                            kind=FactKind.PROSE_NUMBER,
                            statement=f"{column} = {hit.raw} ({row[0] if row else ''})",
                            rule="number:table",
                            confidence=0.65,
                            locator=f"table column {column!r}",
                            payload={
                                "value": hit.value,
                                # The column header *is* the metric; a table cell has no other
                                # attribution, and without it the number cannot be compared to data.
                                "metric": hit.metric or column,
                                "unit": hit.unit,
                                "raw": hit.raw,
                                "labels": {header[0]: row[0]} if row else {},
                                "source_kind": SourceKind.AUTHOR_NOTE.value,
                                "table": True,
                            },
                        ),
                    )


def _extract_todos(out: ExtractionOutput, seen: set[str], file: SourceFile, prose: str) -> None:
    for line in prose.splitlines():
        hit = matches_any(line, TODO_PATTERNS)
        if hit and len(line.split()) > 3:
            _add(out, seen, _fact(file, kind=FactKind.SKILL_GAP, statement=_clean(line),
                                  rule=f"pattern:todo:{hit}", confidence=0.45,
                                  locator=f"line {line_number(prose, prose.find(line))}"))


def _extract_citations(out: ExtractionOutput, seen: set[str], file: SourceFile, tex: str) -> None:
    for match in _TEX_CITE.finditer(tex):
        for key in match.group(1).split(","):
            key = key.strip()
            if not key:
                continue
            _add(
                out,
                seen,
                _fact(
                    file,
                    kind=FactKind.LITERATURE,
                    statement=f"cited: {key}",
                    rule="latex:cite",
                    confidence=0.8,
                    locator=f"line {line_number(tex, match.start())}",
                    payload={"bibtex_key": key, "cited": True},
                ),
            )
