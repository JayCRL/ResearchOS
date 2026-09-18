"""Import facts: the intermediate representation between raw material and research state.

Research Import does not "summarise a repo into a paper". It walks the material, extracts
*attributable facts* (each one carrying a file, a line and a quote), and only then reconstructs
research state — with everything uncertain deferred to a human review queue.

The intermediate representation exists so that every reconstructed object can answer
"which bytes did this come from?" and so that contradicting sources can be detected *before*
they are silently merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..models.common import (
    ClaimStatus,
    EvidenceLevel,
    SourceKind,
    SourceRef,
    StrEnum,
    sha256_text,
)


class FileKind(StrEnum):
    """How a file was classified. Classification drives which extractors run."""

    README = "README"
    MARKDOWN = "MARKDOWN"
    LATEX = "LATEX"
    PAPER_DRAFT = "PAPER_DRAFT"
    AUDIT_REPORT = "AUDIT_REPORT"
    PLAIN_TEXT = "PLAIN_TEXT"
    PDF = "PDF"
    NOTEBOOK = "NOTEBOOK"
    PYTHON = "PYTHON"
    CODE_OTHER = "CODE_OTHER"
    CONFIG = "CONFIG"
    DATA_CSV = "DATA_CSV"
    DATA_JSON = "DATA_JSON"
    DATA_OTHER = "DATA_OTHER"
    LOG = "LOG"
    BIB = "BIB"
    FIGURE = "FIGURE"
    BINARY_UNSUPPORTED = "BINARY_UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


#: Extensions we can read as text.
TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md", ".markdown", ".rst", ".txt", ".tex", ".latex", ".bib", ".bbl",
        ".py", ".ipynb", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".csv", ".tsv", ".log", ".out", ".sh", ".bash", ".ps1", ".r", ".jl", ".m",
    }
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pdf_"})


class FactKind(StrEnum):
    CORE_QUESTION = "CORE_QUESTION"
    SECONDARY_QUESTION = "SECONDARY_QUESTION"
    CLAIM = "CLAIM"
    REJECTED_CLAIM = "REJECTED_CLAIM"
    OPEN_QUESTION = "OPEN_QUESTION"
    EXPERIMENT = "EXPERIMENT"
    CONFIG = "CONFIG"
    METRIC_VALUE = "METRIC_VALUE"
    PROSE_NUMBER = "PROSE_NUMBER"
    EVIDENCE = "EVIDENCE"
    DECISION = "DECISION"
    ANOMALY = "ANOMALY"
    FAILED_RUN = "FAILED_RUN"
    LITERATURE = "LITERATURE"
    NOTE = "NOTE"
    CODE_SURFACE = "CODE_SURFACE"
    GIT_COMMIT = "GIT_COMMIT"
    SKILL_GAP = "SKILL_GAP"
    UNPARSED = "UNPARSED"


@dataclass
class SourceFile:
    """One scanned file, with the digest that pins the interpretation to these exact bytes."""

    rel_path: str
    abs_path: Path
    kind: FileKind
    size: int
    sha256: str
    text: str | None = None
    classification_reason: str = ""
    error: str | None = None

    @property
    def name(self) -> str:
        return Path(self.rel_path).name

    @property
    def directory(self) -> str:
        parent = str(Path(self.rel_path).parent)
        return "" if parent == "." else parent

    def lines(self) -> list[str]:
        return (self.text or "").splitlines()

    def line_of(self, needle: str) -> int | None:
        for index, line in enumerate(self.lines(), start=1):
            if needle in line:
                return index
        return None

    def source_ref(self, *, locator: str | None = None, quote: str | None = None,
                   kind: SourceKind = SourceKind.AUTHOR_NOTE) -> SourceRef:
        return SourceRef(
            path=self.rel_path,
            sha256=self.sha256,
            locator=locator,
            quote=(quote[:1200] if quote else None),
            kind=kind,
        )


@dataclass
class Fact:
    """One attributable statement recovered from the material."""

    kind: FactKind
    statement: str
    rule: str
    source: SourceRef
    confidence: float = 0.6
    payload: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = True
    implied_level: EvidenceLevel | None = None
    files: list[str] = field(default_factory=list)

    def key(self) -> str:
        """Deduplication key: same kind + same normalised statement."""
        return f"{self.kind.value}:{sha256_text(normalise_text(self.statement))[:16]}"

    def numeric(self) -> float | None:
        value = self.payload.get("value")
        return float(value) if isinstance(value, (int, float)) else None

    def metric(self) -> str | None:
        metric = self.payload.get("metric")
        return str(metric) if metric else None


@dataclass
class ExtractionOutput:
    facts: list[Fact] = field(default_factory=list)
    unparsed: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    notes: list[str] = field(default_factory=list)

    def extend(self, other: "ExtractionOutput") -> "ExtractionOutput":
        self.facts.extend(other.facts)
        self.unparsed.extend(other.unparsed)
        self.notes.extend(other.notes)
        return self


# --------------------------------------------------------------------------------------
# Text helpers (deterministic, no NLP model)
# --------------------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\\$])")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_TERMINATED = re.compile(r"[.!?:;]\s*$")

#: Sentence patterns that indicate an assertion the authors are making.
CLAIM_PATTERNS: tuple[tuple[str, str, EvidenceLevel], ...] = (
    (r"\bwe (?:show|find|found|observe|observed|demonstrate|report)\b", "we_show", EvidenceLevel.L2_REPRODUCED),
    (r"\bwe (?:hypothesis|hypothesi[sz]e|speculate|suspect|believe)\b", "we_hypothesise", EvidenceLevel.L0_IDEA),
    (r"\bwe (?:propose|introduce|present)\b", "we_propose", EvidenceLevel.L0_IDEA),
    (r"\b(?:this|these|our) (?:results?|experiments?|findings?) (?:show|shows|indicate|indicates|suggest|suggests)\b", "results_indicate", EvidenceLevel.L2_REPRODUCED),
    (r"\b(?:we conclude|in conclusion|our conclusion)\b", "conclusion", EvidenceLevel.L3_CONTROLLED),
    (r"\b(?:establishes?|proves?|confirms?|demonstrates?) (?:the|that)\b", "strong_claim", EvidenceLevel.L4_INTERVENTION),
    (r"\b(?:is|are) (?:necessary|sufficient) for\b", "necessity_sufficiency", EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY),
    (r"\b(?:causes?|caused|drives?|driven by|leads? to|due to|because of|responsible for)\b", "causal", EvidenceLevel.L4_INTERVENTION),
    (r"\b(?:correlates? with|correlated with|associated with|co-occurs with)\b", "correlational", EvidenceLevel.L1_OBSERVATION),
    (r"\b(?:generalises|generalizes|across (?:models|scales|settings|datasets))\b", "generalisation", EvidenceLevel.L6_CROSS_SETTING_REPLICATION),
)

REJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:we )?(?:rejected|reject|ruled out|rule out|falsified|disproved|abandoned|dropped|gave up on)\b", "rejected"),
    (r"\b(?:did not|does not|doesn't|didn't) (?:hold|work|replicate|survive|generalise|generalize)\b", "did_not_hold"),
    (r"\b(?:no longer|not comparable|should not be cited|stale|invalid)\b", "withdrawn"),
)

DECISION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bwe (?:decided|chose|opted|settled on|agree[ds]? to)\b", "explicit_decision"),
    (r"\bwe (?:switched|moved|changed) (?:to|from)\b", "route_change"),
    (r"\b(?:instead|alternative(?:ly)?),? we\b", "alternative_chosen"),
    (r"\bwe (?:dropped|cut|abandoned) the\b", "scope_reduction"),
)

ANOMALY_PATTERNS: tuple[str, ...] = (
    r"\banomal(?:y|ies|ous)\b",
    r"\bunexpected(?:ly)?\b",
    r"\bsurpris(?:e|ed|ing)\b",
    r"\bweird\b",
    r"\bstrange\b",
    r"\bcontradicts?\b",
    r"\bvanished\b",
    r"\bcollapsed\b",
)

OPEN_QUESTION_PATTERNS: tuple[str, ...] = (
    r"\bopen question\b",
    r"\bunclear whether\b",
    r"\bremains? (?:to be|an open|unclear)\b",
    r"\bwe do not know\b",
    r"\bno clean experiment\b",
    r"\btbd\b",
    r"\bunknown whether\b",
)

FAILURE_PATTERNS: tuple[str, ...] = (
    r"\bfailed\b",
    r"\boom\b",
    r"\bcrash(?:ed)?\b",
    r"\bnan\b",
    r"\bdiverged\b",
    r"\baborted\b",
    r"\btimed out\b",
)

TODO_PATTERNS: tuple[str, ...] = (
    r"\btodo\b",
    r"\bfixme\b",
    r"\bneeds? to be (?:implemented|added|automated|checked)\b",
    r"\bmanually\b",
    r"\bwe had to (?:write|build|hand-?code)\b",
)

#: Named metrics we recognise in prose. Data files contribute their own column names too.
METRIC_LEXICON: tuple[str, ...] = (
    "retention_at_1", "retention_at_4", "retention", "accuracy", "acc", "loss", "perplexity",
    "ppl", "f1", "precision", "recall", "auc", "bleu", "rouge", "exact_match", "param_updates",
    "flops", "wall_clock", "energy", "throughput", "latency", "p_value", "p-value",
)

_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_@.])"
    r"(?P<sign>[-+])?"
    r"(?P<value>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>%|percent|pts?|points?|x|×)?"
    r"(?![A-Za-z_])"
)
_P_VALUE = re.compile(r"\bp\s*[=<>]\s*(?P<p>0?\.\d+(?:e-?\d+)?|\d+(?:\.\d+)?e-?\d+)")
_PLUSMINUS = re.compile(r"(?P<mean>\d+(?:\.\d+)?)\s*(?:±|\+/-)\s*(?P<pm>\d+(?:\.\d+)?)")
#: Dates, versions, identifiers and ranges are full of numerals that are not measurements.
_DATE_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\bv?\d+\.\d+\.\d+\b|\b[0-9a-f]{7,40}\b")
_RANGE_LIKE = re.compile(r"\d+\s*(?:--|–|—|,|to)\s*\d+|\d+\s*-\s*\d+")


def normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, drop markdown emphasis — for dedup and matching."""
    cleaned = re.sub(r"[*_`>#]", " ", text)
    return _WS.sub(" ", cleaned).strip().lower()


def _reflow(paragraph: str) -> list[str]:
    """Group a paragraph into logical units: one per bullet, one per wrapped prose sentence.

    A source line break is *not* a sentence boundary (prose wraps), but a bullet marker is, and a
    terminated line is. Treating wrapped lines as separate units would shred the sentences we most
    need to recover — starting with the research question itself.
    """
    units: list[str] = []
    current = ""
    for raw in paragraph.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if _LIST_ITEM.match(line):
            if current.strip():
                units.append(current.strip())
            current = _LIST_ITEM.sub("", line)
            continue
        if current and not _TERMINATED.search(current):
            current = f"{current} {line.strip()}"
        else:
            if current.strip():
                units.append(current.strip())
            current = line
    if current.strip():
        units.append(current.strip())
    return units


def sentences(text: str) -> list[str]:
    """Split into sentences, never across a paragraph break, heading or list item.

    Structural boundaries matter more than punctuation here: a "sentence" that swallows a heading
    and the following section produces nonsense claims, which is exactly how importers earn a bad
    reputation.
    """
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        cleaned = re.sub(r"^[ \t]*#{1,6}[ \t]+.*$", "", paragraph, flags=re.MULTILINE)
        cleaned = re.sub(r"^[ \t]*\\[a-zA-Z]+\*?\{.*$", "", cleaned, flags=re.MULTILINE)
        for unit in _reflow(cleaned):
            for chunk in _SENTENCE_SPLIT.split(unit):
                stripped = chunk.strip()
                if stripped:
                    out.append(stripped)
    return out


def windows(text: str, pattern: str, *, context: int = 120) -> list[tuple[str, int]]:
    """Return (sentence-ish window, character offset) for each regex hit."""
    out: list[tuple[str, int]] = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = max(0, match.start() - context)
        end = min(len(text), match.end() + context)
        out.append((text[start:end].replace("\n", " ").strip(), match.start()))
    return out


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def implied_level(text: str) -> EvidenceLevel:
    """The strongest evidence level the *language* claims (used to compare against what exists)."""
    best = EvidenceLevel.L1_OBSERVATION
    for pattern, _code, level in CLAIM_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            if level.rank > best.rank:
                best = level
    return best


def matches_any(text: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


@dataclass(frozen=True)
class NumberHit:
    value: float
    raw: str
    offset: int
    unit: str | None = None
    metric: str | None = None
    context: str = ""


def extract_numbers(text: str, *, metric_names: Sequence[str] = ()) -> list[NumberHit]:
    """Extract numerals with a best-effort metric attribution from the nearby window.

    Attribution is deliberately conservative: a numeral is only tied to a metric when a known
    metric token appears within 120 characters before it. Unattributed numerals are still
    recorded (as PROSE_NUMBER facts) because they are exactly what a paper must be able to trace.
    """
    lexicon = [m.lower() for m in (*METRIC_LEXICON, *metric_names)]
    hits: list[NumberHit] = []
    date_spans = [(m.start(), m.end()) for m in _DATE_LIKE.finditer(text)]
    range_spans = [(m.start(), m.end()) for m in _RANGE_LIKE.finditer(text)]
    for match in _NUMBER.finditer(text):
        if any(start <= match.start() < end for start, end in date_spans):
            continue  # a date/version/hash component, not a measurement
        if any(start <= match.start() and match.end() <= end for start, end in range_spans):
            continue  # a range endpoint ("seeds 0--2", "1-4 windows"), not a measurement
        raw = match.group(0).strip()
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            continue
        if match.group("sign") == "-":
            value = -value
        prefix = text[max(0, match.start() - 120) : match.start()].lower()
        metric = None
        best_pos = -1
        for name in lexicon:
            pos = prefix.rfind(name)
            if pos > best_pos:
                best_pos = pos
                metric = name
        hits.append(
            NumberHit(
                value=value,
                raw=raw,
                offset=match.start(),
                unit=match.group("unit"),
                metric=metric,
                context=text[max(0, match.start() - 80) : match.end() + 80].replace("\n", " ").strip(),
            )
        )
    return hits


def extract_p_values(text: str) -> list[NumberHit]:
    hits: list[NumberHit] = []
    for match in _P_VALUE.finditer(text):
        try:
            value = float(match.group("p"))
        except ValueError:
            continue
        hits.append(
            NumberHit(
                value=value,
                raw=match.group(0),
                offset=match.start(),
                metric="p_value",
                context=text[max(0, match.start() - 80) : match.end() + 80].replace("\n", " ").strip(),
            )
        )
    return hits


def extract_pm_pairs(text: str) -> list[NumberHit]:
    hits: list[NumberHit] = []
    for match in _PLUSMINUS.finditer(text):
        try:
            value = float(match.group("mean"))
        except ValueError:
            continue
        hits.append(
            NumberHit(
                value=value,
                raw=match.group(0),
                offset=match.start(),
                metric=None,
                context=text[max(0, match.start() - 80) : match.end() + 80].replace("\n", " ").strip(),
            )
        )
    return hits


def parse_markdown_table(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Parse markdown/pipe tables into (header, rows) pairs. Used for prior numbers and metrics."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells) and cells:
                continue
            current.append(cells)
        else:
            if len(current) >= 2:
                tables.append((current[0], current[1:]))
            current = []
    if len(current) >= 2:
        tables.append((current[0], current[1:]))
    return tables


def iso_or_none(text: str) -> datetime | None:
    """Extract a leading ISO-ish date from a text blob (diary headings)."""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


class StrengthHint(Enum):
    """Not a model: a small enum telling the reconstructor how to phrase a recovered claim."""

    ASSERTION = "ASSERTION"
    HYPOTHESIS = "HYPOTHESIS"
    OBSERVATION = "OBSERVATION"


def claim_status_ceiling(fact: Fact) -> ClaimStatus:
    """The **highest** claim status an imported fact may ever receive.

    Import never certifies anything: at most it records a hypothesis (or an observation-derived
    hypothesis). Promotion to TESTED/SUPPORTED/ROBUST happens later, through the claim lifecycle,
    with evidence and a human in the loop.
    """
    return ClaimStatus.HYPOTHESIS
