"""Extractors for machine-made material: code, configs, data, logs, bibliographies, PDFs, git.

These are the sources that let Research Import reconstruct *real* experiments rather than a
retelling: an argparse surface gives the variables, a config file gives the conditions, a metrics
CSV gives the results, and the git history gives the timeline. All deterministic, all hashed.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import statistics as pystats
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..models.common import EvidenceLevel, SourceKind
from .facts import (
    FAILURE_PATTERNS,
    ExtractionOutput,
    Fact,
    FactKind,
    FileKind,
    SourceFile,
    extract_numbers,
    matches_any,
)

# --------------------------------------------------------------------------------------
# Python source
# --------------------------------------------------------------------------------------

#: argparse names that mean "this is an experiment definition".
EXPERIMENT_ARG_NAMES: frozenset[str] = frozenset(
    {
        "seed", "seeds", "learning_rate", "lr", "steps", "epochs", "batch_size", "model",
        "dataset", "data", "optimizer", "arm", "variant", "sleep_steps", "scale", "config",
        "budget", "tokens", "temperature", "layers", "hidden_size", "n_heads", "window",
    }
)

KV_LINE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[-\d.eE+]+|\"[^\"]*\"|'[^']*')")

#: Columns that identify rows instead of measuring them.
IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {
        "seed", "seeds", "run", "run_id", "id", "index", "idx", "row", "arm", "arms", "variant",
        "condition", "method", "model", "dataset", "scale", "config", "status", "state", "step",
        "steps", "epoch", "epochs", "fold", "trial", "repeat", "replicate", "date", "time", "commit",
        "gpu", "gpus", "host", "note", "notes", "name", "tag", "tags", "group", "level",
    }
)


def normalise_label(column: str) -> str:
    return column.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class CodeSurface:
    """The declared interface of a script: what it can be configured with, and what it emits."""

    argparse: dict[str, Any]
    defaults: dict[str, Any]
    module_constants: dict[str, Any]
    metric_names: list[str]
    seed_usage: list[str]
    writes: list[str]
    looks_like_experiment: bool
    entrypoints: list[str]


def extract_python(file: SourceFile) -> ExtractionOutput:
    out = ExtractionOutput()
    text = file.text or ""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        out.unparsed.append((file.rel_path, f"python syntax error: {exc.msg} (line {exc.lineno})"))
        return out

    surface = CodeSurface({}, {}, {}, [], [], [], False, [])
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {"main", "run", "train", "evaluate"}:
            surface.entrypoints.append(f"{node.name}:{node.lineno}")
        if isinstance(node, ast.Call) and _call_name(node) == "add_argument":
            name = _first_str_arg(node)
            if not name:
                continue
            key = name.lstrip("-").replace("-", "_")
            default = _kwarg(node, "default")
            surface.argparse[key] = default
            if isinstance(default, (int, float, str, bool)):
                surface.defaults[key] = default
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    value = _literal(node.value)
                    if value is not None:
                        surface.module_constants[target.id] = value
                        if target.id in {"METRICS", "METRIC_NAMES", "METRIC"} and isinstance(value, list):
                            surface.metric_names.extend(str(v) for v in value)
        if isinstance(node, ast.Name) and node.id in {"random_seed", "manual_seed", "seed_everything", "set_seed"}:
            surface.seed_usage.append(f"seed-call:{node.id}:{node.lineno}")
        if isinstance(node, ast.Call):
            callee = _call_name(node)
            if callee in {"write_text", "write_bytes", "save", "savefig", "to_csv", "to_json", "dump"}:
                surface.writes.append(f"{callee}:{node.lineno}")

    if not surface.metric_names:
        surface.metric_names = sorted(_metric_like_constants(surface.module_constants, text))
    surface.looks_like_experiment = bool(
        EXPERIMENT_ARG_NAMES & set(surface.argparse)
    ) and ("train" in file.rel_path.lower() or "run" in file.rel_path.lower() or "exp" in file.rel_path.lower())

    out.facts.append(
        Fact(
            kind=FactKind.CODE_SURFACE,
            statement=f"code surface of {file.rel_path}: {len(surface.argparse)} arguments, "
            f"metrics={surface.metric_names or 'none detected'}",
            rule="python:ast",
            source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT, locator="ast"),
            confidence=0.85,
            payload={
                "argparse": surface.argparse,
                "defaults": surface.defaults,
                "module_constants": {k: v for k, v in list(surface.module_constants.items())[:40]},
                "metric_names": surface.metric_names,
                "seed_usage": surface.seed_usage,
                "writes": surface.writes,
                "entrypoints": surface.entrypoints,
            },
            files=[file.rel_path],
        )
    )
    if surface.looks_like_experiment:
        out.facts.append(
            Fact(
                kind=FactKind.EXPERIMENT,
                statement=f"experiment script {file.rel_path} (configured via {len(surface.argparse)} arguments)",
                rule="python:experiment_shape",
                source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT, locator="ast"),
                confidence=0.7,
                payload={
                    "source_file": file.rel_path,
                    "defaults": surface.defaults,
                    "metric_names": surface.metric_names,
                    "declared": True,
                },
                files=[file.rel_path],
            )
        )
    if not surface.argparse and not surface.module_constants:
        out.notes.append(f"{file.rel_path}: no configurable surface detected")
    return out


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _first_str_arg(node: ast.Call) -> str | None:
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _kwarg(node: ast.Call, name: str) -> Any:
    for keyword in node.keywords:
        if keyword.arg == name:
            return _literal(keyword.value)
    return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _metric_like_constants(constants: dict[str, Any], text: str) -> set[str]:
    names: set[str] = set()
    for value in constants.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and re.fullmatch(r"[a-z][a-z0-9_]{2,30}", item):
                    names.add(item)
    for match in re.finditer(r"\b([a-z][a-z0-9_]*_(?:at|acc|loss|score|rate|pct|ratio)\w*)\b", text):
        names.add(match.group(1))
    return names


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------


def extract_config(file: SourceFile) -> ExtractionOutput:
    out = ExtractionOutput()
    text = file.text or ""
    suffix = Path(file.rel_path).suffix.lower()
    payload: dict[str, Any] = {}
    try:
        if suffix == ".json":
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            import yaml

            loaded = yaml.safe_load(text)
            payload = loaded if isinstance(loaded, dict) else {"value": loaded}
        elif suffix == ".toml":
            import tomllib

            payload = tomllib.loads(text)
        else:
            payload = {k: v for k, v in KV_LINE.findall(text)}
    except Exception as exc:  # malformed configs are common in legacy repos
        out.unparsed.append((file.rel_path, f"config not parseable: {type(exc).__name__}: {exc}"))
        return out

    flat = _flatten(payload)
    out.facts.append(
        Fact(
            kind=FactKind.CONFIG,
            statement=f"config {file.rel_path} with {len(flat)} scalar settings",
            rule=f"config:{suffix.lstrip('.') or 'kv'}",
            source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT, locator="whole file"),
            confidence=0.85,
            payload={"config": flat, "source_file": file.rel_path, "model": flat.get("model"),
                     "dataset": flat.get("dataset"), "seeds": flat.get("seeds") or flat.get("seed")},
            files=[file.rel_path],
        )
    )
    return out


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix.rstrip(".")] = obj
        else:
            for index, value in enumerate(obj):
                out.update(_flatten(value, f"{prefix}{index}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


# --------------------------------------------------------------------------------------
# Tabular data
# --------------------------------------------------------------------------------------


@dataclass
class TableProfile:
    columns: list[str]
    numeric_columns: list[str]
    label_columns: list[str]
    row_count: int
    missing: dict[str, int]
    summary: dict[str, dict[str, float]]


def extract_table(file: SourceFile) -> ExtractionOutput:
    """Read a CSV/TSV into metric facts plus a profile. Numbers come from here, not from prose."""
    out = ExtractionOutput()
    text = file.text or ""
    delimiter = "\t" if file.rel_path.endswith(".tsv") else ","
    try:
        rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    except csv.Error as exc:
        out.unparsed.append((file.rel_path, f"csv parse error: {exc}"))
        return out
    if not rows:
        out.unparsed.append((file.rel_path, "no data rows"))
        return out

    columns = [c for c in rows[0].keys() if c is not None]
    numeric_columns: list[str] = []
    label_columns: list[str] = []
    missing: dict[str, int] = {}
    numeric_values: dict[str, list[float]] = {}

    for column in columns:
        values = [row.get(column) for row in rows]
        missing[column] = sum(1 for v in values if v in (None, ""))
        parsed = [_to_float(v) for v in values]
        # Identifier columns are labels even when they are numeric: `seed=2` identifies a run,
        # it is not a measurement. Getting this wrong turns every run grid into "duplicate runs".
        if normalise_label(column) in IDENTIFIER_COLUMNS or sum(1 for p in parsed if p is None) > max(1, len(rows) // 2):
            label_columns.append(column)
        elif sum(1 for p in parsed if p is not None) >= max(1, len(rows) // 2):
            numeric_columns.append(column)
            numeric_values[column] = [p for p in parsed if p is not None]
        else:
            label_columns.append(column)

    profile = TableProfile(
        columns=columns,
        numeric_columns=numeric_columns,
        label_columns=label_columns,
        row_count=len(rows),
        missing=missing,
        summary={
            column: {
                "n": float(len(values)),
                "mean": round(pystats.fmean(values), 6) if values else 0.0,
                "std": round(pystats.stdev(values), 6) if len(values) > 1 else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for column, values in numeric_values.items()
        },
    )

    out.facts.append(
        Fact(
            kind=FactKind.EVIDENCE,
            statement=f"tabular artifact {file.rel_path}: {profile.row_count} rows, "
            f"{len(numeric_columns)} numeric columns",
            rule="table:profile",
            source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT, locator="whole file"),
            confidence=0.9,
            payload={
                "source_file": file.rel_path,
                "columns": columns,
                "numeric_columns": numeric_columns,
                "label_columns": label_columns,
                "row_count": profile.row_count,
                "missing": missing,
                "summary": profile.summary,
                "rows": rows,
            },
            files=[file.rel_path],
        )
    )

    for index, row in enumerate(rows, start=2):  # header is line 1
        labels = {
            column: row.get(column)
            for column in label_columns
            if row.get(column) not in (None, "")
        }
        status = str(labels.get("status", "")).lower()
        for column in numeric_columns:
            value = _to_float(row.get(column))
            if value is None:
                if row.get(column) in (None, ""):
                    out.facts.append(
                        Fact(
                            kind=FactKind.FAILED_RUN if status else FactKind.METRIC_VALUE,
                            statement=f"missing {column} for {labels or f'row {index}'}",
                            rule="table:missing_value",
                            source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                                   locator=f"{file.rel_path}:{index}"),
                            confidence=0.8,
                            payload={"metric": column, "value": None, "labels": labels,
                                     "row": index, "status": status, "missing": True},
                            files=[file.rel_path],
                        )
                    )
                continue
            out.facts.append(
                Fact(
                    kind=FactKind.METRIC_VALUE,
                    statement=f"{column} = {value:g} for {_label_str(labels) or f'row {index}'}",
                    rule="table:metric",
                    source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                           locator=f"{file.rel_path}:{index}"),
                    confidence=0.9,
                    payload={
                        "metric": column,
                        "value": value,
                        "labels": labels,
                        "row": index,
                        "status": status,
                        "unit": _unit_for(column),
                        "table": file.rel_path,
                    },
                    files=[file.rel_path],
                )
            )
    return out


def _label_str(labels: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(labels.items()) if k != "status")


def _unit_for(column: str) -> str | None:
    lowered = column.lower()
    for token, unit in (("pct", "%"), ("percent", "%"), ("rate", None), ("loss", None), ("flops", "FLOPs")):
        if token in lowered:
            return unit
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().replace("%", ""))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# JSON / JSONL data and logs
# --------------------------------------------------------------------------------------


def extract_json_data(file: SourceFile) -> ExtractionOutput:
    out = ExtractionOutput()
    text = file.text or ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        out.unparsed.append((file.rel_path, f"json parse error: {exc.msg} (line {exc.lineno})"))
        return out

    records = payload if isinstance(payload, list) else [payload]
    scalars: dict[str, float] = {}
    for record in records:
        if isinstance(record, dict):
            for key, value in _flatten(record).items():
                number = _to_float(value)
                if number is not None and not key.endswith(("seed", "row", "index")):
                    scalars[key] = number

    out.facts.append(
        Fact(
            kind=FactKind.EVIDENCE,
            statement=f"json result artifact {file.rel_path} with {len(scalars)} numeric fields",
            rule="json:profile",
            source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT, locator="whole file"),
            confidence=0.85,
            payload={"source_file": file.rel_path, "records": len(records), "scalars": scalars},
            files=[file.rel_path],
        )
    )
    for index, (key, value) in enumerate(sorted(scalars.items())):
        out.facts.append(
            Fact(
                kind=FactKind.METRIC_VALUE,
                statement=f"{key} = {value:g}",
                rule="json:scalar",
                source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                       locator=f"{file.rel_path}#{key}"),
                confidence=0.8,
                payload={"metric": key.split(".")[-1], "value": value, "key": key,
                         "source_file": file.rel_path},
                files=[file.rel_path],
            )
        )
    return out


def extract_log(file: SourceFile) -> ExtractionOutput:
    """Parse key=value log lines. Logs are where *partial* runs and failures become visible."""
    out = ExtractionOutput()
    text = file.text or ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        pairs = {k.lower(): v for k, v in KV_LINE.findall(line)}
        numbers: list[tuple[str, float]] = []
        for key, raw in pairs.items():
            value = _to_float(raw.strip("\"'"))
            if value is not None:
                numbers.append((key, value))
        for key, value in numbers:
            out.facts.append(
                Fact(
                    kind=FactKind.METRIC_VALUE,
                    statement=f"{key} = {value:g} (from log {file.rel_path})",
                    rule="log:kv",
                    source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                           locator=f"{file.rel_path}:{line_number}"),
                    confidence=0.7,
                    payload={"metric": key, "value": value, "source_file": file.rel_path,
                             "line": line_number},
                    files=[file.rel_path],
                )
            )
        failure = matches_any(line, FAILURE_PATTERNS)
        if failure:
            out.facts.append(
                Fact(
                    kind=FactKind.FAILED_RUN,
                    statement=f"failure marker in {file.rel_path}: {line.strip()[:160]}",
                    rule=f"log:failure:{failure}",
                    source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                           locator=f"{file.rel_path}:{line_number}"),
                    confidence=0.7,
                    payload={"source_file": file.rel_path, "line": line_number, "raw": line.strip()[:400]},
                    files=[file.rel_path],
                )
            )
        if not pairs:
            for hit in extract_numbers(line):
                if hit.value is not None:
                    out.facts.append(
                        Fact(
                            kind=FactKind.PROSE_NUMBER,
                            statement=f"{hit.raw} in log {file.rel_path}",
                            rule="log:bare_number",
                            source=file.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                                   locator=f"{file.rel_path}:{line_number}"),
                            confidence=0.4,
                            payload={"value": hit.value, "metric": hit.metric, "raw": hit.raw,
                                     "source_file": file.rel_path},
                            files=[file.rel_path],
                        )
                    )
    if not out.facts:
        out.unparsed.append((file.rel_path, "no key=value pairs or numbers found in log"))
    return out


# --------------------------------------------------------------------------------------
# Bibliography
# --------------------------------------------------------------------------------------

_BIB_ENTRY = re.compile(r"@(?P<type>[A-Za-z]+)\s*\{\s*(?P<key>[^,]+),", re.MULTILINE)


def extract_bib(file: SourceFile) -> ExtractionOutput:
    """Parse BibTeX into literature candidates. Metadata only — never mechanism knowledge."""
    out = ExtractionOutput()
    text = file.text or ""
    for entry in parse_bibtex(text):
        out.facts.append(
            Fact(
                kind=FactKind.LITERATURE,
                statement=f"{entry.get('title', entry['key'])} ({entry.get('year', 'n.d.')})",
                rule="bib:entry",
                source=file.source_ref(kind=SourceKind.LITERATURE, locator=f"@{entry['key']}"),
                confidence=0.85,
                payload=entry,
                files=[file.rel_path],
            )
        )
    if not out.facts:
        out.unparsed.append((file.rel_path, "no bibtex entries parsed"))
    return out


def parse_bibtex(text: str) -> list[dict[str, Any]]:
    """A small, dependency-free BibTeX reader: handles nested braces and `and`-separated authors."""
    entries: list[dict[str, Any]] = []
    for match in _BIB_ENTRY.finditer(text):
        entry_type = match.group("type").lower()
        key = match.group("key").strip()
        if entry_type in {"comment", "string", "preamble"}:
            continue
        body, _end = _balanced_body(text, text.find("{", match.start()))
        fields: dict[str, str] = {}
        for name, value in re.findall(r"(\w+)\s*=\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\"[^\"]*\"|[^,]+)", body):
            cleaned = value.strip().strip("{}").strip('"').strip()
            fields[name.lower()] = re.sub(r"\s+", " ", cleaned).strip()
        if not fields.get("title"):
            continue
        authors = [a.strip() for a in re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()]
        entries.append(
            {
                "key": key,
                "type": entry_type,
                "title": fields.get("title", key),
                "authors": authors,
                "year": _to_int(fields.get("year")),
                "venue": fields.get("journal") or fields.get("booktitle") or fields.get("publisher"),
                "doi": fields.get("doi"),
                "arxiv_id": fields.get("eprint") if (fields.get("archiveprefix", "").lower() == "arxiv") else None,
                "url": fields.get("url"),
                "pages": fields.get("pages"),
                "volume": fields.get("volume"),
                "raw_fields": fields,
            }
        )
    return entries


def _balanced_body(text: str, open_index: int) -> tuple[str, int]:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index
    return text[open_index + 1 :], len(text)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d{4}", str(value))
    return int(match.group(0)) if match else None


# --------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------


def extract_pdf(file: SourceFile) -> ExtractionOutput:
    """Extract text from a PDF when a reader is available; otherwise say so honestly.

    ResearchOS never guesses at a paper's contents: an unreadable PDF becomes an UNPARSED record
    asking a human to supply the text (or attach the abstract), not a fabricated summary.
    """
    out = ExtractionOutput()
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(file.abs_path))
        pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                pages.append(f"\n[page {index}]\n" + (page.extract_text() or ""))
            except Exception as exc:  # a single bad page must not lose the paper
                out.notes.append(f"{file.rel_path}: page {index} unreadable ({type(exc).__name__})")
        text = "\n".join(pages).strip()
    except ImportError:
        out.unparsed.append(
            (file.rel_path, "no PDF reader available (install `pypdf`, or attach the text/abstract)")
        )
        return out
    except Exception as exc:
        out.unparsed.append((file.rel_path, f"pdf unreadable: {type(exc).__name__}: {exc}"))
        return out

    if not text:
        out.unparsed.append((file.rel_path, "pdf yielded no extractable text (scanned image?)"))
        return out

    out.facts.append(
        Fact(
            kind=FactKind.EVIDENCE,
            statement=f"pdf text extracted: {file.rel_path} ({len(text.split())} words)",
            rule="pdf:text",
            source=file.source_ref(kind=SourceKind.LITERATURE, locator="whole document"),
            confidence=0.6,
            payload={"source_file": file.rel_path, "text": text[:200_000], "words": len(text.split()),
                     "fulltext": True},
            files=[file.rel_path],
        )
    )
    out.notes.append(
        f"{file.rel_path}: full text extracted — mark it FULLTEXT_VERIFIED only after a human reads it"
    )
    return out


# --------------------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------------------

DECISION_COMMIT_MARKERS: tuple[str, ...] = (
    r"\brevert\b", r"\breject", r"\babandon", r"\bswitch to\b", r"\bdrop\b", r"\bfix: ?(?:data|order|seed)",
    r"\bbreaking\b", r"\bremove (?:the )?(?:old|previous)\b",
)


def git_log(root: str | Path, *, limit: int = 200, runner=None) -> list[dict[str, Any]]:
    """Read the git history *of this material*. Returns ``[]`` when git or a repository is unavailable.

    Two subtleties, both of which produced wrong provenance before they were handled:

    * when the imported directory is a *subdirectory* of a larger repository (very common: a study
      living inside a monorepo), the history must be restricted to the paths under it — otherwise the
      import claims the whole repository's commits as the project's research history;
    * when the imported directory is outside any repository, there is simply no history to report.

    ``runner`` is injectable so history extraction is testable without a real repository.
    """
    if runner is not None:
        return list(runner(root, limit))

    import shutil

    if shutil.which("git") is None:
        return []

    def _git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=20, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    toplevel = _git("rev-parse", "--show-toplevel")
    if not toplevel:
        return []

    pathspec: list[str] = []
    try:
        root_resolved = Path(root).resolve()
        top_resolved = Path(toplevel).resolve()
        if root_resolved != top_resolved:
            relative = root_resolved.relative_to(top_resolved).as_posix()
            pathspec = ["--", relative]
    except (OSError, ValueError):  # pragma: no cover - defensive
        pathspec = []

    fmt = "%H%x1f%ad%x1f%an%x1f%s%x1e"
    try:
        proc = subprocess.run(
            ["git", "log", f"--max-count={limit}", f"--pretty=format:{fmt}", "--date=short", *pathspec],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    commits: list[dict[str, Any]] = []
    for chunk in proc.stdout.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        parts = chunk.split("\x1f")
        if len(parts) < 4:
            continue
        commits.append({"commit": parts[0], "date": parts[1], "author": parts[2], "subject": parts[3]})
    return commits


def extract_git(root: str | Path, *, runner=None, limit: int = 200) -> ExtractionOutput:
    out = ExtractionOutput()
    commits = git_log(root, limit=limit, runner=runner)
    if not commits:
        out.notes.append("no git history available")
        return out
    for commit in commits:
        ref = SourceFile(
            rel_path=".git",
            abs_path=Path(str(root)),
            kind=FileKind.UNKNOWN,
            size=0,
            sha256=commit["commit"],
        )
        subject = str(commit.get("subject", ""))
        out.facts.append(
            Fact(
                kind=FactKind.GIT_COMMIT,
                statement=f"{commit.get('date', '')}: {subject}",
                rule="git:commit",
                source=ref.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                      locator=f"commit {commit['commit'][:10]}", quote=subject),
                confidence=0.95,
                payload=commit,
            )
        )
        if matches_any(subject, DECISION_COMMIT_MARKERS):
            out.facts.append(
                Fact(
                    kind=FactKind.DECISION,
                    statement=f"repository change: {subject}",
                    rule="git:decision_marker",
                    source=ref.source_ref(kind=SourceKind.RAW_EXPERIMENT,
                                          locator=f"commit {commit['commit'][:10]}", quote=subject),
                    confidence=0.6,
                    payload={**commit, "detected_by": "commit message marker"},
                )
            )
    return out
