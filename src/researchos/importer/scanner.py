"""Scanning and classifying the material a researcher hands us.

Deliberately boring and deterministic: walk the tree, skip the noise, hash everything, classify by
name + content, and read text when it is safe to do so. Nothing here interprets research; that is
the extractors' job, and interpretation always keeps a pointer back to these exact bytes.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..models.common import sha256_bytes
from .facts import TEXT_EXTENSIONS, FileKind, SourceFile

#: Directories that are never research material.
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", ".researchos", "__pycache__", "node_modules", ".venv", "venv",
        "env", ".env", "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".idea", ".vscode", "site-packages", ".ipynb_checkpoints", ".cache", "wandb", ".tox",
    }
)

#: File globs that are never research material.
IGNORED_GLOBS: tuple[str, ...] = (
    "*.pyc", "*.pyo", "*.so", "*.dll", "*.dylib", "*.o", "*.a", "*.class", "*.jar",
    "*.zip", "*.tar", "*.gz", "*.bz2", "*.7z", "*.rar", "*.xz",
    "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin", "*.h5", "*.hdf5", "*.npz", "*.npy",
    "*.parquet", "*.feather", "*.sqlite", "*.db", "*.lock", "*.min.js", "*.map",
    "package-lock.json", "poetry.lock", "uv.lock", ".DS_Store", "Thumbs.db",
)

#: Text files larger than this are recorded but not read (they are usually data dumps).
MAX_TEXT_BYTES = 2_000_000
#: Files larger than this are never read at all.
MAX_FILE_BYTES = 64_000_000

PAPER_NAME_HINTS: tuple[str, ...] = ("paper", "main.tex", "manuscript", "draft", "submission", "camera")
AUDIT_NAME_HINTS: tuple[str, ...] = ("audit", "review", "referee", "critique", "checklist", "verification")
NOTE_NAME_HINTS: tuple[str, ...] = ("note", "diary", "log.md", "journal", "scratch", "todo", "thoughts")
EXPERIMENT_NAME_HINTS: tuple[str, ...] = ("train", "run", "exp", "eval", "experiment", "sweep", "main")
CONFIG_EXTENSIONS: frozenset[str] = frozenset({".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"})


@dataclass
class ScanResult:
    files: list[SourceFile] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    total_bytes: int = 0

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.files:
            out[item.kind.value] = out.get(item.kind.value, 0) + 1
        return out

    def text_files(self) -> list[SourceFile]:
        return [f for f in self.files if f.text is not None]

    def of_kind(self, *kinds: FileKind) -> list[SourceFile]:
        wanted = set(kinds)
        return [f for f in self.files if f.kind in wanted]


def is_ignored(path: Path, root: Path) -> str | None:
    """Return a reason string when the path should be skipped, else ``None``."""
    rel_parts = path.relative_to(root).parts
    for part in rel_parts[:-1]:
        if part in IGNORED_DIRS:
            return f"ignored directory {part!r}"
    name = path.name
    for pattern in IGNORED_GLOBS:
        if fnmatch.fnmatch(name, pattern):
            return f"ignored pattern {pattern!r}"
    return None


def classify(path: Path, *, text: str | None = None, root: Path | None = None) -> tuple[FileKind, str]:
    """Classify a file by name first (cheap, reliable) and by content second.

    Returns ``(kind, reason)``; the reason is stored so a reviewer can see why the importer made a
    decision instead of guessing.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()
    rel = str(path.relative_to(root)).replace("\\", "/").lower() if root else name
    directory = str(Path(rel).parent)

    if suffix == ".pdf":
        return FileKind.PDF, "pdf extension"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".eps"}:
        return FileKind.FIGURE, "figure extension"
    if suffix == ".ipynb":
        return FileKind.NOTEBOOK, "jupyter notebook"
    if suffix in {".bib", ".bbl"}:
        return FileKind.BIB, "bibliography"
    if suffix in {".csv", ".tsv"}:
        return FileKind.DATA_CSV, "tabular data"
    if suffix == ".jsonl":
        return FileKind.LOG, "jsonl stream (log-like)"
    if suffix == ".json":
        if "config" in directory or "config" in name or name.endswith("config.json"):
            return FileKind.CONFIG, "config-looking json"
        return FileKind.DATA_JSON, "json data"
    if suffix in {".log", ".out"} or "logs" in directory.split("/"):
        return FileKind.LOG, "log file"
    if suffix == ".py":
        return FileKind.PYTHON, "python source"
    if suffix == ".tex":
        if any(hint in rel for hint in PAPER_NAME_HINTS) or "\\documentclass" in (text or ""):
            return FileKind.PAPER_DRAFT, "latex document"
        return FileKind.LATEX, "latex source"
    if suffix in {".md", ".markdown", ".rst"}:
        if any(hint in name for hint in AUDIT_NAME_HINTS):
            return FileKind.AUDIT_REPORT, "audit/review document"
        if name.startswith("readme"):
            return FileKind.README, "readme"
        if any(hint in rel for hint in NOTE_NAME_HINTS):
            return FileKind.PLAIN_TEXT, "note/diary document"
        return FileKind.MARKDOWN, "markdown document"
    if suffix in CONFIG_EXTENSIONS - {".json"}:
        return FileKind.CONFIG, "configuration file"
    if suffix in TEXT_EXTENSIONS:
        return FileKind.PLAIN_TEXT, "text file"
    if suffix == "" and text:
        return FileKind.PLAIN_TEXT, "extension-less text"
    if suffix in {".sh", ".bash", ".ps1", ".r", ".jl", ".m", ".cpp", ".c", ".h", ".java", ".rs", ".go", ".js", ".ts"}:
        return FileKind.CODE_OTHER, "source code"
    return FileKind.UNKNOWN, f"unrecognised extension {suffix or '(none)'}"


def read_text(path: Path) -> tuple[str | None, str | None]:
    """Read a file as text. Returns ``(text, error)``; never raises."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"unreadable: {exc}"
    if b"\x00" in raw[:4096]:
        return None, "binary content (NUL byte)"
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), None
        except UnicodeDecodeError:
            continue
    return None, "undecodable text"


def scan(
    root: str | Path,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    max_files: int = 5000,
) -> ScanResult:
    """Walk ``root`` and return the material, classified and hashed.

    Deterministic ordering (sorted paths) so an import is reproducible and diffable.
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"{base} is not a directory")
    result = ScanResult()
    count = 0
    for path in sorted(base.rglob("*")):
        if count >= max_files:
            result.skipped.append(("*", f"max_files={max_files} reached"))
            break
        if not path.is_file():
            continue
        reason = is_ignored(path, base)
        if reason:
            result.skipped.append((str(path.relative_to(base)), reason))
            continue
        rel = str(path.relative_to(base)).replace("\\", "/")
        if include and not any(fnmatch.fnmatch(rel, pattern) for pattern in include):
            continue
        if exclude and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
            result.skipped.append((rel, "excluded by pattern"))
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            result.skipped.append((rel, f"stat failed: {exc}"))
            continue
        if size > MAX_FILE_BYTES:
            result.skipped.append((rel, f"too large ({size} bytes)"))
            continue

        text: str | None = None
        error: str | None = None
        if size <= MAX_TEXT_BYTES and path.suffix.lower() in TEXT_EXTENSIONS | {""}:
            text, error = read_text(path)
        kind, why = classify(path, text=text, root=base)
        if size > MAX_TEXT_BYTES:
            error = error or f"not read: {size} bytes exceeds text limit"
            text = None

        digest = sha256_bytes(path.read_bytes()) if size <= MAX_TEXT_BYTES else _hash_large(path)
        result.files.append(
            SourceFile(
                rel_path=rel,
                abs_path=path,
                kind=kind,
                size=size,
                sha256=digest,
                text=text,
                classification_reason=why + (f" | {error}" if error else ""),
                error=error,
            )
        )
        result.total_bytes += size
        count += 1
    return result


def _hash_large(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def group_by_kind(files: Iterable[SourceFile]) -> dict[FileKind, list[SourceFile]]:
    out: dict[FileKind, list[SourceFile]] = {}
    for item in files:
        out.setdefault(item.kind, []).append(item)
    return out
