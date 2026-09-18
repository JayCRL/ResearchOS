"""The ``.researchos/`` layout: where research state actually lives on disk.

Layout is an interface. Every path here is documented in ``ARCHITECTURE.md`` and asserted by
``tests/test_project_layout.py``, because a research repository that reorganises itself silently
is a research repository nobody can audit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ProjectExists, ProjectNotFound

#: Directory name that marks a ResearchOS project.
MARKER = ".researchos"

#: Entity kind -> directory (relative to ``.researchos``).
ENTITY_DIRS: dict[str, str] = {
    "project": ".",
    "research_state": "state",
    "task": "state/tasks",
    "decision": "state/decisions",
    "str": "state/transitions",
    "claim": "claims",
    "experiment": "experiments",
    "evidence": "evidence",
    "analysis": "analysis",
    "literature_paper": "literature/papers",
    "literature_relation": "literature/graph",
    "literature_claim": "literature/claims",
    "query_plan": "literature/search",
    "novelty_audit": "literature/novelty",
    "prior_art_matrix": "literature/prior_art",
    "gap": "literature/gaps",
    "open_question": "state/open_questions",
    "skill": "skills/registry",
    "skill_version": "skills/versions",
    "benchmark_task": "skills/benchmarks/tasks",
    "benchmark_run": "skills/benchmarks/runs",
    "sandbox_report": "skills/sandbox",
    "skill_gap": "skills/gaps",
    "audit": "audits",
    "conflict": "conflicts",
    "review_item": "review",
    "timeline_event": "timeline",
    "author_note": "notes",
    "paper": "paper/artifacts",
    "grounding_report": "paper/audits",
    "red_team_report": "paper/audits",
    "readiness": "paper/audits",
    "interpretation": "analysis/interpretations",}

#: Directories created by ``researchos init``.
STANDARD_DIRS: tuple[str, ...] = (
    "state",
    "state/tasks",
    "state/decisions",
    "state/transitions",
    "claims",
    "experiments",
    "evidence",
    "evidence/raw",
    "analysis",
    "analysis/interpretations",
    "literature",
    "literature/papers",
    "literature/graph",
    "literature/claims",
    "literature/search",
    "literature/prior_art",
    "literature/novelty",
    "literature/gaps",
    "skills",
    "skills/registry",
    "skills/versions",
    "skills/active",
    "skills/candidates",
    "skills/benchmarks",
    "skills/benchmarks/tasks",
    "skills/benchmarks/runs",
    "skills/sandbox",
    "skills/gaps",
    "skills/deprecated",
    "audits",
    "conflicts",
    "review",
    "timeline",
    "notes",
    "paper",
    "paper/artifacts",
    "paper/audits",
    "cache",
)

#: Paths that are never a source of truth and may be deleted at any time.
DISPOSABLE_DIRS: tuple[str, ...] = ("cache",)


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem layout for one project."""

    root: Path

    # ------------------------------------------------------------------ constructors

    @classmethod
    def for_root(cls, root: str | os.PathLike[str]) -> "ProjectPaths":
        return cls(root=Path(root).resolve())

    @classmethod
    def discover(cls, start: str | os.PathLike[str] | None = None) -> "ProjectPaths":
        """Walk upwards from ``start`` looking for a ``.researchos`` directory."""
        current = Path(start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / MARKER).is_dir():
                return cls(root=candidate)
        raise ProjectNotFound(
            f"no {MARKER}/ found in {current} or any parent directory. "
            "Run `researchos init` here, or `researchos import <path>`."
        )

    @classmethod
    def discover_or_none(cls, start: str | os.PathLike[str] | None = None) -> "ProjectPaths | None":
        try:
            return cls.discover(start)
        except ProjectNotFound:
            return None

    # ------------------------------------------------------------------ layout

    @property
    def ros(self) -> Path:
        return self.root / MARKER

    @property
    def project_file(self) -> Path:
        return self.ros / "project.yaml"

    @property
    def research_state_file(self) -> Path:
        return self.ros / "state" / "research_state.yaml"

    @property
    def event_log(self) -> Path:
        return self.ros / "state" / "events.jsonl"

    @property
    def index_db(self) -> Path:
        return self.ros / "cache" / "index.sqlite"

    def dir_for(self, kind: str) -> Path:
        rel = ENTITY_DIRS.get(kind)
        if rel is None:
            raise KeyError(f"unknown entity kind {kind!r}; add it to ENTITY_DIRS")
        return self.ros / rel if rel != "." else self.ros

    def entity_file(self, kind: str, entity_id: str) -> Path:
        return self.dir_for(kind) / f"{entity_id}.yaml"

    def raw_evidence_dir(self) -> Path:
        return self.ros / "evidence" / "raw"

    def artifact_path(self, relative: str) -> Path:
        """Resolve a project-relative artifact path and refuse escapes."""
        candidate = (self.root / relative).resolve()
        if self.root.resolve() not in candidate.parents and candidate != self.root.resolve():
            raise ValueError(f"artifact path {relative!r} escapes the project root")
        return candidate

    def rel(self, path: str | os.PathLike[str]) -> str:
        return os.path.relpath(Path(path), self.root).replace("\\", "/")

    # ------------------------------------------------------------------ lifecycle

    def exists(self) -> bool:
        return self.ros.is_dir() and self.project_file.is_file()

    def ensure(self) -> "ProjectPaths":
        self.ros.mkdir(parents=True, exist_ok=True)
        for rel in STANDARD_DIRS:
            (self.ros / rel).mkdir(parents=True, exist_ok=True)
        return self

    def assert_absent(self) -> None:
        if (self.ros).exists() and any((self.ros).iterdir()):
            raise ProjectExists(f"{self.ros} already exists and is not empty")

    def assert_exists(self) -> None:
        if not self.exists():
            raise ProjectNotFound(
                f"{self.root} is not a ResearchOS project (missing {MARKER}/project.yaml)"
            )

    def describe(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for rel in STANDARD_DIRS:
            target = self.ros / rel
            count = len(list(target.glob("*.yaml"))) if target.is_dir() else 0
            rows.append((rel, f"{count} records" if count else "—"))
        return rows
