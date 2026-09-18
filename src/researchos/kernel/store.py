"""Entity storage: one YAML document per record, atomic writes, optimistic concurrency.

Design decisions, and why:

* **Files, not a database, are the source of truth.** Research state must be reviewable in a pull
  request, diffable, and recoverable without our software. SQLite appears only as a rebuildable
  index under ``cache/``.
* **Atomic replace.** Every write goes to a temp file in the same directory and is then ``os.replace``d,
  so a crash never leaves a half-written claim.
* **Append-only by default.** :meth:`EntityStore.delete` refuses for research entities. Research
  history is retained (invariant 10); records are superseded, never removed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generic, Iterable, Iterator, Sequence, Type, TypeVar

import yaml
from pydantic import ValidationError

from ..models.common import RosModel, canonical_json
from .errors import ImmutableResourceError, ResearchOSError
from .paths import ProjectPaths

T = TypeVar("T", bound=RosModel)

#: Entity kinds whose records may never be deleted through the store API.
NON_DELETABLE_KINDS: frozenset[str] = frozenset(
    {
        "claim",
        "experiment",
        "evidence",
        "analysis",
        "literature_paper",
        "literature_claim",
        "literature_relation",
        "decision",
        "str",
        "audit",
        "conflict",
        "task",
        "novelty_audit",
        "prior_art_matrix",
        "timeline_event",
        "author_note",
        "interpretation",
        "paper",
        "gap",
        "skill",
        "skill_version",
        "benchmark_run",
        "sandbox_report",
        "review_item",
    }
)


class YamlIO:
    """YAML read/write helpers with a stable, diff-friendly style."""

    @staticmethod
    def dump(model: RosModel) -> str:
        return yaml.safe_dump(
            model.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )

    @staticmethod
    def load_text(text: str) -> dict:
        data = yaml.safe_load(text)
        return data or {}

    @staticmethod
    def write_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @staticmethod
    def read(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(path)
        return YamlIO.load_text(path.read_text(encoding="utf-8"))


class EntityStore(Generic[T]):
    """A typed collection of records persisted as individual YAML files."""

    def __init__(self, paths: ProjectPaths, kind: str, model: Type[T], id_field: str) -> None:
        self.paths = paths
        self.kind = kind
        self.model = model
        self.id_field = id_field

    @property
    def directory(self) -> Path:
        return self.paths.dir_for(self.kind)

    def path_for(self, entity_id: str) -> Path:
        return self.paths.entity_file(self.kind, entity_id)

    def exists(self, entity_id: str) -> bool:
        return self.path_for(entity_id).is_file()

    def get(self, entity_id: str) -> T | None:
        path = self.path_for(entity_id)
        if not path.is_file():
            return None
        try:
            return self.model.model_validate(YamlIO.read(path))
        except ValidationError as exc:  # a hand-edited file that no longer fits the schema
            raise ResearchOSError(
                f"{path} does not satisfy the {self.kind} schema: {exc}", resource=str(path)
            ) from exc

    def require(self, entity_id: str) -> T:
        found = self.get(entity_id)
        if found is None:
            raise ResearchOSError(f"{self.kind} {entity_id!r} not found in {self.directory}")
        return found

    def all(self) -> list[T]:
        out: list[T] = []
        if not self.directory.is_dir():
            return out
        for path in sorted(self.directory.glob("*.yaml")):
            try:
                out.append(self.model.model_validate(YamlIO.read(path)))
            except ValidationError as exc:
                raise ResearchOSError(
                    f"{path} does not satisfy the {self.kind} schema: {exc}", resource=str(path)
                ) from exc
        return out

    def ids(self) -> list[str]:
        return [getattr(item, self.id_field) for item in self.all()]

    def filter(self, predicate) -> list[T]:
        return [item for item in self.all() if predicate(item)]

    def find(self, predicate) -> T | None:
        for item in self.all():
            if predicate(item):
                return item
        return None

    def count(self) -> int:
        return len(list(self.directory.glob("*.yaml"))) if self.directory.is_dir() else 0

    def __iter__(self) -> Iterator[T]:
        return iter(self.all())

    def __len__(self) -> int:
        return self.count()

    def save(self, entity: T, *, overwrite: bool = True) -> T:
        entity_id = getattr(entity, self.id_field)
        path = self.path_for(entity_id)
        if path.exists() and not overwrite:
            raise ResearchOSError(f"{self.kind} {entity_id} already exists at {path}")
        YamlIO.write_atomic(path, YamlIO.dump(entity))
        return entity

    def save_all(self, entities: Iterable[T]) -> int:
        count = 0
        for entity in entities:
            self.save(entity)
            count += 1
        return count

    def delete(self, entity_id: str, *, reason: str = "", allow: bool = False) -> None:
        """Refuse by default. Research history is append-only; supersede instead of deleting."""
        if self.kind in NON_DELETABLE_KINDS or not allow:
            raise ImmutableResourceError(
                f"refusing to delete {self.kind} {entity_id}: research records are retained "
                "(mark a claim SUPERSEDED, set an experiment status, or link `supersedes`). "
                f"reason given: {reason or 'none'}",
                resource=f"{self.kind}/{entity_id}",
            )
        path = self.path_for(entity_id)
        if path.is_file():
            path.unlink()

    def digest(self, entity: T) -> str:
        return canonical_json(entity.model_dump(mode="json"))


class ProjectStore:
    """All entity stores for one project."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self._stores: dict[str, EntityStore] = {}

    def register(self, kind: str, model: Type[T], id_field: str) -> EntityStore[T]:
        store: EntityStore[T] = EntityStore(self.paths, kind, model, id_field)
        self._stores[kind] = store
        return store

    def __getitem__(self, kind: str) -> EntityStore:
        return self._stores[kind]

    def kinds(self) -> list[str]:
        return sorted(self._stores)

    def save_yaml(self, relative: str, payload: dict, *, overwrite: bool = True) -> Path:
        path = self.paths.ros / relative
        if path.exists() and not overwrite:
            raise ResearchOSError(f"{path} already exists")
        YamlIO.write_atomic(
            path,
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        )
        return path

    def load_yaml(self, relative: str) -> dict:
        return YamlIO.read(self.paths.ros / relative)


def merge_missing(existing: Sequence[str], incoming: Iterable[str]) -> list[str]:
    """Union of two id lists, order-preserving. Used when linking records without rewriting history."""
    out = list(existing)
    seen = set(out)
    for item in incoming:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
