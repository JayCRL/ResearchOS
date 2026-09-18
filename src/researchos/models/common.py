"""Shared primitives: identity, time, hashing, enums and the strict base model.

Nothing in this module depends on the rest of ResearchOS. Everything else may depend on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------

#: Crockford base32 (no I, L, O, U) — ULID-compatible alphabet.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

#: Prefix for every entity type. Prefixes make an id self-describing in logs and papers.
ID_PREFIXES: Mapping[str, str] = {
    "project": "prj",
    "research_state": "rss",
    "task": "tsk",
    "experiment": "exp",
    "evidence": "evd",
    "analysis": "ana",
    "claim": "clm",
    "decision": "dec",
    "str": "str",
    "literature_paper": "lit",
    "literature_relation": "lrel",
    "literature_claim": "lclm",
    "query_plan": "qp",
    "query": "qry",
    "novelty_audit": "nva",
    "prior_art_matrix": "pam",
    "gap": "gap",
    "skill": "skl",
    "skill_version": "skv",
    "benchmark_task": "btk",
    "benchmark_run": "bnr",
    "audit": "aud",
    "conflict": "cfl",
    "paper": "pap",
    "timeline_event": "tl",
    "finding": "fnd",
    "review_item": "rvi",
    "event": "evt",
    "number": "num",
}


def new_id(kind: str) -> str:
    """Return a new sortable, prefixed identifier, e.g. ``clm_01JBXR4M9K3T7QW2ZP5N8V0D6H``.

    Layout: 48-bit millisecond timestamp + 80 bits of randomness, Crockford base32 encoded
    to 26 characters. Lexicographic order equals creation order, which keeps YAML dumps,
    event logs and git diffs chronologically readable.
    """
    if kind not in ID_PREFIXES:
        raise KeyError(f"unknown id kind {kind!r}; add it to ID_PREFIXES")
    ts = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = secrets.randbits(80)
    value = (ts << 80) | rand
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return f"{ID_PREFIXES[kind]}_{''.join(reversed(chars))}"


def id_kind(identifier: str) -> str:
    """Inverse of :func:`new_id` for validation purposes (``"clm_..."`` -> ``"claim"``)."""
    prefix = identifier.split("_", 1)[0]
    for kind, value in ID_PREFIXES.items():
        if value == prefix:
            return kind
    raise ValueError(f"unknown id prefix in {identifier!r}")


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------


def utcnow() -> datetime:
    """Timezone-aware UTC now. All ResearchOS timestamps are aware UTC."""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# Hashing / canonicalisation
# --------------------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 safe.

    Used for every content hash in the system, so the same logical object always hashes
    to the same digest regardless of dict insertion order or platform.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 over the canonical form of ``obj``. Pydantic models are unwrapped first."""
    return sha256_text(canonical_json(_to_plain(obj)))


def sha256_file(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    """Stream-hash a file. Raises :class:`FileNotFoundError` for missing paths.

    Deliberately *not* tolerant: a missing artifact must surface as an error, never as an
    empty hash that silently "verifies".
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _to_plain(obj: Any) -> Any:
    """Recursively convert pydantic models / enums / paths into JSON-ready primitives."""
    if isinstance(obj, BaseModel):
        return {k: _to_plain(v) for k, v in obj.model_dump(mode="json").items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def content_hash(obj: Any) -> str:
    """Alias used across the codebase for "hash this logical object"."""
    return sha256_text(canonical_json(_to_plain(obj)))


def short_hash(value: str, length: int = 12) -> str:
    return value[:length]


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class StrEnum(str, Enum):
    """String enum whose ``str()`` is its value (nicer YAML/CLI output)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class EvidenceLevel(StrEnum):
    """The evidence ladder. Language strength is *capped* by this level (see ``claims.language``)."""

    L0_IDEA = "L0_IDEA"
    L1_OBSERVATION = "L1_OBSERVATION"
    L2_REPRODUCED = "L2_REPRODUCED"
    L3_CONTROLLED = "L3_CONTROLLED"
    L4_INTERVENTION = "L4_INTERVENTION"
    L5_NECESSITY_OR_SUFFICIENCY = "L5_NECESSITY_OR_SUFFICIENCY"
    L6_CROSS_SETTING_REPLICATION = "L6_CROSS_SETTING_REPLICATION"

    @property
    def rank(self) -> int:
        return int(self.value[1])

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, EvidenceLevel):
            return self.rank < other.rank
        return NotImplemented

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, EvidenceLevel):
            return self.rank <= other.rank
        return NotImplemented

    def __gt__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, EvidenceLevel):
            return self.rank > other.rank
        return NotImplemented

    def __ge__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, EvidenceLevel):
            return self.rank >= other.rank
        return NotImplemented


class ClaimStatus(StrEnum):
    IDEA = "IDEA"
    HYPOTHESIS = "HYPOTHESIS"
    TESTED = "TESTED"
    SUPPORTED = "SUPPORTED"
    ROBUST = "ROBUST"
    WEAKENED = "WEAKENED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class TaskPriority(StrEnum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    EXPLORATORY = "EXPLORATORY"

    @property
    def rank(self) -> int:
        return {"EXPLORATORY": 0, "SECONDARY": 1, "PRIMARY": 2}[self.value]


class TaskPurpose(StrEnum):
    DEBUG = "DEBUG"
    EXPLAIN = "EXPLAIN"
    DESIGN = "DESIGN"
    RUN = "RUN"
    ANALYSE = "ANALYSE"
    LITERATURE = "LITERATURE"
    WRITE = "WRITE"
    IMPORT = "IMPORT"
    AUDIT = "AUDIT"
    SKILL = "SKILL"


class TaskStatus(StrEnum):
    OPEN = "OPEN"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    ABANDONED = "ABANDONED"


class EvidenceType(StrEnum):
    """Evidence maturity. ``RAW`` is immutable; promotion happens by creating a new record."""

    RAW = "RAW"
    ANALYZED = "ANALYZED"
    VERIFIED = "VERIFIED"
    INTERPRETED = "INTERPRETED"
    CLAIMED = "CLAIMED"

    @property
    def rank(self) -> int:
        return ["RAW", "ANALYZED", "VERIFIED", "INTERPRETED", "CLAIMED"].index(self.value)


class SourceKind(StrEnum):
    RAW_EXPERIMENT = "RAW_EXPERIMENT"
    VERIFIED_ANALYSIS = "VERIFIED_ANALYSIS"
    AUDIT = "AUDIT"
    EXPERIMENT_REPORT = "EXPERIMENT_REPORT"
    PAPER_PROSE = "PAPER_PROSE"
    AI_SUMMARY = "AI_SUMMARY"
    LITERATURE = "LITERATURE"
    AUTHOR_NOTE = "AUTHOR_NOTE"


#: Deterministic trust ordering used by conflict resolution (higher wins).
SOURCE_TRUST: Mapping[SourceKind, int] = {
    SourceKind.RAW_EXPERIMENT: 6,
    SourceKind.VERIFIED_ANALYSIS: 5,
    SourceKind.AUDIT: 4,
    SourceKind.EXPERIMENT_REPORT: 3,
    SourceKind.PAPER_PROSE: 2,
    SourceKind.AI_SUMMARY: 1,
    SourceKind.LITERATURE: 3,
    SourceKind.AUTHOR_NOTE: 2,
}


class VerificationStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    HASH_MISMATCH = "HASH_MISMATCH"
    MISSING_ARTIFACT = "MISSING_ARTIFACT"
    PARTIAL = "PARTIAL"


class ExperimentStatus(StrEnum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ChangeClass(StrEnum):
    CORE_QUESTION = "CORE_QUESTION"
    CORE_CLAIM = "CORE_CLAIM"
    PRIORITY = "PRIORITY"
    NON_GOAL = "NON_GOAL"
    SCOPE = "SCOPE"
    OTHER = "OTHER"


class TransitionStatus(StrEnum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"
    STALE = "STALE"


class RelationType(StrEnum):
    CITES = "cites"
    EXTENDS = "extends"
    CONTRADICTS = "contradicts"
    SAME_PROBLEM = "same_problem"
    SAME_MECHANISM = "same_mechanism"
    SIMILAR_MECHANISM = "similar_mechanism"
    DIFFERENT_ASSUMPTION = "different_assumption"
    SAME_METRIC = "same_metric"
    USES_SAME_DATASET = "uses_same_dataset"


class FulltextStatus(StrEnum):
    ABSTRACT_LEVEL_ONLY = "ABSTRACT_LEVEL_ONLY"
    FULLTEXT_VERIFIED = "FULLTEXT_VERIFIED"
    UNKNOWN = "UNKNOWN"


class Tri(StrEnum):
    """Three-valued logic for prior-art matrices. ``UNKNOWN`` is a first-class answer."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class GapKind(StrEnum):
    KNOWN = "KNOWN"
    KNOWN_DIFFERENCE = "KNOWN_DIFFERENCE"
    OPEN_QUESTION = "OPEN_QUESTION"
    CANDIDATE_GAP = "CANDIDATE_GAP"
    VERIFIED_NOVELTY = "VERIFIED_NOVELTY"


class NoveltyVerdict(StrEnum):
    NO_MATCH_FOUND_IN_SEARCHED_COVERAGE = "NO_MATCH_FOUND_IN_SEARCHED_COVERAGE"
    SIMILAR_PRIOR_WORK_FOUND = "SIMILAR_PRIOR_WORK_FOUND"
    NOVELTY_UNCERTAIN = "NOVELTY_UNCERTAIN"


class AuditKind(StrEnum):
    STATISTICAL = "STATISTICAL"
    MECHANISM = "MECHANISM"
    NOVELTY = "NOVELTY"
    STYLE = "STYLE"
    RED_TEAM = "RED_TEAM"
    PROVENANCE = "PROVENANCE"
    CLAIM = "CLAIM"
    EVIDENCE = "EVIDENCE"
    IMPORT = "IMPORT"


class Severity(StrEnum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    BLOCKER = "BLOCKER"


class SkillStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    CANDIDATE = "CANDIDATE"
    SANDBOX = "SANDBOX"
    EVALUATED = "EVALUATED"
    VERIFIED = "VERIFIED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    DEPRECATED = "DEPRECATED"

    @property
    def rank(self) -> int:
        order = [
            "DISCOVERED",
            "CANDIDATE",
            "SANDBOX",
            "EVALUATED",
            "VERIFIED",
            "ACTIVE",
            "DEGRADED",
            "DEPRECATED",
        ]
        return order.index(self.value)


class MechanismRung(StrEnum):
    """The mechanism ladder used by the Mechanism Auditor."""

    OBSERVATION = "OBSERVATION"
    CORRELATION = "CORRELATION"
    CONTROLLED_COMPARISON = "CONTROLLED_COMPARISON"
    INTERVENTION = "INTERVENTION"
    NECESSITY = "NECESSITY"
    SUFFICIENCY = "SUFFICIENCY"
    RESCUE = "RESCUE"
    CROSS_SETTING_REPLICATION = "CROSS_SETTING_REPLICATION"

    @property
    def rank(self) -> int:
        order = [
            "OBSERVATION",
            "CORRELATION",
            "CONTROLLED_COMPARISON",
            "INTERVENTION",
            "NECESSITY",
            "SUFFICIENCY",
            "RESCUE",
            "CROSS_SETTING_REPLICATION",
        ]
        return order.index(self.value)


class ConflictKind(StrEnum):
    VALUE = "VALUE"
    CLAIM = "CLAIM"
    METRIC = "METRIC"
    DESIGN = "DESIGN"
    LITERATURE = "LITERATURE"
    PROVENANCE = "PROVENANCE"


class ConflictResolution(StrEnum):
    A_WINS = "A_WINS"
    B_WINS = "B_WINS"
    BOTH_VALID = "BOTH_VALID"
    UNRESOLVED = "UNRESOLVED"
    MERGED = "MERGED"


class ReviewDecision(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    EDITED = "EDITED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"


class ReadinessDimension(StrEnum):
    EVIDENCE_COMPLETENESS = "EVIDENCE_COMPLETENESS"
    STATISTICAL_COMPLETENESS = "STATISTICAL_COMPLETENESS"
    MECHANISM_EVIDENCE = "MECHANISM_EVIDENCE"
    LITERATURE_COVERAGE = "LITERATURE_COVERAGE"
    CLAIM_GROUNDING = "CLAIM_GROUNDING"
    REPRODUCIBILITY = "REPRODUCIBILITY"
    WRITING_READINESS = "WRITING_READINESS"


# --------------------------------------------------------------------------------------
# Strict base model
# --------------------------------------------------------------------------------------


class RosModel(BaseModel):
    """Base for every ResearchOS object.

    * ``extra="forbid"`` — an unknown key in a YAML file is an error, not a silent no-op.
      This is what makes hand-edited research state trustworthy.
    * ``validate_assignment=True`` — mutations are schema-checked, so ``claim.status = "banana"``
      raises at the point of the mistake.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        arbitrary_types_allowed=False,
        # Fields like Provenance.model_hash are legitimate domain names, not namespace collisions.
        protected_namespaces=(),
    )

    def with_updates(self, **fields: Any) -> "RosModel":
        """Return a validated copy with several fields replaced at once.

        Use this instead of sequential attribute assignment whenever the model has a cross-field
        invariant. With ``validate_assignment=True``, setting ``status`` before the fields it
        requires would fail spuriously; merging first and validating once is both correct and
        simpler to reason about.
        """
        data = self.model_dump(mode="python")
        data.update(fields)
        return type(self).model_validate(data)


class ArtifactRef(RosModel):
    """A pointer to bytes on disk, with the digest that proves we looked at *these* bytes.

    ``sha256`` is computed by reading the file; :meth:`verify` re-reads it. A changed file
    therefore produces ``HASH_MISMATCH`` instead of a silently stale "verified" record.
    """

    path: str = Field(description="Repository-relative path (never absolute, never '..').")
    sha256: str = Field(default="", description="Lowercase hex SHA-256 of the file bytes.")
    bytes: int | None = Field(default=None, ge=0)
    kind: str = Field(default="file", description="file | directory | inline | url")
    note: str | None = None

    def verify(self, root: Path | None = None) -> VerificationStatus:
        target = Path(self.path) if root is None else Path(root) / self.path
        if not target.is_file():
            return VerificationStatus.MISSING_ARTIFACT
        try:
            actual = sha256_file(target)
        except OSError:
            return VerificationStatus.MISSING_ARTIFACT
        if not self.sha256:
            return VerificationStatus.UNVERIFIED
        return (
            VerificationStatus.VERIFIED
            if actual == self.sha256
            else VerificationStatus.HASH_MISMATCH
        )


class Provenance(RosModel):
    """Everything needed to answer "where did this number come from?".

    Fields are individually optional because imported legacy research is frequently
    incomplete — but absence is *recorded* (``missing``), never papered over.
    """

    code_commit: str | None = None
    code_dirty: bool | None = None
    code_remote: str | None = None
    config_hash: str | None = None
    dataset_hashes: list[str] = Field(default_factory=list)
    model_hash: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    random_seeds: list[int] = Field(default_factory=list)
    timestamp: datetime | None = None
    artifact_hash: str | None = None
    command: str | None = None
    host: str | None = None
    python_version: str | None = None
    source_files: list[str] = Field(default_factory=list)

    def missing(self) -> list[str]:
        """Names of provenance fields that are absent — surfaced in audits, not hidden."""
        tracked = (
            "code_commit",
            "config_hash",
            "random_seeds",
            "timestamp",
            "command",
        )
        return [name for name in tracked if not getattr(self, name)]

    def completeness(self) -> float:
        tracked = 5
        return round((tracked - len(self.missing())) / tracked, 4)


class SourceRef(RosModel):
    """A citation into the user's own material (file + locator) — the base unit of grounding."""

    path: str
    sha256: str | None = None
    locator: str | None = Field(
        default=None, description="line:12, page 4, sheet 'runs'!B3, commit abc123, json pointer /a/b"
    )
    quote: str | None = Field(default=None, max_length=1200)
    kind: SourceKind = SourceKind.AUTHOR_NOTE


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def clampi(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def clampf(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean_of(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
