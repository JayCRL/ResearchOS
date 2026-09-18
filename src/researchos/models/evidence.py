"""Evidence objects and the provenance that makes them checkable.

Invariant 1 lives here: ``RAW`` evidence is immutable. ResearchOS does not offer an API that
edits raw observations — you append a new record that supersedes it, and the old one stays.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from .common import (
    ArtifactRef,
    EvidenceLevel,
    EvidenceType,
    Provenance,
    RosModel,
    SourceKind,
    SourceRef,
    VerificationStatus,
    content_hash,
    new_id,
    utcnow,
)


class Evidence(RosModel):
    """A single traceable fact, from a raw observation up to a claimed assertion."""

    evidence_id: str = Field(default_factory=lambda: new_id("evidence"))
    evidence_type: EvidenceType = EvidenceType.RAW
    statement: str = Field(min_length=1, description="What this evidence asserts, in one sentence.")

    source_kind: SourceKind = SourceKind.RAW_EXPERIMENT
    source_path: str | None = None
    artifact_hash: str | None = None
    source_experiment: str | None = None
    source_analysis: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)

    payload: dict[str, Any] = Field(
        default_factory=dict, description="Structured values (numbers) copied from artifacts."
    )
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    derived_from: list[str] = Field(
        default_factory=list, description="Parent evidence ids: the evidence graph edges."
    )

    evidence_level: EvidenceLevel = EvidenceLevel.L0_IDEA
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verified_by: str | None = None
    verified_at: datetime | None = None
    verification_method: str | None = None
    verification_notes: list[str] = Field(default_factory=list)

    claim_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)

    provenance: Provenance = Field(default_factory=Provenance)
    immutable: bool = Field(
        default=False,
        description="True for RAW evidence; the registry refuses updates once this is set.",
    )
    supersedes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "human"
    task_id: str | None = None
    confidence_note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "Evidence":
        if self.evidence_type is EvidenceType.RAW:
            if not (self.artifacts or self.payload or self.source_path):
                raise ValueError(
                    f"evidence {self.evidence_id}: RAW evidence needs an artifact, payload or source_path"
                )
            # RAW is immutable by construction, not by convention.
            object.__setattr__(self, "immutable", True)
        if self.evidence_type.rank >= EvidenceType.VERIFIED.rank:
            if not self.verified_by or not self.verified_at or not self.verification_method:
                raise ValueError(
                    f"evidence {self.evidence_id}: {self.evidence_type.value} evidence requires "
                    "verified_by, verified_at and verification_method"
                )
        if self.evidence_type is EvidenceType.CLAIMED and not self.claim_ids:
            raise ValueError(
                f"evidence {self.evidence_id}: CLAIMED evidence must reference the claim it supports"
            )
        if self.evidence_type is EvidenceType.ANALYZED and not self.source_analysis:
            raise ValueError(
                f"evidence {self.evidence_id}: ANALYZED evidence must reference source_analysis"
            )
        if self.verification_status is VerificationStatus.VERIFIED and not self.verified_by:
            raise ValueError(
                f"evidence {self.evidence_id}: verification_status VERIFIED requires verified_by"
            )
        return self

    def content_hash(self) -> str:
        """Hash of the scientific content (not the metadata) — detects post-hoc edits."""
        return content_hash(
            {
                "statement": self.statement,
                "payload": self.payload,
                "artifacts": [a.model_dump(mode="json") for a in self.artifacts],
                "source_experiment": self.source_experiment,
                "source_analysis": self.source_analysis,
                "evidence_level": self.evidence_level.value,
            }
        )

    def verify_artifacts(self, root: "str | None" = None) -> list[tuple[str, VerificationStatus]]:
        """Re-hash every artifact. Returns ``(path, status)`` pairs — an empty list means clean."""
        from pathlib import Path

        base = Path(root) if root else None
        out: list[tuple[str, VerificationStatus]] = []
        for artifact in self.artifacts:
            status = artifact.verify(base)
            if status is not VerificationStatus.VERIFIED:
                out.append((artifact.path, status))
        return out


class EvidenceLink(RosModel):
    """An explicit edge in the provenance chain claim -> analysis -> experiment -> raw data."""

    from_id: str
    to_id: str
    relation: str = Field(
        description="supports | contradicts | derives_from | verifies | interprets | invalidates"
    )
    weight: float | None = None
    note: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
