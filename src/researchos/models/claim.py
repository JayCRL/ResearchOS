"""Claims — the objects whose strength must never exceed their evidence.

A claim is not a sentence. It is a *state machine* with evidence obligations attached.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from .common import (
    ClaimStatus,
    EvidenceLevel,
    RosModel,
    StrEnum,
    new_id,
    utcnow,
)


class RejectionBasis(StrEnum):
    """Why a claim was rejected. Rejection always leaves a tombstone with a stated basis."""

    NO_EVIDENCE = "NO_EVIDENCE"
    CONTRADICTED = "CONTRADICTED"
    SCOPE_ABANDONED = "SCOPE_ABANDONED"
    SUPERSEDED_BY_DESIGN = "SUPERSEDED_BY_DESIGN"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    OTHER = "OTHER"


class ClaimHistoryEntry(RosModel):
    """Append-only claim history. Nothing in ResearchOS rewrites a claim's past."""

    at: datetime = Field(default_factory=utcnow)
    by: str = Field(description="Principal name, e.g. 'human', 'claim_manager', 'importer'.")
    from_status: ClaimStatus | None = None
    to_status: ClaimStatus
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    str_id: str | None = None


class Claim(RosModel):
    """A falsifiable assertion with an explicit evidence obligation."""

    claim_id: str = Field(default_factory=lambda: new_id("claim"))
    statement: str = Field(min_length=1)
    status: ClaimStatus = ClaimStatus.IDEA
    scope: str = Field(
        default="",
        description="Where the claim is asserted to hold: model family, scale, dataset, regime.",
    )
    evidence_level: EvidenceLevel = EvidenceLevel.L0_IDEA

    evidence_ids: list[str] = Field(default_factory=list)
    literature_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    supporting_experiments: list[str] = Field(default_factory=list)
    contradicting_experiments: list[str] = Field(default_factory=list)

    parent_question: str | None = Field(
        default=None, description="Which question this claim answers; None means it floats free."
    )
    is_core: bool = Field(
        default=False, description="Core claims live in research_state.core_claims and are guarded."
    )

    language_strength: str | None = Field(
        default=None,
        description="Calibrated verb class permitted by evidence_level, e.g. 'ASSOCIATION'.",
    )
    interpretation_ids: list[str] = Field(default_factory=list)
    audit_ids: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: str = "human"
    task_id: str | None = None

    superseded_by: str | None = None
    rejection_reason: str | None = None
    rejection_basis: RejectionBasis | None = None
    rejected_at: datetime | None = None
    tombstone: bool = Field(
        default=False,
        description="A retained record of a dead claim. Guarded: cannot be deleted (invariant 10).",
    )

    history: list[ClaimHistoryEntry] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------ validators

    @model_validator(mode="after")
    def _check_status_obligations(self) -> "Claim":
        status = self.status

        if status in (ClaimStatus.SUPPORTED, ClaimStatus.ROBUST) and not self.evidence_ids:
            raise ValueError(
                f"claim {self.claim_id}: status {status.value} requires at least one evidence_id "
                "(invariant 4: every claim has evidence)"
            )
        if status in (ClaimStatus.TESTED, ClaimStatus.SUPPORTED, ClaimStatus.ROBUST):
            if not self.supporting_experiments and not self.contradicting_experiments:
                raise ValueError(
                    f"claim {self.claim_id}: status {status.value} requires at least one "
                    "experiment reference (supporting or contradicting)"
                )
        if status is ClaimStatus.REJECTED:
            if not self.rejection_reason:
                raise ValueError(
                    f"claim {self.claim_id}: a rejected claim must state rejection_reason"
                )
            if self.rejection_basis is None:
                raise ValueError(
                    f"claim {self.claim_id}: a rejected claim must state rejection_basis"
                )
            if (
                self.rejection_basis is RejectionBasis.CONTRADICTED
                and not self.contradicting_experiments
                and not self.evidence_ids
            ):
                raise ValueError(
                    f"claim {self.claim_id}: rejection_basis=CONTRADICTED requires contradicting "
                    "experiments or evidence"
                )
        if status is ClaimStatus.SUPERSEDED and not self.superseded_by:
            raise ValueError(
                f"claim {self.claim_id}: status SUPERSEDED requires superseded_by"
            )
        if status is not ClaimStatus.REJECTED and self.rejection_basis is not None:
            raise ValueError(
                f"claim {self.claim_id}: rejection_basis is only meaningful for REJECTED claims"
            )
        return self

    # ------------------------------------------------------------------ helpers

    @property
    def is_live(self) -> bool:
        """A claim that may still be used to justify paper language."""
        return self.status not in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED)

    @property
    def promotable_to_paper(self) -> bool:
        return self.status in (ClaimStatus.SUPPORTED, ClaimStatus.ROBUST)

    def evidence_ids_required(self) -> set[str]:
        return set(self.evidence_ids)

    def record(
        self,
        to_status: ClaimStatus,
        *,
        by: str,
        reason: str = "",
        evidence_ids: list[str] | None = None,
        str_id: str | None = None,
    ) -> ClaimHistoryEntry:
        """Append a history entry. Callers must go through ``claims.lifecycle`` to mutate status."""
        entry = ClaimHistoryEntry(
            by=by,
            from_status=self.status,
            to_status=to_status,
            reason=reason,
            evidence_ids=list(evidence_ids or []),
            str_id=str_id,
        )
        self.history = [*self.history, entry]
        return entry


class Interpretation(RosModel):
    """An *approved* reading of evidence. The paper compiler consumes interpretations, not raw prose."""

    interpretation_id: str = Field(default_factory=lambda: new_id("analysis"))
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    ruled_out: list[str] = Field(default_factory=list)
    approved_by: str | None = Field(
        default=None, description="None means not yet approved and therefore unusable in the paper."
    )
    approved_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    confidence_note: str | None = None

    @property
    def is_approved(self) -> bool:
        return self.approved_by is not None
