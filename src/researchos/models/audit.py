"""Audit and conflict objects — the system's ability to disagree with itself, in writing."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .common import (
    AuditKind,
    ConflictKind,
    ConflictResolution,
    RosModel,
    Severity,
    SourceKind,
    SOURCE_TRUST,
    StrEnum,
    new_id,
    utcnow,
)


class AuditVerdict(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class Finding(RosModel):
    """One issue found by an auditor. Findings never mutate state; they inform transitions."""

    finding_id: str = Field(default_factory=lambda: new_id("finding"))
    code: str = Field(min_length=1, description="Machine-readable code, e.g. UNSUPPORTED_CAUSAL_LANGUAGE.")
    severity: Severity = Severity.MEDIUM
    message: str = Field(min_length=1)
    target_ref: str | None = Field(default=None, description="What is being criticised.")
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    literature_claim_ids: list[str] = Field(default_factory=list)
    quote: str | None = Field(default=None, max_length=2000)
    location: str | None = None
    suggestion: str | None = None
    blocks_publication: bool = False
    deterministic: bool = Field(
        default=True, description="False when a heuristic/LLM judgement was involved."
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    details: dict[str, object] = Field(default_factory=dict)


class Audit(RosModel):
    """A completed audit. Audits are immutable records: a re-run produces a *new* audit."""

    audit_id: str = Field(default_factory=lambda: new_id("audit"))
    kind: AuditKind
    title: str = Field(min_length=1)
    subjects: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""
    verdict: AuditVerdict = AuditVerdict.INCONCLUSIVE

    inputs_hashed: dict[str, str] = Field(
        default_factory=dict, description="artifact path -> sha256 at audit time; makes re-runs comparable."
    )
    deterministic: bool = True
    tool: str | None = Field(default=None, description="Which auditor produced this, with version.")
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "auditor"
    task_id: str | None = None
    respects_state: bool = Field(
        default=True, description="Auditors report; they do not silently change official state."
    )

    @model_validator(mode="after")
    def _check(self) -> "Audit":
        if self.verdict is AuditVerdict.PASS and any(
            f.blocks_publication for f in self.findings
        ):
            raise ValueError(
                f"audit {self.audit_id}: cannot PASS while a blocking finding is present"
            )
        if not self.respects_state:
            raise ValueError(
                f"audit {self.audit_id}: an auditor must not mutate official state; report instead"
            )
        return self

    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.blocks_publication]

    def worst_severity(self) -> Severity:
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.BLOCKER]
        if not self.findings:
            return Severity.INFO
        return max((f.severity for f in self.findings), key=order.index)

    def counts_by_severity(self) -> dict[str, int]:
        out: dict[str, int] = {s.value: 0 for s in Severity}
        for finding in self.findings:
            out[finding.severity.value] += 1
        return out


class ConflictSource(RosModel):
    kind: SourceKind
    ref: str = Field(min_length=1, description="File path, evidence id, experiment id or prose location.")
    value: object | None = None
    artifact_hash: str | None = None
    locator: str | None = None
    captured_at: datetime = Field(default_factory=utcnow)

    @property
    def trust(self) -> int:
        return SOURCE_TRUST[self.kind]


class Conflict(RosModel):
    """A disagreement between two sources. Never resolved by deleting one of them.

    The trust order (raw experiment > verified analysis > audit > experiment report > paper prose >
    AI summary) decides the *default* resolution, but the loser is preserved and marked, and
    unresolved conflicts block claim promotion.
    """

    conflict_id: str = Field(default_factory=lambda: new_id("conflict"))
    kind: ConflictKind = ConflictKind.VALUE
    subject: str = Field(min_length=1, description="What the two sources disagree about.")
    source_a: ConflictSource
    source_b: ConflictSource
    difference: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "importer"
    detected_by: str = Field(default="deterministic", description="deterministic | llm_assisted | human")

    resolution: ConflictResolution = ConflictResolution.UNRESOLVED
    resolution_reason: str | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    winning_ref: str | None = None
    superseded_ref: str | None = None
    preserved: bool = Field(
        default=True, description="Guard-adjacent: both sources are retained regardless of resolution."
    )
    claim_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    blocks_claim_promotion: bool = Field(default=True)

    @model_validator(mode="after")
    def _check(self) -> "Conflict":
        if not self.preserved:
            raise ValueError(
                f"conflict {self.conflict_id}: resolution never deletes a source "
                "(research history is retained)"
            )
        if self.resolution is ConflictResolution.UNRESOLVED and not self.blocks_claim_promotion:
            raise ValueError(
                f"conflict {self.conflict_id}: an unresolved conflict keeps blocking claim promotion"
            )
        if self.resolution in (ConflictResolution.A_WINS, ConflictResolution.B_WINS):
            if not self.resolution_reason or not self.resolved_by:
                raise ValueError(
                    f"conflict {self.conflict_id}: a decided conflict needs resolution_reason and resolved_by"
                )
            if self.winning_ref is None or self.superseded_ref is None:
                raise ValueError(
                    f"conflict {self.conflict_id}: a decided conflict must name winning_ref and "
                    "superseded_ref (marked, not deleted)"
                )
        if self.resolution in (ConflictResolution.BOTH_VALID, ConflictResolution.MERGED) and not self.resolution_reason:
            raise ValueError(f"conflict {self.conflict_id}: state why both sources can stand")
        return self

    def preferred_by_trust(self) -> ConflictSource | None:
        """Default resolution by the deterministic trust ranking; ``None`` when the trust ties."""
        if self.source_a.trust == self.source_b.trust:
            return None
        return self.source_a if self.source_a.trust > self.source_b.trust else self.source_b

    def is_unresolved(self) -> bool:
        return self.resolution is ConflictResolution.UNRESOLVED
