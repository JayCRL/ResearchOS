"""Evidence OS: the registry, verification, and the provenance graph.

Invariants owned here:

* **raw evidence is immutable** — the registry has no update path for it at all,
* **verification means re-hashing**, so an artifact that changed after registration surfaces as
  ``HASH_MISMATCH`` instead of a silently stale "verified" record,
* an unverifiable artifact produces a conflict, not a warning that everyone ignores.
"""

from __future__ import annotations

from typing import Sequence

from ..kernel.errors import EvidenceError, PermissionDenied, VerificationError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..kernel.provenance import verify_artifacts
from ..models.audit import ConflictSource
from ..models.common import (
    ConflictKind,
    EvidenceLevel,
    EvidenceType,
    SourceKind,
    VerificationStatus,
    utcnow,
)
from ..models.evidence import Evidence
from ..models.timeline import TimelineEventKind

#: The highest evidence maturity each principal may create. Nobody self-certifies.
CEILING_BY_PRINCIPAL: dict[str, EvidenceType] = {
    "human": EvidenceType.CLAIMED,
    "importer": EvidenceType.ANALYZED,
    "experiment": EvidenceType.RAW,
    "analysis": EvidenceType.VERIFIED,
    "literature_researcher": EvidenceType.ANALYZED,
    "novelty_auditor": EvidenceType.ANALYZED,
    "claim_manager": EvidenceType.ANALYZED,
    "system": EvidenceType.RAW,
}

#: Evidence that may never be modified once written.
IMMUTABLE_TYPES: frozenset[EvidenceType] = frozenset({EvidenceType.RAW})


class EvidenceRegistry:
    """Create, link and verify evidence. Never edit raw observations."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ------------------------------------------------------------------ create

    def create(self, principal: Principal, evidence: Evidence) -> Evidence:
        principal.require(Cap.EVIDENCE_CREATE, "evidence.create")
        if evidence.evidence_type is EvidenceType.RAW:
            principal.require(Cap.EVIDENCE_RAW_WRITE, "evidence.create:raw")
        if evidence.evidence_type is EvidenceType.VERIFIED:
            principal.require(Cap.EVIDENCE_VERIFY, "evidence.create:verified")

        ceiling = CEILING_BY_PRINCIPAL.get(principal.name, EvidenceType.ANALYZED)
        if evidence.evidence_type.rank > ceiling.rank:
            raise EvidenceError(
                f"principal {principal.name!r} may create evidence up to {ceiling.value}, not "
                f"{evidence.evidence_type.value}. Verification is a separate, auditable step.",
                principal=principal.name,
                resource=evidence.evidence_type.value,
            )
        if evidence.evidence_type is EvidenceType.RAW:
            evidence = evidence.with_updates(immutable=True)
        self.kernel.evidence.save(evidence)
        self.kernel.events.append(
            "evidence.created",
            actor=principal.name,
            task_id=evidence.task_id,
            payload={
                "evidence_id": evidence.evidence_id,
                "type": evidence.evidence_type.value,
                "level": evidence.evidence_level.value,
                "source_experiment": evidence.source_experiment,
                "source_analysis": evidence.source_analysis,
                "artifacts": [a.path for a in evidence.artifacts],
                "content_hash": evidence.content_hash()[:16],
            },
        )
        return evidence

    # ------------------------------------------------------------------ verify

    def verify(
        self,
        principal: Principal,
        evidence_id: str,
        *,
        method: str,
        notes: Sequence[str] = (),
        level: EvidenceLevel | None = None,
    ) -> Evidence:
        """Verify evidence by re-hashing its artifacts and recording who did it and how."""
        principal.require(Cap.EVIDENCE_VERIFY, f"evidence:{evidence_id}:verify")
        evidence = self.kernel.evidence.get(evidence_id)
        if evidence is None:
            raise EvidenceError(f"evidence {evidence_id!r} not found")
        if evidence.immutable:
            # Verification *appends* verification metadata; it never rewrites the observation.
            pass

        results = verify_artifacts(self.kernel.root, evidence.artifacts)
        mismatches = [(ref, status, detail) for ref, status, detail in results if status is not VerificationStatus.VERIFIED]
        notes = list(notes)
        for ref, status, detail in mismatches:
            notes.append(f"{status.value}: {detail}")

        if mismatches and not evidence.artifacts:
            status = VerificationStatus.MISSING_ARTIFACT
        elif mismatches:
            status = VerificationStatus.HASH_MISMATCH
        elif evidence.artifacts:
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.PARTIAL

        updated = evidence.with_updates(
            evidence_type=(
                evidence.evidence_type
                if evidence.evidence_type.rank >= EvidenceType.VERIFIED.rank
                else EvidenceType.VERIFIED
            ),
            verification_status=status,
            verified_by=principal.name,
            verified_at=utcnow(),
            verification_method=method,
            verification_notes=notes,
            evidence_level=level or evidence.evidence_level,
        )
        self.kernel.evidence.save(updated)

        for ref, _status, detail in mismatches:
            self.kernel.ledger.create(
                principal,
                kind=ConflictKind.PROVENANCE,
                subject=f"artifact {ref.path} for evidence {evidence_id}",
                source_a=ConflictSource(
                    kind=SourceKind.RAW_EXPERIMENT,
                    ref=f"{ref.path} as recorded",
                    value=ref.sha256,
                    artifact_hash=ref.sha256,
                ),
                source_b=ConflictSource(
                    kind=SourceKind.RAW_EXPERIMENT,
                    ref=f"{ref.path} as found on disk",
                    value=None,
                    artifact_hash=None,
                ),
                difference=(
                    f"artifact hash mismatch: {detail or 'recorded hash does not match the file'}. "
                    "The bytes backing this evidence changed after it was registered."
                ),
                evidence_ids=[evidence_id],
                auto_resolve_by_trust=False,
            )

        self.kernel.events.append(
            "evidence.verified",
            actor=principal.name,
            task_id=evidence.task_id,
            payload={
                "evidence_id": evidence_id,
                "status": status.value,
                "method": method,
                "mismatches": [ref.path for ref, _s, _d in mismatches],
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.VERIFICATION,
            f"Evidence {evidence_id} verification: {status.value}",
            detail=method,
            actor=principal.name,
            task_id=evidence.task_id,
            refs=[evidence_id],
            state_revision=self.kernel.state.revision(),
        )
        return updated

    # ------------------------------------------------------------------ linking

    def link_to_claim(
        self, principal: Principal, evidence_id: str, claim_id: str, *, note: str = ""
    ) -> Evidence:
        """Attach evidence to a claim. The claim keeps its own history of what supports it."""
        principal.require(Cap.EVIDENCE_CREATE, f"evidence:{evidence_id}:link")
        evidence = self.kernel.evidence.get(evidence_id)
        claim = self.kernel.claims.get(claim_id)
        if evidence is None or claim is None:
            raise EvidenceError(f"cannot link: evidence={evidence_id!r} claim={claim_id!r}")
        if claim.claim_id not in evidence.claim_ids:
            evidence = evidence.with_updates(claim_ids=[*evidence.claim_ids, claim_id])
            self.kernel.evidence.save(evidence)
        if evidence_id not in claim.evidence_ids:
            claim = claim.with_updates(
                evidence_ids=[*claim.evidence_ids, evidence_id],
                updated_at=utcnow(),
            )
            self.kernel.claims.save(claim)
        self.kernel.events.append(
            "evidence.linked",
            actor=principal.name,
            payload={"evidence_id": evidence_id, "claim_id": claim_id, "note": note[:200]},
        )
        return evidence

    # ------------------------------------------------------------------ guard

    def update(self, principal: Principal, evidence: Evidence, *, reason: str) -> Evidence:
        """Update is only legal for non-raw, non-verified evidence. Everything else is append-only."""
        existing = self.kernel.evidence.get(evidence.evidence_id)
        if existing is None:
            raise EvidenceError(f"evidence {evidence.evidence_id!r} not found")
        if existing.evidence_type in IMMUTABLE_TYPES or existing.immutable:
            raise EvidenceError(
                f"evidence {existing.evidence_id} is {existing.evidence_type.value} and immutable: "
                f"{reason or 'no reason given'}. Append a new evidence record that supersedes it "
                "(`supersedes=[...]`) so the original observation remains inspectable.",
                principal=principal.name,
                resource=existing.evidence_id,
            )
        if existing.content_hash() != evidence.content_hash():
            raise EvidenceError(
                f"refusing to change the scientific content of evidence {existing.evidence_id} "
                f"({reason or 'no reason given'}); content changes require a new record",
                principal=principal.name,
                resource=existing.evidence_id,
            )
        self.kernel.evidence.save(evidence)
        self.kernel.events.append(
            "evidence.updated",
            actor=principal.name,
            payload={"evidence_id": evidence.evidence_id, "reason": reason[:300]},
        )
        return evidence

    # ------------------------------------------------------------------ queries

    def for_claim(self, claim_id: str) -> list[Evidence]:
        return [e for e in self.kernel.evidence.all() if claim_id in e.claim_ids]

    def for_experiment(self, experiment_id: str) -> list[Evidence]:
        return [e for e in self.kernel.evidence.all() if experiment_id in e.experiment_ids or e.source_experiment == experiment_id]

    def unverified(self) -> list[Evidence]:
        return [
            e
            for e in self.kernel.evidence.all()
            if e.verification_status is not VerificationStatus.VERIFIED
        ]

    def hash_mismatches(self) -> list[tuple[str, str, VerificationStatus]]:
        """Every evidence artifact whose bytes no longer match the recorded digest."""
        out: list[tuple[str, str, VerificationStatus]] = []
        for evidence in self.kernel.evidence.all():
            for path, status in evidence.verify_artifacts(self.kernel.root):
                out.append((evidence.evidence_id, path, status))
        return out

    def summary(self) -> dict[str, object]:
        items = self.kernel.evidence.all()
        by_type: dict[str, int] = {t.value: 0 for t in EvidenceType}
        by_status: dict[str, int] = {s.value: 0 for s in VerificationStatus}
        by_level: dict[str, int] = {}
        for item in items:
            by_type[item.evidence_type.value] += 1
            by_status[item.verification_status.value] += 1
            by_level[item.evidence_level.value] = by_level.get(item.evidence_level.value, 0) + 1
        return {
            "total": len(items),
            "by_type": by_type,
            "by_verification": by_status,
            "by_level": by_level,
            "immutable": sum(1 for i in items if i.immutable),
            "with_provenance": sum(1 for i in items if i.provenance.code_commit or i.provenance.timestamp),
            "unverified": len(self.unverified()),
        }


class EvidenceGraph:
    """The traceability chain: claim → analysis → experiment → raw artifact → code/config."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel
        self.registry = EvidenceRegistry(kernel)

    def trace_claim(self, claim_id: str) -> dict[str, object]:
        claim = self.kernel.claims.get(claim_id)
        if claim is None:
            raise EvidenceError(f"claim {claim_id!r} not found")
        chains: list[dict[str, object]] = []
        dangling: list[str] = []
        for evidence_id in claim.evidence_ids:
            evidence = self.kernel.evidence.get(evidence_id)
            if evidence is None:
                dangling.append(f"evidence {evidence_id} referenced by claim but missing")
                continue
            analysis = (
                self.kernel.analyses.get(evidence.source_analysis)
                if evidence.source_analysis
                else None
            )
            if evidence.source_analysis and analysis is None:
                dangling.append(f"analysis {evidence.source_analysis} referenced by {evidence_id} but missing")
            experiment = (
                self.kernel.experiments.get(evidence.source_experiment)
                if evidence.source_experiment
                else None
            )
            if evidence.source_experiment and experiment is None:
                dangling.append(
                    f"experiment {evidence.source_experiment} referenced by {evidence_id} but missing"
                )
            artifacts = [a.path for a in evidence.artifacts]
            if experiment:
                artifacts.extend(a.path for a in experiment.raw_artifacts)
            chains.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": evidence_id,
                    "evidence_type": evidence.evidence_type.value,
                    "evidence_level": evidence.evidence_level.value,
                    "verification": evidence.verification_status.value,
                    "analysis_id": evidence.source_analysis,
                    "experiment_id": evidence.source_experiment,
                    "artifacts": sorted(set(artifacts)),
                    "provenance": {
                        "code_commit": evidence.provenance.code_commit,
                        "config_hash": evidence.provenance.config_hash,
                        "seeds": evidence.provenance.random_seeds,
                        "missing": evidence.provenance.missing(),
                    },
                }
            )
        return {
            "claim_id": claim_id,
            "status": claim.status.value,
            "chains": chains,
            "dangling": dangling,
            "complete": bool(chains) and not dangling,
        }

    def dataset_lineage(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for experiment in self.kernel.experiments.all():
            out.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "title": experiment.title,
                    "code_commit": experiment.provenance.code_commit or experiment.code_commit,
                    "config_hash": experiment.config_hash or experiment.provenance.config_hash,
                    "dataset": experiment.dataset,
                    "dataset_hash": experiment.dataset_hash,
                    "seeds": experiment.seeds,
                    "raw_artifacts": [a.path for a in experiment.raw_artifacts],
                    "analysis_artifacts": [a.path for a in experiment.analysis_artifacts],
                    "provenance_completeness": experiment.provenance.completeness(),
                    "provenance_missing": experiment.provenance.missing(),
                }
            )
        return out

    def orphan_evidence(self) -> list[str]:
        """Evidence that no claim, experiment or analysis references — either incomplete work or noise."""
        out: list[str] = []
        for evidence in self.kernel.evidence.all():
            if not evidence.claim_ids and not evidence.source_experiment and not evidence.source_analysis:
                out.append(evidence.evidence_id)
        return out

    def integrity(self) -> dict[str, object]:
        mismatches = self.registry.hash_mismatches()
        return {
            "evidence": len(self.kernel.evidence.all()),
            "orphans": self.orphan_evidence(),
            "hash_mismatches": [
                {"evidence_id": eid, "path": path, "status": status.value}
                for eid, path, status in mismatches
            ],
            "claims_without_evidence": [
                c.claim_id for c in self.kernel.claims.all() if not c.evidence_ids
            ],
            "ok": not mismatches and not self.orphan_evidence(),
        }
