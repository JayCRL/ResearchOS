"""Claim registry queries: what is live, what is usable in a paper, what still needs work."""

from __future__ import annotations

from typing import Sequence

from ..kernel.kernel import ResearchKernel
from ..models.claim import Claim
from ..models.common import ClaimStatus, EvidenceLevel
from .language import calibrate, overreaches, permitted_verb

#: Statuses whose claims may appear in a compiled paper.
PAPER_READY_STATUSES: frozenset[ClaimStatus] = frozenset(
    {ClaimStatus.SUPPORTED, ClaimStatus.ROBUST}
)

#: Statuses that are recorded history rather than live research.
RETIRED_STATUSES: frozenset[ClaimStatus] = frozenset(
    {ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED}
)


class ClaimRegistry:
    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ------------------------------------------------------------------ queries

    def all(self) -> list[Claim]:
        return self.kernel.claims.all()

    def by_status(self, *statuses: ClaimStatus) -> list[Claim]:
        wanted = set(statuses)
        return [c for c in self.all() if c.status in wanted]

    def live(self) -> list[Claim]:
        return [c for c in self.all() if c.status not in RETIRED_STATUSES]

    def retired(self) -> list[Claim]:
        return [c for c in self.all() if c.status in RETIRED_STATUSES]

    def paper_ready(self) -> list[Claim]:
        return [c for c in self.all() if c.status in PAPER_READY_STATUSES]

    def core(self) -> list[Claim]:
        state = self.kernel.research_state()
        out: list[Claim] = []
        for claim_id in state.core_claims:
            claim = self.kernel.claims.get(claim_id)
            if claim is not None:
                out.append(claim)
        return out

    def without_evidence(self) -> list[Claim]:
        return [c for c in self.live() if not c.evidence_ids]

    def without_experiments(self) -> list[Claim]:
        return [c for c in self.live() if not c.supporting_experiments]

    def blocked(self) -> list[tuple[Claim, list[str]]]:
        out: list[tuple[Claim, list[str]]] = []
        for claim in self.live():
            conflicts = self.kernel.ledger.blocking_for_claim(claim.claim_id)
            if conflicts:
                out.append((claim, [c.difference for c in conflicts]))
        return out

    # ------------------------------------------------------------------ language safety

    def safe_statement(self, claim: Claim, level: EvidenceLevel) -> str:
        """The claim's statement weakened to what its evidence supports.

        A paper may only print this version — never the original, if the original overreaches.
        """
        return calibrate(claim.statement, level)

    def overclaiming(self, level_by_claim: dict[str, EvidenceLevel] | None = None) -> list[tuple[Claim, str]]:
        """Claims whose wording demands more evidence than they have, with the safe rewrite."""
        out: list[tuple[Claim, str]] = []
        for claim in self.live():
            level = (level_by_claim or {}).get(claim.claim_id)
            if level is None:
                level = claim.evidence_level
            if overreaches(claim.statement, level):
                out.append((claim, calibrate(claim.statement, level)))
        return out

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict[str, object]:
        claims = self.all()
        by_status: dict[str, int] = {s.value: 0 for s in ClaimStatus}
        for claim in claims:
            by_status[claim.status.value] += 1
        return {
            "total": len(claims),
            "by_status": by_status,
            "live": len(self.live()),
            "paper_ready": len(self.paper_ready()),
            "without_evidence": len(self.without_evidence()),
            "without_experiments": len(self.without_experiments()),
            "blocked_by_conflict": len(self.blocked()),
            "core_claims": len(self.kernel.research_state().core_claims),
            "permitted_language": {
                c.claim_id: permitted_verb(c.evidence_level) for c in claims[:20]
            },
        }

    def table(self, claims: Sequence[Claim] | None = None) -> list[tuple[str, str, str, str, int]]:
        """(claim_id, status, level, permitted language, #evidence) for display."""
        rows: list[tuple[str, str, str, str, int]] = []
        for claim in claims if claims is not None else self.all():
            rows.append(
                (
                    claim.claim_id,
                    claim.status.value,
                    claim.evidence_level.value,
                    permitted_verb(claim.evidence_level),
                    len(claim.evidence_ids),
                )
            )
        return rows
