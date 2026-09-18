"""The human-in-the-loop actions the dashboard may perform.

The rule this module exists to keep: **a UI is not a second authority.** Every action here runs
through the *same* kernel gate as the CLI, as the ``human`` principal, with the same obligations
enforced. Nothing here can bypass a capability check, promote a claim whose evidence is missing, or
approve a stale transition.

Actions are:

* **disabled by default** (``researchos api --enable-actions`` turns them on),
* **localhost-only** when enabled,
* **audited**: each one appends a ``ui.action`` event naming the action and the client address, so the
  event log distinguishes a click from a CLI command.

Only three action families are exposed, because they are the ones a human must do anyway: deciding the
review queue produced by Research Import, approving or rejecting a filed state transition, and
re-verifying evidence by re-hashing its artifacts. Everything else stays in the CLI, where the
destination of a mutation is explicit.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..kernel.kernel import ResearchKernel
from ..kernel.errors import ResearchOSError

#: Addresses treated as local. A dashboard bound to localhost is a personal tool; exposing write
#: actions to the network would need authentication, which this module deliberately does not invent.
LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})


class ReviewDecisionBody(BaseModel):
    review_item_id: str
    decision: str = Field(description="ACCEPTED | EDITED | REJECTED | DEFERRED")
    note: str = ""


class TransitionDecisionBody(BaseModel):
    str_id: str
    note: str = ""


class TransitionRejectBody(BaseModel):
    str_id: str
    reason: str = ""


class EvidenceVerifyBody(BaseModel):
    evidence_id: str
    method: str = "re-verify artifacts by re-hashing them (dashboard)"


def _guard(request: Request, enabled: bool) -> None:
    if not enabled:
        raise HTTPException(
            status_code=403,
            detail=(
                "write actions are disabled. Start the server with `researchos api --enable-actions` "
                "to decide review items, approve transitions and verify evidence from the dashboard. "
                "All other mutations belong to the CLI, where the acting principal is explicit."
            ),
        )
    client = request.client.host if request.client else ""
    if client not in LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail=f"write actions are restricted to local clients; request came from {client!r}",
        )


def build_actions_router(kernel: ResearchKernel, *, enabled: bool) -> APIRouter:
    """Build the optional action router. When disabled, every route answers 403 with guidance."""
    router = APIRouter(prefix="/actions", tags=["actions"])

    def _audit(request: Request, action: str, payload: dict[str, Any]) -> None:
        kernel.events.append(
            "ui.action",
            actor=kernel.human().name,
            payload={
                "action": action,
                "client": request.client.host if request.client else "unknown",
                "origin": "dashboard",
                **payload,
            },
        )

    @router.get("/status")
    def status(request: Request) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "local_only": True,
            "available": [
                "review.decide",
                "transition.approve",
                "transition.reject",
                "evidence.verify",
            ],
            "note": (
                "actions run as the human principal through the same kernel gate as the CLI; the "
                "dashboard has no privileges of its own"
            ),
        }

    @router.post("/review/decide")
    def decide_review(request: Request, body: ReviewDecisionBody) -> dict[str, Any]:
        _guard(request, enabled)
        from ..models.common import utcnow
        from ..models.review import ReviewDecision

        item = kernel.reviews.get(body.review_item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"review item {body.review_item_id} not found")
        try:
            decision = ReviewDecision(body.decision.upper())
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"unknown decision {body.decision!r}; expected one of ACCEPTED, EDITED, REJECTED, DEFERRED",
            )
        updated = item.with_updates(
            status=decision,
            decided_by=kernel.human().name,
            decided_at=utcnow(),
            decision_note=body.note,
        )
        kernel.reviews.save(updated)
        _audit(request, "review.decide", {"review_item_id": body.review_item_id, "decision": decision.value})
        return {"review_item_id": updated.review_item_id, "status": updated.status.value}

    @router.post("/transition/approve")
    def approve_transition(request: Request, body: TransitionDecisionBody) -> dict[str, Any]:
        _guard(request, enabled)
        try:
            approved, decision = kernel.transitions.approve(kernel.human(), body.str_id, note=body.note)
        except ResearchOSError as exc:
            # the kernel's refusal is the answer, not an internal error: surface it verbatim
            raise HTTPException(status_code=409, detail=str(exc))
        _audit(request, "transition.approve", {"str_id": body.str_id, "decision_id": decision.decision_id})
        return {
            "str_id": approved.str_id,
            "status": approved.status.value,
            "applied_revision": approved.applied_revision,
            "decision_id": decision.decision_id,
        }

    @router.post("/transition/reject")
    def reject_transition(request: Request, body: TransitionRejectBody) -> dict[str, Any]:
        _guard(request, enabled)
        try:
            rejected = kernel.transitions.reject(kernel.human(), body.str_id, reason=body.reason)
        except ResearchOSError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        _audit(request, "transition.reject", {"str_id": body.str_id})
        return {"str_id": rejected.str_id, "status": rejected.status.value}

    @router.post("/evidence/verify")
    def verify_evidence(request: Request, body: EvidenceVerifyBody) -> dict[str, Any]:
        _guard(request, enabled)
        from ..evidence import EvidenceRegistry

        try:
            verified = EvidenceRegistry(kernel).verify(
                kernel.human(), body.evidence_id, method=body.method
            )
        except ResearchOSError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        _audit(
            request,
            "evidence.verify",
            {"evidence_id": body.evidence_id, "status": verified.verification_status.value},
        )
        return {
            "evidence_id": verified.evidence_id,
            "verification_status": verified.verification_status.value,
            "evidence_type": verified.evidence_type.value,
            "notes": verified.verification_notes[:5],
        }

    return router
