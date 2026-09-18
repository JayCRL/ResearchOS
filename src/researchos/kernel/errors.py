"""Kernel error taxonomy.

Every error here corresponds to a specific way research software usually goes wrong silently:
an agent overwrote raw data, a local task redefined the project, two writers raced, an audit was
quietly edited, a transition skipped a rung. Making them *loud* is most of the value of the kernel.
"""

from __future__ import annotations


class ResearchOSError(Exception):
    """Base class. Every kernel error carries the principal and resource involved."""

    def __init__(self, message: str, *, principal: str | None = None, resource: str | None = None) -> None:
        super().__init__(message)
        self.principal = principal
        self.resource = resource


class PermissionDenied(ResearchOSError):
    """A principal tried to exercise a capability it does not hold."""


class TaskBoundaryViolation(ResearchOSError):
    """An action was attempted outside the boundaries of the active task."""


class SilentStateChangeError(ResearchOSError):
    """A guarded research-state field was about to change without an approved STR.

    This is the anti-drift error: *"debugging a metric is not a new core research question"*.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str,
        principal: str | None = None,
        task_id: str | None = None,
    ) -> None:
        super().__init__(message, principal=principal, resource=path)
        self.path = path
        self.task_id = task_id

    def remedy(self) -> str:
        return (
            f"File a State Transition Request for {self.path!r} "
            "(`researchos task transition request ...`) and have a human approve it."
        )


class ImmutableResourceError(ResearchOSError):
    """An append-only/immutable resource was mutated. Never allowed, for any principal."""


class StateRevisionConflict(ResearchOSError):
    """Optimistic-concurrency failure: the state changed under the writer."""


class TransitionError(ResearchOSError):
    """An STR or claim-status transition was attempted that the lifecycle forbids."""


class ClaimTransitionError(TransitionError):
    """A claim lifecycle jump that is explicitly disallowed (e.g. HYPOTHESIS -> SUPPORTED)."""


class EvidenceError(ResearchOSError):
    """Evidence was referenced, created or verified in a way the evidence model forbids."""


class VerificationError(ResearchOSError):
    """An artifact failed verification (hash mismatch, missing file)."""


class ProvenanceError(ResearchOSError):
    """Provenance is missing or inconsistent for something that requires it."""


class ConflictError(ResearchOSError):
    """A conflict was created or resolved in a way that would delete or hide a source."""


class GroundingError(ResearchOSError):
    """The paper compiler refused to emit text that is not grounded in verified material."""


class LiteratureError(ResearchOSError):
    """A literature operation violated sourcing or coverage rules."""


class SkillError(ResearchOSError):
    """A skill lifecycle gate was violated (activation without benchmark, sandbox failure, ...)."""


class ImportError_(ResearchOSError):
    """Research Import failed to reconstruct state from the supplied material."""


class ProjectNotFound(ResearchOSError):
    """No ``.researchos/`` project was found at or above the given path."""


class ProjectExists(ResearchOSError):
    """``researchos init`` was run where a project already exists."""
