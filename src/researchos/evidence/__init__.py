"""Evidence OS — the registry, verification and the provenance graph.

See :mod:`researchos.evidence.registry`. The split is intentional: the registry owns mutation and
the graph owns read-only traceability, so an audit can walk the graph without any ability to change it.
"""

from .registry import (
    CEILING_BY_PRINCIPAL,
    IMMUTABLE_TYPES,
    EvidenceGraph,
    EvidenceRegistry,
)

__all__ = [
    "CEILING_BY_PRINCIPAL",
    "IMMUTABLE_TYPES",
    "EvidenceGraph",
    "EvidenceRegistry",
]
