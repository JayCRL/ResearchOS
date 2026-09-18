"""Research Import — Research Archaeology for material that already exists.

``researchos import <path>`` reconstructs the research state (question, claims, experiments,
evidence, literature, decisions, rejected claims, conflicts, open questions) from a directory or
repository, and puts everything uncertain into a human review queue.

It never approves a claim, and it never summarises the material into a paper.
"""

from .facts import ExtractionOutput, Fact, FactKind, FileKind, SourceFile
from .pipeline import ImportPipeline, ImportResult
from .report import ImportConfidence, ImportReport
from .scanner import ScanResult, scan

__all__ = [
    "ExtractionOutput",
    "Fact",
    "FactKind",
    "FileKind",
    "ImportConfidence",
    "ImportPipeline",
    "ImportReport",
    "ImportResult",
    "ScanResult",
    "SourceFile",
    "scan",
]
