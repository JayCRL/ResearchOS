"""Paper compilation: turn research state into a grounded artifact, not free-form text.

Package layout:
    compiler.py      — evidence-grounded compilation of sections into a PaperArtifact
    grounding.py     — the gates: numbers, claims, citations and language must all resolve
    style_audit.py   — deterministic AI-style / overclaim auditor
    readiness.py     — the seven separate readiness dimensions (never one total score)
"""

from .compiler import (
    DEFAULT_SECTIONS,
    PRINTABLE_FIELDS,
    CompilationContext,
    PaperCompiler,
    TextProposer,
    compile_paper,
)
from .grounding import (
    extract_numerals,
    verify_citations,
    verify_claims,
    verify_language,
    verify_numbers,
    verify_paper,
)
from .readiness import ReadinessAssessor
from .style_audit import StyleAuditor

__all__ = [
    "CompilationContext",
    "DEFAULT_SECTIONS",
    "PRINTABLE_FIELDS",
    "PaperCompiler",
    "ReadinessAssessor",
    "StyleAuditor",
    "TextProposer",
    "compile_paper",
    "extract_numerals",
    "verify_citations",
    "verify_claims",
    "verify_language",
    "verify_numbers",
    "verify_paper",
]
