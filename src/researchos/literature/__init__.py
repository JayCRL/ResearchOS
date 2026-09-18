"""Literature OS — broad, recorded, coverage-measured search.

Not a helper for the paper writer: this package reconstructs prior work, builds the map, runs
novelty audits and keeps every claim about another paper attached to a page or section.
"""

from .coverage import compute_coverage, coverage_report, is_sufficient, required_families, thresholds
from .graph import LiteratureGraph, jaccard, tokens
from .novelty import NoveltyAuditor, build_match, safe_novelty_sentence
from .prior_art import (
    CURRENT_WORK_REF,
    DEFAULT_COLUMNS,
    PriorArtMatrixBuilder,
    cell_provenance,
    coverage_line,
    render_markdown,
)

__all__ = [
    "CURRENT_WORK_REF",
    "DEFAULT_COLUMNS",
    "LiteratureGraph",
    "NoveltyAuditor",
    "PriorArtMatrixBuilder",
    "build_match",
    "cell_provenance",
    "compute_coverage",
    "coverage_line",
    "coverage_report",
    "is_sufficient",
    "jaccard",
    "render_markdown",
    "required_families",
    "safe_novelty_sentence",
    "thresholds",
    "tokens",
]
