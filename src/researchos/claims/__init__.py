"""Claim OS: registry, lifecycle, language calibration.

Invariants owned here:

* a claim may not skip a rung (``HYPOTHESIS → SUPPORTED`` is refused),
* every claim that is ``SUPPORTED`` or above has evidence, an analysis artifact and no unresolved
  conflict,
* a rejected claim is a tombstone — retained, explainable, and never resurrected,
* language strength is capped by evidence level, and calibration only ever weakens.
"""

from .language import (
    LANGUAGE_CLASS_MIN_LEVEL,
    VERB_BY_LEVEL,
    WEAKENING_RULES,
    calibrate,
    detect_drift,
    language_class,
    minimum_level_for,
    overreach_explanation,
    overreaches,
    permitted_verb,
    summarise_levels,
)
from .lifecycle import (
    CAPABILITY_BY_TARGET,
    HUMAN_GATED_TARGETS,
    LEGAL_TRANSITIONS,
    ROBUST_MIN_EXPERIMENTS,
    ROBUST_MIN_LEVEL,
    SUPPORTED_MIN_LEVEL,
    ClaimLifecycle,
)
from .registry import PAPER_READY_STATUSES, RETIRED_STATUSES, ClaimRegistry

__all__ = [
    "CAPABILITY_BY_TARGET",
    "ClaimLifecycle",
    "ClaimRegistry",
    "HUMAN_GATED_TARGETS",
    "LANGUAGE_CLASS_MIN_LEVEL",
    "LEGAL_TRANSITIONS",
    "PAPER_READY_STATUSES",
    "RETIRED_STATUSES",
    "ROBUST_MIN_EXPERIMENTS",
    "ROBUST_MIN_LEVEL",
    "SUPPORTED_MIN_LEVEL",
    "VERB_BY_LEVEL",
    "WEAKENING_RULES",
    "calibrate",
    "detect_drift",
    "language_class",
    "minimum_level_for",
    "overreach_explanation",
    "overreaches",
    "permitted_verb",
    "summarise_levels",
]
