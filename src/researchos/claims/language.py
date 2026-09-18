"""Language calibration: the bridge between an evidence level and the words a paper may use.

The rule is one-directional. Evidence level *caps* language strength; language never raises the
level. :func:`calibrate` therefore only ever weakens.

Why a separate module: the same mapping is needed by the claim lifecycle (to record what a claim is
allowed to say), by the paper compiler (to gate sentences) and by the style auditor (to detect
drift). One table, three consumers, no drift between them.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..models.common import EvidenceLevel

#: The verb class each evidence level permits, and the strongest verb it permits.
VERB_BY_LEVEL: dict[EvidenceLevel, str] = {
    EvidenceLevel.L0_IDEA: "we hypothesise",
    EvidenceLevel.L1_OBSERVATION: "we observe",
    EvidenceLevel.L2_REPRODUCED: "we find",
    EvidenceLevel.L3_CONTROLLED: "we show an association",
    EvidenceLevel.L4_INTERVENTION: "causes",
    EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY: "is necessary for",
    EvidenceLevel.L6_CROSS_SETTING_REPLICATION: "generalises across settings",
}

#: Reporting verbs that may be replaced by the permitted verb class during calibration.
REPORTING_VERBS = re.compile(
    r"\bwe (?:observe|observed|find|found|show|showed|demonstrate|demonstrated|establish|established"
    r"|prove|proved|confirm|confirmed|report|reported|reveal|revealed|indicate|indicated)\b"
    r"|\bour (?:results?|experiments?|findings?) (?:show|shows|indicate|indicates|suggest|suggests|reveal|reveals)\b",
    re.IGNORECASE,
)

#: Language classes that require a minimum level.
LANGUAGE_CLASS_MIN_LEVEL: dict[str, EvidenceLevel] = {
    # A plain descriptive sentence makes no evidential claim of its own (its numbers still have to be
    # grounded, and a load-bearing sentence still has to cite a claim).
    "DESCRIPTION": EvidenceLevel.L0_IDEA,
    "HYPOTHESIS": EvidenceLevel.L0_IDEA,
    "OBSERVATION": EvidenceLevel.L1_OBSERVATION,
    "FINDING": EvidenceLevel.L2_REPRODUCED,
    "ASSOCIATION": EvidenceLevel.L3_CONTROLLED,
    "CAUSATION": EvidenceLevel.L4_INTERVENTION,
    "NECESSITY": EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY,
    "GENERALISATION": EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
}

#: Phrases -> the language class they imply. Ordered strongest first.
LANGUAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    # "establishing a mechanism" needs necessity/sufficiency evidence, not merely an intervention:
    # a controlled comparison shows that something matters, not how it works.
    (
        r"\b(?:establishes?|demonstrates?|shows?|proves?|confirms?|identifies?)\s+the mechanism\b"
        r"|\bproves? the mechanism\b|\bis the mechanism\b"
        r"|\bresponsible for the (?:effect|improvement)\b",
        "NECESSITY",
    ),
    (r"\bis (?:necessary|required|essential) (?:for|to)\b", "NECESSITY"),
    (r"\bis sufficient (?:for|to)\b", "NECESSITY"),
    (r"\b(?:causes?|caused|causing|drives?|driven by|leads? to|led to|results? in|because of|due to)\b", "CAUSATION"),
    (r"\b(?:establishes?|proves?|proven|confirms?|demonstrates?)\b", "CAUSATION"),
    (r"\b(?:generalises|generalizes|holds across|replicates across|across (?:models|scales|settings|datasets|families))\b", "GENERALISATION"),
    (r"\b(?:associations?|correlates? with|correlated with|associated with|co-occurs with|is linked to)\b", "ASSOCIATION"),
    # "we find" is a reproduced finding; "we observe" is a bare observation and must not be upgraded.
    (r"\b(?:we|our results?|our experiments?|our findings?)\s+(?:show|shows|find|finds|demonstrate|demonstrates|indicate|indicates|reveal|reveals)\b", "FINDING"),
    (r"\b(?:we|our results?)\s+(?:observe|observes|notice|notices|see|report)\b", "OBSERVATION"),
    (r"\b(?:we )?(?:hypothesi[sz]e|speculate|suspect|conjecture|propose|expect)\b", "HYPOTHESIS"),
    (r"\bmay\b|\bmight\b|\bcould\b|\bpossibly\b|\bperhaps\b|\bwe are not aware\b", "HYPOTHESIS"),
)

#: Deterministic weakenings applied by :func:`calibrate`, strongest -> weakest.
WEAKENING_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(?:establishes?|demonstrates?|shows?|proves?|confirms?)\s+the mechanism\b", "may bear on the mechanism"),
    (r"\bestablishes the mechanism\b", "is consistent with a role for the mechanism"),
    (r"\bestablishes\b", "provides evidence about"),
    (r"\bproves\b", "supports"),
    (r"\bproven\b", "supported"),
    (r"\bconfirm(?:s|ed)?\b", "is consistent with"),
    (r"\bcauses\b", "is associated with"),
    (r"\bcaused\b", "was associated with"),
    (r"\bcausing\b", "associated with"),
    (r"\bdrives\b", "is associated with"),
    (r"\bdriven by\b", "associated with"),
    (r"\bleads to\b", "is associated with"),
    (r"\bled to\b", "was associated with"),
    (r"\bdemonstrates\b", "shows"),
    (r"\bis necessary for\b", "is associated with"),
    (r"\bis required for\b", "is associated with"),
    (r"\bis sufficient for\b", "is associated with"),
    (r"\bgeneralises across\b", "was tested in"),
    (r"\bcorrelates with\b", "may be related to"),
    (r"\bcorrelated with\b", "may be related to"),
    (r"\bis associated with\b", "may be related to"),
    (r"\bare associated with\b", "may be related to"),
    (r"\bis linked to\b", "may be related to"),
    (r"\bshows? that\b", "suggests that"),
    (r"\bdemonstrating that\b", "suggesting that"),
    (r"\bfundamentally\b", ""),
    (r"\bprofoundly\b", ""),
    (r"\buniversally\b", ""),
    (r"\bcompletely\b", ""),
    (r"\balways\b", "in the settings studied"),
    (r"\bnever\b", "not in the settings studied"),
)


def language_class(text: str) -> str:
    """The strongest language class present in ``text`` (default: DESCRIPTION)."""
    lowered = text.lower()
    for pattern, klass in LANGUAGE_PATTERNS:
        if re.search(pattern, lowered):
            return klass
    return "DESCRIPTION"


def minimum_level_for(text: str) -> EvidenceLevel:
    return LANGUAGE_CLASS_MIN_LEVEL.get(language_class(text), EvidenceLevel.L0_IDEA)


def overreaches(text: str, level: EvidenceLevel) -> bool:
    """True when the sentence's language demands more evidence than ``level`` provides."""
    return minimum_level_for(text).rank > level.rank


def permitted_verb(level: EvidenceLevel) -> str:
    return VERB_BY_LEVEL[level]


def calibrate(text: str, level: EvidenceLevel) -> str:
    """Weaken ``text`` until it no longer overreaches ``level``.

    The function never adds information: it only replaces overclaiming verbs with the class the
    evidence permits, and drops intensifiers that carry no evidential weight.
    """
    if not overreaches(text, level):
        return text
    weakened = text
    for pattern, replacement in WEAKENING_RULES:
        if not overreaches(weakened, level):
            break
        weakened = re.sub(pattern, replacement, weakened, flags=re.IGNORECASE)
    weakened = re.sub(r"\s{2,}", " ", weakened)
    weakened = re.sub(r"\s+([.,;])", r"\1", weakened)
    if overreaches(weakened, level):
        # Final step: replace whatever reporting verb is left with the one this level permits. This is
        # what makes calibration total — every sentence can be brought inside its evidence, and the
        # replacement is always *weaker or equal*, never stronger.
        replaced = REPORTING_VERBS.sub(permitted_verb(level), weakened)
        if replaced != weakened:
            weakened = replaced
        else:
            weakened = re.sub(
                r"\b(?:demonstrates?|shows?|proves?|establishes?|confirms?)\b",
                "is consistent with",
                weakened,
                flags=re.IGNORECASE,
            )
    return weakened.strip()


def overreach_explanation(text: str, level: EvidenceLevel) -> str:
    """A reviewer-facing explanation of *why* a sentence was weakened."""
    klass = language_class(text)
    needed = LANGUAGE_CLASS_MIN_LEVEL.get(klass, EvidenceLevel.L1_OBSERVATION)
    return (
        f"language class {klass} requires {needed.value} but the evidence provides {level.value}; "
        f"the strongest permitted phrasing is {permitted_verb(level)!r}"
    )


def detect_drift(original: str, rewritten: str, level: EvidenceLevel) -> list[str]:
    """Report whether a rewrite made a claim *stronger* than it was — the one forbidden direction."""
    problems: list[str] = []
    if overreaches(original, level) and not overreaches(rewritten, level):
        return problems
    if minimum_level_for(rewritten).rank > minimum_level_for(original).rank:
        problems.append(
            f"rewrite strengthens the language class from {language_class(original)} to "
            f"{language_class(rewritten)}"
        )
    if overreaches(rewritten, level):
        problems.append(f"rewrite still overreaches: {overreach_explanation(rewritten, level)}")
    return problems


def summarise_levels(levels: Sequence[EvidenceLevel]) -> str:
    if not levels:
        return "no evidence"
    strongest = max(levels, key=lambda level: level.rank)
    return f"strongest of {len(levels)} evidence level(s): {strongest.value} -> {permitted_verb(strongest)!r}"
