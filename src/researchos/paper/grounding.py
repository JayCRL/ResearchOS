"""Paper grounding: the gates that make a paper a *compilation* rather than a generation.

Three independent gates run over the rendered text:

1. **numbers** — every numeral must resolve to a ``NumberRef`` taken from an analysis artifact,
2. **citations** — every prior-work statement must resolve to a literature claim with a locator,
   and mechanism-level statements additionally require a verified full text,
3. **claims and language** — every load-bearing sentence must resolve to an approved claim whose
   evidence level permits its wording.

A violation is not a warning. It blocks compilation, because a paper that fails these gates is not
a paper — it is a draft with unknown provenance.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from ..kernel.kernel import ResearchKernel
from ..models.common import ClaimStatus, EvidenceLevel, Severity
from ..models.paper import (
    CitationRef,
    GroundedSentence,
    GroundingReport,
    GroundingViolation,
    GroundingViolationCode,
    NumberRef,
    PaperArtifact,
    PaperSection,
)
from ..claims.language import minimum_level_for, overreaches

#: Numerals in prose. Deliberately broad: a false negative here would let an invented number through.
_NUMERAL = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(%|percent)?")

#: Numerals that are structural rather than scientific and never need an artifact.
STRUCTURAL_NUMERALS: frozenset[str] = frozenset(
    {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "100"}
)

#: Words that reveal a mechanism-level statement about prior work.
MECHANISM_WORDS: tuple[str, ...] = (
    "mechanism", "how it works", "internally", "internally computes", "the method computes",
    "the architecture", "is implemented as", "consists of", "relies on an internal",
)


def extract_numerals(text: str) -> list[tuple[float, str, int]]:
    """Return ``(value, raw, offset)`` for every numeral in ``text``."""
    out: list[tuple[float, str, int]] = []
    for match in _NUMERAL.finditer(text):
        try:
            out.append((float(match.group(1)), match.group(0).strip(), match.start()))
        except ValueError:  # pragma: no cover - regex guarantees a float
            continue
    return out


def _allowed_lookup(numbers: Iterable[NumberRef]) -> list[NumberRef]:
    return list(numbers)


def _is_identifier_numeral(sentence: str, offset: int, raw: str) -> bool:
    """Numerals inside identifiers (``gpt2``, ``wikitext-103``, ``124M``) are names, not measurements.

    Without this rule every model and dataset name would be reported as an ungrounded number, which
    would train users to ignore the gate — the worst possible outcome for a guard.
    """
    before = sentence[offset - 1] if offset > 0 else " "
    after = sentence[offset + len(raw)] if offset + len(raw) < len(sentence) else " "
    if before.isalpha():
        return True
    if before in "-_/" and offset >= 2 and sentence[offset - 2].isalnum():
        return True
    if after.isalpha():
        return True
    return False


def _numeral_is_structural(raw: str, sentence: str, offset: int) -> bool:
    """Structural numerals: section references, seeds, "two arms", and so on.

    These still have to be *justified by the paper's own structure*, so we only exempt the small
    integers that appear in ordinary scientific connective text.
    """
    value = raw.strip().rstrip("%").rstrip()
    if value not in STRUCTURAL_NUMERALS:
        return False
    context = sentence[max(0, offset - 24) : offset + len(raw) + 24].lower()
    structural_markers = (
        "section", "sec.", "figure", "fig.", "table", "appendix", "equation", "eq.",
        "arm", "arms", "seed", "seeds", "step", "steps", "epoch", "epochs", "layer", "layers",
        "of the", "two", "three", "four", "five", "both",
    )
    return any(marker in context for marker in structural_markers)


def verify_numbers(
    sentences: Sequence[GroundedSentence],
    numbers: Sequence[NumberRef],
    *,
    tolerance: float = 0.0,
) -> list[GroundingViolation]:
    """Every numeral in every sentence must match a declared ``NumberRef``.

    This is the gate that makes *"the writer cannot invent a number"* mechanical rather than
    aspirational.
    """
    violations: list[GroundingViolation] = []
    pool = _allowed_lookup(numbers)
    declared_ids = {n.number_id for n in pool}

    for sentence in sentences:
        # Bibliographic numerals (years, volumes, DOIs) and appendix sentences — which quote evidence
        # records verbatim and carry their evidence ids — are not paper measurements and are not
        # subject to the number gate. Everything a *result* sentence says still is.
        if sentence.section in (PaperSection.REFERENCES, PaperSection.APPENDIX):
            continue
        declared_here = {n.number_id for n in sentence.numbers}
        for number in sentence.numbers:
            if number.number_id not in declared_ids:
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.UNRESOLVED_NUMBER,
                        severity=Severity.BLOCKER,
                        message=(
                            f"number {number.number_id} is attached to a sentence but is not in the "
                            "compilation's declared number pool"
                        ),
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                        expected=f"analysis {number.analysis_id}:{number.result_id}",
                        suggestion="Recompile: the number pool must be built from analysis artifacts.",
                    )
                )
        for value, raw, offset in extract_numerals(sentence.text):
            if any(n.matches(value) or abs(n.value - value) <= tolerance for n in sentence.numbers):
                continue
            if any(abs(n.value - value) <= n.tolerance for n in pool if n.number_id in declared_here):
                continue
            if _is_identifier_numeral(sentence.text, offset, raw):
                continue
            if _numeral_is_structural(raw, sentence.text, offset):
                continue
            violations.append(
                GroundingViolation(
                    code=GroundingViolationCode.UNGROUNDED_NUMBER,
                    severity=Severity.BLOCKER,
                    message=(
                        f"numeral {raw!r} in this sentence has no analysis artifact behind it. "
                        "Numbers in a ResearchOS paper are references, not values."
                    ),
                    section=sentence.section,
                    sentence_id=sentence.sentence_id,
                    quote=sentence.text[:240],
                    found=raw,
                    expected="a NumberRef from an analysis artifact",
                    suggestion=(
                        "Remove the number, or add the artifact that produced it "
                        "(researchos analysis compute / researchos experiment register)."
                    ),
                )
            )
    return violations


def verify_citations(
    sentences: Sequence[GroundedSentence],
    literature_claims: Mapping[str, object],
    papers: Mapping[str, object],
) -> list[GroundingViolation]:
    """Every citation must resolve to a literature claim with a locator, and full text for mechanisms."""
    violations: list[GroundingViolation] = []
    for sentence in sentences:
        for citation in sentence.citations:
            claim = literature_claims.get(citation.literature_claim_id)
            if claim is None:
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.CITATION_UNRESOLVED,
                        severity=Severity.BLOCKER,
                        message=(
                            f"citation {citation.literature_claim_id} does not resolve to a "
                            "literature claim"
                        ),
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                        quote=sentence.text[:240],
                        expected=f"literature claim {citation.literature_claim_id}",
                        suggestion="Create the literature claim with a page/section locator first.",
                    )
                )
                continue
            if not (citation.page or citation.section or citation.quote):
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.CITATION_SOURCE_MISSING,
                        severity=Severity.BLOCKER,
                        message=f"citation {citation.literature_claim_id} has no page/section locator",
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                        suggestion="Locate the statement in the source (page or section).",
                    )
                )
            paper = papers.get(citation.paper_id)
            lowered = sentence.text.lower()
            if (
                any(word in lowered for word in MECHANISM_WORDS)
                and paper is not None
                and not getattr(paper, "supports_mechanism_claims", lambda: False)()
            ):
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.CITATION_FULLTEXT_REQUIRED,
                        severity=Severity.BLOCKER,
                        message=(
                            f"sentence makes a mechanism-level statement about {citation.paper_id}, "
                            "but that paper's full text has not been verified"
                        ),
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                        quote=sentence.text[:240],
                        expected="fulltext_status=FULLTEXT_VERIFIED",
                        suggestion=(
                            "Read the full text and record page-level literature claims, or remove the "
                            "mechanism-level statement."
                        ),
                    )
                )
    return violations


def verify_claims(
    sentences: Sequence[GroundedSentence],
    claims: Mapping[str, object],
    evidence: Mapping[str, object],
) -> list[GroundingViolation]:
    """Load-bearing sentences must rest on approved claims with evidence; dead claims are refused."""
    violations: list[GroundingViolation] = []
    for sentence in sentences:
        if not sentence.is_load_bearing and not sentence.claim_ids:
            continue
        if sentence.is_load_bearing and not (
            sentence.claim_ids or sentence.evidence_ids or sentence.interpretation_ids
        ):
            violations.append(
                GroundingViolation(
                    code=GroundingViolationCode.UNGROUNDED_CLAIM,
                    severity=Severity.BLOCKER,
                    message=(
                        "load-bearing sentence carries no claim, evidence or approved interpretation"
                    ),
                    section=sentence.section,
                    sentence_id=sentence.sentence_id,
                    quote=sentence.text[:240],
                    suggestion="Ground the sentence or mark it as non-load-bearing background.",
                )
            )
        for claim_id in sentence.claim_ids:
            claim = claims.get(claim_id)
            if claim is None:
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.UNGROUNDED_CLAIM,
                        severity=Severity.BLOCKER,
                        message=f"sentence cites claim {claim_id} which does not exist",
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                    )
                )
                continue
            status = claim.status if isinstance(claim.status, ClaimStatus) else ClaimStatus(str(claim.status))
            if status in (ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED):
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.REJECTED_CLAIM_CITED,
                        severity=Severity.BLOCKER,
                        message=(
                            f"sentence cites claim {claim_id}, which is {status.value}. Retired claims "
                            "must not re-enter a paper."
                        ),
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                        quote=sentence.text[:240],
                        suggestion="Use the claim that superseded it, or remove the sentence.",
                    )
                )
            elif status not in (ClaimStatus.SUPPORTED, ClaimStatus.ROBUST):
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.CLAIM_NOT_APPROVED,
                        severity=Severity.BLOCKER,
                        message=(
                            f"sentence cites claim {claim_id} with status {status.value}; only "
                            "SUPPORTED or ROBUST claims may carry paper language"
                        ),
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                        expected="SUPPORTED or ROBUST",
                        found=status.value,
                        suggestion=(
                            "Promote the claim through the lifecycle (evidence + analysis + human "
                            "approval) or state it as an open question in Limitations."
                        ),
                    )
                )
            missing = [e for e in claim.evidence_ids if e not in evidence]
            if missing:
                violations.append(
                    GroundingViolation(
                        code=GroundingViolationCode.UNGROUNDED_CLAIM,
                        severity=Severity.BLOCKER,
                        message=f"claim {claim_id} references missing evidence {missing}",
                        section=sentence.section,
                        sentence_id=sentence.sentence_id,
                    )
                )
    return violations


def verify_language(
    sentences: Sequence[GroundedSentence],
    levels: Mapping[str, EvidenceLevel],
) -> list[GroundingViolation]:
    """Language strength may never exceed the evidence level behind the sentence."""
    violations: list[GroundingViolation] = []
    for sentence in sentences:
        if not sentence.is_load_bearing:
            continue
        level = levels.get(sentence.sentence_id)
        if level is None:
            level = max(
                (levels.get(cid, EvidenceLevel.L0_IDEA) for cid in sentence.claim_ids),
                key=lambda l: l.rank,
                default=sentence.language_level,
            )
        if overreaches(sentence.text, level):
            minimum = minimum_level_for(sentence.text)
            violations.append(
                GroundingViolation(
                    code=GroundingViolationCode.LANGUAGE_EXCEEDS_EVIDENCE,
                    severity=Severity.HIGH,
                    message=(
                        f"sentence language requires {minimum.value} but the evidence is {level.value}"
                    ),
                    section=sentence.section,
                    sentence_id=sentence.sentence_id,
                    quote=sentence.text[:240],
                    expected=level.value,
                    found=minimum.value,
                    suggestion=(
                        "Weaken the wording to the permitted verb class for this evidence level "
                        "(researchos claim audit shows the calibrated phrasing)."
                    ),
                )
            )
    return violations


def verify_paper(
    paper: PaperArtifact,
    *,
    kernel: ResearchKernel,
    numbers: Sequence[NumberRef] | None = None,
    levels: Mapping[str, EvidenceLevel] | None = None,
) -> GroundingReport:
    """Run every gate over a compiled paper and return the report (never raises)."""
    sentences = paper.all_sentences()
    claims = {c.claim_id: c for c in kernel.claims.all()}
    evidence = {e.evidence_id: e for e in kernel.evidence.all()}
    literature_claims = {c.literature_claim_id: c for c in kernel.literature_claims.all()}
    papers = {p.paper_id: p for p in kernel.papers.all()}

    number_pool = list(numbers) if numbers is not None else [
        n for sentence in sentences for n in sentence.numbers
    ]
    level_map = dict(levels or {})
    if not level_map:
        for claim in claims.values():
            level_map[claim.claim_id] = claim.evidence_level

    violations = [
        *verify_numbers(sentences, number_pool),
        *verify_claims(sentences, claims, evidence),
        *verify_citations(sentences, literature_claims, papers),
        *verify_language(sentences, level_map),
    ]
    report = GroundingReport(
        sentences_checked=len(sentences),
        load_bearing_sentences=sum(1 for s in sentences if s.is_load_bearing),
        numbers_checked=sum(len(extract_numerals(s.text)) for s in sentences),
        numbers_unmatched=sum(
            1 for v in violations if v.code is GroundingViolationCode.UNGROUNDED_NUMBER
        ),
        citations_checked=sum(len(s.citations) for s in sentences),
        citations_unresolved=sum(
            1
            for v in violations
            if v.code in (GroundingViolationCode.CITATION_UNRESOLVED, GroundingViolationCode.CITATION_SOURCE_MISSING)
        ),
        violations=violations,
    )
    paper.grounding_report = report
    return report
