"""Deterministic AI-style auditing: "looks like AI wrote it" as a measurable property.

Why this module exists
----------------------
Nothing here is a vibe. Every finding is a rule over the characters on the page plus the research
state that is actually stored, so two runs over identical input produce identical output and a
reviewer can disagree with a *specific rule* instead of with an impression. The auditor never
rewrites a paper and never touches research state: it reports, and the compiler/author decide.

Two scientific rules drive most of the checks:

* **Language strength must never exceed evidence level** (``ARCHITECTURE.md`` §4.8). "X causes Y" is
  a different scientific claim from "X is associated with Y"; only an intervention licenses the
  former. Verb classes are therefore compared against :class:`EvidenceLevel`, not against taste.
* **The paper must tell the real research story.** ``HISTORY_MISMATCH`` compares the prose with the
  recorded :class:`TimelineEvent` values: "as we hypothesised, confirming our hypothesis" is a false
  statement of method whenever the timeline contains an anomaly, a hypothesis revision, a rejected
  claim or a failed experiment.

Design notes
------------
* One :class:`StyleProfile` is built per document and reused by every rule, so sentence splitting
  and tokenisation cannot disagree between rules.
* Findings are sorted by ``(category, position in text)``; identical input therefore yields an
  identical list, and a diff between two audit runs is meaningful.
* ``audit_text`` treats a missing ``evidence_level`` as ``L0_IDEA`` and says so in the message:
  nothing has been shown to license strong language, so the conservative reading is the honest one.

Standard library only. No LLM, no network, no randomness.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..kernel.permissions import Cap, Principal
from ..models import (
    Audit,
    AuditKind,
    AuditVerdict,
    Claim,
    ClaimStatus,
    EvidenceLevel,
    Finding,
    GroundedSentence,
    PaperArtifact,
    PaperSection,
    Severity,
    StyleCategory,
    StyleFinding,
    TimelineEvent,
    TimelineEventKind,
)
from ..models.literature import FORBIDDEN_NOVELTY_PHRASES

#: Version string recorded on every audit produced by this module.
STYLE_AUDIT_VERSION = "1.0.0"
STYLE_AUDIT_TOOL = f"style_audit/{STYLE_AUDIT_VERSION}"

# --------------------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------------------

#: Buzzword hits per 1000 words above which the document is reported as buzzword-inflated.
BUZZWORD_DENSITY_LIMIT: float = 6.0

#: Sentence-length uniformity: below this coefficient of variation, prose is template-flat.
#: Human academic prose typically sits at 0.4–0.6; generated prose clusters near 0.1–0.25.
MIN_SENTENCE_CV: float = 0.25

#: Above this CV the length distribution is so skewed that mean±std describes nothing. We abstain
#: from the uniformity verdict rather than reporting a comparison the data cannot support — and we
#: say so in a finding, because a silent abstention looks like a pass.
MAX_SENTENCE_CV: float = 1.5

#: Hard caps so a pathological document cannot produce an unbounded audit. Both are applied after
#: deterministic sorting, never by sampling.
MAX_FINDINGS_PER_RULE: int = 25
MAX_QUOTE_CHARS: int = 300

#: Findings that block publication, and only at this severity or above.
BLOCKING_CODES: frozenset[str] = frozenset(
    {
        "LANGUAGE_EXCEEDS_EVIDENCE",
        "NOVELTY_UNSUPPORTED",
        "HISTORY_MISMATCH",
        "CAUSAL_WITHOUT_INTERVENTION",
    }
)
BLOCKING_MIN_SEVERITY: Severity = Severity.HIGH

_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.BLOCKER,
)


def _at_least(severity: Severity, floor: Severity) -> bool:
    return _SEVERITY_ORDER.index(severity) >= _SEVERITY_ORDER.index(floor)


# --------------------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------------------

#: Buzzwords, keyed by the canonical word. Values are the surface forms counted as that word.
#: A *density* rule (not a ban) is used because these words are legitimate in moderation; what is
#: measurable is that a document leans on them instead of on specifics.
BUZZWORDS: Mapping[str, tuple[str, ...]] = {
    "novel": ("novel", "novelty", "novelties"),
    "comprehensive": ("comprehensive", "comprehensively", "comprehensiveness"),
    "robust": ("robust", "robustness"),
    "framework": ("framework", "frameworks"),
    "paradigm": ("paradigm", "paradigms", "paradigm shift"),
    "leverage": ("leverage", "leverages", "leveraging", "leveraged"),
    "delve": ("delve", "delves", "delving", "delved"),
    "tapestry": ("tapestry",),
    "landscape": ("landscape", "landscapes"),
    "multifaceted": ("multifaceted",),
    "holistic": ("holistic", "holistically"),
    "seamless": ("seamless", "seamlessly"),
    "cutting-edge": ("cutting-edge", "cutting edge"),
    "pivotal": ("pivotal",),
    "intricate": ("intricate", "intricacies"),
    "realm": ("realm", "realms"),
    "underscore": ("underscore", "underscores", "underscoring", "underscored"),
    "unprecedented": ("unprecedented",),
    "revolutionary": ("revolutionary", "revolutionize", "revolutionizing", "revolutionise"),
    "game-changing": ("game-changing", "game changer", "game-changer"),
    "synergy": ("synergy", "synergies", "synergistic"),
    "transformative": ("transformative",),
    "promising": ("promising",),
    "crucial": ("crucial", "crucially"),
    "breakthrough": ("breakthrough", "breakthroughs"),
    "elegant": ("elegant", "elegantly"),
    "profound": ("profound", "profoundly"),
    "fundamental": ("fundamental", "fundamentally"),
    "emerging": ("emerging",),
    "versatile": ("versatile", "versatility"),
}

#: Clichéd sentence frames that carry no project-specific content. One finding per matched phrase.
TEMPLATE_PHRASES: tuple[str, ...] = (
    "In this paper, we propose",
    "To the best of our knowledge",
    "has been widely studied",
    "have been widely studied",
    "a large body of work",
    "extensive experiments demonstrate",
    "we conduct extensive experiments",
    "achieves state-of-the-art results",
    "opening up new avenues",
    "paves the way for",
    "sheds light on",
    "it is worth noting that",
    "it is important to note that",
    "in the realm of",
    "a growing body of literature",
    "plays a pivotal role",
    "in recent years",
)

#: Phrases that are the strongest cheap signal of machine-written prose. Kept as a separate,
#: explicitly named tuple because these are the ones a reader can point at in a diff.
KNOWN_AI_PHRASES: tuple[str, ...] = (
    "This important finding",
    "significant insights",
    "fundamental challenge",
    "promising results",
    "our novel framework",
    "demonstrates the mechanism",
    "delve into",
    "tapestry of",
    "underscores the importance",
    "in today's rapidly evolving",
    "comprehensive understanding",
    "nuanced understanding",
    "multifaceted challenge",
    "holistic approach",
    "leverage the power of",
    "seamlessly integrates",
    "stands as a testament",
    "it is crucial to note",
    "rich tapestry",
    "opens up exciting",
)

#: Adjectives/adverbs whose use is *conditional*: they assert a judgement (importance, size,
#: novelty) that must be licensed by evidence, so they are reported per occurrence.
RHETORICAL_ADJECTIVES: tuple[str, ...] = (
    "groundbreaking",
    "profound",
    "profoundly",
    "fundamental",
    "fundamentally",
    "crucial",
    "crucially",
    "remarkable",
    "remarkably",
    "novel",
    "novelty",
    "very",
    "highly",
    "significantly",
    "pivotal",
    "transformative",
    "revolutionary",
    "extremely",
    "incredibly",
    "elegant",
    "dramatically",
    "vastly",
)

#: Empty-background forms: sentences that assert a field is important, popular or fast-moving and
#: commit to nothing checkable. Written as regexes so the shapes (not just exact strings) are caught.
EMPTY_BACKGROUND_PATTERNS: tuple[str, ...] = (
    r"\b(?:is|are|was|were|remains?|has become|have become)\s+(?:very\s+|particularly\s+|especially\s+)?"
    r"(?:important|significant|critical|crucial|essential|vital|fundamental|central)\b",
    r"\b(?:has|have)\s+(?:attracted|received|drawn)\s+"
    r"(?:increasing|growing|considerable|significant|much|substantial|a lot of)\s+attention\b",
    r"\bin recent years\b",
    r"\bplays?\s+(?:a|an)\s+(?:crucial|key|important|vital|central|significant|critical)\s+role\b",
    r"\bhas become\s+(?:an?\s+)?(?:increasingly\s+)?"
    r"(?:important|popular|central|essential|critical|active|attractive)\s+"
    r"(?:area|topic|field|research|direction|problem|subject)\b",
    r"\bis\s+(?:an?\s+)?(?:active|important|growing|rapidly evolving|hot|thriving)\s+"
    r"(?:area|topic|field|research area|research direction|line of work)\b",
    r"\bwith the rapid development of\b",
    r"\bhas\s+(?:gained|attracted)\s+(?:widespread|significant|considerable)\s+interest\b",
)

#: Hedges. One hedge is scientific caution; two or more in one sentence is a sentence that says
#: nothing, and the reader cannot tell which of the two the author actually believes.
HEDGES: tuple[str, ...] = (
    "may",
    "might",
    "could",
    "possibly",
    "potentially",
    "perhaps",
    "arguably",
    "seems",
    "seem",
    "seemed",
    "appears",
    "appear",
    "appeared",
    "suggests",
    "suggest",
    "suggested",
    "likely",
    "probably",
    "somewhat",
    "relatively",
    "tends to",
    "tend to",
    "it is possible",
    "to some extent",
    "generally",
    "often",
    "sometimes",
    "we believe",
    "presumably",
    "apparently",
    "in some cases",
    "roughly",
    "approximately",
    "more or less",
)

#: Causal verbs. Each of these asserts that a variable *produces* a change, which the ladder only
#: permits at ``L4_INTERVENTION`` (or above). Phrases deliberately excluded as too weak to be a
#: causal claim: "allows", "makes possible", "is compatible with".
CAUSAL_VERBS: tuple[str, ...] = (
    "causes",
    "cause",
    "caused",
    "causing",
    "is caused by",
    "are caused by",
    "leads to",
    "lead to",
    "led to",
    "leading to",
    "results in",
    "result in",
    "resulted in",
    "resulting in",
    "produces",
    "produce",
    "produced",
    "producing",
    "induces",
    "induce",
    "induced",
    "inducing",
    "drives",
    "drive",
    "drove",
    "driven by",
    "gives rise to",
    "give rise to",
    "gave rise to",
    "is responsible for",
    "are responsible for",
    "is due to",
    "are due to",
    "determines",
    "determine",
    "determined",
    "affects",
    "affect",
    "affected",
    "affecting",
)

#: First-person and absolute novelty framings. Without a coverage-gated ``NoveltyAudit`` in state,
#: none of these is expressible: absence of a match in our search is not absence of prior work.
_FIRST_PERSON_NOVELTY: tuple[str, ...] = (
    "we are the first",
    "we introduce the first",
    "the first to",
    "first work to",
    "first framework to",
    "first method to",
    "first study to",
    "for the first time",
    "no prior work",
    "no existing work",
    "no previous work",
    "nobody has",
    "no one has",
    "never been attempted",
    "never been studied",
    "unprecedented",
    "entirely novel",
    "completely novel",
    "we pioneer",
)

#: Union of our first-person framings and the literature module's forbidden phrases, so the two
#: modules can never drift apart: the paper side and the literature side agree on what is banned.
NOVELTY_PHRASES: tuple[str, ...] = tuple(
    dict.fromkeys((*_FIRST_PERSON_NOVELTY, *FORBIDDEN_NOVELTY_PHRASES))
)

#: The calibrated replacements offered for a novelty claim. Never stronger than the evidence.
CALIBRATED_NOVELTY_FRAMES: tuple[str, ...] = (
    "We are not aware of prior work that reports this exact result; the searched coverage is "
    "stated in the text.",
    "Within the searched coverage, we found no prior work reporting this exact result.",
)

#: Statements that are true of any method and therefore carry no information about *this* project.
GENERIC_STATEMENT_PATTERNS: tuple[str, ...] = (
    r"\bdeep learning (?:models|systems|approaches|methods)\s+(?:learn|require|need|are trained)",
    r"\bneural networks\s+(?:are|can|learn|require|need)\b",
    r"\blanguage models\s+(?:are|have been)\s+(?:trained|pretrained|pre-trained)\b",
    r"\bmachine learning (?:models|algorithms|systems|methods)\s+(?:are|require|learn|need)\b",
    r"\btraining\s+(?:a|the)\s+model\s+(?:requires|needs)\b",
    r"\btransformers\s+(?:are|have)\s+(?:become|been)\b",
    r"\bit is\s+(?:well[- ]known|widely\s+(?:known|accepted|used)|generally\s+(?:known|accepted))\s+that\b",
    r"\bas\s+(?:is|we)\s+(?:well[- ]known|all know)\b",
    r"\bin general,\s+(?:models|methods|systems|networks)\b",
    r"\bmodels\s+(?:tend to|generally)\s+(?:learn|perform|suffer|benefit)\b",
    r"\battention\s+(?:is|has become)\s+(?:a|an|the)\s+(?:key|central|core|important)\s+"
    r"(?:mechanism|component|ingredient|part)\b",
    r"\bdata\s+(?:is|are)\s+(?:essential|important|crucial)\s+(?:for|to)\s+(?:training|learning)\b",
)

#: Confirmation framings. Used only to compare prose against the recorded timeline.
CONFIRMATION_PHRASES: tuple[str, ...] = (
    "as we hypothesised",
    "as we hypothesized",
    "as hypothesised",
    "as hypothesized",
    "confirming our hypothesis",
    "confirms our hypothesis",
    "confirmed our hypothesis",
    "as expected",
    "as we predicted",
    "as predicted",
    "consistent with our hypothesis",
    "consistent with our prediction",
    "exactly as we expected",
)

#: Timeline kinds that make a clean "hypothesis → confirmation" arc a false statement of method.
HISTORY_COUNTER_KINDS: tuple[TimelineEventKind, ...] = (
    TimelineEventKind.ANOMALY,
    TimelineEventKind.CLAIM_REVISION,
    TimelineEventKind.CLAIM_REJECTED,
    TimelineEventKind.EXPERIMENT_FAILED,
)

#: Reasons-why markers. Their *absence* is the finding, so this list has to be about the author's
#: reasoning rather than about reporting: "we ran three seeds" is not a reason.
REASONING_MARKERS: tuple[str, ...] = (
    "because",
    "we expected",
    "we hypothesised",
    "we hypothesized",
    "this surprised us",
    "surprised us",
    "so we",
    "to test whether",
    "to check whether",
    "we revised",
    "we changed",
    "motivated by",
    "the reason",
    "why we",
)

#: Sentence transitions. Rational connectives are fine; *most* sentences starting with one is a
#: structural tell, and it flattens the argument into a list of equally weighted steps.
TRANSITION_WORDS: tuple[str, ...] = (
    "However",
    "Moreover",
    "Furthermore",
    "Additionally",
    "In addition",
    "Therefore",
    "Thus",
    "Consequently",
    "Overall",
    "Notably",
)
TRANSITION_OVERUSE_RATIO: float = 0.4

#: Paragraph shapes considered "listy": runs of identically shaped sentences.
LISTY_RUN_LENGTH: int = 3
MIN_STRUCTURE_SENTENCES: int = 8

#: Similarity thresholds.
PARAPHRASE_JACCARD: float = 0.75
NGRAM_LENGTH: int = 6
REPEATED_OPENING_LENGTH: int = 3
MIN_SENTENCES_FOR_OPENING_RULE: int = 3

#: Claim-restatement detection thresholds (sentence vs claim statement).
RESTATEMENT_MIN_SHARED: int = 3
RESTATEMENT_JACCARD: float = 0.4
RESTATEMENT_CONTAINMENT: float = 0.6

#: Intensifiers stripped by :meth:`StyleAuditor.suggest_weakening`.
INTENSIFIERS: tuple[str, ...] = (
    "groundbreaking",
    "profound",
    "profoundly",
    "fundamental",
    "fundamentally",
    "crucial",
    "crucially",
    "remarkably",
    "remarkable",
    "extremely",
    "incredibly",
    "very",
    "highly",
    "perfectly",
    "truly",
    "novel",
    "novelty",
    "pivotal",
    "transformative",
    "revolutionary",
    "universally",
    "completely",
    "entirely",
    "always",
    "clearly",
    "obviously",
    "undeniably",
    "dramatically",
    "vastly",
)

#: Words that assert an absolute. Removable ones are listed here; "never" is handled separately
#: because deleting it can invert the claim.
ABSOLUTE_WORDS: tuple[str, ...] = (
    "always",
    "never",
    "completely",
    "entirely",
    "universally",
    "undeniably",
    "without exception",
    "in all cases",
    "all models",
    "all methods",
    "all datasets",
    "all settings",
    "all tasks",
    "every model",
    "every method",
    "every dataset",
    "every setting",
    "every case",
)

#: Scope qualifiers. An absolute that is explicitly scoped ("all three seeds we tested") is a
#: precise statement, not an overclaim — precision must not be punished.
_SCOPE_QUALIFIERS: tuple[str, ...] = (
    r"\b(?:we|that we)\s+(?:tested|studied|examined|ran|used|observe|observed|consider|considered|"
    r"search|searched|surveyed|measure|measured|evaluate|evaluated)\b",
    r"\bin (?:our|these|the)\s+(?:experiments|runs|settings|models|datasets|seeds|sample|study|"
    r"evaluation|benchmarks)\b",
    r"\bon (?:this|these|the)\s+(?:dataset|datasets|model|models|task|tasks|split|benchmark|benchmarks)\b",
    r"\bfor the\s+(?:models|datasets|seeds|settings|runs|tasks)\b",
    r"\bwithin\b",
    r"\bamong\b",
    r"\bunder\b",
    r"\bas far as we\b",
    r"\bin the setting\b",
)
_SCOPE_RE = re.compile("|".join(_SCOPE_QUALIFIERS), re.IGNORECASE)

#: Verb classes a calibrated sentence may use, by evidence level.
PERMITTED_VERB_CLASS: Mapping[EvidenceLevel, str] = {
    EvidenceLevel.L0_IDEA: "we hypothesise that",
    EvidenceLevel.L1_OBSERVATION: "we observe that",
    EvidenceLevel.L2_REPRODUCED: "we find that",
    EvidenceLevel.L3_CONTROLLED: "we show an association (we find that ... is associated with)",
    EvidenceLevel.L4_INTERVENTION: "causes",
    EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY: "is necessary for",
    EvidenceLevel.L6_CROSS_SETTING_REPLICATION: "generalises across the settings we tested",
}

#: Level implied by a claim *status*: what the claim's lifecycle permits it to be stated as.
CLAIM_STATUS_LEVEL_CAP: Mapping[ClaimStatus, EvidenceLevel] = {
    ClaimStatus.IDEA: EvidenceLevel.L0_IDEA,
    ClaimStatus.HYPOTHESIS: EvidenceLevel.L1_OBSERVATION,
    ClaimStatus.TESTED: EvidenceLevel.L2_REPRODUCED,
    ClaimStatus.SUPPORTED: EvidenceLevel.L3_CONTROLLED,
    ClaimStatus.ROBUST: EvidenceLevel.L4_INTERVENTION,
    ClaimStatus.WEAKENED: EvidenceLevel.L1_OBSERVATION,
    ClaimStatus.REJECTED: EvidenceLevel.L0_IDEA,
    ClaimStatus.SUPERSEDED: EvidenceLevel.L0_IDEA,
}

#: Placeholders used by :meth:`StyleAuditor.suggest_weakening`. Numbers and citations are *masked*
#: rather than copied (which would smuggle a number past the audit) or deleted (which would silently
#: drop grounding the author must re-attach).
MASK_NUMBER = "<value>"
MASK_CITATION = "<citation>"

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "we", "our", "ours", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "for", "with", "and", "or", "but", "that", "this", "these",
        "those", "it", "its", "as", "by", "at", "from", "than", "then", "there", "here", "their",
        "they", "them", "he", "she", "his", "her", "not", "no", "do", "does", "did", "has",
        "have", "had", "can", "could", "may", "might", "will", "would", "should", "if", "when",
        "while", "which", "who", "whom", "what", "also", "such", "into", "over", "under", "about",
        "more", "most", "less", "up", "out", "so", "because", "between", "both", "each", "other",
    }
)


# --------------------------------------------------------------------------------------
# Text primitives
# --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-_]*")

_ABBREVIATIONS: tuple[str, ...] = (
    "e.g.", "i.e.", "cf.", "et al.", "vs.", "fig.", "figs.", "eq.", "eqs.", "sec.", "secs.",
    "no.", "approx.", "dr.", "prof.", "al.", "ref.", "refs.", "tab.", "tabs.", "app.", "resp.",
)

_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)|\n+")


def _masked_for_splitting(text: str) -> str:
    """Replace periods that are *not* sentence ends with a same-length placeholder.

    Offsets are preserved, so sentences can be sliced out of the original text with their true
    character positions — which is what makes ``location`` stable across runs.
    """
    out = list(text)
    for match in _DECIMAL_RE.finditer(text):
        out[match.start()] = "\x01"
    lowered = text.lower()
    for abbreviation in _ABBREVIATIONS:
        for match in re.finditer(re.escape(abbreviation), lowered):
            index = match.end() - 1
            if index < len(out) and text[index] == ".":
                out[index] = "\x01"
    return "".join(out)


def split_sentences(text: str) -> tuple[tuple[str, int], ...]:
    """Split ``text`` into ``(sentence, start_offset)`` pairs.

    Deliberately conservative: decimals, common abbreviations and bullet/newline boundaries are
    handled, because a sentence splitter that mis-splits would corrupt every downstream rule.
    """
    masked = _masked_for_splitting(text)
    spans: list[tuple[str, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(masked):
        end = match.end()
        chunk = text[start:end]
        stripped = chunk.strip()
        if stripped:
            spans.append((stripped, start + (len(chunk) - len(chunk.lstrip()))))
        start = end
    tail = text[start:]
    if tail.strip():
        spans.append((tail.strip(), start + (len(tail) - len(tail.lstrip()))))
    return tuple(spans)


def word_tokens(text: str) -> tuple[str, ...]:
    """Surface tokens (punctuation dropped) — the basis of every length/word-count statistic."""
    return tuple(_TOKEN_RE.findall(text))


def normalised_tokens(text: str) -> tuple[str, ...]:
    """Lowercased tokens, used for n-grams, openings and similarity."""
    return tuple(token.lower() for token in _TOKEN_RE.findall(text))


def content_tokens(text: str) -> frozenset[str]:
    """Stopword-free token set: similarity is measured on content, not on grammar words."""
    return frozenset(t for t in normalised_tokens(text) if t not in _STOPWORDS)


def _phrase_pattern(phrases: Iterable[str]) -> re.Pattern[str]:
    """Alternation regex for a phrase list; multiword phrases tolerate any whitespace."""
    parts: list[str] = []
    for phrase in sorted(set(phrases), key=len, reverse=True):
        escaped = re.escape(phrase).replace("\\ ", r"\s+").replace(" ", r"\s+")
        parts.append(escaped)
    if not parts:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(parts) + r")(?![A-Za-z])", re.IGNORECASE)


_CAUSAL_RE = _phrase_pattern(CAUSAL_VERBS)
_HEDGE_RE = _phrase_pattern(HEDGES)
_NOVELTY_RE = _phrase_pattern(NOVELTY_PHRASES)
_CONFIRMATION_RE = _phrase_pattern(CONFIRMATION_PHRASES)
_REASONING_RE = _phrase_pattern(REASONING_MARKERS)
_TEMPLATE_RE = _phrase_pattern(TEMPLATE_PHRASES)
_KNOWN_AI_RE = _phrase_pattern(KNOWN_AI_PHRASES)
_ABSOLUTE_RE = _phrase_pattern(ABSOLUTE_WORDS)
_RHETORICAL_RE = _phrase_pattern(RHETORICAL_ADJECTIVES)
_INTENSIFIER_RE = _phrase_pattern(INTENSIFIERS)

_EMPTY_BACKGROUND_RE = re.compile("|".join(EMPTY_BACKGROUND_PATTERNS), re.IGNORECASE)
_GENERIC_RE = re.compile("|".join(GENERIC_STATEMENT_PATTERNS), re.IGNORECASE)

_CITATION_RE = re.compile(
    r"\\cite[a-zA-Z]*\{[^}]*\}"
    r"|\[\d+(?:\s*[,;\u2013-]\s*\d+)*\]"
    r"|\([A-Z][A-Za-z.'\-]+(?:\s+(?:et al\.?|and|&)\s*[A-Za-z.'\-]+)*,?\s*(?:19|20)\d{2}[a-z]?\)"
    r"|\b[A-Z][A-Za-z.'\-]+ et al\.,?\s*(?:19|20)\d{2}"
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:\s*(?:%|percent|x))?")
_ABSOLUTE_REMOVABLE_RE = _phrase_pattern(
    ("always", "completely", "entirely", "universally", "undeniably", "clearly", "obviously")
)
_NEVER_SAFE_RE = re.compile(r"\bnever\s+(been|observed|reported|shown|found|seen)\b", re.IGNORECASE)

# --------------------------------------------------------------------------------------
# Language-strength ladder (text -> minimum evidence level the wording demands)
# --------------------------------------------------------------------------------------

_MECHANISM_RE = re.compile(
    r"\b(?:demonstrates?|establish(?:es)?|reveals?|identif(?:y|ies))\s+the\s+mechanism"
    r"(?:\s+(?:of|behind|by|for))?\b",
    re.IGNORECASE,
)
_NECESSITY_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:necessary|sufficient)\s+(?:for|to)\b", re.IGNORECASE
)
_CONFIRM_RE = re.compile(r"\b(?:confirms?|confirmed|validates?|validated)\b", re.IGNORECASE)
_PROOF_RE = re.compile(
    r"\b(?:proves?|proven|establishes?|established|definitively|conclusively)\b", re.IGNORECASE
)
_SHOW_RE = re.compile(r"\bshows?\b|\bshown\b", re.IGNORECASE)
_DEMONSTRATE_RE = re.compile(r"\bdemonstrates?\b|\bdemonstrated\b|\breveals?\b", re.IGNORECASE)

#: (regex, level that first permits the wording). Ordered strongest first.
_STRENGTH_MARKERS: tuple[tuple[re.Pattern[str], EvidenceLevel], ...] = (
    (_PROOF_RE, EvidenceLevel.L6_CROSS_SETTING_REPLICATION),
    (
        re.compile(r"\buniversally\b|\bwithout exception\b|\bin all cases\b", re.IGNORECASE),
        EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
    ),
    (_NECESSITY_RE, EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY),
    (_MECHANISM_RE, EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY),
    (_CAUSAL_RE, EvidenceLevel.L4_INTERVENTION),
    (_CONFIRM_RE, EvidenceLevel.L4_INTERVENTION),
    (re.compile(r"\b(?:demonstrates?|shown|shows?)\b", re.IGNORECASE), EvidenceLevel.L3_CONTROLLED),
    (
        re.compile(
            r"\b(?:is|are|was|were)\s+associated\s+with\b|\bcorrelates?\s+with\b|"
            r"\bco-?occurs?\s+with\b|\bco-?vary\w*\s+with\b",
            re.IGNORECASE,
        ),
        EvidenceLevel.L3_CONTROLLED,
    ),
    (
        re.compile(
            r"\b(?:finds?|found|reports?|reported|reproduces?|reproduced|replicat\w+)\b", re.IGNORECASE
        ),
        EvidenceLevel.L2_REPRODUCED,
    ),
    (re.compile(r"\b(?:observes?|observed)\b|\bwe\s+see\b", re.IGNORECASE), EvidenceLevel.L1_OBSERVATION),
    (
        re.compile(r"\b(?:hypothes\w+|speculat\w+|conjectur\w+|posits?)\b", re.IGNORECASE),
        EvidenceLevel.L0_IDEA,
    ),
)

#: Strong verbs checked by the OVERCLAIM rule (causal verbs are reported by their own rule so the
#: two findings cannot double-report the same mistake).
_OVERCLAIM_MARKERS: tuple[tuple[re.Pattern[str], EvidenceLevel], ...] = (
    (_PROOF_RE, EvidenceLevel.L6_CROSS_SETTING_REPLICATION),
    (
        re.compile(r"\buniversally\b|\bwithout exception\b|\bin all cases\b", re.IGNORECASE),
        EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
    ),
    (_NECESSITY_RE, EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY),
    (_MECHANISM_RE, EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY),
    (_CONFIRM_RE, EvidenceLevel.L4_INTERVENTION),
)

#: Reporting clauses and the level each verb demands. Used to calibrate and to rewrite.
_CLAUSE_RE = re.compile(
    r"^(?P<lead>We|Our\s+(?:results|findings|analysis|experiments|data|measurements)|"
    r"The\s+results|These\s+results|This\s+work|Our\s+work)\s+(?P<verb>[A-Za-z]+)\b"
)
_CLAUSE_VERB_REQUIREMENT: Mapping[str, EvidenceLevel] = {
    "hypothesise": EvidenceLevel.L0_IDEA,
    "hypothesize": EvidenceLevel.L0_IDEA,
    "speculate": EvidenceLevel.L0_IDEA,
    "expect": EvidenceLevel.L0_IDEA,
    "predict": EvidenceLevel.L0_IDEA,
    "observe": EvidenceLevel.L1_OBSERVATION,
    "see": EvidenceLevel.L1_OBSERVATION,
    "note": EvidenceLevel.L1_OBSERVATION,
    "find": EvidenceLevel.L2_REPRODUCED,
    "found": EvidenceLevel.L2_REPRODUCED,
    "report": EvidenceLevel.L2_REPRODUCED,
    "measure": EvidenceLevel.L2_REPRODUCED,
    "reproduce": EvidenceLevel.L2_REPRODUCED,
    "show": EvidenceLevel.L3_CONTROLLED,
    "demonstrate": EvidenceLevel.L3_CONTROLLED,
    "reveal": EvidenceLevel.L3_CONTROLLED,
    "confirm": EvidenceLevel.L4_INTERVENTION,
    "validate": EvidenceLevel.L4_INTERVENTION,
    "prove": EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
    "establish": EvidenceLevel.L6_CROSS_SETTING_REPLICATION,
}
_TARGET_CLAUSE_VERB: Mapping[EvidenceLevel, str] = {
    EvidenceLevel.L0_IDEA: "hypothesise",
    EvidenceLevel.L1_OBSERVATION: "observe",
    EvidenceLevel.L2_REPRODUCED: "find",
    EvidenceLevel.L3_CONTROLLED: "find",
    EvidenceLevel.L4_INTERVENTION: "find",
    EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY: "find",
    EvidenceLevel.L6_CROSS_SETTING_REPLICATION: "find",
}


def required_evidence_level(text: str) -> EvidenceLevel:
    """The *minimum* evidence level the wording itself demands.

    This is the calibration primitive: "universally" demands cross-setting replication, "causes"
    demands an intervention, "is associated with" demands a controlled comparison. Public because
    readiness and the claim registry need the same ladder as the style rules.
    """
    best = EvidenceLevel.L0_IDEA
    for pattern, level in _STRENGTH_MARKERS:
        if pattern.search(text) and level.rank > best.rank:
            best = level
    return best


def claim_permitted_level(claim: Claim) -> EvidenceLevel:
    """The strongest wording a claim's status *and* evidence level jointly permit."""
    cap = CLAIM_STATUS_LEVEL_CAP.get(claim.status, EvidenceLevel.L0_IDEA)
    return cap if cap.rank <= claim.evidence_level.rank else claim.evidence_level


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _buzzword_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for canonical, variants in BUZZWORDS.items():
        pattern = _phrase_pattern(variants)
        hits = len(pattern.findall(text))
        if hits:
            counts[canonical] = hits
    return counts


def _truncate(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "\u2026"


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


# --------------------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleProfile:
    """A document-level view built once and reused by every rule.

    Rules that agreed on their own private tokenisation would be impossible to reconcile when two
    of them disagreed, so splitting, tokenising and the length distribution all live here.
    """

    sentences: tuple[str, ...]
    words: tuple[str, ...]
    word_count: int
    sentence_lengths: tuple[int, ...]
    mean_sentence_length: float
    sentence_length_cv: float
    conditional_adjectives: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "StyleProfile":
        sentences = tuple(sentence for sentence, _ in split_sentences(text))
        words = normalised_tokens(text)
        lengths = tuple(len(word_tokens(sentence)) for sentence in sentences)
        mean = (sum(lengths) / len(lengths)) if lengths else 0.0
        if len(lengths) >= 2 and mean > 0:
            cv = statistics.pstdev(lengths) / mean
        else:
            cv = 0.0
        adjectives: list[str] = []
        for adjective in RHETORICAL_ADJECTIVES:
            if adjective not in adjectives and _phrase_pattern((adjective,)).search(text):
                adjectives.append(adjective)
        return cls(
            sentences=sentences,
            words=words,
            word_count=len(words),
            sentence_lengths=lengths,
            mean_sentence_length=round(mean, 4),
            sentence_length_cv=round(cv, 6),
            conditional_adjectives=tuple(adjectives),
        )


@dataclass(frozen=True)
class _Candidate:
    """An internal finding plus the position used for deterministic ordering."""

    category: StyleCategory
    code: str
    severity: Severity
    message: str
    position: int
    quote: str | None = None
    location: str | None = None
    section: PaperSection | None = None
    suggested_rewrite: str | None = None
    evidence_level: EvidenceLevel | None = None
    claim_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def to_finding(self) -> StyleFinding:
        return StyleFinding(
            category=self.category,
            code=self.code,
            severity=self.severity,
            message=self.message,
            quote=_truncate(self.quote) if self.quote else None,
            location=self.location,
            section=self.section,
            suggested_rewrite=self.suggested_rewrite,
            evidence_level=self.evidence_level,
        )


def _sort_key(candidate: _Candidate) -> tuple[str, int, str, str]:
    return (candidate.category.value, candidate.position, candidate.code, candidate.quote or "")


# --------------------------------------------------------------------------------------
# The auditor
# --------------------------------------------------------------------------------------


class StyleAuditor:
    """Deterministic style and calibration auditor.

    The auditor is stateless apart from configuration, so the same instance may be reused and two
    runs over identical input are guaranteed to produce identical findings.
    """

    def __init__(self, *, buzzword_limit: float | None = None) -> None:
        self.buzzword_limit = float(BUZZWORD_DENSITY_LIMIT if buzzword_limit is None else buzzword_limit)

    # ------------------------------------------------------------------ public API

    def audit_text(
        self,
        text: str,
        *,
        section: PaperSection | None = None,
        evidence_level: EvidenceLevel | None = None,
        claim_texts: Sequence[str] = (),
        timeline: Sequence[TimelineEvent] = (),
    ) -> list[StyleFinding]:
        """Audit free text (or a rendered section).

        ``evidence_level=None`` is read as ``L0_IDEA`` and stated in the finding messages: with no
        supplied level, nothing licences strong wording, and pretending otherwise would defeat the
        purpose of the audit.
        """
        profile = StyleProfile.from_text(text)
        offsets = tuple(offset for _, offset in split_sentences(text))
        level = evidence_level if evidence_level is not None else EvidenceLevel.L0_IDEA
        levels = tuple(level for _ in profile.sentences)
        return self._audit(
            profile,
            offsets=offsets,
            section=section,
            levels=levels,
            sentence_claims=tuple(() for _ in profile.sentences),
            permitted=levels,
            claim_texts=tuple(claim_texts),
            timeline=tuple(timeline),
            level_was_supplied=evidence_level is not None,
        )

    def audit_sentences(
        self,
        sentences: Sequence[GroundedSentence],
        *,
        claim_levels: Mapping[str, EvidenceLevel] | None = None,
        timeline: Sequence[TimelineEvent] = (),
    ) -> list[StyleFinding]:
        """Audit compiled sentences, each with *its own* declared language level.

        A sentence's permitted strength is the weakest level among the claims it cites (from
        ``claim_levels``), falling back to its own ``language_level``. That is what makes
        ``CLAIM_STRENGTH_DRIFT`` checkable: the sentence is compared with the claim it invokes.
        """
        sentences = list(sentences)
        text = " \n".join(sentence.text for sentence in sentences)
        profile = StyleProfile.from_text(text)
        offsets_list: list[int] = []
        cursor = 0
        for sentence in sentences:
            offsets_list.append(cursor)
            cursor += len(sentence.text) + 2
        offsets = tuple(offsets_list)
        levels = tuple(sentence.language_level for sentence in sentences)

        known = dict(claim_levels or {})
        permitted_list: list[EvidenceLevel] = []
        for sentence, own in zip(sentences, levels):
            referenced = [known[cid] for cid in sentence.claim_ids if cid in known]
            if referenced:
                permitted_list.append(min(referenced, key=lambda level: level.rank))
            else:
                permitted_list.append(own)
        permitted = tuple(permitted_list)

        return self._audit(
            profile,
            offsets=offsets,
            section=None,
            levels=levels,
            sentence_claims=tuple(tuple(sentence.claim_ids) for sentence in sentences),
            permitted=permitted,
            claim_texts=(),
            timeline=tuple(timeline),
            level_was_supplied=True,
            sections=tuple(sentence.section for sentence in sentences),
            sentence_numbers=True,
        )

    def audit_paper(
        self,
        principal: Principal,
        paper: PaperArtifact,
        *,
        kernel: "object | None" = None,
        claims: Sequence[Claim] = (),
        timeline: Sequence[TimelineEvent] = (),
        task_id: str | None = None,
    ) -> Audit:
        """Audit a whole :class:`PaperArtifact` and persist the resulting audit.

        The audit is saved through ``kernel.save_audit`` when a kernel is supplied, so the style
        verdict becomes part of the replayable project history. The paper itself is never modified:
        an auditor reports, it does not rewrite.
        """
        principal.require(Cap.PAPER_STYLE_AUDIT, "paper.style_audit")

        claim_levels: dict[str, EvidenceLevel] = {
            claim.claim_id: claim_permitted_level(claim) for claim in claims
        }
        candidates: list[_Candidate] = []
        section_order: dict[PaperSection, int] = {}
        for index, draft in enumerate(paper.sections):
            section_order.setdefault(draft.section, index)

        if paper.title.strip():
            candidates.extend(
                self._with_rank(
                    self.audit_text(paper.title, section=PaperSection.TITLE, timeline=timeline), 0, 0
                )
            )
        if paper.abstract.strip():
            candidates.extend(
                self._with_rank(
                    self.audit_text(
                        paper.abstract,
                        section=PaperSection.ABSTRACT,
                        timeline=timeline,
                        claim_texts=tuple(claim.statement for claim in claims),
                    ),
                    1,
                    0,
                )
            )

        for index, draft in enumerate(paper.sections):
            rank = index + 2
            findings = self.audit_sentences(
                draft.sentences, claim_levels=claim_levels, timeline=timeline
            )
            for finding in findings:
                candidates.append(
                    _Candidate(
                        category=finding.category,
                        code=finding.code,
                        severity=finding.severity,
                        message=finding.message,
                        position=rank * 100000 + self._location_position(finding.location),
                        quote=finding.quote,
                        location=finding.location,
                        section=draft.section,
                        suggested_rewrite=finding.suggested_rewrite,
                        evidence_level=finding.evidence_level,
                        claim_ids=self._claims_for_quote(draft.sentences, finding),
                        evidence_ids=self._evidence_for_quote(draft.sentences, finding),
                    )
                )
            if draft.sentences:
                text = " ".join(sentence.text for sentence in draft.sentences)
                for finding in self.audit_text(text, section=draft.section, timeline=timeline):
                    if finding.code in {
                        "UNIFORM_SENTENCE_LENGTH",
                        "SENTENCE_UNIFORMITY_UNRELIABLE",
                        "LISTY_PARAGRAPH",
                        "TRICOLON",
                        "TRANSITION_OVERUSE",
                        "BUZZWORD_DENSITY",
                        "REPEATED_NGRAM",
                        "REPEATED_SENTENCE_OPENING",
                        "PARAPHRASE_LOOP",
                        "MISSING_RESEARCHER_REASONING",
                        "HISTORY_MISMATCH",
                    }:
                        candidates.append(
                            _Candidate(
                                category=finding.category,
                                code=finding.code,
                                severity=finding.severity,
                                message=finding.message,
                                position=rank * 100000 + 9999,
                                quote=finding.quote,
                                location=finding.location,
                                section=draft.section,
                                suggested_rewrite=finding.suggested_rewrite,
                                evidence_level=finding.evidence_level,
                            )
                        )

        candidates.sort(key=_sort_key)
        audit_findings = [self._to_audit_finding(candidate, paper) for candidate in candidates]
        blockers = [finding for finding in audit_findings if finding.blocks_publication]
        verdict = (
            AuditVerdict.FAIL
            if blockers
            else (AuditVerdict.WARN if audit_findings else AuditVerdict.PASS)
        )
        audit = Audit(
            kind=AuditKind.STYLE,
            title=f"AI-style audit of {paper.title or paper.paper_id}",
            subjects=[paper.paper_id, *paper.claim_ids],
            findings=audit_findings,
            summary=(
                f"{len(audit_findings)} style findings "
                f"({len(blockers)} blocking) over {len(paper.sections)} sections, "
                f"{paper.word_count()} words"
            ),
            verdict=verdict,
            deterministic=True,
            tool=STYLE_AUDIT_TOOL,
            created_by=principal.name,
            task_id=task_id,
        )
        saver = getattr(kernel, "save_audit", None)
        if callable(saver):
            saver(principal, audit)
        return audit

    def density(self, text: str) -> dict[str, float]:
        """Buzzword density in *this* text — reported, never assumed."""
        counts = _buzzword_counts(text)
        words = len(normalised_tokens(text))
        hits = sum(counts.values())
        return {
            "words": float(words),
            "buzzword_hits": float(hits),
            "per_1000_words": round(1000.0 * hits / words, 4) if words else 0.0,
            "limit_per_1000_words": self.buzzword_limit,
        }

    def suggest_weakening(self, text: str, *, level: EvidenceLevel) -> str:
        """Return a *usable* sentence whose language the given level actually permits.

        Guarantees, in order of importance:

        * no intensifier from :data:`INTENSIFIERS` survives;
        * verb classes above the level are replaced by the level's permitted class
          (``we observe`` / ``we find`` / ``we show an association`` / ``causes`` / ``is necessary for``);
        * no number, citation or mechanism is **introduced** (numbers and citations already present
          are masked as ``<value>``/``<citation>``: copying them would smuggle a numeral past the
          audit, deleting them would silently drop grounding the author must re-attach);
        * the sentence stays readable — an unreadable suggestion is never adopted.
        """
        had_clause = bool(_CLAUSE_RE.match(text.strip()))
        needs_framing = required_evidence_level(text).rank > level.rank

        out = _CITATION_RE.sub(MASK_CITATION, text)
        out = _NUMBER_RE.sub(MASK_NUMBER, out)

        out = self._downgrade_clause(out, level)
        out = self._downgrade_verbs(out, level)
        out = self._strip_intensifiers(out)
        out = self._soften_absolutes(out)
        out = self._tidy(out)

        if needs_framing and not had_clause and out:
            lead = _TARGET_CLAUSE_VERB[level]
            if not out.lstrip().lower().startswith(("we ", "our ", "the ", "this ")):
                out = f"We {lead} that {out[0].lower()}{out[1:]}"
            else:
                lowered = out[0].lower() + out[1:]
                out = f"We {lead} that {lowered}"
            out = self._tidy(out)
        return out

    # ------------------------------------------------------------------ rule driver

    def _audit(
        self,
        profile: StyleProfile,
        *,
        offsets: tuple[int, ...],
        section: PaperSection | None,
        levels: tuple[EvidenceLevel, ...],
        sentence_claims: tuple[tuple[str, ...], ...],
        permitted: tuple[EvidenceLevel, ...],
        claim_texts: tuple[str, ...],
        timeline: tuple[TimelineEvent, ...],
        level_was_supplied: bool,
        sections: tuple[PaperSection, ...] | None = None,
        sentence_numbers: bool = False,
    ) -> list[StyleFinding]:
        candidates: list[_Candidate] = []
        for index, sentence in enumerate(profile.sentences):
            level = levels[index] if index < len(levels) else EvidenceLevel.L0_IDEA
            permit = permitted[index] if index < len(permitted) else level
            claims = sentence_claims[index] if index < len(sentence_claims) else ()
            position = offsets[index] if index < len(offsets) else 0
            location = (
                f"sentence {index + 1}"
                if sentence_numbers
                else f"chars {position}-{position + len(sentence)}"
            )
            candidates.extend(
                self._sentence_rules(
                    sentence,
                    position=position,
                    location=location,
                    section=section,
                    level=level,
                    permitted=permit,
                    claim_ids=claims,
                    level_was_supplied=level_was_supplied,
                )
            )
        candidates.extend(
            self._document_rules(
                profile,
                offsets=offsets,
                section=section,
                sections=sections,
                timeline=timeline,
            )
        )
        candidates.extend(
            self._restatement_rules(
                profile,
                offsets=offsets,
                section=section,
                claim_texts=claim_texts,
                sentence_numbers=sentence_numbers,
            )
        )
        candidates.sort(key=_sort_key)
        return [candidate.to_finding() for candidate in candidates]

    # ------------------------------------------------------------------ sentence rules

    def _sentence_rules(
        self,
        sentence: str,
        *,
        position: int,
        location: str,
        section: PaperSection | None,
        level: EvidenceLevel,
        permitted: EvidenceLevel,
        claim_ids: tuple[str, ...],
        level_was_supplied: bool,
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        out.extend(self._rule_empty_background(sentence, position, location, section))
        out.extend(self._rule_template_phrase(sentence, position, location, section))
        out.extend(
            self._rule_causal(sentence, position, location, section, level, level_was_supplied)
        )
        out.extend(self._rule_novelty(sentence, position, location, section))
        out.extend(self._rule_overclaim(sentence, position, location, section, level))
        out.extend(self._rule_generic(sentence, position, location, section))
        out.extend(self._rule_rhetorical(sentence, position, location, section))
        out.extend(
            self._rule_claim_drift(
                sentence,
                position=position,
                location=location,
                section=section,
                permitted=permitted,
                claim_ids=claim_ids,
            )
        )
        out.extend(self._rule_hedge(sentence, position, location, section))
        return out

    def _rule_empty_background(
        self, sentence: str, position: int, location: str, section: PaperSection | None
    ) -> list[_Candidate]:
        match = _EMPTY_BACKGROUND_RE.search(sentence)
        if not match:
            return []
        grounded = _has_citation(sentence) or _has_number(sentence)
        severity = Severity.LOW if grounded else Severity.MEDIUM
        note = (
            "It carries a citation or a numeral, so it is at least anchored."
            if grounded
            else "It cites nothing and states no numeral, so the reader cannot check it."
        )
        return [
            _Candidate(
                category=StyleCategory.EMPTY_BACKGROUND,
                code="EMPTY_BACKGROUND_SENTENCE",
                severity=severity,
                message=(
                    f"Empty background: '{match.group(0)}' asserts that the topic matters without "
                    f"committing to a checkable claim. {note}"
                ),
                position=position + match.start(),
                quote=sentence,
                location=location,
                section=section,
                suggested_rewrite=(
                    "Replace it with the specific observation that motivated this work, including "
                    "what was measured and where the reader can verify it."
                ),
            )
        ]

    def _rule_template_phrase(
        self, sentence: str, position: int, location: str, section: PaperSection | None
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        for phrase in TEMPLATE_PHRASES:
            match = _phrase_pattern((phrase,)).search(sentence)
            if match:
                out.append(
                    _Candidate(
                        category=StyleCategory.TEMPLATE_PHRASE,
                        code="TEMPLATE_PHRASE",
                        severity=Severity.LOW,
                        message=f"Template phrase '{match.group(0)}'.",
                        position=position + match.start(),
                        quote=sentence,
                        location=location,
                        section=section,
                        suggested_rewrite=(
                            "State what was done or found instead of framing the section; delete "
                            "the frame rather than paraphrasing it."
                        ),
                    )
                )
        for phrase in KNOWN_AI_PHRASES:
            match = _phrase_pattern((phrase,)).search(sentence)
            if match:
                out.append(
                    _Candidate(
                        category=StyleCategory.TEMPLATE_PHRASE,
                        code="TEMPLATE_PHRASE",
                        severity=Severity.MEDIUM,
                        message=(
                            f"Known AI-associated phrase '{match.group(0)}'. These are the phrasings "
                            "that make a text read as generated rather than written."
                        ),
                        position=position + match.start(),
                        quote=sentence,
                        location=location,
                        section=section,
                        suggested_rewrite=(
                            "Name the finding itself: the number, the comparison, or the mechanism "
                            "you actually tested."
                        ),
                    )
                )
        return out

    def _rule_causal(
        self,
        sentence: str,
        position: int,
        location: str,
        section: PaperSection | None,
        level: EvidenceLevel,
        level_was_supplied: bool,
    ) -> list[_Candidate]:
        match = _CAUSAL_RE.search(sentence)
        if not match or level.rank >= EvidenceLevel.L4_INTERVENTION.rank:
            return []
        supplied = (
            f"the supplied evidence level is {level.value}"
            if level_was_supplied
            else "no evidence level was supplied, which is read as L0_IDEA"
        )
        return [
            _Candidate(
                category=StyleCategory.UNSUPPORTED_CAUSAL,
                code="CAUSAL_WITHOUT_INTERVENTION",
                severity=Severity.HIGH,
                message=(
                    f"Causal wording '{match.group(0)}' but {supplied}; an intervention "
                    "(L4_INTERVENTION) is required before a cause can be asserted. Permitted at "
                    f"{level.value}: '{PERMITTED_VERB_CLASS[level]}'."
                ),
                position=position + match.start(),
                quote=sentence,
                location=location,
                section=section,
                suggested_rewrite=self.suggest_weakening(sentence, level=level),
                evidence_level=level,
            )
        ]

    def _rule_novelty(
        self, sentence: str, position: int, location: str, section: PaperSection | None
    ) -> list[_Candidate]:
        match = _NOVELTY_RE.search(sentence)
        if not match:
            return []
        return [
            _Candidate(
                category=StyleCategory.UNSUPPORTED_NOVELTY,
                code="NOVELTY_UNSUPPORTED",
                severity=Severity.HIGH,
                message=(
                    f"Novelty wording '{match.group(0)}' is not licensed by the recorded search "
                    "coverage. Absence of a match is not absence of prior work: only a "
                    "coverage-gated novelty audit may support a novelty sentence."
                ),
                position=position + match.start(),
                quote=sentence,
                location=location,
                section=section,
                suggested_rewrite=CALIBRATED_NOVELTY_FRAMES[0],
            )
        ]

    def _rule_overclaim(
        self,
        sentence: str,
        position: int,
        location: str,
        section: PaperSection | None,
        level: EvidenceLevel,
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        hits: list[tuple[str, EvidenceLevel]] = []
        for pattern, required in _OVERCLAIM_MARKERS:
            match = pattern.search(sentence)
            if match and required.rank > level.rank:
                hits.append((match.group(0), required))
        if hits:
            worst = max(required for _, required in hits)
            gap = worst.rank - level.rank
            out.append(
                _Candidate(
                    category=StyleCategory.OVERCLAIM,
                    code="LANGUAGE_EXCEEDS_EVIDENCE",
                    severity=Severity.BLOCKER if gap >= 2 else Severity.HIGH,
                    message=(
                        "Wording "
                        + ", ".join(f"'{word}' (needs {required.value})" for word, required in hits)
                        + f" exceeds the evidence level {level.value}. Permitted at {level.value}: "
                        f"'{PERMITTED_VERB_CLASS[level]}'."
                    ),
                    position=position,
                    quote=sentence,
                    location=location,
                    section=section,
                    suggested_rewrite=self.suggest_weakening(sentence, level=level),
                    evidence_level=level,
                )
            )

        absolute = _ABSOLUTE_RE.search(sentence)
        if absolute and not _SCOPE_RE.search(sentence):
            out.append(
                _Candidate(
                    category=StyleCategory.OVERCLAIM,
                    code="ABSOLUTE_LANGUAGE",
                    severity=Severity.MEDIUM,
                    message=(
                        f"Absolute wording '{absolute.group(0)}' with no stated scope; an absolute "
                        "claim needs either a scope (which settings, which seeds) or removal."
                    ),
                    position=position + absolute.start(),
                    quote=sentence,
                    location=location,
                    section=section,
                    suggested_rewrite=self.suggest_weakening(sentence, level=level),
                    evidence_level=level,
                )
            )
        return out

    def _rule_generic(
        self, sentence: str, position: int, location: str, section: PaperSection | None
    ) -> list[_Candidate]:
        match = _GENERIC_RE.search(sentence)
        if not match:
            return []
        return [
            _Candidate(
                category=StyleCategory.GENERIC_STATEMENT,
                code="GENERIC_STATEMENT",
                severity=Severity.LOW,
                message=(
                    f"Generic statement '{match.group(0)}': true of any method in the field, so it "
                    "carries no information about this project."
                ),
                position=position + match.start(),
                quote=sentence,
                location=location,
                section=section,
                suggested_rewrite=(
                    "Delete it, or replace it with the specific result this project produced."
                ),
            )
        ]

    def _rule_rhetorical(
        self, sentence: str, position: int, location: str, section: PaperSection | None
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        for adjective in RHETORICAL_ADJECTIVES:
            match = _phrase_pattern((adjective,)).search(sentence)
            if not match:
                continue
            if adjective == "significantly" and (
                _has_number(sentence)
                or re.search(r"\bp\s*[<>=]", sentence)
                or re.search(r"\bCI\b|confidence interval", sentence)
            ):
                # statistical use of "significantly" is precise, not rhetorical
                continue
            if adjective in ("novel", "novelty") and (
                re.search(r"searched coverage", sentence, re.IGNORECASE)
                or re.search(r"not aware of prior work", sentence, re.IGNORECASE)
            ):
                # a calibrated novelty sentence is exactly what we want authors to write
                continue
            out.append(
                _Candidate(
                    category=StyleCategory.RHETORICAL_ADJECTIVE,
                    code="RHETORICAL_ADJECTIVE",
                    severity=Severity.LOW if adjective not in ("novel", "novelty") else Severity.MEDIUM,
                    message=(
                        f"Conditional adjective '{match.group(0)}': it asserts a judgement "
                        "(importance, size, novelty) that the evidence has to license."
                    ),
                    position=position + match.start(),
                    quote=sentence,
                    location=location,
                    section=section,
                    suggested_rewrite=self.suggest_weakening(sentence, level=EvidenceLevel.L3_CONTROLLED),
                )
            )
        return out

    def _rule_claim_drift(
        self,
        sentence: str,
        *,
        position: int,
        location: str,
        section: PaperSection | None,
        permitted: EvidenceLevel,
        claim_ids: tuple[str, ...],
    ) -> list[_Candidate]:
        if not claim_ids:
            return []
        required = required_evidence_level(sentence)
        if required.rank <= permitted.rank:
            return []
        return [
            _Candidate(
                category=StyleCategory.CLAIM_STRENGTH_DRIFT,
                code="CLAIM_STRENGTH_DRIFT",
                severity=Severity.HIGH,
                message=(
                    f"The sentence cites {', '.join(claim_ids)} and is written at {required.value}, "
                    f"but those claims permit only {permitted.value}. A hypothesis stated as a "
                    "finding is a different claim from the one that was tested."
                ),
                position=position,
                quote=sentence,
                location=location,
                section=section,
                suggested_rewrite=self.suggest_weakening(sentence, level=permitted),
                evidence_level=permitted,
                claim_ids=claim_ids,
            )
        ]

    def _rule_hedge(
        self, sentence: str, position: int, location: str, section: PaperSection | None
    ) -> list[_Candidate]:
        matches = list(_HEDGE_RE.finditer(sentence))
        distinct = {match.group(0).lower() for match in matches}
        if len(matches) < 2:
            return []
        return [
            _Candidate(
                category=StyleCategory.HEDGE_ABUSE,
                code="HEDGE_STACKING",
                severity=Severity.MEDIUM,
                message=(
                    f"{len(matches)} hedges in one sentence ({', '.join(sorted(distinct))}): the "
                    "reader cannot tell which of them the author means."
                ),
                position=position + matches[0].start(),
                quote=sentence,
                location=location,
                section=section,
                suggested_rewrite=(
                    "Keep exactly one hedge and state the condition it stands for (sample, setting "
                    "or measurement limit)."
                ),
            )
        ]

    # ------------------------------------------------------------------ document rules

    def _document_rules(
        self,
        profile: StyleProfile,
        *,
        offsets: tuple[int, ...],
        section: PaperSection | None,
        sections: tuple[PaperSection, ...] | None,
        timeline: tuple[TimelineEvent, ...],
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        out.extend(self._rule_repetition(profile, offsets, section))
        out.extend(self._rule_density(profile, section))
        out.extend(self._rule_structure(profile, offsets, section))
        out.extend(self._rule_reasoning(profile, offsets, section, sections))
        out.extend(self._rule_history(profile, offsets, section, timeline))
        return out

    def _rule_repetition(
        self, profile: StyleProfile, offsets: tuple[int, ...], section: PaperSection | None
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        sentences = profile.sentences
        tokenised = [normalised_tokens(sentence) for sentence in sentences]

        ngram_sentences: dict[tuple[str, ...], list[int]] = {}
        for index, tokens in enumerate(tokenised):
            seen: set[tuple[str, ...]] = set()
            for start in range(0, max(0, len(tokens) - NGRAM_LENGTH + 1)):
                gram = tokens[start : start + NGRAM_LENGTH]
                if gram in seen:
                    continue
                seen.add(gram)
                ngram_sentences.setdefault(gram, []).append(index)
        repeated = [
            (gram, indexes) for gram, indexes in ngram_sentences.items() if len(set(indexes)) >= 2
        ]
        repeated.sort(key=lambda item: (min(item[1]), item[0]))
        for gram, indexes in repeated[:MAX_FINDINGS_PER_RULE]:
            first = min(indexes)
            out.append(
                _Candidate(
                    category=StyleCategory.REPETITION,
                    code="REPEATED_NGRAM",
                    severity=Severity.MEDIUM,
                    message=(
                        f"The {NGRAM_LENGTH}-gram '{' '.join(gram)}' recurs in sentences "
                        f"{', '.join(str(i + 1) for i in sorted(set(indexes)))}; reused phrasing "
                        "reads as generation rather than argument."
                    ),
                    position=offsets[first] if first < len(offsets) else 0,
                    quote=sentences[first],
                    location=f"sentences {', '.join(str(i + 1) for i in sorted(set(indexes)))}",
                    section=section,
                    suggested_rewrite="State the second occurrence in terms of what is different there.",
                )
            )

        if len(sentences) >= MIN_SENTENCES_FOR_OPENING_RULE:
            openings: dict[tuple[str, ...], list[int]] = {}
            for index, tokens in enumerate(tokenised):
                if len(tokens) < REPEATED_OPENING_LENGTH:
                    continue
                openings.setdefault(tokens[:REPEATED_OPENING_LENGTH], []).append(index)
            for opening, indexes in sorted(openings.items(), key=lambda item: min(item[1])):
                if len(indexes) < MIN_SENTENCES_FOR_OPENING_RULE:
                    continue
                first = min(indexes)
                out.append(
                    _Candidate(
                        category=StyleCategory.REPETITION,
                        code="REPEATED_SENTENCE_OPENING",
                        severity=Severity.MEDIUM,
                        message=(
                            f"{len(indexes)} sentences open with '{' '.join(opening)}' "
                            f"(sentences {', '.join(str(i + 1) for i in indexes)}); parallel openings "
                            "flatten the argument into a list."
                        ),
                        position=offsets[first] if first < len(offsets) else 0,
                        quote=sentences[first],
                        location=f"sentences {', '.join(str(i + 1) for i in indexes)}",
                        section=section,
                        suggested_rewrite=(
                            "Vary the subject: lead with the object, the condition or the measurement."
                        ),
                    )
                )

        content = [content_tokens(sentence) for sentence in sentences]
        pairs: list[tuple[float, int, int]] = []
        for i in range(len(sentences)):
            if len(content[i]) < 4:
                continue
            for j in range(i + 1, len(sentences)):
                if len(content[j]) < 4:
                    continue
                score = _jaccard(content[i], content[j])
                if score >= PARAPHRASE_JACCARD:
                    pairs.append((score, i, j))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        for score, i, j in pairs[:MAX_FINDINGS_PER_RULE]:
            out.append(
                _Candidate(
                    category=StyleCategory.REPETITION,
                    code="PARAPHRASE_LOOP",
                    severity=Severity.MEDIUM,
                    message=(
                        f"Sentences {i + 1} and {j + 1} are near-duplicates "
                        f"(content-token Jaccard {score:.2f} \u2265 {PARAPHRASE_JACCARD}); the same "
                        "point is being made twice."
                    ),
                    position=offsets[i] if i < len(offsets) else 0,
                    quote=sentences[i],
                    location=f"sentences {i + 1}, {j + 1}",
                    section=section,
                    suggested_rewrite="Keep the stronger sentence and delete the other.",
                )
            )
        return out

    def _rule_density(self, profile: StyleProfile, section: PaperSection | None) -> list[_Candidate]:
        if not profile.sentences:
            return []
        counts = _buzzword_counts(" ".join(profile.sentences))
        hits = sum(counts.values())
        density = 1000.0 * hits / profile.word_count if profile.word_count else 0.0
        if density <= self.buzzword_limit:
            return []
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        return [
            _Candidate(
                category=StyleCategory.BUZZWORD_DENSITY,
                code="BUZZWORD_DENSITY",
                severity=Severity.MEDIUM,
                message=(
                    f"Buzzword density {density:.2f} per 1000 words exceeds the limit "
                    f"{self.buzzword_limit:.2f} ({hits} hits in {profile.word_count} words). "
                    "Top contributors: " + ", ".join(f"{word}\u00d7{count}" for word, count in top) + "."
                ),
                position=0,
                quote=None,
                location="document",
                section=section,
                suggested_rewrite=(
                    "Replace each buzzword with the specific thing it stands for: which component, "
                    "which measurement, which comparison."
                ),
            )
        ]

    def _rule_structure(
        self, profile: StyleProfile, offsets: tuple[int, ...], section: PaperSection | None
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        count = len(profile.sentences)
        if count >= MIN_STRUCTURE_SENTENCES:
            cv = profile.sentence_length_cv
            if cv < MIN_SENTENCE_CV:
                out.append(
                    _Candidate(
                        category=StyleCategory.AI_LIKE_STRUCTURE,
                        code="UNIFORM_SENTENCE_LENGTH",
                        severity=Severity.MEDIUM,
                        message=(
                            f"Sentence lengths are uniform: CV {cv:.3f} < {MIN_SENTENCE_CV} over "
                            f"{count} sentences (mean {profile.mean_sentence_length:.1f} words). "
                            "Uniform rhythm is a strong structural signal of generated prose."
                        ),
                        position=0,
                        quote=None,
                        location="document",
                        section=section,
                        suggested_rewrite=(
                            "Merge two short sentences and split one long one so length follows "
                            "content rather than a template."
                        ),
                    )
                )
            elif cv > MAX_SENTENCE_CV:
                out.append(
                    _Candidate(
                        category=StyleCategory.SENTENCE_UNIFORMITY,
                        code="SENTENCE_UNIFORMITY_UNRELIABLE",
                        severity=Severity.INFO,
                        message=(
                            f"Sentence lengths are too dispersed (CV {cv:.3f} > {MAX_SENTENCE_CV}) "
                            "for a mean-based uniformity verdict; no uniformity judgement is made "
                            "here, and that abstention is reported rather than hidden."
                        ),
                        position=0,
                        quote=None,
                        location="document",
                        section=section,
                        suggested_rewrite=None,
                    )
                )

        shapes: list[tuple[str, str, int]] = []
        for sentence in profile.sentences:
            tokens = normalised_tokens(sentence)
            if not tokens:
                shapes.append(("", "", 0))
                continue
            shapes.append((tokens[0], tokens[-1], len(tokens)))
        run_start = 0
        for index in range(1, len(profile.sentences) + 1):
            same = index < len(profile.sentences) and shapes[index][:2] == shapes[run_start][:2] and (
                abs(shapes[index][2] - shapes[run_start][2]) <= 2
            )
            if same:
                continue
            run_length = index - run_start
            if run_length >= LISTY_RUN_LENGTH and shapes[run_start][0]:
                out.append(
                    _Candidate(
                        category=StyleCategory.AI_LIKE_STRUCTURE,
                        code="LISTY_PARAGRAPH",
                        severity=Severity.MEDIUM,
                        message=(
                            f"{run_length} consecutive sentences share one shape "
                            f"(start '{shapes[run_start][0]}', end '{shapes[run_start][1]}', "
                            f"~{shapes[run_start][2]} words) — a list wearing prose clothes."
                        ),
                        position=offsets[run_start] if run_start < len(offsets) else 0,
                        quote=profile.sentences[run_start],
                        location=f"sentences {run_start + 1}-{index}",
                        section=section,
                        suggested_rewrite=(
                            "Turn the repeated frame into one sentence with an explicit enumeration, "
                            "or make each sentence carry a different subject."
                        ),
                    )
                )
            run_start = index

        for match in re.finditer(
            r"([^,;.]{2,60}?),\s*([^,;.]{2,60}?),\s*(?:and|or)\s+([^,;.]{2,60})", " ".join(profile.sentences)
        ):
            items = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]
            lengths = [len(item.split()) for item in items]
            firsts = [item.split()[0].lower().strip("'\"") for item in items]
            parallel = (
                len(set(lengths)) == 1
                or len(set(firsts)) == 1
                or all(len(item.split()) >= 2 for item in items)
                and len({item.split()[-1][-3:].lower() for item in items}) == 1
            )
            if not parallel:
                continue
            out.append(
                _Candidate(
                    category=StyleCategory.AI_LIKE_STRUCTURE,
                    code="TRICOLON",
                    severity=Severity.LOW,
                    message=(
                        f"Three-part parallel list '{match.group(0)}' — the rhetorical triple is a "
                        "cadence, not an argument."
                    ),
                    position=0,
                    quote=match.group(0),
                    location="document",
                    section=section,
                    suggested_rewrite=(
                        "Keep the two items the evidence supports and drop the third, or say what is "
                        "different about each."
                    ),
                )
            )
            if len(out) >= MAX_FINDINGS_PER_RULE:
                break

        if profile.sentences:
            transitions = sum(
                1
                for sentence in profile.sentences
                if any(
                    re.match(rf"^{re.escape(word)}\b", sentence.strip(), re.IGNORECASE)
                    for word in TRANSITION_WORDS
                )
            )
            ratio = transitions / len(profile.sentences)
            if transitions >= 3 and ratio >= TRANSITION_OVERUSE_RATIO:
                out.append(
                    _Candidate(
                        category=StyleCategory.AI_LIKE_STRUCTURE,
                        code="TRANSITION_OVERUSE",
                        severity=Severity.MEDIUM,
                        message=(
                            f"{transitions}/{len(profile.sentences)} sentences "
                            f"({ratio:.0%}) open with a transition word; the connectives are doing "
                            "the work the argument should do."
                        ),
                        position=0,
                        quote=None,
                        location="document",
                        section=section,
                        suggested_rewrite=(
                            "Delete the transition and let the previous sentence's content imply the "
                            "relation."
                        ),
                    )
                )
        return out

    def _rule_reasoning(
        self,
        profile: StyleProfile,
        offsets: tuple[int, ...],
        section: PaperSection | None,
        sections: tuple[PaperSection, ...] | None,
    ) -> list[_Candidate]:
        result_sections = {
            PaperSection.RESULTS,
            PaperSection.ANALYSIS,
            PaperSection.EXPERIMENTS,
        }
        if not profile.sentences:
            return []
        groups: list[tuple[PaperSection | None, list[int]]] = []
        if sections:
            by_section: dict[PaperSection, list[int]] = {}
            for index, item in enumerate(sections):
                by_section.setdefault(item, []).append(index)
            groups = list(by_section.items())
        elif section in result_sections:
            groups = [(section, list(range(len(profile.sentences))))]
        else:
            return []

        out: list[_Candidate] = []
        for group_section, indexes in groups:
            if group_section not in result_sections:
                continue
            body = " ".join(profile.sentences[index] for index in indexes if index < len(profile.sentences))
            if _REASONING_RE.search(body):
                continue
            out.append(
                _Candidate(
                    category=StyleCategory.MISSING_RESEARCHER_REASONING,
                    code="MISSING_RESEARCHER_REASONING",
                    severity=Severity.MEDIUM,
                    message=(
                        f"{group_section.value} reports results but never says why an experiment was "
                        "run or revised: no 'because', 'we expected', 'this surprised us', 'so we' or "
                        "'to test whether'. The decisions that produced these results are missing."
                    ),
                    position=offsets[indexes[0]] if indexes and indexes[0] < len(offsets) else 0,
                    quote=None,
                    location="section",
                    section=group_section,
                    suggested_rewrite=(
                        "Add one sentence of researcher reasoning, e.g. \"We ran this because ...\" "
                        "or \"This surprised us, so we ...\" — filled in from the decision log and "
                        "author notes, not from a template."
                    ),
                )
            )
        return out

    def _rule_history(
        self,
        profile: StyleProfile,
        offsets: tuple[int, ...],
        section: PaperSection | None,
        timeline: tuple[TimelineEvent, ...],
    ) -> list[_Candidate]:
        if not timeline or not profile.sentences:
            return []
        counters = [event for event in timeline if event.kind in HISTORY_COUNTER_KINDS]
        if not counters:
            return []
        out: list[_Candidate] = []
        for index, sentence in enumerate(profile.sentences):
            match = _CONFIRMATION_RE.search(sentence)
            if not match:
                continue
            kinds = sorted({event.kind.value for event in counters})
            out.append(
                _Candidate(
                    category=StyleCategory.HISTORY_MISMATCH,
                    code="HISTORY_MISMATCH",
                    severity=Severity.HIGH,
                    message=(
                        f"The text claims a clean confirmation ('{match.group(0)}'), but the recorded "
                        f"timeline contains {', '.join(kinds)}. The real path to this result was not "
                        "a straight hypothesis-to-confirmation arc, and the paper must not tell a "
                        "tidier story than the log."
                    ),
                    position=offsets[index] if index < len(offsets) else 0,
                    quote=sentence,
                    location=f"sentence {index + 1}",
                    section=section,
                    suggested_rewrite=(
                        "State what was actually revised and why, e.g. \"Our initial hypothesis "
                        "predicted X; after the anomaly in ..., we revised it to Y and tested that.\""
                    ),
                )
            )
        return out

    def _restatement_rules(
        self,
        profile: StyleProfile,
        *,
        offsets: tuple[int, ...],
        section: PaperSection | None,
        claim_texts: tuple[str, ...],
        sentence_numbers: bool,
    ) -> list[_Candidate]:
        """Compare sentences with the claim statements they appear to restate.

        This is the free-text half of ``CLAIM_STRENGTH_DRIFT``: with no claim ids available (plain
        text, abstracts), a sentence that reproduces a claim's content in stronger words is still
        detectable, because both the content and the verb class are computable from the text.
        """
        if not claim_texts:
            return []
        claims = [(text, required_evidence_level(text), content_tokens(text)) for text in claim_texts]
        out: list[_Candidate] = []
        for index, sentence in enumerate(profile.sentences):
            tokens = content_tokens(sentence)
            if len(tokens) < RESTATEMENT_MIN_SHARED:
                continue
            required = required_evidence_level(sentence)
            for claim_text, claim_level, claim_tokens in claims:
                shared = len(tokens & claim_tokens)
                if shared < RESTATEMENT_MIN_SHARED:
                    continue
                similar = _jaccard(tokens, claim_tokens) >= RESTATEMENT_JACCARD or (
                    _containment(tokens, claim_tokens) >= RESTATEMENT_CONTAINMENT
                )
                if not similar or required.rank <= claim_level.rank:
                    continue
                out.append(
                    _Candidate(
                        category=StyleCategory.CLAIM_STRENGTH_DRIFT,
                        code="CLAIM_STRENGTH_DRIFT",
                        severity=Severity.HIGH,
                        message=(
                            f"This sentence restates the claim \"{_truncate(claim_text, 120)}\" but at "
                            f"{required.value} instead of the claim's {claim_level.value}: the "
                            "sentence asserts more than the claim records."
                        ),
                        position=offsets[index] if index < len(offsets) else 0,
                        quote=sentence,
                        location=f"sentence {index + 1}" if sentence_numbers else f"chars {offsets[index] if index < len(offsets) else 0}",
                        section=section,
                        suggested_rewrite=self.suggest_weakening(sentence, level=claim_level),
                        evidence_level=claim_level,
                    )
                )
                break
        return out

    # ------------------------------------------------------------------ rewrite helpers

    def _downgrade_clause(self, text: str, level: EvidenceLevel) -> str:
        match = _CLAUSE_RE.match(text.strip())
        if not match:
            return text
        required = _CLAUSE_VERB_REQUIREMENT.get(match.group("verb").lower())
        if required is None or required.rank <= level.rank:
            return text
        replacement = _TARGET_CLAUSE_VERB[level]
        start = match.start("verb")
        end = match.end("verb")
        return text[:start] + replacement + text[end:]

    def _downgrade_verbs(self, text: str, level: EvidenceLevel) -> str:
        out = text
        if level.rank < EvidenceLevel.L4_INTERVENTION.rank:
            replacement = (
                "is associated with" if level.rank >= EvidenceLevel.L3_CONTROLLED.rank else "co-occurs with"
            )
            out = _CAUSAL_RE.sub(replacement, out)
        if level.rank < EvidenceLevel.L5_NECESSITY_OR_SUFFICIENCY.rank:
            out = _MECHANISM_RE.sub("is associated with", out)
            out = _NECESSITY_RE.sub("is associated with", out)
        if level.rank < EvidenceLevel.L6_CROSS_SETTING_REPLICATION.rank:
            out = _PROOF_RE.sub("provides evidence that", out)
        if level.rank < EvidenceLevel.L4_INTERVENTION.rank:
            out = _CONFIRM_RE.sub("is consistent with", out)
        if level.rank < EvidenceLevel.L3_CONTROLLED.rank:
            out = _DEMONSTRATE_RE.sub("indicates", out)
            out = _SHOW_RE.sub("indicates", out)
        return out

    def _strip_intensifiers(self, text: str) -> str:
        out = _INTENSIFIER_RE.sub(" ", text)
        # "significantly <value>" is a statistical statement; keep it in that one form.
        out = re.sub(r"\bsignificantly\b(?!\s*(?:<value>|p\b))", " ", out, flags=re.IGNORECASE)
        return out

    def _soften_absolutes(self, text: str) -> str:
        out = _ABSOLUTE_REMOVABLE_RE.sub(" ", text)
        out = _NEVER_SAFE_RE.sub(lambda m: f"not {m.group(1)}", out)
        return out

    @staticmethod
    def _tidy(text: str) -> str:
        out = " ".join(text.split())
        out = re.sub(r"\s+([,.;:])", r"\1", out)
        out = re.sub(r",\s*,+", ",", out)
        out = re.sub(
            r"\b(a|an|the|our|this|that|its|their|these|those|of|with|for)\s+and\s+",
            lambda match: match.group(1) + " ",
            out,
        )
        out = re.sub(r"^\s*and\s+", "", out, flags=re.IGNORECASE)
        out = re.sub(r",\s+and\s+", " and ", out)
        out = re.sub(r"\s+\.", ".", out)
        out = re.sub(r"\(\s*\)", "", out)
        out = " ".join(out.split()).strip()
        if out and out[0].islower():
            out = out[0].upper() + out[1:]
        return out

    # ------------------------------------------------------------------ audit mapping

    def _to_audit_finding(self, candidate: _Candidate, paper: PaperArtifact) -> Finding:
        blocking = candidate.code in BLOCKING_CODES and _at_least(
            candidate.severity, BLOCKING_MIN_SEVERITY
        )
        return Finding(
            code=candidate.code,
            severity=candidate.severity,
            message=candidate.message,
            target_ref=paper.paper_id,
            claim_ids=list(candidate.claim_ids),
            evidence_ids=list(candidate.evidence_ids),
            quote=(_truncate(candidate.quote, 2000) if candidate.quote else None),
            location=candidate.location,
            suggestion=candidate.suggested_rewrite,
            blocks_publication=blocking,
            deterministic=True,
            confidence=1.0,
            details={
                "style_category": candidate.category.value,
                "section": candidate.section.value if candidate.section else None,
                "evidence_level": candidate.evidence_level.value if candidate.evidence_level else None,
            },
        )

    @staticmethod
    def _with_rank(findings: Sequence[StyleFinding], rank: int, base: int) -> list[_Candidate]:
        out: list[_Candidate] = []
        for index, finding in enumerate(findings):
            out.append(
                _Candidate(
                    category=finding.category,
                    code=finding.code,
                    severity=finding.severity,
                    message=finding.message,
                    position=rank * 100000 + base + index,
                    quote=finding.quote,
                    location=finding.location,
                    section=finding.section,
                    suggested_rewrite=finding.suggested_rewrite,
                    evidence_level=finding.evidence_level,
                )
            )
        return out

    @staticmethod
    def _location_position(location: str | None) -> int:
        if not location:
            return 0
        digits = re.findall(r"\d+", location)
        return int(digits[-1]) if digits else 0

    @staticmethod
    def _claims_for_quote(
        sentences: Sequence[GroundedSentence], finding: StyleFinding
    ) -> tuple[str, ...]:
        quote = (finding.quote or "").strip()
        for sentence in sentences:
            if sentence.text.strip() == quote or quote in sentence.text or sentence.text in quote:
                return tuple(sentence.claim_ids)
        return ()

    @staticmethod
    def _evidence_for_quote(
        sentences: Sequence[GroundedSentence], finding: StyleFinding
    ) -> tuple[str, ...]:
        quote = (finding.quote or "").strip()
        for sentence in sentences:
            if sentence.text.strip() == quote or quote in sentence.text or sentence.text in quote:
                return tuple(sentence.evidence_ids)
        return ()


__all__ = [
    "ABSOLUTE_WORDS",
    "BLOCKING_CODES",
    "BUZZWORD_DENSITY_LIMIT",
    "BUZZWORDS",
    "CALIBRATED_NOVELTY_FRAMES",
    "CAUSAL_VERBS",
    "CLAIM_STATUS_LEVEL_CAP",
    "CONFIRMATION_PHRASES",
    "EMPTY_BACKGROUND_PATTERNS",
    "GENERIC_STATEMENT_PATTERNS",
    "HEDGES",
    "HISTORY_COUNTER_KINDS",
    "INTENSIFIERS",
    "KNOWN_AI_PHRASES",
    "MASK_CITATION",
    "MASK_NUMBER",
    "MAX_SENTENCE_CV",
    "MIN_SENTENCE_CV",
    "NOVELTY_PHRASES",
    "PERMITTED_VERB_CLASS",
    "REASONING_MARKERS",
    "RHETORICAL_ADJECTIVES",
    "STYLE_AUDIT_TOOL",
    "STYLE_AUDIT_VERSION",
    "TEMPLATE_PHRASES",
    "TRANSITION_WORDS",
    "StyleAuditor",
    "StyleProfile",
    "claim_permitted_level",
    "content_tokens",
    "normalised_tokens",
    "required_evidence_level",
    "split_sentences",
    "word_tokens",
]
