"""Researcher voice: the paper narrative must follow the research that actually happened.

Two failure modes this module exists to prevent:

1. **The faked arc.** A draft that reads ``hypothesis → experiment → confirmation`` when the timeline
   shows ``observation → anomaly → hypothesis revision → control → evidence → claim`` is not a stylistic
   problem; it is a false account of the method. :meth:`ResearcherVoice.arc` reconstructs the real arc
   from records, and :meth:`ResearcherVoice.distortion` names the difference.
2. **The generic voice.** Published prose is a poor model of how *this* researcher writes. Their own
   notes, diaries and decision records are a much better one, and they are already in the project.

Everything here is assembled from records — timeline events, author notes, decisions, claim history —
and every sentence carries the ids it came from so the compiler can ground it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Sequence

from ..kernel.kernel import ResearchKernel
from ..models.common import SourceRef
from ..models.paper import PaperSection
from ..models.timeline import AuthorNote, NoteKind, TimelineEvent, TimelineEventKind

#: Phrases that claim a clean confirmation arc.
CONFIRMATION_PHRASES: tuple[str, ...] = (
    "as we hypothesised",
    "as we hypothesized",
    "as expected",
    "confirming our hypothesis",
    "as predicted",
    "our hypothesis was confirmed",
    "in line with our expectations",
)

#: Event kinds that indicate the research did *not* go straight to a confirmation.
REVISION_KINDS: frozenset[TimelineEventKind] = frozenset(
    {
        TimelineEventKind.ANOMALY,
        TimelineEventKind.CLAIM_REVISION,
        TimelineEventKind.CLAIM_REJECTED,
        TimelineEventKind.EXPERIMENT_FAILED,
        TimelineEventKind.DECISION,
        TimelineEventKind.STATE_TRANSITION,
        TimelineEventKind.AUDIT,
    }
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class VoiceStep:
    """One step of the real research arc, with the records it came from."""

    phase: str
    title: str
    detail: str = ""
    at: str | None = None
    refs: list[str] = field(default_factory=list)
    note_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)


@dataclass
class VoiceProfile:
    """A style fingerprint taken from the researcher's own notes, not from published prose."""

    sample_size: int
    mean_sentence_words: float
    sentence_words_std: float
    mean_note_words: float
    first_person_ratio: float
    hedge_ratio: float
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "sample_size": self.sample_size,
            "mean_sentence_words": round(self.mean_sentence_words, 2),
            "sentence_words_std": round(self.sentence_words_std, 2),
            "mean_note_words": round(self.mean_note_words, 2),
            "first_person_ratio": round(self.first_person_ratio, 3),
            "hedge_ratio": round(self.hedge_ratio, 3),
        }


class ResearcherVoice:
    """Reconstructs the real arc and the researcher's own voice from project records."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ------------------------------------------------------------------ notes

    def notes(self) -> list[AuthorNote]:
        return sorted(self.kernel.notes.all(), key=lambda note: note.created_at)

    def notes_of_kind(self, *kinds: NoteKind) -> list[AuthorNote]:
        wanted = set(kinds)
        return [note for note in self.notes() if note.kind in wanted]

    # ------------------------------------------------------------------ the arc

    def arc(self) -> list[VoiceStep]:
        """The real research arc, assembled from timeline events, notes and decisions.

        Phases, in the order a project usually passes through them: observation, hypothesis, anomaly,
        control, revision, evidence, claim. Phases with no records are simply absent — the arc is a
        report, not a template.
        """
        steps: list[VoiceStep] = []
        events = self.kernel.history.all()

        def add(phase: str, title: str, detail: str = "", at: str | None = None, refs: Sequence[str] = ()) -> None:
            steps.append(VoiceStep(phase=phase, title=title, detail=detail, at=at, refs=list(refs)))

        description_events = [
            event
            for event in events
            if event.kind
            in {
                TimelineEventKind.PROJECT_CREATED,
                TimelineEventKind.IMPORT,
                TimelineEventKind.IDEA,
                TimelineEventKind.FINDING,
            }
        ]
        for event in description_events[:4]:
            add("observation", event.title, event.detail, event.at.isoformat(), event.refs)

        for note in self.notes_of_kind(NoteKind.WHY_THIS_EXPERIMENT, NoteKind.INTUITION)[:3]:
            step = VoiceStep(
                phase="hypothesis",
                title=note.text[:240],
                detail=f"author note ({note.kind.value})",
                at=note.created_at.isoformat(),
                note_ids=[note.note_id],
            )
            steps.append(step)

        for note in self.notes_of_kind(NoteKind.ANOMALY, NoteKind.FAILED_EXPERIMENT)[:4]:
            steps.append(
                VoiceStep(
                    phase="anomaly",
                    title=note.text[:240],
                    detail=f"author note ({note.kind.value})",
                    at=note.created_at.isoformat(),
                    note_ids=[note.note_id],
                )
            )

        for experiment in self.kernel.experiments.all():
            if experiment.control is None:
                continue
            related = [
                decision
                for decision in self.kernel.decisions.all()
                if experiment.experiment_id in decision.affected_experiments
                or decision.kind.value in {"CONTROL_ADDED", "EXPERIMENT_ROUTE"}
            ]
            steps.append(
                VoiceStep(
                    phase="control",
                    title=f"{experiment.title}: control arm {experiment.control.name!r}",
                    detail="matched on " + ", ".join(experiment.matched_conditions or ["nothing recorded"]),
                    refs=[experiment.experiment_id],
                    decision_ids=[decision.decision_id for decision in related[:2]],
                )
            )

        for note in self.notes_of_kind(NoteKind.CLAIM_REVISION, NoteKind.DECISION_NOTE)[:4]:
            steps.append(
                VoiceStep(
                    phase="revision",
                    title=note.text[:240],
                    detail=f"author note ({note.kind.value})",
                    at=note.created_at.isoformat(),
                    note_ids=[note.note_id],
                )
            )
        for decision in self.kernel.decisions.all():
            steps.append(
                VoiceStep(
                    phase="revision",
                    title=decision.summary[:240],
                    detail=decision.rationale[:240],
                    at=decision.created_at.isoformat(),
                    decision_ids=[decision.decision_id],
                    refs=list(decision.affected_claims) + list(decision.affected_experiments),
                )
            )

        for claim in self.kernel.claims.all():
            if claim.status.value not in {"SUPPORTED", "ROBUST"}:
                continue
            related = [
                decision
                for decision in self.kernel.decisions.all()
                if claim.claim_id in decision.affected_claims
            ]
            steps.append(
                VoiceStep(
                    phase="claim",
                    title=claim.statement[:240],
                    detail=f"status {claim.status.value}, evidence level {claim.evidence_level.value}",
                    refs=[claim.claim_id],
                    decision_ids=[decision.decision_id for decision in related[:2]],
                )
            )

        order = {"observation": 0, "hypothesis": 1, "anomaly": 2, "control": 3, "revision": 4, "evidence": 5, "claim": 6}
        steps.sort(key=lambda step: (order.get(step.phase, 9), step.at or ""))
        return steps

    def arc_phases(self) -> list[str]:
        seen: list[str] = []
        for step in self.arc():
            if step.phase not in seen:
                seen.append(step.phase)
        return seen

    # ------------------------------------------------------------------ distortion

    def distortion(self, text: str) -> list[str]:
        """Name the ways ``text`` misrepresents the project's own history."""
        problems: list[str] = []
        lowered = text.lower()
        revision_events = [event for event in self.kernel.history.all() if event.kind in REVISION_KINDS]
        revisions = [
            step
            for step in self.arc()
            if step.phase in {"anomaly", "revision"}
        ]
        for phrase in CONFIRMATION_PHRASES:
            if phrase in lowered and revisions:
                problems.append(
                    f"the draft says {phrase!r} but the project records {len(revisions)} anomaly/revision "
                    "step(s) before the result (see `researchos timeline`)"
                )
        if "we hypothesised" in lowered or "we hypothesized" in lowered:
            if not self.notes_of_kind(NoteKind.WHY_THIS_EXPERIMENT, NoteKind.INTUITION):
                problems.append(
                    "the draft describes a hypothesis, but no hypothesis note exists in the project: "
                    "either add the note from your own record or describe the observation that started it"
                )
        if revision_events and "first" not in lowered and "initially" not in lowered:
            problems.append(
                "the draft does not mention that the research direction was revised, although the "
                "timeline records it"
            )
        return problems

    # ------------------------------------------------------------------ profile

    def profile(self) -> VoiceProfile:
        """A style fingerprint from the researcher's own notes."""
        notes = self.notes()
        texts = [note.text for note in notes if note.text.strip()]
        sentences = [s for text in texts for s in _SENTENCE.split(text) if s.strip()]
        lengths = [len(sentence.split()) for sentence in sentences] or [0]
        words = [word for text in texts for word in text.lower().split()]
        first_person = sum(1 for word in words if word in {"i", "we", "my", "our", "me", "us"})
        hedges = sum(
            1
            for word in words
            if word in {"may", "might", "perhaps", "possibly", "seems", "looks", "unclear", "suspect"}
        )
        return VoiceProfile(
            sample_size=len(texts),
            mean_sentence_words=mean(lengths),
            sentence_words_std=pstdev(lengths) if len(lengths) > 1 else 0.0,
            mean_note_words=mean([len(text.split()) for text in texts]) if texts else 0.0,
            first_person_ratio=first_person / len(words) if words else 0.0,
            hedge_ratio=hedges / len(words) if words else 0.0,
            examples=texts[:3],
        )

    # ------------------------------------------------------------------ sentences

    def narrative_sentences(
        self, *, limit: int = 6
    ) -> list[tuple[str, PaperSection, list[str], list[str], list[str]]]:
        """Grounded sentences for the paper, each with its record ids.

        Returns ``(text, section, note_ids, decision_ids, claim_ids)``. The text deliberately contains no
        numerals: a number in the narrative would have to come from an analysis artifact, and the
        narrative is about process, not results.
        """
        out: list[tuple[str, PaperSection, list[str], list[str], list[str]]] = []
        arc = self.arc()

        def claim_refs(refs: Sequence[str]) -> list[str]:
            return [ref for ref in refs if ref.startswith("clm_")]

        anomalies = [step for step in arc if step.phase == "anomaly"]
        controls = [step for step in arc if step.phase == "control"]
        revisions = [step for step in arc if step.phase == "revision"]

        if anomalies:
            first = anomalies[0]
            out.append(
                (
                    "An unexpected observation changed the course of the work: "
                    + _strip_numerals(first.title).rstrip(".") + ".",
                    PaperSection.INTRODUCTION,
                    first.note_ids,
                    first.decision_ids,
                    [],
                )
            )
        if controls:
            first = controls[0]
            out.append(
                (
                    "To separate the effect from its alternatives we added a control condition: "
                    + _strip_numerals(first.title).rstrip(".") + ".",
                    PaperSection.METHOD,
                    [],
                    first.decision_ids,
                    [],
                )
            )
        if revisions:
            first = revisions[0]
            out.append(
                (
                    "We revised our working hypothesis rather than fitting the story to the data: "
                    + _strip_numerals(first.title).rstrip(".") + ".",
                    PaperSection.DISCUSSION,
                    first.note_ids,
                    first.decision_ids,
                    [],
                )
            )
        for step in [s for s in arc if s.phase == "claim"][:2]:
            out.append(
                (
                    "What survived that process is the claim we report: "
                    + _strip_numerals(step.title).rstrip(".") + ".",
                    PaperSection.CONCLUSION,
                    step.note_ids,
                    step.decision_ids,
                    claim_refs(step.refs),
                )
            )
        # Only sentences that point at a record are offered: an ungrounded narrative sentence is exactly
        # the kind of fluent filler this system exists to refuse.
        grounded = [item for item in out if item[2] or item[3] or item[4]]
        return grounded[:limit]


def _strip_numerals(text: str) -> str:
    """Remove numerals from narrative text.

    The narrative is about *process*; every number in the paper must come from an analysis artifact, so
    the voice layer refuses to carry one rather than forcing the compiler to reject the sentence.

    Two shapes are handled separately because they read differently: scale tokens (``124M``, ``1B``,
    ``3k``) are dropped entirely, while genuine measurements (``0.418``) become a placeholder phrase —
    deleting them would leave a hole in the sentence.
    """
    cleaned = re.sub(r"\b\d+(?:[.,]\d+)?\s*[A-Za-z]{1,3}\b", "", text)          # 124M, 1B, 3k
    cleaned = re.sub(r"\b\d+(?:[.,]\d+)?%?\b", "the measured value", cleaned)   # 0.418, 12%
    cleaned = re.sub(r"\bthe\s+the\b", "the", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip().replace(" ,", ",").replace(" .", ".")


def voice_context(kernel: ResearchKernel) -> dict[str, object]:
    """A compact, structured description of the researcher's voice, for a writer or a model."""
    voice = ResearcherVoice(kernel)
    return {
        "arc": [
            {"phase": step.phase, "title": step.title, "detail": step.detail, "at": step.at}
            for step in voice.arc()
        ],
        "profile": voice.profile().as_dict(),
        "distortion_warnings": voice.distortion(
            " ".join(note.text for note in voice.notes())
        ),
        "source_refs": [
            SourceRef(path=f".researchos/notes/{note.note_id}.yaml", locator=note.note_id).model_dump(mode="json")
            for note in voice.notes()[:5]
        ],
    }
