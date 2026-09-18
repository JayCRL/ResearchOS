"""The LLM router: where a model may be used, and where it may not.

The router exists to make the boundary *explicit and checkable* rather than a matter of developer
discipline. Its public surface only exposes proposal-shaped operations:

* ``propose_queries``   — expand a literature search (the resulting query set is recorded in the plan),
* ``propose_hypotheses``— generate candidate hypotheses (they enter as IDEA claims, never more),
* ``propose_language``  — rewrite prose that is already grounded (numbers and citations are re-verified),
* ``summarise_evidence``— summarise *for a human reader*, never for the claim registry.

Every response is checked for the two things a model must never smuggle in: a numeral that is not in the
provided set, and a novelty phrase without coverage.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import LLMMessage, LLMProvider, LLMResponse, system_prompt, user_prompt

#: The instruction set handed to every provider, per task kind.
ROLE_PROMPTS: Mapping[str, Sequence[str]] = {
    "queries": (
        "Propose additional literature search queries that a keyword search would miss.",
        "Include historical terminology, mechanistic equivalents and neighbouring communities.",
        "Return a JSON array of strings. Do not justify each query.",
    ),
    "hypotheses": (
        "Propose falsifiable hypotheses consistent with the given state.",
        "Each hypothesis must be testable with an experiment that could refute it.",
        "Return a JSON array of objects with 'statement' and 'test' fields.",
    ),
    "language": (
        "Rewrite the given sentences to be clearer, keeping the scientific content identical.",
        "You may not add, remove or change any number, citation, mechanism or scope.",
        "Return a JSON object mapping sentence ids to rewritten text.",
    ),
    "summary": (
        "Summarise the supplied evidence for a human reader.",
        "State uncertainty explicitly. Never assert more than the evidence levels permit.",
    ),
}


class LLMRouter:
    """Routes proposal tasks to a provider and validates what comes back."""

    def __init__(self, provider: LLMProvider, *, enabled: bool = True) -> None:
        self.provider = provider
        self.enabled = enabled
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ plumbing

    def _call(
        self,
        kind: str,
        instruction: str,
        *,
        context: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        constraints = ROLE_PROMPTS.get(kind)
        if constraints is None:
            raise ValueError(f"unknown LLM task kind {kind!r}; add it to ROLE_PROMPTS")
        messages = [system_prompt(role=f"{kind} proposer", constraints=constraints), user_prompt(instruction, context=context)]
        response = self.provider.complete(messages, temperature=temperature, max_tokens=max_tokens)
        self.calls.append({"kind": kind, "provider": response.provider, "chars": len(response.text)})
        return response

    @property
    def deterministic(self) -> bool:
        return getattr(self.provider, "deterministic", False)

    # ------------------------------------------------------------------ proposals

    def propose_queries(self, target: str, *, existing: Sequence[str] = (), limit: int = 8) -> list[str]:
        if not self.enabled:
            return []
        response = self._call(
            "queries",
            f"Target: {target}\nAlready planned queries: {list(existing)}",
            context={"target": target, "existing_queries": list(existing), "limit": limit},
            max_tokens=600,
        )
        return [str(item).strip() for item in _as_list(response.text) if str(item).strip()][:limit]

    def propose_hypotheses(
        self, *, state_summary: Mapping[str, Any], limit: int = 5
    ) -> list[dict[str, str]]:
        if not self.enabled:
            return []
        response = self._call(
            "hypotheses",
            f"Propose up to {limit} hypotheses.",
            context=state_summary,
            temperature=0.4,
            max_tokens=900,
        )
        out: list[dict[str, str]] = []
        for item in _as_list(response.text):
            if isinstance(item, Mapping) and item.get("statement"):
                out.append({"statement": str(item["statement"]), "test": str(item.get("test", ""))})
        return out[:limit]

    def propose_language(
        self,
        sentences: Mapping[str, str],
        *,
        allowed_numbers: Sequence[float] = (),
        evidence_levels: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Rewrite grounded sentences. Any response that changes a number is discarded wholesale."""
        if not self.enabled:
            return {}
        response = self._call(
            "language",
            "Rewrite the sentences, keeping every number and citation identical.",
            context={
                "sentences": dict(sentences),
                "allowed_numbers": list(allowed_numbers),
                "evidence_levels": dict(evidence_levels or {}),
            },
            max_tokens=1200,
        )
        try:
            proposed = response.json()
        except Exception:  # noqa: BLE001 - a malformed rewrite is simply ignored
            return {}
        if not isinstance(proposed, Mapping):
            return {}
        checked: dict[str, str] = {}
        for key, value in proposed.items():
            original = sentences.get(str(key))
            if original is None or not isinstance(value, str):
                continue
            if not _numbers_preserved(original, value):
                continue
            checked[str(key)] = value
        return checked

    def summarise_evidence(self, *, evidence: Mapping[str, Any]) -> str:
        if not self.enabled:
            return ""
        response = self._call("summary", "Summarise the evidence below.", context=evidence, max_tokens=900)
        return response.text

    # ------------------------------------------------------------------ guard

    def refuse(self, use: str) -> None:
        """Explicit refusal for the prohibited uses, so callers cannot "just try it"."""
        from .base import PROHIBITED_USES

        if use in PROHIBITED_USES:
            raise ValueError(
                f"refusing to route {use!r} through a language model: it is a deterministic "
                "responsibility of the kernel or the analysis layer"
            )


def _as_list(text: str) -> list[Any]:
    """Parse a JSON array out of a response, or fall back to newline-separated items."""
    import json

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
    try:
        parsed = json.loads(stripped)
    except Exception:  # noqa: BLE001
        return [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, Mapping):
        for key in ("queries", "items", "results", "hypotheses"):
            if isinstance(parsed.get(key), list):
                return list(parsed[key])
    return []


def _numbers_preserved(original: str, rewritten: str) -> bool:
    from ..paper.grounding import extract_numerals

    before = sorted(value for value, _raw, _offset in extract_numerals(original))
    after = sorted(value for value, _raw, _offset in extract_numerals(rewritten))
    return before == after
