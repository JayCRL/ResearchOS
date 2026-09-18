"""LLM adapter interface.

ResearchOS is deliberately *capable* of using language models and deliberately *not dependent* on them.
Everything in this package is optional: the kernel, evidence, claims, analysis and paper grounding do
not import it (enforced by ``tests/test_layering_and_cli.py``).

What a model may do here is propose: hypotheses, plan steps, query expansions, prose drafts. What it
may never do is decide a fact — numbers, hashes, statistics, citation metadata, state transitions and
grounding verification are computed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Uses this layer must never be put to. Kept as data so it can be asserted in tests.
PROHIBITED_USES: tuple[str, ...] = (
    "computing a statistic",
    "deciding whether a claim is supported",
    "approving a state transition",
    "verifying an artifact hash",
    "resolving a conflict by trust order",
    "judging novelty coverage",
    "generating a number for a paper",
    "authoring a citation",
)


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    deterministic: bool = False

    def json(self) -> Any:
        """Parse the response as JSON, tolerating a fenced code block."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[1] if "\n" in text else text
        return json.loads(text)


class LLMProvider(ABC):
    """Minimal provider interface. Implementations must not be required by the truth path."""

    name: str = "provider"
    requires_network: bool = True

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "network": self.requires_network, "available": self.available()}


class EchoProvider(LLMProvider):
    """Deterministic provider used by tests and by ``--no-llm`` runs.

    It never fabricates content: it echoes the last instruction, which makes it obvious in a transcript
    that no model was involved.
    """

    name = "echo"
    requires_network = False

    def __init__(self, response: str | None = None) -> None:
        self._response = response
        self.calls: list[list[LLMMessage]] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._response is not None:
            text = self._response
        else:
            last = messages[-1].content if messages else ""
            text = f"[echo] {last}".strip()
        return LLMResponse(
            text=text, provider=self.name, model="echo-1", deterministic=True, usage={"calls": len(self.calls)}
        )


def system_prompt(*, role: str, constraints: Sequence[str]) -> LLMMessage:
    """Build a system message that states the constraints explicitly rather than implying them."""
    body = [f"You are the {role} in ResearchOS.", "", "Hard constraints:"]
    body.extend(f"- {constraint}" for constraint in constraints)
    body.append("")
    body.append(
        "You never invent numbers, citations, experiments or mechanisms. If information is missing, say "
        "so and stop."
    )
    return LLMMessage(role="system", content="\n".join(body))


def user_prompt(instruction: str, *, context: Mapping[str, Any] | None = None) -> LLMMessage:
    """Build a user message from structured state, never from a chat history."""
    parts = [instruction]
    if context:
        parts.append("")
        parts.append("Structured context (authoritative):")
        parts.append(json.dumps(context, indent=2, default=str, sort_keys=True))
    return LLMMessage(role="user", content="\n".join(parts))
