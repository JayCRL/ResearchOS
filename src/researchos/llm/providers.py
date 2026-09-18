"""Concrete LLM providers.

All of them are optional and lazily configured: no credentials, no import, no problem. The HTTP
providers take an injectable ``poster`` so they are testable without a network.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping, Sequence

from .base import LLMMessage, LLMProvider, LLMResponse

Poster = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


def _urllib_poster(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    """Default HTTP POST. Imported here so the package has no hard dependency at import time."""
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={**dict(headers), "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit endpoint
        return json.loads(response.read().decode("utf-8"))


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible endpoint: OpenAI, vLLM, LM Studio, Together, Ollama's compat layer."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        poster: Poster | None = None,
        timeout: float = 60.0,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.poster = poster or _urllib_poster
        self.timeout = timeout
        if name:
            self.name = name

    def available(self) -> bool:
        return bool(self.api_key) or "localhost" in self.base_url or "127.0.0.1" in self.base_url

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        data = self.poster(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            payload,
            self.timeout,
        )
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content", "")
        return LLMResponse(
            text=text or "",
            provider=self.name,
            model=self.model,
            usage=data.get("usage") or {},
            raw=data,
            deterministic=False,
        )


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        poster: Poster | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.poster = poster or _urllib_poster
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system = "\n".join(m.content for m in messages if m.role == "system")
        turns = [m.as_dict() for m in messages if m.role != "system"]
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or 2048,
            "temperature": temperature,
            "system": system,
            "messages": turns,
        }
        data = self.poster(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            payload,
            self.timeout,
        )
        blocks = data.get("content") or [{}]
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, Mapping))
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=data.get("usage") or {},
            raw=data,
            deterministic=False,
        )


class OllamaProvider(OpenAICompatProvider):
    """A local model through Ollama's OpenAI-compatible endpoint. No API key, no data egress."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434/v1",
        poster: Poster | None = None,
    ) -> None:
        super().__init__(model=model, base_url=base_url, api_key="ollama", poster=poster)

    def available(self) -> bool:
        return True  # reachability is checked at call time, not by a guess


def provider_from_env(**overrides: Any) -> LLMProvider:
    """Choose a provider from the environment, falling back to the deterministic echo provider.

    A missing configuration is not an error: ResearchOS works without any model, which is the point.
    """
    if os.environ.get("RESEARCHOS_LLM") == "echo":
        from .base import EchoProvider

        return EchoProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(**overrides)
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return OpenAICompatProvider(
            model=os.environ.get("RESEARCHOS_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            **overrides,
        )
    if os.environ.get("RESEARCHOS_OLLAMA"):
        return OllamaProvider(**overrides)
    from .base import EchoProvider

    return EchoProvider()
