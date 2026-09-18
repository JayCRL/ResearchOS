"""LLM adapters — optional, provider-agnostic, and never on the truth path.

See :mod:`researchos.llm.base` for the boundary this package enforces.
"""

from .base import (
    PROHIBITED_USES,
    EchoProvider,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    system_prompt,
    user_prompt,
)
from .providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAICompatProvider,
    provider_from_env,
)
from .router import ROLE_PROMPTS, LLMRouter

__all__ = [
    "AnthropicProvider",
    "EchoProvider",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "OllamaProvider",
    "OpenAICompatProvider",
    "PROHIBITED_USES",
    "ROLE_PROMPTS",
    "provider_from_env",
    "system_prompt",
    "user_prompt",
]
