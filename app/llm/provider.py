"""
LLM provider factory.

Returns a provider instance based on settings.llm_provider.
Only Groq is fully implemented; the others are functional stubs that raise
NotImplementedError so the architecture is complete and easy to fill in later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from app.config import settings


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the assistant message text."""


# ── Groq (fully implemented via httpx) ───────────────────────────────────────

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseLLMProvider):
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in configuration.")
        self._api_key = settings.groq_api_key
        self._model = settings.groq_model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            # Force JSON output — Groq honours OpenAI's response_format field
            "response_format": {"type": "json_object"},
            "temperature": 0.0,   # deterministic for structured extraction
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(_GROQ_CHAT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


# ── Stubs ─────────────────────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Ollama provider not yet implemented.")


class OpenAIProvider(BaseLLMProvider):
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("OpenAI provider not yet implemented.")


class AnthropicProvider(BaseLLMProvider):
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Anthropic provider not yet implemented.")


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_provider() -> BaseLLMProvider:
    """Instantiate and return the provider configured in settings."""
    provider = settings.llm_provider
    if provider == "groq":
        return GroqProvider()
    if provider == "ollama":
        return OllamaProvider()
    if provider == "openai":
        return OpenAIProvider()
    if provider == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
