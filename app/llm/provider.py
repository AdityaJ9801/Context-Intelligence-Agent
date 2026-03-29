"""LLM provider factory — returns the correct async client based on settings."""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from app.config import settings

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class BaseLLMProvider(ABC):
    """Abstract base for all LLM provider implementations."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the assistant message text."""


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI chat-completion provider via httpx."""

    def __init__(self) -> None:
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            raise ValueError("Azure OpenAI configuration is missing in settings.")
        self._api_key = settings.azure_openai_api_key
        base_url = settings.azure_openai_endpoint.rstrip("/")
        deployment = settings.azure_openai_deployment_name
        api_version = settings.azure_openai_api_version
        self._url = f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Azure OpenAI chat-completions endpoint and return the response text."""
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self._url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class GroqProvider(BaseLLMProvider):
    """Groq chat-completion provider via httpx (fully async, no SDK dependency)."""

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in configuration.")
        self._api_key = settings.groq_api_key
        self._model = settings.groq_model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Groq chat-completions endpoint and return the response text."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(_GROQ_CHAT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class OllamaProvider(BaseLLMProvider):
    """Ollama provider stub — not yet implemented."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Raise NotImplementedError until Ollama support is added."""
        raise NotImplementedError("Ollama provider not yet implemented.")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI chat-completion provider via httpx."""

    def __init__(self) -> None:
        if not getattr(settings, "openai_api_key", None):
            raise ValueError("OPENAI_API_KEY is not set in configuration.")
        self._api_key = settings.openai_api_key
        self._model = getattr(settings, "openai_model", "gpt-4o")

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI chat-completions endpoint and return the response text."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider stub — not yet implemented."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Raise NotImplementedError until Anthropic support is added."""
        raise NotImplementedError("Anthropic provider not yet implemented.")


def get_llm_provider() -> BaseLLMProvider:
    """Instantiate and return the provider configured in settings."""
    provider = settings.llm_provider
    if provider == "azure_openai":
        return AzureOpenAIProvider()
    if provider == "groq":
        return GroqProvider()
    if provider == "ollama":
        return OllamaProvider()
    if provider == "openai":
        return OpenAIProvider()
    if provider == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
