"""
Multi-model LLM Provider Gateway for Gojo.
Provides unified, resilient access across OpenAI, Google Gemini, Anthropic,
and zero-cost offline local inference via Ollama (localhost:11434).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import re
from typing import Any

import httpx

OLLAMA_DEFAULT_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _strip_thinking_tags(text: str) -> str:
    """Strip internal reasoning tags (e.g. <think>...</think>) from reasoning models."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    return cleaned.strip()


class LLMProvider(ABC):
    """Abstract interface for LLM completions."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str: ...


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.default_model = os.getenv("GOJO_HN_MODEL", "gpt-4o-mini")

    @property
    def provider_name(self) -> str:
        return "openai"

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        use_model = model or self.default_model

        resp = await client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
        )

        if not resp.choices or not resp.choices[0].message.content:
            raise RuntimeError("OpenAI returned an empty response.")

        return resp.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    """Google Gemini API provider via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.default_model = os.getenv("GOJO_HN_GEMINI_MODEL", "gemini-2.0-flash")

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        use_model = model or self.default_model

        resp = await client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
        )

        if not resp.choices or not resp.choices[0].message.content:
            raise RuntimeError("Gemini returned an empty response.")

        return resp.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    """Local offline inference provider via Ollama."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("GOJO_HN_OLLAMA_MODEL")

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def get_available_models(self) -> list[str]:
        """Query installed models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                if r.status_code == 200:
                    data = r.json()
                    return [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
        return []

    async def is_available(self) -> bool:
        """Check if local Ollama daemon is active."""
        models = await self.get_available_models()
        return len(models) > 0

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        available = await self.get_available_models()
        if not available:
            raise RuntimeError(
                f"Ollama server at '{self.base_url}' is unreachable or has no models installed."
            )

        target_model: str | None = None
        if model and model in available:
            target_model = model
        elif self.model and self.model in available:
            target_model = self.model
        elif available:
            target_model = available[0]

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Try /v1/chat/completions (OpenAI compatible endpoint)
            v1_url = f"{self.base_url}/v1/chat/completions"
            v1_payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }
            try:
                r = await client.post(v1_url, json=v1_payload)
                if r.status_code == 200:
                    data = r.json()
                    if hasattr(data, "__await__"):
                        data = await data
                    choices = data.get("choices") or []
                    if choices and choices[0].get("message"):
                        msg = choices[0]["message"].get("content", "")
                        cleaned = _strip_thinking_tags(msg)
                        if cleaned:
                            return cleaned

                # Fallback to /api/generate
                gen_url = f"{self.base_url}/api/generate"
                full_prompt = f"{system_prompt}\n\n{user_message}"
                gen_payload = {
                    "model": target_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.3},
                }
                r2 = await client.post(gen_url, json=gen_payload)
                r2.raise_for_status()
                data2 = r2.json()
                if hasattr(data2, "__await__"):
                    data2 = await data2
                msg2 = data2.get("response", "")
                cleaned2 = _strip_thinking_tags(msg2)
                if not cleaned2:
                    raise RuntimeError("Ollama returned an empty response.")
                return cleaned2
            except Exception as e:
                raise RuntimeError(f"Ollama generation failed: {e}") from e


class AutoLLMProvider(LLMProvider):
    """
    Automatic provider selector:
    Probes environment variables and local services in order:
    1. OpenAI (if OPENAI_API_KEY)
    2. Gemini (if GEMINI_API_KEY)
    3. Ollama (if running on localhost:11434 with models)
    """

    def __init__(self) -> None:
        self._cached_provider: LLMProvider | None = None

    @property
    def provider_name(self) -> str:
        return self._cached_provider.provider_name if self._cached_provider else "auto"

    async def resolve_provider(self) -> LLMProvider:
        if self._cached_provider is not None:
            return self._cached_provider

        # 1. OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self._cached_provider = OpenAIProvider()
            return self._cached_provider

        # 2. Gemini
        if os.getenv("GEMINI_API_KEY"):
            self._cached_provider = GeminiProvider()
            return self._cached_provider

        # 3. Local Ollama probe
        ollama = OllamaProvider()
        if await ollama.is_available():
            self._cached_provider = ollama
            return self._cached_provider

        # If none found, return helpful diagnostic
        raise RuntimeError(
            "No LLM provider available. Please do one of the following:\n"
            "  1. Set OPENAI_API_KEY in environment or .env\n"
            "  2. Set GEMINI_API_KEY in environment or .env\n"
            "  3. Start Ollama locally (http://localhost:11434) with any model (e.g. 'ollama run qwen2.5')\n"
            "  4. Provide a custom_summarizer async function to GojoHNService."
        )

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        provider = await self.resolve_provider()
        return await provider.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            max_tokens=max_tokens,
        )
