"""
Multi-model LLM Provider Gateway for Gojo.
Provides unified, resilient access across OpenAI, Google Gemini, Anthropic,
and zero-cost offline local inference via Ollama (localhost:11434).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import os
import re
import time
from typing import Any

import httpx

OLLAMA_DEFAULT_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

_NO_PROVIDER_GUIDANCE = (
    "No LLM provider available. Please do one of the following:\n"
    "  1. Set OPENAI_API_KEY in environment or .env\n"
    "  2. Set GEMINI_API_KEY in environment or .env\n"
    "  3. Start Ollama locally (http://localhost:11434) with any model (e.g. 'ollama run qwen2.5')\n"
    "  4. Provide a custom_summarizer async function to GojoHNService."
)


def _strip_thinking_tags(text: str) -> str:
    """Strip internal reasoning tags (e.g. <think>...</think>) from reasoning models."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    return cleaned.strip()


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if an exception represents a transient network/rate-limit/5xx failure."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        code = exc.response.status_code
        if code == 429 or 500 <= code < 600:
            return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_transient_error(cause)
    return False


class LLMProvider(ABC):
    """Abstract interface for LLM completions."""

    default_model: str | None = None
    last_used_model: str | None = None
    last_used_provider: str | None = None

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

    async def close(self) -> None:
        """Release any pooled network resources held by this provider."""
        return None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider with pooled AsyncOpenAI client."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.default_model: str = os.getenv("GOJO_HN_MODEL", "gpt-4o-mini")
        self.last_used_model: str | None = None
        self.last_used_provider: str | None = None
        self._client: Any | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            close_fn = getattr(self._client, "close", None)
            if close_fn is not None:
                res = close_fn()
                if hasattr(res, "__await__"):
                    await res
            self._client = None

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

        client = self._get_client()
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

        cleaned = _strip_thinking_tags(resp.choices[0].message.content)
        if not cleaned:
            raise RuntimeError("OpenAI returned an empty response.")

        self.last_used_model = use_model
        self.last_used_provider = self.provider_name
        return cleaned


class GeminiProvider(LLMProvider):
    """Google Gemini API provider via OpenAI-compatible endpoint with pooled client."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.default_model: str = os.getenv("GOJO_HN_GEMINI_MODEL", "gemini-2.5-flash")
        self.last_used_model: str | None = None
        self.last_used_provider: str | None = None
        self._client: Any | None = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            close_fn = getattr(self._client, "close", None)
            if close_fn is not None:
                res = close_fn()
                if hasattr(res, "__await__"):
                    await res
            self._client = None

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

        client = self._get_client()
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

        cleaned = _strip_thinking_tags(resp.choices[0].message.content)
        if not cleaned:
            raise RuntimeError("Gemini returned an empty response.")

        self.last_used_model = use_model
        self.last_used_provider = self.provider_name
        return cleaned


class OllamaProvider(LLMProvider):
    """Local offline inference provider via Ollama with pooled httpx.AsyncClient."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("GOJO_HN_OLLAMA_MODEL")
        self.default_model: str | None = self.model
        self.last_used_model: str | None = None
        self.last_used_provider: str | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=120.0,
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def get_available_models(self) -> list[str]:
        """Query installed models from Ollama."""
        try:
            client = self._get_client()
            r = await client.get(f"{self.base_url}/api/tags", timeout=3.0)
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

        client = self._get_client()
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
                        self.last_used_model = target_model
                        self.last_used_provider = self.provider_name
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
            self.last_used_model = target_model
            self.last_used_provider = self.provider_name
            return cleaned2
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e


class AutoLLMProvider(LLMProvider):
    """
    Automatic provider selector with runtime failover cascade and cooldown circuit breaker.
    Probes environment variables and local services in order:
    1. OpenAI (if OPENAI_API_KEY)
    2. Gemini (if GEMINI_API_KEY)
    3. Ollama (if running on localhost:11434 with models)
    """

    def __init__(
        self,
        *,
        providers: list[LLMProvider] | None = None,
        cooldown_seconds: float = 60.0,
        max_retries_per_provider: int = 1,
        retry_backoff_s: float = 0.25,
    ) -> None:
        self._explicit_providers: list[LLMProvider] | None = providers
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.max_retries_per_provider = max(0, max_retries_per_provider)
        self.retry_backoff_s = max(0.0, retry_backoff_s)

        self._cached_provider: LLMProvider | None = None
        self._pool: dict[str, LLMProvider] = {}
        self._cooldown_until: dict[str, float] = {}
        self.last_used_model: str | None = None
        self.last_used_provider: str | None = None

    @property
    def provider_name(self) -> str:
        if self.last_used_provider:
            return self.last_used_provider
        return self._cached_provider.provider_name if self._cached_provider else "auto"

    def _is_in_cooldown(self, provider_name: str) -> bool:
        until = self._cooldown_until.get(provider_name, 0.0)
        if until <= 0.0:
            return False
        if time.monotonic() < until:
            return True
        self._cooldown_until.pop(provider_name, None)
        return False

    def _mark_failed(self, provider_name: str) -> None:
        if self.cooldown_seconds > 0.0:
            self._cooldown_until[provider_name] = time.monotonic() + self.cooldown_seconds

    def _clear_cooldown(self, provider_name: str) -> None:
        self._cooldown_until.pop(provider_name, None)

    def _get_or_create_openai(self) -> OpenAIProvider:
        p = self._pool.get("openai")
        if not isinstance(p, OpenAIProvider):
            p = OpenAIProvider()
            self._pool["openai"] = p
        return p

    def _get_or_create_gemini(self) -> GeminiProvider:
        p = self._pool.get("gemini")
        if not isinstance(p, GeminiProvider):
            p = GeminiProvider()
            self._pool["gemini"] = p
        return p

    def _get_or_create_ollama(self) -> OllamaProvider:
        p = self._pool.get("ollama")
        if not isinstance(p, OllamaProvider):
            p = OllamaProvider()
            self._pool["ollama"] = p
        return p

    async def close(self) -> None:
        """Close all instantiated provider clients."""
        seen: set[int] = set()
        to_close: list[LLMProvider] = []
        for p in list(self._pool.values()) + list(self._explicit_providers or []):
            if id(p) not in seen:
                seen.add(id(p))
                to_close.append(p)
        for p in to_close:
            await p.close()
        self._pool.clear()
        self._cached_provider = None

    async def resolve_provider(self) -> LLMProvider:
        if self._cached_provider is not None and not self._is_in_cooldown(
            self._cached_provider.provider_name
        ):
            return self._cached_provider

        if self._explicit_providers is not None:
            for p in self._explicit_providers:
                if not self._is_in_cooldown(p.provider_name):
                    self._cached_provider = p
                    return p
            raise RuntimeError(_NO_PROVIDER_GUIDANCE)

        # 1. OpenAI
        if os.getenv("OPENAI_API_KEY") and not self._is_in_cooldown("openai"):
            self._cached_provider = self._get_or_create_openai()
            return self._cached_provider

        # 2. Gemini
        if os.getenv("GEMINI_API_KEY") and not self._is_in_cooldown("gemini"):
            self._cached_provider = self._get_or_create_gemini()
            return self._cached_provider

        # 3. Local Ollama probe
        if not self._is_in_cooldown("ollama"):
            ollama = self._get_or_create_ollama()
            if await ollama.is_available():
                self._cached_provider = ollama
                return self._cached_provider

        raise RuntimeError(_NO_PROVIDER_GUIDANCE)

    # Backward-compatible alias
    resolve = resolve_provider

    async def _iter_candidates(self) -> list[LLMProvider]:
        """Collect configured providers in cascade order without probing Ollama unless needed."""
        if self._explicit_providers is not None:
            return list(self._explicit_providers)

        candidates: list[LLMProvider] = []
        if os.getenv("OPENAI_API_KEY"):
            candidates.append(self._get_or_create_openai())
        if os.getenv("GEMINI_API_KEY"):
            candidates.append(self._get_or_create_gemini())
        return candidates

    async def _try_provider(
        self,
        provider: LLMProvider,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None,
        max_tokens: int,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries_per_provider + 1):
            try:
                result = await provider.complete(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=model,
                    max_tokens=max_tokens,
                )
                self._clear_cooldown(provider.provider_name)
                self._cached_provider = provider
                self.last_used_provider = provider.provider_name
                self.last_used_model = (
                    provider.last_used_model
                    or model
                    or getattr(provider, "default_model", None)
                )
                return result
            except Exception as exc:
                last_exc = exc
                if _is_transient_error(exc) and attempt < self.max_retries_per_provider:
                    await asyncio.sleep(self.retry_backoff_s * (2**attempt))
                    continue
                self._mark_failed(provider.provider_name)
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Provider '{provider.provider_name}' failed unexpectedly.")

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        errors: list[str] = []

        candidates = await self._iter_candidates()

        # Evaluate cloud / explicit candidates first
        for idx, provider in enumerate(candidates):
            name = provider.provider_name
            if self._is_in_cooldown(name):
                errors.append(f"{name}: skipped (cooldown active)")
                continue

            effective_model = model if idx == 0 else None
            try:
                return await self._try_provider(
                    provider,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=effective_model,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        # Lazy Ollama fallback (only probed if explicit providers were not injected and earlier providers failed/unavailable)
        if self._explicit_providers is None:
            if self._is_in_cooldown("ollama"):
                errors.append("ollama: skipped (cooldown active)")
            else:
                ollama = self._get_or_create_ollama()
                if await ollama.is_available():
                    effective_model = model if len(candidates) == 0 else None
                    try:
                        return await self._try_provider(
                            ollama,
                            system_prompt=system_prompt,
                            user_message=user_message,
                            model=effective_model,
                            max_tokens=max_tokens,
                        )
                    except Exception as exc:
                        errors.append(f"ollama: {exc}")

        detail = "\nProvider errors:\n" + "\n".join(f"  - {e}" for e in errors) if errors else ""
        raise RuntimeError(f"{_NO_PROVIDER_GUIDANCE}{detail}")
