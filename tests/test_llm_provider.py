from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gojo.providers.llm import (
    AutoLLMProvider,
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    _strip_thinking_tags,
)


def _make_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    return resp


class _StubProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        *,
        default_model: str,
        side_effect: list[object] | None = None,
        response: str = "ok",
    ) -> None:
        self._name = name
        self.default_model = default_model
        self.last_used_model: str | None = None
        self.last_used_provider: str | None = None
        self._side_effect = list(side_effect) if side_effect is not None else None
        self._response = response
        self.calls: list[dict[str, object]] = []
        self.closed = False

    @property
    def provider_name(self) -> str:
        return self._name

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        use_model = model or self.default_model
        self.calls.append({"model": model, "effective_model": use_model})
        if self._side_effect:
            nxt = self._side_effect.pop(0)
            if isinstance(nxt, BaseException):
                raise nxt
            self.last_used_model = use_model
            self.last_used_provider = self._name
            return str(nxt)
        self.last_used_model = use_model
        self.last_used_provider = self._name
        return self._response

    async def close(self) -> None:
        self.closed = True


def test_strip_thinking_tags() -> None:
    raw = "<think>Internal deep reasoning about the universe</think>\n# Final Answer"
    cleaned = _strip_thinking_tags(raw)
    assert cleaned == "# Final Answer"


@pytest.mark.asyncio
async def test_ollama_provider_complete_mocked() -> None:
    provider = OllamaProvider(base_url="http://mock-ollama:11434", model="test-model")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": "<think>reasoning</think>## Summary\nOffline AI generated."}}
        ]
    }
    mock_resp.raise_for_status = lambda: None

    try:
        with patch.object(provider, "get_available_models", new=AsyncMock(return_value=["test-model"])):
            with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
                res = await provider.complete(
                    system_prompt="system",
                    user_message="user",
                )
                assert res == "## Summary\nOffline AI generated."
                assert provider.last_used_model == "test-model"
                assert provider.last_used_provider == "ollama"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openai_and_gemini_strip_thinking_pool_and_default_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOJO_HN_MODEL", raising=False)
    monkeypatch.delenv("GOJO_HN_GEMINI_MODEL", raising=False)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_chat_response("<think>hidden</think>Clean OpenAI output")
    )
    mock_client.close = AsyncMock()

    openai_p = OpenAIProvider(api_key="sk-test")
    with patch.object(openai_p, "_get_client", return_value=mock_client):
        out = await openai_p.complete(system_prompt="sys", user_message="usr", model=None)
        assert out == "Clean OpenAI output"
        assert openai_p.last_used_model == "gpt-4o-mini"
        assert openai_p.last_used_provider == "openai"
        assert mock_client.chat.completions.create.await_args.kwargs["model"] == "gpt-4o-mini"

    # Pool reuse + close
    openai_p._client = mock_client
    await openai_p.close()
    mock_client.close.assert_awaited_once()
    assert openai_p._client is None

    mock_gemini_client = MagicMock()
    mock_gemini_client.chat.completions.create = AsyncMock(
        return_value=_make_chat_response("<think>thoughts</think>Clean Gemini output")
    )
    mock_gemini_client.close = AsyncMock()

    gemini_p = GeminiProvider(api_key="gem-test")
    with patch.object(gemini_p, "_get_client", return_value=mock_gemini_client):
        out_g = await gemini_p.complete(system_prompt="sys", user_message="usr", model=None)
        assert out_g == "Clean Gemini output"
        assert gemini_p.last_used_model == "gemini-2.5-flash"
        assert gemini_p.last_used_provider == "gemini"
        assert mock_gemini_client.chat.completions.create.await_args.kwargs["model"] == "gemini-2.5-flash"

    gemini_p._client = mock_gemini_client
    await gemini_p.close()
    mock_gemini_client.close.assert_awaited_once()
    assert gemini_p._client is None


@pytest.mark.asyncio
async def test_auto_llm_provider_resolution_with_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. When OPENAI_API_KEY is present
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    auto = AutoLLMProvider()
    p = await auto.resolve_provider()
    assert isinstance(p, OpenAIProvider)
    assert p.provider_name == "openai"
    # Backward-compatible resolve() alias
    assert await auto.resolve() is p
    await auto.close()

    # 2. When GEMINI_API_KEY is present (no OpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-mock-key")
    auto2 = AutoLLMProvider()
    p2 = await auto2.resolve_provider()
    assert isinstance(p2, GeminiProvider)
    assert p2.provider_name == "gemini"
    await auto2.close()


@pytest.mark.asyncio
async def test_auto_llm_failover_cascade_and_cooldown() -> None:
    p_openai = _StubProvider(
        "openai",
        default_model="gpt-4o-mini",
        side_effect=[RuntimeError("401 invalid api key")],
    )
    p_gemini = _StubProvider(
        "gemini",
        default_model="gemini-2.5-flash",
        response="Gemini rescued",
    )
    p_ollama = _StubProvider(
        "ollama",
        default_model="qwen2.5",
        response="Ollama rescued",
    )

    auto = AutoLLMProvider(
        providers=[p_openai, p_gemini, p_ollama],
        cooldown_seconds=60.0,
        max_retries_per_provider=1,
        retry_backoff_s=0.001,
    )

    # First call: OpenAI fails -> fails over to Gemini with Gemini's own default model
    res1 = await auto.complete(system_prompt="sys", user_message="msg", model="gpt-4o-mini")
    assert res1 == "Gemini rescued"
    assert auto.last_used_provider == "gemini"
    assert auto.last_used_model == "gemini-2.5-flash"
    assert len(p_openai.calls) == 1
    assert p_gemini.calls[0]["model"] is None
    assert p_gemini.calls[0]["effective_model"] == "gemini-2.5-flash"

    # Second call: OpenAI is in cooldown so it is skipped completely
    res2 = await auto.complete(system_prompt="sys", user_message="msg")
    assert res2 == "Gemini rescued"
    assert len(p_openai.calls) == 1
    assert len(p_gemini.calls) == 2

    await auto.close()
    assert p_openai.closed and p_gemini.closed and p_ollama.closed


@pytest.mark.asyncio
async def test_auto_llm_transient_retry_before_failover() -> None:
    p_openai = _StubProvider(
        "openai",
        default_model="gpt-4o-mini",
        side_effect=[httpx.ReadTimeout("transient timeout"), "OpenAI recovered"],
    )
    p_gemini = _StubProvider("gemini", default_model="gemini-2.5-flash", response="Gemini backup")

    auto = AutoLLMProvider(
        providers=[p_openai, p_gemini],
        cooldown_seconds=60.0,
        max_retries_per_provider=1,
        retry_backoff_s=0.001,
    )

    res = await auto.complete(system_prompt="sys", user_message="msg")
    assert res == "OpenAI recovered"
    assert auto.last_used_provider == "openai"
    assert auto.last_used_model == "gpt-4o-mini"
    assert len(p_openai.calls) == 2
    assert len(p_gemini.calls) == 0


@pytest.mark.asyncio
async def test_auto_llm_all_providers_fail_raises_actionable_error() -> None:
    p_openai = _StubProvider("openai", default_model="gpt-4o-mini", side_effect=[RuntimeError("bad key")])
    p_gemini = _StubProvider("gemini", default_model="gemini-2.5-flash", side_effect=[RuntimeError("quota")])

    auto = AutoLLMProvider(
        providers=[p_openai, p_gemini],
        cooldown_seconds=60.0,
        max_retries_per_provider=0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await auto.complete(system_prompt="sys", user_message="msg")

    msg = str(exc_info.value)
    assert "No LLM provider available" in msg
    assert "openai: bad key" in msg
    assert "gemini: quota" in msg
