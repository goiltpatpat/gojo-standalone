from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from gojo.providers.llm import (
    AutoLLMProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    _strip_thinking_tags,
)


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

    with patch.object(provider, "get_available_models", new=AsyncMock(return_value=["test-model"])):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            res = await provider.complete(
                system_prompt="system",
                user_message="user",
            )
            assert res == "## Summary\nOffline AI generated."


@pytest.mark.asyncio
async def test_auto_llm_provider_resolution_with_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. When OPENAI_API_KEY is present
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    auto = AutoLLMProvider()
    p = await auto.resolve_provider()
    assert isinstance(p, OpenAIProvider)
    assert p.provider_name == "openai"

    # 2. When GEMINI_API_KEY is present (no OpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-mock-key")
    auto2 = AutoLLMProvider()
    p2 = await auto2.resolve_provider()
    assert isinstance(p2, GeminiProvider)
    assert p2.provider_name == "gemini"
