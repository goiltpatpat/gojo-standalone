from unittest.mock import AsyncMock, patch
import pytest

from gojo.service import GojoHNService, _prefer_main_html, _strip_html
from gojo.storage.memory import MemoryStorage
from gojo.types import DigestFeedItem


def test_html_helpers() -> None:
    html = "<html><body><script>alert(1)</script><article><p>Crucial Content Here</p></article></body></html>"
    preferred = _prefer_main_html(html)
    assert "<script>" in html
    # _prefer_main_html keeps full article if > 400 chars, else returns html
    stripped = _strip_html(html)
    assert "alert" not in stripped
    assert "Crucial Content Here" in stripped


@pytest.mark.asyncio
async def test_service_list_feed_and_cache() -> None:
    storage = MemoryStorage()
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.list_page.return_value = (
        [
            DigestFeedItem(
                feed_provider="hn",
                external_id="101",
                title="Python 3.14 Released",
                score=300,
            )
        ],
        False,
    )

    svc = GojoHNService(feed_source=mock_source, storage=storage)

    # First fetch (hits source)
    res1 = await svc.list_feed(feed_id="hn_top", page=1)
    assert res1.completion.status == "success"
    assert res1.completion.cache_layer == "origin"
    assert len(res1.items) == 1
    assert mock_source.list_page.call_count == 1

    # Second fetch (hits cache)
    res2 = await svc.list_feed(feed_id="hn_top", page=1)
    assert res2.completion.status == "success"
    assert res2.completion.cache_layer == "sqlite" or res2.completion.cache_layer == "memory"
    assert len(res2.items) == 1
    assert mock_source.list_page.call_count == 1  # Not called again!


@pytest.mark.asyncio
async def test_service_summarize_idempotent() -> None:
    storage = MemoryStorage()
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.get_item.return_value = {
        "id": 555,
        "title": "Quantum Computing Milestone",
        "url": "https://example.com/quantum",
    }

    async def fake_summarizer(**kwargs):
        return (
            "## Summary\n"
            "Physicists achieved quantum supremacy.\n\n"
            "### Why it matters\n"
            "It speeds up computation exponentially.\n\n"
            "### Key takeaway\n"
            "Qubits are stable.\n\n"
            "### Gojo note\n"
            "Watch out for error correction overhead.\n"
        )

    svc = GojoHNService(
        feed_source=mock_source,
        storage=storage,
        custom_summarizer=fake_summarizer,
    )

    # First summarize call (generates)
    res1 = await svc.summarize(external_id="555", summary_type="quick")
    assert res1.completion.status == "success"
    assert res1.completion.cache_layer == "origin"
    assert res1.artifact.title == "Quantum Computing Milestone"
    assert res1.artifact.summary_extra.get("key_takeaway") == "Qubits are stable."

    # Second summarize call (idempotent cache hit)
    res2 = await svc.summarize(external_id="555", summary_type="quick")
    assert res2.completion.status == "cached"
    assert res2.artifact.fingerprint == res1.artifact.fingerprint
    assert res2.artifact.id == res1.artifact.id
