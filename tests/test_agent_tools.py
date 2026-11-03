from unittest.mock import AsyncMock
import pytest

from gojo.agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    filter_items_by_query,
    format_search_for_agent,
    format_summary_for_agent,
    tool_get_hn_story,
    tool_search_hn,
    tool_summarize_hn,
)
from gojo.types import (
    DigestArtifact,
    DigestCompletion,
    DigestFeedItem,
    DigestFeedResponse,
    DigestSummaryResponse,
)


def test_filter_items_by_query() -> None:
    items = [
        DigestFeedItem(feed_provider="hn", external_id="1", title="Show HN: A Fast Rust Terminal"),
        DigestFeedItem(feed_provider="hn", external_id="2", title="Ask HN: Best Books on Systems?"),
        DigestFeedItem(feed_provider="hn", external_id="3", title="SQLite 3.45 Released"),
    ]
    filtered = filter_items_by_query(items, "rust")
    assert len(filtered) == 1
    assert filtered[0].external_id == "1"

    all_items = filter_items_by_query(items, None)
    assert len(all_items) == 3


def test_agent_tool_definitions_valid() -> None:
    assert len(HN_AGENT_TOOL_DEFINITIONS) == 3
    names = [t["function"]["name"] for t in HN_AGENT_TOOL_DEFINITIONS]
    assert "hn_search" in names
    assert "hn_summarize" in names
    assert "hn_get_story" in names


@pytest.mark.asyncio
async def test_tool_search_hn() -> None:
    mock_svc = AsyncMock()
    mock_svc.list_feed.return_value = DigestFeedResponse(
        feed_id="hn_top",
        page=1,
        items=[
            DigestFeedItem(
                feed_provider="hn",
                external_id="10",
                title="Antigravity v2",
                score=100,
                descendants=25,
            )
        ],
        has_more=False,
        completion=DigestCompletion(status="success", cache_layer="origin"),
    )

    out = await tool_search_hn(mock_svc, feed_id="hn_top", page=1, query="antigravity")
    assert out["feed_id"] == "hn_top"
    assert out["total_returned"] == 1
    assert out["items"][0]["title"] == "Antigravity v2"

    fmt = format_search_for_agent(out)
    assert "Antigravity v2" in fmt
    assert "Score: 100" in fmt


@pytest.mark.asyncio
async def test_tool_summarize_hn() -> None:
    mock_svc = AsyncMock()
    mock_svc.summarize.return_value = DigestSummaryResponse(
        artifact=DigestArtifact(
            fingerprint="abc" * 21 + "a",
            feed_provider="hn",
            external_id="10",
            title="Antigravity v2",
            locale="en",
            prompt_version="v5",
            summary_markdown="## Summary\nGreat release.",
        ),
        completion=DigestCompletion(status="success", cache_layer="origin"),
    )

    out = await tool_summarize_hn(mock_svc, external_id="10", summary_type="quick")
    assert out["artifact"]["title"] == "Antigravity v2"

    fmt = format_summary_for_agent(out)
    assert "Antigravity v2" in fmt
    assert "Great release" in fmt
