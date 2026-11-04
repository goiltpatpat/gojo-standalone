from unittest.mock import AsyncMock, patch
import pytest

from gojo.providers.algolia import (
    HackerNewsAlgoliaProvider,
    _clean_comment_html,
    flatten_comments,
)


def test_clean_comment_html() -> None:
    raw = "<p>First line</p><p>Second line with <a href=\"https://example.com\">link text</a></p>"
    cleaned = _clean_comment_html(raw)
    assert "First line" in cleaned
    assert "Second line with link text (https://example.com)" in cleaned
    assert "<p>" not in cleaned
    assert "</a>" not in cleaned


def test_flatten_comments() -> None:
    sample_children = [
        {
            "id": 101,
            "author": "alice",
            "text": "<p>This is a top-level critique of the system architecture.</p>",
            "children": [
                {
                    "id": 102,
                    "author": "bob",
                    "text": "<p>I agree, and we had this exact outage in production last year.</p>",
                    "children": [],
                }
            ],
        },
        {
            "id": 103,
            "author": "charlie",
            "text": None,  # Deleted comment
            "children": [],
        },
    ]

    flattened = flatten_comments(sample_children)
    assert len(flattened) == 2
    assert flattened[0]["id"] == 101
    assert flattened[0]["author"] == "alice"
    assert flattened[0]["depth"] == 0
    assert flattened[1]["id"] == 102
    assert flattened[1]["author"] == "bob"
    assert flattened[1]["depth"] == 1


@pytest.mark.asyncio
async def test_algolia_provider_fetch_discussion_context() -> None:
    provider = HackerNewsAlgoliaProvider()

    mock_item = {
        "id": 999,
        "title": "New Framework Announcement",
        "author": "founder",
        "children": [
            {
                "id": 1001,
                "author": "engineer1",
                "text": "<p>Have you benchmarked this against Redis?</p>",
                "children": [
                    {
                        "id": 1002,
                        "author": "founder",
                        "text": "<p>Yes, we are 2x faster for read-heavy workloads.</p>",
                        "children": [],
                    }
                ],
            }
        ],
    }

    with patch.object(provider, "get_item", new=AsyncMock(return_value=mock_item)):
        ctx = await provider.fetch_discussion_context("999")
        assert "Community Discussion" in ctx
        assert "[@engineer1]" in ctx
        assert "Have you benchmarked this against Redis?" in ctx
        assert "[@founder]" in ctx

    await provider.close()
