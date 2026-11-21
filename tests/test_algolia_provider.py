from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from gojo.providers.algolia import (
    ALGOLIA_SEARCH_BY_DATE_URL,
    ALGOLIA_SEARCH_URL,
    HackerNewsAlgoliaProvider,
    _clean_comment_html,
    flatten_comments,
    score_comment_subtree,
)


def test_clean_comment_html() -> None:
    raw = (
        "<p>First line &amp; &#x27;quoted&#x27;</p>"
        "<p>Second line with <a href=\"https://example.com\">link text</a> &lt;code&gt;</p>"
    )
    cleaned = _clean_comment_html(raw)
    assert "First line & 'quoted'" in cleaned
    assert "Second line with link text (https://example.com) <code>" in cleaned
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


def test_score_comment_subtree_prefers_rich_nested_thread() -> None:
    shallow = {
        "id": 1,
        "author": "quick",
        "points": 5,
        "text": "<p>Nice project, thanks for sharing!</p>",
        "children": [],
    }
    deep = {
        "id": 2,
        "author": "architect",
        "points": 12,
        "text": "<p>We tested this memory layout under 50k RPS and hit NUMA contention.</p>",
        "children": [
            {
                "id": 3,
                "author": "maintainer",
                "points": 8,
                "text": "<p>Good catch — pinning worker arenas per socket eliminates that stall.</p>",
                "children": [
                    {
                        "id": 4,
                        "author": "architect",
                        "points": 4,
                        "text": "<p>Confirmed, p99 dropped from 42ms to 6ms with per-socket arenas.</p>",
                        "children": [],
                    }
                ],
            }
        ],
    }

    s_shallow = score_comment_subtree(shallow)
    s_deep = score_comment_subtree(deep)
    assert s_deep > s_shallow
    assert s_deep[0] == 2  # total_descendants
    assert s_deep[2] == 3  # depth_diversity
    assert s_deep[3] == 24  # points


@pytest.mark.asyncio
async def test_algolia_provider_fetch_discussion_context_subtree_ranking_and_paths() -> None:
    provider = HackerNewsAlgoliaProvider()

    mock_item = {
        "id": 999,
        "title": "New Framework Announcement",
        "author": "founder",
        "children": [
            {
                "id": 2001,
                "author": "short_commenter",
                "text": "<p>Looks interesting, will try later.</p>",
                "children": [],
            },
            {
                "id": 1001,
                "author": "engineer1",
                "text": "<p>Have you benchmarked this against Redis under high contention?</p>",
                "children": [
                    {
                        "id": 1002,
                        "author": "founder",
                        "text": "<p>Yes, we are 2x faster for read-heavy workloads with zero GC pauses.</p>",
                        "children": [],
                    },
                    {
                        "id": 1003,
                        "author": "skeptic",
                        "text": "<p>Your benchmark disables fsync while Redis had AOF enabled.</p>",
                        "children": [],
                    },
                ],
            },
        ],
    }

    with patch.object(provider, "get_item", new=AsyncMock(return_value=mock_item)):
        ctx = await provider.fetch_discussion_context("999")
        assert "Community Discussion" in ctx
        # Richer subtree (engineer1) should win [1] ahead of short_commenter [2]
        assert "[1] @engineer1:" in ctx
        assert "  [1.1] @founder:" in ctx
        assert "  [1.2] @skeptic:" in ctx
        assert "[2] @short_commenter:" in ctx
        assert ctx.index("[1] @engineer1:") < ctx.index("[1.1] @founder:")
        assert ctx.index("[1.1] @founder:") < ctx.index("[2] @short_commenter:")

    await provider.close()


@pytest.mark.asyncio
async def test_algolia_provider_search_stories_relevance_and_date() -> None:
    provider = HackerNewsAlgoliaProvider()

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "hits": [
            {
                "objectID": "40123456",
                "title": "SQLite on the Edge",
                "url": "https://example.com/sqlite-edge",
                "author": "db_eng",
                "points": 321,
                "num_comments": 144,
                "created_at_i": 1700000000,
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_resp)
    mock_client.is_closed = False

    with patch.object(provider, "_get_client", new=AsyncMock(return_value=mock_client)):
        items = await provider.search_stories("sqlite", limit=10, sort_by="relevance")
        assert len(items) == 1
        assert items[0].external_id == "40123456"
        assert items[0].title == "SQLite on the Edge"
        assert items[0].score == 321
        assert items[0].descendants == 144
        mock_client.get.assert_called_with(
            ALGOLIA_SEARCH_URL,
            params={"query": "sqlite", "tags": "story", "hitsPerPage": 10},
        )

        items_date = await provider.search_stories("sqlite", limit=5, sort_by="date")
        assert len(items_date) == 1
        mock_client.get.assert_called_with(
            ALGOLIA_SEARCH_BY_DATE_URL,
            params={"query": "sqlite", "tags": "story", "hitsPerPage": 5},
        )

    await provider.close()
