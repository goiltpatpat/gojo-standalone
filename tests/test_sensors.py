from datetime import datetime, timezone
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gojo.providers import FeedSource, GitHubTrendingFeedSource, LobstersFeedSource
from gojo.types import ALL_SUPPORTED_FEED_IDS, DigestFeedId, DigestFeedItem, FeedItem


def test_types_and_provider_exports() -> None:
    expected_feeds = (
        "hn_top",
        "hn_new",
        "hn_best",
        "hn_ask",
        "hn_show",
        "lobsters_hot",
        "lobsters_new",
        "github_trending",
    )
    assert ALL_SUPPORTED_FEED_IDS == expected_feeds
    assert set(get_args(DigestFeedId)) == set(expected_feeds)
    assert FeedItem is DigestFeedItem
    assert issubclass(LobstersFeedSource, FeedSource)
    assert issubclass(GitHubTrendingFeedSource, FeedSource)
    assert LobstersFeedSource.provider_id == "lobsters"
    assert GitHubTrendingFeedSource.provider_id == "github"


# ---------------------------------------------------------------------------
# Lobste.rs Sensor Tests
# ---------------------------------------------------------------------------


def test_lobsters_feed_endpoints() -> None:
    source = LobstersFeedSource()
    assert source._list_url("hot") == "https://lobste.rs/hottest.json"
    assert source._list_url("hottest") == "https://lobste.rs/hottest.json"
    assert source._list_url("lobsters_hot") == "https://lobste.rs/hottest.json"
    assert source._list_url("new") == "https://lobste.rs/newest.json"
    assert source._list_url("newest") == "https://lobste.rs/newest.json"
    assert source._list_url("lobsters_new") == "https://lobste.rs/newest.json"
    assert source._list_url("unknown") == "https://lobste.rs/hottest.json"


@pytest.mark.asyncio
async def test_lobsters_list_page_hot_and_new_pagination() -> None:
    source = LobstersFeedSource()
    sample_stories = [
        {
            "short_id": "abc123",
            "title": "Writing a Minimal Kernel in Rust",
            "url": "https://example.org/rust-kernel",
            "comments_url": "https://lobste.rs/s/abc123/writing_minimal_kernel",
            "score": 87,
            "comment_count": 34,
            "submitter_user": "alice",
            "created_at": "2026-11-20T12:30:00.000-06:00",
        },
        {
            "short_id": "def456",
            "title": "Ask Lobsters: Best SQLite Tricks?",
            "url": "",
            "comments_url": "https://lobste.rs/s/def456/ask_lobsters_sqlite",
            "score": 42,
            "comment_count": 19,
            "submitter_user": {"username": "bob"},
            "created_at": "2026-11-20T18:00:00Z",
            "description": "Share your favorite pragmas.",
        },
        {
            "short_id": "ghi789",
            "title": "Third Story Beyond Page Size",
            "url": "https://example.org/third",
            "comments_url": "https://lobste.rs/s/ghi789/third",
            "score": 15,
            "comment_count": 3,
            "submitter_user": "carol",
            "created_at": "2026-11-20T19:00:00Z",
        },
    ]

    try:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_stories
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            items, has_more = await source.list_page(feed_variant="lobsters_hot", page=2, page_size=2)
            assert mock_get.call_args.args[0] == "https://lobste.rs/hottest.json?page=2"
            assert len(items) == 2
            assert has_more is True

            first = items[0]
            assert first.feed_provider == "lobsters"
            assert first.external_id == "abc123"
            assert first.title == "Writing a Minimal Kernel in Rust"
            assert first.url == "https://example.org/rust-kernel"
            assert first.hn_url == "https://lobste.rs/s/abc123/writing_minimal_kernel"
            assert first.score == 87
            assert first.descendants == 34
            assert first.author == "alice"
            assert first.by == "alice"
            assert isinstance(first.created_at, datetime)
            assert first.time == int(first.created_at.timestamp())
            assert first.raw == sample_stories[0]

            # Empty url falls back to comments_url; dict submitter_user supported
            second = items[1]
            assert second.external_id == "def456"
            assert second.url == "https://lobste.rs/s/def456/ask_lobsters_sqlite"
            assert second.hn_url == "https://lobste.rs/s/def456/ask_lobsters_sqlite"
            assert second.author == "bob"
            assert second.by == "bob"
            assert second.text == "Share your favorite pragmas."
            assert second.created_at == datetime(2026, 11, 20, 18, 0, 0, tzinfo=timezone.utc)

            # New variant with page_size larger than returned list (<25) -> has_more False
            items_new, has_more_new = await source.list_page(feed_variant="new", page=1, page_size=10)
            assert mock_get.call_args.args[0] == "https://lobste.rs/newest.json?page=1"
            assert len(items_new) == 3
            assert has_more_new is False
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_lobsters_get_item() -> None:
    source = LobstersFeedSource()
    payload = {"short_id": "xyz999", "title": "Single Lobsters Story", "score": 55}

    try:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = payload
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            result = await source.get_item(external_id="xyz999")
            assert result == payload
            assert mock_get.call_args.args[0] == "https://lobste.rs/s/xyz999.json"
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_lobsters_client_pooling_and_close() -> None:
    source = LobstersFeedSource()
    c1 = source._get_client()
    c2 = source._get_client()
    assert c1 is c2
    assert not c1.is_closed

    await source.close()
    assert c1.is_closed
    assert source._client is None


@pytest.mark.asyncio
async def test_lobsters_transient_retry_and_no_retry_404() -> None:
    source = LobstersFeedSource(max_retries=2, retry_backoff_s=0.001)
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {"short_id": "ok1", "title": "Recovered"}
    ok_resp.raise_for_status.return_value = None

    err_429 = MagicMock()
    err_429.status_code = 429

    try:
        with patch(
            "httpx.AsyncClient.get",
            new=AsyncMock(side_effect=[httpx.ReadTimeout("timeout"), err_429, ok_resp]),
        ) as mock_get:
            item = await source.get_item(external_id="ok1")
            assert item == {"short_id": "ok1", "title": "Recovered"}
            assert mock_get.await_count == 3

        req = httpx.Request("GET", "https://lobste.rs/s/missing.json")
        resp_404 = httpx.Response(404, request=req)
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=resp_404)) as mock_get_404:
            with pytest.raises(httpx.HTTPStatusError):
                await source.get_item(external_id="missing")
            assert mock_get_404.await_count == 1
    finally:
        await source.close()


# ---------------------------------------------------------------------------
# GitHub Trending Sensor Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_github_trending_list_page() -> None:
    source = GitHubTrendingFeedSource()
    payload = {
        "total_count": 120,
        "items": [
            {
                "full_name": "acme/hyper-db",
                "description": "Fast embedded columnar engine",
                "html_url": "https://github.com/acme/hyper-db",
                "stargazers_count": 1450,
                "forks_count": 98,
                "owner": {"login": "acme"},
                "created_at": "2026-11-18T09:15:00Z",
            },
            {
                "full_name": "solo/no-desc-repo",
                "description": None,
                "html_url": "https://github.com/solo/no-desc-repo",
                "stargazers_count": 320,
                "forks_count": 12,
                "owner": {"login": "solo"},
                "created_at": "2026-11-19T14:00:00Z",
            },
        ],
    }

    try:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = payload
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            items, has_more = await source.list_page(feed_variant="github_trending", page=1, page_size=2)
            assert mock_get.call_args.args[0] == "https://api.github.com/search/repositories"
            params = mock_get.call_args.kwargs["params"]
            assert params["q"].startswith("created:>=")
            assert params["sort"] == "stars"
            assert params["order"] == "desc"
            assert params["per_page"] == 2
            assert params["page"] == 1

            assert len(items) == 2
            assert has_more is True

            first = items[0]
            assert first.feed_provider == "github"
            assert first.external_id == "acme/hyper-db"
            assert first.title == "acme/hyper-db: Fast embedded columnar engine"
            assert first.url == "https://github.com/acme/hyper-db"
            assert first.hn_url == "https://github.com/acme/hyper-db"
            assert first.score == 1450
            assert first.descendants == 98
            assert first.author == "acme"
            assert first.by == "acme"
            assert first.created_at == datetime(2026, 11, 18, 9, 15, 0, tzinfo=timezone.utc)
            assert first.time == int(first.created_at.timestamp())
            assert first.raw == payload["items"][0]

            second = items[1]
            assert second.external_id == "solo/no-desc-repo"
            assert second.title == "solo/no-desc-repo"
            assert second.author == "solo"
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_github_trending_token_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    source_no_token = GitHubTrendingFeedSource()
    try:
        client = source_no_token._get_client()
        assert client.headers["Accept"] == "application/vnd.github+json"
        assert "Authorization" not in client.headers
    finally:
        await source_no_token.close()

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret_token_123")
    source_with_env = GitHubTrendingFeedSource()
    try:
        client = source_with_env._get_client()
        assert client.headers["Accept"] == "application/vnd.github+json"
        assert client.headers["Authorization"] == "Bearer ghp_secret_token_123"
    finally:
        await source_with_env.close()

    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    source_blank_env = GitHubTrendingFeedSource()
    try:
        client = source_blank_env._get_client()
        assert "Authorization" not in client.headers
    finally:
        await source_blank_env.close()


@pytest.mark.asyncio
async def test_github_trending_get_item() -> None:
    source = GitHubTrendingFeedSource()
    repo_payload = {
        "full_name": "acme/hyper-db",
        "html_url": "https://github.com/acme/hyper-db",
        "stargazers_count": 1500,
    }

    try:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = repo_payload
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            result = await source.get_item(external_id="acme/hyper-db")
            assert result == repo_payload
            assert mock_get.call_args.args[0] == "https://api.github.com/repos/acme/hyper-db"
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_github_trending_client_pooling_and_close() -> None:
    source = GitHubTrendingFeedSource()
    c1 = source._get_client()
    c2 = source._get_client()
    assert c1 is c2
    assert not c1.is_closed

    await source.close()
    assert c1.is_closed
    assert source._client is None


@pytest.mark.asyncio
async def test_github_trending_transient_retry_and_no_retry_404() -> None:
    source = GitHubTrendingFeedSource(max_retries=2, retry_backoff_s=0.001)
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {"full_name": "acme/ok", "stargazers_count": 10}
    ok_resp.raise_for_status.return_value = None

    err_502 = MagicMock()
    err_502.status_code = 502

    try:
        with patch(
            "httpx.AsyncClient.get",
            new=AsyncMock(side_effect=[httpx.ConnectTimeout("timeout"), err_502, ok_resp]),
        ) as mock_get:
            item = await source.get_item(external_id="acme/ok")
            assert item == {"full_name": "acme/ok", "stargazers_count": 10}
            assert mock_get.await_count == 3

        req = httpx.Request("GET", "https://api.github.com/repos/acme/missing")
        resp_404 = httpx.Response(404, request=req)
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=resp_404)) as mock_get_404:
            with pytest.raises(httpx.HTTPStatusError):
                await source.get_item(external_id="acme/missing")
            assert mock_get_404.await_count == 1
    finally:
        await source.close()
