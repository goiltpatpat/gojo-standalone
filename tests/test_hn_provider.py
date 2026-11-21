import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gojo.providers.hn import HackerNewsFeedSource


@pytest.mark.asyncio
async def test_hn_feed_source_to_feed_item() -> None:
    source = HackerNewsFeedSource()
    raw = {
        "id": 123456,
        "title": "Show HN: My New Compiler",
        "url": "https://github.com/example/compiler",
        "score": 150,
        "by": "compilerdev",
        "time": 1700000000,
        "descendants": 42,
        "type": "story",
    }
    item = source._to_feed_item(raw)
    assert item.external_id == "123456"
    assert item.title == "Show HN: My New Compiler"
    assert item.score == 150
    assert item.descendants == 42
    assert item.feed_provider == "hn"


@pytest.mark.asyncio
async def test_hn_list_page_mocked() -> None:
    source = HackerNewsFeedSource()

    fake_top_ids = [101, 102, 103, 104, 105]
    fake_item = {
        "id": 101,
        "title": "Story 101",
        "url": "https://example.com/101",
        "score": 10,
    }

    async def mock_fetch_item(client, eid):
        return {**fake_item, "id": int(eid), "title": f"Story {eid}"}

    try:
        with patch.object(source, "_fetch_item", side_effect=mock_fetch_item):
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_resp = MagicMock()
                mock_resp.json.return_value = fake_top_ids
                mock_resp.raise_for_status.return_value = None
                mock_get.return_value = mock_resp

                items, has_more = await source.list_page(feed_variant="top", page=1, page_size=2)
                assert len(items) == 2
                assert items[0].external_id == "101"
                assert items[1].external_id == "102"
                assert has_more is True
    finally:
        await source.close()


def test_hn_extra_feed_endpoints() -> None:
    source = HackerNewsFeedSource()
    assert source._list_url("top").endswith("/v0/topstories.json")
    assert source._list_url("hn_top").endswith("/v0/topstories.json")
    assert source._list_url("new").endswith("/v0/newstories.json")
    assert source._list_url("hn_new").endswith("/v0/newstories.json")
    assert source._list_url("best").endswith("/v0/beststories.json")
    assert source._list_url("hn_best").endswith("/v0/beststories.json")
    assert source._list_url("ask").endswith("/v0/askstories.json")
    assert source._list_url("hn_ask").endswith("/v0/askstories.json")
    assert source._list_url("show").endswith("/v0/showstories.json")
    assert source._list_url("hn_show").endswith("/v0/showstories.json")


@pytest.mark.asyncio
async def test_hn_client_pooling_and_close() -> None:
    source = HackerNewsFeedSource()
    c1 = source._get_client()
    c2 = source._get_client()
    assert c1 is c2
    assert not c1.is_closed

    await source.close()
    assert c1.is_closed
    assert source._client is None


@pytest.mark.asyncio
async def test_hn_semaphore_bounds_concurrency() -> None:
    source = HackerNewsFeedSource(max_concurrency=3)
    in_flight = 0
    peak_in_flight = 0

    async def slow_fetch_item(client, eid):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {"id": int(eid), "title": f"Story {eid}"}

    try:
        with patch.object(source, "_fetch_item", side_effect=slow_fetch_item):
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = list(range(1, 11))
                mock_resp.raise_for_status.return_value = None
                mock_get.return_value = mock_resp

                items, has_more = await source.list_page(feed_variant="best", page=1, page_size=10)
                assert len(items) == 10
                assert has_more is False
                assert peak_in_flight <= 3
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_hn_transient_retry_on_timeout_and_5xx() -> None:
    source = HackerNewsFeedSource(max_retries=2, retry_backoff_s=0.001)
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {"id": 42, "title": "Recovered"}
    ok_resp.raise_for_status.return_value = None

    err_503 = MagicMock()
    err_503.status_code = 503
    err_503.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=httpx.Request("GET", "https://example.com"), response=httpx.Response(503)
    )

    try:
        with patch(
            "httpx.AsyncClient.get",
            new=AsyncMock(side_effect=[httpx.ReadTimeout("timeout"), err_503, ok_resp]),
        ) as mock_get:
            item = await source.get_item(external_id="42")
            assert item == {"id": 42, "title": "Recovered"}
            assert mock_get.await_count == 3
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_hn_does_not_retry_404() -> None:
    source = HackerNewsFeedSource(max_retries=2, retry_backoff_s=0.001)
    req = httpx.Request("GET", "https://hacker-news.firebaseio.com/v0/item/999.json")
    resp_404 = httpx.Response(404, request=req)

    try:
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=resp_404)) as mock_get:
            with pytest.raises(httpx.HTTPStatusError):
                await source.get_item(external_id="999")
            assert mock_get.await_count == 1
    finally:
        await source.close()
