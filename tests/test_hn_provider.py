import pytest
from unittest.mock import AsyncMock, patch

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

    with patch.object(source, "_fetch_item", side_effect=mock_fetch_item):
        from unittest.mock import MagicMock
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
