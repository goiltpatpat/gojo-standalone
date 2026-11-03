"""
Hacker News feed source implementation using the official Firebase API.
Endpoints:
- Top stories: https://hacker-news.firebaseio.com/v0/topstories.json
- New stories: https://hacker-news.firebaseio.com/v0/newstories.json
- Item: https://hacker-news.firebaseio.com/v0/item/{id}.json
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..types import DigestFeedItem
from .base import FeedSource

_HN_BASE = "https://hacker-news.firebaseio.com/v0"


class HackerNewsFeedSource(FeedSource):
    """Fetches Hacker News feed items and stories concurrently."""

    provider_id = "hn"

    def __init__(self, *, timeout_s: float = 20.0) -> None:
        self._timeout = timeout_s

    def _list_url(self, feed_variant: str) -> str:
        if feed_variant == "new":
            return f"{_HN_BASE}/newstories.json"
        return f"{_HN_BASE}/topstories.json"

    async def list_page(
        self, *, feed_variant: str, page: int, page_size: int
    ) -> tuple[list[DigestFeedItem], bool]:
        """Fetch a paginated slice of Hacker News stories concurrently."""
        page = max(1, page)
        list_url = self._list_url(feed_variant)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(list_url)
            r.raise_for_status()
            ids = r.json()
            if not isinstance(ids, list):
                return [], False

            start = (page - 1) * page_size
            end = start + page_size
            slice_ids = ids[start:end]
            has_more = end < len(ids)

            fetched = await asyncio.gather(
                *[self._fetch_item(client, str(eid)) for eid in slice_ids],
                return_exceptions=True,
            )

        items: list[DigestFeedItem] = []
        for it in fetched:
            if isinstance(it, dict):
                items.append(self._to_feed_item(it))

        return items, has_more

    async def _fetch_item(self, client: httpx.AsyncClient, external_id: str) -> dict[str, Any] | None:
        url = f"{_HN_BASE}/item/{external_id}.json"
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else None

    async def get_item(self, *, external_id: str) -> dict[str, Any] | None:
        """Fetch single story details by HN ID."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._fetch_item(client, external_id)

    def _to_feed_item(self, raw: dict[str, Any]) -> DigestFeedItem:
        return DigestFeedItem(
            feed_provider=self.provider_id,
            external_id=str(raw.get("id", "")),
            title=raw.get("title"),
            url=raw.get("url"),
            text=raw.get("text"),
            score=raw.get("score"),
            by=raw.get("by"),
            time=raw.get("time"),
            type=raw.get("type"),
            descendants=raw.get("descendants"),
        )
