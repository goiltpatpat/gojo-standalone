"""
Hacker News feed source implementation using the official Firebase API.
Endpoints:
- Top stories:  https://hacker-news.firebaseio.com/v0/topstories.json
- New stories:  https://hacker-news.firebaseio.com/v0/newstories.json
- Best stories: https://hacker-news.firebaseio.com/v0/beststories.json
- Ask HN:       https://hacker-news.firebaseio.com/v0/askstories.json
- Show HN:      https://hacker-news.firebaseio.com/v0/showstories.json
- Item:         https://hacker-news.firebaseio.com/v0/item/{id}.json
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..types import DigestFeedItem
from .base import FeedSource

_HN_BASE = "https://hacker-news.firebaseio.com/v0"

SUPPORTED_FEED_VARIANTS: tuple[str, ...] = ("top", "new", "best", "ask", "show")
SUPPORTED_FEED_IDS: tuple[str, ...] = ("hn_top", "hn_new", "hn_best", "hn_ask", "hn_show")

_FEED_ENDPOINTS: dict[str, str] = {
    "top": "topstories.json",
    "hn_top": "topstories.json",
    "new": "newstories.json",
    "hn_new": "newstories.json",
    "best": "beststories.json",
    "hn_best": "beststories.json",
    "ask": "askstories.json",
    "hn_ask": "askstories.json",
    "show": "showstories.json",
    "hn_show": "showstories.json",
}


class HackerNewsFeedSource(FeedSource):
    """Fetches Hacker News feed items and stories concurrently with connection pooling."""

    provider_id = "hn"

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        max_concurrency: int = 12,
        max_retries: int = 2,
        retry_backoff_s: float = 0.25,
    ) -> None:
        self._timeout = timeout_s
        self._max_concurrency = max(1, max_concurrency)
        self._max_retries = max(0, max_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and reuse a pooled httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._client

    async def close(self) -> None:
        """Close the pooled HTTP client if open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _list_url(self, feed_variant: str) -> str:
        endpoint = _FEED_ENDPOINTS.get(feed_variant.strip().lower(), "topstories.json")
        return f"{_HN_BASE}/{endpoint}"

    async def _get_json(self, client: httpx.AsyncClient, url: str) -> Any:
        """GET JSON with lightweight exponential-backoff retry on timeouts, 429, and 5xx."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                r = await client.get(url)
                status = getattr(r, "status_code", None)
                if isinstance(status, int) and (status == 429 or 500 <= status < 600):
                    if attempt < self._max_retries:
                        await asyncio.sleep(self._retry_backoff_s * (2**attempt))
                        continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                code = exc.response.status_code if exc.response is not None else 0
                if (code == 429 or 500 <= code < 600) and attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff_s * (2**attempt))
                    continue
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff_s * (2**attempt))
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Failed to fetch {url}")

    async def list_page(
        self, *, feed_variant: str, page: int, page_size: int
    ) -> tuple[list[DigestFeedItem], bool]:
        """Fetch a paginated slice of Hacker News stories with bounded concurrency."""
        page = max(1, page)
        list_url = self._list_url(feed_variant)
        client = self._get_client()

        ids = await self._get_json(client, list_url)
        if not isinstance(ids, list):
            return [], False

        start = (page - 1) * page_size
        end = start + page_size
        slice_ids = ids[start:end]
        has_more = end < len(ids)

        async def _bounded_fetch(eid: str) -> dict[str, Any] | None:
            async with self._semaphore:
                return await self._fetch_item(client, eid)

        fetched = await asyncio.gather(
            *[_bounded_fetch(str(eid)) for eid in slice_ids],
            return_exceptions=True,
        )

        items: list[DigestFeedItem] = []
        for it in fetched:
            if isinstance(it, dict):
                items.append(self._to_feed_item(it))

        return items, has_more

    async def _fetch_item(self, client: httpx.AsyncClient, external_id: str) -> dict[str, Any] | None:
        url = f"{_HN_BASE}/item/{external_id}.json"
        data = await self._get_json(client, url)
        return data if isinstance(data, dict) else None

    async def get_item(self, *, external_id: str) -> dict[str, Any] | None:
        """Fetch single story details by HN ID."""
        client = self._get_client()
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
