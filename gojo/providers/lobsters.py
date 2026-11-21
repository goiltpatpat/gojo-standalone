"""
Lobste.rs feed source implementation using the public JSON API.
Endpoints:
- Hottest stories: https://lobste.rs/hottest.json?page={page}
- Newest stories:  https://lobste.rs/newest.json?page={page}
- Single story:    https://lobste.rs/s/{short_id}.json
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from ..types import DigestFeedItem
from .base import FeedSource

_LOBSTERS_BASE = "https://lobste.rs"
_LOBSTERS_PAGE_SIZE = 25

SUPPORTED_LOBSTERS_VARIANTS: tuple[str, ...] = ("hot", "hottest", "lobsters_hot", "new", "newest", "lobsters_new")

_FEED_ENDPOINTS: dict[str, str] = {
    "hot": "hottest.json",
    "hottest": "hottest.json",
    "lobsters_hot": "hottest.json",
    "new": "newest.json",
    "newest": "newest.json",
    "lobsters_new": "newest.json",
}


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


class LobstersFeedSource(FeedSource):
    """Fetches Lobste.rs stories with connection pooling and transient retry/backoff."""

    provider_id = "lobsters"

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.25,
    ) -> None:
        self._timeout = timeout_s
        self._max_retries = max(0, max_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
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
        endpoint = _FEED_ENDPOINTS.get(feed_variant.strip().lower(), "hottest.json")
        return f"{_LOBSTERS_BASE}/{endpoint}"

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
        """Fetch a paginated page of Lobste.rs stories."""
        page = max(1, page)
        page_size = max(1, page_size)
        url = f"{self._list_url(feed_variant)}?page={page}"
        client = self._get_client()

        stories = await self._get_json(client, url)
        if not isinstance(stories, list):
            return [], False

        dict_stories = [s for s in stories if isinstance(s, dict)]
        sliced = dict_stories[:page_size]
        items = [self._to_feed_item(story) for story in sliced]
        has_more = len(dict_stories) > page_size or len(dict_stories) >= _LOBSTERS_PAGE_SIZE
        return items, has_more

    async def get_item(self, *, external_id: str) -> dict[str, Any] | None:
        """Fetch single story details by Lobste.rs short_id."""
        client = self._get_client()
        url = f"{_LOBSTERS_BASE}/s/{external_id}.json"
        data = await self._get_json(client, url)
        return data if isinstance(data, dict) else None

    def _to_feed_item(self, story: dict[str, Any]) -> DigestFeedItem:
        submitter = story.get("submitter_user")
        if isinstance(submitter, dict):
            author = submitter.get("username")
        elif isinstance(submitter, str):
            author = submitter
        else:
            author = None

        created_at = _parse_iso_datetime(story.get("created_at"))
        unix_time = int(created_at.timestamp()) if created_at is not None else None
        comments_url = story.get("comments_url")
        url = story.get("url") or comments_url
        description = story.get("description")
        text = description if isinstance(description, str) and description.strip() else None

        return DigestFeedItem(
            feed_provider=self.provider_id,
            external_id=str(story.get("short_id", "")),
            title=story.get("title"),
            url=url,
            hn_url=comments_url,
            text=text,
            score=story.get("score"),
            by=author,
            author=author,
            time=unix_time,
            created_at=created_at,
            type="story",
            descendants=story.get("comment_count"),
            raw=story,
        )
