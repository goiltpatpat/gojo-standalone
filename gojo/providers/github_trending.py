"""
GitHub Trending feed source implementation using the official GitHub Search Repositories API.
Endpoints:
- Search repositories: https://api.github.com/search/repositories
- Single repository:   https://api.github.com/repos/{owner}/{repo}
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..types import DigestFeedItem
from .base import FeedSource

_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_SEARCH_CAP = 1000


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


class GitHubTrendingFeedSource(FeedSource):
    """Fetches recently created, high-star GitHub repositories with connection pooling and retry."""

    provider_id = "github"

    def __init__(
        self,
        *,
        github_token: str | None = None,
        timeout_s: float = 20.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.25,
        lookback_days: int = 7,
    ) -> None:
        self._github_token = github_token
        self._timeout = timeout_s
        self._max_retries = max(0, max_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        self._lookback_days = max(1, lookback_days)
        self._client: httpx.AsyncClient | None = None

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "gojo-hn",
        }
        raw_token = self._github_token if self._github_token is not None else os.environ.get("GITHUB_TOKEN")
        token = (raw_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and reuse a pooled httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._build_headers(),
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._client

    async def close(self) -> None:
        """Close the pooled HTTP client if open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """GET JSON with lightweight exponential-backoff retry on timeouts, 429, and 5xx."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if params is not None:
                    r = await client.get(url, params=params)
                else:
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
        self, *, feed_variant: str = "github_trending", page: int = 1, page_size: int = 30
    ) -> tuple[list[DigestFeedItem], bool]:
        """Fetch trending repositories created in the last 7 days sorted by stars desc."""
        _ = feed_variant
        page = max(1, page)
        per_page = min(max(1, page_size), 100)
        since_date = (datetime.now(timezone.utc) - timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")

        client = self._get_client()
        url = f"{_GITHUB_API_BASE}/search/repositories"
        params = {
            "q": f"created:>={since_date}",
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }
        data = await self._get_json(client, url, params=params)
        if not isinstance(data, dict):
            return [], False

        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            return [], False

        items = [self._to_feed_item(repo) for repo in raw_items if isinstance(repo, dict)]
        total_count = data.get("total_count")
        if isinstance(total_count, int) and total_count > 0:
            has_more = (page * per_page) < min(total_count, _GITHUB_SEARCH_CAP)
        else:
            has_more = len(items) >= per_page
        return items, has_more

    async def get_item(self, *, external_id: str) -> dict[str, Any] | None:
        """Fetch single repository details by full_name (owner/repo)."""
        client = self._get_client()
        repo_path = external_id.strip().lstrip("/")
        url = f"{_GITHUB_API_BASE}/repos/{repo_path}"
        data = await self._get_json(client, url)
        return data if isinstance(data, dict) else None

    def _to_feed_item(self, repo: dict[str, Any]) -> DigestFeedItem:
        full_name = str(repo.get("full_name", ""))
        desc = (repo.get("description") or "").strip()
        title = f"{full_name}: {desc}" if desc else full_name
        html_url = repo.get("html_url")
        owner = repo.get("owner")
        author = owner.get("login") if isinstance(owner, dict) else None
        created_at = _parse_iso_datetime(repo.get("created_at"))
        unix_time = int(created_at.timestamp()) if created_at is not None else None

        return DigestFeedItem(
            feed_provider=self.provider_id,
            external_id=full_name,
            title=title,
            url=html_url,
            hn_url=html_url,
            text=desc or None,
            score=repo.get("stargazers_count"),
            by=author,
            author=author,
            time=unix_time,
            created_at=created_at,
            type="repository",
            descendants=repo.get("forks_count"),
            raw=repo,
        )
