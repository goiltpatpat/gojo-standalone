"""
Algolia Hacker News Search API provider.
Provides high-performance, single-request access to full nested comment trees
and submission metadata without N+1 query overhead.
"""

from __future__ import annotations

import html
import re
from typing import Any

import httpx

ALGOLIA_ITEM_URL = "https://hn.algolia.com/api/v1/items/{item_id}"
DEFAULT_TIMEOUT_S = 10.0


def _clean_comment_html(raw_html: str | None) -> str:
    """Convert comment HTML markup into clean, scannable plaintext."""
    if not raw_html:
        return ""
    text = raw_html
    # Replace <p> tags with double newlines
    text = re.sub(r"<p>", "\n\n", text, flags=re.I)
    text = re.sub(r"</p>", "", text, flags=re.I)
    # Convert <a href="...">text</a> to text (url)
    text = re.sub(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', r"\2 (\1)", text, flags=re.I)
    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities
    text = html.unescape(text)
    # Normalize excessive spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flatten_comments(
    children: list[dict[str, Any]],
    depth: int = 0,
    max_depth: int = 3,
    collected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten nested Algolia comment tree into a linear list with depth indicators."""
    if collected is None:
        collected = []
    if depth > max_depth:
        return collected

    for child in children:
        text = _clean_comment_html(child.get("text"))
        author = child.get("author") or "anonymous"
        cid = child.get("id")
        
        # Skip empty, deleted, or dead comments
        if text and len(text) >= 15 and cid:
            collected.append({
                "id": cid,
                "author": author,
                "depth": depth,
                "text": text,
                "children_count": len(child.get("children") or []),
            })

        nested = child.get("children")
        if nested and isinstance(nested, list):
            flatten_comments(nested, depth=depth + 1, max_depth=max_depth, collected=collected)

    return collected


class HackerNewsAlgoliaProvider:
    """Algolia HN API client for 1-call full discussion tree harvesting."""

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_item(self, external_id: str) -> dict[str, Any] | None:
        """Fetch complete item data with full nested comment tree in 1 HTTP call."""
        client = await self._get_client()
        url = ALGOLIA_ITEM_URL.format(item_id=str(external_id).strip())
        try:
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    async def fetch_discussion_context(
        self,
        external_id: str,
        *,
        max_comments: int = 15,
        max_chars: int = 4000,
    ) -> str:
        """
        Fetch and format top discussion comments into a structured transcript
        suitable for feeding directly into LLM prompts.
        """
        data = await self.get_item(external_id)
        if not data:
            return ""

        children = data.get("children") or []
        flattened = flatten_comments(children, max_depth=3)
        if not flattened:
            return ""

        # Prioritize top comments with high engagement / substance
        # Score comments by depth (lower depth = top level) and text length
        def comment_sort_key(c: dict[str, Any]) -> tuple[int, int]:
            # Favor top-level first (depth 0, 1), then length of substantive commentary
            return (-c["depth"], min(len(c["text"]), 500))

        # Take up to max_comments
        selected = flattened[:max_comments]

        lines = [
            f"--- Hacker News Community Discussion ({len(flattened)} total comments, showing top {len(selected)}) ---"
        ]
        total_len = 0

        for c in selected:
            indent = "  " * c["depth"]
            prefix = f"{indent}[@{c['author']}]: "
            text_snippet = c["text"]
            if len(text_snippet) > 600:
                text_snippet = text_snippet[:600] + "..."
            
            entry = f"{prefix}{text_snippet}"
            if total_len + len(entry) > max_chars:
                break
            lines.append(entry)
            total_len += len(entry)

        return "\n\n".join(lines)
