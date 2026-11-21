"""
Algolia Hacker News Search API provider.
Provides high-performance, single-request access to full nested comment trees,
subtree-scored discussion transcripts with hierarchical thread-path IDs,
and full Hacker News archive search.
"""

from __future__ import annotations

import html
import re
from typing import Any

import httpx

from ..types import DigestFeedItem

ALGOLIA_ITEM_URL = "https://hn.algolia.com/api/v1/items/{item_id}"
ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ALGOLIA_SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_TIMEOUT_S = 10.0


def _clean_comment_html(raw_html: str | None) -> str:
    """Convert comment HTML markup into clean, scannable plaintext."""
    if not raw_html:
        return ""
    text = raw_html
    # Replace paragraph and line-break tags with newlines
    text = re.sub(r"<p>", "\n\n", text, flags=re.I)
    text = re.sub(r"</p>", "", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    # Convert <a href="...">text</a> to text (url)
    text = re.sub(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', r"\2 (\1)", text, flags=re.I | re.S)
    # Strip remaining HTML tags before unescaping entities so escaped code stays intact
    text = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities (including numeric entities like &#x27;)
    text = html.unescape(text)
    # Normalize excessive spaces and newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
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
            collected.append(
                {
                    "id": cid,
                    "author": author,
                    "depth": depth,
                    "text": text,
                    "children_count": len(child.get("children") or []),
                }
            )

        nested = child.get("children")
        if nested and isinstance(nested, list):
            flatten_comments(nested, depth=depth + 1, max_depth=max_depth, collected=collected)

    return collected


def _prune_comment_subtree(
    node: dict[str, Any],
    *,
    depth: int = 0,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Prune deleted/trivial comments while preserving causal parent->child structure."""
    if depth > max_depth:
        return []

    raw_children = node.get("children") or []
    pruned_children: list[dict[str, Any]] = []
    if isinstance(raw_children, list):
        for child in raw_children:
            if isinstance(child, dict):
                pruned_children.extend(
                    _prune_comment_subtree(child, depth=depth + 1, max_depth=max_depth)
                )

    text = _clean_comment_html(node.get("text"))
    cid = node.get("id")
    if text and len(text) >= 15 and cid:
        raw_points = node.get("points")
        try:
            points = int(raw_points) if raw_points is not None else 0
        except (TypeError, ValueError):
            points = 0
        return [
            {
                "id": cid,
                "author": node.get("author") or "anonymous",
                "depth": depth,
                "text": text,
                "points": points,
                "children": pruned_children,
            }
        ]

    # If a parent comment was deleted/empty, promote valid descendants
    return pruned_children


def _walk_pruned_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node]
    for child in node.get("children") or []:
        nodes.extend(_walk_pruned_nodes(child))
    return nodes


def score_comment_subtree(
    node: dict[str, Any],
    *,
    max_depth: int = 3,
) -> tuple[int, int, int, int]:
    """
    Score a top-level comment's subtree by:
      (total_descendants, substantive_char_weight, depth_diversity, points)
    Accepts either a raw Algolia comment dict or an already-pruned subtree node.
    """
    if "points" in node and isinstance(node.get("text"), str) and "<" not in node["text"] and "depth" in node:
        pruned_roots = [node]
    else:
        pruned_roots = _prune_comment_subtree(node, depth=0, max_depth=max_depth)

    if not pruned_roots:
        return (0, 0, 0, 0)

    all_nodes: list[dict[str, Any]] = []
    for root in pruned_roots:
        all_nodes.extend(_walk_pruned_nodes(root))

    total_descendants = max(len(all_nodes) - 1, 0)
    substantive_char_weight = sum(min(len(n["text"]), 500) for n in all_nodes)
    depth_diversity = len({n["depth"] for n in all_nodes})
    points = sum(int(n.get("points") or 0) for n in all_nodes)
    return (total_descendants, substantive_char_weight, depth_diversity, points)


def _serialize_subtree_entries(
    node: dict[str, Any],
    *,
    path: str,
    rel_depth: int,
    out: list[tuple[str, str, int, str]],
) -> None:
    """Emit (path, author, rel_depth, text) tuples in causal parent->child order."""
    out.append((path, node["author"], rel_depth, node["text"]))
    for idx, child in enumerate(node.get("children") or [], start=1):
        _serialize_subtree_entries(
            child,
            path=f"{path}.{idx}",
            rel_depth=rel_depth + 1,
            out=out,
        )


def _hit_to_feed_item(hit: dict[str, Any]) -> DigestFeedItem | None:
    ext_id = hit.get("objectID") or hit.get("story_id")
    if not ext_id:
        return None
    title = hit.get("title") or hit.get("story_title")
    url = hit.get("url") or hit.get("story_url")
    text = hit.get("story_text")
    points = hit.get("points")
    num_comments = hit.get("num_comments")
    created_at_i = hit.get("created_at_i")
    return DigestFeedItem(
        feed_provider="hn",
        external_id=str(ext_id),
        title=title,
        url=url,
        text=text,
        score=int(points) if points is not None else None,
        by=hit.get("author"),
        time=int(created_at_i) if created_at_i is not None else None,
        type="story",
        descendants=int(num_comments) if num_comments is not None else None,
    )


class HackerNewsAlgoliaProvider:
    """Algolia HN API client for 1-call full discussion tree harvesting and archive search."""

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

    async def search_stories(
        self,
        query: str,
        limit: int = 20,
        sort_by: str = "relevance",
    ) -> list[DigestFeedItem]:
        """
        Query the Algolia HN archive for stories matching `query`.
        Uses `/api/v1/search` for relevance (default) or `/api/v1/search_by_date` when `sort_by == "date"`.
        """
        q = (query or "").strip()
        if not q:
            return []

        endpoint = ALGOLIA_SEARCH_BY_DATE_URL if (sort_by or "").strip().lower() == "date" else ALGOLIA_SEARCH_URL
        hits_per_page = max(1, min(int(limit), 100))
        client = await self._get_client()
        try:
            resp = await client.get(
                endpoint,
                params={
                    "query": q,
                    "tags": "story",
                    "hitsPerPage": hits_per_page,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return []

        hits = payload.get("hits") or []
        items: list[DigestFeedItem] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            item = _hit_to_feed_item(hit)
            if item is not None:
                items.append(item)
        return items[:hits_per_page]

    async def fetch_discussion_context(
        self,
        external_id: str,
        *,
        max_comments: int = 15,
        max_chars: int = 4000,
        max_depth: int = 3,
    ) -> str:
        """
        Fetch and format top discussion subtrees into a structured transcript
        with hierarchical thread-path IDs (`[1] @author`, `[1.1] @reply`)
        suitable for feeding directly into LLM prompts.
        """
        data = await self.get_item(external_id)
        if not data:
            return ""

        raw_children = data.get("children") or []
        pruned_roots: list[dict[str, Any]] = []
        for child in raw_children:
            if isinstance(child, dict):
                pruned_roots.extend(_prune_comment_subtree(child, depth=0, max_depth=max_depth))

        if not pruned_roots:
            return ""

        total_valid = sum(len(_walk_pruned_nodes(r)) for r in pruned_roots)

        # Rank top-level subtrees by (total_descendants, substantive_char_weight, depth_diversity, points)
        ranked_roots = sorted(
            enumerate(pruned_roots),
            key=lambda pair: (score_comment_subtree(pair[1], max_depth=max_depth), -pair[0]),
            reverse=True,
        )

        entries: list[str] = []
        total_len = 0
        selected_count = 0

        for root_num, (_, root_node) in enumerate(ranked_roots, start=1):
            if selected_count >= max_comments:
                break

            subtree_flat: list[tuple[str, str, int, str]] = []
            _serialize_subtree_entries(
                root_node,
                path=str(root_num),
                rel_depth=0,
                out=subtree_flat,
            )

            budget_exhausted = False
            for path, author, rel_depth, text in subtree_flat:
                if selected_count >= max_comments:
                    budget_exhausted = True
                    break

                indent = "  " * rel_depth
                text_snippet = text if len(text) <= 600 else text[:600] + "..."
                entry = f"{indent}[{path}] @{author}: {text_snippet}"

                if total_len + len(entry) > max_chars:
                    budget_exhausted = True
                    break

                entries.append(entry)
                total_len += len(entry)
                selected_count += 1

            if budget_exhausted:
                break

        if not entries:
            return ""

        header = (
            f"--- Hacker News Community Discussion "
            f"({total_valid} total comments, showing top {selected_count}) ---"
        )
        return "\n\n".join([header, *entries])
