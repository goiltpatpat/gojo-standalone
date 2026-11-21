"""
Gojo Hacker News Agent Tools:
Native function schemas, executors, and string formatters for AI Agents.
Compatible with OpenAI Function Calling, Anthropic Tools, Gemini, LangChain, and Vercel AI SDK.
"""

from __future__ import annotations

from typing import Any

from .providers.hn import SUPPORTED_FEED_IDS
from .service import GojoHNService
from .signals import rank_items_by_focus
from .types import DigestFeedItem


def _apply_max_chars(text: str, max_chars: int | None) -> str:
    """Enforce an optional hard character budget for agent context windows."""
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = "\n... [truncated]"
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def filter_items_by_query(items: list[DigestFeedItem], query: str | None) -> list[DigestFeedItem]:
    """Filter feed items by case-insensitive title search."""
    if not query or not query.strip():
        return items
    q = query.strip().lower()
    return [it for it in items if it.title and q in it.title.lower()]


async def tool_search_hn(
    svc: GojoHNService,
    *,
    feed_id: str = "hn_top",
    page: int = 1,
    query: str | None = None,
    focus: str | None = None,
) -> dict[str, Any]:
    """
    Search or browse Hacker News feeds ('hn_top', 'hn_new', 'hn_best', 'hn_ask', 'hn_show').
    Optionally filters by title substring and/or re-ranks by BM25 focus query.
    """
    feed = await svc.list_feed(feed_id=feed_id, page=page)
    filtered = filter_items_by_query(feed.items, query)
    focus_applied = bool(focus and focus.strip())
    if focus_applied and filtered:
        ranked = rank_items_by_focus(filtered, focus.strip())  # type: ignore[union-attr]
        filtered = [it for it, _score in ranked]

    return {
        "feed_id": feed.feed_id,
        "page": feed.page,
        "total_returned": len(filtered),
        "has_more": feed.has_more,
        "query_applied": bool(query and query.strip()),
        "focus_applied": focus_applied,
        "items": [it.model_dump(mode="json") for it in filtered],
    }


async def tool_summarize_hn(
    svc: GojoHNService,
    *,
    external_id: str,
    summary_type: str = "quick",
    locale: str = "en",
    include_comments: bool = True,
) -> dict[str, Any]:
    """
    Summarize a Hacker News story by ID using AI with anti-slop guidelines
    and optional community discussion synthesis.
    """
    res = await svc.summarize(
        external_id=str(external_id),
        summary_type=summary_type,
        locale=locale,
        include_comments=include_comments,
    )
    return {
        "artifact": res.artifact.model_dump(mode="json"),
        "completion": res.completion.model_dump(mode="json"),
    }


async def tool_briefing_hn(
    svc: GojoHNService,
    *,
    feed_id: str = "hn_top",
    limit: int = 10,
    locale: str = "th",
    focus: str | None = None,
    since_last: bool = False,
) -> dict[str, Any]:
    """
    Generate an authoritative executive technology briefing synthesizing top Hacker News stories.
    """
    briefing_md = await svc.create_briefing(
        feed_id=feed_id,
        limit=limit,
        locale=locale,
        focus=focus,
        since_last=since_last,
    )
    return {
        "feed_id": feed_id,
        "limit": limit,
        "locale": locale,
        "focus": focus,
        "since_last": since_last,
        "briefing_markdown": briefing_md,
    }


async def tool_get_hn_story(svc: GojoHNService, *, external_id: str) -> dict[str, Any] | None:
    """Fetch raw Hacker News story details."""
    return await svc.source.get_item(external_id=str(external_id))


async def tool_get_artifact(svc: GojoHNService, *, artifact_id: str) -> dict[str, Any] | None:
    """Retrieve an existing generated summary artifact by its UUID."""
    art = await svc.get_artifact(artifact_id=artifact_id)
    return art.model_dump(mode="json") if art else None


async def tool_search_archive(
    svc: GojoHNService,
    *,
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Search the local SQLite FTS5 / memory cache of previously summarized stories."""
    artifacts = await svc.search_archive(query=query, limit=limit)
    return {
        "query": query,
        "total_returned": len(artifacts),
        "artifacts": [art.model_dump(mode="json") for art in artifacts],
    }


def format_search_for_agent(
    out: dict[str, Any],
    max_items: int = 20,
    *,
    max_chars: int | None = None,
) -> str:
    """Format search results as clean Markdown for LLM prompt context."""
    items = out.get("items") or []
    lines = [
        f"Feed: {out.get('feed_id')} | Page: {out.get('page')} | Items: {len(items)} "
        f"| Has More: {out.get('has_more')}",
        "",
    ]
    for it in items[:max_items]:
        eid = it.get("external_id")
        title = it.get("title") or "Untitled"
        score = it.get("score") or 0
        comments = it.get("descendants") or 0
        url = it.get("url") or ""
        badges = it.get("signal_badges") or []
        badge_str = f" [{', '.join(badges)}]" if badges else ""
        lines.append(
            f"- [ID: {eid}] {title} (Score: {score}, Comments: {comments}){badge_str}"
        )
        if url:
            lines.append(f"  URL: {url}")

    if len(items) > max_items:
        lines.append(f"... and {len(items) - max_items} more.")
    return _apply_max_chars("\n".join(lines), max_chars)


def format_summary_for_agent(
    out: dict[str, Any],
    *,
    max_chars: int | None = None,
) -> str:
    """Format summary response for LLM agent turns."""
    art = out.get("artifact") or {}
    comp = out.get("completion") or {}
    formatted = (
        f"Artifact ID: {art.get('id')}\n"
        f"Story ID: {art.get('external_id')}\n"
        f"Title: {art.get('title')}\n"
        f"Status: {comp.get('status')} ({comp.get('cache_layer')})\n"
        f"Type: {art.get('summary_type')} | Locale: {art.get('locale')}\n\n"
        f"{art.get('summary_markdown')}"
    )
    return _apply_max_chars(formatted, max_chars)


def format_archive_search_for_agent(
    out: dict[str, Any],
    *,
    max_chars: int | None = None,
) -> str:
    """Format local archive FTS5 search results for LLM agent turns."""
    artifacts = out.get("artifacts") or []
    lines = [
        f"Archive Query: {out.get('query')} | Matches: {len(artifacts)}",
        "",
    ]
    for art in artifacts:
        lines.append(
            f"- [Artifact: {art.get('id')}] [Story: {art.get('external_id')}] "
            f"{art.get('title') or 'Untitled'} ({art.get('summary_type')}, {art.get('locale')})"
        )
        takeaway = (art.get("summary_extra") or {}).get("key_takeaway")
        if takeaway:
            lines.append(f"  Takeaway: {takeaway}")
    if not artifacts:
        lines.append("No matching archived summaries found.")
    return _apply_max_chars("\n".join(lines), max_chars)


HN_AGENT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "hn_search",
            "description": (
                "Search and browse Hacker News headlines. Feed IDs: 'hn_top' (front page), "
                "'hn_new' (newest), 'hn_best' (best), 'hn_ask' (Ask HN), or 'hn_show' (Show HN). "
                "Optional query filters titles; optional focus re-ranks by topic relevance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feed_id": {
                        "type": "string",
                        "enum": list(SUPPORTED_FEED_IDS),
                        "default": "hn_top",
                        "description": "Feed category to retrieve.",
                    },
                    "page": {
                        "type": "integer",
                        "default": 1,
                        "description": "Page number (1-based).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional title substring filter.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional BM25 focus topic to re-rank stories by relevance.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hn_summarize",
            "description": (
                "Fetch full article and generate an intelligent, structured technical summary "
                "for a Hacker News story ID with anti-slop guidelines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "external_id": {
                        "type": "string",
                        "description": "Hacker News story ID.",
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["quick", "technical", "bullet", "thai_concise"],
                        "default": "quick",
                        "description": "Format and depth of the summary.",
                    },
                    "locale": {
                        "type": "string",
                        "default": "en",
                        "description": "Target language ISO code (e.g. 'en', 'th').",
                    },
                    "include_comments": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to fetch and synthesize community comment discussions.",
                    },
                },
                "required": ["external_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hn_briefing",
            "description": (
                "Generate a synthesized executive technology briefing from Hacker News stories "
                "with signal badges, clustering, optional focus re-ranking, and since-last deltas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feed_id": {
                        "type": "string",
                        "enum": list(SUPPORTED_FEED_IDS),
                        "default": "hn_top",
                        "description": "Feed to synthesize.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of stories to include.",
                    },
                    "locale": {
                        "type": "string",
                        "default": "th",
                        "description": "Target language (e.g. 'th', 'en').",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional BM25 focus topic to prioritize relevant stories.",
                    },
                    "since_last": {
                        "type": "boolean",
                        "default": False,
                        "description": "Compare against prior sighting snapshots to highlight NEW and surging stories.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hn_get_story",
            "description": "Fetch raw metadata (score, author, URL, comments count) for a Hacker News story.",
            "parameters": {
                "type": "object",
                "properties": {
                    "external_id": {
                        "type": "string",
                        "description": "Hacker News story ID.",
                    },
                },
                "required": ["external_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hn_search_archive",
            "description": (
                "Search previously summarized Hacker News digest artifacts in the local SQLite FTS5 cache."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Full-text search query over archived summaries and titles.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of matching artifacts to return.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hn_get_artifact",
            "description": "Retrieve a previously generated digest summary artifact by its UUID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "UUID of the stored digest artifact.",
                    },
                },
                "required": ["artifact_id"],
            },
        },
    },
]
