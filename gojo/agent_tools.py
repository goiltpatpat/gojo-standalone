"""
Gojo Hacker News Agent Tools:
Native function schemas, executors, and string formatters for AI Agents.
Compatible with OpenAI Function Calling, Anthropic Tools, Gemini, LangChain, and Vercel AI SDK.
"""

from __future__ import annotations

from typing import Any

from .service import GojoHNService
from .types import DigestFeedItem


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
) -> dict[str, Any]:
    """
    Search or browse Hacker News top/new stories.
    """
    feed = await svc.list_feed(feed_id=feed_id, page=page)
    filtered = filter_items_by_query(feed.items, query)
    return {
        "feed_id": feed.feed_id,
        "page": feed.page,
        "total_returned": len(filtered),
        "has_more": feed.has_more,
        "query_applied": bool(query and query.strip()),
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
    locale: str = "en",
) -> dict[str, Any]:
    """
    Generate an authoritative executive technology briefing synthesizing top Hacker News stories.
    """
    briefing_md = await svc.create_briefing(feed_id=feed_id, limit=limit, locale=locale)
    return {
        "feed_id": feed_id,
        "limit": limit,
        "locale": locale,
        "briefing_markdown": briefing_md,
    }


async def tool_get_hn_story(svc: GojoHNService, *, external_id: str) -> dict[str, Any] | None:
    """Fetch raw Hacker News story details."""
    return await svc.source.get_item(external_id=str(external_id))


async def tool_get_artifact(svc: GojoHNService, *, artifact_id: str) -> dict[str, Any] | None:
    """Retrieve an existing generated summary artifact by its UUID."""
    art = await svc.get_artifact(artifact_id=artifact_id)
    return art.model_dump(mode="json") if art else None


def format_search_for_agent(out: dict[str, Any], max_items: int = 20) -> str:
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
        lines.append(f"- [ID: {eid}] {title} (Score: {score}, Comments: {comments})")
        if url:
            lines.append(f"  URL: {url}")

    if len(items) > max_items:
        lines.append(f"... and {len(items) - max_items} more.")
    return "\n".join(lines)


def format_summary_for_agent(out: dict[str, Any]) -> str:
    """Format summary response for LLM agent turns."""
    art = out.get("artifact") or {}
    comp = out.get("completion") or {}
    return (
        f"Artifact ID: {art.get('id')}\n"
        f"Story ID: {art.get('external_id')}\n"
        f"Title: {art.get('title')}\n"
        f"Status: {comp.get('status')} ({comp.get('cache_layer')})\n"
        f"Type: {art.get('summary_type')} | Locale: {art.get('locale')}\n\n"
        f"{art.get('summary_markdown')}"
    )


HN_AGENT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "hn_search",
            "description": (
                "Search and browse Hacker News headlines. Feed IDs: 'hn_top' (front page) "
                "or 'hn_new' (newest). Optional query filters titles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feed_id": {
                        "type": "string",
                        "enum": ["hn_top", "hn_new"],
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
                        "description": "Optional title search filter.",
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
            "description": "Generate a synthesized executive technology briefing from top Hacker News stories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "feed_id": {
                        "type": "string",
                        "enum": ["hn_top", "hn_new"],
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
]
