"""
Gojo Hacker News Agent Tools:
Native function schemas, executors, and string formatters for AI Agents.
Compatible with OpenAI Function Calling, Anthropic Tools, Gemini, LangChain, and Vercel AI SDK.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

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


async def tool_search_hn_full(
    svc: GojoHNService,
    *,
    query: str,
    limit: int = 15,
    sort_by: str = "relevance",
) -> dict[str, Any]:
    """Search the full historical Hacker News archive via Algolia with Six Eyes signals."""
    items = await svc.search_hn_archive(query=query, limit=limit, sort_by=sort_by)
    return {
        "query": query,
        "sort_by": sort_by,
        "total_returned": len(items),
        "items": [it.model_dump(mode="json") for it in items],
    }


async def tool_radar_hn(
    svc: GojoHNService,
    *,
    feeds: list[str] | None = None,
    top_n: int = 10,
    focus: str | None = None,
    badges: list[str] | None = None,
) -> dict[str, Any]:
    """Run the zero-LLM-token Six Eyes Signal Radar across HN feeds."""
    return await svc.radar(
        feeds=feeds,
        top_n=top_n,
        focus=focus,
        badges=badges,
    )


async def tool_deep_dive_hn(
    svc: GojoHNService,
    *,
    topic: str,
    max_stories: int = 3,
    summary_type: str = "technical",
    locale: str = "en",
    include_comments: bool = True,
) -> dict[str, Any]:
    """Run a compound multi-story research deep-dive on a topic across local + HN archives."""
    return await svc.deep_dive(
        topic=topic,
        max_stories=max_stories,
        summary_type=summary_type,
        locale=locale,
        include_comments=include_comments,
    )


def format_search_for_agent(
    out: dict[str, Any],
    max_items: int = 20,
    *,
    max_chars: int | None = None,
) -> str:
    """Format search results as clean Markdown for LLM prompt context."""
    items = out.get("items") or []
    if out.get("feed_id") is not None:
        header = (
            f"Feed: {out.get('feed_id')} | Page: {out.get('page')} | Items: {len(items)} "
            f"| Has More: {out.get('has_more')}"
        )
    else:
        header = (
            f"HN Archive Query: {out.get('query')} | Sort: {out.get('sort_by', 'relevance')} "
            f"| Items: {len(items)}"
        )
    lines = [header, ""]
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


def format_radar_for_agent(
    out: dict[str, Any],
    *,
    max_chars: int | None = None,
) -> str:
    """Format Six Eyes Signal Radar output as compact Markdown for LLM agent turns."""
    feeds = ", ".join(out.get("feeds_scanned") or [])
    total = out.get("total_scanned", 0)
    top_items = out.get("top_items") or []
    clusters = out.get("clusters") or []
    focus = out.get("focus_applied")
    badges_filter = out.get("badges_filter") or []

    header_parts = [
        f"Six Eyes Radar | Feeds: {feeds}",
        f"Scanned: {total}",
        f"Top: {len(top_items)}",
        f"Clusters: {len(clusters)}",
    ]
    if focus:
        header_parts.append(f"Focus: {focus}")
    if badges_filter:
        header_parts.append(f"Badges: {', '.join(badges_filter)}")

    lines = [" | ".join(header_parts), ""]
    if not top_items:
        lines.append("No stories matched the radar filters.")
        return _apply_max_chars("\n".join(lines), max_chars)

    for cluster in clusters:
        c_items = cluster.get("items") or []
        if len(c_items) > 1:
            lines.append(
                f"### Cluster {cluster.get('cluster_id')} ({len(c_items)} corroborated stories)"
            )
        for it in c_items:
            eid = it.get("external_id")
            title = it.get("title") or "Untitled"
            score = it.get("score") or 0
            comments = it.get("descendants") or 0
            vel = it.get("velocity")
            badges = it.get("signal_badges") or []
            badge_str = f" [{', '.join(badges)}]" if badges else ""
            vel_str = f", Vel: {vel:.1f}" if isinstance(vel, (int, float)) else ""
            lines.append(
                f"- [ID: {eid}] {title} (Score: {score}, Comments: {comments}{vel_str}){badge_str}"
            )
            if it.get("url"):
                lines.append(f"  URL: {it.get('url')}")
    return _apply_max_chars("\n".join(lines), max_chars)


def format_deep_dive_for_agent(
    out: dict[str, Any],
    *,
    max_chars: int | None = None,
) -> str:
    """Format compound deep-dive research dossier as Markdown for LLM agent turns."""
    topic = out.get("topic") or ""
    analyzed = out.get("stories_analyzed", 0)
    cached_hits = out.get("cached_hits", 0)
    summaries = out.get("summaries") or []
    local_matches = out.get("local_archive_matches") or []
    errors = out.get("errors") or []

    lines = [
        f"# Deep Dive Dossier: {topic}",
        f"Stories Analyzed: {analyzed} | Cached Hits: {cached_hits} | Local Archive Matches: {len(local_matches)}",
        "",
    ]

    for idx, entry in enumerate(summaries, start=1):
        art = entry.get("artifact") or {}
        comp = entry.get("completion") or {}
        lines.append(
            f"## {idx}. [{art.get('external_id')}] {art.get('title') or 'Untitled'} "
            f"({comp.get('status')}/{comp.get('cache_layer')})"
        )
        if art.get("source_url"):
            lines.append(f"URL: {art.get('source_url')}")
        lines.append("")
        lines.append(str(art.get("summary_markdown") or "").strip())
        lines.append("")

    if not summaries:
        lines.append("No stories were summarized for this topic.")
        lines.append("")

    if local_matches:
        lines.append("### Related Local Archive Matches")
        for art in local_matches[:5]:
            lines.append(
                f"- [Artifact: {art.get('id')}] [Story: {art.get('external_id')}] "
                f"{art.get('title') or 'Untitled'}"
            )
        lines.append("")

    if errors:
        lines.append("### Partial Errors")
        for err in errors:
            target = err.get("external_id") or err.get("stage") or "unknown"
            lines.append(f"- {target}: {err.get('error')}")

    return _apply_max_chars("\n".join(lines).strip(), max_chars)


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
            "name": "hn_search_full",
            "description": (
                "Search the full historical Hacker News archive via Algolia with Six Eyes "
                "signal enrichment (velocity, debate_ratio, signal_badges)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Full-text search query across historical Hacker News stories.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 15,
                        "description": "Maximum number of stories to return.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "date"],
                        "default": "relevance",
                        "description": "Sort results by relevance or newest date first.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hn_radar",
            "description": (
                "Zero-LLM-token Six Eyes Signal Radar: scans multiple HN feeds concurrently, "
                "deduplicates stories, computes velocity/debate signals, filters by badges, "
                "optionally re-ranks by BM25 focus, and clusters related stories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feeds": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SUPPORTED_FEED_IDS)},
                        "description": "Feeds to scan concurrently (default: ['hn_top', 'hn_best']).",
                    },
                    "top_n": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of top-ranked stories to return.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional BM25 focus topic to re-rank stories.",
                    },
                    "badges": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["breakout", "rising", "high-debate", "slow-burn"],
                        },
                        "description": "Optional signal badge filter (e.g. ['breakout', 'high-debate']).",
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
            "name": "hn_deep_dive",
            "description": (
                "Compound topic research pipeline: searches both local FTS5 archive and full HN "
                "Algolia archive, selects top distinct stories, summarizes them concurrently "
                "(reusing deterministic SHA-256 cache), and returns a research dossier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Technical topic or keyword to research across Hacker News.",
                    },
                    "max_stories": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum number of distinct stories to summarize.",
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["quick", "technical", "bullet", "thai_concise"],
                        "default": "technical",
                        "description": "Summary style for each analyzed story.",
                    },
                    "locale": {
                        "type": "string",
                        "default": "en",
                        "description": "Target language ISO code (e.g. 'en', 'th').",
                    },
                    "include_comments": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to synthesize HN community comments for each story.",
                    },
                },
                "required": ["topic"],
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


class GojoAgentToolkit:
    """
    Unified agent-facing toolkit wrapping GojoHNService with multi-framework tool schemas
    (OpenAI, Anthropic, Gemini) and safe, crash-proof tool dispatch with optional max_chars budgeting.
    """

    def __init__(self, service: GojoHNService | None = None) -> None:
        self._owns_service = service is None
        self._service: GojoHNService | None = service

    @property
    def service(self) -> GojoHNService:
        if self._service is None:
            self._service = GojoHNService()
        return self._service

    async def __aenter__(self) -> GojoAgentToolkit:
        _ = self.service
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_service and self._service is not None:
            await self._service.close()
            self._service = None

    def schemas(
        self,
        provider: Literal["openai", "anthropic", "gemini"] = "openai",
    ) -> list[dict[str, Any]]:
        """Export all 9 Gojo tool definitions formatted for OpenAI, Anthropic, or Gemini."""
        if provider == "openai":
            return copy.deepcopy(HN_AGENT_TOOL_DEFINITIONS)
        if provider == "anthropic":
            return [
                {
                    "name": fn["name"],
                    "description": fn["description"],
                    "input_schema": copy.deepcopy(fn["parameters"]),
                }
                for fn in (t["function"] for t in HN_AGENT_TOOL_DEFINITIONS)
            ]
        if provider == "gemini":
            return [
                {
                    "name": fn["name"],
                    "description": fn["description"],
                    "parameters": copy.deepcopy(fn["parameters"]),
                }
                for fn in (t["function"] for t in HN_AGENT_TOOL_DEFINITIONS)
            ]
        raise ValueError(
            f"Unsupported schema provider '{provider}'. Choose 'openai', 'anthropic', or 'gemini'."
        )

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        max_chars: int | None = None,
        as_text: bool = True,
    ) -> dict[str, Any] | str:
        """
        Safely parse tool arguments, execute the requested Gojo tool, and format output.
        Never raises on unknown tools, invalid JSON, or runtime errors — returns a structured
        error envelope (as_text=False) or concise error string (as_text=True).
        """
        def _err(msg: str) -> dict[str, Any] | str:
            if as_text:
                return _apply_max_chars(f"Error [{name}]: {msg}", max_chars)
            return {"ok": False, "tool": name, "error": msg}

        if arguments is None:
            kwargs: dict[str, Any] = {}
        elif isinstance(arguments, str):
            stripped = arguments.strip()
            if not stripped:
                kwargs = {}
            else:
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    return _err(f"Invalid JSON arguments: {exc}")
                if not isinstance(parsed, dict):
                    return _err("JSON arguments must decode to an object.")
                kwargs = parsed
        elif isinstance(arguments, dict):
            kwargs = dict(arguments)
        else:
            return _err("Arguments must be a dict, JSON object string, or None.")

        svc = self.service
        try:
            if name == "hn_search":
                res = await tool_search_hn(svc, **kwargs)
                return format_search_for_agent(res, max_chars=max_chars) if as_text else res
            if name == "hn_search_full":
                res = await tool_search_hn_full(svc, **kwargs)
                return format_search_for_agent(res, max_chars=max_chars) if as_text else res
            if name == "hn_radar":
                res = await tool_radar_hn(svc, **kwargs)
                return format_radar_for_agent(res, max_chars=max_chars) if as_text else res
            if name == "hn_summarize":
                res = await tool_summarize_hn(svc, **kwargs)
                return format_summary_for_agent(res, max_chars=max_chars) if as_text else res
            if name == "hn_deep_dive":
                res = await tool_deep_dive_hn(svc, **kwargs)
                return format_deep_dive_for_agent(res, max_chars=max_chars) if as_text else res
            if name == "hn_briefing":
                res = await tool_briefing_hn(svc, **kwargs)
                if as_text:
                    return _apply_max_chars(str(res.get("briefing_markdown") or ""), max_chars)
                return res
            if name == "hn_get_story":
                story = await tool_get_hn_story(svc, **kwargs)
                if story is None:
                    return _err(f"Story '{kwargs.get('external_id', '')}' not found.")
                if as_text:
                    return _apply_max_chars(
                        json.dumps(story, indent=2, ensure_ascii=False), max_chars
                    )
                return story
            if name == "hn_search_archive":
                res = await tool_search_archive(svc, **kwargs)
                return (
                    format_archive_search_for_agent(res, max_chars=max_chars)
                    if as_text
                    else res
                )
            if name == "hn_get_artifact":
                art = await tool_get_artifact(svc, **kwargs)
                if art is None:
                    return _err(f"Artifact '{kwargs.get('artifact_id', '')}' not found.")
                if as_text:
                    return _apply_max_chars(
                        json.dumps(art, indent=2, ensure_ascii=False), max_chars
                    )
                return art
            return _err(f"Unknown tool '{name}'.")
        except Exception as exc:
            return _err(str(exc))
