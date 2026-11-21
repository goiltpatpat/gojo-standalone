"""
Optional FastMCP server wrapper for Gojo Standalone.

Importing this module has zero mandatory third-party MCP dependencies.
`mcp` (specifically `mcp.server.fastmcp.FastMCP`) is imported lazily when
`create_mcp_server()` or `run_mcp_server()` is invoked.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from .agent_tools import (
    _apply_max_chars,
    format_archive_search_for_agent,
    format_deep_dive_for_agent,
    format_radar_for_agent,
    format_search_for_agent,
    format_summary_for_agent,
    format_topic_pulse_for_agent,
    tool_briefing_hn,
    tool_deep_dive_hn,
    tool_get_artifact,
    tool_get_hn_story,
    tool_radar_hn,
    tool_search_archive,
    tool_search_hn,
    tool_search_hn_full,
    tool_summarize_hn,
    tool_topic_pulse,
)
from .providers.llm import AutoLLMProvider, OllamaProvider
from .service import GojoHNService

_MCP_INSTALL_HINT = (
    "The optional 'mcp' package is required to run the Gojo MCP server. "
    'Install it with: pip install -e ".[mcp]"'
)


def _import_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_MCP_INSTALL_HINT) from exc
    return FastMCP


def create_mcp_server(
    *,
    name: str = "gojo-hn",
    service_factory: Callable[[], GojoHNService] = GojoHNService,
) -> Any:
    """
    Create and configure a FastMCP server exposing all 10 Gojo Six Eyes tools,
    plus reusable MCP prompts and resources when supported by FastMCP:
    Tools:
    - hn_search
    - hn_search_full
    - hn_radar
    - hn_topic_pulse
    - hn_summarize
    - hn_deep_dive
    - hn_briefing
    - hn_get_story
    - hn_search_archive
    - hn_get_artifact
    Prompts:
    - morning_tech_radar
    - topic_deep_dive
    - adaptive_briefing
    Resources:
    - gojo://providers
    - gojo://archive/recent
    - gojo://profile/focus
    """
    FastMCP = _import_fastmcp()
    mcp = FastMCP(name)

    shared_service: GojoHNService | None = None

    def _get_service() -> GojoHNService:
        nonlocal shared_service
        if shared_service is None:
            shared_service = service_factory()
        return shared_service

    @mcp.tool()
    async def hn_search(
        feed_id: str = "hn_top",
        page: int = 1,
        query: str | None = None,
        focus: str | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Browse or search Hacker News feeds ('hn_top', 'hn_new', 'hn_best', 'hn_ask', 'hn_show')."""
        svc = _get_service()
        res = await tool_search_hn(
            svc, feed_id=feed_id, page=page, query=query, focus=focus
        )
        return format_search_for_agent(res, max_chars=max_chars)

    @mcp.tool()
    async def hn_search_full(
        query: str,
        limit: int = 15,
        sort_by: str = "relevance",
        max_chars: int | None = None,
    ) -> str:
        """Search the full historical Hacker News archive via Algolia with Six Eyes signals."""
        svc = _get_service()
        res = await tool_search_hn_full(
            svc, query=query, limit=limit, sort_by=sort_by
        )
        return format_search_for_agent(res, max_chars=max_chars)

    @mcp.tool()
    async def hn_radar(
        feeds: list[str] | None = None,
        top_n: int = 10,
        focus: str | None = None,
        badges: list[str] | None = None,
        personalized: bool = False,
        include_sensors: bool = False,
        max_chars: int | None = None,
    ) -> str:
        """Zero-LLM-token Six Eyes Signal Radar across HN and optional multi-sensor feeds with adaptive focus and corroboration."""
        svc = _get_service()
        res = await tool_radar_hn(
            svc,
            feeds=feeds,
            top_n=top_n,
            focus=focus,
            badges=badges,
            personalized=personalized,
            include_sensors=include_sensors,
        )
        return format_radar_for_agent(res, max_chars=max_chars)

    @mcp.tool()
    async def hn_topic_pulse(
        topic: str,
        days: int = 30,
        limit: int = 50,
        max_chars: int | None = None,
    ) -> str:
        """Track historical topic momentum, score deltas, debate ratio, and archived takeaways over a lookback window."""
        svc = _get_service()
        res = await tool_topic_pulse(svc, topic=topic, days=days, limit=limit)
        return format_topic_pulse_for_agent(res, max_chars=max_chars)

    @mcp.tool()
    async def hn_summarize(
        external_id: str,
        summary_type: str = "quick",
        locale: str = "en",
        include_comments: bool = True,
        max_chars: int | None = None,
    ) -> str:
        """Fetch full article and generate an anti-slop technical summary with community discussion synthesis."""
        svc = _get_service()
        res = await tool_summarize_hn(
            svc,
            external_id=external_id,
            summary_type=summary_type,
            locale=locale,
            include_comments=include_comments,
        )
        return format_summary_for_agent(res, max_chars=max_chars)

    @mcp.tool()
    async def hn_deep_dive(
        topic: str,
        max_stories: int = 3,
        summary_type: str = "technical",
        locale: str = "en",
        include_comments: bool = True,
        max_chars: int | None = None,
    ) -> str:
        """Compound research pipeline across local + HN archives with concurrent cached summarization."""
        svc = _get_service()
        res = await tool_deep_dive_hn(
            svc,
            topic=topic,
            max_stories=max_stories,
            summary_type=summary_type,
            locale=locale,
            include_comments=include_comments,
        )
        return format_deep_dive_for_agent(res, max_chars=max_chars)

    @mcp.tool()
    async def hn_briefing(
        feed_id: str = "hn_top",
        limit: int = 10,
        locale: str = "th",
        focus: str | None = None,
        since_last: bool = False,
        max_chars: int | None = None,
    ) -> str:
        """Generate a synthesized executive technology briefing from Hacker News stories."""
        svc = _get_service()
        res = await tool_briefing_hn(
            svc,
            feed_id=feed_id,
            limit=limit,
            locale=locale,
            focus=focus,
            since_last=since_last,
        )
        return _apply_max_chars(str(res["briefing_markdown"]), max_chars)

    @mcp.tool()
    async def hn_get_story(external_id: str) -> str:
        """Fetch raw submission metadata (score, author, URL, comments count) for a Hacker News story."""
        svc = _get_service()
        story = await tool_get_hn_story(svc, external_id=external_id)
        if story is None:
            return json.dumps({"error": f"Story '{external_id}' not found."})
        return json.dumps(story, indent=2, ensure_ascii=False)

    @mcp.tool()
    async def hn_search_archive(
        query: str,
        limit: int = 10,
        max_chars: int | None = None,
    ) -> str:
        """Search locally cached Hacker News summaries via SQLite FTS5."""
        svc = _get_service()
        res = await tool_search_archive(svc, query=query, limit=limit)
        return format_archive_search_for_agent(res, max_chars=max_chars)

    @mcp.tool()
    async def hn_get_artifact(artifact_id: str) -> str:
        """Retrieve an existing cached summary artifact by its UUID."""
        svc = _get_service()
        artifact = await tool_get_artifact(svc, artifact_id=artifact_id)
        if artifact is None:
            return json.dumps({"error": f"Artifact '{artifact_id}' not found."})
        return json.dumps(artifact, indent=2, ensure_ascii=False)

    if hasattr(mcp, "prompt"):
        @mcp.prompt()
        def morning_tech_radar(focus: str = "") -> str:
            """Concise morning Hacker News radar workflow prompt."""
            focus_clause = (
                f" with focus='{focus.strip()}'" if focus and focus.strip() else ""
            )
            return (
                f"Run `hn_radar`{focus_clause} across hn_top and hn_best to surface "
                "breakout, rising, and high-debate stories at zero LLM token cost. "
                "Pick the top 2-3 most consequential engineering stories and call `hn_summarize` "
                "(summary_type='technical') on each. Report concrete mechanisms, benchmarks, "
                "and community consensus vs. counterarguments. Skip hype and filler."
            )

        @mcp.prompt()
        def topic_deep_dive(topic: str) -> str:
            """Concise topic deep-dive research workflow prompt."""
            clean_topic = (topic or "").strip() or "systems engineering"
            return (
                f"Run `hn_deep_dive` with topic='{clean_topic}' (max_stories=3, "
                "summary_type='technical', include_comments=True). Cross-reference any "
                "local archive matches, then synthesize concrete technical findings, "
                "trade-offs, production pitfalls, and cited HN thread disagreements. "
                "Keep the brief direct, specific, and free of filler."
            )

        @mcp.prompt()
        def adaptive_briefing() -> str:
            """Personalized multi-sensor Six Eyes briefing workflow prompt."""
            return (
                "Inspect `gojo://profile/focus` to review the learned adaptive focus terms, "
                "then run `hn_radar` with `personalized=True` and `include_sensors=True` to "
                "surface cross-source corroborated clusters across Hacker News, Lobste.rs, and "
                "GitHub Trending at zero LLM token cost. For the top 1-2 recurring themes, "
                "call `hn_topic_pulse` (days=30) to verify whether momentum is rising, steady, "
                "or cooling, and call `hn_summarize` (summary_type='technical') on the highest-signal "
                "corroborated stories. Deliver a direct engineering brief without hype or filler."
            )

    if hasattr(mcp, "resource"):
        @mcp.resource("gojo://providers")
        async def resource_providers() -> str:
            """Inspect configured and active Gojo LLM providers as JSON."""
            has_openai = bool(os.getenv("OPENAI_API_KEY"))
            has_gemini = bool(os.getenv("GEMINI_API_KEY"))
            ollama = OllamaProvider()
            auto = AutoLLMProvider()
            try:
                ollama_ok = await ollama.is_available()
                ollama_models = await ollama.get_available_models() if ollama_ok else []
                active_provider: str | None = None
                active_error: str | None = None
                try:
                    p = await auto.resolve_provider()
                    active_provider = p.provider_name
                except Exception as exc:
                    active_error = str(exc)
                payload = {
                    "openai_configured": has_openai,
                    "gemini_configured": has_gemini,
                    "ollama_available": ollama_ok,
                    "ollama_base_url": ollama.base_url,
                    "ollama_models": ollama_models,
                    "active_provider": active_provider,
                    "active_error": active_error,
                }
            finally:
                await ollama.close()
                await auto.close()
            return json.dumps(payload, indent=2, ensure_ascii=False)

        @mcp.resource("gojo://archive/recent")
        async def resource_archive_recent() -> str:
            """Return recent locally cached summary artifacts as JSON."""
            svc = _get_service()
            artifacts = await svc.list_recent_artifacts(limit=20)
            return json.dumps(
                [art.model_dump(mode="json") for art in artifacts],
                indent=2,
                ensure_ascii=False,
            )

        @mcp.resource("gojo://profile/focus")
        async def resource_profile_focus() -> str:
            """Return the learned adaptive focus profile from recent summary history as JSON."""
            svc = _get_service()
            profile = await svc.get_adaptive_focus_profile()
            return json.dumps(profile, indent=2, ensure_ascii=False)

    return mcp


def run_mcp_server(*, name: str = "gojo-hn") -> None:
    """Create and run the Gojo FastMCP server over stdio."""
    server = create_mcp_server(name=name)
    server.run()
