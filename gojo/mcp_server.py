"""
Optional FastMCP server wrapper for Gojo Standalone.

Importing this module has zero mandatory third-party MCP dependencies.
`mcp` (specifically `mcp.server.fastmcp.FastMCP`) is imported lazily when
`create_mcp_server()` or `run_mcp_server()` is invoked.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .agent_tools import (
    format_archive_search_for_agent,
    format_search_for_agent,
    format_summary_for_agent,
    tool_briefing_hn,
    tool_get_artifact,
    tool_get_hn_story,
    tool_search_archive,
    tool_search_hn,
    tool_summarize_hn,
)
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
    Create and configure a FastMCP server exposing Gojo Hacker News tools:
    - hn_search
    - hn_summarize
    - hn_briefing
    - hn_get_story
    - hn_search_archive
    - hn_get_artifact
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
    ) -> str:
        """Browse or search Hacker News feeds ('hn_top', 'hn_new', 'hn_best', 'hn_ask', 'hn_show')."""
        svc = _get_service()
        res = await tool_search_hn(
            svc, feed_id=feed_id, page=page, query=query, focus=focus
        )
        return format_search_for_agent(res)

    @mcp.tool()
    async def hn_summarize(
        external_id: str,
        summary_type: str = "quick",
        locale: str = "en",
        include_comments: bool = True,
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
        return format_summary_for_agent(res)

    @mcp.tool()
    async def hn_briefing(
        feed_id: str = "hn_top",
        limit: int = 10,
        locale: str = "th",
        focus: str | None = None,
        since_last: bool = False,
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
        return str(res["briefing_markdown"])

    @mcp.tool()
    async def hn_get_story(external_id: str) -> str:
        """Fetch raw submission metadata (score, author, URL, comments count) for a Hacker News story."""
        svc = _get_service()
        story = await tool_get_hn_story(svc, external_id=external_id)
        if story is None:
            return json.dumps({"error": f"Story '{external_id}' not found."})
        return json.dumps(story, indent=2, ensure_ascii=False)

    @mcp.tool()
    async def hn_search_archive(query: str, limit: int = 10) -> str:
        """Search locally cached Hacker News summaries via SQLite FTS5."""
        svc = _get_service()
        res = await tool_search_archive(svc, query=query, limit=limit)
        return format_archive_search_for_agent(res)

    @mcp.tool()
    async def hn_get_artifact(artifact_id: str) -> str:
        """Retrieve an existing cached summary artifact by its UUID."""
        svc = _get_service()
        artifact = await tool_get_artifact(svc, artifact_id=artifact_id)
        if artifact is None:
            return json.dumps({"error": f"Artifact '{artifact_id}' not found."})
        return json.dumps(artifact, indent=2, ensure_ascii=False)

    return mcp


def run_mcp_server(*, name: str = "gojo-hn") -> None:
    """Create and run the Gojo FastMCP server over stdio."""
    server = create_mcp_server(name=name)
    server.run()
