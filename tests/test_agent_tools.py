import builtins
from unittest.mock import AsyncMock, patch

import pytest

from gojo.agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    filter_items_by_query,
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
from gojo.cli import build_parser
from gojo.types import (
    DigestArtifact,
    DigestCompletion,
    DigestFeedItem,
    DigestFeedResponse,
    DigestSummaryResponse,
)


def test_filter_items_by_query() -> None:
    items = [
        DigestFeedItem(feed_provider="hn", external_id="1", title="Show HN: A Fast Rust Terminal"),
        DigestFeedItem(feed_provider="hn", external_id="2", title="Ask HN: Best Books on Systems?"),
        DigestFeedItem(feed_provider="hn", external_id="3", title="SQLite 3.45 Released"),
    ]
    filtered = filter_items_by_query(items, "rust")
    assert len(filtered) == 1
    assert filtered[0].external_id == "1"

    all_items = filter_items_by_query(items, None)
    assert len(all_items) == 3


def test_agent_tool_definitions_valid() -> None:
    assert len(HN_AGENT_TOOL_DEFINITIONS) >= 6
    by_name = {t["function"]["name"]: t["function"] for t in HN_AGENT_TOOL_DEFINITIONS}
    assert "hn_search" in by_name
    assert "hn_summarize" in by_name
    assert "hn_briefing" in by_name
    assert "hn_get_story" in by_name
    assert "hn_search_archive" in by_name
    assert "hn_get_artifact" in by_name

    # Verify reconciled defaults and 5-feed enum
    search_feeds = by_name["hn_search"]["parameters"]["properties"]["feed_id"]["enum"]
    assert set(search_feeds) == {"hn_top", "hn_new", "hn_best", "hn_ask", "hn_show"}
    assert "focus" in by_name["hn_search"]["parameters"]["properties"]
    assert by_name["hn_briefing"]["parameters"]["properties"]["locale"]["default"] == "th"


@pytest.mark.asyncio
async def test_tool_search_hn() -> None:
    mock_svc = AsyncMock()
    mock_svc.list_feed.return_value = DigestFeedResponse(
        feed_id="hn_top",
        page=1,
        items=[
            DigestFeedItem(
                feed_provider="hn",
                external_id="10",
                title="Antigravity v2",
                score=100,
                descendants=25,
                signal_badges=["breakout", "high-debate"],
            ),
            DigestFeedItem(
                feed_provider="hn",
                external_id="11",
                title="Antigravity Gardening Tips",
                score=10,
                descendants=1,
            ),
        ],
        has_more=False,
        completion=DigestCompletion(status="success", cache_layer="origin"),
    )

    out = await tool_search_hn(
        mock_svc, feed_id="hn_top", page=1, query="antigravity", focus="v2"
    )
    assert out["feed_id"] == "hn_top"
    assert out["total_returned"] == 2
    assert out["focus_applied"] is True
    assert out["items"][0]["title"] == "Antigravity v2"

    fmt = format_search_for_agent(out)
    assert "Antigravity v2" in fmt
    assert "Score: 100" in fmt
    assert "[breakout, high-debate]" in fmt

    # max_chars token budget enforcement
    capped = format_search_for_agent(out, max_chars=80)
    assert len(capped) <= 80
    assert capped.endswith("... [truncated]")


@pytest.mark.asyncio
async def test_tool_summarize_hn() -> None:
    mock_svc = AsyncMock()
    mock_svc.summarize.return_value = DigestSummaryResponse(
        artifact=DigestArtifact(
            fingerprint="abc" * 21 + "a",
            feed_provider="hn",
            external_id="10",
            title="Antigravity v2",
            locale="en",
            prompt_version="v5",
            summary_markdown="## Summary\nGreat release with lots of technical details.",
        ),
        completion=DigestCompletion(status="success", cache_layer="origin"),
    )

    out = await tool_summarize_hn(mock_svc, external_id="10", summary_type="quick")
    assert out["artifact"]["title"] == "Antigravity v2"

    fmt = format_summary_for_agent(out)
    assert "Antigravity v2" in fmt
    assert "Great release" in fmt

    capped = format_summary_for_agent(out, max_chars=90)
    assert len(capped) <= 90
    assert capped.endswith("... [truncated]")


@pytest.mark.asyncio
async def test_tool_briefing_hn_default_locale_th() -> None:
    mock_svc = AsyncMock()
    mock_svc.create_briefing.return_value = "# Executive Technology Briefing\nThai summary."

    out = await tool_briefing_hn(mock_svc, feed_id="hn_best", limit=12, focus="AI", since_last=True)
    assert out["locale"] == "th"
    assert out["feed_id"] == "hn_best"
    assert out["focus"] == "AI"
    assert out["since_last"] is True
    mock_svc.create_briefing.assert_awaited_once_with(
        feed_id="hn_best", limit=12, locale="th", focus="AI", since_last=True
    )


@pytest.mark.asyncio
async def test_tool_search_archive_and_get_artifact() -> None:
    art = DigestArtifact(
        fingerprint="def" * 21 + "d",
        feed_provider="hn",
        external_id="42",
        title="SQLite WAL Internals",
        locale="en",
        prompt_version="v5",
        summary_markdown="## Summary\nWAL enables concurrent readers.",
        summary_extra={"key_takeaway": "WAL improves concurrency."},
    )
    mock_svc = AsyncMock()
    mock_svc.search_archive.return_value = [art]
    mock_svc.get_artifact.return_value = art

    arch_out = await tool_search_archive(mock_svc, query="SQLite", limit=5)
    assert arch_out["total_returned"] == 1
    assert arch_out["artifacts"][0]["title"] == "SQLite WAL Internals"

    fmt_arch = format_archive_search_for_agent(arch_out)
    assert "SQLite WAL Internals" in fmt_arch
    assert "WAL improves concurrency." in fmt_arch

    got = await tool_get_artifact(mock_svc, artifact_id=art.id)
    assert got is not None
    assert got["id"] == art.id


def test_cli_parser_new_flags_and_subcommands() -> None:
    parser = build_parser()

    args_list = parser.parse_args(["list", "--feed", "best", "--focus", "rust", "--force", "--json"])
    assert args_list.feed == "best"
    assert args_list.focus == "rust"
    assert args_list.force is True
    assert args_list.json is True

    args_brief = parser.parse_args(
        ["briefing", "--feed", "ask", "--limit", "20", "--focus", "security", "--since-last", "--json"]
    )
    assert args_brief.feed == "ask"
    assert args_brief.limit == 20
    assert args_brief.focus == "security"
    assert args_brief.since_last is True
    assert args_brief.json is True

    args_arch = parser.parse_args(["archive-search", "sqlite", "--limit", "5", "--json"])
    assert args_arch.command == "archive-search"
    assert args_arch.query == "sqlite"
    assert args_arch.limit == 5

    args_prov = parser.parse_args(["providers", "--json"])
    assert args_prov.json is True

    args_mcp = parser.parse_args(["mcp"])
    assert args_mcp.command == "mcp"


def test_mcp_server_import_safety_and_actionable_error() -> None:
    import gojo.mcp_server as mcp_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError) as exc_info:
            mcp_mod.create_mcp_server()
        assert 'pip install -e ".[mcp]"' in str(exc_info.value)


@pytest.mark.asyncio
async def test_mcp_server_tool_parity_and_shared_lazy_service() -> None:
    import json
    import gojo.mcp_server as mcp_mod

    class _FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn

            return decorator

    art = DigestArtifact(
        fingerprint="mcp" * 21 + "m",
        feed_provider="hn",
        external_id="42",
        title="MCP Parity Story",
        locale="en",
        prompt_version="v5",
        summary_markdown="## Summary\nArtifact details.",
    )
    mock_svc = AsyncMock()
    mock_svc.get_artifact.return_value = art
    mock_svc.source.get_item.return_value = {"id": 42, "title": "MCP Parity Story"}

    factory_calls = 0

    def fake_factory():
        nonlocal factory_calls
        factory_calls += 1
        return mock_svc

    with patch.object(mcp_mod, "_import_fastmcp", return_value=_FakeFastMCP):
        server = mcp_mod.create_mcp_server(service_factory=fake_factory)

    # Full parity with HN_AGENT_TOOL_DEFINITIONS (6 tools)
    expected_tools = {t["function"]["name"] for t in HN_AGENT_TOOL_DEFINITIONS}
    assert set(server.tools.keys()) == expected_tools
    assert "hn_get_artifact" in server.tools

    # Lazy instantiation: factory not called yet
    assert factory_calls == 0

    # Calling multiple tools reuses a single shared service without closing per call
    got_art_json = await server.tools["hn_get_artifact"](artifact_id=art.id)
    got_art = json.loads(got_art_json)
    assert got_art["id"] == art.id
    assert got_art["title"] == "MCP Parity Story"

    got_story_json = await server.tools["hn_get_story"](external_id="42")
    got_story = json.loads(got_story_json)
    assert got_story["id"] == 42

    # Not-found artifact returns structured JSON error
    mock_svc.get_artifact.return_value = None
    missing_json = await server.tools["hn_get_artifact"](artifact_id="nonexistent")
    assert "error" in json.loads(missing_json)

    assert factory_calls == 1
    mock_svc.close.assert_not_awaited()
