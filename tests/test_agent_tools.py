import builtins
from unittest.mock import AsyncMock, patch

import pytest

from gojo.agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    GojoAgentToolkit,
    filter_items_by_query,
    format_archive_search_for_agent,
    format_deep_dive_for_agent,
    format_radar_for_agent,
    format_search_for_agent,
    format_summary_for_agent,
    tool_briefing_hn,
    tool_deep_dive_hn,
    tool_get_artifact,
    tool_get_hn_story,
    tool_radar_hn,
    tool_search_archive,
    tool_search_hn,
    tool_search_hn_full,
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
    assert len(HN_AGENT_TOOL_DEFINITIONS) == 9
    by_name = {t["function"]["name"]: t["function"] for t in HN_AGENT_TOOL_DEFINITIONS}
    assert set(by_name.keys()) == {
        "hn_search",
        "hn_search_full",
        "hn_radar",
        "hn_summarize",
        "hn_deep_dive",
        "hn_briefing",
        "hn_get_story",
        "hn_search_archive",
        "hn_get_artifact",
    }

    # Verify reconciled defaults and 5-feed enum
    search_feeds = by_name["hn_search"]["parameters"]["properties"]["feed_id"]["enum"]
    assert set(search_feeds) == {"hn_top", "hn_new", "hn_best", "hn_ask", "hn_show"}
    assert "focus" in by_name["hn_search"]["parameters"]["properties"]
    assert by_name["hn_briefing"]["parameters"]["properties"]["locale"]["default"] == "th"
    assert by_name["hn_search_full"]["parameters"]["required"] == ["query"]
    assert by_name["hn_deep_dive"]["parameters"]["required"] == ["topic"]


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


@pytest.mark.asyncio
async def test_new_tool_executors_and_formatters() -> None:
    mock_svc = AsyncMock()
    mock_svc.search_hn_archive.return_value = [
        DigestFeedItem(
            feed_provider="hn",
            external_id="501",
            title="PostgreSQL 18 Async I/O",
            score=420,
            descendants=180,
            url="https://example.com/pg18",
            signal_badges=["breakout"],
        )
    ]
    full_out = await tool_search_hn_full(mock_svc, query="PostgreSQL", limit=5, sort_by="date")
    assert full_out["query"] == "PostgreSQL"
    assert full_out["sort_by"] == "date"
    assert full_out["total_returned"] == 1
    fmt_full = format_search_for_agent(full_out)
    assert "HN Archive Query: PostgreSQL" in fmt_full
    assert "PostgreSQL 18 Async I/O" in fmt_full

    mock_svc.radar.return_value = {
        "feeds_scanned": ["hn_top", "hn_best"],
        "total_scanned": 30,
        "focus_applied": "postgres",
        "badges_filter": ["breakout"],
        "top_items": [
            {
                "external_id": "501",
                "title": "PostgreSQL 18 Async I/O",
                "score": 420,
                "descendants": 180,
                "velocity": 42.5,
                "signal_badges": ["breakout"],
                "url": "https://example.com/pg18",
            }
        ],
        "clusters": [
            {
                "cluster_id": 1,
                "size": 1,
                "lead_external_id": "501",
                "lead_title": "PostgreSQL 18 Async I/O",
                "items": [
                    {
                        "external_id": "501",
                        "title": "PostgreSQL 18 Async I/O",
                        "score": 420,
                        "descendants": 180,
                        "velocity": 42.5,
                        "signal_badges": ["breakout"],
                        "url": "https://example.com/pg18",
                    }
                ],
            }
        ],
    }
    radar_out = await tool_radar_hn(
        mock_svc, feeds=["hn_top", "hn_best"], top_n=5, focus="postgres", badges=["breakout"]
    )
    assert radar_out["total_scanned"] == 30
    fmt_radar = format_radar_for_agent(radar_out)
    assert "Six Eyes Radar" in fmt_radar
    assert "PostgreSQL 18 Async I/O" in fmt_radar
    assert len(format_radar_for_agent(radar_out, max_chars=70)) <= 70

    mock_svc.deep_dive.return_value = {
        "topic": "PostgreSQL",
        "stories_analyzed": 1,
        "cached_hits": 1,
        "summaries": [
            {
                "artifact": {
                    "id": "art-1",
                    "external_id": "501",
                    "title": "PostgreSQL 18 Async I/O",
                    "source_url": "https://example.com/pg18",
                    "summary_markdown": "## Summary\nAsync I/O cuts read latency.",
                },
                "completion": {"status": "cached", "cache_layer": "sqlite"},
            }
        ],
        "local_archive_matches": [],
        "errors": [],
    }
    dd_out = await tool_deep_dive_hn(mock_svc, topic="PostgreSQL", max_stories=2)
    assert dd_out["stories_analyzed"] == 1
    fmt_dd = format_deep_dive_for_agent(dd_out)
    assert "# Deep Dive Dossier: PostgreSQL" in fmt_dd
    assert "Async I/O cuts read latency." in fmt_dd
    assert len(format_deep_dive_for_agent(dd_out, max_chars=80)) <= 80


@pytest.mark.asyncio
async def test_gojo_agent_toolkit_schemas_and_lifecycle() -> None:
    ext_svc = AsyncMock()
    async with GojoAgentToolkit(service=ext_svc) as kit:
        openai_schemas = kit.schemas("openai")
        anthropic_schemas = kit.schemas("anthropic")
        gemini_schemas = kit.schemas("gemini")

        assert len(openai_schemas) == 9
        assert len(anthropic_schemas) == 9
        assert len(gemini_schemas) == 9

        assert openai_schemas[0]["type"] == "function"
        assert "function" in openai_schemas[0]

        assert "input_schema" in anthropic_schemas[0]
        assert "name" in anthropic_schemas[0]
        assert "description" in anthropic_schemas[0]

        assert "parameters" in gemini_schemas[0]
        assert "name" in gemini_schemas[0]
        assert "description" in gemini_schemas[0]

        with pytest.raises(ValueError, match="Unsupported schema provider"):
            kit.schemas("unknown")  # type: ignore[arg-type]

    # Externally supplied service must NOT be closed by toolkit
    ext_svc.close.assert_not_awaited()

    # Auto-created service MUST be closed on exit
    owned_svc = AsyncMock()
    with patch("gojo.agent_tools.GojoHNService", return_value=owned_svc):
        async with GojoAgentToolkit() as owned_kit:
            assert owned_kit.service is owned_svc
    owned_svc.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_gojo_agent_toolkit_dispatch_success_json_args_max_chars_and_errors() -> None:
    import json

    mock_svc = AsyncMock()
    mock_svc.radar.return_value = {
        "feeds_scanned": ["hn_top"],
        "total_scanned": 5,
        "focus_applied": "rust",
        "badges_filter": [],
        "top_items": [
            {
                "external_id": "1",
                "title": "Rust Compiler Speedups",
                "score": 300,
                "descendants": 90,
                "velocity": 25.0,
                "signal_badges": ["breakout"],
            }
        ],
        "clusters": [
            {
                "cluster_id": 1,
                "size": 1,
                "lead_external_id": "1",
                "lead_title": "Rust Compiler Speedups",
                "items": [
                    {
                        "external_id": "1",
                        "title": "Rust Compiler Speedups",
                        "score": 300,
                        "descendants": 90,
                        "velocity": 25.0,
                        "signal_badges": ["breakout"],
                    }
                ],
            }
        ],
    }
    mock_svc.source.get_item.return_value = None
    mock_svc.search_hn_archive.side_effect = RuntimeError("Algolia timeout")

    async with GojoAgentToolkit(service=mock_svc) as kit:
        # JSON string arguments + as_text=True + max_chars
        txt = await kit.dispatch(
            "hn_radar",
            json.dumps({"feeds": ["hn_top"], "focus": "rust", "top_n": 3}),
            max_chars=75,
            as_text=True,
        )
        assert isinstance(txt, str)
        assert len(txt) <= 75
        assert txt.endswith("... [truncated]")

        # Dict arguments + as_text=False
        raw = await kit.dispatch("hn_radar", {"feeds": ["hn_top"]}, as_text=False)
        assert isinstance(raw, dict)
        assert raw["total_scanned"] == 5

        # Invalid JSON string -> error envelope (no crash)
        bad_json = await kit.dispatch("hn_radar", "{not valid json", as_text=False)
        assert isinstance(bad_json, dict)
        assert bad_json["ok"] is False
        assert "Invalid JSON" in bad_json["error"]

        # Unknown tool -> error envelope (no crash)
        unknown = await kit.dispatch("hn_nonexistent", {}, as_text=False)
        assert isinstance(unknown, dict)
        assert unknown["ok"] is False
        assert "Unknown tool" in unknown["error"]

        # Runtime exception inside tool -> caught into error envelope
        err_text = await kit.dispatch("hn_search_full", {"query": "rust"}, as_text=True)
        assert isinstance(err_text, str)
        assert "Algolia timeout" in err_text

        # Missing story -> structured error envelope
        missing = await kit.dispatch("hn_get_story", {"external_id": "999999"}, as_text=False)
        assert isinstance(missing, dict)
        assert missing["ok"] is False


@pytest.mark.asyncio
async def test_mcp_server_prompts_resources_and_9_tools() -> None:
    import json
    import gojo.mcp_server as mcp_mod

    class _FullFakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, object] = {}
            self.prompts: dict[str, object] = {}
            self.resources: dict[str, object] = {}

        def tool(self):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn

            return decorator

        def prompt(self):
            def decorator(fn):
                self.prompts[fn.__name__] = fn
                return fn

            return decorator

        def resource(self, uri: str):
            def decorator(fn):
                self.resources[uri] = fn
                return fn

            return decorator

    art = DigestArtifact(
        fingerprint="res" * 21 + "r",
        feed_provider="hn",
        external_id="77",
        title="Recent Cached Story",
        locale="en",
        prompt_version="v5",
        summary_markdown="## Summary\nCached details.",
    )
    mock_svc = AsyncMock()
    mock_svc.list_recent_artifacts.return_value = [art]
    mock_svc.radar.return_value = {
        "feeds_scanned": ["hn_top"],
        "total_scanned": 1,
        "focus_applied": None,
        "badges_filter": [],
        "top_items": [],
        "clusters": [],
    }

    with patch.object(mcp_mod, "_import_fastmcp", return_value=_FullFakeFastMCP):
        server = mcp_mod.create_mcp_server(service_factory=lambda: mock_svc)

    # All 9 tools registered
    assert len(server.tools) == 9
    assert {"hn_search_full", "hn_radar", "hn_deep_dive"}.issubset(server.tools.keys())

    # Prompts registered and return concise anti-slop text
    assert set(server.prompts.keys()) == {"morning_tech_radar", "topic_deep_dive"}
    p1 = server.prompts["morning_tech_radar"](focus="rust")
    assert "hn_radar" in p1 and "rust" in p1
    p2 = server.prompts["topic_deep_dive"](topic="sqlite")
    assert "hn_deep_dive" in p2 and "sqlite" in p2

    # Resources registered
    assert set(server.resources.keys()) == {"gojo://providers", "gojo://archive/recent"}
    recent_json = await server.resources["gojo://archive/recent"]()
    recent_data = json.loads(recent_json)
    assert len(recent_data) == 1
    assert recent_data[0]["title"] == "Recent Cached Story"

    with (
        patch.object(mcp_mod.OllamaProvider, "is_available", new_callable=AsyncMock, return_value=False),
        patch.object(mcp_mod.OllamaProvider, "close", new_callable=AsyncMock),
        patch.object(mcp_mod.AutoLLMProvider, "resolve_provider", new_callable=AsyncMock, side_effect=RuntimeError("no key")),
        patch.object(mcp_mod.AutoLLMProvider, "close", new_callable=AsyncMock),
    ):
        prov_json = await server.resources["gojo://providers"]()
        prov_data = json.loads(prov_json)
        assert "openai_configured" in prov_data
        assert "ollama_available" in prov_data


def test_cli_parser_supercommands_and_exports() -> None:
    import gojo

    for sym in (
        "GojoAgentToolkit",
        "tool_search_hn_full",
        "tool_radar_hn",
        "tool_deep_dive_hn",
        "format_radar_for_agent",
        "format_deep_dive_for_agent",
    ):
        assert hasattr(gojo, sym)

    parser = build_parser()

    args_radar = parser.parse_args(
        ["radar", "--feeds", "top", "best", "--top", "7", "--focus", "ai", "--badges", "breakout", "--json"]
    )
    assert args_radar.command == "radar"
    assert args_radar.feeds == ["top", "best"]
    assert args_radar.top == 7
    assert args_radar.focus == "ai"
    assert args_radar.badges == ["breakout"]
    assert args_radar.json is True

    args_hn_arch = parser.parse_args(["hn-archive", "ebpf", "--limit", "12", "--sort", "date", "--json"])
    assert args_hn_arch.command == "hn-archive"
    assert args_hn_arch.query == "ebpf"
    assert args_hn_arch.limit == 12
    assert args_hn_arch.sort == "date"
    assert args_hn_arch.json is True

    args_dd = parser.parse_args(
        ["deep-dive", "WebAssembly", "--max-stories", "4", "--type", "bullet", "--locale", "th", "--json"]
    )
    assert args_dd.command == "deep-dive"
    assert args_dd.topic == "WebAssembly"
    assert args_dd.max_stories == 4
    assert args_dd.type == "bullet"
    assert args_dd.locale == "th"
    assert args_dd.json is True
