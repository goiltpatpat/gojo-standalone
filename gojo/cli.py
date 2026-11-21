"""
Command line interface for Gojo Hacker News & Multi-Sensor workflow.
Usage:
    python -m gojo.cli list [--feed top|new|best|ask|show|lobsters_hot|lobsters_new|github_trending] [--page 1] [--focus "..."] [--force]
    python -m gojo.cli search <query> [--feed top|new|best|ask|show|lobsters_hot|lobsters_new|github_trending] [--focus "..."]
    python -m gojo.cli radar [--feeds top best] [--top 10] [--focus "..."] [--badges breakout] [--personalized] [--multi-sensor] [--json]
    python -m gojo.cli pulse <topic> [--days 30] [--limit 50] [--json]
    python -m gojo.cli hn-archive <query> [--limit 15] [--sort relevance|date] [--json]
    python -m gojo.cli archive-search <query> [--limit 10] [--json]
    python -m gojo.cli get <story_id>
    python -m gojo.cli summarize <story_id> [--type quick|technical|bullet|thai_concise] [--no-comments]
    python -m gojo.cli deep-dive <topic> [--max-stories 3] [--type technical] [--locale en] [--json]
    python -m gojo.cli briefing [--feed top|new|best|ask|show|lobsters_hot|lobsters_new|github_trending] [--limit 10] [--locale th|en] [--focus "..."] [--since-last] [--json]
    python -m gojo.cli providers [--json]
    python -m gojo.cli mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from .agent_tools import (
    filter_items_by_query,
    format_deep_dive_for_agent,
    format_radar_for_agent,
    format_topic_pulse_for_agent,
)
from .mcp_server import run_mcp_server
from .providers.hn import SUPPORTED_FEED_VARIANTS
from .providers.llm import AutoLLMProvider, OllamaProvider
from .service import GojoHNService
from .signals import rank_items_by_focus
from .types import ALL_SUPPORTED_FEED_IDS

_FEED_MAP: dict[str, str] = {
    "top": "hn_top",
    "new": "hn_new",
    "best": "hn_best",
    "ask": "hn_ask",
    "show": "hn_show",
    "lobsters_hot": "lobsters_hot",
    "lobsters_new": "lobsters_new",
    "github_trending": "github_trending",
}


def _resolve_feed_id(feed_arg: str) -> str:
    cleaned = (feed_arg or "").strip()
    if cleaned in _FEED_MAP:
        return _FEED_MAP[cleaned]
    if cleaned in ALL_SUPPORTED_FEED_IDS:
        return cleaned
    return "hn_top"


def _parse_multi_arg(values: list[str] | str | None) -> list[str] | None:
    if not values:
        return None
    raw_list = [values] if isinstance(values, str) else list(values)
    out: list[str] = []
    for entry in raw_list:
        for part in str(entry).split(","):
            cleaned = part.strip()
            if cleaned:
                out.append(cleaned)
    return out or None


def _print_items(items, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps([it.model_dump(mode="json") for it in items], indent=2, ensure_ascii=False))
        return

    if not items:
        print("No items found.")
        return

    for it in items:
        badges = getattr(it, "signal_badges", None) or []
        badge_str = f" \033[1;33m[{', '.join(badges)}]\033[0m" if badges else ""
        print(f"\033[1;36m[{it.external_id}]\033[0m \033[1m{it.title}\033[0m{badge_str}")
        meta = []
        if it.score is not None:
            meta.append(f"⭐ {it.score}")
        if it.by:
            meta.append(f"👤 {it.by}")
        if it.descendants is not None:
            meta.append(f"💬 {it.descendants}")
        print(f"    {' · '.join(meta)}")
        if it.url:
            print(f"    \033[90m{it.url}\033[0m")
        print()


async def _run_cli(args: argparse.Namespace) -> None:
    async with GojoHNService() as svc:
        if args.command == "list":
            feed_id = _resolve_feed_id(args.feed)
            res = await svc.list_feed(
                feed_id=feed_id,
                page=args.page,
                page_size=args.limit,
                force_refresh=getattr(args, "force", False),
            )
            items = res.items
            focus = getattr(args, "focus", None)
            if focus and focus.strip() and items:
                items = [it for it, _score in rank_items_by_focus(items, focus.strip())]
                res = res.model_copy(update={"items": items})

            if args.json:
                print(json.dumps(res.model_dump(mode="json"), indent=2, ensure_ascii=False))
            else:
                focus_note = f" | Focus: '{focus.strip()}'" if focus and focus.strip() else ""
                print(
                    f"--- Hacker News ({feed_id.upper()}) Page {res.page} "
                    f"[Cache: {res.completion.cache_layer}]{focus_note} ---"
                )
                _print_items(res.items)

        elif args.command == "search":
            feed_id = _resolve_feed_id(args.feed)
            res = await svc.list_feed(feed_id=feed_id, page=args.page, page_size=args.limit or 30)
            filtered = filter_items_by_query(res.items, args.query)
            focus = getattr(args, "focus", None)
            if focus and focus.strip() and filtered:
                filtered = [it for it, _score in rank_items_by_focus(filtered, focus.strip())]

            if args.json:
                print(json.dumps([it.model_dump(mode="json") for it in filtered], indent=2, ensure_ascii=False))
            else:
                print(f"--- Search results for '{args.query}' in {feed_id.upper()} ---")
                _print_items(filtered)

        elif args.command == "archive-search":
            artifacts = await svc.search_archive(query=args.query, limit=args.limit)
            if args.json:
                print(
                    json.dumps(
                        [art.model_dump(mode="json") for art in artifacts],
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(f"--- Local Archive Search for '{args.query}' ({len(artifacts)} matches) ---")
                if not artifacts:
                    print("No matching archived summaries found.")
                for art in artifacts:
                    print(
                        f"\033[1;36m[{art.external_id}]\033[0m \033[1m{art.title or 'Untitled'}\033[0m "
                        f"({art.summary_type}, {art.locale}) · Artifact: {art.id}"
                    )
                    takeaway = (art.summary_extra or {}).get("key_takeaway")
                    if takeaway:
                        print(f"    Takeaway: {takeaway}")
                    print()

        elif args.command == "get":
            story = await svc.source.get_item(external_id=str(args.story_id))
            if not story:
                print(f"Story ID {args.story_id} not found.", file=sys.stderr)
                sys.exit(1)
            if args.json:
                print(json.dumps(story, indent=2, ensure_ascii=False))
            else:
                print(f"\033[1;36m[{story.get('id')}]\033[0m \033[1m{story.get('title')}\033[0m")
                print(f"By: {story.get('by')} | Score: {story.get('score')} | Comments: {story.get('descendants')}")
                if story.get("url"):
                    print(f"URL: {story.get('url')}")
                if story.get("text"):
                    print(f"\n{story.get('text')}")

        elif args.command == "summarize":
            comments_note = "with comments" if getattr(args, "comments", True) else "no comments"
            print(
                f"Summarizing story {args.story_id} ({args.type}, locale: {args.locale}, {comments_note})...",
                file=sys.stderr,
            )
            res = await svc.summarize(
                external_id=str(args.story_id),
                summary_type=args.type,
                locale=args.locale,
                model=args.model,
                force_regenerate=args.force,
                include_comments=getattr(args, "comments", True),
            )
            if args.json:
                print(json.dumps(res.model_dump(mode="json"), indent=2, ensure_ascii=False))
            else:
                art = res.artifact
                print(f"\n# {art.title or 'Hacker News Summary'}")
                print(f"*Status: {res.completion.status} ({res.completion.cache_layer}) · Model: {art.model}*\n")
                print(art.summary_markdown)
                if art.summary_extra:
                    print("\n--- Structured Insights ---")
                    for k, v in art.summary_extra.items():
                        print(f"**{k}**: {v}")

        elif args.command == "briefing":
            feed_id = _resolve_feed_id(args.feed)
            focus = getattr(args, "focus", None)
            since_last = getattr(args, "since_last", False)
            print(
                f"Generating Executive Briefing for {feed_id.upper()} "
                f"(top {args.limit} stories, locale: {args.locale})...\n",
                file=sys.stderr,
            )
            briefing_md = await svc.create_briefing(
                feed_id=feed_id,
                limit=args.limit,
                locale=args.locale,
                model=args.model,
                focus=focus,
                since_last=since_last,
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "feed_id": feed_id,
                            "limit": args.limit,
                            "locale": args.locale,
                            "focus": focus,
                            "since_last": since_last,
                            "briefing_markdown": briefing_md,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(briefing_md)

        elif args.command == "providers":
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
                except Exception as e:
                    active_error = str(e)
            finally:
                await ollama.close()
                await auto.close()

            if args.json:
                print(
                    json.dumps(
                        {
                            "openai_configured": has_openai,
                            "gemini_configured": has_gemini,
                            "ollama_available": ollama_ok,
                            "ollama_base_url": ollama.base_url,
                            "ollama_models": ollama_models,
                            "active_provider": active_provider,
                            "active_error": active_error,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print("--- Gojo LLM Provider Status ---")
                print(f"• OpenAI API Key: {'[Configured]' if has_openai else '[Not Set]'}")
                print(f"• Gemini API Key: {'[Configured]' if has_gemini else '[Not Set]'}")
                if ollama_ok:
                    print(f"• Local Ollama: [Active] at {ollama.base_url}")
                    print(f"  Installed models: {', '.join(ollama_models)}")
                else:
                    print(f"• Local Ollama: [Not Running or No Models] ({ollama.base_url})")

                if active_provider:
                    print(f"\nActive Auto-Selected Provider: \033[1;32m{active_provider.upper()}\033[0m")
                else:
                    print(f"\nActive Auto-Selected Provider: \033[1;31mNone ({active_error})\033[0m")

        elif args.command == "radar":
            raw_feeds = _parse_multi_arg(getattr(args, "feeds", None))
            resolved_feeds = (
                [_resolve_feed_id(f) for f in raw_feeds] if raw_feeds else None
            )
            badges = _parse_multi_arg(getattr(args, "badges", None))
            res = await svc.radar(
                feeds=resolved_feeds,
                top_n=args.top,
                focus=getattr(args, "focus", None),
                badges=badges,
                personalized=getattr(args, "personalized", False),
                include_sensors=getattr(args, "multi_sensor", False),
            )
            if args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                print(format_radar_for_agent(res))

        elif args.command == "pulse":
            res = await svc.topic_pulse(
                topic=args.topic,
                days=args.days,
                limit=args.limit,
            )
            if args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                print(format_topic_pulse_for_agent(res))

        elif args.command == "hn-archive":
            items = await svc.search_hn_archive(
                query=args.query,
                limit=args.limit,
                sort_by=args.sort,
            )
            if args.json:
                print(
                    json.dumps(
                        [it.model_dump(mode="json") for it in items],
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(
                    f"--- HN Full Archive Search for '{args.query}' "
                    f"(sort: {args.sort}, {len(items)} hits) ---"
                )
                _print_items(items)

        elif args.command == "deep-dive":
            print(
                f"Running Deep Dive for '{args.topic}' "
                f"(max {args.max_stories} stories, type: {args.type}, locale: {args.locale})...\n",
                file=sys.stderr,
            )
            res = await svc.deep_dive(
                topic=args.topic,
                max_stories=args.max_stories,
                summary_type=args.type,
                locale=args.locale,
                include_comments=getattr(args, "comments", True),
            )
            if args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                print(format_deep_dive_for_agent(res))

        else:
            print(
                "Please specify a command: list, search, radar, pulse, hn-archive, archive-search, "
                "get, summarize, deep-dive, briefing, providers, or mcp. Run with --help for details."
            )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the Gojo CLI argument parser."""
    parser = argparse.ArgumentParser(description="Gojo Standalone Six Eyes CLI")
    subparsers = parser.add_subparsers(dest="command")
    feed_choices = list(SUPPORTED_FEED_VARIANTS) + [
        "lobsters_hot",
        "lobsters_new",
        "github_trending",
    ]

    # list
    p_list = subparsers.add_parser("list", help="List Hacker News stories from a feed")
    p_list.add_argument("--feed", choices=feed_choices, default="top", help="Feed to fetch")
    p_list.add_argument("--page", type=int, default=1, help="Page number (1-based)")
    p_list.add_argument("--limit", type=int, default=15, help="Number of items to fetch")
    p_list.add_argument("--focus", default=None, help="BM25 focus query to re-rank stories")
    p_list.add_argument("--force", action="store_true", help="Bypass feed cache and refresh from origin")
    p_list.add_argument("--json", action="store_true", help="Output raw JSON")

    # search
    p_search = subparsers.add_parser("search", help="Search headlines in feed")
    p_search.add_argument("query", help="Title substring query")
    p_search.add_argument("--feed", choices=feed_choices, default="top", help="Feed to search")
    p_search.add_argument("--page", type=int, default=1, help="Page number")
    p_search.add_argument("--limit", type=int, default=30, help="Number of stories to scan")
    p_search.add_argument("--focus", default=None, help="BM25 focus query to re-rank matches")
    p_search.add_argument("--json", action="store_true", help="Output raw JSON")

    # archive-search
    p_arch = subparsers.add_parser("archive-search", help="Search local SQLite FTS5 summary archive")
    p_arch.add_argument("query", help="Full-text search query")
    p_arch.add_argument("--limit", type=int, default=10, help="Maximum matching artifacts")
    p_arch.add_argument("--json", action="store_true", help="Output raw JSON")

    # get
    p_get = subparsers.add_parser("get", help="Get raw story details by ID")
    p_get.add_argument("story_id", help="Hacker News story ID")
    p_get.add_argument("--json", action="store_true", help="Output raw JSON")

    # summarize
    p_sum = subparsers.add_parser("summarize", help="Generate or fetch AI summary for a story")
    p_sum.add_argument("story_id", help="Hacker News story ID")
    p_sum.add_argument(
        "--type",
        choices=["quick", "technical", "bullet", "thai_concise"],
        default="quick",
        help="Summary style and depth",
    )
    p_sum.add_argument("--locale", default="en", help="Language code (e.g. en, th)")
    p_sum.add_argument("--model", default=None, help="Model override")
    p_sum.add_argument("--force", action="store_true", help="Force regenerate instead of cache")
    p_sum.add_argument(
        "--no-comments",
        dest="comments",
        action="store_false",
        default=True,
        help="Disable community comments",
    )
    p_sum.add_argument("--json", action="store_true", help="Output full JSON response")

    # briefing
    p_brief = subparsers.add_parser("briefing", help="Generate executive technical briefing for top stories")
    p_brief.add_argument("--feed", choices=feed_choices, default="top", help="Feed to synthesize")
    p_brief.add_argument("--limit", type=int, default=10, help="Number of stories to include (default 10)")
    p_brief.add_argument("--locale", default="th", help="Language code (e.g. th, en)")
    p_brief.add_argument("--model", default=None, help="Model override")
    p_brief.add_argument("--focus", default=None, help="BM25 focus profile to prioritize stories")
    p_brief.add_argument(
        "--since-last",
        dest="since_last",
        action="store_true",
        default=False,
        help="Annotate NEW and surging stories compared to prior snapshot",
    )
    p_brief.add_argument("--json", action="store_true", help="Output structured JSON")

    # providers
    p_prov = subparsers.add_parser("providers", help="Inspect available LLM providers and models")
    p_prov.add_argument("--json", action="store_true", help="Output structured JSON")

    # radar
    p_radar = subparsers.add_parser(
        "radar",
        help="Zero-LLM-token Six Eyes Signal Radar across HN and multi-sensor feeds",
    )
    p_radar.add_argument(
        "--feeds",
        nargs="+",
        default=None,
        help="Feeds to scan (e.g. top best lobsters_hot github_trending or comma-separated)",
    )
    p_radar.add_argument("--top", type=int, default=10, help="Maximum top stories to return")
    p_radar.add_argument("--focus", default=None, help="BM25 focus query to re-rank stories")
    p_radar.add_argument(
        "--badges",
        nargs="+",
        default=None,
        help="Filter by signal badges (e.g. breakout high-debate corroborated)",
    )
    p_radar.add_argument(
        "--personalized",
        action="store_true",
        default=False,
        help="Apply learned adaptive focus profile from local summary history",
    )
    p_radar.add_argument(
        "--multi-sensor",
        dest="multi_sensor",
        action="store_true",
        default=False,
        help="Include lobsters_hot and github_trending alongside HN feeds for cross-source corroboration",
    )
    p_radar.add_argument("--json", action="store_true", help="Output structured JSON")

    # pulse
    p_pulse = subparsers.add_parser(
        "pulse",
        help="Temporal memory pulse/timeline for a topic across historical snapshots and archive",
    )
    p_pulse.add_argument("topic", help="Topic query (e.g. 'rust async' or 'rust, sqlite')")
    p_pulse.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30)")
    p_pulse.add_argument("--limit", type=int, default=50, help="Maximum timeline stories to return")
    p_pulse.add_argument("--json", action="store_true", help="Output structured JSON")

    # hn-archive
    p_hn_arch = subparsers.add_parser(
        "hn-archive",
        help="Search the full historical Hacker News archive via Algolia",
    )
    p_hn_arch.add_argument("query", help="Full-text search query")
    p_hn_arch.add_argument("--limit", type=int, default=15, help="Maximum stories to return")
    p_hn_arch.add_argument(
        "--sort",
        choices=["relevance", "date"],
        default="relevance",
        help="Sort order (relevance or date)",
    )
    p_hn_arch.add_argument("--json", action="store_true", help="Output raw JSON")

    # deep-dive
    p_dd = subparsers.add_parser(
        "deep-dive",
        help="Compound research dossier across local + HN archives with cached summarization",
    )
    p_dd.add_argument("topic", help="Topic to research across Hacker News")
    p_dd.add_argument(
        "--max-stories",
        dest="max_stories",
        type=int,
        default=3,
        help="Maximum distinct stories to summarize",
    )
    p_dd.add_argument(
        "--type",
        choices=["quick", "technical", "bullet", "thai_concise"],
        default="technical",
        help="Summary style and depth",
    )
    p_dd.add_argument("--locale", default="en", help="Language code (e.g. en, th)")
    p_dd.add_argument(
        "--no-comments",
        dest="comments",
        action="store_false",
        default=True,
        help="Disable community comments",
    )
    p_dd.add_argument("--json", action="store_true", help="Output structured JSON")

    # mcp
    subparsers.add_parser("mcp", help="Run the optional Gojo FastMCP server over stdio")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "mcp":
        try:
            run_mcp_server()
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        return

    asyncio.run(_run_cli(args))


if __name__ == "__main__":
    main()
