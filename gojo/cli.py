"""
Command line interface for Gojo Hacker News workflow.
Usage:
    python -m gojo.cli list [--feed top|new] [--page 1]
    python -m gojo.cli search <query>
    python -m gojo.cli get <story_id>
    python -m gojo.cli summarize <story_id> [--type quick|technical|bullet|thai_concise]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .agent_tools import filter_items_by_query
from .service import GojoHNService


def _print_items(items, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps([it.model_dump(mode="json") for it in items], indent=2, ensure_ascii=False))
        return

    if not items:
        print("No items found.")
        return

    for i, it in enumerate(items, 1):
        print(f"\033[1;36m[{it.external_id}]\033[0m \033[1m{it.title}\033[0m")
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
    svc = GojoHNService()
    try:
        if args.command == "list":
            feed_id = "hn_new" if args.feed == "new" else "hn_top"
            res = await svc.list_feed(feed_id=feed_id, page=args.page, page_size=args.limit)
            if args.json:
                print(json.dumps(res.model_dump(mode="json"), indent=2, ensure_ascii=False))
            else:
                print(f"--- Hacker News ({feed_id.upper()}) Page {res.page} [Cache: {res.completion.cache_layer}] ---")
                _print_items(res.items)

        elif args.command == "search":
            feed_id = "hn_new" if args.feed == "new" else "hn_top"
            res = await svc.list_feed(feed_id=feed_id, page=args.page, page_size=args.limit or 30)
            filtered = filter_items_by_query(res.items, args.query)
            if args.json:
                print(json.dumps([it.model_dump(mode="json") for it in filtered], indent=2, ensure_ascii=False))
            else:
                print(f"--- Search results for '{args.query}' in {feed_id.upper()} ---")
                _print_items(filtered)

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
            print(f"Summarizing story {args.story_id} ({args.type}, locale: {args.locale})...", file=sys.stderr)
            res = await svc.summarize(
                external_id=str(args.story_id),
                summary_type=args.type,
                locale=args.locale,
                model=args.model,
                force_regenerate=args.force,
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

        else:
            print("Please specify a command: list, search, get, or summarize. Run with --help for details.")
    finally:
        await svc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gojo Standalone Hacker News CLI")
    subparsers = parser.add_subparsers(dest="command")

    # list
    p_list = subparsers.add_parser("list", help="List Hacker News front page or new stories")
    p_list.add_argument("--feed", choices=["top", "new"], default="top", help="Feed to fetch")
    p_list.add_argument("--page", type=int, default=1, help="Page number (1-based)")
    p_list.add_argument("--limit", type=int, default=15, help="Number of items to fetch")
    p_list.add_argument("--json", action="store_true", help="Output raw JSON")

    # search
    p_search = subparsers.add_parser("search", help="Search headlines in feed")
    p_search.add_argument("query", help="Title substring query")
    p_search.add_argument("--feed", choices=["top", "new"], default="top", help="Feed to search")
    p_search.add_argument("--page", type=int, default=1, help="Page number")
    p_search.add_argument("--limit", type=int, default=30, help="Number of stories to scan")
    p_search.add_argument("--json", action="store_true", help="Output raw JSON")

    # get
    p_get = subparsers.add_parser("get", help="Get raw story details by ID")
    p_get.add_argument("story_id", help="Hacker News item ID")
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
    p_sum.add_argument("--model", default=None, help="OpenAI model override")
    p_sum.add_argument("--force", action="store_true", help="Force regenerate instead of cache")
    p_sum.add_argument("--json", action="store_true", help="Output full JSON response")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    asyncio.run(_run_cli(args))


if __name__ == "__main__":
    main()
