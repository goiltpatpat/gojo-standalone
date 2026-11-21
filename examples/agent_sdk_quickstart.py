"""Quickstart example for the Gojo Standalone Agent SDK (GojoAgentToolkit).

Demonstrates exporting provider-native tool schemas for OpenAI, Anthropic,
and Gemini, and dispatching token-budgeted Hacker News intelligence tools.
"""

from __future__ import annotations

import asyncio

from gojo import GojoAgentToolkit


async def main() -> None:
    async with GojoAgentToolkit() as kit:
        # Export tool definitions for major LLM SDKs
        openai_tools = kit.schemas("openai")
        anthropic_tools = kit.schemas("anthropic")
        gemini_tools = kit.schemas("gemini")

        print(
            f"Exported schemas -> OpenAI: {len(openai_tools)}, "
            f"Anthropic: {len(anthropic_tools)}, Gemini: {len(gemini_tools)}"
        )

        # Zero-token Six Eyes radar with BM25 focus re-ranking
        radar_output = await kit.dispatch(
            "hn_radar",
            {"focus": "rust, ai systems", "top_n": 5},
            max_chars=3000,
        )
        print("\n=== HN Radar ===\n")
        print(radar_output)

        # Query previously archived summaries via local SQLite FTS5
        archive_output = await kit.dispatch(
            "hn_search_archive",
            {"query": "kernel"},
            max_chars=2000,
        )
        print("\n=== Archive Search ===\n")
        print(archive_output)


if __name__ == "__main__":
    asyncio.run(main())
