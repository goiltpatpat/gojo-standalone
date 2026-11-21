# Gojo Standalone — Six Eyes (Rikugan) Hacker News Intelligence Engine

<p align="center">
  <img src="assets/gojo_mascot.png" width="280" alt="Gojo Mascot — The Boundless Tech Intelligence Engine" />
</p>

A self-contained, production-grade Hacker News intelligence pipeline, summarization engine, and AI agent tool suite. Extracted and distilled from `patzelife-plugin-agentdiscord` into a standalone, modular package powered by the **Six Eyes (Rikugan)** signal, extraction, and local-first archive architecture.

---

## 🌟 Key Features

1. **Multi-Feed Hacker News & Algolia Providers (`gojo.providers.hn`, `gojo.providers.algolia`)**:
   - Full coverage of all 5 Hacker News Firebase feeds: `hn_top` (`topstories`), `hn_new` (`newstories`), `hn_best` (`beststories`), `hn_ask` (`askstories`), and `hn_show` (`showstories`), plus `/v0/item/{id}.json`.
   - Pooled `httpx.AsyncClient` connections with `asyncio.Semaphore(12)` concurrency control, exponential backoff retries, and clean `close()` teardown.
   - Single-request **Algolia Search API** client for nested comment trees with root-subtree scoring that preserves parent→child causality and emits `[1]` / `[1.1]` thread-path citations, plus full-archive `search_stories()`.

2. **SSRF & Transport Security (`gojo.ssrf`, `gojo.service`)**:
   - Static validation (`is_public_http_url_for_fetch`) plus async DNS A/AAAA resolution (`validate_public_url_with_dns`) rejecting private RFC 1918 ranges, loopback, link-local, CGNAT, non-standard IPv4 literals (decimal, octal, hex, shorthand), IPv4-mapped IPv6, NAT64/SIIT, 6to4, Teredo, and URL userinfo credentials.
   - Hop-by-hop redirect validation (`follow_redirects=False`, max 5 hops) with a 2MB streaming byte cap (`MAX_RESPONSE_BYTES`) and defense-in-depth validation on the Jina Reader fallback.

3. **Async Content-Type-Aware Extraction Pipeline (`trafilatura` + PDF + Jina Fallback)**:
   - Concurrent article + Algolia discussion fetching via `asyncio.gather`, with CPU-bound extraction offloaded via `asyncio.to_thread`.
   - **Tier 1 (Trafilatura / PDF)**: Tuned Trafilatura DOM extraction preserving headers, code blocks, and tables; optional PDF extraction via `pypdf` (`pip install -e ".[docs]"`); automatic rejection of non-text binary payloads.
   - **Tier 2 (Semantic HTML Parsing)**: In-memory fallback prioritizing `<article>` and `<main>` tags.
   - **Tier 3 (Jina Reader)**: Built-in SSRF-guarded remote fallback (`https://r.jina.ai/{url}`) with Markdown newline preservation for complex SPAs.

4. **Multi-Model LLM Gateway with Runtime Failover (`gojo.providers.llm`)**:
   - **OpenAI** (`OPENAI_API_KEY`), **Google Gemini** (`GEMINI_API_KEY`), and offline **Local Ollama** (`http://localhost:11434` with automatic model discovery) using pooled HTTP/LLM clients.
   - **`AutoLLMProvider` Runtime Failover**: Automatic cascade `OpenAI -> Gemini -> Ollama` with a cooldown circuit breaker on transient errors (429 / 5xx / transport), universal `<think>` tag stripping for reasoning models, and accurate `last_used_model` / `last_used_provider` recording.

5. **Six Eyes Signal Engine (`gojo.signals`)**:
   - Pure-Python, zero-dependency story enrichment computing HN gravity-normalized `velocity`, `debate_ratio`, and deterministic `signal_badges` (`breakout`, `rising`, `high-debate`, `slow-burn`).
   - URL canonicalization (strips `utm_*`, tracking params, fragments, and default ports) and 64-bit deterministic SimHash + Jaccard event clustering to group duplicate/related stories covering the same event.
   - Zero-token local Okapi BM25 `--focus` re-ranking across title, URL, and body text.

6. **Multi-Style AI Summarization & Multilingual Parsing (`gojo.prompts`, `gojo.interpretation`)**:
   - Four anti-slop styles: **`quick`**, **`technical`**, **`bullet`**, and **`thai_concise`**.
   - **Consensus vs. Contrarian / Debunked Synthesis**: Prompt templates separate peer consensus, contrarian pushback, and cited thread paths (`[1]`, `[1.1]`) alongside alternative tools.
   - **Multilingual Structured Extraction**: `interpretation.py` parses `why_matters`, `key_takeaway`, `note`, `community_consensus`, and `alternative_solutions` from both English and Thai headings.

7. **Executive Tech Briefing & Delta Mode (`create_briefing`)**:
   - Clusters top stories into thematic categories (*Security & Privacy*, *AI & Models*, *Systems & Embedded*, *Web & Platforms*) with signal badges, event clustering, optional `--focus` BM25 prioritization, and `--since-last` incremental delta annotations (`NEW` / surging stories) backed by `hn_story_snapshots`.

8. **Local-First WAL SQLite + FTS5 Archive (`gojo.storage`)**:
   - Zero-config **SQLite** backend (`gojo_hn.db`) configured with `PRAGMA journal_mode=WAL`, `busy_timeout=5000`, and `asyncio.to_thread` async execution.
   - Deterministic SHA-256 fingerprinting (`SHA256("{provider}:{id}:{locale}:{prompt_version}:{type}")`) for idempotent caching.
   - Built-in **FTS5** full-text search index (`hn_artifacts_fts` + BM25 ranking) for searching archived summaries and `hn_story_snapshots` for cross-run delta tracking.
   - Ephemeral **In-Memory** backend for tests and extensible `StorageBackend` ABC.

9. **AI Agent Tools & Optional FastMCP Server (`gojo.agent_tools`, `gojo.mcp_server`)**:
   - 6 ready-to-use function schemas in `HN_AGENT_TOOL_DEFINITIONS` (`hn_search`, `hn_summarize`, `hn_briefing`, `hn_get_story`, `hn_search_archive`, `hn_get_artifact`) with `max_chars` token budgeting on agent formatters.
   - Optional stdio **FastMCP** server (`python3 -m gojo.cli mcp`) enabled via `pip install -e ".[mcp]"`.

---

## 📁 Architecture Overview

```text
gojo-standalone/
├── gojo/
│   ├── __init__.py           # Package exports
│   ├── types.py              # Normalized Pydantic models (DigestFeedItem, DigestArtifact, etc.)
│   ├── fingerprint.py        # SHA-256 idempotency key generation
│   ├── ssrf.py               # Static + async DNS SSRF validator & IP classification
│   ├── signals.py            # Six Eyes Signal Engine (velocity, badges, SimHash, clustering, BM25)
│   ├── prompts.py            # Anti-slop prompt templates, Consensus/Contrarian & briefing modes
│   ├── interpretation.py     # English & Thai regex extractor for structured summary insights
│   ├── service.py            # Core GojoHNService orchestrator & hop-by-hop extraction pipeline
│   ├── agent_tools.py        # 6 AI Agent tool schemas, executors & max_chars formatters
│   ├── mcp_server.py         # Optional lazy-loaded FastMCP stdio server wrapper
│   ├── cli.py                # Standalone Command-Line Interface (gojo-hn)
│   ├── providers/
│   │   ├── base.py           # FeedSource ABC
│   │   ├── hn.py             # Firebase HN API (5 feeds, pooled httpx, semaphore + retries)
│   │   ├── algolia.py        # Algolia subtree-scored comments & archive story search
│   │   └── llm.py            # OpenAI, Gemini, Ollama & AutoLLMProvider runtime failover
│   └── storage/
│       ├── base.py           # StorageBackend ABC
│       ├── memory.py         # Ephemeral In-Memory backend
│       └── sqlite.py         # WAL SQLite backend with FTS5 archive & story snapshots
├── tests/                    # Automated pytest suite (83 unit tests)
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Installation

```bash
cd gojo-standalone

# Core dependencies
pip install -r requirements.txt

# Or editable install with optional extras:
pip install -e ".[ai]"          # OpenAI / Gemini SDK
pip install -e ".[mcp]"         # Optional FastMCP stdio server
pip install -e ".[docs]"        # Optional PDF extraction (pypdf)
pip install -e ".[ai,mcp,docs,test]"
```

### 2. Configuration (Optional)

Create a `.env` file (or export environment variables):

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GOJO_HN_MODEL=gpt-4o-mini
GOJO_HN_DB_PATH=gojo_hn.db
GOJO_HN_JINA_FALLBACK=true
```

If neither cloud key is set, Gojo automatically probes local Ollama at `http://localhost:11434` (`OLLAMA_HOST`).

---

## 💻 CLI Usage

### Browse Feeds (`top`, `new`, `best`, `ask`, `show`)
```bash
# List top stories with signal badges
python3 -m gojo.cli list --feed top --limit 10

# List Show HN stories re-ranked by BM25 focus, bypassing cache
python3 -m gojo.cli list --feed show --focus "rust sqlite" --force --json
```

### Search Feed Headlines
```bash
python3 -m gojo.cli search "Rust" --feed best --limit 30 --focus "compiler"
```

### Search Local FTS5 Summary Archive
```bash
python3 -m gojo.cli archive-search "kernel ebpf" --limit 10 --json
```

### Get Raw Story Metadata
```bash
python3 -m gojo.cli get 50354960 --json
```

### Summarize a Story with AI
```bash
# Quick summary with subtree-scored community comments (English)
python3 -m gojo.cli summarize 50354960 --type quick

# High-density Thai summary
python3 -m gojo.cli summarize 50354960 --type thai_concise

# Bypass cache or disable community comments
python3 -m gojo.cli summarize 50354960 --type technical --force --no-comments --json
```

### Executive Daily Briefing & `--since-last` Deltas
```bash
# English briefing of top 10 stories focused on systems/security
python3 -m gojo.cli briefing --feed top --limit 10 --locale en --focus "security kernel"

# Incremental Thai briefing highlighting NEW and surging stories since last run
python3 -m gojo.cli briefing --feed top --limit 10 --locale th --since-last --json
```

### Inspect LLM Providers
```bash
python3 -m gojo.cli providers --json
```

### Run Optional FastMCP Server (stdio)
```bash
# Requires: pip install -e ".[mcp]"
python3 -m gojo.cli mcp
```

---

## 🤖 Python API & Agent Integration

### Using `GojoHNService` in Python Code

```python
import asyncio
from gojo import GojoHNService

async def main():
    svc = GojoHNService()
    try:
        # 1. Fetch top feed (enriched with velocity, debate_ratio, signal_badges)
        feed = await svc.list_feed(feed_id="hn_top", page=1)
        for story in feed.items[:5]:
            badges = f" [{', '.join(story.signal_badges)}]" if story.signal_badges else ""
            print(f"[{story.external_id}] {story.title} ({story.score} pts){badges}")

        # 2. Summarize a story (idempotent, checks local WAL SQLite first)
        if feed.items:
            res = await svc.summarize(
                external_id=feed.items[0].external_id,
                summary_type="quick",
                locale="en",
            )
            print("\n--- Summary ---")
            print(res.artifact.summary_markdown)
            print("Why it matters:", res.artifact.summary_extra.get("why_matters"))

        # 3. Query local FTS5 archive
        archived = await svc.search_archive("rust", limit=5)
        print(f"Archive matches: {len(archived)}")
    finally:
        await svc.close()

asyncio.run(main())
```

### Using with AI Agents (Function Calling)

```python
from gojo import GojoHNService
from gojo.agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    tool_search_hn,
    tool_summarize_hn,
    tool_briefing_hn,
    tool_get_hn_story,
    tool_search_archive,
    tool_get_artifact,
    format_search_for_agent,
    format_summary_for_agent,
    format_archive_search_for_agent,
)

# Register HN_AGENT_TOOL_DEFINITIONS (6 tools) with your OpenAI / Anthropic / Gemini client
async def handle_agent_call(name: str, args: dict):
    svc = GojoHNService()
    try:
        if name == "hn_search":
            res = await tool_search_hn(svc, **args)
            return format_search_for_agent(res, max_chars=4000)
        elif name == "hn_summarize":
            res = await tool_summarize_hn(svc, **args)
            return format_summary_for_agent(res, max_chars=6000)
        elif name == "hn_briefing":
            res = await tool_briefing_hn(svc, **args)
            return res["briefing_markdown"]
        elif name == "hn_get_story":
            return await tool_get_hn_story(svc, **args)
        elif name == "hn_search_archive":
            res = await tool_search_archive(svc, **args)
            return format_archive_search_for_agent(res, max_chars=4000)
        elif name == "hn_get_artifact":
            return await tool_get_artifact(svc, **args)
    finally:
        await svc.close()
```

---

## 🧪 Testing

Run the full automated test suite (83 unit tests):

```bash
python3 -m pytest tests -v
```
