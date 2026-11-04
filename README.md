# Gojo Standalone — Hacker News Intelligence Workflow

<p align="center">
  <img src="assets/gojo_mascot.png" width="280" alt="Gojo Mascot — The Boundless Tech Intelligence Engine" />
</p>

A self-contained, production-grade Hacker News workflow, summarization engine, and AI agent tool suite. Extracted and distilled from `patzelife-plugin-agentdiscord` into a standalone, modular package.

---

## 🌟 Key Features

1. **Hacker News API & Algolia Provider (`gojo.providers.hn`, `gojo.providers.algolia`)**:
   - Direct integration with Hacker News official Firebase API (`/v0/topstories.json`, `/v0/newstories.json`, `/v0/item/{id}.json`).
   - High-performance **Algolia Search API client** fetching complete nested discussion comment trees in a single asynchronous request (eliminates the N+1 query problem).
   - High-throughput asynchronous batch item fetching with concurrency management (`asyncio.gather`).

2. **Security & SSRF Mitigation (`gojo.ssrf`)**:
   - Validates outbound article URLs against private networks (RFC 1918), loopback (`127.0.0.1`, `localhost`), link-local (`169.254.0.0/16`), Carrier-Grade NAT, and IPv6 unroutable addresses before fetching.

3. **Article Extraction Pipeline (`trafilatura` + Jina Fallback)**:
   - **Tier 1 (Trafilatura)**: Precision DOM density & text extraction preserving markdown headers, code blocks (`<pre><code>`), and tables while discarding layout boilerplate.
   - **Tier 2 (Semantic Tag Parsing)**: In-memory fallback prioritizing `<article>` and `<main>` tags.
   - **Tier 3 (Jina Reader)**: Built-in remote fallback (`https://r.jina.ai/{url}`) for complex SPAs.

4. **Multi-Model Provider Gateway (`gojo.providers.llm`)**:
   - **OpenAI**: Native support for GPT-4o / GPT-4o-mini via `OPENAI_API_KEY`.
   - **Google Gemini**: OpenAI-compatible endpoint support via `GEMINI_API_KEY`.
   - **Offline Local LLM (Ollama)**: Zero-cost offline local inference via `http://localhost:11434` with automatic model discovery and thinking tag stripping (`<think>`).
   - **Auto-Discovery**: Dynamically routes to the best available provider or local server without crashing.

5. **Multi-Style AI Summarization & Community Synthesis (`gojo.prompts`)**:
   - **`quick`**: Readable, substantive 2–5 paragraph technical brief with names, numbers, and mechanisms.
   - **`technical`**: Deep dive focusing on stack, APIs, CVEs, benchmarks, and architecture.
   - **`bullet`**: Dense, nested scannable bullet points matching source density.
   - **`thai_concise`**: Natural, high-density Thai language summary.
   - **Community Synthesis**: Synthesizes HN peer reactions, consensus, and alternative tools recommended by commenters.
   - Built-in anti-slop guidelines (bans filler phrases like *"delve"*, *"landscape"*, *"From the title alone"*, *"It is important to note"*).

6. **Executive Tech Briefing Mode (`create_briefing`)**:
   - Holistic synthesis of top stories clustered into thematic categories (*Security & Privacy*, *AI & Models*, *Systems & Embedded*, *Web & Platforms*) with macro-level trend analysis.

7. **Deduplication & Deterministic Fingerprinting (`gojo.fingerprint`)**:
   - SHA-256 fingerprinting: `SHA256("{provider}:{id}:{locale}:{prompt_version}:{type}")`.
   - Guaranteed idempotency: identical requests hit cache without duplicate LLM API costs or token usage.

8. **Pluggable Storage (`gojo.storage`)**:
   - Zero-dependency **SQLite** storage (`gojo_hn.db`) for standalone local usage.
   - **In-Memory** storage for unit testing and ephemeral runs.
   - Extensible `StorageBackend` abstract class for Redis/Postgres/Supabase.

9. **AI Agent Tool Integrations (`gojo.agent_tools`)**:
   - Ready-to-use function definitions and schemas compatible with OpenAI Function Calling, Claude/Anthropic Tools, Gemini, LangChain, and Vercel AI SDK (`hn_search`, `hn_summarize`, `hn_get_story`, `hn_briefing`).

---

## 📁 Architecture Overview

```text
gojo-standalone/
├── gojo/
│   ├── __init__.py           # Package exports
│   ├── types.py              # Normalized Pydantic models (FeedItem, Artifact, etc.)
│   ├── fingerprint.py        # SHA-256 idempotency key generation
│   ├── ssrf.py               # Outbound URL security validator
│   ├── prompts.py            # Anti-slop prompt templates & multi-style modes
│   ├── interpretation.py     # Regex extractor for why_matters, key_takeaway, note
│   ├── service.py            # Core GojoHNService orchestrator
│   ├── agent_tools.py        # AI Agent function schemas & tools
│   ├── cli.py                # Standalone Command-Line Interface
│   ├── providers/
│   │   ├── base.py           # FeedSource ABC
│   │   └── hn.py             # Firebase HN API implementation
│   └── storage/
│       ├── base.py           # StorageBackend ABC
│       ├── memory.py         # Ephemeral In-Memory backend
│       └── sqlite.py         # Self-contained SQLite backend
├── tests/                    # Comprehensive pytest suite (100% passing)
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone or navigate to directory
cd gojo-standalone

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (Optional)

Create a `.env` file (or set environment variables):

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key_here
GOJO_HN_MODEL=gpt-4o-mini
GOJO_HN_DB_PATH=gojo_hn.db
GOJO_HN_JINA_FALLBACK=true
```

---

## 💻 CLI Usage

### Browse Top or New Stories
```bash
# List top stories
python3 -m gojo.cli list --feed top --limit 10

# List new stories as JSON
python3 -m gojo.cli list --feed new --limit 5 --json
```

### Search Headlines
```bash
python3 -m gojo.cli search "Rust" --limit 30
```

### Get Raw Story Metadata
```bash
python3 -m gojo.cli get 50354960
```

### Summarize a Story with AI
```bash
# Quick summary with community comments (English)
python3 -m gojo.cli summarize 50354960 --type quick

# Technical summary in Thai with community synthesis
python3 -m gojo.cli summarize 50354960 --type thai_concise

# Summarize article only without community comments
python3 -m gojo.cli summarize 50354960 --type quick --no-comments
```

### Executive Daily Briefing
Synthesize the top stories on Hacker News into an executive macro-briefing:
```bash
# English briefing of top 10 stories
python3 -m gojo.cli briefing --feed top --limit 10 --locale en

# Thai briefing of top 10 stories
python3 -m gojo.cli briefing --feed top --limit 10 --locale th
```

### Inspect LLM Providers
Check status of configured cloud keys and local Ollama daemon:
```bash
python3 -m gojo.cli providers
```

---

## 🤖 Python API & Agent Integration

### Using `GojoHNService` in Python Code

```python
import asyncio
from gojo import GojoHNService

async def main():
    svc = GojoHNService()
    
    # 1. Fetch top feed
    feed = await svc.list_feed(feed_id="hn_top", page=1)
    for story in feed.items[:5]:
        print(f"[{story.external_id}] {story.title} ({story.score} pts)")

    # 2. Summarize a story (idempotent, checks local SQLite first)
    if feed.items:
        story_id = feed.items[0].external_id
        res = await svc.summarize(
            external_id=story_id,
            summary_type="quick",
            locale="en",
        )
        print("\n--- Summary ---")
        print(res.artifact.summary_markdown)
        print("Why it matters:", res.artifact.summary_extra.get("why_matters"))

    await svc.close()

asyncio.run(main())
```

### Using with AI Agents (Function Calling)

```python
from gojo.agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    tool_search_hn,
    tool_summarize_hn,
    format_search_for_agent,
)
from gojo import GojoHNService

# Register HN_AGENT_TOOL_DEFINITIONS with your OpenAI / Claude client:
# tools = HN_AGENT_TOOL_DEFINITIONS

# When the agent calls 'hn_search':
async def handle_agent_call(name, args):
    svc = GojoHNService()
    if name == "hn_search":
        result = await tool_search_hn(svc, **args)
        return format_search_for_agent(result)
    elif name == "hn_summarize":
        result = await tool_summarize_hn(svc, **args)
        return result["artifact"]["summary_markdown"]
```

---

## 🧪 Testing

Run the full automated test suite:

```bash
python3 -m pytest tests -v
```
