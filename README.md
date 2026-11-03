# Gojo Standalone — Hacker News Intelligence Workflow

A self-contained, production-grade Hacker News workflow, summarization engine, and AI agent tool suite. Extracted and distilled from `patzelife-plugin-agentdiscord` into a standalone, modular package.

---

## 🌟 Key Features

1. **Hacker News API Provider (`gojo.providers.hn`)**:
   - Direct integration with Hacker News official Firebase API (`/v0/topstories.json`, `/v0/newstories.json`, `/v0/item/{id}.json`).
   - High-throughput asynchronous batch item fetching with concurrency management (`asyncio.gather`).

2. **Security & SSRF Mitigation (`gojo.ssrf`)**:
   - Validates outbound article URLs against private networks (RFC 1918), loopback (`127.0.0.1`, `localhost`), link-local (`169.254.0.0/16`), Carrier-Grade NAT, and IPv6 unroutable addresses before fetching.

3. **Article Extraction & Jina Fallback (`gojo.service`)**:
   - In-memory HTML parsing prioritizing semantic `<article>` and `<main>` tags while stripping scripts, tracking codes, and noisy boilerplate.
   - Built-in Jina Reader fallback (`https://r.jina.ai/{url}`) for clean markdown extraction of paywalled or complex SPAs.

4. **Multi-Style AI Summarization (`gojo.prompts`)**:
   - **`quick`**: Readable, substantive 2–5 paragraph technical brief with names, numbers, and mechanisms.
   - **`technical`**: Deep dive focusing on stack, APIs, CVEs, benchmarks, and architecture.
   - **`bullet`**: Dense, nested scannable bullet points matching source density.
   - **`thai_concise`**: Natural, high-density Thai language summary.
   - Built-in anti-slop guidelines (bans filler phrases like *"delve"*, *"landscape"*, *"From the title alone"*, *"It is important to note"*).
   - Structured sections: `## Summary`, `### Why it matters`, `### Key takeaway`, `### Gojo note`.

5. **Deduplication & Deterministic Fingerprinting (`gojo.fingerprint`)**:
   - SHA-256 fingerprinting: `SHA256("{provider}:{id}:{locale}:{prompt_version}:{type}")`.
   - Guaranteed idempotency: identical requests hit cache without duplicate LLM API costs or token usage.

6. **Pluggable Storage (`gojo.storage`)**:
   - Zero-dependency **SQLite** storage (`gojo_hn.db`) for standalone local usage.
   - **In-Memory** storage for unit testing and ephemeral runs.
   - Extensible `StorageBackend` abstract class for Redis/Postgres/Supabase.

7. **AI Agent Tool Integrations (`gojo.agent_tools`)**:
   - Ready-to-use function definitions and schemas compatible with OpenAI Function Calling, Claude/Anthropic Tools, Gemini, LangChain, and Vercel AI SDK.
   - Built-in Markdown formatters (`format_search_for_agent`, `format_summary_for_agent`) to feed clean context into LLM prompts.

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
# Quick summary (English)
python3 -m gojo.cli summarize 50354960 --type quick

# Technical summary in Thai
python3 -m gojo.cli summarize 50354960 --type thai_concise

# Bullet mode
python3 -m gojo.cli summarize 50354960 --type bullet
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
