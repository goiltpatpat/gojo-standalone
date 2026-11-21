# AGENTS.md — Gojo Standalone ("Six Eyes / Rikugan") Intelligence Engine

<p align="center">
  <img src="assets/gojo_logo.png" width="160" alt="Gojo Intelligence Engine Badge" />
</p>

> **Notice to AI Agents (Cursor, Claude Code, Antigravity, Codex, Devin, Cline, LangChain, CrewAI):**
> This file is the authoritative machine-readable guide and operational contract for interacting with, extending, or embedding **Gojo Standalone** (`gojo-hn` v2.1.0).

---

## 1. System Identity & Mission

**Gojo** is an autonomous, production-grade Hacker News intelligence pipeline and tech harvesting engine (`v2.1.0`, PEP 561 typed via `gojo/py.typed`). It serves as a specialized sensory organ for AI agents, providing:
* Multi-feed asynchronous ingestion across all 5 Firebase HN feeds (`hn_top`, `hn_new`, `hn_best`, `hn_ask`, `hn_show`) with pooled `httpx.AsyncClient`, `asyncio.Semaphore(12)`, exponential backoff retries, and `async with GojoHNService() as svc:` lifecycle management
* Single-request Algolia discussion harvesting with root-subtree scoring, parent→child `[1]`/`[1.1]` path citations, and full-archive story search (`search_stories` / `hn_search_full`)
* Concurrent article + Algolia fetching (`asyncio.gather`) and Content-Type-aware extraction (`trafilatura` / optional `[docs]` `pypdf` via `asyncio.to_thread` $\rightarrow$ semantic HTML $\rightarrow$ SSRF-guarded Jina Reader with Markdown newline preservation)
* Multi-model LLM gateway with true `AutoLLMProvider` runtime failover (`OpenAI` $\rightarrow$ `Gemini` $\rightarrow$ `Ollama` at `localhost:11434`), cooldown circuit breaker, universal `<think>` tag stripping, and accurate `last_used_model` recording
* Six Eyes Signal Engine (`gojo/signals.py`): pure-Python `velocity`, `debate_ratio`, `signal_badges` (`breakout`, `rising`, `high-debate`, `slow-burn`), 64-bit SimHash + URL canonicalization event clustering, zero-token Okapi BM25 `--focus` re-ranking, and multi-feed zero-token `hn_radar` scanning
* Unified **`GojoAgentToolkit`** (`gojo/agent_tools.py`): multi-framework tool schema export (`openai`, `anthropic`, `gemini`) across all 9 tools plus crash-proof `dispatch()` with optional `max_chars` token budgeting
* Anti-slop prompt engineering across 4 styles (`quick`, `technical`, `bullet`, `thai_concise`) with Consensus vs. Contrarian/Debunked synthesis and English + Thai structured parsing (`gojo/interpretation.py`)
* Hardened SSRF & transport security: static + async DNS A/AAAA resolution (`validate_public_url_with_dns`), hop-by-hop redirect validation (`follow_redirects=False`, max 5 hops), 2MB streaming byte cap, and non-standard IPv4 / IPv4-mapped IPv6 / NAT64 / SIIT / 6to4 / Teredo blocking
* Local-first WAL SQLite storage (`busy_timeout=5000`, `asyncio.to_thread`) with deterministic SHA-256 fingerprint caching, zero-dependency FTS5 archive search (`hn_artifacts_fts` + BM25), and `hn_story_snapshots` for `--since-last` incremental delta briefings
* Optional lazy-loaded FastMCP stdio server (`gojo/mcp_server.py`, `pip install -e ".[mcp]"`) exposing 9 tools with `max_chars` budgeting, 2 MCP Prompts (`morning_tech_radar`, `topic_deep_dive`), and 2 MCP Resources (`gojo://providers`, `gojo://archive/recent`)

---

## 2. Ingestion & Execution Modes for Agents

Agents can interface with Gojo through three execution patterns:

### Pattern A: Python In-Process via `GojoAgentToolkit` (Recommended for Coding & Subagent Frameworks)
Use `GojoAgentToolkit` to export provider-native schemas and dispatch any of the 9 tools without manual `if/elif` routing boilerplate (see [examples/agent_sdk_quickstart.py](examples/agent_sdk_quickstart.py)):
```python
from gojo import GojoAgentToolkit

async def run_agent_tool(name: str, args: dict | str):
    async with GojoAgentToolkit() as kit:
        # 1. Export all 9 tool schemas for your LLM SDK ("openai" | "anthropic" | "gemini")
        tools = kit.schemas("openai")

        # 2. Dispatch safely (handles JSON string or dict args, catches errors, enforces max_chars)
        return await kit.dispatch(name, args, max_chars=4000, as_text=True)
```
For direct service access without tool dispatch, use `GojoHNService` as an async context manager:
```python
from gojo import GojoHNService

async with GojoHNService() as svc:
    radar = await svc.radar(feeds=["hn_top", "hn_best"], top_n=10, focus="rust kernel")
    dossier = await svc.deep_dive(topic="ebpf", max_stories=3, summary_type="technical")
```

### Pattern B: CLI Subprocess (Recommended for Terminal / Shell Agents)
Agents can execute CLI commands directly (`python3 -m gojo.cli` or installed `gojo-hn` entry point) and parse stdout:
```bash
# Browse any of the 5 feeds (top|new|best|ask|show) with optional BM25 focus & cache bypass
python3 -m gojo.cli list --feed top --limit 5 --focus "rust kernel" --force --json

# Search headlines matching query
python3 -m gojo.cli search "Kernel" --feed best --limit 20 --focus "linux" --json

# Zero-LLM-token Six Eyes Signal Radar across multiple feeds with clustering & badge filters
python3 -m gojo.cli radar --feeds top best --top 10 --focus "ai systems" --badges breakout high-debate --json

# Search the full historical Hacker News archive via Algolia
python3 -m gojo.cli hn-archive "sqlite wal" --limit 15 --sort relevance --json

# Search locally archived summaries via SQLite FTS5
python3 -m gojo.cli archive-search "ebpf" --limit 10 --json

# Fetch single story metadata
python3 -m gojo.cli get 50354960 --json

# Generate AI summary with subtree-scored community discussions (auto-cached)
python3 -m gojo.cli summarize 50354960 --type quick --json

# Run a concurrent, cache-aware multi-story research dossier across local + HN archives
python3 -m gojo.cli deep-dive "rust async" --max-stories 3 --type technical --locale en --json

# Generate executive briefing with focus & since-last snapshot deltas
python3 -m gojo.cli briefing --feed top --limit 10 --locale en --focus "ai systems" --since-last --json

# Inspect active LLM providers and available local models
python3 -m gojo.cli providers --json
```

### Pattern C: FastMCP Server over Stdio (Optional `[mcp]` Extra)
Install with `pip install -e ".[mcp]"` (or `pip install -e ".[all]"` for AI + MCP + PDF extraction), then launch:
```bash
python3 -m gojo.cli mcp
```
See [examples/mcp_client_config.json](examples/mcp_client_config.json) for a drop-in MCP client configuration. `gojo.mcp_server.create_mcp_server()` exposes:
* **9 Tools** (with `max_chars` context-window budgeting on formatted outputs): `hn_search`, `hn_search_full`, `hn_radar`, `hn_summarize`, `hn_deep_dive`, `hn_briefing`, `hn_get_story`, `hn_search_archive`, `hn_get_artifact`.
* **2 MCP Prompts**:
  * `morning_tech_radar(focus: str = "")`: Zero-token radar scan across `hn_top` and `hn_best`, followed by technical summaries of the top 2–3 engineering stories.
  * `topic_deep_dive(topic: str)`: Compound `hn_deep_dive` workflow cross-referencing local archive matches and HN thread disagreements.
* **2 MCP Resources**:
  * `gojo://providers`: JSON snapshot of configured cloud keys, local Ollama availability/models, and resolved active provider.
  * `gojo://archive/recent`: JSON list of the 20 most recent locally cached summary artifacts.

---

## 3. Tool Specifications & Schemas

All 9 function definitions are exported via `gojo.agent_tools.HN_AGENT_TOOL_DEFINITIONS` (and reformatted via `GojoAgentToolkit.schemas("openai" | "anthropic" | "gemini")`). Formatters (`format_search_for_agent`, `format_radar_for_agent`, `format_summary_for_agent`, `format_deep_dive_for_agent`, `format_archive_search_for_agent`) accept an optional `max_chars: int | None` hard character budget for agent context windows.

### `hn_search`
Search or browse Hacker News feeds with signal badges and optional BM25 focus re-ranking.
* **`feed_id`** (`string`, optional, default: `"hn_top"`): One of `"hn_top"`, `"hn_new"`, `"hn_best"`, `"hn_ask"`, `"hn_show"`.
* **`page`** (`integer`, optional, default: `1`): 1-based page index.
* **`query`** (`string`, optional): Case-insensitive title substring filter.
* **`focus`** (`string`, optional): Zero-token Okapi BM25 topic query to re-rank returned stories by relevance.

### `hn_search_full`
Search the full historical Hacker News archive via Algolia with Six Eyes signal enrichment (`velocity`, `debate_ratio`, `signal_badges`).
* **`query`** (`string`, **required**): Full-text search query across historical Hacker News stories.
* **`limit`** (`integer`, optional, default: `15`): Maximum number of stories to return.
* **`sort_by`** (`string`, optional, default: `"relevance"`): Sort order — `"relevance"` or `"date"`.

### `hn_radar`
Zero-LLM-token Six Eyes Signal Radar: scans multiple HN feeds concurrently, deduplicates stories, computes velocity and debate signals, filters by badges, optionally re-ranks by BM25 focus, and clusters related stories.
* **`feeds`** (`array[string]`, optional, default: `["hn_top", "hn_best"]`): Feeds to scan concurrently (`"hn_top"`, `"hn_new"`, `"hn_best"`, `"hn_ask"`, `"hn_show"`).
* **`top_n`** (`integer`, optional, default: `10`): Maximum number of top-ranked stories to return.
* **`focus`** (`string`, optional): Zero-token Okapi BM25 focus topic to re-rank stories.
* **`badges`** (`array[string]`, optional): Optional signal badge filter (`"breakout"`, `"rising"`, `"high-debate"`, `"slow-burn"`).

### `hn_summarize`
Fetch full article and generate an anti-slop technical summary with Consensus vs. Contrarian/Debunked community synthesis and `[1]`/`[1.1]` thread-path citations.
* **`external_id`** (`string`, **required**): Hacker News story ID (e.g. `"50354960"`).
* **`summary_type`** (`string`, optional, default: `"quick"`):
  * `"quick"`: Substantive 2–5 paragraph technical brief with names, numbers, and mechanisms.
  * `"technical"`: Deep dive focusing on stack, APIs, CVEs, benchmarks, and architecture.
  * `"bullet"`: Dense, nested scannable bullet points matching source density.
  * `"thai_concise"`: Natural, high-density Thai language summary.
* **`locale`** (`string`, optional, default: `"en"`): Target language ISO code (e.g. `"en"`, `"th"`).
* **`include_comments`** (`boolean`, optional, default: `true`): Whether to fetch and synthesize subtree-scored peer comments from Algolia.

### `hn_deep_dive`
Compound topic research pipeline: searches both the local SQLite FTS5 archive and the full HN Algolia archive, selects top distinct stories, summarizes them concurrently (reusing the deterministic SHA-256 cache), and returns a structured research dossier.
* **`topic`** (`string`, **required**): Technical topic or keyword to research across Hacker News.
* **`max_stories`** (`integer`, optional, default: `3`): Maximum number of distinct stories to summarize.
* **`summary_type`** (`string`, optional, default: `"technical"`): Summary style (`"quick"`, `"technical"`, `"bullet"`, `"thai_concise"`).
* **`locale`** (`string`, optional, default: `"en"`): Target language ISO code (e.g. `"en"`, `"th"`).
* **`include_comments`** (`boolean`, optional, default: `true`): Whether to synthesize HN community comments for each story.

### `hn_briefing`
Synthesize Hacker News developments into an Executive Technology Briefing with thematic clustering, signal badges, SimHash event clustering, optional focus re-ranking, and snapshot deltas.
* **`feed_id`** (`string`, optional, default: `"hn_top"`): One of `"hn_top"`, `"hn_new"`, `"hn_best"`, `"hn_ask"`, `"hn_show"`.
* **`limit`** (`integer`, optional, default: `10`): Number of stories to include.
* **`locale`** (`string`, optional, default: `"th"`): Language ISO code (`"th"` or `"en"`).
* **`focus`** (`string`, optional): BM25 focus topic to prioritize relevant stories.
* **`since_last`** (`boolean`, optional, default: `false`): Compare against prior `hn_story_snapshots` sightings to annotate `NEW` and surging stories.

### `hn_get_story`
Fetch raw submission metadata (score, author, URL, comments count) for a Hacker News story.
* **`external_id`** (`string`, **required**): Hacker News story ID.

### `hn_search_archive`
Search previously summarized Hacker News digest artifacts in the local SQLite FTS5 (`hn_artifacts_fts`) or memory cache.
* **`query`** (`string`, **required**): Full-text search query over archived titles and summaries.
* **`limit`** (`integer`, optional, default: `10`): Maximum number of matching artifacts to return.

### `hn_get_artifact`
Retrieve a previously generated digest summary artifact by its UUID.
* **`artifact_id`** (`string`, **required**): UUID of the stored digest artifact.

---

## 4. Architectural Invariants for Agents

When contributing to or modifying Gojo, agents **must strictly preserve** these architectural invariants:

1. **SSRF Guard Invariance (`gojo/ssrf.py`, `gojo/service.py`):**
   * Never bypass or weaken `is_public_http_url_for_fetch()` or `validate_public_url_with_dns()`.
   * Outbound HTTP fetches must enforce hop-by-hop redirect validation (`follow_redirects=False`, max 5 hops), the 2MB streaming byte cap, and defense-in-depth validation on Jina Reader fallbacks.
   * Outbound requests must never reach private IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.1`), link-local (`169.254.0.0/16`), CGNAT (`100.64.0.0/10`), non-standard IPv4 literals (decimal/octal/hex/shorthand), IPv4-mapped IPv6, NAT64/SIIT/6to4/Teredo embeddings, or userinfo credentials in URLs.

2. **Deterministic Fingerprint Invariance (`gojo/fingerprint.py`):**
   * Cache identity must remain strictly deterministic:
     $$\text{Fingerprint} = \text{SHA256}(\text{feed\_provider} : \text{external\_id} : \text{locale} : \text{prompt\_version} : \text{summary\_type})$$
   * Do not alter the fingerprint formula without bumping `prompt_version` or `fingerprint_version`.

3. **Anti-Slop Content Discipline (`gojo/prompts.py`):**
   * Banned filler phrases must never be introduced:
     * *"delve"*, *"landscape"*, *"leverage"*, *"robust"*, *"synergy"*, *"game-changer"*
     * *"From the title alone"*, *"It is important to note"*, *"In today's world"*, *"In conclusion"*
   * Maintain the voice: confident editor briefing an engineering colleague with concrete nouns and variable sentence lengths.

4. **Multi-Tier Degradation:**
   * Content extraction must gracefully degrade:
     $$\text{Trafilatura / PDF} \longrightarrow \text{In-Memory Semantic HTML} \longrightarrow \text{Jina Reader} \longrightarrow \text{Title Fallback}$$
   * LLM provider resolution and runtime failover must gracefully cascade:
     $$\text{OpenAI Key} \longrightarrow \text{Gemini Key} \longrightarrow \text{Local Ollama (localhost:11434)} \longrightarrow \text{Actionable Error}$$

---

## 5. Verification & Test Suite

Before committing or delivering any code changes, execute the automated test suite (also enforced in `.github/workflows/ci.yml` across Python 3.10–3.12):

```bash
# Install dev dependencies (pytest, pytest-asyncio, ruff)
pip install -e ".[dev]"

# Run complete test suite (must pass 100%)
python3 -m pytest tests -v
```

All 91 unit tests covering storage (WAL SQLite, FTS5, snapshots), SSRF security (static, DNS, redirect hops), Six Eyes signals (velocity, SimHash, clustering, BM25), prompt parsing (EN/TH), Algolia subtree discussions, LLM failover, `GojoAgentToolkit` (9 tools, multi-framework schemas, dispatch), MCP prompts/resources, CLI supercommands, and briefing workflows must pass.
