# AGENTS.md — Gojo Standalone ("Six Eyes / Rikugan") Intelligence Engine

<p align="center">
  <img src="assets/gojo_logo.png" width="160" alt="Gojo Intelligence Engine Badge" />
</p>

> **Notice to AI Agents (Cursor, Claude Code, Antigravity, Codex, Devin, Cline, LangChain, CrewAI):**
> This file is the authoritative machine-readable guide and operational contract for interacting with, extending, or embedding **Gojo Standalone**.

---

## 1. System Identity & Mission

**Gojo** is an autonomous, production-grade Hacker News intelligence pipeline and tech harvesting engine. It serves as a specialized sensory organ for AI agents, providing:
* Multi-feed asynchronous ingestion across all 5 Firebase HN feeds (`hn_top`, `hn_new`, `hn_best`, `hn_ask`, `hn_show`) with pooled `httpx.AsyncClient`, `asyncio.Semaphore(12)`, and exponential backoff retries
* Single-request Algolia discussion harvesting with root-subtree scoring, parent→child `[1]`/`[1.1]` path citations, and full-archive story search (`search_stories`)
* Concurrent article + Algolia fetching (`asyncio.gather`) and Content-Type-aware extraction (`trafilatura` / optional `[docs]` `pypdf` via `asyncio.to_thread` $\rightarrow$ semantic HTML $\rightarrow$ SSRF-guarded Jina Reader with Markdown newline preservation)
* Multi-model LLM gateway with true `AutoLLMProvider` runtime failover (`OpenAI` $\rightarrow$ `Gemini` $\rightarrow$ `Ollama` at `localhost:11434`), cooldown circuit breaker, universal `<think>` tag stripping, and accurate `last_used_model` recording
* Six Eyes Signal Engine (`gojo/signals.py`): pure-Python `velocity`, `debate_ratio`, `signal_badges` (`breakout`, `rising`, `high-debate`, `slow-burn`), 64-bit SimHash + URL canonicalization event clustering, and zero-token Okapi BM25 `--focus` re-ranking
* Anti-slop prompt engineering across 4 styles (`quick`, `technical`, `bullet`, `thai_concise`) with Consensus vs. Contrarian/Debunked synthesis and English + Thai structured parsing (`gojo/interpretation.py`)
* Hardened SSRF & transport security: static + async DNS A/AAAA resolution (`validate_public_url_with_dns`), hop-by-hop redirect validation (`follow_redirects=False`, max 5 hops), 2MB streaming byte cap, and non-standard IPv4 / IPv4-mapped IPv6 / NAT64 / SIIT / 6to4 / Teredo blocking
* Local-first WAL SQLite storage (`busy_timeout=5000`, `asyncio.to_thread`) with deterministic SHA-256 fingerprint caching, zero-dependency FTS5 archive search (`hn_artifacts_fts` + BM25), and `hn_story_snapshots` for `--since-last` incremental delta briefings
* Optional lazy-loaded FastMCP stdio server (`gojo/mcp_server.py`, `pip install -e ".[mcp]"`)

---

## 2. Ingestion & Execution Modes for Agents

Agents can interface with Gojo through three execution patterns:

### Pattern A: Python In-Process (Recommended for Coding & Subagent Frameworks)
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

# 1. Provide HN_AGENT_TOOL_DEFINITIONS (6 tools) directly to LLM tool registers
# 2. Execute tool requests asynchronously with optional max_chars token budgeting:
async def handle_tool(name: str, args: dict):
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

### Pattern B: CLI Subprocess (Recommended for Terminal / Shell Agents)
Agents can execute CLI commands directly and parse stdout:
```bash
# Browse any of the 5 feeds (top|new|best|ask|show) with optional BM25 focus & cache bypass
python3 -m gojo.cli list --feed top --limit 5 --focus "rust kernel" --force --json

# Search headlines matching query
python3 -m gojo.cli search "Kernel" --feed best --limit 20 --focus "linux" --json

# Search locally archived summaries via SQLite FTS5
python3 -m gojo.cli archive-search "ebpf" --limit 10 --json

# Fetch single story metadata
python3 -m gojo.cli get 50354960 --json

# Generate AI summary with subtree-scored community discussions (auto-cached)
python3 -m gojo.cli summarize 50354960 --type quick --json

# Generate executive briefing with focus & since-last snapshot deltas
python3 -m gojo.cli briefing --feed top --limit 10 --locale en --focus "ai systems" --since-last --json

# Inspect active LLM providers and available local models
python3 -m gojo.cli providers --json
```

### Pattern C: FastMCP Server over Stdio (Optional `[mcp]` Extra)
Install with `pip install -e ".[mcp]"` (and optionally `pip install -e ".[docs]"` for PDF extraction), then launch:
```bash
python3 -m gojo.cli mcp
```
Exposes all 6 tools (`hn_search`, `hn_summarize`, `hn_briefing`, `hn_get_story`, `hn_search_archive`, `hn_get_artifact`) over stdio via `gojo.mcp_server.create_mcp_server()`.

---

## 3. Tool Specifications & Schemas

All 6 function definitions are exported via `gojo.agent_tools.HN_AGENT_TOOL_DEFINITIONS`. Formatters (`format_search_for_agent`, `format_summary_for_agent`, `format_archive_search_for_agent`) accept an optional `max_chars: int | None` hard character budget for agent context windows.

### `hn_search`
Search or browse Hacker News feeds with signal badges and optional BM25 focus re-ranking.
* **`feed_id`** (`string`, optional, default: `"hn_top"`): One of `"hn_top"`, `"hn_new"`, `"hn_best"`, `"hn_ask"`, `"hn_show"`.
* **`page`** (`integer`, optional, default: `1`): 1-based page index.
* **`query`** (`string`, optional): Case-insensitive title substring filter.
* **`focus`** (`string`, optional): Zero-token Okapi BM25 topic query to re-rank returned stories by relevance.

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

Before committing or delivering any code changes, execute the automated test suite:

```bash
# Run complete test suite (must pass 100%)
python3 -m pytest tests -v
```

All 83 unit tests covering storage (WAL SQLite, FTS5, snapshots), SSRF security (static, DNS, redirect hops), Six Eyes signals (velocity, SimHash, clustering, BM25), prompt parsing (EN/TH), Algolia subtree discussions, LLM failover, agent tools, CLI, and briefing workflows must pass.
