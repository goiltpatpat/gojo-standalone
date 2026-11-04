# AGENTS.md — Gojo Standalone Intelligence Engine

<p align="center">
  <img src="assets/gojo_logo.png" width="160" alt="Gojo Intelligence Engine Badge" />
</p>

> **Notice to AI Agents (Cursor, Claude Code, Antigravity, Codex, Devin, Cline, LangChain, CrewAI):**  
> This file is the authoritative machine-readable guide and operational contract for interacting with, extending, or embedding **Gojo Standalone**.

---

## 1. System Identity & Mission

**Gojo** is an autonomous, production-grade Hacker News intelligence pipeline and tech harvesting engine. It serves as a specialized sensory organ for AI agents, providing:
* High-throughput asynchronous feed ingestion (Firebase API)
* 1-request full nested comment discussion harvesting (Algolia Search API)
* Multi-tier high-precision content extraction (`trafilatura` $\rightarrow$ semantic HTML $\rightarrow$ Jina Reader)
* Multi-model provider routing (OpenAI, Google Gemini, and offline local Ollama at `localhost:11434`)
* Anti-slop prompt engineering across 4 styles (`quick`, `technical`, `bullet`, `thai_concise`)
* SSRF mitigation against private networks, cloud metadata, and loopback addresses
* Idempotent SHA-256 fingerprint deduplication with zero-config SQLite/Memory storage

---

## 2. Ingestion & Execution Modes for Agents

Agents can interface with Gojo through two primary execution patterns:

### Pattern A: Python In-Process (Recommended for Coding & Subagent Frameworks)
```python
from gojo import GojoHNService
from gojo.agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    tool_search_hn,
    tool_summarize_hn,
    tool_briefing_hn,
    tool_get_hn_story,
    format_search_for_agent,
    format_summary_for_agent,
)

# 1. Provide HN_AGENT_TOOL_DEFINITIONS directly to LLM tool registers
# 2. Execute tool requests asynchronously:
async def handle_tool(name: str, args: dict):
    svc = GojoHNService()
    try:
        if name == "hn_search":
            res = await tool_search_hn(svc, **args)
            return format_search_for_agent(res)  # Token-optimized Markdown
        elif name == "hn_summarize":
            res = await tool_summarize_hn(svc, **args)
            return format_summary_for_agent(res)
        elif name == "hn_briefing":
            res = await tool_briefing_hn(svc, **args)
            return res["briefing_markdown"]
        elif name == "hn_get_story":
            return await tool_get_hn_story(svc, **args)
    finally:
        await svc.close()
```

### Pattern B: CLI Subprocess (Recommended for Terminal / Shell Agents)
Agents can execute CLI commands directly and parse stdout:
```bash
# Browse top stories as structured JSON
python3 -m gojo.cli list --feed top --limit 5 --json

# Search headlines matching query
python3 -m gojo.cli search "Kernel" --limit 20 --json

# Fetch single story metadata
python3 -m gojo.cli get 50354960 --json

# Generate AI summary with community discussions (auto-cached)
python3 -m gojo.cli summarize 50354960 --type quick --json

# Generate executive daily briefing
python3 -m gojo.cli briefing --feed top --limit 10 --locale en

# Inspect active LLM providers and available local models
python3 -m gojo.cli providers
```

---

## 3. Tool Specifications & Schemas

The following function definitions are exported via `gojo.agent_tools.HN_AGENT_TOOL_DEFINITIONS`:

### `hn_search`
Search or browse Hacker News headlines.
* **`feed_id`** (`string`, optional, default: `"hn_top"`): `"hn_top"` (front page) or `"hn_new"` (newest).
* **`page`** (`integer`, optional, default: `1`): 1-based page index.
* **`query`** (`string`, optional): Case-insensitive title substring filter.

### `hn_summarize`
Fetch full article and generate an intelligent, structured technical summary with anti-slop guidelines and peer community discussion synthesis.
* **`external_id`** (`string`, **required**): Hacker News story ID (e.g. `"50354960"`).
* **`summary_type`** (`string`, optional, default: `"quick"`):
  * `"quick"`: Substantive 2–5 paragraph technical brief with names, numbers, and mechanisms.
  * `"technical"`: Deep dive focusing on stack, APIs, CVEs, benchmarks, and architecture.
  * `"bullet"`: Dense, nested scannable bullet points matching source density.
  * `"thai_concise"`: Natural, high-density Thai language summary.
* **`locale`** (`string`, optional, default: `"en"`): Target language ISO code (e.g. `"en"`, `"th"`).
* **`include_comments`** (`boolean`, optional, default: `true`): Whether to fetch and synthesize peer comments from Algolia.

### `hn_briefing`
Synthesize top Hacker News developments into an authoritative Executive Technology Briefing with thematic clustering and macro-level takeaways.
* **`feed_id`** (`string`, optional, default: `"hn_top"`): `"hn_top"` or `"hn_new"`.
* **`limit`** (`integer`, optional, default: `10`): Number of stories to include.
* **`locale`** (`string`, optional, default: `"th"`): Language ISO code (`"th"` or `"en"`).

### `hn_get_story`
Fetch raw submission metadata (score, author, URL, comments count) for a Hacker News story.
* **`external_id`** (`string`, **required**): Hacker News story ID.

---

## 4. Architectural Invariants for Agents

When contributing to or modifying Gojo, agents **must strictly preserve** these architectural invariants:

1. **SSRF Guard Invariance (`gojo/ssrf.py`):**
   * Never bypass or weaken `is_public_http_url_for_fetch()`.
   * Outbound HTTP requests must never reach private IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.1`), link-local (`169.254.0.0/16`), or userinfo credentials in URLs.

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
     $$\text{Trafilatura} \longrightarrow \text{In-Memory Semantic HTML} \longrightarrow \text{Jina Reader} \longrightarrow \text{Title Fallback}$$
   * LLM provider resolution must gracefully probe:
     $$\text{OpenAI Key} \longrightarrow \text{Gemini Key} \longrightarrow \text{Local Ollama (localhost:11434)} \longrightarrow \text{Actionable Error}$$

---

## 5. Verification & Test Suite

Before committing or delivering any code changes, execute the automated test suite:

```bash
# Run complete test suite (must pass 100%)
python3 -m pytest tests -v
```

All 30 unit tests covering storage, SSRF security, prompt parsing, Algolia discussions, LLM providers, and briefing workflows must pass.
