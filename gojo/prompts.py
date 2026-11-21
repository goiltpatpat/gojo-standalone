"""
Digest summarization prompts and anti-slop guidelines across multiple summary styles:
- quick: readable, substantive summary
- technical: deep stack, APIs, benchmarks, architectural decisions
- bullet: scannable bullet points matching source density
- thai_concise: concise and natural Thai summary
"""

from __future__ import annotations

from .types import SummaryType


def normalize_summary_type(raw: str) -> SummaryType:
    s = (raw or "quick").strip().lower()
    if s in ("quick", "technical", "bullet", "thai_concise"):
        return s  # type: ignore[return-value]
    return "quick"


def effective_locale_for_summary(locale: str, summary_type: SummaryType) -> str:
    if summary_type == "thai_concise":
        return "th"
    return locale or "en"


def build_digest_system_prompt(
    *,
    summary_type: SummaryType,
    locale: str,
    include_discussion: bool = False,
) -> str:
    el = effective_locale_for_summary(locale, summary_type)
    base_lang = f"Write in the language matching locale '{el}' (ISO-style)."

    discussion_sections = (
        "### Community consensus\n"
        "(Separate **Consensus** — what engineers agreed on or corroborated from real-world experience — "
        "from **Contrarian / Debunked** — claims challenged, caveats raised, or benchmarks disputed. "
        "Anchor key points with thread-path and handle citations like `[1] @handle` or `[1.2] @handle` when comments are provided.)\n"
        "### Alternative solutions\n"
        "(prior art, alternative tools, or libraries mentioned in discussion, anchored with `[path] @handle` if any)\n"
        if include_discussion
        else ""
    )

    structure = (
        "Output GitHub-flavored Markdown with this structure:\n"
        "## Summary\n"
        "(main narrative — follow the mode below; lead with the most useful facts)\n"
        f"{discussion_sections}"
        "### Why it matters\n"
        "(2–4 sentences when the source is rich; 1–2 when thin; stay specific to this story)\n"
        "### Key takeaway\n"
        "(one or two tight lines when the story warrants nuance)\n"
        "### Gojo note\n"
        "(one line: practical angle or caveat for builders; concrete, no hype)\n"
    )

    voice = (
        "Voice: confident editor briefing a colleague—concrete nouns, varied sentence length, no lecture tone.\n"
        "Anti-slop: do not use these (or close variants): "
        "\"From the title alone\", \"I only have the title\", \"It is important to note\", "
        "\"In today's world\", \"In conclusion\", "
        "\"delve\", \"landscape\" (figurative), \"leverage\", \"robust\", \"synergy\", \"game-changer\", "
        "\"at the end of the day\", \"Let's unpack\".\n"
        "When the input is thin, state limits in one direct sentence—no performative hedging or filler lists.\n"
        "Do not mention AI, models, or prompts. No meta disclaimers about \"lack of data\" unless the source truly "
        "offers nothing beyond a title; then one honest line only.\n"
    )

    length_policy = (
        "Length policy: scale to the source. Long or detailed article text → long Summary (often "
        "roughly 350–900 words in the Summary section alone when substance supports it—multiple paragraphs, "
        "concrete detail). Medium article → several paragraphs. Do not compress a rich article into a few "
        "short sentences; a reader who skips the link should still understand what happened and why it matters. "
        "When the input is only a title or a few lines, stay short and honest.\n"
    )

    mode_lines: dict[SummaryType, str] = {
        "quick": (
            "Summary mode: **Quick** — still means readable, not tiny. When the body is available, deliver a "
            "substantive Summary: often 10–28 sentences across 2–5 paragraphs for rich reporting, with "
            "names, numbers, mechanisms, and timeline. Prefer completeness of insight over brevity when the "
            "source is long. When the source is thin, keep it short and factual."
        ),
        "technical": (
            "Summary mode: **Technical** — go deep where the source supports it: stack, APIs, versions, CVE IDs, "
            "benchmarks, migration or rollout notes when present. Use as many paragraphs as needed; do not stop "
            "early while important details remain in the source. Keep subsections scannable."
        ),
        "bullet": (
            "Summary mode: **Bullet** — use bullets in the Summary section; sub-bullets allowed for nested facts; "
            "use many bullets when the article is long or dense—match list length to source density."
        ),
        "thai_concise": (
            "Summary mode: **Thai concise** — ภาษาไทยกระชับแต่ครบความ ตามความยาวของต้นฉบับ "
            "แบ่งย่อหน้าตามโครงสร้างด้านบนอย่างชัดเจนและสละสลวย"
        ),
    }

    return (
        "You summarize technical articles for a busy reader.\n"
        f"{base_lang}\n"
        f"{voice}\n"
        f"{length_policy}\n"
        f"{mode_lines[summary_type]}\n"
        f"{structure}\n"
        "- Anchor claims in the provided text; prefer quoted terms and numbers from the source when present.\n"
        "- Do not pad with filler; match depth to evidence in the source.\n"
        "- Keep Gojo note grounded; no AI self-reference."
    )


def build_digest_user_message(
    content: str,
    *,
    source_fetch_status: str,
    discussion_context: str | None = None,
) -> str:
    st = (source_fetch_status or "fallback_title").strip().lower()
    if st == "ok":
        n = len(content)
        depth = (
            "Write at proportional length: substantial article text below → **long, detailed** Summary "
            "(see system length policy); do not under-write.\n"
            if n > 2500
            else (
                "Aim for several full paragraphs in Summary when the text below has enough substance.\n"
                if n > 900
                else ""
            )
        )
        preamble = (
            "Source: full article text was retrieved below.\n"
            "Ground the Summary in the Content section; use the title only for orientation.\n"
            f"{depth}"
        )
    elif st == "blocked":
        preamble = (
            "Source: external URL was not fetched (policy). Work only from the title and any text below—"
            "do not invent reporting.\n"
        )
    elif st == "error":
        preamble = (
            "Source: page fetch failed; you may only have the title line below. "
            "Be honest about limits in one sentence; do not fabricate article details.\n"
        )
    else:
        preamble = (
            "Source: thin input (e.g. title-only). Summarize what can be responsibly inferred; "
            "state gaps plainly without stock phrases.\n"
        )

    body = f"{preamble}\n\n{content}"
    if discussion_context and discussion_context.strip():
        body += f"\n\n{discussion_context.strip()}"
    return body


def build_briefing_system_prompt(*, locale: str = "en") -> str:
    """System prompt for multi-story executive technical briefings."""
    base_lang = f"Write in the language matching locale '{locale}' (ISO-style, e.g. 'th' for Thai, 'en' for English)."
    return (
        "You are an executive technology editor delivering an authoritative, high-density briefing on top tech developments.\n"
        f"{base_lang}\n"
        "Voice: confident editor briefing technical leaders—concrete facts, numbers, mechanisms, and cross-cutting implications.\n"
        "Anti-slop: banned terms: \"delve\", \"landscape\", \"robust\", \"leverage\", \"synergy\", \"game-changer\", \"In today's world\".\n\n"
        "Output GitHub-flavored Markdown with this exact structure:\n"
        "# Executive Technology Briefing\n\n"
        "## Key Themes & Macro Shifts\n"
        "(2-3 paragraphs synthesizing the overarching narratives across these stories, highlighting security boundaries, AI developments, or infrastructure realities)\n\n"
        "## Thematic Clusters\n"
        "(Group the stories into 3-4 thematic categories like: Security & Privacy, AI & Models, Systems & Embedded, Web & Platforms)\n"
        "For each category:\n"
        "### [Category Name]\n"
        "- **[Story Title]** (Score: X, Comments: Y)\n"
        "  - Core fact & mechanism (1-2 tight sentences)\n"
        "  - Industry impact or engineering takeaway\n\n"
        "## Actionable Takeaways for Builders\n"
        "(3-5 bullet points of concrete advice, cautionary lessons, or architectural patterns engineers should adopt today)"
    )

