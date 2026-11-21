"""
Parse structured insight sections from generated summary markdown.
Extracts: why_matters, key_takeaway, note, community_consensus, alternative_solutions
across English and Thai headings.
"""

from __future__ import annotations

import re
from typing import Any

_BOUNDARY = r"(?=\n#{2,3}\s|\Z)"

_SECTION_PATTERNS = {
    "why_matters": re.compile(
        rf"#{{2,3}}\s*(?:Why it matters|ทำไมเรื่องนี้ถึงสำคัญ|ทำไมถึงสำคัญ|ความสำคัญ)\s*:?\s*\n([\s\S]*?){_BOUNDARY}",
        re.IGNORECASE,
    ),
    "key_takeaway": re.compile(
        rf"#{{2,3}}\s*(?:Key takeaways?|ประเด็นสำคัญ|ข้อสรุปสำคัญ|สรุปประเด็นสำคัญ)\s*:?\s*\n([\s\S]*?){_BOUNDARY}",
        re.IGNORECASE,
    ),
    "note": re.compile(
        rf"#{{2,3}}\s*(?:(?:Patze|Gojo)?\s*note|Gojo\s*(?:โน้ต|ข้อสังเกต|หมายเหตุ)|ข้อสังเกต(?:\s*Gojo)?|หมายเหตุ(?:\s*Gojo)?)\s*:?\s*\n([\s\S]*?){_BOUNDARY}",
        re.IGNORECASE,
    ),
    "community_consensus": re.compile(
        rf"#{{2,3}}\s*(?:Community consensus|Community reaction|Discussion|ความเห็นชุมชน|เสียงสะท้อนจากชุมชน|มุมมองชุมชน|การถกเถียงในชุมชน)\s*:?\s*\n([\s\S]*?){_BOUNDARY}",
        re.IGNORECASE,
    ),
    "alternative_solutions": re.compile(
        rf"#{{2,3}}\s*(?:Alternative solutions|Prior art|Alternatives|ทางเลือกอื่น|โซลูชันทางเลือก|เครื่องมือทางเลือก)\s*:?\s*\n([\s\S]*?){_BOUNDARY}",
        re.IGNORECASE,
    ),
}


def _strip_md_noise(s: str) -> str:
    t = s.strip()
    t = re.sub(r"\s+", " ", t)
    return t[:8000]


def extract_summary_extra_from_markdown(body: str) -> dict[str, Any]:
    """Extract structured interpretation fields from model markdown (EN & TH)."""
    text = body or ""
    extra: dict[str, Any] = {}
    for api_key, pat in _SECTION_PATTERNS.items():
        mm = pat.search(text)
        if mm:
            cleaned = _strip_md_noise(mm.group(1))
            if cleaned:
                extra[api_key] = cleaned
    return extra


extract_interpretation_fields = extract_summary_extra_from_markdown
