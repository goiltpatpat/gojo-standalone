"""
Parse structured insight sections from generated summary markdown.
Extracts: why_matters, key_takeaway, note.
"""

from __future__ import annotations

import re
from typing import Any

_SECTION_PATTERNS = {
    "why_matters": re.compile(
        r"###\s*Why it matters\s*\n([\s\S]*?)(?=\n###|\n##\s|\Z)", re.IGNORECASE
    ),
    "key_takeaway": re.compile(
        r"###\s*Key takeaway\s*\n([\s\S]*?)(?=\n###|\n##\s|\Z)", re.IGNORECASE
    ),
    "note": re.compile(
        r"###\s*(?:Patze|Gojo)?\s*note\s*\n([\s\S]*?)(?=\n###|\n##\s|\Z)", re.IGNORECASE
    ),
}


def _strip_md_noise(s: str) -> str:
    t = s.strip()
    t = re.sub(r"\s+", " ", t)
    return t[:8000]


def extract_summary_extra_from_markdown(body: str) -> dict[str, Any]:
    """Extract structured interpretation fields from model markdown."""
    text = body or ""
    extra: dict[str, Any] = {}
    for api_key, pat in _SECTION_PATTERNS.items():
        mm = pat.search(text)
        if mm:
            extra[api_key] = _strip_md_noise(mm.group(1))
    return extra
