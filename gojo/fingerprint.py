"""
Stable fingerprint generator for idempotent summary deduplication and cache lookup.
Matches the v2 fingerprinting standard from the reference architecture.
"""

from __future__ import annotations

import hashlib


def digest_fingerprint(
    *,
    feed_provider: str,
    external_id: str,
    locale: str,
    prompt_version: str,
    summary_type: str = "quick",
) -> str:
    """Generate SHA256 deterministic key for story summary identity."""
    st = (summary_type or "quick").strip().lower()
    raw = f"{feed_provider}:{external_id}:{locale.lower().strip()}:{prompt_version}:{st}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
