from gojo.prompts import (
    build_digest_system_prompt,
    build_digest_user_message,
    effective_locale_for_summary,
    normalize_summary_type,
)


def test_normalize_summary_type() -> None:
    assert normalize_summary_type("QUICK") == "quick"
    assert normalize_summary_type("technical") == "technical"
    assert normalize_summary_type("bullet") == "bullet"
    assert normalize_summary_type("thai_concise") == "thai_concise"
    assert normalize_summary_type("unknown") == "quick"


def test_effective_locale_forced_thai_for_thai_concise() -> None:
    assert effective_locale_for_summary("en", "thai_concise") == "th"
    assert effective_locale_for_summary("fr", "quick") == "fr"


def test_build_digest_system_prompt_anti_slop_and_structure() -> None:
    p = build_digest_system_prompt(summary_type="quick", locale="en")
    assert "Anti-slop" in p
    assert "From the title alone" in p
    assert "## Summary" in p
    assert "### Why it matters" in p
    assert "### Key takeaway" in p
    assert "### Gojo note" in p


def test_build_digest_user_message_variants() -> None:
    msg_ok = build_digest_user_message("Sample content", source_fetch_status="ok")
    assert "full article text was retrieved" in msg_ok

    msg_blocked = build_digest_user_message("Sample content", source_fetch_status="blocked")
    assert "not fetched (policy)" in msg_blocked

    msg_error = build_digest_user_message("Sample content", source_fetch_status="error")
    assert "page fetch failed" in msg_error
