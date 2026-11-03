from gojo.interpretation import extract_summary_extra_from_markdown


def test_extract_summary_extra_success() -> None:
    markdown = """
## Summary
This is the main summary.

### Why it matters
It represents a major breakthrough in distributed consensus protocols.

### Key takeaway
Latency is reduced by 40% with zero loss in safety.

### Gojo note
Ensure your node configuration has adequate memory before migrating.
"""
    extra = extract_summary_extra_from_markdown(markdown)
    assert "why_matters" in extra
    assert "breakthrough" in extra["why_matters"]
    assert "key_takeaway" in extra
    assert "Latency is reduced" in extra["key_takeaway"]
    assert "note" in extra
    assert "Ensure your node" in extra["note"]


def test_extract_summary_extra_missing_sections() -> None:
    markdown = "Just a raw summary without headers."
    extra = extract_summary_extra_from_markdown(markdown)
    assert extra == {}
