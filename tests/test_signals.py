from datetime import datetime, timezone

import pytest

from gojo.signals import (
    canonicalize_url,
    cluster_related_stories,
    compute_story_signals,
    enrich_items_with_signals,
    hamming_distance,
    rank_items_by_focus,
    simhash64,
)
from gojo.types import DigestFeedItem, FeedItem


def test_feed_item_alias_and_default_signal_fields() -> None:
    assert FeedItem is DigestFeedItem
    item = DigestFeedItem(external_id="1", title="Example")
    assert item.velocity is None
    assert item.debate_ratio is None
    assert item.signal_badges == []


def test_compute_story_signals_breakout_and_high_debate() -> None:
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    one_hour_ago = int(now.timestamp() - 3600)

    item = DigestFeedItem(
        external_id="100",
        title="Major Kernel Release",
        score=200,
        descendants=180,
        time=one_hour_ago,
    )
    sig = compute_story_signals(item, now=now)

    expected_velocity = (200 - 1) / ((1.0 + 2.0) ** 1.8)
    assert sig["velocity"] == pytest.approx(expected_velocity, rel=1e-4)
    assert sig["debate_ratio"] == pytest.approx(180 / 200, rel=1e-4)
    assert "breakout" in sig["signal_badges"]
    assert "high-debate" in sig["signal_badges"]
    assert "rising" not in sig["signal_badges"]
    assert "slow-burn" not in sig["signal_badges"]


def test_compute_story_signals_rising_and_slow_burn() -> None:
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

    rising_item = DigestFeedItem(
        external_id="101",
        title="Fast climbing Show HN",
        score=60,
        descendants=12,
        time=int(now.timestamp() - 3600),
    )
    rising_sig = compute_story_signals(rising_item, now=now)
    assert "rising" in rising_sig["signal_badges"]
    assert "breakout" not in rising_sig["signal_badges"]

    slow_burn_item = DigestFeedItem(
        external_id="102",
        title="Deep dive posted yesterday",
        score=240,
        descendants=90,
        time=int(now.timestamp() - 18 * 3600),
    )
    slow_sig = compute_story_signals(slow_burn_item, now=now)
    assert "slow-burn" in slow_sig["signal_badges"]
    assert "breakout" not in slow_sig["signal_badges"]


def test_enrich_items_with_signals_preserves_originals() -> None:
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    orig = DigestFeedItem(
        external_id="103",
        title="Test Story",
        score=120,
        descendants=100,
        time=int(now.timestamp() - 1800),
    )
    enriched = enrich_items_with_signals([orig], now=now)
    assert len(enriched) == 1
    assert orig.velocity is None
    assert enriched[0].velocity is not None and enriched[0].velocity > 15.0
    assert enriched[0].debate_ratio == pytest.approx(100 / 120, rel=1e-4)
    assert "breakout" in enriched[0].signal_badges
    assert "high-debate" in enriched[0].signal_badges


def test_canonicalize_url_strips_tracking_fragments_ports_and_slashes() -> None:
    u1 = "HTTPS://Example.COM:443/blog/post/?utm_source=hn&utm_medium=social&id=42&ref=frontpage#comments"
    u2 = "https://example.com/blog/post?id=42"
    assert canonicalize_url(u1) == u2
    assert canonicalize_url("http://example.com:80/") == "http://example.com"
    assert canonicalize_url("https://example.com/a?b=2&a=1&fbclid=xyz") == "https://example.com/a?a=1&b=2"
    assert canonicalize_url("") == ""


def test_simhash64_and_hamming_distance() -> None:
    h_same_1 = simhash64("Rust 1.86 released with trait upcasting support")
    h_same_2 = simhash64("Rust 1.86 released with trait upcasting support")
    assert h_same_1 == h_same_2
    assert hamming_distance(h_same_1, h_same_2) == 0

    h_near = simhash64("Rust 1.86 released with full trait upcasting support")
    h_far = simhash64("Sourdough bread fermentation temperature guide for beginners")

    dist_near = hamming_distance(h_same_1, h_near)
    dist_far = hamming_distance(h_same_1, h_far)
    assert dist_near < dist_far
    assert dist_near <= 12
    assert simhash64("") == 0


def test_cluster_related_stories_url_and_title_similarity() -> None:
    items = [
        DigestFeedItem(
            external_id="1",
            title="SQLite 3.50 Released with Concurrent Writes",
            url="https://sqlite.org/release-3-50/?utm_source=hn",
        ),
        DigestFeedItem(
            external_id="2",
            title="Linux 6.15 Kernel Scheduler Benchmarks",
            url="https://lwn.net/Articles/999999/",
        ),
        DigestFeedItem(
            external_id="3",
            title="SQLite 3.50 Announcement",
            url="https://SQLITE.org/release-3-50#top",
        ),
        DigestFeedItem(
            external_id="4",
            title="Benchmarks for the Linux 6.15 Kernel Scheduler",
            url="https://phoronix.com/news/linux-615-scheduler",
        ),
        DigestFeedItem(
            external_id="5",
            title="Show HN: Minimalist Vector Graphics Editor in Zig",
            url="https://example.org/zig-svg",
        ),
    ]

    clusters = cluster_related_stories(items)
    assert len(clusters) == 3
    # Rank order preserved across clusters and within clusters
    assert [i.external_id for i in clusters[0]] == ["1", "3"]
    assert [i.external_id for i in clusters[1]] == ["2", "4"]
    assert [i.external_id for i in clusters[2]] == ["5"]


def test_rank_items_by_focus_bm25() -> None:
    items = [
        DigestFeedItem(
            external_id="1",
            title="Building a Frontend CSS Framework",
            url="https://css.example.com/grid",
        ),
        DigestFeedItem(
            external_id="2",
            title="Writing an Async Rust Kernel Driver for NVMe",
            url="https://rust-osdev.org/nvme-kernel",
        ),
        DigestFeedItem(
            external_id="3",
            title="Rust Compiler Diagnostics Improvements",
            url="https://blog.rust-lang.org/diagnostics",
        ),
        DigestFeedItem(
            external_id="4",
            title="PostgreSQL Vacuum Internals",
            url="https://postgres.example.com/vacuum",
        ),
    ]

    ranked = rank_items_by_focus(items, "rust, kernel, nvme")
    assert ranked[0][0].external_id == "2"
    assert ranked[0][1] > ranked[1][1]
    assert ranked[1][0].external_id == "3"
    assert ranked[1][1] > 0.0
    # Non-matching items get 0.0 and keep relative order
    assert ranked[2][1] == 0.0
    assert ranked[3][1] == 0.0
    assert [ranked[2][0].external_id, ranked[3][0].external_id] == ["1", "4"]
