from datetime import datetime, timezone

import pytest

from gojo.signals import (
    annotate_cross_source_clusters,
    build_adaptive_focus_profile,
    canonicalize_url,
    cluster_related_stories,
    compute_story_signals,
    enrich_items_with_signals,
    hamming_distance,
    profile_to_focus_query,
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


def test_build_adaptive_focus_profile_empty_and_limits() -> None:
    assert build_adaptive_focus_profile([]) == []
    assert build_adaptive_focus_profile([{"title": "", "key_takeaway": ""}]) == []
    assert build_adaptive_focus_profile([{"title": "Rust eBPF", "key_takeaway": "SQLite"}], top_terms=0) == []


def test_build_adaptive_focus_profile_title_weighting_and_stopwords() -> None:
    corpus = [
        {
            "title": "A New C++ and Go eBPF Runtime for AI DB Workloads",
            "key_takeaway": "The article summary says Rust LLM SQLite 2026 x y z",
            "source_url": "https://example.com/ebpf-runtime",
        }
    ]
    profile = build_adaptive_focus_profile(corpus, top_terms=20, decay=0.92)
    weights = dict(profile)

    # Preserved short and special tech tokens
    for tech_token in ("c++", "go", "ai", "db", "ebpf", "rust", "llm", "sqlite"):
        assert tech_token in weights

    # Filtered stopwords, boilerplate, single-char tokens, and pure numbers
    for filtered in ("a", "new", "and", "for", "the", "article", "summary", "says", "2026", "x", "y", "z"):
        assert filtered not in weights

    # Title tokens receive 2.0x weight compared to key_takeaway tokens (1.0x)
    assert weights["ebpf"] == pytest.approx(2.0 * weights["sqlite"], rel=1e-5)
    assert weights["c++"] == pytest.approx(2.0 * weights["rust"], rel=1e-5)


def test_build_adaptive_focus_profile_recency_decay_and_idf() -> None:
    # Newest-first corpus: index 0 has decay**0 = 1.0, index 2 has decay**2
    corpus = [
        {"title": "eBPF tracing", "key_takeaway": "kernel telemetry"},
        {"title": "SQLite WAL", "key_takeaway": "kernel storage"},
        {"title": "Zig allocator", "key_takeaway": "kernel memory"},
    ]
    decay = 0.8
    profile = build_adaptive_focus_profile(corpus, top_terms=10, decay=decay)
    weights = dict(profile)

    # All three unique title tokens have df=1, so recency decay strictly orders them
    assert weights["ebpf"] > weights["sqlite"] > weights["zig"]
    assert weights["sqlite"] == pytest.approx(weights["ebpf"] * decay, rel=1e-5)
    assert weights["zig"] == pytest.approx(weights["ebpf"] * (decay**2), rel=1e-5)

    # Returned list is sorted descending by weight and respects top_terms
    top3 = build_adaptive_focus_profile(corpus, top_terms=3, decay=decay)
    assert len(top3) == 3
    assert [w for _, w in top3] == sorted((w for _, w in top3), reverse=True)


def test_profile_to_focus_query_and_bm25_integration() -> None:
    assert profile_to_focus_query([]) == ""
    assert profile_to_focus_query([("rust", 5.0), ("ebpf", 3.0)], max_terms=0) == ""

    corpus = [
        {"title": "Rust Async Kernel Driver", "key_takeaway": "Zero-copy NVMe IO in Rust"},
        {"title": "eBPF Network Filtering in Rust", "key_takeaway": "XDP packet processing"},
    ]
    profile = build_adaptive_focus_profile(corpus, top_terms=10)
    query = profile_to_focus_query(profile, max_terms=5)
    tokens = query.split()
    assert 1 <= len(tokens) <= 5
    assert "rust" in tokens

    candidates = [
        DigestFeedItem(external_id="1", title="CSS Grid Layout Tips"),
        DigestFeedItem(external_id="2", title="Writing an eBPF Packet Filter in Rust"),
        DigestFeedItem(external_id="3", title="Baking Sourdough Bread"),
    ]
    ranked = rank_items_by_focus(candidates, query)
    assert ranked[0][0].external_id == "2"
    assert ranked[0][1] > 0.0


def test_annotate_cross_source_clusters_detection_badges_and_ranking() -> None:
    assert annotate_cross_source_clusters([]) == []
    assert annotate_cross_source_clusters([[]]) == []

    # 3-source corroborated cluster (hn + lobsters + github) with moderate velocity
    tri_hn = DigestFeedItem(
        external_id="10",
        feed_provider="hn",
        title="SQLite 3.51 Concurrent WAL Mode",
        url="https://sqlite.org/wal-351",
        velocity=12.5,
        signal_badges=["rising"],
    )
    tri_lobsters = DigestFeedItem(
        external_id="lob-10",
        feed_provider="lobsters",
        title="SQLite 3.51 Concurrent WAL Mode",
        url="https://sqlite.org/wal-351?utm_source=lobsters",
        velocity=8.0,
        signal_badges=["corroborated"],  # already present -> must not duplicate
    )
    tri_github = DigestFeedItem(
        external_id="gh-10",
        feed_provider="github",
        title="SQLite 3.51 Concurrent WAL Mode Release",
        url="https://sqlite.org/wal-351#readme",
        velocity=6.0,
        signal_badges=[],
    )

    # 2-source corroborated cluster (hn + lobsters) with higher velocity
    dual_hn = DigestFeedItem(
        external_id="20",
        feed_provider="hn",
        title="Linux 6.16 eBPF Verifier Rewrite",
        url="https://lwn.net/Articles/616000/",
        velocity=25.0,
        signal_badges=["breakout"],
    )
    dual_lobsters = DigestFeedItem(
        external_id="lob-20",
        feed_provider="lobsters",
        title="Linux 6.16 eBPF Verifier Rewrite",
        url="https://lwn.net/Articles/616000/",
        velocity=10.0,
        signal_badges=[],
    )

    # Single-source high-velocity cluster (HN only, 2 items -> still not corroborated)
    single_hi_1 = DigestFeedItem(
        external_id="30",
        feed_provider="hn",
        title="Show HN: Tiny Zig HTTP Server",
        url="https://example.org/zig-http",
        velocity=40.0,
        signal_badges=["breakout"],
    )
    single_hi_2 = DigestFeedItem(
        external_id="31",
        feed_provider="hn",
        title="Tiny Zig HTTP Server Benchmarks",
        url="https://example.org/zig-http?ref=hn",
        velocity=5.0,
        signal_badges=[],
    )

    # Single-source low-velocity cluster
    single_lo = DigestFeedItem(
        external_id="40",
        feed_provider="hn",
        title="Old Retro Computing Post",
        url="https://example.org/retro",
        velocity=2.0,
        signal_badges=[],
    )

    # Cluster via cluster_related_stories() first to verify full pipeline
    clusters = cluster_related_stories(
        [single_lo, single_hi_1, single_hi_2, dual_hn, dual_lobsters, tri_hn, tri_lobsters, tri_github]
    )
    annotated = annotate_cross_source_clusters(clusters)

    assert len(annotated) == 4

    # 1st: 3-source corroborated cluster
    first = annotated[0]
    assert first["is_corroborated"] is True
    assert first["sources"] == ["github", "hn", "lobsters"]
    assert first["corroboration_count"] == 3
    assert first["cluster_size"] == 3
    assert first["max_velocity"] == pytest.approx(12.5)
    assert first["primary"].external_id == "10"
    for item in first["items"]:
        assert "corroborated" in item.signal_badges
        assert item.signal_badges.count("corroborated") == 1

    # 2nd: 2-source corroborated cluster
    second = annotated[1]
    assert second["is_corroborated"] is True
    assert second["sources"] == ["hn", "lobsters"]
    assert second["corroboration_count"] == 2
    assert second["cluster_size"] == 2
    assert second["max_velocity"] == pytest.approx(25.0)
    assert "corroborated" in second["primary"].signal_badges

    # 3rd: Single-source high-velocity cluster (beats low-velocity single-source)
    third = annotated[2]
    assert third["is_corroborated"] is False
    assert third["sources"] == ["hn"]
    assert third["corroboration_count"] == 1
    assert third["cluster_size"] == 2
    assert third["max_velocity"] == pytest.approx(40.0)
    for item in third["items"]:
        assert "corroborated" not in item.signal_badges

    # 4th: Single-source low-velocity cluster
    fourth = annotated[3]
    assert fourth["is_corroborated"] is False
    assert fourth["max_velocity"] == pytest.approx(2.0)

    # Original caller objects were NOT mutated
    assert "corroborated" not in tri_hn.signal_badges
    assert "corroborated" not in tri_github.signal_badges
    assert "corroborated" not in dual_hn.signal_badges
