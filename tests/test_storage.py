import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from gojo.storage.memory import MemoryStorage
from gojo.storage.sqlite import SQLiteStorage
from gojo.types import DigestArtifact, DigestFeedItem


@pytest.fixture
def sample_artifact() -> DigestArtifact:
    return DigestArtifact(
        fingerprint="test_fp_1234567890abcdef" * 2,
        feed_provider="hn",
        external_id="999",
        title="Test Headline",
        source_url="https://example.com/test",
        locale="en",
        prompt_version="v5",
        summary_markdown="## Summary\nGreat news.",
        summary_type="quick",
        summary_extra={"why_matters": "Very important"},
    )


@pytest.fixture
def tmp_db_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ("", "-wal", "-shm"):
        candidate = f"{path}{suffix}"
        if os.path.exists(candidate):
            os.remove(candidate)


@pytest.mark.asyncio
async def test_memory_storage(sample_artifact: DigestArtifact) -> None:
    storage = MemoryStorage()
    assert storage.layer_name == "memory"

    saved = await storage.save_artifact(sample_artifact)
    assert saved.id == sample_artifact.id

    fetched = await storage.get_artifact_by_fingerprint(sample_artifact.fingerprint)
    assert fetched is not None
    assert fetched.title == "Test Headline"

    by_id = await storage.get_artifact_by_id(sample_artifact.id)
    assert by_id is not None
    assert by_id.fingerprint == sample_artifact.fingerprint

    # Feed cache (default page_size=15 backward compatibility)
    await storage.set_feed_cache("hn_top", 1, {"items": [1, 2, 3]}, ttl_seconds=10)
    cached = await storage.get_feed_cache("hn_top", 1)
    assert cached == {"items": [1, 2, 3]}


@pytest.mark.asyncio
async def test_sqlite_storage(sample_artifact: DigestArtifact, tmp_db_path: str) -> None:
    storage = SQLiteStorage(tmp_db_path)
    assert storage.layer_name == "sqlite"

    await storage.save_artifact(sample_artifact)

    fetched = await storage.get_artifact_by_fingerprint(sample_artifact.fingerprint)
    assert fetched is not None
    assert fetched.title == "Test Headline"
    assert fetched.summary_extra == {"why_matters": "Very important"}

    by_id = await storage.get_artifact_by_id(sample_artifact.id)
    assert by_id is not None
    assert by_id.title == "Test Headline"

    recent = await storage.list_recent_artifacts(limit=5)
    assert len(recent) == 1
    assert recent[0].external_id == "999"

    # Feed cache (default page_size=15 backward compatibility)
    await storage.set_feed_cache("hn_top", 1, {"items": ["a", "b"]}, ttl_seconds=60)
    cache_data = await storage.get_feed_cache("hn_top", 1)
    assert cache_data == {"items": ["a", "b"]}


@pytest.mark.asyncio
async def test_sqlite_wal_and_pragmas(tmp_db_path: str) -> None:
    storage = SQLiteStorage(tmp_db_path)
    conn = storage._connect()
    try:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert str(journal_mode).lower() == "wal"
        assert int(synchronous) == 1  # NORMAL
        assert int(busy_timeout) == 5000
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_sqlite_upsert_refreshes_metadata(tmp_db_path: str) -> None:
    storage = SQLiteStorage(tmp_db_path)
    fp = "upsert_fp_" + ("a" * 54)
    t1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)

    first = DigestArtifact(
        fingerprint=fp,
        external_id="1001",
        title="Original Kernel Story",
        source_url="https://example.com/v1",
        summary_markdown="Initial summary about legacy scheduler.",
        model="gpt-4o-mini",
        content_length=120,
        generated_ms=500,
        source_fetch_status="title_only",
        model_version="v1",
        created_at=t1,
    )
    s1 = await storage.save_artifact(first)

    updated = DigestArtifact(
        fingerprint=fp,
        external_id="1001",
        title="Updated Kernel Story",
        source_url="https://example.com/v2",
        summary_markdown="Regenerated deep dive on io_uring zero-copy.",
        model="gemini-2.5-flash",
        content_length=4200,
        generated_ms=950,
        source_fetch_status="ok",
        model_version="v2",
        created_at=t2,
    )
    assert updated.id != s1.id
    s2 = await storage.save_artifact(updated)
    assert s2.id == s1.id

    by_id = await storage.get_artifact_by_id(s2.id)
    assert by_id is not None
    assert by_id.id == s1.id
    assert by_id.summary_markdown == "Regenerated deep dive on io_uring zero-copy."

    fetched = await storage.get_artifact_by_fingerprint(fp)
    assert fetched is not None
    assert fetched.summary_markdown == "Regenerated deep dive on io_uring zero-copy."
    assert fetched.model == "gemini-2.5-flash"
    assert fetched.content_length == 4200
    assert fetched.generated_ms == 950
    assert fetched.source_fetch_status == "ok"
    assert fetched.model_version == "v2"
    assert fetched.created_at == t2

    # FTS5 AFTER UPDATE trigger should index the new text and drop the old text
    hits_new = await storage.search_artifacts("io_uring")
    assert len(hits_new) == 1
    assert hits_new[0].fingerprint == fp
    hits_old = await storage.search_artifacts("legacy")
    assert hits_old == []


@pytest.mark.asyncio
async def test_feed_cache_page_size_isolation_and_prune(tmp_db_path: str) -> None:
    for storage in (MemoryStorage(), SQLiteStorage(tmp_db_path)):
        await storage.set_feed_cache("hn_top", 1, {"size": 5}, ttl_seconds=60, page_size=5)
        await storage.set_feed_cache("hn_top", 1, {"size": 30}, ttl_seconds=60, page_size=30)
        await storage.set_feed_cache("hn_top", 1, {"size": 15}, ttl_seconds=60)

        assert await storage.get_feed_cache("hn_top", 1, page_size=5) == {"size": 5}
        assert await storage.get_feed_cache("hn_top", 1, page_size=30) == {"size": 30}
        assert await storage.get_feed_cache("hn_top", 1) == {"size": 15}
        assert await storage.get_feed_cache("hn_top", 1, page_size=10) is None

        # Expired entries are pruned while active ones remain
        await storage.set_feed_cache("hn_new", 1, {"stale": True}, ttl_seconds=-10, page_size=5)
        await storage.set_feed_cache("hn_new", 2, {"stale": True}, ttl_seconds=-1, page_size=15)
        pruned = await storage.prune_expired_feed_cache()
        assert pruned == 2
        assert await storage.get_feed_cache("hn_top", 1, page_size=5) == {"size": 5}


@pytest.mark.asyncio
async def test_memory_list_recent_artifacts_newest_first() -> None:
    storage = MemoryStorage()
    for idx in range(1, 4):
        await storage.save_artifact(
            DigestArtifact(
                fingerprint=f"fp_{idx}_" + ("0" * 58),
                external_id=str(idx),
                title=f"Story {idx}",
                summary_markdown=f"Summary {idx}",
            )
        )

    recent = await storage.list_recent_artifacts(limit=2)
    assert [a.external_id for a in recent] == ["3", "2"]
    assert await storage.list_recent_artifacts(limit=0) == []


@pytest.mark.asyncio
async def test_sqlite_fts5_search_special_chars_and_backfill(tmp_db_path: str) -> None:
    # Seed a legacy DB with hn_artifacts BEFORE FTS5 exists to verify backfill
    conn = sqlite3.connect(tmp_db_path)
    try:
        conn.execute(
            """
            CREATE TABLE hn_artifacts (
                id TEXT PRIMARY KEY,
                fingerprint TEXT UNIQUE NOT NULL,
                fingerprint_version TEXT DEFAULT 'v2',
                feed_provider TEXT NOT NULL,
                external_id TEXT NOT NULL,
                title TEXT,
                source_url TEXT,
                locale TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model TEXT,
                summary_markdown TEXT NOT NULL,
                summary_type TEXT DEFAULT 'quick',
                summary_extra TEXT,
                content_length INTEGER,
                generated_ms INTEGER,
                source_fetch_status TEXT,
                model_version TEXT,
                created_at TEXT,
                created_by_user_id TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO hn_artifacts (
                id, fingerprint, feed_provider, external_id, title, source_url,
                locale, prompt_version, summary_markdown
            ) VALUES (?, ?, 'hn', '500', 'Preexisting Rust-Lang Async Runtime',
                      'https://rust-lang.org/blog/async', 'en', 'v5',
                      'Detailed benchmark of tokio work-stealing scheduler.');
            """,
            ("legacy-id-1", "legacy_fp_" + ("b" * 54)),
        )
        conn.commit()
    finally:
        conn.close()

    storage = SQLiteStorage(tmp_db_path)

    # Backfilled row is immediately searchable
    backfilled = await storage.search_artifacts("tokio scheduler")
    assert len(backfilled) == 1
    assert backfilled[0].external_id == "500"

    # Add more artifacts via normal async save_artifact
    await storage.save_artifact(
        DigestArtifact(
            fingerprint="fts_fp_1_" + ("c" * 55),
            external_id="501",
            title="SQLite FTS5 BM25 Ranking Guide",
            source_url="https://sqlite.org/fts5.html",
            summary_markdown="SQLite FTS5 enables fast full-text search with BM25 ranking.",
        )
    )
    await storage.save_artifact(
        DigestArtifact(
            fingerprint="fts_fp_2_" + ("d" * 55),
            external_id="502",
            title="PostgreSQL Indexing Internals",
            source_url="https://postgresql.org/docs/indexes",
            summary_markdown="Brief mention of sqlite in comparison to Postgres GIN.",
        )
    )

    # Relevance ranking: FTS5 doc mentioning SQLite multiple times in title & body ranks first
    hits = await storage.search_artifacts("sqlite")
    assert [h.external_id for h in hits] == ["501", "502"]

    # Special FTS5 characters must be handled gracefully without OperationalError
    hyphen_hits = await storage.search_artifacts("rust-lang")
    assert len(hyphen_hits) == 1
    assert hyphen_hits[0].external_id == "500"

    for tricky_query in [
        "",
        "   ",
        "***",
        "---",
        '""',
        "foo:bar",
        '"unbalanced quote',
        "NOT OR AND",
        "title:sqlite -postgres * (test)",
        "https://sqlite.org/fts5.html",
    ]:
        res = await storage.search_artifacts(tricky_query)
        assert isinstance(res, list)

    # Verify AFTER DELETE trigger keeps FTS5 consistent
    conn = storage._connect()
    try:
        with conn:
            conn.execute("DELETE FROM hn_artifacts WHERE external_id = '501';")
    finally:
        conn.close()
    assert [h.external_id for h in await storage.search_artifacts("BM25")] == []


@pytest.mark.asyncio
async def test_memory_search_artifacts() -> None:
    storage = MemoryStorage()
    await storage.save_artifact(
        DigestArtifact(
            fingerprint="mem_s1_" + ("e" * 57),
            external_id="601",
            title="Postgres Query Planner",
            source_url="https://example.com/pg",
            summary_markdown="Mentions rust bindings briefly.",
        )
    )
    await storage.save_artifact(
        DigestArtifact(
            fingerprint="mem_s2_" + ("f" * 57),
            external_id="602",
            title="Rust Compiler Performance",
            source_url="https://rust-lang.org/perf",
            summary_markdown="Rust LLVM backend optimizations for faster builds.",
        )
    )

    hits = await storage.search_artifacts("rust")
    assert [h.external_id for h in hits] == ["602", "601"]
    assert [h.external_id for h in await storage.search_artifacts("rust llvm")] == ["602"]
    assert await storage.search_artifacts("") == []
    assert await storage.search_artifacts("---***") == []


@pytest.mark.asyncio
async def test_story_snapshots(tmp_db_path: str) -> None:
    for storage in (MemoryStorage(), SQLiteStorage(tmp_db_path)):
        assert await storage.record_story_snapshots("hn_top", []) == 0
        assert await storage.get_latest_snapshots([]) == {}

        batch1 = [
            DigestFeedItem(external_id="101", title="Alpha", score=50, descendants=10),
            DigestFeedItem(external_id="102", title="Beta", score=30, descendants=5),
        ]
        count1 = await storage.record_story_snapshots("hn_top", batch1)
        assert count1 == 2

        batch2 = [
            DigestFeedItem(external_id="102", title="Beta", score=120, descendants=42),
            DigestFeedItem(external_id="103", title="Gamma", score=80, descendants=15),
        ]
        count2 = await storage.record_story_snapshots("hn_top", batch2)
        assert count2 == 2

        latest = await storage.get_latest_snapshots(["101", "102", "103", "999"])
        assert set(latest.keys()) == {"101", "102", "103"}
        assert latest["101"]["rank"] == 1
        assert latest["101"]["score"] == 50
        assert latest["101"]["descendants"] == 10
        assert latest["101"]["seen_at"]

        # Story 102 should reflect batch2 (rank 1, score 120, descendants 42)
        assert latest["102"]["rank"] == 1
        assert latest["102"]["score"] == 120
        assert latest["102"]["descendants"] == 42

        assert latest["103"]["rank"] == 2
        assert latest["103"]["score"] == 80
        assert latest["103"]["descendants"] == 15


@pytest.mark.asyncio
async def test_sqlite_legacy_snapshots_schema_migration(tmp_db_path: str) -> None:
    # Seed a legacy DB where hn_story_snapshots lacks title & url columns
    conn = sqlite3.connect(tmp_db_path)
    try:
        conn.execute(
            """
            CREATE TABLE hn_story_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT NOT NULL,
                feed_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                score INTEGER,
                descendants INTEGER,
                seen_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO hn_story_snapshots (
                external_id, feed_id, rank, score, descendants, seen_at
            ) VALUES ('7001', 'hn_top', 1, 95, 22, '2026-11-01T10:00:00+00:00');
            """
        )
        conn.commit()
    finally:
        conn.close()

    # Initialize SQLiteStorage -> should auto-migrate without losing legacy row
    storage = SQLiteStorage(tmp_db_path)

    # Re-init must be idempotent
    storage = SQLiteStorage(tmp_db_path)

    conn = storage._connect()
    try:
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(hn_story_snapshots);").fetchall()
        }
        assert {"title", "url"}.issubset(cols)
        legacy_row = conn.execute(
            "SELECT * FROM hn_story_snapshots WHERE external_id = '7001';"
        ).fetchone()
        assert legacy_row is not None
        assert legacy_row["score"] == 95
        assert legacy_row["title"] == ""
        assert legacy_row["url"] == ""
    finally:
        conn.close()

    # New snapshots persist title & url cleanly
    count = await storage.record_story_snapshots(
        "hn_best",
        [
            DigestFeedItem(
                external_id="7002",
                title="SQLite WAL Internals",
                url="https://sqlite.org/wal.html",
                score=210,
                descendants=64,
            )
        ],
    )
    assert count == 1

    conn = storage._connect()
    try:
        new_row = conn.execute(
            "SELECT title, url, score FROM hn_story_snapshots WHERE external_id = '7002';"
        ).fetchone()
        assert new_row is not None
        assert new_row["title"] == "SQLite WAL Internals"
        assert new_row["url"] == "https://sqlite.org/wal.html"
        assert new_row["score"] == 210
    finally:
        conn.close()


def _insert_snapshot_with_time(
    storage: MemoryStorage | SQLiteStorage,
    *,
    external_id: str,
    feed_id: str,
    rank: int,
    score: int,
    descendants: int,
    seen_at: datetime,
    title: str,
    url: str,
) -> None:
    seen_iso = seen_at.isoformat()
    if isinstance(storage, MemoryStorage):
        storage._snapshots.setdefault(external_id, []).append(
            {
                "external_id": external_id,
                "feed_id": feed_id,
                "rank": rank,
                "score": score,
                "descendants": descendants,
                "seen_at": seen_iso,
                "title": title,
                "url": url,
            }
        )
    else:
        conn = storage._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO hn_story_snapshots (
                        external_id, feed_id, rank, score, descendants, seen_at, title, url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (external_id, feed_id, rank, score, descendants, seen_iso, title, url),
                )
        finally:
            conn.close()


@pytest.mark.asyncio
async def test_topic_timeline_trends_and_takeaways(tmp_db_path: str) -> None:
    now = datetime.now(timezone.utc)

    for idx, storage in enumerate((MemoryStorage(), SQLiteStorage(tmp_db_path))):
        # Rising topic: "rust" (1 story in older half, 3 stories in newer half; one has 2 snapshots)
        _insert_snapshot_with_time(
            storage,
            external_id="r1",
            feed_id="hn_top",
            rank=5,
            score=80,
            descendants=40,
            seen_at=now - timedelta(days=22),
            title="Rust Async Cancellation Patterns",
            url="https://example.com/rust-async",
        )
        _insert_snapshot_with_time(
            storage,
            external_id="r2",
            feed_id="hn_top",
            rank=3,
            score=100,
            descendants=50,
            seen_at=now - timedelta(days=4),
            title="Rust Compiler Frontend Speedups",
            url="https://rust-lang.org/blog/perf",
        )
        _insert_snapshot_with_time(
            storage,
            external_id="r2",
            feed_id="hn_best",
            rank=1,
            score=260,
            descendants=130,
            seen_at=now - timedelta(days=2),
            title="Rust Compiler Frontend Speedups",
            url="https://rust-lang.org/blog/perf",
        )
        _insert_snapshot_with_time(
            storage,
            external_id="r3",
            feed_id="hn_show",
            rank=2,
            score=140,
            descendants=70,
            seen_at=now - timedelta(days=1),
            title="Show HN: Zero-Copy Parser",
            url="https://github.com/org/rust-zerocopy",
        )
        # Outside 30-day window (should be ignored)
        _insert_snapshot_with_time(
            storage,
            external_id="r_old",
            feed_id="hn_top",
            rank=1,
            score=999,
            descendants=500,
            seen_at=now - timedelta(days=45),
            title="Ancient Rust News",
            url="https://example.com/old-rust",
        )

        # Cooling topic: "ebpf" (3 stories in older half, 1 in newer half)
        for j, age in enumerate((25, 21, 18), start=1):
            _insert_snapshot_with_time(
                storage,
                external_id=f"e_old_{j}",
                feed_id="hn_top",
                rank=j,
                score=200 + j * 10,
                descendants=60,
                seen_at=now - timedelta(days=age),
                title=f"eBPF Tracing Deep Dive {j}",
                url=f"https://ebpf.io/post-{j}",
            )
        _insert_snapshot_with_time(
            storage,
            external_id="e_new_1",
            feed_id="hn_new",
            rank=8,
            score=40,
            descendants=10,
            seen_at=now - timedelta(days=2),
            title="eBPF Verifier Note",
            url="https://ebpf.io/verifier",
        )

        # Steady topic: "sqlite" (2 older, 2 newer, similar scores)
        _insert_snapshot_with_time(
            storage,
            external_id="s1",
            feed_id="hn_top",
            rank=2,
            score=150,
            descendants=30,
            seen_at=now - timedelta(days=24),
            title="SQLite B-Tree Layout",
            url="https://sqlite.org/btree",
        )
        _insert_snapshot_with_time(
            storage,
            external_id="s2",
            feed_id="hn_top",
            rank=4,
            score=170,
            descendants=34,
            seen_at=now - timedelta(days=19),
            title="SQLite JSONB Benchmarks",
            url="https://sqlite.org/jsonb",
        )
        _insert_snapshot_with_time(
            storage,
            external_id="s3",
            feed_id="hn_best",
            rank=1,
            score=160,
            descendants=32,
            seen_at=now - timedelta(days=5),
            title="SQLite WAL Checkpoint Tuning",
            url="https://sqlite.org/wal",
        )
        _insert_snapshot_with_time(
            storage,
            external_id="s4",
            feed_id="hn_best",
            rank=3,
            score=165,
            descendants=33,
            seen_at=now - timedelta(days=1),
            title="SQLite FTS5 Query Planner",
            url="https://sqlite.org/fts5",
        )

        # Insufficient data topic: "zig" (only 1 distinct story)
        _insert_snapshot_with_time(
            storage,
            external_id="z1",
            feed_id="hn_top",
            rank=1,
            score=120,
            descendants=40,
            seen_at=now - timedelta(days=3),
            title="Zig 0.14 Release Notes",
            url="https://ziglang.org/download/0.14",
        )

        # Save an archived artifact for "rust"
        art_time = now - timedelta(days=2)
        saved_art = await storage.save_artifact(
            DigestArtifact(
                fingerprint=f"tl_fp_{idx}_" + ("a" * 55),
                external_id="r2",
                title="Rust Compiler Frontend Speedups",
                source_url="https://rust-lang.org/blog/perf",
                summary_markdown="## Summary\nParallel frontend cuts compile times by 30%.",
                summary_extra={
                    "key_takeaway": "Parallel rustc frontend lowers incremental build latency.",
                    "community_consensus": "Users confirm 20-35% gains on large workspaces.",
                },
                created_at=art_time,
            )
        )

        # 1. Rising timeline check (also verifies URL match r3 + multi-snapshot r2 delta)
        rust_tl = await storage.get_topic_timeline("rust", days=30, limit=50)
        assert json.dumps(rust_tl)  # strictly JSON-serializable
        assert rust_tl["topic"] == "rust"
        assert rust_tl["window_days"] == 30
        assert rust_tl["distinct_stories"] == 3
        assert rust_tl["total_snapshots"] == 4
        assert rust_tl["peak_score"] == 260
        assert rust_tl["avg_score"] == round((80 + 260 + 140) / 3, 2)
        assert rust_tl["avg_debate_ratio"] == 0.5
        assert rust_tl["trend_direction"] == "rising"

        # Sorted newest first: r3 (1d ago), r2 (2d ago), r1 (22d ago)
        assert [s["external_id"] for s in rust_tl["timeline_stories"]] == ["r3", "r2", "r1"]
        r2_entry = rust_tl["timeline_stories"][1]
        assert r2_entry["first_score"] == 100
        assert r2_entry["latest_score"] == 260
        assert r2_entry["score_delta"] == 160
        assert r2_entry["latest_descendants"] == 130
        assert r2_entry["feed_id"] == "hn_best"

        assert len(rust_tl["archived_takeaways"]) == 1
        tk = rust_tl["archived_takeaways"][0]
        assert tk["artifact_id"] == saved_art.id
        assert tk["external_id"] == "r2"
        assert tk["title"] == "Rust Compiler Frontend Speedups"
        assert tk["key_takeaway"] == "Parallel rustc frontend lowers incremental build latency."
        assert tk["community_consensus"] == "Users confirm 20-35% gains on large workspaces."
        assert tk["created_at"] == art_time.isoformat()

        # 2. Cooling timeline check
        ebpf_tl = await storage.get_topic_timeline("ebpf", days=30)
        assert ebpf_tl["distinct_stories"] == 4
        assert ebpf_tl["trend_direction"] == "cooling"

        # 3. Steady timeline check
        sqlite_tl = await storage.get_topic_timeline("sqlite", days=30)
        assert sqlite_tl["distinct_stories"] == 4
        assert sqlite_tl["trend_direction"] == "steady"

        # 4. Insufficient data check
        zig_tl = await storage.get_topic_timeline("zig", days=30)
        assert zig_tl["distinct_stories"] == 1
        assert zig_tl["trend_direction"] == "insufficient_data"

        # 5. Multi-word & comma-separated queries
        multi_word = await storage.get_topic_timeline("rust async", days=30)
        assert [s["external_id"] for s in multi_word["timeline_stories"]] == ["r1"]
        assert multi_word["trend_direction"] == "insufficient_data"

        comma_or = await storage.get_topic_timeline("zig, rust async", days=30)
        assert {s["external_id"] for s in comma_or["timeline_stories"]} == {"z1", "r1"}

        # 6. Empty / zero edge cases
        empty_tl = await storage.get_topic_timeline("   ", days=30)
        assert empty_tl["distinct_stories"] == 0
        assert empty_tl["trend_direction"] == "insufficient_data"
        assert empty_tl["timeline_stories"] == []


@pytest.mark.asyncio
async def test_recent_interest_corpus(tmp_db_path: str) -> None:
    t1 = datetime(2026, 11, 10, 8, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 11, 9, 30, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc)

    for idx, storage in enumerate((MemoryStorage(), SQLiteStorage(tmp_db_path))):
        await storage.save_artifact(
            DigestArtifact(
                fingerprint=f"corp_1_{idx}_" + ("1" * 54),
                external_id="c1",
                title="First Story on WebAssembly",
                source_url="https://example.com/wasm",
                summary_markdown="Summary 1",
                summary_extra={"key_takeaway": "Wasm component model stabilizes."},
                created_at=t1,
            )
        )
        await storage.save_artifact(
            DigestArtifact(
                fingerprint=f"corp_2_{idx}_" + ("2" * 54),
                external_id="c2",
                title="Second Story Without Takeaway",
                source_url=None,
                summary_markdown="Summary 2",
                summary_extra={},
                created_at=t2,
            )
        )
        await storage.save_artifact(
            DigestArtifact(
                fingerprint=f"corp_3_{idx}_" + ("3" * 54),
                external_id="c3",
                title="Third Story on Linux io_uring",
                source_url="https://kernel.org/io_uring",
                summary_markdown="Summary 3",
                summary_extra={"key_takeaway": "Zero-copy networking hits line rate."},
                created_at=t3,
            )
        )

        corpus = await storage.get_recent_interest_corpus(limit=2)
        assert len(corpus) == 2
        assert corpus[0] == {
            "title": "Third Story on Linux io_uring",
            "key_takeaway": "Zero-copy networking hits line rate.",
            "source_url": "https://kernel.org/io_uring",
            "created_at": t3.isoformat(),
        }
        assert corpus[1] == {
            "title": "Second Story Without Takeaway",
            "key_takeaway": "",
            "source_url": "",
            "created_at": t2.isoformat(),
        }
        assert await storage.get_recent_interest_corpus(limit=0) == []
