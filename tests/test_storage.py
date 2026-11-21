from datetime import datetime, timezone
import os
import sqlite3
import tempfile
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
    await storage.save_artifact(first)

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
    await storage.save_artifact(updated)

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
