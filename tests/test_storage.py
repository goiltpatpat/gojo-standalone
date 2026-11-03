import os
import tempfile
import pytest

from gojo.storage.memory import MemoryStorage
from gojo.storage.sqlite import SQLiteStorage
from gojo.types import DigestArtifact


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


@pytest.mark.asyncio
async def test_memory_storage(sample_artifact: DigestArtifact) -> None:
    storage = MemoryStorage()
    saved = await storage.save_artifact(sample_artifact)
    assert saved.id == sample_artifact.id

    fetched = await storage.get_artifact_by_fingerprint(sample_artifact.fingerprint)
    assert fetched is not None
    assert fetched.title == "Test Headline"

    by_id = await storage.get_artifact_by_id(sample_artifact.id)
    assert by_id is not None
    assert by_id.fingerprint == sample_artifact.fingerprint

    # Feed cache
    await storage.set_feed_cache("hn_top", 1, {"items": [1, 2, 3]}, ttl_seconds=10)
    cached = await storage.get_feed_cache("hn_top", 1)
    assert cached == {"items": [1, 2, 3]}


@pytest.mark.asyncio
async def test_sqlite_storage(sample_artifact: DigestArtifact) -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        storage = SQLiteStorage(tmp_db)
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

        # Feed cache
        await storage.set_feed_cache("hn_top", 1, {"items": ["a", "b"]}, ttl_seconds=60)
        cache_data = await storage.get_feed_cache("hn_top", 1)
        assert cache_data == {"items": ["a", "b"]}
    finally:
        if os.path.exists(tmp_db):
            os.remove(tmp_db)
