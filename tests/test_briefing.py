from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from gojo.fingerprint import digest_fingerprint
from gojo.service import PROMPT_VERSION, GojoHNService
from gojo.storage.memory import MemoryStorage
from gojo.types import DigestArtifact, DigestFeedItem


@pytest.mark.asyncio
async def test_create_briefing_workflow() -> None:
    storage = MemoryStorage()
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.list_page.return_value = (
        [
            DigestFeedItem(
                feed_provider="hn",
                external_id="101",
                title="Major Vulnerability in Postman Discovered",
                score=500,
                descendants=120,
                by="security_researcher",
                url="https://example.com/cve-1",
            ),
            DigestFeedItem(
                feed_provider="hn",
                external_id="102",
                title="Linux Kernel Exploit via Coredump Pattern",
                score=350,
                descendants=80,
                by="hacker",
                url="https://example.com/exploit",
            ),
        ],
        False,
    )

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        "# Executive Technology Briefing\n\n"
        "## Key Themes & Macro Shifts\n"
        "Two major security vulnerabilities were disclosed today.\n\n"
        "## Thematic Clusters\n"
        "### Security & Privacy\n"
        "- Postman Hook Injection\n"
        "- Kernel Exploit\n\n"
        "## Actionable Takeaways for Builders\n"
        "- Audit your ~/.codex configuration."
    )

    svc = GojoHNService(
        feed_source=mock_source,
        storage=storage,
        llm_provider=mock_llm,
    )

    briefing = await svc.create_briefing(feed_id="hn_top", limit=5, locale="en")
    assert "# Executive Technology Briefing" in briefing
    assert "Two major security vulnerabilities were disclosed today." in briefing
    assert mock_llm.complete.call_count == 1

    await svc.close()


@pytest.mark.asyncio
async def test_create_briefing_limit_gt_15_not_truncated() -> None:
    storage = MemoryStorage()
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    twenty_items = [
        DigestFeedItem(
            feed_provider="hn",
            external_id=str(1000 + i),
            title=f"Story Number {i} on Distributed Systems",
            score=100 + i,
            descendants=20,
            by="author",
            url=f"https://example.com/story-{i}",
        )
        for i in range(20)
    ]
    mock_source.list_page.return_value = (twenty_items, False)

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Executive Technology Briefing\nAll 20 stories."

    svc = GojoHNService(feed_source=mock_source, storage=storage, llm_provider=mock_llm)
    await svc.create_briefing(feed_id="hn_top", limit=20, locale="en")

    # list_page must have been called with page_size >= 20
    assert mock_source.list_page.call_args.kwargs["page_size"] >= 20
    user_msg = mock_llm.complete.call_args.kwargs["user_message"]
    assert "Story Number 19 on Distributed Systems" in user_msg
    assert "Top 20 Stories" in user_msg

    await svc.close()


@pytest.mark.asyncio
async def test_create_briefing_focus_clusters_badges_and_cached_takeaway() -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)

    # Pre-seed a cached summary for story 201
    fp = digest_fingerprint(
        feed_provider="hn",
        external_id="201",
        locale="en",
        prompt_version=PROMPT_VERSION,
        summary_type="quick",
    )
    await storage.save_artifact(
        DigestArtifact(
            fingerprint=fp,
            feed_provider="hn",
            external_id="201",
            title="Rust Async Runtime Zero-Copy Benchmarks",
            locale="en",
            prompt_version=PROMPT_VERSION,
            summary_type="quick",
            summary_markdown="## Summary\nRust runtime cuts latency by 40%.",
            summary_extra={"key_takeaway": "Zero-copy io_uring cuts tail latency by 40%."},
        )
    )

    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.list_page.return_value = (
        [
            DigestFeedItem(
                feed_provider="hn",
                external_id="201",
                title="Rust Async Runtime Zero-Copy Benchmarks",
                score=420,
                descendants=310,
                by="rustacean",
                url="https://example.com/rust-io-uring?utm_source=hn",
                published_at=now - timedelta(hours=1),
            ),
            DigestFeedItem(
                feed_provider="hn",
                external_id="202",
                title="Rust Async Runtime Zero-Copy Benchmarks (Mirror)",
                score=180,
                descendants=40,
                by="mirror",
                url="https://example.com/rust-io-uring",
                published_at=now - timedelta(hours=2),
            ),
            DigestFeedItem(
                feed_provider="hn",
                external_id="203",
                title="Sourdough Bread Fermentation Tips",
                score=90,
                descendants=15,
                by="baker",
                url="https://example.com/bread",
                published_at=now - timedelta(hours=5),
            ),
        ],
        False,
    )

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Executive Technology Briefing\nRust focus."

    svc = GojoHNService(feed_source=mock_source, storage=storage, llm_provider=mock_llm)
    await svc.create_briefing(feed_id="hn_top", limit=2, locale="en", focus="Rust async")

    user_msg = mock_llm.complete.call_args.kwargs["user_message"]
    assert "Focus Profile: Rust async" in user_msg
    assert "Corroborated Cluster" in user_msg
    assert "breakout" in user_msg
    assert "Zero-copy io_uring cuts tail latency by 40%." in user_msg
    assert "Sourdough Bread" not in user_msg

    await svc.close()


@pytest.mark.asyncio
async def test_create_briefing_since_last_deltas() -> None:
    storage = MemoryStorage()

    # Record a prior sighting for story 301 at score=100
    await storage.record_story_snapshots(
        "hn_top",
        [
            DigestFeedItem(
                feed_provider="hn",
                external_id="301",
                title="Surging Compiler Release",
                score=100,
                descendants=20,
            )
        ],
    )

    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.list_page.return_value = (
        [
            DigestFeedItem(
                feed_provider="hn",
                external_id="301",
                title="Surging Compiler Release",
                score=275,
                descendants=90,
            ),
            DigestFeedItem(
                feed_provider="hn",
                external_id="302",
                title="Brand New Database Engine",
                score=140,
                descendants=35,
            ),
        ],
        False,
    )

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Executive Technology Briefing\nDelta briefing."

    svc = GojoHNService(feed_source=mock_source, storage=storage, llm_provider=mock_llm)
    await svc.create_briefing(feed_id="hn_top", limit=5, locale="en", since_last=True)

    user_msg = mock_llm.complete.call_args.kwargs["user_message"]
    assert "SURGING Δ+175 pts" in user_msg
    assert "NEW since last check" in user_msg

    # Verify latest snapshot was updated to 275 after the briefing
    snaps = await storage.get_latest_snapshots(["301", "302"])
    assert snaps["301"]["score"] == 275
    assert snaps["302"]["score"] == 140

    await svc.close()
