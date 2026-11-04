from unittest.mock import AsyncMock
import pytest

from gojo.service import GojoHNService
from gojo.storage.memory import MemoryStorage
from gojo.types import DigestFeedItem


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
