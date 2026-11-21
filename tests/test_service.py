import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gojo.service import GojoHNService, _prefer_main_html, _strip_html
from gojo.storage.memory import MemoryStorage
from gojo.types import DigestFeedItem


def test_html_helpers() -> None:
    html = (
        "<html><body><script>alert(1)</script><style>.x{color:red}</style>"
        '<iframe src="https://evil.example/ad">ad frame</iframe>'
        '<img src="https://cdn.example/a.png" alt="pic" />'
        "<article><p>Crucial &amp; Clean &quot;Content&quot; Here</p></article>"
        "</body></html>"
    )
    preferred = _prefer_main_html(html)
    assert "<script>" in html
    assert preferred == html  # <article> is <= 400 chars
    stripped = _strip_html(html)
    assert "alert" not in stripped
    assert "<img" not in stripped
    assert "<iframe" not in stripped
    assert "ad frame" not in stripped
    assert 'Crucial & Clean "Content" Here' in stripped


@pytest.mark.asyncio
async def test_service_list_feed_and_cache() -> None:
    storage = MemoryStorage()
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.list_page.return_value = (
        [
            DigestFeedItem(
                feed_provider="hn",
                external_id="101",
                title="Python 3.14 Released",
                score=300,
                descendants=150,
                published_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )
        ],
        False,
    )

    svc = GojoHNService(feed_source=mock_source, storage=storage)

    # First fetch (hits source, enriches signals, records snapshot)
    res1 = await svc.list_feed(feed_id="hn_top", page=1, page_size=15)
    assert res1.completion.status == "success"
    assert res1.completion.cache_layer == "origin"
    assert len(res1.items) == 1
    assert res1.items[0].velocity is not None
    assert "breakout" in res1.items[0].signal_badges
    assert mock_source.list_page.call_count == 1

    # Snapshot recorded in storage
    snaps = await storage.get_latest_snapshots(["101"])
    assert "101" in snaps
    assert snaps["101"]["score"] == 300

    # Second fetch same page_size (hits memory cache and reports layer_name == "memory")
    res2 = await svc.list_feed(feed_id="hn_top", page=1, page_size=15)
    assert res2.completion.status == "success"
    assert res2.completion.cache_layer == "memory"
    assert len(res2.items) == 1
    assert mock_source.list_page.call_count == 1

    # Different page_size must NOT collide in cache
    res3 = await svc.list_feed(feed_id="hn_top", page=1, page_size=30)
    assert res3.completion.cache_layer == "origin"
    assert mock_source.list_page.call_count == 2

    # Additional HN feeds (best/ask/show) accepted
    for fid in ("hn_best", "hn_ask", "hn_show"):
        r = await svc.list_feed(feed_id=fid, page=1)
        assert r.feed_id == fid

    await svc.close()


@pytest.mark.asyncio
async def test_service_summarize_idempotent_and_get_artifact() -> None:
    storage = MemoryStorage()
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.get_item.return_value = {
        "id": 555,
        "title": "Quantum Computing Milestone",
        "url": "https://example.com/quantum",
    }

    async def fake_summarizer(**kwargs):
        return (
            "## Summary\n"
            "Physicists achieved quantum supremacy.\n\n"
            "### Why it matters\n"
            "It speeds up computation exponentially.\n\n"
            "### Key takeaway\n"
            "Qubits are stable.\n\n"
            "### Gojo note\n"
            "Watch out for error correction overhead.\n"
        )

    svc = GojoHNService(
        feed_source=mock_source,
        storage=storage,
        custom_summarizer=fake_summarizer,
    )

    # First summarize call (generates)
    res1 = await svc.summarize(external_id="555", summary_type="quick")
    assert res1.completion.status == "success"
    assert res1.completion.cache_layer == "origin"
    assert res1.artifact.title == "Quantum Computing Milestone"
    assert res1.artifact.summary_extra.get("key_takeaway") == "Qubits are stable."

    # Second summarize call (idempotent cache hit with accurate layer_name)
    res2 = await svc.summarize(external_id="555", summary_type="quick")
    assert res2.completion.status == "cached"
    assert res2.completion.cache_layer == "memory"
    assert res2.artifact.fingerprint == res1.artifact.fingerprint
    assert res2.artifact.id == res1.artifact.id

    # get_artifact() delegates to storage.get_artifact_by_id()
    fetched = await svc.get_artifact(res1.artifact.id)
    assert fetched is not None
    assert fetched.id == res1.artifact.id
    assert fetched.title == "Quantum Computing Milestone"

    await svc.close()


@pytest.mark.asyncio
async def test_service_ssrf_hop_by_hop_redirect_blocking() -> None:
    """A public URL redirecting to cloud metadata IP must be blocked before fetching target."""
    visited_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        visited_urls.append(str(request.url))
        if str(request.url) == "https://public.example.com/start":
            return httpx.Response(
                302,
                headers={"Location": "http://169.254.169.254/latest/meta-data/"},
            )
        return httpx.Response(200, text="SECRET_METADATA")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)

    svc = GojoHNService(storage=MemoryStorage(), enable_jina_fallback=True)
    svc._article_http = client

    checked_urls: list[str] = []

    async def fake_validate(u: str) -> tuple[bool, str]:
        checked_urls.append(u)
        if "169.254.169.254" in u:
            return False, "Blocked link-local metadata IP"
        return True, ""

    with (
        patch("gojo.service.validate_public_url_with_dns", side_effect=fake_validate),
        patch.object(svc, "_fetch_via_jina_reader", new_callable=AsyncMock) as mock_jina,
    ):
        text, status = await svc._fetch_article_text(
            "Redirect Attack", "https://public.example.com/start"
        )

    assert status == "blocked"
    assert "SECRET_METADATA" not in text
    assert checked_urls == [
        "https://public.example.com/start",
        "http://169.254.169.254/latest/meta-data/",
    ]
    assert visited_urls == ["https://public.example.com/start"]
    mock_jina.assert_not_called()
    await svc.close()


@pytest.mark.asyncio
async def test_service_byte_capped_streaming() -> None:
    """_safe_fetch_response must enforce max_bytes cap while streaming."""
    chunk = b"A" * 512

    class BigStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(20):  # 10 KB total
                yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            stream=BigStream(),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    svc = GojoHNService(storage=MemoryStorage())
    svc._article_http = client

    with patch("gojo.service.validate_public_url_with_dns", return_value=(True, "")):
        body, ct, final_url, status = await svc._safe_fetch_response(
            "https://public.example.com/huge.txt",
            headers={},
            max_bytes=1200,
        )

    assert status == "ok"
    assert ct == "text/plain"
    assert final_url == "https://public.example.com/huge.txt"
    assert len(body) == 1200
    await svc.close()


@pytest.mark.asyncio
async def test_service_pdf_graceful_fallback_and_jina_preserves_newlines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith(".pdf"):
            return httpx.Response(
                200,
                headers={"Content-Type": "application/pdf"},
                content=b"%PDF-1.4\x00\xffbinary_garbage_payload",
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/markdown"},
            content=b"# Heading\n\nFirst paragraph line.\n\n- Bullet one\n- Bullet two\n" + b"x" * 100,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    svc = GojoHNService(storage=MemoryStorage(), enable_jina_fallback=True)
    svc._article_http = client

    with (
        patch("gojo.service.validate_public_url_with_dns", return_value=(True, "")),
        patch("gojo.service._extract_pdf_text", return_value=None),
    ):
        pdf_content, pdf_status = await svc._fetch_article_text(
            "Whitepaper", "https://public.example.com/paper.pdf"
        )
        assert pdf_status == "fallback_title"
        assert "binary_garbage_payload" not in pdf_content
        assert "pypdf" in pdf_content

        jina_md = await svc._fetch_via_jina_reader("https://public.example.com/post")
        assert "\n\n" in jina_md
        assert "# Heading\n\nFirst paragraph line." in jina_md

    await svc.close()


@pytest.mark.asyncio
async def test_service_summarize_concurrent_article_and_algolia() -> None:
    """_fetch_article_text and algolia.fetch_discussion_context must execute concurrently."""
    article_started = asyncio.Event()
    algolia_started = asyncio.Event()

    async def slow_article(title: str, url: str) -> tuple[str, str]:
        article_started.set()
        # Will time out if sequential!
        await asyncio.wait_for(algolia_started.wait(), timeout=1.0)
        return f"Title: {title}\nContent: Full body text here.", "ok"

    async def slow_algolia(external_id: str) -> str:
        algolia_started.set()
        await asyncio.wait_for(article_started.wait(), timeout=1.0)
        return "[1] @alice: Great benchmark."

    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.get_item.return_value = {
        "id": 777,
        "title": "Concurrent Pipeline Test",
        "url": "https://example.com/concurrent",
    }
    mock_algolia = AsyncMock()
    mock_algolia.fetch_discussion_context.side_effect = slow_algolia

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "## Summary\nDone."
    mock_llm.last_used_model = "gemini-2.5-flash"

    svc = GojoHNService(
        feed_source=mock_source,
        algolia_provider=mock_algolia,
        storage=MemoryStorage(),
        llm_provider=mock_llm,
    )

    with patch.object(svc, "_fetch_article_text", side_effect=slow_article):
        res = await svc.summarize(external_id="777", include_comments=True)

    assert res.completion.status == "success"
    assert article_started.is_set() and algolia_started.is_set()
    await svc.close()


@pytest.mark.asyncio
async def test_service_model_passthrough_none_and_records_last_used_model(monkeypatch) -> None:
    """When no explicit model or GOJO_HN_MODEL is set, pass model=None and record provider.last_used_model."""
    monkeypatch.delenv("GOJO_HN_MODEL", raising=False)
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"
    mock_source.get_item.return_value = {
        "id": 888,
        "title": "Ask HN: Self-post HTML",
        "text": "<p>Hello &amp; <b>world</b></p>",
    }

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "## Summary\nGemini summarized this."
    mock_llm.last_used_model = "gemini-2.5-flash"

    svc = GojoHNService(
        feed_source=mock_source,
        algolia_provider=AsyncMock(),
        storage=MemoryStorage(),
        llm_provider=mock_llm,
    )

    res = await svc.summarize(external_id="888", include_comments=False)
    # model=None passed through so Gemini/Ollama use their own default
    call_kwargs = mock_llm.complete.call_args.kwargs
    assert call_kwargs["model"] is None
    # Recorded actual model used by provider
    assert res.artifact.model == "gemini-2.5-flash"
    assert res.artifact.model_version == "gemini-2.5-flash"
    # Self-post HTML was cleaned
    assert "<p>" not in call_kwargs["user_message"]
    assert "Hello & world" in call_kwargs["user_message"]

    await svc.close()


@pytest.mark.asyncio
async def test_service_close_closes_all_pooled_clients() -> None:
    mock_source = AsyncMock()
    mock_algolia = AsyncMock()
    mock_llm = AsyncMock()
    svc = GojoHNService(
        feed_source=mock_source,
        algolia_provider=mock_algolia,
        storage=MemoryStorage(),
        llm_provider=mock_llm,
    )
    # Force creation of _article_http
    client = await svc._article_client()
    assert not client.is_closed

    await svc.close()
    assert client.is_closed
    mock_source.close.assert_awaited_once()
    mock_algolia.close.assert_awaited_once()
    mock_llm.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_search_archive_and_hn_archive() -> None:
    storage = MemoryStorage()
    mock_algolia = AsyncMock()
    mock_algolia.search_stories.return_value = [
        DigestFeedItem(
            feed_provider="hn",
            external_id="901",
            title="SQLite FTS5 Deep Dive",
            score=400,
            descendants=120,
            published_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
    ]

    svc = GojoHNService(
        feed_source=AsyncMock(),
        algolia_provider=mock_algolia,
        storage=storage,
        llm_provider=AsyncMock(),
    )

    hn_hits = await svc.search_hn_archive("SQLite", limit=5, sort_by="date")
    assert len(hn_hits) == 1
    assert hn_hits[0].velocity is not None
    mock_algolia.search_stories.assert_awaited_once_with("SQLite", limit=5, sort_by="date")

    arch_hits = await svc.search_archive("SQLite", limit=5)
    assert arch_hits == []

    await svc.close()


@pytest.mark.asyncio
async def test_service_uses_auto_llm_provider_when_openai_key_in_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_API_KEY in env must still use AutoLLMProvider (keeping failover/cooldowns active);
    only explicit openai_api_key pins OpenAIProvider directly."""
    from gojo.providers.llm import AutoLLMProvider, OpenAIProvider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

    svc_env = GojoHNService(storage=MemoryStorage())
    assert isinstance(svc_env.llm_provider, AutoLLMProvider)
    await svc_env.close()

    svc_explicit = GojoHNService(storage=MemoryStorage(), openai_api_key="sk-explicit-key")
    assert isinstance(svc_explicit.llm_provider, OpenAIProvider)
    await svc_explicit.close()


@pytest.mark.asyncio
async def test_service_jina_reader_blocks_private_urls_directly() -> None:
    """Defense-in-depth: direct calls to _fetch_via_jina_reader must reject private/loopback URLs."""
    svc = GojoHNService(storage=MemoryStorage(), enable_jina_fallback=True)
    with patch.object(svc, "_safe_fetch_response", new_callable=AsyncMock) as mock_fetch:
        assert await svc._fetch_via_jina_reader("http://169.254.169.254/latest/meta-data/") == ""
        assert await svc._fetch_via_jina_reader("http://127.0.0.1:8080/admin") == ""
        assert await svc._fetch_via_jina_reader("http://10.0.0.1/internal") == ""
        mock_fetch.assert_not_called()
    await svc.close()


@pytest.mark.asyncio
async def test_service_async_context_manager_closes_on_exit() -> None:
    mock_source = AsyncMock()
    mock_algolia = AsyncMock()
    mock_llm = AsyncMock()

    async with GojoHNService(
        feed_source=mock_source,
        algolia_provider=mock_algolia,
        storage=MemoryStorage(),
        llm_provider=mock_llm,
    ) as svc:
        client = await svc._article_client()
        assert not client.is_closed

    assert client.is_closed
    mock_source.close.assert_awaited_once()
    mock_algolia.close.assert_awaited_once()
    mock_llm.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_radar_dedup_signals_badges_focus_and_clusters() -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    recent_ts = int((now - timedelta(minutes=30)).timestamp())

    story_a = DigestFeedItem(
        feed_provider="hn",
        external_id="1001",
        title="Rust Kernel Driver Framework Released",
        url="https://example.com/rust-kernel",
        score=280,
        descendants=240,
        time=recent_ts,
        published_at=now - timedelta(minutes=30),
    )
    story_a_dup = DigestFeedItem(
        feed_provider="hn",
        external_id="1002",
        title="Rust Kernel Driver Framework Released",
        url="https://example.com/rust-kernel?utm_source=hn",
        score=210,
        descendants=190,
        time=recent_ts,
        published_at=now - timedelta(minutes=30),
    )
    story_b = DigestFeedItem(
        feed_provider="hn",
        external_id="1003",
        title="Sourdough Baking Tips for Beginners",
        url="https://example.com/bread",
        score=12,
        descendants=2,
        time=recent_ts,
        published_at=now - timedelta(minutes=30),
    )

    mock_source = AsyncMock()
    mock_source.provider_id = "hn"

    async def fake_list_page(feed_variant: str, page: int, page_size: int):
        if feed_variant == "top":
            return [story_a, story_b], False
        if feed_variant == "best":
            # story_a repeated across feeds + story_a_dup same canonical URL
            return [story_a, story_a_dup], False
        return [], False

    mock_source.list_page.side_effect = fake_list_page
    mock_llm = AsyncMock()

    async with GojoHNService(
        feed_source=mock_source,
        storage=storage,
        llm_provider=mock_llm,
    ) as svc:
        res = await svc.radar(
            feeds=["hn_top", "hn_best"],
            limit_per_feed=10,
            top_n=5,
            focus="rust kernel",
            badges=["breakout", "high-debate"],
        )

    assert res["feeds_scanned"] == ["hn_top", "hn_best"]
    assert res["total_scanned"] == 3  # 1001, 1003, 1002 (deduped 1001)
    assert res["focus_applied"] == "rust kernel"
    assert res["badges_filter"] == ["breakout", "high-debate"]
    # story_b filtered out by badges; 1001 and 1002 remain and cluster together via canonical URL
    assert len(res["top_items"]) == 2
    assert res["top_items"][0]["external_id"] in {"1001", "1002"}
    assert res["top_items"][0]["velocity"] is not None
    assert "breakout" in res["top_items"][0]["signal_badges"]
    assert len(res["clusters"]) == 1
    assert res["clusters"][0]["size"] == 2
    # Zero LLM tokens used
    mock_llm.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_service_deep_dive_concurrent_cache_and_partial_errors() -> None:
    storage = MemoryStorage()
    mock_source = AsyncMock()
    mock_source.provider_id = "hn"

    async def fake_get_item(external_id: str):
        if external_id == "201":
            return {"id": 201, "title": "SQLite WAL Architecture", "text": "WAL details."}
        if external_id == "202":
            return {"id": 202, "title": "SQLite FTS5 Benchmarks", "text": "FTS5 details."}
        return None  # 203 triggers ValueError inside summarize

    mock_source.get_item.side_effect = fake_get_item

    mock_algolia = AsyncMock()
    mock_algolia.fetch_discussion_context.return_value = "[1] @alice: Great insight."
    mock_algolia.search_stories.return_value = [
        DigestFeedItem(feed_provider="hn", external_id="201", title="SQLite WAL Architecture", score=300),
        DigestFeedItem(feed_provider="hn", external_id="202", title="SQLite FTS5 Benchmarks", score=250),
        DigestFeedItem(feed_provider="hn", external_id="203", title="Missing Story", score=50),
    ]

    async def fake_summarizer(**kwargs):
        return "## Summary\nTechnical breakdown.\n\n### Key takeaway\nSQLite is fast."

    async with GojoHNService(
        feed_source=mock_source,
        algolia_provider=mock_algolia,
        storage=storage,
        custom_summarizer=fake_summarizer,
    ) as svc:
        # Pre-warm cache for story 201 to verify cached_hits tracking & local_archive_matches
        await svc.summarize(external_id="201", summary_type="technical", locale="en")

        dossier = await svc.deep_dive(
            topic="SQLite",
            max_stories=3,
            summary_type="technical",
            locale="en",
            include_comments=True,
        )

        with pytest.raises(ValueError, match="non-empty"):
            await svc.deep_dive(topic="   ")

    assert dossier["topic"] == "SQLite"
    assert dossier["stories_analyzed"] == 2
    assert dossier["cached_hits"] == 1
    assert len(dossier["summaries"]) == 2
    assert len(dossier["local_archive_matches"]) >= 1
    assert len(dossier["errors"]) == 1
    assert dossier["errors"][0]["external_id"] == "203"
