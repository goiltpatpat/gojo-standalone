"""
GojoHNService: Main orchestration engine for Hacker News feed fetching,
SSRF-safe article content extraction with Trafilatura and Jina fallback,
community discussion synthesis via Algolia, multi-model AI summarization,
fingerprint deduplication, and persistent caching.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
import re
import time
from typing import Any, Callable, Coroutine
from urllib.parse import urlparse

import httpx

from .fingerprint import digest_fingerprint
from .interpretation import extract_summary_extra_from_markdown
from .prompts import (
    build_briefing_system_prompt,
    build_digest_system_prompt,
    build_digest_user_message,
    effective_locale_for_summary,
    normalize_summary_type,
)
from .providers.algolia import HackerNewsAlgoliaProvider
from .providers.base import FeedSource
from .providers.hn import HackerNewsFeedSource
from .providers.llm import (
    AutoLLMProvider,
    LLMProvider,
    OpenAIProvider,
)
from .ssrf import is_public_http_url_for_fetch
from .storage.base import StorageBackend
from .storage.sqlite import SQLiteStorage
from .types import (
    DigestArtifact,
    DigestCompletion,
    DigestFeedItem,
    DigestFeedResponse,
    DigestSummaryRequest,
    DigestSummaryResponse,
    SummaryType,
)

PROMPT_VERSION = "v5"
LIST_CACHE_TTL_S = 900
PAGE_SIZE = 15
PAGE_FETCH_TIMEOUT_S = 12.0


def _strip_html(html: str) -> str:
    """Strip script/style tags and HTML markup to retain readable plain text."""
    t = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.I)
    t = re.sub(r"<style[\s\S]*?</style>", "", t, flags=re.I)
    t = re.sub(r"<(?!img|iframe|/iframe)[^>]+>", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:20000]


def _prefer_main_html(html: str) -> str:
    """Prefer <article> or <main> tag to eliminate navigation chrome/header/footer."""
    for pat in (
        r"<article\b[^>]*>([\s\S]*?)</article>",
        r"<main\b[^>]*>([\s\S]*?)</main>",
    ):
        m = re.search(pat, html, re.I)
        if m and len(m.group(1).strip()) > 400:
            return m.group(1)
    return html


class GojoHNService:
    """Production service for Hacker News feeds, discussion trees, and intelligent summaries."""

    def __init__(
        self,
        *,
        feed_source: FeedSource | None = None,
        algolia_provider: HackerNewsAlgoliaProvider | None = None,
        storage: StorageBackend | None = None,
        llm_provider: LLMProvider | None = None,
        openai_api_key: str | None = None,
        default_model: str | None = None,
        enable_jina_fallback: bool | None = None,
        custom_summarizer: Callable[..., Coroutine[Any, Any, str]] | None = None,
    ) -> None:
        self.source = feed_source or HackerNewsFeedSource()
        self.algolia = algolia_provider or HackerNewsAlgoliaProvider()
        self.storage = storage or SQLiteStorage(
            os.getenv("GOJO_HN_DB_PATH", "gojo_hn.db")
        )
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model or os.getenv("GOJO_HN_MODEL", "gpt-4o-mini")

        if llm_provider is not None:
            self.llm_provider = llm_provider
        elif self.api_key:
            self.llm_provider = OpenAIProvider(api_key=self.api_key)
        else:
            self.llm_provider = AutoLLMProvider()

        jina_env = os.getenv("GOJO_HN_JINA_FALLBACK", "true").lower() in ("true", "1", "yes")
        self.enable_jina_fallback = enable_jina_fallback if enable_jina_fallback is not None else jina_env
        self.custom_summarizer = custom_summarizer

        self._article_http: httpx.AsyncClient | None = None

    async def _article_client(self) -> httpx.AsyncClient:
        if self._article_http is None or self._article_http.is_closed:
            self._article_http = httpx.AsyncClient(
                timeout=PAGE_FETCH_TIMEOUT_S,
                follow_redirects=True,
                max_redirects=5,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._article_http

    async def close(self) -> None:
        """Close open HTTP client connections."""
        if self._article_http is not None and not self._article_http.is_closed:
            await self._article_http.aclose()
            self._article_http = None
        await self.algolia.close()

    async def list_feed(
        self,
        *,
        feed_id: str = "hn_top",
        page: int = 1,
        page_size: int = PAGE_SIZE,
        force_refresh: bool = False,
    ) -> DigestFeedResponse:
        """
        List stories for feed_id ('hn_top' or 'hn_new').
        Returns cached list if present and unexpired unless force_refresh is True.
        """
        if feed_id not in ("hn_top", "hn_new"):
            raise ValueError(f"Invalid feed_id '{feed_id}'. Must be 'hn_top' or 'hn_new'.")

        page = max(1, page)
        if not force_refresh:
            cached = await self.storage.get_feed_cache(feed_id, page)
            if cached and isinstance(cached, dict):
                items = [DigestFeedItem.model_validate(x) for x in cached.get("items", [])]
                ref_at = None
                if cached.get("refreshed_at"):
                    try:
                        ref_at = datetime.fromisoformat(cached["refreshed_at"])
                    except Exception:
                        pass
                return DigestFeedResponse(
                    feed_id=feed_id,
                    page=page,
                    items=items,
                    has_more=bool(cached.get("has_more")),
                    completion=DigestCompletion(status="success", cache_layer="sqlite"),
                    feed_refreshed_at=ref_at,
                    list_cache_ttl_seconds=LIST_CACHE_TTL_S,
                )

        variant = "new" if feed_id == "hn_new" else "top"
        items, has_more = await self.source.list_page(
            feed_variant=variant, page=page, page_size=page_size
        )

        now = datetime.now(timezone.utc)
        payload = {
            "feed_id": feed_id,
            "page": page,
            "has_more": has_more,
            "refreshed_at": now.isoformat(),
            "items": [it.model_dump(mode="json") for it in items],
        }
        await self.storage.set_feed_cache(feed_id, page, payload, ttl_seconds=LIST_CACHE_TTL_S)

        return DigestFeedResponse(
            feed_id=feed_id,
            page=page,
            items=items,
            has_more=has_more,
            completion=DigestCompletion(status="success", cache_layer="origin"),
            feed_refreshed_at=now,
            list_cache_ttl_seconds=LIST_CACHE_TTL_S,
        )

    async def get_artifact(self, artifact_id: str) -> DigestArtifact | None:
        """Retrieve a saved digest artifact by its primary key UUID."""
        return await self.storage.get_artifact(artifact_id)

    async def get_artifact_by_external_id(
        self,
        external_id: str,
        *,
        summary_type: str = "quick",
        locale: str = "en",
    ) -> DigestArtifact | None:
        """Find artifact matching external ID, summary type, and locale."""
        st = normalize_summary_type(summary_type)
        eff_locale = effective_locale_for_summary(locale, st)
        fp = digest_fingerprint(
            feed_provider=self.source.provider_id,
            external_id=str(external_id),
            locale=eff_locale,
            prompt_version=PROMPT_VERSION,
            summary_type=st,
        )
        return await self.storage.get_artifact_by_fingerprint(fp)

    async def summarize(
        self,
        *,
        external_id: str,
        summary_type: str = "quick",
        locale: str = "en",
        model: str | None = None,
        force_regenerate: bool = False,
        user_id: str | None = None,
        include_comments: bool = True,
    ) -> DigestSummaryResponse:
        """
        Generate or fetch cached summary for a Hacker News story.
        Idempotent by fingerprint: identical inputs return cached artifact unless force_regenerate is True.
        """
        st = normalize_summary_type(summary_type)
        eff_locale = effective_locale_for_summary(locale, st)
        fp = digest_fingerprint(
            feed_provider=self.source.provider_id,
            external_id=str(external_id),
            locale=eff_locale,
            prompt_version=PROMPT_VERSION,
            summary_type=st,
        )

        if not force_regenerate:
            cached_art = await self.storage.get_artifact_by_fingerprint(fp)
            if cached_art:
                return DigestSummaryResponse(
                    artifact=cached_art,
                    completion=DigestCompletion(
                        status="cached", cache_layer="sqlite", hydrated=False
                    ),
                )

        raw = await self.source.get_item(external_id=str(external_id))
        if not raw:
            raise ValueError(f"Hacker News story ID '{external_id}' not found.")

        title = str(raw.get("title") or "")
        source_url = raw.get("url")
        content_to_summarize = title
        source_fetch_status = "fallback_title"

        if source_url and isinstance(source_url, str):
            if is_public_http_url_for_fetch(source_url):
                content_to_summarize, source_fetch_status = await self._fetch_article_text(
                    title, source_url
                )
            else:
                content_to_summarize = (
                    f"{title}\n(URL not fetched for security; summarize from title and context)."
                )
                source_fetch_status = "blocked"
        elif raw.get("text"):
            content_to_summarize = f"{title}\nContent: {raw.get('text')}"
            source_fetch_status = "ok"

        # Harvest discussion context from Algolia in 1 request
        discussion_context = ""
        if include_comments:
            try:
                discussion_context = await self.algolia.fetch_discussion_context(str(external_id))
            except Exception:
                discussion_context = ""

        use_model = model or self.default_model
        t0 = time.perf_counter()

        summary_markdown = await self._generate_summary(
            model=use_model,
            locale=eff_locale,
            content=content_to_summarize,
            summary_type=st,
            source_fetch_status=source_fetch_status,
            discussion_context=discussion_context,
        )
        generated_ms = int((time.perf_counter() - t0) * 1000)
        summary_extra = extract_summary_extra_from_markdown(summary_markdown)
        content_length = len(summary_markdown.encode("utf-8"))

        artifact = DigestArtifact(
            fingerprint=fp,
            fingerprint_version="v2",
            feed_provider=self.source.provider_id,
            external_id=str(external_id),
            title=title or None,
            source_url=str(source_url) if source_url else None,
            locale=eff_locale,
            prompt_version=PROMPT_VERSION,
            model=use_model,
            summary_markdown=summary_markdown,
            summary_type=st,
            summary_extra=summary_extra,
            content_length=content_length,
            generated_ms=generated_ms,
            source_fetch_status=source_fetch_status,
            model_version=use_model,
            created_at=datetime.now(timezone.utc),
            created_by_user_id=user_id,
        )

        saved = await self.storage.save_artifact(artifact)
        return DigestSummaryResponse(
            artifact=saved,
            completion=DigestCompletion(status="success", cache_layer="origin", hydrated=False),
        )

    async def _fetch_article_text(self, title: str, url: str) -> tuple[str, str]:
        """Fetch article HTML using Trafilatura or fallback via Jina reader."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        direct_body = ""
        direct_ok = False
        try:
            client = await self._article_client()
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            raw_html = r.text

            # Tier 1: Trafilatura (structural markdown content extraction)
            try:
                import trafilatura
                trafil_text = trafilatura.extract(raw_html, output_format="markdown", include_links=False)
                if trafil_text and len(trafil_text.strip()) >= 250:
                    direct_body = trafil_text.strip()
                    direct_ok = True
            except Exception:
                pass

            # Tier 2: Prefer main/article tag + regex strip
            if not direct_ok:
                html = _prefer_main_html(raw_html)
                direct_body = _strip_html(html)
                direct_ok = True
        except Exception:
            pass

        if direct_ok and len(direct_body) >= 500:
            return f"Title: {title}\nContent: {direct_body[:25000]}", "ok"

        # Tier 3: Jina Reader fallback
        if self.enable_jina_fallback:
            jina_text = await self._fetch_via_jina_reader(url)
            if jina_text and len(jina_text) >= 280:
                return f"Title: {title}\nContent: {jina_text[:25000]}", "ok"

        if direct_ok and len(direct_body) >= 80:
            return f"Title: {title}\nContent: {direct_body[:25000]}", "ok"

        return f"{title}\n(Could not fetch page content; summarize from title.)", "error"

    async def _fetch_via_jina_reader(self, article_url: str) -> str:
        """Fallback to Jina Reader (https://r.jina.ai/) for reader-mode markdown."""
        jina_url = f"https://r.jina.ai/{article_url}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/plain,text/markdown,*/*;q=0.8",
        }
        try:
            client = await self._article_client()
            r = await client.get(jina_url, headers=headers, timeout=httpx.Timeout(20.0, connect=10.0))
            r.raise_for_status()
            text = (r.text or "").strip()
            if len(text) < 80:
                return ""
            text = re.sub(r"\s+", " ", text).strip()
            return text[:20000]
        except Exception:
            return ""

    async def _generate_summary(
        self,
        *,
        model: str,
        locale: str,
        content: str,
        summary_type: SummaryType,
        source_fetch_status: str,
        discussion_context: str | None = None,
    ) -> str:
        """Call LLM summarizer using custom callback or configured LLM provider."""
        if self.custom_summarizer is not None:
            try:
                return await self.custom_summarizer(
                    model=model,
                    locale=locale,
                    content=content,
                    summary_type=summary_type,
                    source_fetch_status=source_fetch_status,
                    discussion_context=discussion_context,
                )
            except TypeError:
                return await self.custom_summarizer(
                    model=model,
                    locale=locale,
                    content=content,
                    summary_type=summary_type,
                    source_fetch_status=source_fetch_status,
                )

        system_prompt = build_digest_system_prompt(
            summary_type=summary_type,
            locale=locale,
            include_discussion=bool(discussion_context and discussion_context.strip()),
        )
        user_message = build_digest_user_message(
            content,
            source_fetch_status=source_fetch_status,
            discussion_context=discussion_context,
        )

        return await self.llm_provider.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            max_tokens=4096,
        )

    async def create_briefing(
        self,
        *,
        feed_id: str = "hn_top",
        limit: int = 10,
        locale: str = "en",
        model: str | None = None,
    ) -> str:
        """
        Generate a multi-story executive technical briefing synthesizing top stories.
        """
        feed = await self.list_feed(feed_id=feed_id, page=1)
        items = feed.items[:limit]
        if not items:
            return "No stories available to create briefing."

        context_lines = [
            f"=== Hacker News Feed: {feed_id.upper()} (Top {len(items)} Stories) ===\n"
        ]
        for idx, it in enumerate(items, 1):
            context_lines.append(
                f"{idx}. [{it.external_id}] {it.title}\n"
                f"   Score: {it.score} pts | Comments: {it.descendants} | By: @{it.by or 'unknown'}\n"
                f"   URL: {it.url or 'HN Discussion'}\n"
            )

        stories_context = "\n".join(context_lines)
        system_prompt = build_briefing_system_prompt(locale=locale)
        user_msg = (
            f"Please synthesize the following top Hacker News developments into an authoritative Executive Briefing:\n\n"
            f"{stories_context}"
        )

        use_model = model or self.default_model
        return await self.llm_provider.complete(
            system_prompt=system_prompt,
            user_message=user_msg,
            model=use_model,
            max_tokens=4096,
        )
