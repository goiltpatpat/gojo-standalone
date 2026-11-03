"""
GojoHNService: Main orchestration engine for Hacker News feed fetching,
SSRF-safe article content extraction, AI summarization, fingerprint deduplication,
and persistent caching.
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
    build_digest_system_prompt,
    build_digest_user_message,
    effective_locale_for_summary,
    normalize_summary_type,
)
from .providers.base import FeedSource
from .providers.hn import HackerNewsFeedSource
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
    """Production service for Hacker News feeds and intelligent summaries."""

    def __init__(
        self,
        *,
        feed_source: FeedSource | None = None,
        storage: StorageBackend | None = None,
        openai_api_key: str | None = None,
        default_model: str | None = None,
        enable_jina_fallback: bool | None = None,
        custom_summarizer: Callable[..., Coroutine[Any, Any, str]] | None = None,
    ) -> None:
        self.source = feed_source or HackerNewsFeedSource()
        self.storage = storage or SQLiteStorage(
            os.getenv("GOJO_HN_DB_PATH", "gojo_hn.db")
        )
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model or os.getenv("GOJO_HN_MODEL", "gpt-4o-mini")
        
        jina_env = os.getenv("GOJO_HN_JINA_FALLBACK", "true").lower() in ("true", "1", "yes")
        self.enable_jina_fallback = enable_jina_fallback if enable_jina_fallback is not None else jina_env
        self.custom_summarizer = custom_summarizer

        self._article_http: httpx.AsyncClient | None = None

    async def _article_client(self) -> httpx.AsyncClient:
        if self._article_http is None:
            self._article_http = httpx.AsyncClient(
                timeout=PAGE_FETCH_TIMEOUT_S,
                follow_redirects=True,
                max_redirects=5,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._article_http

    async def close(self) -> None:
        """Close open HTTP client connections."""
        if self._article_http is not None:
            await self._article_http.aclose()
            self._article_http = None

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
            "items": [i.model_dump(mode="json") for i in items],
            "has_more": has_more,
            "refreshed_at": now.isoformat(),
        }
        await self.storage.set_feed_cache(feed_id, page, payload, LIST_CACHE_TTL_S)

        return DigestFeedResponse(
            feed_id=feed_id,
            page=page,
            items=items,
            has_more=has_more,
            completion=DigestCompletion(status="success", cache_layer="origin"),
            feed_refreshed_at=now,
            list_cache_ttl_seconds=LIST_CACHE_TTL_S,
        )

    async def get_cached_summary(
        self,
        *,
        external_id: str,
        summary_type: str = "quick",
        locale: str = "en",
    ) -> DigestSummaryResponse | None:
        """Lookup stored summary without triggering external fetch or AI generation."""
        st = normalize_summary_type(summary_type)
        eff_locale = effective_locale_for_summary(locale, st)
        fp = digest_fingerprint(
            feed_provider=self.source.provider_id,
            external_id=str(external_id),
            locale=eff_locale,
            prompt_version=PROMPT_VERSION,
            summary_type=st,
        )

        art = await self.storage.get_artifact_by_fingerprint(fp)
        if art:
            return DigestSummaryResponse(
                artifact=art,
                completion=DigestCompletion(status="cached", cache_layer="sqlite", hydrated=True),
            )
        return None

    async def get_artifact(self, *, artifact_id: str) -> DigestArtifact | None:
        """Retrieve stored artifact by UUID."""
        return await self.storage.get_artifact_by_id(artifact_id)

    async def summarize(
        self,
        *,
        external_id: str,
        summary_type: str = "quick",
        locale: str = "en",
        user_id: str = "standalone",
        model: str | None = None,
        force_regenerate: bool = False,
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

        use_model = model or self.default_model
        t0 = time.perf_counter()

        summary_markdown = await self._generate_summary(
            model=use_model,
            locale=eff_locale,
            content=content_to_summarize,
            summary_type=st,
            source_fetch_status=source_fetch_status,
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
        """Fetch article HTML or fallback via Jina reader."""
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
            html = _prefer_main_html(r.text)
            direct_body = _strip_html(html)
            direct_ok = True
        except Exception:
            pass

        if direct_ok and len(direct_body) >= 500:
            return f"Title: {title}\nContent: {direct_body}", "ok"

        if self.enable_jina_fallback:
            jina_text = await self._fetch_via_jina_reader(url)
            if jina_text and len(jina_text) >= 280:
                return f"Title: {title}\nContent: {jina_text}", "ok"

        if direct_ok and len(direct_body) >= 80:
            return f"Title: {title}\nContent: {direct_body}", "ok"

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
    ) -> str:
        """Call LLM summarizer using custom callback or OpenAI client."""
        if self.custom_summarizer is not None:
            return await self.custom_summarizer(
                model=model,
                locale=locale,
                content=content,
                summary_type=summary_type,
                source_fetch_status=source_fetch_status,
            )

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide an API key or a custom_summarizer function."
            )

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        system_prompt = build_digest_system_prompt(summary_type=summary_type, locale=locale)
        user_message = build_digest_user_message(content, source_fetch_status=source_fetch_status)

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=4096,
        )

        if not resp.choices or not resp.choices[0].message.content:
            raise RuntimeError("Summarization model returned an empty response.")

        return resp.choices[0].message.content.strip()
