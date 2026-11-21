"""
GojoHNService: Main orchestration engine for Hacker News feed fetching,
SSRF-safe hop-by-hop article content extraction with Trafilatura and Jina fallback,
community discussion synthesis via Algolia, multi-model AI summarization,
fingerprint deduplication, signal enrichment, and persistent caching.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import html as html_lib
import io
import os
import re
import time
from typing import Any, Callable, Coroutine
from urllib.parse import urljoin

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
from .providers.hn import SUPPORTED_FEED_IDS, HackerNewsFeedSource
from .providers.llm import (
    AutoLLMProvider,
    LLMProvider,
    OpenAIProvider,
)
from .signals import (
    cluster_related_stories,
    enrich_items_with_signals,
    rank_items_by_focus,
)
from .ssrf import is_public_http_url_for_fetch, validate_public_url_with_dns
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
MAX_REDIRECT_HOPS = 5
MAX_RESPONSE_BYTES = 2_000_000  # 2MB hard cap for streamed responses

_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_BINARY_CONTENT_PREFIXES = ("image/", "video/", "audio/", "font/")
_BINARY_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "application/zip",
        "application/gzip",
        "application/x-tar",
        "application/x-7z-compressed",
        "application/vnd.ms-excel",
        "application/wasm",
    }
)


def _strip_html(raw_html: str) -> str:
    """Strip script/style/iframe/img tags and HTML markup to retain readable plain text."""
    t = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.I)
    t = re.sub(r"<style[\s\S]*?</style>", " ", t, flags=re.I)
    t = re.sub(r"<iframe[\s\S]*?</iframe>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html_lib.unescape(t)
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


def _extract_trafilatura(raw_html: str, url: str) -> str:
    """Synchronous Trafilatura extraction helper (intended for asyncio.to_thread)."""
    import trafilatura

    extracted = trafilatura.extract(
        raw_html,
        url=url,
        output_format="markdown",
        include_links=False,
        include_tables=True,
        favor_precision=True,
        deduplicate=True,
    )
    return (extracted or "").strip()


def _extract_pdf_text(pdf_bytes: bytes) -> str | None:
    """
    Synchronous PDF text extraction using optional `pypdf`.
    Returns None if `pypdf` is not installed; returns extracted text (possibly empty) otherwise.
    """
    try:
        import pypdf
    except ImportError:
        return None

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts: list[str] = []
        total_chars = 0
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                parts.append(page_text)
                total_chars += len(page_text)
                if total_chars >= 25000:
                    break
        return "\n\n".join(parts)[:25000].strip()
    except Exception:
        return ""


def _normalize_markdown_whitespace(text: str) -> str:
    """Normalize Markdown text while preserving paragraph and line breaks."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


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

        env_model = os.getenv("GOJO_HN_MODEL")
        self._explicit_default_model: str | None = default_model or (
            env_model.strip() if env_model and env_model.strip() else None
        )
        self.default_model = self._explicit_default_model or "gpt-4o-mini"

        if llm_provider is not None:
            self.llm_provider = llm_provider
        elif openai_api_key:
            self.llm_provider = OpenAIProvider(api_key=openai_api_key)
        else:
            self.llm_provider = AutoLLMProvider()

        jina_env = os.getenv("GOJO_HN_JINA_FALLBACK", "true").lower() in ("true", "1", "yes")
        self.enable_jina_fallback = (
            enable_jina_fallback if enable_jina_fallback is not None else jina_env
        )
        self.custom_summarizer = custom_summarizer

        self._article_http: httpx.AsyncClient | None = None

    async def _article_client(self) -> httpx.AsyncClient:
        if self._article_http is None or self._article_http.is_closed:
            self._article_http = httpx.AsyncClient(
                timeout=PAGE_FETCH_TIMEOUT_S,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._article_http

    async def close(self) -> None:
        """Close all open HTTP client connections across article fetcher, feeds, Algolia, and LLM."""
        if self._article_http is not None and not self._article_http.is_closed:
            await self._article_http.aclose()
            self._article_http = None

        for resource in (self.source, self.algolia, self.llm_provider):
            close_fn = getattr(resource, "close", None)
            if callable(close_fn):
                res = close_fn()
                if asyncio.iscoroutine(res):
                    await res

    def _storage_layer_name(self) -> str:
        layer = getattr(self.storage, "layer_name", None)
        if isinstance(layer, str) and layer.strip():
            return layer.strip()
        return "sqlite"

    def _resolve_requested_model(self, model: str | None) -> str | None:
        """Return explicit model override if provided by caller or GOJO_HN_MODEL, else None."""
        if model is not None and str(model).strip():
            return str(model).strip()
        return self._explicit_default_model

    def _resolve_recorded_model(self, requested_model: str | None) -> str:
        """Return actual model reported by LLM provider, falling back to requested/default model."""
        if self.custom_summarizer is None:
            last_used = getattr(self.llm_provider, "last_used_model", None)
            if isinstance(last_used, str) and last_used.strip():
                return last_used.strip()
        return requested_model or self.default_model

    async def list_feed(
        self,
        *,
        feed_id: str = "hn_top",
        page: int = 1,
        page_size: int = PAGE_SIZE,
        force_refresh: bool = False,
        _record_snapshots: bool = True,
    ) -> DigestFeedResponse:
        """
        List stories for feed_id ('hn_top', 'hn_new', 'hn_best', 'hn_ask', 'hn_show').
        Returns cached list if present and unexpired unless force_refresh is True.
        """
        if feed_id not in SUPPORTED_FEED_IDS:
            allowed = ", ".join(f"'{fid}'" for fid in SUPPORTED_FEED_IDS)
            raise ValueError(f"Invalid feed_id '{feed_id}'. Must be one of: {allowed}.")

        page = max(1, page)
        page_size = max(1, page_size)
        if not force_refresh:
            cached = await self.storage.get_feed_cache(feed_id, page, page_size=page_size)
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
                    completion=DigestCompletion(
                        status="success", cache_layer=self._storage_layer_name()
                    ),
                    feed_refreshed_at=ref_at,
                    list_cache_ttl_seconds=LIST_CACHE_TTL_S,
                )

        variant = feed_id.removeprefix("hn_")
        raw_items, has_more = await self.source.list_page(
            feed_variant=variant, page=page, page_size=page_size
        )
        items = enrich_items_with_signals(raw_items)

        if _record_snapshots and items:
            try:
                await self.storage.record_story_snapshots(feed_id, items)
            except Exception:
                pass

        now = datetime.now(timezone.utc)
        payload = {
            "feed_id": feed_id,
            "page": page,
            "has_more": has_more,
            "refreshed_at": now.isoformat(),
            "items": [it.model_dump(mode="json") for it in items],
        }
        await self.storage.set_feed_cache(
            feed_id, page, payload, ttl_seconds=LIST_CACHE_TTL_S, page_size=page_size
        )

        return DigestFeedResponse(
            feed_id=feed_id,
            page=page,
            items=items,
            has_more=has_more,
            completion=DigestCompletion(status="success", cache_layer="origin"),
            feed_refreshed_at=now,
            list_cache_ttl_seconds=LIST_CACHE_TTL_S,
        )

    async def search_archive(self, query: str, limit: int = 10) -> list[DigestArtifact]:
        """Search locally cached/summarized digest artifacts using storage FTS5 / text search."""
        return await self.storage.search_artifacts(query, limit=limit)

    async def search_hn_archive(
        self,
        query: str,
        limit: int = 20,
        sort_by: str = "relevance",
    ) -> list[DigestFeedItem]:
        """Search historical Hacker News stories via Algolia and enrich with Six Eyes signals."""
        items = await self.algolia.search_stories(query, limit=limit, sort_by=sort_by)
        return enrich_items_with_signals(items)

    async def get_artifact(self, artifact_id: str) -> DigestArtifact | None:
        """Retrieve a saved digest artifact by its primary key UUID."""
        return await self.storage.get_artifact_by_id(artifact_id)

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
                        status="cached",
                        cache_layer=self._storage_layer_name(),
                        hydrated=False,
                    ),
                )

        raw = await self.source.get_item(external_id=str(external_id))
        if not raw:
            raise ValueError(f"Hacker News story ID '{external_id}' not found.")

        title = str(raw.get("title") or "")
        source_url = raw.get("url")

        async def _resolve_article() -> tuple[str, str]:
            if source_url and isinstance(source_url, str):
                if is_public_http_url_for_fetch(source_url):
                    return await self._fetch_article_text(title, source_url)
                return (
                    f"{title}\n(URL not fetched for security; summarize from title and context).",
                    "blocked",
                )
            if raw.get("text"):
                cleaned_text = _strip_html(str(raw.get("text")))
                return f"{title}\nContent: {cleaned_text}", "ok"
            return title, "fallback_title"

        async def _resolve_discussion() -> str:
            if not include_comments:
                return ""
            try:
                return await self.algolia.fetch_discussion_context(str(external_id))
            except Exception:
                return ""

        (content_to_summarize, source_fetch_status), discussion_context = await asyncio.gather(
            _resolve_article(),
            _resolve_discussion(),
        )

        requested_model = self._resolve_requested_model(model)
        t0 = time.perf_counter()

        summary_markdown = await self._generate_summary(
            model=requested_model,
            locale=eff_locale,
            content=content_to_summarize,
            summary_type=st,
            source_fetch_status=source_fetch_status,
            discussion_context=discussion_context,
        )
        generated_ms = int((time.perf_counter() - t0) * 1000)
        recorded_model = self._resolve_recorded_model(requested_model)
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
            model=recorded_model,
            summary_markdown=summary_markdown,
            summary_type=st,
            summary_extra=summary_extra,
            content_length=content_length,
            generated_ms=generated_ms,
            source_fetch_status=source_fetch_status,
            model_version=recorded_model,
            created_at=datetime.now(timezone.utc),
            created_by_user_id=user_id,
        )

        saved = await self.storage.save_artifact(artifact)
        return DigestSummaryResponse(
            artifact=saved,
            completion=DigestCompletion(status="success", cache_layer="origin", hydrated=False),
        )

    async def _safe_fetch_response(
        self,
        url: str,
        *,
        headers: dict[str, str],
        max_hops: int = MAX_REDIRECT_HOPS,
        max_bytes: int = MAX_RESPONSE_BYTES,
        timeout: httpx.Timeout | float | None = None,
    ) -> tuple[bytes, str, str, str]:
        """
        Execute an SSRF-hardened GET request with manual hop-by-hop redirect validation
        and byte-capped streaming.

        Returns:
            (body_bytes, content_type, final_url, status)
            where status is 'ok', 'blocked', or 'error'.
        """
        current_url = url
        client = await self._article_client()

        for hop in range(max_hops + 1):
            is_safe, _reason = await validate_public_url_with_dns(current_url)
            if not is_safe:
                return b"", "", current_url, "blocked"

            stream_kwargs: dict[str, Any] = {"headers": headers}
            if timeout is not None:
                stream_kwargs["timeout"] = timeout

            try:
                async with client.stream("GET", current_url, **stream_kwargs) as r:
                    if r.status_code in _REDIRECT_STATUS_CODES:
                        location = r.headers.get("location")
                        if not location or hop >= max_hops:
                            return b"", "", current_url, "error"
                        current_url = urljoin(current_url, location)
                        continue

                    r.raise_for_status()
                    content_type = (
                        (r.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
                    )

                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in r.aiter_bytes():
                        if not chunk:
                            continue
                        remaining = max_bytes - total
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunks.append(chunk[:remaining])
                            total += remaining
                            break
                        chunks.append(chunk)
                        total += len(chunk)

                    return b"".join(chunks), content_type, current_url, "ok"
            except Exception:
                return b"", "", current_url, "error"

        return b"", "", current_url, "error"

    async def _fetch_article_text(self, title: str, url: str) -> tuple[str, str]:
        """
        Fetch article content with 4-tier degradation:
        Tier 1: Trafilatura (offloaded to thread)
        Tier 2: Semantic HTML (<article>/<main> + _strip_html)
        Tier 3: Jina Reader fallback (newline-preserving Markdown)
        Tier 4: Title fallback
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        body_bytes, content_type, final_url, fetch_status = await self._safe_fetch_response(
            url,
            headers=headers,
            max_hops=MAX_REDIRECT_HOPS,
            max_bytes=MAX_RESPONSE_BYTES,
        )

        if fetch_status == "blocked":
            return (
                f"{title}\n(URL not fetched for security; summarize from title and context).",
                "blocked",
            )

        direct_body = ""
        direct_ok = False

        if fetch_status == "ok":
            # Content-Type: PDF
            is_pdf = content_type == "application/pdf" or (
                not content_type and final_url.lower().endswith(".pdf")
            )
            if is_pdf:
                pdf_text = await asyncio.to_thread(_extract_pdf_text, body_bytes)
                if pdf_text and len(pdf_text) >= 80:
                    return f"Title: {title}\nContent: {pdf_text[:25000]}", "ok"
                return (
                    f"{title}\n(PDF document detected; text extraction unavailable — "
                    "install optional 'pypdf' dependency or summarize from title and discussion.)",
                    "fallback_title",
                )

            # Content-Type: Non-text binary assets (avoid feeding binary garbage to LLM)
            if content_type.startswith(_BINARY_CONTENT_PREFIXES) or content_type in _BINARY_CONTENT_TYPES:
                return (
                    f"{title}\n(Non-text content type '{content_type}' skipped; "
                    "summarize from title and discussion.)",
                    "fallback_title",
                )

            # Content-Type: Plain text / Markdown / JSON
            if content_type in ("text/plain", "text/markdown", "application/json"):
                decoded_plain = _normalize_markdown_whitespace(
                    body_bytes.decode("utf-8", errors="replace")
                )
                if len(decoded_plain) >= 80:
                    return f"Title: {title}\nContent: {decoded_plain[:25000]}", "ok"

            raw_html = body_bytes.decode("utf-8", errors="replace")

            # Tier 1: Trafilatura (structural Markdown content extraction, offloaded to thread)
            try:
                trafil_text = await asyncio.to_thread(_extract_trafilatura, raw_html, final_url)
                if trafil_text and len(trafil_text) >= 250:
                    direct_body = trafil_text
                    direct_ok = True
            except Exception:
                pass

            # Tier 2: Prefer main/article tag + cleaned HTML strip
            if not direct_ok:
                html_slice = _prefer_main_html(raw_html)
                direct_body = _strip_html(html_slice)
                direct_ok = True

        if direct_ok and len(direct_body) >= 500:
            return f"Title: {title}\nContent: {direct_body[:25000]}", "ok"

        # Tier 3: Jina Reader fallback
        if self.enable_jina_fallback:
            jina_text = await self._fetch_via_jina_reader(url)
            if jina_text and len(jina_text) >= 280:
                return f"Title: {title}\nContent: {jina_text[:25000]}", "ok"

        if direct_ok and len(direct_body) >= 80:
            return f"Title: {title}\nContent: {direct_body[:25000]}", "ok"

        # Tier 4: Title fallback
        return f"{title}\n(Could not fetch page content; summarize from title.)", "error"

    async def _fetch_via_jina_reader(self, article_url: str) -> str:
        """Fallback to Jina Reader (https://r.jina.ai/) for reader-mode Markdown (preserving newlines)."""
        if not is_public_http_url_for_fetch(article_url):
            return ""
        jina_url = f"https://r.jina.ai/{article_url}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/plain,text/markdown,*/*;q=0.8",
        }
        try:
            body_bytes, _ct, _final, status = await self._safe_fetch_response(
                jina_url,
                headers=headers,
                max_hops=MAX_REDIRECT_HOPS,
                max_bytes=MAX_RESPONSE_BYTES,
                timeout=httpx.Timeout(20.0, connect=10.0),
            )
            if status != "ok" or not body_bytes:
                return ""
            text = _normalize_markdown_whitespace(body_bytes.decode("utf-8", errors="replace"))
            if len(text) < 80:
                return ""
            return text[:20000]
        except Exception:
            return ""

    async def _generate_summary(
        self,
        *,
        model: str | None,
        locale: str,
        content: str,
        summary_type: SummaryType,
        source_fetch_status: str,
        discussion_context: str | None = None,
    ) -> str:
        """Call LLM summarizer using custom callback or configured LLM provider."""
        if self.custom_summarizer is not None:
            effective_custom_model = model or self.default_model
            try:
                return await self.custom_summarizer(
                    model=effective_custom_model,
                    locale=locale,
                    content=content,
                    summary_type=summary_type,
                    source_fetch_status=source_fetch_status,
                    discussion_context=discussion_context,
                )
            except TypeError:
                return await self.custom_summarizer(
                    model=effective_custom_model,
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

    async def _lookup_cached_takeaway(self, external_id: str, locale: str) -> str | None:
        """Check storage for an already-summarized artifact takeaway for briefing enrichment."""
        locales_to_try = [locale]
        if locale != "en":
            locales_to_try.append("en")

        for loc in locales_to_try:
            for st in ("quick", "technical", "bullet", "thai_concise"):
                try:
                    art = await self.get_artifact_by_external_id(
                        external_id, summary_type=st, locale=loc
                    )
                except Exception:
                    art = None
                if art:
                    takeaway = (art.summary_extra or {}).get("key_takeaway")
                    if isinstance(takeaway, str) and takeaway.strip():
                        return takeaway.strip()
                    # Fall back to first non-heading line of summary_markdown
                    for line in (art.summary_markdown or "").splitlines():
                        cleaned = line.strip()
                        if cleaned and not cleaned.startswith("#"):
                            return cleaned[:240]
        return None

    async def create_briefing(
        self,
        *,
        feed_id: str = "hn_top",
        limit: int = 10,
        locale: str = "en",
        model: str | None = None,
        since_last: bool = False,
        focus: str | None = None,
    ) -> str:
        """
        Generate a multi-story executive technical briefing synthesizing top stories,
        enriched with signal badges, story clustering, cached takeaways, optional focus
        BM25 re-ranking, and optional since-last snapshot deltas.
        """
        limit = max(1, limit)
        fetch_size = max(limit, PAGE_SIZE)
        if focus and focus.strip():
            fetch_size = max(fetch_size, min(limit * 2, 50))

        feed = await self.list_feed(
            feed_id=feed_id,
            page=1,
            page_size=fetch_size,
            _record_snapshots=not since_last,
        )
        candidates = list(feed.items)
        if not candidates:
            return "No stories available to create briefing."

        if focus and focus.strip():
            ranked = rank_items_by_focus(candidates, focus.strip())
            items = [it for it, _score in ranked][:limit]
        else:
            items = candidates[:limit]

        prior_snapshots: dict[str, dict[str, Any]] = {}
        if since_last:
            try:
                prior_snapshots = await self.storage.get_latest_snapshots(
                    [it.external_id for it in items]
                )
            except Exception:
                prior_snapshots = {}
            # Record current sightings now that we have captured the prior baseline
            try:
                await self.storage.record_story_snapshots(feed_id, feed.items)
            except Exception:
                pass

        # Look up any cached takeaways concurrently
        takeaways_list = await asyncio.gather(
            *(self._lookup_cached_takeaway(it.external_id, locale) for it in items)
        )
        takeaways_by_id = {
            it.external_id: tk for it, tk in zip(items, takeaways_list) if tk
        }

        clusters = cluster_related_stories(items)

        header = f"=== Hacker News Feed: {feed_id.upper()} (Top {len(items)} Stories, {len(clusters)} Clusters) ==="
        if focus and focus.strip():
            header += f"\nFocus Profile: {focus.strip()}"
        if since_last:
            header += "\nDelta Mode: Comparing against prior snapshot (NEW vs SURGING Δscore)"

        context_lines = [header + "\n"]
        story_idx = 0
        for c_idx, cluster in enumerate(clusters, 1):
            if len(cluster) > 1:
                context_lines.append(
                    f"--- Corroborated Cluster {c_idx} ({len(cluster)} related stories) ---"
                )
            for it in cluster:
                story_idx += 1
                annotations: list[str] = []
                if it.signal_badges:
                    annotations.append(f"Signals: {', '.join(it.signal_badges)}")
                if since_last:
                    prev = prior_snapshots.get(it.external_id)
                    if prev is None:
                        annotations.append("NEW since last check")
                    else:
                        prev_score = int(prev.get("score") or 0)
                        cur_score = int(it.score or 0)
                        delta = cur_score - prev_score
                        if delta > 0:
                            annotations.append(f"SURGING Δ+{delta} pts")
                        elif delta < 0:
                            annotations.append(f"Δ{delta} pts")
                        else:
                            annotations.append("UNCHANGED")

                badge_str = f" [{' | '.join(annotations)}]" if annotations else ""
                block = (
                    f"{story_idx}. [{it.external_id}] {it.title}{badge_str}\n"
                    f"   Score: {it.score} pts | Comments: {it.descendants} | By: @{it.by or 'unknown'}\n"
                    f"   URL: {it.url or 'HN Discussion'}"
                )
                cached_tk = takeaways_by_id.get(it.external_id)
                if cached_tk:
                    block += f"\n   Cached Takeaway: {cached_tk}"
                context_lines.append(block + "\n")

        stories_context = "\n".join(context_lines)
        system_prompt = build_briefing_system_prompt(locale=locale)
        focus_note = (
            f" Prioritize insights relevant to the focus profile: '{focus.strip()}'."
            if focus and focus.strip()
            else ""
        )
        user_msg = (
            "Please synthesize the following top Hacker News developments into an authoritative "
            f"Executive Briefing.{focus_note}\n\n"
            f"{stories_context}"
        )

        requested_model = self._resolve_requested_model(model)
        return await self.llm_provider.complete(
            system_prompt=system_prompt,
            user_message=user_msg,
            model=requested_model,
            max_tokens=4096,
        )
