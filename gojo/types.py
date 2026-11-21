"""
Data types and schemas for Gojo Hacker News workflow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field

FeedProviderId = Literal["hn"]
DigestFeedId = Literal["hn_top", "hn_new"]
SummaryType = Literal["quick", "technical", "bullet", "thai_concise"]
DigestCacheLayer = Literal["memory", "sqlite", "origin"]


class DigestFeedItem(BaseModel):
    """Normalized Hacker News item representation."""

    feed_provider: str = "hn"
    external_id: str
    title: str | None = None
    url: str | None = None
    text: str | None = None
    score: int | None = None
    by: str | None = None
    time: int | None = None
    type: str | None = None
    descendants: int | None = None
    velocity: float | None = None
    debate_ratio: float | None = None
    signal_badges: list[str] = Field(default_factory=list)


FeedItem = DigestFeedItem


class DigestArtifact(BaseModel):
    """Durable generated summary artifact with stable fingerprint."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint: str
    fingerprint_version: str = "v2"
    feed_provider: str = "hn"
    external_id: str
    title: str | None = None
    source_url: str | None = None
    locale: str = "en"
    prompt_version: str = "v5"
    model: str | None = None
    summary_markdown: str
    summary_type: str = "quick"
    summary_extra: dict[str, Any] = Field(default_factory=dict)
    content_length: int | None = None
    generated_ms: int | None = None
    source_fetch_status: str | None = None
    model_version: str | None = None
    created_at: datetime | None = None
    created_by_user_id: str | None = None


class DigestCompletion(BaseModel):
    """Execution and cache metadata."""

    status: Literal["success", "cached"]
    cache_layer: DigestCacheLayer | None = None
    hydrated: bool | None = None


class DigestFeedResponse(BaseModel):
    """Paginated feed response."""

    feed_id: str
    page: int
    items: list[DigestFeedItem]
    has_more: bool
    completion: DigestCompletion
    feed_refreshed_at: datetime | None = None
    list_cache_ttl_seconds: int = 900


class DigestSummaryRequest(BaseModel):
    """Request payload for story summarization."""

    feed_provider: str = "hn"
    external_id: str
    locale: str = "en"
    summary_type: str = "quick"


class DigestSummaryResponse(BaseModel):
    """Response containing summary artifact and execution metadata."""

    artifact: DigestArtifact
    completion: DigestCompletion


def artifact_row_to_model(row: dict[str, Any]) -> DigestArtifact:
    """Helper to deserialize raw database/dict record to DigestArtifact."""
    extra = row.get("summary_extra")
    if not isinstance(extra, dict):
        extra = {}

    created = row.get("created_at")
    if isinstance(created, str):
        try:
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
            created = None

    return DigestArtifact(
        id=str(row.get("id")),
        fingerprint=str(row.get("fingerprint")),
        fingerprint_version=str(row.get("fingerprint_version") or "v2"),
        feed_provider=str(row.get("feed_provider") or "hn"),
        external_id=str(row.get("external_id")),
        title=row.get("title"),
        source_url=row.get("source_url"),
        locale=str(row.get("locale") or "en"),
        prompt_version=str(row.get("prompt_version") or "v5"),
        model=row.get("model"),
        summary_markdown=str(row.get("summary_markdown") or ""),
        summary_type=str(row.get("summary_type") or "quick"),
        summary_extra=extra,
        content_length=row.get("content_length") or len(str(row.get("summary_markdown") or "").encode("utf-8")),
        generated_ms=row.get("generated_ms"),
        source_fetch_status=row.get("source_fetch_status"),
        model_version=row.get("model_version"),
        created_at=created,
        created_by_user_id=row.get("created_by_user_id"),
    )
