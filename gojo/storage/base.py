"""
Abstract storage backend interface for artifacts and feed cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from ..types import DigestArtifact, DigestFeedItem


def parse_topic_clauses(topic: str) -> list[list[str]]:
    """Parse a topic string into OR-ed clauses of AND-ed lowercase terms.

    Examples:
        "rust" -> [["rust"]]
        "rust async" -> [["rust", "async"]]
        "rust, sqlite" -> [["rust"], ["sqlite"]]
    """
    if not topic or not topic.strip():
        return []
    clauses: list[list[str]] = []
    for raw_clause in topic.split(","):
        terms = [tok.strip().lower() for tok in raw_clause.split() if tok.strip()]
        if terms:
            clauses.append(terms)
    return clauses


def matches_topic_clauses(
    title: str | None, url: str | None, clauses: list[list[str]]
) -> bool:
    """Return True if any clause has all of its terms present in title or url."""
    if not clauses:
        return False
    haystack = f"{title or ''} {url or ''}".lower()
    return any(all(term in haystack for term in clause) for clause in clauses)


def parse_iso_utc(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_topic_timeline(
    topic: str,
    days: int,
    limit: int,
    snapshots: list[dict[str, Any]],
    artifacts: list[DigestArtifact],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate matching story snapshots and archived takeaways into a topic timeline."""
    clean_topic = (topic or "").strip()
    window_days = max(0, int(days))
    clauses = parse_topic_clauses(clean_topic)
    ref_now = now if now is not None else datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)
    else:
        ref_now = ref_now.astimezone(timezone.utc)

    archived_takeaways = [
        {
            "artifact_id": art.id,
            "external_id": art.external_id,
            "title": art.title or "",
            "created_at": art.created_at.isoformat() if art.created_at else None,
            "key_takeaway": str(art.summary_extra.get("key_takeaway") or ""),
            "community_consensus": str(art.summary_extra.get("community_consensus") or ""),
        }
        for art in artifacts[: max(0, min(limit, 15))]
    ]

    if not clauses or window_days <= 0 or limit <= 0:
        return {
            "topic": clean_topic,
            "window_days": window_days,
            "distinct_stories": 0,
            "total_snapshots": 0,
            "peak_score": 0,
            "avg_score": 0.0,
            "avg_debate_ratio": 0.0,
            "trend_direction": "insufficient_data",
            "timeline_stories": [],
            "archived_takeaways": archived_takeaways,
        }

    cutoff = ref_now - timedelta(days=window_days)
    midpoint = cutoff + timedelta(days=window_days / 2.0)

    matched: list[tuple[datetime, int, dict[str, Any]]] = []
    for seq, snap in enumerate(snapshots):
        seen_dt = parse_iso_utc(snap.get("seen_at"))
        if seen_dt is None or seen_dt < cutoff:
            continue
        if not matches_topic_clauses(snap.get("title"), snap.get("url"), clauses):
            continue
        matched.append((seen_dt, seq, snap))

    by_story: dict[str, list[tuple[datetime, int, dict[str, Any]]]] = {}
    all_scores: list[int] = []
    for item in matched:
        snap = item[2]
        ext_id = str(snap["external_id"])
        by_story.setdefault(ext_id, []).append(item)
        if snap.get("score") is not None:
            all_scores.append(int(snap["score"]))

    story_records: list[tuple[datetime, int, dict[str, Any]]] = []
    latest_scores: list[int] = []
    debate_ratios: list[float] = []
    older_scores: list[int] = []
    newer_scores: list[int] = []
    older_count = 0
    newer_count = 0

    for ext_id, entries in by_story.items():
        entries.sort(key=lambda e: (e[0], e[1]))
        first_dt, _, first_snap = entries[0]
        last_dt, _, last_snap = entries[-1]

        title = ""
        url = ""
        for _, _, s in reversed(entries):
            if not title and s.get("title"):
                title = str(s["title"])
            if not url and s.get("url"):
                url = str(s["url"])
            if title and url:
                break

        first_score = int(first_snap["score"]) if first_snap.get("score") is not None else None
        latest_score = int(last_snap["score"]) if last_snap.get("score") is not None else None
        latest_desc = (
            int(last_snap["descendants"]) if last_snap.get("descendants") is not None else None
        )
        score_delta = (
            (latest_score - first_score)
            if (latest_score is not None and first_score is not None)
            else 0
        )

        if latest_score is not None:
            latest_scores.append(latest_score)
            if latest_desc is not None:
                debate_ratios.append(max(0, latest_desc) / max(latest_score, 1))

        if last_dt >= midpoint:
            newer_count += 1
            if latest_score is not None:
                newer_scores.append(latest_score)
        else:
            older_count += 1
            if latest_score is not None:
                older_scores.append(latest_score)

        story_records.append(
            (
                last_dt,
                latest_score or 0,
                {
                    "external_id": ext_id,
                    "feed_id": str(last_snap.get("feed_id") or ""),
                    "title": title,
                    "url": url,
                    "first_seen": str(first_snap.get("seen_at") or first_dt.isoformat()),
                    "last_seen": str(last_snap.get("seen_at") or last_dt.isoformat()),
                    "first_score": first_score,
                    "latest_score": latest_score,
                    "score_delta": score_delta,
                    "latest_descendants": latest_desc,
                },
            )
        )

    distinct_stories = len(story_records)
    if distinct_stories < 2:
        trend_direction = "insufficient_data"
    elif older_count == 0:
        trend_direction = "rising"
    elif newer_count == 0:
        trend_direction = "cooling"
    else:
        older_avg = sum(older_scores) / len(older_scores) if older_scores else 0.0
        newer_avg = sum(newer_scores) / len(newer_scores) if newer_scores else 0.0
        if newer_count > older_count * 1.25 or (
            newer_count >= older_count and newer_avg > older_avg * 1.25
        ):
            trend_direction = "rising"
        elif newer_count < older_count * 0.75 or (
            newer_count <= older_count and newer_avg < older_avg * 0.75
        ):
            trend_direction = "cooling"
        else:
            trend_direction = "steady"

    story_records.sort(key=lambda rec: (rec[0], rec[1]), reverse=True)
    timeline_stories = [rec[2] for rec in story_records[:limit]]

    return {
        "topic": clean_topic,
        "window_days": window_days,
        "distinct_stories": distinct_stories,
        "total_snapshots": len(matched),
        "peak_score": max(all_scores) if all_scores else 0,
        "avg_score": round(sum(latest_scores) / len(latest_scores), 2) if latest_scores else 0.0,
        "avg_debate_ratio": (
            round(sum(debate_ratios) / len(debate_ratios), 3) if debate_ratios else 0.0
        ),
        "trend_direction": trend_direction,
        "timeline_stories": timeline_stories,
        "archived_takeaways": archived_takeaways,
    }


class StorageBackend(ABC):
    """Storage contract for artifact persistence, feed caching, search, and snapshots."""

    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Identifier for this storage backend layer (e.g., 'sqlite', 'memory')."""

    @abstractmethod
    async def get_artifact_by_fingerprint(self, fingerprint: str) -> DigestArtifact | None:
        """Retrieve stored artifact by SHA256 fingerprint."""

    @abstractmethod
    async def get_artifact_by_id(self, artifact_id: str) -> DigestArtifact | None:
        """Retrieve stored artifact by UUID."""

    @abstractmethod
    async def save_artifact(self, artifact: DigestArtifact) -> DigestArtifact:
        """Persist or update an artifact."""

    @abstractmethod
    async def get_feed_cache(
        self, feed_id: str, page: int, page_size: int = 15
    ) -> dict[str, Any] | None:
        """Retrieve cached feed payload if still within TTL."""

    @abstractmethod
    async def set_feed_cache(
        self,
        feed_id: str,
        page: int,
        data: dict[str, Any],
        ttl_seconds: int,
        page_size: int = 15,
    ) -> None:
        """Store feed payload with TTL in seconds."""

    @abstractmethod
    async def prune_expired_feed_cache(self) -> int:
        """Remove expired feed cache entries and return the number deleted."""

    @abstractmethod
    async def list_recent_artifacts(self, limit: int = 20) -> list[DigestArtifact]:
        """List recently summarized artifacts (newest first)."""

    @abstractmethod
    async def search_artifacts(self, query: str, limit: int = 10) -> list[DigestArtifact]:
        """Search stored artifacts by text query, ordered by relevance."""

    @abstractmethod
    async def record_story_snapshots(
        self, feed_id: str, items: list[DigestFeedItem]
    ) -> int:
        """Record lightweight story sighting snapshots and return count recorded."""

    @abstractmethod
    async def get_latest_snapshots(
        self, external_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Return the most recent prior snapshot per story external_id."""

    @abstractmethod
    async def get_topic_timeline(
        self, topic: str, *, days: int = 30, limit: int = 50
    ) -> dict[str, Any]:
        """Build a temporal memory timeline for a topic from snapshots and archived artifacts."""

    @abstractmethod
    async def get_recent_interest_corpus(self, *, limit: int = 40) -> list[dict[str, str]]:
        """Return recent summarized artifacts as a lightweight corpus for adaptive focus."""
