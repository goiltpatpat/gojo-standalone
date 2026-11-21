"""
In-memory storage backend for testing and zero-file ephemeral runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
import time
from typing import Any

from ..types import DigestArtifact, DigestFeedItem
from .base import StorageBackend


class MemoryStorage(StorageBackend):
    """Stores artifacts, feed cache, and story snapshots in process memory."""

    def __init__(self) -> None:
        self._artifacts_by_fp: dict[str, DigestArtifact] = {}
        self._artifacts_by_id: dict[str, DigestArtifact] = {}
        self._feed_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._snapshots: dict[str, list[dict[str, Any]]] = {}

    @property
    def layer_name(self) -> str:
        return "memory"

    async def get_artifact_by_fingerprint(self, fingerprint: str) -> DigestArtifact | None:
        return self._artifacts_by_fp.get(fingerprint)

    async def get_artifact_by_id(self, artifact_id: str) -> DigestArtifact | None:
        return self._artifacts_by_id.get(artifact_id)

    async def save_artifact(self, artifact: DigestArtifact) -> DigestArtifact:
        existing = self._artifacts_by_fp.get(artifact.fingerprint)
        if existing is not None and existing.id != artifact.id:
            self._artifacts_by_id.pop(existing.id, None)
        self._artifacts_by_fp[artifact.fingerprint] = artifact
        self._artifacts_by_id[artifact.id] = artifact
        return artifact

    async def get_feed_cache(
        self, feed_id: str, page: int, page_size: int = 15
    ) -> dict[str, Any] | None:
        key = f"{feed_id}:{page}:{page_size}"
        item = self._feed_cache.get(key)
        if not item:
            return None
        expires_at, data = item
        if time.time() > expires_at:
            del self._feed_cache[key]
            return None
        return data

    async def set_feed_cache(
        self,
        feed_id: str,
        page: int,
        data: dict[str, Any],
        ttl_seconds: int,
        page_size: int = 15,
    ) -> None:
        key = f"{feed_id}:{page}:{page_size}"
        expires_at = time.time() + ttl_seconds
        self._feed_cache[key] = (expires_at, data)

    async def prune_expired_feed_cache(self) -> int:
        now = time.time()
        expired = [k for k, (expires_at, _) in self._feed_cache.items() if now > expires_at]
        for k in expired:
            del self._feed_cache[k]
        return len(expired)

    async def list_recent_artifacts(self, limit: int = 20) -> list[DigestArtifact]:
        if limit <= 0:
            return []
        items = list(self._artifacts_by_id.values())
        return list(reversed(items))[:limit]

    async def search_artifacts(self, query: str, limit: int = 10) -> list[DigestArtifact]:
        if limit <= 0 or not query or not query.strip():
            return []
        terms = re.findall(r"\w+", query.lower())
        if not terms:
            return []

        scored: list[tuple[float, int, DigestArtifact]] = []
        artifacts = list(self._artifacts_by_id.values())
        for idx, art in enumerate(artifacts):
            title_l = (art.title or "").lower()
            summary_l = (art.summary_markdown or "").lower()
            url_l = (art.source_url or "").lower()
            combined = f"{title_l}\n{summary_l}\n{url_l}"
            if not all(term in combined for term in terms):
                continue

            score = 0.0
            for term in terms:
                score += title_l.count(term) * 3.0
                score += summary_l.count(term) * 1.0
                score += url_l.count(term) * 0.5
            scored.append((score, idx, art))

        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [art for _, _, art in scored[:limit]]

    async def record_story_snapshots(
        self, feed_id: str, items: list[DigestFeedItem]
    ) -> int:
        if not items:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        for rank, item in enumerate(items, start=1):
            ext_id = str(item.external_id)
            snap = {
                "external_id": ext_id,
                "feed_id": feed_id,
                "rank": rank,
                "score": item.score,
                "descendants": item.descendants,
                "seen_at": now_iso,
            }
            self._snapshots.setdefault(ext_id, []).append(snap)
        return len(items)

    async def get_latest_snapshots(
        self, external_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not external_ids:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for ext_id in external_ids:
            key = str(ext_id)
            history = self._snapshots.get(key)
            if history:
                latest = history[-1]
                result[key] = {
                    "rank": latest["rank"],
                    "score": latest["score"],
                    "descendants": latest["descendants"],
                    "seen_at": latest["seen_at"],
                    "feed_id": latest["feed_id"],
                }
        return result
