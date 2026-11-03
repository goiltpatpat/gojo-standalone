"""
In-memory storage backend for testing and zero-file ephemeral runs.
"""

from __future__ import annotations

import time
from typing import Any

from ..types import DigestArtifact
from .base import StorageBackend


class MemoryStorage(StorageBackend):
    """Stores artifacts and cache in process memory."""

    def __init__(self) -> None:
        self._artifacts_by_fp: dict[str, DigestArtifact] = {}
        self._artifacts_by_id: dict[str, DigestArtifact] = {}
        self._feed_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def get_artifact_by_fingerprint(self, fingerprint: str) -> DigestArtifact | None:
        return self._artifacts_by_fp.get(fingerprint)

    async def get_artifact_by_id(self, artifact_id: str) -> DigestArtifact | None:
        return self._artifacts_by_id.get(artifact_id)

    async def save_artifact(self, artifact: DigestArtifact) -> DigestArtifact:
        self._artifacts_by_fp[artifact.fingerprint] = artifact
        self._artifacts_by_id[artifact.id] = artifact
        return artifact

    async def get_feed_cache(self, feed_id: str, page: int) -> dict[str, Any] | None:
        key = f"{feed_id}:{page}"
        item = self._feed_cache.get(key)
        if not item:
            return None
        expires_at, data = item
        if time.time() > expires_at:
            del self._feed_cache[key]
            return None
        return data

    async def set_feed_cache(
        self, feed_id: str, page: int, data: dict[str, Any], ttl_seconds: int
    ) -> None:
        key = f"{feed_id}:{page}"
        expires_at = time.time() + ttl_seconds
        self._feed_cache[key] = (expires_at, data)

    async def list_recent_artifacts(self, limit: int = 20) -> list[DigestArtifact]:
        items = list(self._artifacts_by_id.values())
        return items[-limit:]
