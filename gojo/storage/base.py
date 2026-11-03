"""
Abstract storage backend interface for artifacts and feed cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import DigestArtifact


class StorageBackend(ABC):
    """Storage contract for artifact persistence and feed caching."""

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
    async def get_feed_cache(self, feed_id: str, page: int) -> dict[str, Any] | None:
        """Retrieve cached feed payload if still within TTL."""

    @abstractmethod
    async def set_feed_cache(
        self, feed_id: str, page: int, data: dict[str, Any], ttl_seconds: int
    ) -> None:
        """Store feed payload with TTL in seconds."""

    @abstractmethod
    async def list_recent_artifacts(self, limit: int = 20) -> list[DigestArtifact]:
        """List recently summarized artifacts."""
