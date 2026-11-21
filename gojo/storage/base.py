"""
Abstract storage backend interface for artifacts and feed cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import DigestArtifact, DigestFeedItem


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
