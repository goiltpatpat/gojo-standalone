"""
Feed source abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import DigestFeedItem


class FeedSource(ABC):
    """Abstract base class for feed providers."""

    provider_id: str

    @abstractmethod
    async def list_page(
        self, *, feed_variant: str, page: int, page_size: int
    ) -> tuple[list[DigestFeedItem], bool]:
        """Return items for page (1-based) and whether more pages exist."""

    @abstractmethod
    async def get_item(self, *, external_id: str) -> dict[str, Any] | None:
        """Fetch raw item dict by external ID."""
