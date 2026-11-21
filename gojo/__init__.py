"""
Gojo Standalone — Production-grade Hacker News workflow and agent tool engine.
Extracted and refined from patzelife-plugin-agentdiscord.
"""

from .agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    format_archive_search_for_agent,
    format_search_for_agent,
    format_summary_for_agent,
    tool_briefing_hn,
    tool_get_artifact,
    tool_get_hn_story,
    tool_search_archive,
    tool_search_hn,
    tool_summarize_hn,
)
from .mcp_server import create_mcp_server, run_mcp_server
from .providers.algolia import HackerNewsAlgoliaProvider
from .providers.hn import (
    SUPPORTED_FEED_IDS,
    SUPPORTED_FEED_VARIANTS,
    HackerNewsFeedSource,
)
from .providers.llm import (
    AutoLLMProvider,
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
)
from .service import GojoHNService
from .signals import (
    cluster_related_stories,
    compute_story_signals,
    enrich_items_with_signals,
    rank_items_by_focus,
)
from .types import (
    DigestArtifact,
    DigestCompletion,
    DigestFeedItem,
    DigestFeedResponse,
    DigestSummaryRequest,
    DigestSummaryResponse,
    FeedItem,
    SummaryType,
)

__all__ = [
    "GojoHNService",
    "HackerNewsFeedSource",
    "SUPPORTED_FEED_IDS",
    "SUPPORTED_FEED_VARIANTS",
    "HackerNewsAlgoliaProvider",
    "AutoLLMProvider",
    "LLMProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "DigestArtifact",
    "DigestCompletion",
    "DigestFeedItem",
    "FeedItem",
    "DigestFeedResponse",
    "DigestSummaryRequest",
    "DigestSummaryResponse",
    "SummaryType",
    "compute_story_signals",
    "enrich_items_with_signals",
    "cluster_related_stories",
    "rank_items_by_focus",
    "HN_AGENT_TOOL_DEFINITIONS",
    "tool_search_hn",
    "tool_summarize_hn",
    "tool_briefing_hn",
    "tool_get_hn_story",
    "tool_get_artifact",
    "tool_search_archive",
    "format_search_for_agent",
    "format_summary_for_agent",
    "format_archive_search_for_agent",
    "create_mcp_server",
    "run_mcp_server",
]
