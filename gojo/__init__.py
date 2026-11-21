"""
Gojo Standalone — Production-grade Hacker News workflow and agent tool engine.
Extracted and refined from patzelife-plugin-agentdiscord.
"""

from .agent_tools import (
    HN_AGENT_TOOL_DEFINITIONS,
    GojoAgentToolkit,
    format_archive_search_for_agent,
    format_deep_dive_for_agent,
    format_radar_for_agent,
    format_search_for_agent,
    format_summary_for_agent,
    format_topic_pulse_for_agent,
    tool_briefing_hn,
    tool_deep_dive_hn,
    tool_get_artifact,
    tool_get_hn_story,
    tool_radar_hn,
    tool_search_archive,
    tool_search_hn,
    tool_search_hn_full,
    tool_summarize_hn,
    tool_topic_pulse,
)
from .mcp_server import create_mcp_server, run_mcp_server
from .providers.algolia import HackerNewsAlgoliaProvider
from .providers.github_trending import GitHubTrendingFeedSource
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
from .providers.lobsters import LobstersFeedSource
from .service import GojoHNService
from .signals import (
    annotate_cross_source_clusters,
    build_adaptive_focus_profile,
    cluster_related_stories,
    compute_story_signals,
    enrich_items_with_signals,
    profile_to_focus_query,
    rank_items_by_focus,
)
from .types import (
    ALL_SUPPORTED_FEED_IDS,
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
    "LobstersFeedSource",
    "GitHubTrendingFeedSource",
    "SUPPORTED_FEED_IDS",
    "SUPPORTED_FEED_VARIANTS",
    "ALL_SUPPORTED_FEED_IDS",
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
    "annotate_cross_source_clusters",
    "rank_items_by_focus",
    "build_adaptive_focus_profile",
    "profile_to_focus_query",
    "HN_AGENT_TOOL_DEFINITIONS",
    "GojoAgentToolkit",
    "tool_search_hn",
    "tool_search_hn_full",
    "tool_radar_hn",
    "tool_topic_pulse",
    "tool_summarize_hn",
    "tool_deep_dive_hn",
    "tool_briefing_hn",
    "tool_get_hn_story",
    "tool_get_artifact",
    "tool_search_archive",
    "format_search_for_agent",
    "format_radar_for_agent",
    "format_topic_pulse_for_agent",
    "format_summary_for_agent",
    "format_deep_dive_for_agent",
    "format_archive_search_for_agent",
    "create_mcp_server",
    "run_mcp_server",
]
