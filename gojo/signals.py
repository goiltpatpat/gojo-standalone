"""
Six Eyes Signal Engine — Pure-Python, zero-dependency intelligence primitives:
- Gravity-normalized velocity, debate ratio, and deterministic signal badges
- URL canonicalization (tracking-parameter & fragment stripping)
- 64-bit deterministic SimHash & Hamming distance for near-duplicate detection
- Event-level story clustering (URL match, Jaccard overlap, SimHash proximity)
- Local Okapi BM25 focus-profile relevance ranking (zero LLM token cost)
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .types import DigestFeedItem

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "for",
        "from",
        "get",
        "has",
        "have",
        "hn",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "more",
        "my",
        "new",
        "no",
        "not",
        "now",
        "of",
        "on",
        "one",
        "or",
        "our",
        "out",
        "over",
        "show",
        "ask",
        "tell",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "us",
        "use",
        "using",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "ref",
        "ref_src",
        "ref_url",
        "fbclid",
        "gclid",
        "yclid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "si",
        "mkt_tok",
        "_hsenc",
        "_hsmi",
        "spm",
    }
)


def _utc_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def compute_story_signals(
    item: DigestFeedItem,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Compute HN gravity-normalized velocity, debate_ratio, and deterministic signal_badges.

    Formulas:
      - velocity = max(score - 1, 0) / ((age_hours + 2.0) ** 1.8)
      - debate_ratio = descendants / max(score, 1)
    """
    ref_now = _utc_now(now)
    score = max(int(item.score or 0), 0)
    descendants = max(int(item.descendants or 0), 0)

    if item.time is not None:
        age_hours = max((ref_now.timestamp() - float(item.time)) / 3600.0, 0.0)
    else:
        age_hours = 0.0

    velocity = max(score - 1, 0) / ((age_hours + 2.0) ** 1.8)
    debate_ratio = descendants / max(score, 1)

    badges: list[str] = []
    if velocity >= 15.0:
        badges.append("breakout")
    elif velocity >= 5.0:
        badges.append("rising")

    if descendants >= 20 and debate_ratio >= 0.75:
        badges.append("high-debate")

    if age_hours >= 12.0 and (score >= 100 or descendants >= 50) and "breakout" not in badges and "rising" not in badges:
        badges.append("slow-burn")

    return {
        "velocity": round(velocity, 4),
        "debate_ratio": round(debate_ratio, 4),
        "age_hours": round(age_hours, 4),
        "signal_badges": badges,
    }


def enrich_items_with_signals(
    items: list[DigestFeedItem],
    now: datetime | None = None,
) -> list[DigestFeedItem]:
    """Return copies of feed items populated with velocity, debate_ratio, and signal_badges."""
    ref_now = _utc_now(now)
    enriched: list[DigestFeedItem] = []
    for item in items:
        sig = compute_story_signals(item, now=ref_now)
        enriched.append(
            item.model_copy(
                update={
                    "velocity": sig["velocity"],
                    "debate_ratio": sig["debate_ratio"],
                    "signal_badges": sig["signal_badges"],
                }
            )
        )
    return enriched


def canonicalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication:
    - lowercase scheme and host
    - strip default ports (:80, :443)
    - strip tracking query params (utm_*, ref, fbclid, gclid, etc.)
    - sort remaining query params deterministically
    - strip fragments and trailing path slashes
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if not scheme or not hostname:
        return raw.rstrip("/")

    port = parts.port
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = re.sub(r"/+", "/", parts.path or "").rstrip("/")

    kept_params: list[tuple[str, str]] = []
    if parts.query:
        for k, v in parse_qsl(parts.query, keep_blank_values=False):
            kl = k.lower()
            if kl.startswith("utm_") or kl in _TRACKING_PARAMS:
                continue
            kept_params.append((k, v))
        kept_params.sort(key=lambda kv: (kv[0].lower(), kv[0], kv[1]))

    query = urlencode(kept_params, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]
    if drop_stopwords:
        return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
    return tokens


def simhash64(text: str) -> int:
    """
    Compute a deterministic 64-bit Charikar SimHash fingerprint for near-duplicate detection.
    Combines normalized word unigrams and 4-char sliding shingles with BLAKE2b-64 so short
    headlines remain stable under minor insertions or rewordings across processes.
    """
    tokens = _tokenize(text, drop_stopwords=False)
    if not tokens:
        return 0

    features: Counter[str] = Counter()
    for tok in tokens:
        features[f"w:{tok}"] += 3

    joined = " ".join(tokens)
    if len(joined) >= 4:
        for i in range(len(joined) - 3):
            features[f"c:{joined[i : i + 4]}"] += 1

    vector = [0] * 64
    for feat, weight in features.items():
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        for bit in range(64):
            if (h >> bit) & 1:
                vector[bit] += weight
            else:
                vector[bit] -= weight

    fingerprint = 0
    for bit in range(64):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Compute the 64-bit Hamming distance between two SimHash integers."""
    mask = (1 << 64) - 1
    return ((a ^ b) & mask).bit_count()


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def cluster_related_stories(
    items: list[DigestFeedItem],
    max_hamming: int = 12,
    jaccard_threshold: float = 0.45,
) -> list[list[DigestFeedItem]]:
    """
    Group duplicate or related stories covering the same event while preserving rank order.

    Matching criteria (any triggers cluster merge):
      1. Canonical URL equality (when both items have non-empty URLs)
      2. Significant title-token Jaccard overlap >= jaccard_threshold (>= 2 significant tokens each)
      3. SimHash Hamming distance <= max_hamming on significant title tokens (>= 3 tokens each)
    """
    if not items:
        return []

    clusters: list[list[DigestFeedItem]] = []
    cluster_meta: list[list[tuple[str, set[str], int, int]]] = []

    for item in items:
        canon_url = canonicalize_url(item.url or "")
        sig_tokens = _tokenize(item.title or "", drop_stopwords=True)
        token_set = set(sig_tokens)
        sig_hash = simhash64(" ".join(sig_tokens)) if sig_tokens else 0
        sig_len = len(sig_tokens)

        matched_idx: int | None = None
        for idx, members_meta in enumerate(cluster_meta):
            for m_url, m_tokens, m_hash, m_len in members_meta:
                if canon_url and m_url and canon_url == m_url:
                    matched_idx = idx
                    break
                if len(token_set) >= 2 and len(m_tokens) >= 2:
                    if _jaccard_similarity(token_set, m_tokens) >= jaccard_threshold:
                        matched_idx = idx
                        break
                if sig_len >= 3 and m_len >= 3:
                    if hamming_distance(sig_hash, m_hash) <= max_hamming:
                        matched_idx = idx
                        break
            if matched_idx is not None:
                break

        meta_entry = (canon_url, token_set, sig_hash, sig_len)
        if matched_idx is not None:
            clusters[matched_idx].append(item)
            cluster_meta[matched_idx].append(meta_entry)
        else:
            clusters.append([item])
            cluster_meta.append([meta_entry])

    return clusters


def _extract_item_doc_tokens(item: DigestFeedItem) -> list[str]:
    """Build weighted token list from title (2x), URL host/path, and body text."""
    title_tokens = _tokenize(item.title or "", drop_stopwords=False)
    url_tokens: list[str] = []
    if item.url:
        parts = urlsplit(item.url)
        host = (parts.hostname or "").lower().removeprefix("www.")
        url_tokens = _tokenize(f"{host} {parts.path}", drop_stopwords=False)
    text_tokens = _tokenize(item.text or "", drop_stopwords=False)
    # Weight title tokens twice so headline relevance dominates noisy URLs/bodies
    return title_tokens + title_tokens + url_tokens + text_tokens


def rank_items_by_focus(
    items: list[DigestFeedItem],
    focus_query: str,
) -> list[tuple[DigestFeedItem, float]]:
    """
    Pure-Python Okapi BM25 local relevance ranker (zero LLM token cost).
    Supports comma- or space-separated focus terms and preserves input order on ties.
    """
    if not items:
        return []

    raw_query = (focus_query or "").strip()
    if not raw_query:
        return [(item, 0.0) for item in items]

    # Support comma-separated phrases as well as whitespace-separated terms
    phrases = [p.strip().lower() for p in raw_query.split(",") if p.strip()]
    query_terms: list[str] = []
    for phrase in phrases:
        for tok in _tokenize(phrase, drop_stopwords=False):
            if tok not in _STOPWORDS or len(phrases) == 1:
                query_terms.append(tok)

    # Deduplicate while preserving order; fallback to raw tokens if all were stopwords
    if not query_terms:
        query_terms = _tokenize(raw_query, drop_stopwords=False)
    unique_terms = list(dict.fromkeys(query_terms))
    if not unique_terms:
        return [(item, 0.0) for item in items]

    docs = [_extract_item_doc_tokens(item) for item in items]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n_docs, 1)
    if avgdl <= 0.0:
        return [(item, 0.0) for item in items]

    doc_counters = [Counter(d) for d in docs]
    df: dict[str, int] = {
        term: sum(1 for c in doc_counters if c.get(term, 0) > 0)
        for term in unique_terms
    }

    k1 = 1.5
    b = 0.75
    scored: list[tuple[int, DigestFeedItem, float]] = []

    for idx, (item, tokens, counter) in enumerate(zip(items, docs, doc_counters)):
        dl = len(tokens)
        score = 0.0
        for term in unique_terms:
            tf = counter.get(term, 0)
            if tf <= 0:
                continue
            # Non-negative BM25+ style IDF so single-doc / all-matching corpora still score > 0
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * (tf * (k1 + 1.0)) / denom

        # Small exact multi-word phrase boost in title
        title_lower = (item.title or "").lower()
        for phrase in phrases:
            if " " in phrase and phrase in title_lower:
                score += 0.5

        scored.append((idx, item, round(score, 6)))

    scored.sort(key=lambda row: (-row[2], row[0]))
    return [(item, score) for _, item, score in scored]
