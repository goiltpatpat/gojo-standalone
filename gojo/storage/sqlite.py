"""
SQLite storage backend for standalone persistent workflows.
Stores artifacts, feed cache, FTS5 search index, and story snapshots in a local .db file.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from ..types import DigestArtifact, DigestFeedItem, artifact_row_to_model
from .base import StorageBackend

_SNAPSHOT_CHUNK_SIZE = 500


def _sanitize_fts_query(query: str) -> str | None:
    """Convert an arbitrary query string into a safe FTS5 MATCH expression."""
    if not query or not query.strip():
        return None
    terms: list[str] = []
    for raw_token in query.split():
        cleaned = raw_token.replace('"', "").strip()
        if cleaned and re.search(r"\w", cleaned):
            terms.append(f'"{cleaned}"')
    if not terms:
        return None
    return " ".join(terms)


def _row_to_artifact(row: sqlite3.Row) -> DigestArtifact:
    data = dict(row)
    if data.get("summary_extra"):
        try:
            data["summary_extra"] = json.loads(data["summary_extra"])
        except Exception:
            data["summary_extra"] = {}
    return artifact_row_to_model(data)


class SQLiteStorage(StorageBackend):
    """File-backed SQLite storage for standalone deployments."""

    def __init__(self, db_path: str | Path = "gojo_hn.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    @property
    def layer_name(self) -> str:
        return "sqlite"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        return self._connect()

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hn_artifacts (
                        id TEXT PRIMARY KEY,
                        fingerprint TEXT UNIQUE NOT NULL,
                        fingerprint_version TEXT DEFAULT 'v2',
                        feed_provider TEXT NOT NULL,
                        external_id TEXT NOT NULL,
                        title TEXT,
                        source_url TEXT,
                        locale TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        model TEXT,
                        summary_markdown TEXT NOT NULL,
                        summary_type TEXT DEFAULT 'quick',
                        summary_extra TEXT,
                        content_length INTEGER,
                        generated_ms INTEGER,
                        source_fetch_status TEXT,
                        model_version TEXT,
                        created_at TEXT,
                        created_by_user_id TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_hn_artifacts_fingerprint
                    ON hn_artifacts(fingerprint);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_hn_artifacts_ext_id
                    ON hn_artifacts(external_id);
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hn_feed_cache (
                        cache_key TEXT PRIMARY KEY,
                        feed_id TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS hn_artifacts_fts USING fts5(
                        title,
                        summary_markdown,
                        source_url,
                        content='hn_artifacts',
                        content_rowid='rowid'
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS hn_artifacts_ai
                    AFTER INSERT ON hn_artifacts BEGIN
                        INSERT INTO hn_artifacts_fts(rowid, title, summary_markdown, source_url)
                        VALUES (
                            new.rowid,
                            COALESCE(new.title, ''),
                            COALESCE(new.summary_markdown, ''),
                            COALESCE(new.source_url, '')
                        );
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS hn_artifacts_ad
                    AFTER DELETE ON hn_artifacts BEGIN
                        INSERT INTO hn_artifacts_fts(
                            hn_artifacts_fts, rowid, title, summary_markdown, source_url
                        )
                        VALUES (
                            'delete',
                            old.rowid,
                            COALESCE(old.title, ''),
                            COALESCE(old.summary_markdown, ''),
                            COALESCE(old.source_url, '')
                        );
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS hn_artifacts_au
                    AFTER UPDATE ON hn_artifacts BEGIN
                        INSERT INTO hn_artifacts_fts(
                            hn_artifacts_fts, rowid, title, summary_markdown, source_url
                        )
                        VALUES (
                            'delete',
                            old.rowid,
                            COALESCE(old.title, ''),
                            COALESCE(old.summary_markdown, ''),
                            COALESCE(old.source_url, '')
                        );
                        INSERT INTO hn_artifacts_fts(rowid, title, summary_markdown, source_url)
                        VALUES (
                            new.rowid,
                            COALESCE(new.title, ''),
                            COALESCE(new.summary_markdown, ''),
                            COALESCE(new.source_url, '')
                        );
                    END;
                    """
                )
                art_count = conn.execute("SELECT COUNT(*) FROM hn_artifacts;").fetchone()[0]
                fts_count = conn.execute(
                    "SELECT COUNT(*) FROM hn_artifacts_fts_docsize;"
                ).fetchone()[0]
                if art_count != fts_count:
                    conn.execute(
                        "INSERT INTO hn_artifacts_fts(hn_artifacts_fts) VALUES('rebuild');"
                    )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hn_story_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        external_id TEXT NOT NULL,
                        feed_id TEXT NOT NULL,
                        rank INTEGER NOT NULL,
                        score INTEGER,
                        descendants INTEGER,
                        seen_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_hn_story_snapshots_ext_seen
                    ON hn_story_snapshots(external_id, seen_at DESC, id DESC);
                    """
                )
        finally:
            conn.close()

    def _get_artifact_by_fingerprint_sync(self, fingerprint: str) -> DigestArtifact | None:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT * FROM hn_artifacts WHERE fingerprint = ? LIMIT 1;", (fingerprint,)
            )
            row = cur.fetchone()
            return _row_to_artifact(row) if row else None
        finally:
            conn.close()

    async def get_artifact_by_fingerprint(self, fingerprint: str) -> DigestArtifact | None:
        return await asyncio.to_thread(self._get_artifact_by_fingerprint_sync, fingerprint)

    def _get_artifact_by_id_sync(self, artifact_id: str) -> DigestArtifact | None:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT * FROM hn_artifacts WHERE id = ? LIMIT 1;", (artifact_id,)
            )
            row = cur.fetchone()
            return _row_to_artifact(row) if row else None
        finally:
            conn.close()

    async def get_artifact_by_id(self, artifact_id: str) -> DigestArtifact | None:
        return await asyncio.to_thread(self._get_artifact_by_id_sync, artifact_id)

    def _save_artifact_sync(self, artifact: DigestArtifact) -> DigestArtifact:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO hn_artifacts (
                        id, fingerprint, fingerprint_version, feed_provider, external_id,
                        title, source_url, locale, prompt_version, model,
                        summary_markdown, summary_type, summary_extra, content_length,
                        generated_ms, source_fetch_status, model_version, created_at, created_by_user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(fingerprint) DO UPDATE SET
                        title = excluded.title,
                        source_url = excluded.source_url,
                        summary_markdown = excluded.summary_markdown,
                        summary_extra = excluded.summary_extra,
                        model = excluded.model,
                        content_length = excluded.content_length,
                        generated_ms = excluded.generated_ms,
                        source_fetch_status = excluded.source_fetch_status,
                        model_version = excluded.model_version,
                        created_at = excluded.created_at,
                        created_by_user_id = excluded.created_by_user_id;
                    """,
                    (
                        artifact.id,
                        artifact.fingerprint,
                        artifact.fingerprint_version,
                        artifact.feed_provider,
                        artifact.external_id,
                        artifact.title,
                        artifact.source_url,
                        artifact.locale,
                        artifact.prompt_version,
                        artifact.model,
                        artifact.summary_markdown,
                        artifact.summary_type,
                        json.dumps(artifact.summary_extra, ensure_ascii=False),
                        artifact.content_length,
                        artifact.generated_ms,
                        artifact.source_fetch_status,
                        artifact.model_version,
                        artifact.created_at.isoformat() if artifact.created_at else None,
                        artifact.created_by_user_id,
                    ),
                )
            return artifact
        finally:
            conn.close()

    async def save_artifact(self, artifact: DigestArtifact) -> DigestArtifact:
        return await asyncio.to_thread(self._save_artifact_sync, artifact)

    def _get_feed_cache_sync(
        self, feed_id: str, page: int, page_size: int
    ) -> dict[str, Any] | None:
        key = f"{feed_id}:{page}:{page_size}"
        now = time.time()
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT payload, expires_at FROM hn_feed_cache WHERE cache_key = ?;", (key,)
            )
            row = cur.fetchone()
            if not row:
                return None
            if now > row["expires_at"]:
                with conn:
                    conn.execute("DELETE FROM hn_feed_cache WHERE cache_key = ?;", (key,))
                return None
            try:
                return json.loads(row["payload"])
            except Exception:
                return None
        finally:
            conn.close()

    async def get_feed_cache(
        self, feed_id: str, page: int, page_size: int = 15
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_feed_cache_sync, feed_id, page, page_size)

    def _set_feed_cache_sync(
        self,
        feed_id: str,
        page: int,
        data: dict[str, Any],
        ttl_seconds: int,
        page_size: int,
    ) -> None:
        key = f"{feed_id}:{page}:{page_size}"
        expires_at = time.time() + ttl_seconds
        payload = json.dumps(data, ensure_ascii=False)
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO hn_feed_cache (cache_key, feed_id, page, payload, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload = excluded.payload,
                        expires_at = excluded.expires_at;
                    """,
                    (key, feed_id, page, payload, expires_at),
                )
        finally:
            conn.close()

    async def set_feed_cache(
        self,
        feed_id: str,
        page: int,
        data: dict[str, Any],
        ttl_seconds: int,
        page_size: int = 15,
    ) -> None:
        await asyncio.to_thread(
            self._set_feed_cache_sync, feed_id, page, data, ttl_seconds, page_size
        )

    def _prune_expired_feed_cache_sync(self) -> int:
        now = time.time()
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM hn_feed_cache WHERE expires_at < ?;", (now,)
                )
                return max(0, cur.rowcount)
        finally:
            conn.close()

    async def prune_expired_feed_cache(self) -> int:
        return await asyncio.to_thread(self._prune_expired_feed_cache_sync)

    def _list_recent_artifacts_sync(self, limit: int) -> list[DigestArtifact]:
        if limit <= 0:
            return []
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT * FROM hn_artifacts ORDER BY rowid DESC LIMIT ?;", (limit,)
            )
            return [_row_to_artifact(r) for r in cur.fetchall()]
        finally:
            conn.close()

    async def list_recent_artifacts(self, limit: int = 20) -> list[DigestArtifact]:
        return await asyncio.to_thread(self._list_recent_artifacts_sync, limit)

    def _search_artifacts_sync(self, query: str, limit: int) -> list[DigestArtifact]:
        if limit <= 0:
            return []
        fts_query = _sanitize_fts_query(query)
        if not fts_query:
            return []
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT a.*
                FROM hn_artifacts_fts f
                JOIN hn_artifacts a ON a.rowid = f.rowid
                WHERE hn_artifacts_fts MATCH ?
                ORDER BY bm25(hn_artifacts_fts)
                LIMIT ?;
                """,
                (fts_query, limit),
            )
            return [_row_to_artifact(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    async def search_artifacts(self, query: str, limit: int = 10) -> list[DigestArtifact]:
        return await asyncio.to_thread(self._search_artifacts_sync, query, limit)

    def _record_story_snapshots_sync(
        self, feed_id: str, items: list[DigestFeedItem]
    ) -> int:
        if not items:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                str(item.external_id),
                feed_id,
                rank,
                item.score,
                item.descendants,
                now_iso,
            )
            for rank, item in enumerate(items, start=1)
        ]
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO hn_story_snapshots (
                        external_id, feed_id, rank, score, descendants, seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    rows,
                )
            return len(rows)
        finally:
            conn.close()

    async def record_story_snapshots(
        self, feed_id: str, items: list[DigestFeedItem]
    ) -> int:
        return await asyncio.to_thread(self._record_story_snapshots_sync, feed_id, items)

    def _get_latest_snapshots_sync(
        self, external_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not external_ids:
            return {}
        unique_ids = list(dict.fromkeys(str(eid) for eid in external_ids))
        result: dict[str, dict[str, Any]] = {}
        conn = self._connect()
        try:
            for start in range(0, len(unique_ids), _SNAPSHOT_CHUNK_SIZE):
                chunk = unique_ids[start : start + _SNAPSHOT_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"""
                    SELECT external_id, feed_id, rank, score, descendants, seen_at
                    FROM (
                        SELECT
                            external_id,
                            feed_id,
                            rank,
                            score,
                            descendants,
                            seen_at,
                            ROW_NUMBER() OVER (
                                PARTITION BY external_id
                                ORDER BY seen_at DESC, id DESC
                            ) AS rn
                        FROM hn_story_snapshots
                        WHERE external_id IN ({placeholders})
                    )
                    WHERE rn = 1;
                    """,
                    chunk,
                )
                for row in cur.fetchall():
                    result[str(row["external_id"])] = {
                        "rank": row["rank"],
                        "score": row["score"],
                        "descendants": row["descendants"],
                        "seen_at": row["seen_at"],
                        "feed_id": row["feed_id"],
                    }
            return result
        finally:
            conn.close()

    async def get_latest_snapshots(
        self, external_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(self._get_latest_snapshots_sync, external_ids)
