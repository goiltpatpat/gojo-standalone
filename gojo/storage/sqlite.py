"""
SQLite storage backend for standalone persistent workflows.
Stores artifacts and feed cache in a local .db file without external database servers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from ..types import DigestArtifact, artifact_row_to_model
from .base import StorageBackend


class SQLiteStorage(StorageBackend):
    """File-backed SQLite storage for standalone deployments."""

    def __init__(self, db_path: str | Path = "gojo_hn.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
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
                CREATE INDEX IF NOT EXISTS idx_hn_artifacts_fingerprint ON hn_artifacts(fingerprint);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hn_artifacts_ext_id ON hn_artifacts(external_id);
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
            conn.commit()

    async def get_artifact_by_fingerprint(self, fingerprint: str) -> DigestArtifact | None:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM hn_artifacts WHERE fingerprint = ? LIMIT 1;", (fingerprint,)
            )
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("summary_extra"):
                try:
                    data["summary_extra"] = json.loads(data["summary_extra"])
                except Exception:
                    data["summary_extra"] = {}
            return artifact_row_to_model(data)

    async def get_artifact_by_id(self, artifact_id: str) -> DigestArtifact | None:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM hn_artifacts WHERE id = ? LIMIT 1;", (artifact_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("summary_extra"):
                try:
                    data["summary_extra"] = json.loads(data["summary_extra"])
                except Exception:
                    data["summary_extra"] = {}
            return artifact_row_to_model(data)

    async def save_artifact(self, artifact: DigestArtifact) -> DigestArtifact:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO hn_artifacts (
                    id, fingerprint, fingerprint_version, feed_provider, external_id,
                    title, source_url, locale, prompt_version, model,
                    summary_markdown, summary_type, summary_extra, content_length,
                    generated_ms, source_fetch_status, model_version, created_at, created_by_user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    summary_markdown = excluded.summary_markdown,
                    summary_extra = excluded.summary_extra,
                    model = excluded.model,
                    generated_ms = excluded.generated_ms;
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
            conn.commit()
        return artifact

    async def get_feed_cache(self, feed_id: str, page: int) -> dict[str, Any] | None:
        key = f"{feed_id}:{page}"
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT payload, expires_at FROM hn_feed_cache WHERE cache_key = ?;", (key,)
            )
            row = cur.fetchone()
            if not row:
                return None
            if now > row["expires_at"]:
                conn.execute("DELETE FROM hn_feed_cache WHERE cache_key = ?;", (key,))
                conn.commit()
                return None
            try:
                return json.loads(row["payload"])
            except Exception:
                return None

    async def set_feed_cache(
        self, feed_id: str, page: int, data: dict[str, Any], ttl_seconds: int
    ) -> None:
        key = f"{feed_id}:{page}"
        expires_at = time.time() + ttl_seconds
        payload = json.dumps(data, ensure_ascii=False)
        with self._get_connection() as conn:
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
            conn.commit()

    async def list_recent_artifacts(self, limit: int = 20) -> list[DigestArtifact]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM hn_artifacts ORDER BY rowid DESC LIMIT ?;", (limit,)
            )
            rows = cur.fetchall()
            results: list[DigestArtifact] = []
            for r in rows:
                data = dict(r)
                if data.get("summary_extra"):
                    try:
                        data["summary_extra"] = json.loads(data["summary_extra"])
                    except Exception:
                        data["summary_extra"] = {}
                results.append(artifact_row_to_model(data))
            return results
