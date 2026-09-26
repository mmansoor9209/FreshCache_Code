"""
freshcache/l3_cache.py
SQLite-backed URL-content (evidence) cache — Cache Tier L3.

This is the most stable and highest-value cache tier.
It stores fetched web page content keyed by canonical URL.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Optional
from urllib.parse import urlparse

from .models import L3Entry

_LOCAL = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS l3_cache (
    canonical_url      TEXT PRIMARY KEY,
    extracted_text     TEXT NOT NULL,
    content_hash       TEXT NOT NULL,
    fetched_at         REAL NOT NULL,
    etag               TEXT,
    last_modified      TEXT,
    status_code        INTEGER DEFAULT 200,
    extraction_version TEXT    DEFAULT '1.0',
    raw_headers        TEXT    DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_l3_fetched
    ON l3_cache(fetched_at);
"""


def _canonical_url(url: str) -> str:
    """Normalize URL: lowercase, strip trailing slash, drop common tracking params."""
    url = url.strip()
    parsed = urlparse(url)
    if parsed.query:
        kept = [p for p in parsed.query.split("&")
                if not p.startswith(("utm_", "ref=", "source=", "fbclid="))]
        query = "&".join(kept)
        path = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        url = f"{path}?{query}" if kept else path
    else:
        url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return url.lower().rstrip("/")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class L3Cache:
    """
    Evidence cache: URL → extracted text + HTTP metadata.

    Thread-safe via threading.local connections.
    Write failures are logged but never raise — the cache is advisory.
    """

    def __init__(self, db_path: str = "freshcache_l3.db") -> None:
        self.db_path = db_path
        self._bootstrap()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """Create tables in a one-off connection (called once at init)."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(_LOCAL, "l3_conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _LOCAL.l3_conn = conn
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, url: str) -> Optional[L3Entry]:
        """Return cached entry for *url*, or None on miss."""
        canon = _canonical_url(url)
        try:
            row = self._conn().execute(
                "SELECT * FROM l3_cache WHERE canonical_url = ?", (canon,)
            ).fetchone()
        except sqlite3.Error:
            return None

        if row is None:
            return None

        try:
            raw_headers: dict = json.loads(row["raw_headers"] or "{}")
        except (json.JSONDecodeError, TypeError):
            raw_headers = {}

        return L3Entry(
            canonical_url=row["canonical_url"],
            extracted_text=row["extracted_text"],
            content_hash=row["content_hash"],
            fetched_at=row["fetched_at"],
            etag=row["etag"],
            last_modified=row["last_modified"],
            status_code=row["status_code"] or 200,
            extraction_version=row["extraction_version"] or "1.0",
            raw_headers=raw_headers,
        )

    def put(
        self,
        url: str,
        extracted_text: str,
        headers: Optional[dict] = None,
        status_code: int = 200,
        extraction_version: str = "1.0",
    ) -> L3Entry:
        """Store *extracted_text* for *url*. Returns the stored entry."""
        canon = _canonical_url(url)
        h = {k.lower(): v for k, v in (headers or {}).items()}
        entry = L3Entry(
            canonical_url=canon,
            extracted_text=extracted_text,
            content_hash=_content_hash(extracted_text),
            fetched_at=time.time(),
            etag=h.get("etag"),
            last_modified=h.get("last-modified"),
            status_code=status_code,
            extraction_version=extraction_version,
            raw_headers=h,
        )
        try:
            self._conn().execute(
                """
                INSERT OR REPLACE INTO l3_cache
                  (canonical_url, extracted_text, content_hash, fetched_at,
                   etag, last_modified, status_code, extraction_version, raw_headers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.canonical_url,
                    entry.extracted_text,
                    entry.content_hash,
                    entry.fetched_at,
                    entry.etag,
                    entry.last_modified,
                    entry.status_code,
                    entry.extraction_version,
                    json.dumps(entry.raw_headers),
                ),
            )
            self._conn().commit()
        except sqlite3.Error as exc:
            print(f"[L3Cache] Write error for {canon}: {exc}")
        return entry

    def invalidate(self, url: str) -> bool:
        """Remove an entry. Returns True if a row was deleted."""
        canon = _canonical_url(url)
        try:
            cur = self._conn().execute(
                "DELETE FROM l3_cache WHERE canonical_url = ?", (canon,)
            )
            self._conn().commit()
            return cur.rowcount > 0
        except sqlite3.Error:
            return False

    def evict_older_than(self, max_age_seconds: float) -> int:
        """Delete entries older than *max_age_seconds*. Returns number deleted."""
        cutoff = time.time() - max_age_seconds
        try:
            cur = self._conn().execute(
                "DELETE FROM l3_cache WHERE fetched_at < ?", (cutoff,)
            )
            self._conn().commit()
            return cur.rowcount
        except sqlite3.Error:
            return 0

    def close(self) -> None:
        """Close and discard the thread-local connection. Call in tests and shutdown."""
        conn = getattr(_LOCAL, "l3_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            _LOCAL.l3_conn = None

    def stats(self) -> dict:
        try:
            row = self._conn().execute(
                "SELECT COUNT(*) AS cnt, AVG(? - fetched_at) AS avg_age "
                "FROM l3_cache",
                (time.time(),),
            ).fetchone()
            return {
                "total_entries": row["cnt"],
                "avg_age_seconds": round(row["avg_age"] or 0.0, 1),
            }
        except sqlite3.Error:
            return {"total_entries": 0, "avg_age_seconds": 0.0}
