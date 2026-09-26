"""
freshcache/l2_cache.py
SQLite-backed URL-list (retrieval-result) cache — Cache Tier L2.

Stores search-API results so that similar queries can skip the search call.
Nearest-neighbor lookup uses TF-IDF cosine similarity.

Production note: replace the linear-scan similarity search with a
vector index (FAISS, pgvector, Chroma, etc.) once the trace grows large.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

from .embeddings import query_hash, query_similarity
from .models import FreshnessClass, L2Entry

_LOCAL = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS l2_cache (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text       TEXT NOT NULL,
    query_hash_val   TEXT NOT NULL,
    url_list         TEXT NOT NULL,
    retrieved_at     REAL NOT NULL,
    search_engine    TEXT DEFAULT 'unknown',
    freshness_class  TEXT DEFAULT 'MEDIUM',
    result_snippets  TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_l2_hash ON l2_cache(query_hash_val);
CREATE INDEX IF NOT EXISTS idx_l2_time ON l2_cache(retrieved_at);
"""

_DEFAULT_THRESHOLD = 0.75   # Minimum cosine similarity for an L2 hit
_SCAN_LIMIT        = 500    # Max rows to scan in similarity search


class L2Cache:
    """
    Retrieval-result cache: query → URL list.

    Hit conditions (enforced in core.py, not here):
      • retrieval_intent_equivalent(q_new, q_cached) — checked via similarity ≥ threshold
      • P_stale(url_list | q, age) ≤ ε_url_list
      • query class is not REAL_TIME
    """

    def __init__(
        self,
        db_path: str = "freshcache_l2.db",
        similarity_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.db_path = db_path
        self.threshold = similarity_threshold
        self._bootstrap()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(_LOCAL, "l2_conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            _LOCAL.l2_conn = conn
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[L2Entry, float]]:
        """
        Return the best matching cached entry and its similarity score,
        or None if no entry meets the threshold.
        """
        thr = threshold if threshold is not None else self.threshold
        qh  = query_hash(query)

        try:
            # Fast path: exact hash match
            row = self._conn().execute(
                "SELECT * FROM l2_cache WHERE query_hash_val = ? "
                "ORDER BY retrieved_at DESC LIMIT 1",
                (qh,),
            ).fetchone()
            if row is not None:
                return self._row_to_entry(row), 1.0

            # Slow path: linear cosine scan over recent entries
            rows = self._conn().execute(
                "SELECT * FROM l2_cache ORDER BY retrieved_at DESC LIMIT ?",
                (_SCAN_LIMIT,),
            ).fetchall()
        except sqlite3.Error:
            return None

        best_entry: Optional[L2Entry] = None
        best_sim = 0.0

        for row in rows:
            sim = query_similarity(query, row["query_text"])
            if sim > best_sim:
                best_sim = sim
                best_entry = self._row_to_entry(row)

        if best_entry is not None and best_sim >= thr:
            return best_entry, best_sim
        return None

    def put(
        self,
        query: str,
        url_list: List[str],
        freshness_class: FreshnessClass = FreshnessClass.MEDIUM,
        search_engine: str = "unknown",
        snippets: Optional[List[str]] = None,
    ) -> L2Entry:
        """Store a URL list for a query."""
        qh = query_hash(query)
        entry = L2Entry(
            query_text=query,
            query_hash=qh,
            url_list=url_list,
            retrieved_at=time.time(),
            search_engine=search_engine,
            freshness_class=freshness_class.value,
            result_snippets=snippets or [],
        )
        try:
            self._conn().execute(
                """
                INSERT INTO l2_cache
                  (query_text, query_hash_val, url_list, retrieved_at,
                   search_engine, freshness_class, result_snippets)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.query_text,
                    entry.query_hash,
                    json.dumps(entry.url_list),
                    entry.retrieved_at,
                    entry.search_engine,
                    entry.freshness_class,
                    json.dumps(entry.result_snippets),
                ),
            )
            self._conn().commit()
        except sqlite3.Error as exc:
            print(f"[L2Cache] Write error: {exc}")
        return entry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_entry(self, row: sqlite3.Row) -> L2Entry:
        try:
            url_list = json.loads(row["url_list"])
        except (json.JSONDecodeError, TypeError):
            url_list = []
        try:
            snippets = json.loads(row["result_snippets"])
        except (json.JSONDecodeError, TypeError):
            snippets = []
        return L2Entry(
            query_text=row["query_text"],
            query_hash=row["query_hash_val"],
            url_list=url_list,
            retrieved_at=row["retrieved_at"],
            search_engine=row["search_engine"] or "unknown",
            freshness_class=row["freshness_class"] or FreshnessClass.MEDIUM.value,
            result_snippets=snippets,
        )

    def close(self) -> None:
        """Close and discard the thread-local connection."""
        conn = getattr(_LOCAL, "l2_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            _LOCAL.l2_conn = None

    def stats(self) -> dict:
        try:
            row = self._conn().execute(
                "SELECT COUNT(*) AS cnt FROM l2_cache"
            ).fetchone()
            return {"total_entries": row["cnt"]}
        except sqlite3.Error:
            return {"total_entries": 0}
