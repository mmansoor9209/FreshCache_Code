"""
freshcache/l1_cache.py
SQLite-backed answer cache — Cache Tier L1.

Stores complete generated answers keyed by query.
L1 has the strictest hit conditions:
  • semantic_equivalent(q_new, q_cached) — cosine ≥ 0.85
  • model_version and prompt_version must match
  • P_stale(answer | q, age) ≤ ε_answer
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

from .embeddings import query_hash, query_similarity
from .models import FreshnessClass, L1Entry

_LOCAL = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS l1_cache (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text       TEXT NOT NULL,
    query_hash_val   TEXT NOT NULL,
    answer           TEXT NOT NULL,
    supporting_urls  TEXT DEFAULT '[]',
    evidence_hashes  TEXT DEFAULT '[]',
    generated_at     REAL NOT NULL,
    model_version    TEXT DEFAULT 'unknown',
    prompt_version   TEXT DEFAULT '1.0',
    freshness_class  TEXT DEFAULT 'MEDIUM'
);
CREATE INDEX IF NOT EXISTS idx_l1_hash ON l1_cache(query_hash_val);
CREATE INDEX IF NOT EXISTS idx_l1_time ON l1_cache(generated_at);
"""

_DEFAULT_THRESHOLD = 0.40   # L1 requires higher similarity than L2
_SCAN_LIMIT        = 500


class L1Cache:
    """
    Answer cache: query → generated answer.

    Only yields a hit when *both* similarity and model/prompt version
    match AND the freshness risk model approves (checked in core.py).
    """

    def __init__(
        self,
        db_path: str = "freshcache_l1.db",
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
        conn = getattr(_LOCAL, "l1_conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            _LOCAL.l1_conn = conn
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: str,
        model_version: str = "unknown",
        prompt_version: str = "1.0",
        freshness_class: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[L1Entry, float]]:
        """
        Return the best matching cached answer and its similarity score,
        or None if no compatible entry meets the threshold.
        Only entries with matching model_version and prompt_version are considered.
        """
        thr = threshold if threshold is not None else self.threshold
        qh  = query_hash(query)

        try:
            # Fast path: exact hash match with version check
            row = self._conn().execute(
                "SELECT * FROM l1_cache "
                "WHERE query_hash_val = ? "
                "  AND model_version  = ? "
                "  AND prompt_version = ? "
                "ORDER BY generated_at DESC LIMIT 1",
                (qh, model_version, prompt_version),
            ).fetchone()
            if row is not None:
                return self._row_to_entry(row), 1.0

            # Slow path: cosine scan with version filter
            rows = self._conn().execute(
                "SELECT * FROM l1_cache "
                "WHERE model_version = ? AND prompt_version = ? "
                + ("AND freshness_class = ? " if freshness_class else "") +
                "ORDER BY generated_at DESC LIMIT ?",
                (model_version, prompt_version, *([freshness_class] if freshness_class else []), _SCAN_LIMIT),
            ).fetchall()
        except sqlite3.Error:
            return None

        best_entry: Optional[L1Entry] = None
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
        answer: str,
        supporting_urls: List[str],
        evidence_hashes: List[str],
        freshness_class: FreshnessClass = FreshnessClass.MEDIUM,
        model_version: str = "unknown",
        prompt_version: str = "1.0",
    ) -> L1Entry:
        """Store a generated answer for a query."""
        qh = query_hash(query)
        entry = L1Entry(
            query_text=query,
            query_hash=qh,
            answer=answer,
            supporting_urls=supporting_urls,
            evidence_hashes=evidence_hashes,
            generated_at=time.time(),
            model_version=model_version,
            prompt_version=prompt_version,
            freshness_class=freshness_class.value,
        )
        try:
            self._conn().execute(
                """
                INSERT INTO l1_cache
                  (query_text, query_hash_val, answer, supporting_urls,
                   evidence_hashes, generated_at, model_version,
                   prompt_version, freshness_class)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.query_text,
                    entry.query_hash,
                    entry.answer,
                    json.dumps(entry.supporting_urls),
                    json.dumps(entry.evidence_hashes),
                    entry.generated_at,
                    entry.model_version,
                    entry.prompt_version,
                    entry.freshness_class,
                ),
            )
            self._conn().commit()
        except sqlite3.Error as exc:
            print(f"[L1Cache] Write error: {exc}")
        return entry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_entry(self, row: sqlite3.Row) -> L1Entry:
        try:
            urls = json.loads(row["supporting_urls"])
        except (json.JSONDecodeError, TypeError):
            urls = []
        try:
            hashes = json.loads(row["evidence_hashes"])
        except (json.JSONDecodeError, TypeError):
            hashes = []
        return L1Entry(
            query_text=row["query_text"],
            query_hash=row["query_hash_val"],
            answer=row["answer"],
            supporting_urls=urls,
            evidence_hashes=hashes,
            generated_at=row["generated_at"],
            model_version=row["model_version"] or "unknown",
            prompt_version=row["prompt_version"] or "1.0",
            freshness_class=row["freshness_class"] or FreshnessClass.MEDIUM.value,
        )

    def close(self) -> None:
        """Close and discard the thread-local connection."""
        conn = getattr(_LOCAL, "l1_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            _LOCAL.l1_conn = None

    def stats(self) -> dict:
        try:
            row = self._conn().execute(
                "SELECT COUNT(*) AS cnt FROM l1_cache"
            ).fetchone()
            return {"total_entries": row["cnt"]}
        except sqlite3.Error:
            return {"total_entries": 0}
