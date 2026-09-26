"""
freshcache/workload.py
Workload characterization and query trace recorder.

Records every query + its features so you can compute:
  - exact / semantic repetition rates
  - freshness class distribution
  - per-tier hit rates
  - average stale risk at hit time
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Dict, List

from .embeddings import query_similarity
from .models import QueryFeatures


class WorkloadTracer:
    """
    In-memory query trace recorder.

    For long-running deployments, flush to a JSONL file or database
    via export_trace().
    """

    def __init__(self) -> None:
        self._queries: List[dict] = []
        self._hits:    List[dict] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_query(self, query: str, features: QueryFeatures) -> None:
        self._queries.append({
            "timestamp":       time.time(),
            "query":           query,
            "freshness_class": features.freshness_class.value,
            "answer_type":     features.answer_type,
            "has_temporal_cue": features.has_temporal_cue,
            "language":        features.language,
        })

    def record_hit(self, tier: str, p_stale: float) -> None:
        self._hits.append({
            "timestamp": time.time(),
            "tier":      tier,
            "p_stale":   p_stale,
        })

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def exact_repetition_rate(self) -> float:
        """Fraction of queries that exactly repeat a prior query."""
        if len(self._queries) < 2:
            return 0.0
        seen: set = set()
        repeats = 0
        for record in self._queries:
            normalized = record["query"].strip().lower()
            if normalized in seen:
                repeats += 1
            seen.add(normalized)
        return repeats / len(self._queries)

    def semantic_repetition_rate(self, threshold: float = 0.75) -> float:
        """
        Fraction of queries with a semantically similar prior query.
        O(n²) — call only on small traces (< 1 000 queries).
        """
        texts = [r["query"] for r in self._queries]
        n = len(texts)
        if n < 2:
            return 0.0
        repeats = 0
        for i in range(1, n):
            for j in range(i):
                if query_similarity(texts[i], texts[j]) >= threshold:
                    repeats += 1
                    break
        return repeats / n

    def summary(self) -> dict:
        """Return a human-readable workload summary dict."""
        total = len(self._queries)
        if total == 0:
            return {"total_queries": 0}

        class_dist = Counter(q["freshness_class"] for q in self._queries)
        type_dist  = Counter(
            q["answer_type"] for q in self._queries if q["answer_type"]
        )
        lang_dist  = Counter(q["language"] for q in self._queries)
        temporal_pct = (
            sum(1 for q in self._queries if q["has_temporal_cue"]) / total
        )

        hit_by_tier: Dict[str, List[dict]] = defaultdict(list)
        for h in self._hits:
            hit_by_tier[h["tier"]].append(h)

        hit_rates = {
            tier: round(len(hits) / total, 3)
            for tier, hits in hit_by_tier.items()
        }
        avg_stale_at_hit = {
            tier: round(sum(h["p_stale"] for h in hits) / len(hits), 4)
            for tier, hits in hit_by_tier.items()
            if hits
        }

        return {
            "total_queries":               total,
            "freshness_class_distribution": dict(class_dist),
            "answer_type_distribution":     dict(type_dist),
            "language_distribution":        dict(lang_dist),
            "temporal_cue_pct":             round(temporal_pct * 100, 1),
            "cache_hit_rates":              hit_rates,
            "avg_stale_risk_at_hit":        avg_stale_at_hit,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_trace(self) -> List[dict]:
        """Return the full query trace list (copy)."""
        return list(self._queries)
