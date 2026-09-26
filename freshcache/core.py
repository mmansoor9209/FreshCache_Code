"""
freshcache/core.py
Main FreshCache orchestrator — implements the lookup algorithm from Section 9.

Usage:
    from freshcache import FreshCache

    fc = FreshCache(
        search_fn=my_search_api,
        fetch_fn=my_web_fetcher,
        generate_fn=my_llm,
    )
    result = fc.lookup("What is today's USD/KRW rate?")
    print(result.answer, result.source.value)
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable, List, Optional, Tuple

from .l1_cache import L1Cache
from .l2_cache import L2Cache
from .l3_cache import L3Cache
from .models import (
    CacheTier,
    FreshnessClass,
    L3Entry,
    LookupResult,
    LookupSource,
    ValidationResult,
)
from .risk_model import (
    DEFAULT_EPS_ANSWER,
    DEFAULT_EPS_CONTENT,
    DEFAULT_EPS_URL_LIST,
    RuleBasedRiskModel,
)
from .validator import LightweightValidator
from .workload import WorkloadTracer

# Type aliases for the three injectable functions
SearchFn   = Callable[[str], List[str]]          # query  → list[url]
FetchFn    = Callable[[str], str]                # url    → extracted text
GenerateFn = Callable[[str, List[str]], str]     # query, evidence_list → answer


def _evidence_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class FreshCache:
    """
    Risk-constrained tiered semantic cache for open-web RAG systems.

    Three pluggable functions handle all external I/O:
        search_fn   — wraps your search API (Google, Bing, Tavily, …)
        fetch_fn    — fetches a URL and returns plain text
        generate_fn — calls your LLM with query + evidence

    Three SQLite databases persist the cache tiers across restarts.
    All risk thresholds are configurable.
    """

    def __init__(
        self,
        search_fn:   SearchFn,
        fetch_fn:    FetchFn,
        generate_fn: GenerateFn,
        l1_db:  str   = "freshcache_l1.db",
        l2_db:  str   = "freshcache_l2.db",
        l3_db:  str   = "freshcache_l3.db",
        eps_answer:   float = DEFAULT_EPS_ANSWER,
        eps_url_list: float = DEFAULT_EPS_URL_LIST,
        eps_content:  float = DEFAULT_EPS_CONTENT,
        model_version:  str = "1.0",
        prompt_version: str = "1.0",
        enable_http_validation: bool = True,
    ) -> None:
        self.search_fn   = search_fn
        self.fetch_fn    = fetch_fn
        self.generate_fn = generate_fn

        self.l1 = L1Cache(l1_db)
        self.l2 = L2Cache(l2_db)
        self.l3 = L3Cache(l3_db)

        self.risk_model = RuleBasedRiskModel()
        self.validator  = LightweightValidator()
        self.tracer     = WorkloadTracer()

        self.eps_answer   = eps_answer
        self.eps_url_list = eps_url_list
        self.eps_content  = eps_content
        self.model_version  = model_version
        self.prompt_version = prompt_version
        self.enable_http_validation = enable_http_validation

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> LookupResult:
        """
        Process a query through the three-tier cache.

        Returns a LookupResult with the answer, the cache source,
        and cost counters (search calls, fetches, cache hits per tier).
        """
        t_start = time.time()

        cache_hits: dict       = {"l1": 0, "l2": 0, "l3": 0}
        search_api_called      = False
        fetch_count            = [0]   # mutable so _resolve_content can update it

        # ----------------------------------------------------------
        # Step 1: Extract query features
        # ----------------------------------------------------------
        features = self.risk_model.extract_query_features(query)
        self.tracer.record_query(query, features)

        # ----------------------------------------------------------
        # Step 2: Try L1 answer cache
        # ----------------------------------------------------------
        if features.freshness_class != FreshnessClass.REAL_TIME:
            l1_result = self.l1.find_similar(
                query,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
            )
            if l1_result is not None:
                l1_entry, _sim = l1_result
                p_stale_l1 = self.risk_model.estimate_risk(
                    query=query,
                    tier=CacheTier.ANSWER,
                    age_seconds=l1_entry.age(),
                    features=features,
                )
                if p_stale_l1 <= self.eps_answer:
                    cache_hits["l1"] = 1
                    self.tracer.record_hit("l1", p_stale_l1)
                    return LookupResult(
                        answer=l1_entry.answer,
                        source=LookupSource.L1_HIT,
                        urls_used=l1_entry.supporting_urls,
                        latency_ms=self._ms(t_start),
                        search_api_called=False,
                        fetches_performed=0,
                        cache_hits=cache_hits,
                        stale_risk=p_stale_l1,
                    )

        # ----------------------------------------------------------
        # Step 3: Try L2 URL-list cache (skip for REAL_TIME queries)
        # ----------------------------------------------------------
        url_list: Optional[List[str]] = None
        source = LookupSource.LIVE_SEARCH

        if features.freshness_class != FreshnessClass.REAL_TIME:
            l2_result = self.l2.find_similar(query)
            if l2_result is not None:
                l2_entry, _sim = l2_result
                p_stale_l2 = self.risk_model.estimate_risk(
                    query=query,
                    tier=CacheTier.URL_LIST,
                    age_seconds=l2_entry.age(),
                    features=features,
                )
                if p_stale_l2 <= self.eps_url_list:
                    url_list = l2_entry.url_list
                    source   = LookupSource.L2_HIT
                    cache_hits["l2"] = 1
                    self.tracer.record_hit("l2", p_stale_l2)

        # ----------------------------------------------------------
        # Step 4: Live search if L2 miss
        # ----------------------------------------------------------
        if url_list is None:
            url_list = self.search_fn(query)
            self.l2.put(
                query=query,
                url_list=url_list,
                freshness_class=features.freshness_class,
            )
            search_api_called = True
            source = LookupSource.LIVE_SEARCH

        # ----------------------------------------------------------
        # Step 5: Resolve URL contents via L3
        # ----------------------------------------------------------
        evidence_texts: List[str] = []
        urls_used:      List[str] = []

        for url in url_list:
            content, was_fetched = self._resolve_content(
                url=url,
                query=query,
                features=features,
                cache_hits=cache_hits,
            )
            if content is not None:
                evidence_texts.append(content)
                urls_used.append(url)
                if was_fetched:
                    fetch_count[0] += 1

        # ----------------------------------------------------------
        # Step 6: Generate answer
        # ----------------------------------------------------------
        answer = self.generate_fn(query, evidence_texts)

        # ----------------------------------------------------------
        # Step 7: Update L1 cache
        # ----------------------------------------------------------
        evidence_hashes = [_evidence_hash(t) for t in evidence_texts]
        self.l1.put(
            query=query,
            answer=answer,
            supporting_urls=urls_used,
            evidence_hashes=evidence_hashes,
            freshness_class=features.freshness_class,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )

        return LookupResult(
            answer=answer,
            source=source,
            urls_used=urls_used,
            latency_ms=self._ms(t_start),
            search_api_called=search_api_called,
            fetches_performed=fetch_count[0],
            cache_hits=cache_hits,
            stale_risk=0.0,
        )

    # ------------------------------------------------------------------
    # Internal: L3 content resolution
    # ------------------------------------------------------------------

    def _resolve_content(
        self,
        url: str,
        query: str,
        features,
        cache_hits: dict,
    ) -> Tuple[Optional[str], bool]:
        """
        Resolve *url* to its text content.

        Returns:
            (text, was_fetched):
                text        — extracted content, or None on permanent error
                was_fetched — True if a real HTTP fetch was performed
        """
        l3_entry: Optional[L3Entry] = self.l3.get(url)

        # ---- Cache miss: fetch and store ----
        if l3_entry is None:
            text = self._safe_fetch(url)
            if text is not None:
                self.l3.put(url, text)
            return text, True

        # ---- Cache hit: estimate staleness ----
        p_stale = self.risk_model.estimate_risk(
            query=query,
            tier=CacheTier.CONTENT,
            age_seconds=l3_entry.age(),
            url=url,
            features=features,
        )

        # Low risk → reuse directly
        if p_stale <= self.eps_content:
            cache_hits["l3"] = cache_hits.get("l3", 0) + 1
            self.tracer.record_hit("l3", p_stale)
            return l3_entry.extracted_text, False

        # Moderate/high risk → validate first
        validation = self.validator.validate(
            url=url,
            entry=l3_entry,
            features=features,
            use_http=self.enable_http_validation,
        )

        if validation == ValidationResult.FRESH:
            # Touch the fetched_at timestamp so TTL resets
            self.l3.put(
                url,
                l3_entry.extracted_text,
                headers=l3_entry.raw_headers,
                status_code=l3_entry.status_code,
            )
            cache_hits["l3"] = cache_hits.get("l3", 0) + 1
            self.tracer.record_hit("l3", p_stale)
            return l3_entry.extracted_text, False

        # STALE or UNKNOWN → full refetch
        text = self._safe_fetch(url)
        if text is not None:
            self.l3.put(url, text)
            return text, True

        # Fetch failed → fall back to cached content with a warning
        print(f"[FreshCache] Refetch failed for {url}; reusing stale cache entry")
        return l3_entry.extracted_text, False

    def _safe_fetch(self, url: str) -> Optional[str]:
        """Fetch *url* via fetch_fn; return None on exception."""
        try:
            return self.fetch_fn(url)
        except Exception as exc:
            print(f"[FreshCache] Fetch error for {url}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ms(t_start: float) -> float:
        return round((time.time() - t_start) * 1000, 2)

    def close(self) -> None:
        """Close all cache connections. Call in tests and graceful shutdown."""
        self.l1.close()
        self.l2.close()
        self.l3.close()

    def stats(self) -> dict:
        """Return cache statistics for all tiers plus workload summary."""
        return {
            "l1":       self.l1.stats(),
            "l2":       self.l2.stats(),
            "l3":       self.l3.stats(),
            "workload": self.tracer.summary(),
        }
