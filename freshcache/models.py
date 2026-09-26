"""
freshcache/models.py
All data classes and enums used across FreshCache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FreshnessClass(Enum):
    TIMELESS  = "TIMELESS"
    SLOW      = "SLOW"
    MEDIUM    = "MEDIUM"
    FAST      = "FAST"
    REAL_TIME = "REAL_TIME"


class CacheTier(Enum):
    ANSWER   = "answer"
    URL_LIST = "url_list"
    CONTENT  = "content"


class ValidationResult(Enum):
    FRESH   = "FRESH"
    STALE   = "STALE"
    UNKNOWN = "UNKNOWN"
    ERROR   = "ERROR"


class LookupSource(Enum):
    L1_HIT           = "L1_HIT"
    L2_HIT           = "L2_HIT"
    SEARCHLESS_REUSE = "SEARCHLESS_REUSE"
    LIVE_SEARCH      = "LIVE_SEARCH"


@dataclass
class QueryFeatures:
    query:            str
    has_temporal_cue: bool              = False
    temporal_keywords: List[str]        = field(default_factory=list)
    answer_type:      Optional[str]     = None
    freshness_class:  FreshnessClass    = FreshnessClass.MEDIUM
    language:         str               = "en"
    entities:         List[str]         = field(default_factory=list)


@dataclass
class RiskProfile:
    freshness_class:   FreshnessClass
    p_stale_answer:    float   # P(stale | tier=answer)
    p_stale_url_list:  float   # P(stale | tier=url_list)
    p_stale_content:   float   # P(stale | tier=content)
    confidence:        float   = 1.0
    reasoning:         str     = ""


@dataclass
class L3Entry:
    canonical_url:      str
    extracted_text:     str
    content_hash:       str
    fetched_at:         float
    etag:               Optional[str]        = None
    last_modified:      Optional[str]        = None
    status_code:        int                  = 200
    extraction_version: str                  = "1.0"
    raw_headers:        Dict[str, str]       = field(default_factory=dict)

    def age(self) -> float:
        """Age in seconds since this entry was fetched."""
        return time.time() - self.fetched_at


@dataclass
class L2Entry:
    query_text:      str
    query_hash:      str
    url_list:        List[str]
    retrieved_at:    float
    search_engine:   str        = "unknown"
    freshness_class: str        = FreshnessClass.MEDIUM.value
    result_snippets: List[str]  = field(default_factory=list)

    def age(self) -> float:
        return time.time() - self.retrieved_at


@dataclass
class L1Entry:
    query_text:       str
    query_hash:       str
    answer:           str
    supporting_urls:  List[str]
    evidence_hashes:  List[str]
    generated_at:     float
    model_version:    str  = "unknown"
    prompt_version:   str  = "1.0"
    freshness_class:  str  = FreshnessClass.MEDIUM.value

    def age(self) -> float:
        return time.time() - self.generated_at


@dataclass
class LookupResult:
    answer:             str
    source:             LookupSource
    urls_used:          List[str]
    latency_ms:         float
    search_api_called:  bool
    fetches_performed:  int
    cache_hits:         Dict[str, int]
    stale_risk:         float = 0.0
