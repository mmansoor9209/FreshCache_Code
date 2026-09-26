"""
freshcache/risk_model.py
Freshness risk model — rule-based baseline with data-driven calibration.

Estimates P(stale | query, age, tier, url) using:
  - Temporal keyword detection
  - Answer-type classification
  - Freshness class assignment
  - Age-based exponential decay (λ fitted from change_log.jsonl)
  - Domain volatility adjustment (fitted from url_manifest.jsonl)

Risk threshold defaults:
  ε_answer   = 0.10  (most conservative)
  ε_url_list = 0.20
  ε_content  = 0.35  (most permissive)
"""

from __future__ import annotations

import json
import math
import re
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import numpy as np
import warnings
warnings.filterwarnings("ignore")
# # At the top of risk_model.py, after existing imports
# from .utils import is_noise_url   # if you move it to utils

from urllib.parse import urlparse

from .models import CacheTier, FreshnessClass, QueryFeatures, RiskProfile

# ---------------------------------------------------------------------------
# Default risk budgets
# ---------------------------------------------------------------------------
DEFAULT_EPS_ANSWER   = 0.10
DEFAULT_EPS_URL_LIST = 0.20
DEFAULT_EPS_CONTENT  = 0.35

# ---------------------------------------------------------------------------
# Temporal keyword sets
# ---------------------------------------------------------------------------
_TEMPORAL_EN: FrozenSet[str] = frozenset([
    "today", "now", "current", "currently", "latest", "live",
    "real-time", "realtime", "breaking", "just", "recent",
    "this week", "this month", "price", "score", "rate",
    "weather", "stock", "market", "exchange", "election",
    "vote", "result", "match", "game", "update",
])

_TEMPORAL_KO: FrozenSet[str] = frozenset([
    "현재", "오늘", "지금", "최신", "실시간", "이번주", "이번달",
    "최근", "가격", "환율", "날씨", "주가", "점수", "결과",
    "경기", "속보", "업데이트",
])

# ---------------------------------------------------------------------------
# Answer-type vocabulary — must match ANSWER_TYPES in train_risk_model.py
# ---------------------------------------------------------------------------
_ANSWER_TYPES_LIST = [
    "price", "score", "weather", "exchange_rate", "stock",
    "definition", "biography", "news", "schedule",
]
_ANSWER_TYPE_TO_IDX: Dict[str, int] = {
    a: i for i, a in enumerate(_ANSWER_TYPES_LIST)
}
_N_ANSWER_TYPES = len(_ANSWER_TYPES_LIST)   # 9
_MLP_INPUT_DIM  = 5 + 1 + 1 + _N_ANSWER_TYPES + 1   # = 17

# ---------------------------------------------------------------------------
# Default half-lives (seconds) — overridden by calibrate() once data arrives
# ---------------------------------------------------------------------------
_DEFAULT_HALF_LIFE: Dict[FreshnessClass, float] = {
    FreshnessClass.REAL_TIME :            30.0,   # 30 sec  (kept as prior — REAL_TIME pages often blocked by scrapers)
    FreshnessClass.FAST      :         11373.4,   # calibrated from observed change rates
    FreshnessClass.MEDIUM    :         54194.5,   # calibrated from observed change rates
    FreshnessClass.SLOW      :       1349104.0,   # calibrated from observed change rates
    FreshnessClass.TIMELESS  :       1866942.7,   # calibrated from observed change rates
}

# Tier multiplier — answer cache goes stale fastest
_TIER_MULT: Dict[CacheTier, float] = {
    CacheTier.ANSWER:   1.5,
    CacheTier.URL_LIST: 1.2,
    CacheTier.CONTENT:  1.0,
}

# ---------------------------------------------------------------------------
# Default domain volatility priors (overridden by calibrate())
# ---------------------------------------------------------------------------
_DEFAULT_DOMAIN_VOLATILITY: Dict[str, float] = {
    "twitter.com":           0.95,
    "x.com":                 0.95,
    "reddit.com":            0.85,
    "finance.yahoo.com":     0.95,
    "investing.com":         0.90,
    "coindesk.com": 1.0,
    "bloomberg.com":         0.85,
    "reuters.com":           0.80,
    "bbc.com":               0.75,
    "cnn.com":               0.75,
    "nytimes.com":           0.70,
    "theguardian.com":       0.70,
    "weather.com":           0.90,
    "accuweather.com":       0.90,
    "techcrunch.com":        0.55,
    "arstechnica.com":       0.45,
    "github.com":            0.35,
    "medium.com":            0.30,
    "stackoverflow.com":     0.15,
    "wikipedia.org":         0.12,
    "ko.wikipedia.org":      0.12,
    "arxiv.org":             0.05,
    "docs.python.org":       0.05,
    "developer.mozilla.org": 0.08,
    # "namu.wiki":             0.40,
    "namu.wiki":             0.46,   # was 0.40 — wiki with dynamic edit counters

    "coinmarketcap.com": 1.0,
    "coingecko.com":         0.95,
    "m.stock.naver.com":     0.95,
    "kr.investing.com":      0.90,
    "tradingeconomics.com":  0.85,
    "worldbank.org":         0.20,
    "statista.com":          0.25,
    "ign.com":               0.30,
    "socialblade.com":       0.50,
    "boxofficemojo.com":     0.45,

    # These domains have dynamic page elements that cause false-positive
    # change detection in TIMELESS content. Volatility set just above
    # the TIMELESS ceiling (0.45) so they are excluded from TIMELESS
    # calibration but remain valid for SLOW calibration.
    # "namu.wiki":             0.46,   # was 0.40 — wiki with dynamic edit counters
    "m.blog.naver.com":      0.46,   # Naver blog with view counters in body
    "blog.naver.com":        0.46,   # same
    "brainly.com":           0.46,   # Q&A site with ad content in body
    "2news.com": 1.0,   # local TV news with rotating video widget
    "newsbytesapp.com": 1.0,   # news app with "In the news" trending sidebar

    # Sports and entertainment — dynamic nav sections
    "sports.yahoo.com":      0.65,   # Yahoo Sports with trending nav

    # Genuine misclassifications — these are not TIMELESS content
    "vietcombank.com.vn":    0.95,   # live exchange rate table
    "en.vietnamplus.vn": 1.0,   # Vietnamese news agency
}

# ── Domain volatility ceiling per freshness class ──────────────────────────
# If a URL's domain volatility exceeds this ceiling for its assigned query
# class, the URL is a search quality mismatch and is excluded from
# class-level calibration.
CLASS_VOL_CEILING: Dict[str, float] = {
    "TIMELESS":  0.45,   # was 0.25 — raised so unknown domains (default 0.40) pass through
    "SLOW":      0.50,
    "MEDIUM":    0.70,
    "FAST":      0.90,
    "REAL_TIME": 1.00,
}

_DEFAULT_DOMAIN_VOL = 0.40   # fallback for unknown domains


def is_noise_url(domain: str, freshness_class: str) -> bool:
    """
    Return True if this URL's domain volatility is too high for its
    assigned query class. Scales automatically to any dataset size.
    """
    domain = (domain or "").lower().removeprefix("www.")
    vol    = _DEFAULT_DOMAIN_VOLATILITY.get(domain, _DEFAULT_DOMAIN_VOL)
    # Partial match for subdomains
    if vol == _DEFAULT_DOMAIN_VOL:
        for known, v in _DEFAULT_DOMAIN_VOLATILITY.items():
            if domain.endswith("." + known):
                vol = v
                break
    ceiling = CLASS_VOL_CEILING.get(freshness_class, 0.50)
    return vol > ceiling

# ---------------------------------------------------------------------------
# Answer-type keyword patterns
# ---------------------------------------------------------------------------
_ANSWER_TYPE_PATTERNS: Dict[str, list] = {
    "price":         ["price", "cost", "worth", "가격", "비용"],
    "score":         ["score", "result", "point", "점수", "결과", "골"],
    "weather":       ["weather", "temperature", "forecast", "날씨", "기온"],
    "exchange_rate": ["exchange rate", "dollar", "won", "yen", "환율", "달러"],
    "stock":         ["stock", "share", "equity", "nasdaq", "주가", "주식"],
    "definition":    ["what is", "explain", "define", "설명", "뭐야", "무엇", "뜻"],
    "biography":     ["who is", "biography", "born", "누구", "인물"],
    "news":          ["news", "breaking", "뉴스", "속보", "사건"],
    "schedule":      ["schedule", "when does", "date", "일정", "언제", "날짜"],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_temporal_keywords(query: str) -> list:
    q = query.lower()
    return [kw for kw in (_TEMPORAL_EN | _TEMPORAL_KO) if kw in q]


def _detect_answer_type(query: str) -> Optional[str]:
    q = query.lower()
    for atype, keywords in _ANSWER_TYPE_PATTERNS.items():
        if any(kw in q for kw in keywords):
            return atype
    return None


def _detect_language(query: str) -> str:
    korean_count = len(re.findall(r"[가-힣]", query))
    return "ko" if korean_count > len(query) * 0.2 else "en"


def _classify_freshness(
    query: str,
    temporal_keywords: list,
    answer_type: Optional[str],
) -> FreshnessClass:
    q = query.lower()

    # REAL_TIME
    if answer_type == "score":
        return FreshnessClass.REAL_TIME
    rt_kws = {"live", "real-time", "realtime", "실시간", "지금 경기", "현재 점수"}
    if any(kw in q for kw in rt_kws):
        return FreshnessClass.REAL_TIME

    # FAST
    fast_types = {"price", "weather", "exchange_rate", "stock", "news"}
    fast_kws   = {"today", "오늘", "now", "지금", "current", "현재",
                  "latest", "최신", "breaking", "속보"}
    if answer_type in fast_types:
        return FreshnessClass.FAST
    if any(kw in q for kw in fast_kws):
        return FreshnessClass.FAST

    # MEDIUM
    medium_kws = {"this week", "this month", "이번주", "이번달",
                  "recent", "최근", "election", "선거", "schedule", "일정"}
    if answer_type == "schedule" or any(kw in q for kw in medium_kws):
        return FreshnessClass.MEDIUM

    # TIMELESS
    timeless_types = {"definition", "biography"}
    timeless_kws   = {"what is", "explain", "define", "설명", "뭐야", "뜻",
                      "theorem", "law of", "history of"}
    if answer_type in timeless_types and not temporal_keywords:
        return FreshnessClass.TIMELESS
    if any(kw in q for kw in timeless_kws) and not temporal_keywords:
        return FreshnessClass.TIMELESS

    return FreshnessClass.SLOW if temporal_keywords else FreshnessClass.MEDIUM


def _domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return ""


def _fit_lambda(change_rate: float, age_seconds: float) -> float:
    """
    Fit λ from a single observed change rate at a given age.
    P(stale) = 1 - exp(-λ × age) → λ = -ln(1 - P) / age
    Clamp change_rate to avoid log(0).
    """
    p = max(0.01, min(0.99, change_rate))
    return -math.log(1.0 - p) / age_seconds


def _half_life_from_lambda(lam: float) -> float:
    return math.log(2) / lam


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class RuleBasedRiskModel:
    """
    Freshness risk model with optional data-driven calibration.

    Usage:
        model = RuleBasedRiskModel()
        model.calibrate("data/change_log.jsonl", "data/url_manifest.jsonl")
        p = model.estimate_risk(query, tier, age_seconds, url)
    """

    def __init__(self) -> None:
        self._half_life       = dict(_DEFAULT_HALF_LIFE)
        self._domain_vol      = dict(_DEFAULT_DOMAIN_VOLATILITY)
        self._calibrated      = False
        self._calibration_log: List[str] = []

    # ------------------------------------------------------------------
    # Calibration — fit λ from real change_log data
    # ------------------------------------------------------------------

    def calibrate(
    self,
    change_log_path: str = "data/change_log.jsonl",
    manifest_path:   str = "data/url_manifest.jsonl",
    excluded_runs:   list = None,
    ) -> None:
        excluded = set(excluded_runs or [])
        """
        Fit staleness rates from collected data.

        For each freshness class and run_id (time delta):
          1. Count URLs that changed (from change_log)
          2. Count total URLs tracked (from manifest, snapshot_available=True)
          3. Fit λ = -ln(1 - change_rate) / age_seconds
          4. Average λ across all time deltas → convert to half_life

        Also fits domain volatility from per-domain change rates.
        """
        cl_path = Path(change_log_path)
        mf_path = Path(manifest_path)

        if not cl_path.exists() or not mf_path.exists():
            self._calibration_log.append("Calibration skipped — data files not found.")
            return

        # Parse run_id → age_seconds mapping
        run_ages = {
            "rerun_1h":  3_600,
            "rerun_6h":  21_600,
            "rerun_12h": 43_200,
            "rerun_24h": 86_400,
            "rerun_48h": 172_800,
            "rerun_72h": 259_200,
        }

        # Load change_log
        changes = []
        for line in cl_path.open(encoding="utf-8"):
            try:
                changes.append(json.loads(line))
            except Exception:
                continue

        # Load manifest — count total tracked URLs per (run_id, freshness_class)
        tracked: Dict[tuple, int] = defaultdict(int)
        for line in mf_path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("snapshot_available") and r.get("run_id", "run_00") != "run_00":
                    if r.get("run_id") in excluded:          # ← add this line
                        continue                             # ← add this line
                    domain          = r.get("domain", "")
                    freshness_class = r.get("freshness_class", "")
                    if freshness_class and is_noise_url(domain, freshness_class):
                        continue
                    key = (r["run_id"], r["freshness_class"])
                    tracked[key] += 1
            except Exception:
                continue

        # Count changes per (run_id, freshness_class)
        # Skip noise-flagged records — these are search quality mismatches
        changed: Dict[tuple, int] = defaultdict(int)
        for c in changes:
            if c.get("noise", False) or c.get("run_id") in excluded:   # ← add excluded check
                continue
            key = (c["run_id"], c["freshness_class"])
            changed[key] += 1

        # Fit λ per freshness class across all available time deltas
        class_lambdas: Dict[str, List[tuple]] = defaultdict(list)  # [(lam, age_seconds)]
        for (run_id, fc), total in tracked.items():
            if run_id not in run_ages or total == 0:
                continue
            age      = run_ages[run_id]
            n_change = changed.get((run_id, fc), 0)
            rate     = n_change / total
            lam = _fit_lambda(rate, age)
            class_lambdas[fc].append((lam, age))
            self._calibration_log.append(
                f"  {fc} @ {run_id}: {n_change}/{total} changed "
                f"(rate={rate:.3f}, λ={lam:.6f}, "
                f"half_life={_half_life_from_lambda(lam)/3600:.1f}h)"
            )

        # Update half_lives from fitted λ values
        for fc_str, lam_age_pairs in class_lambdas.items():
            try:
                fc      = FreshnessClass(fc_str)
                lambdas = [p[0] for p in lam_age_pairs]
                ages    = [p[1] for p in lam_age_pairs]
                avg_lam = sum(l * a for l, a in zip(lambdas, ages)) / sum(ages)
                new_hl  = _half_life_from_lambda(avg_lam)   # ← this line was missing
                self._half_life[fc] = new_hl
                self._calibration_log.append(
                    f"  → {fc_str}: half_life updated to "
                    f"{_half_life_from_lambda(avg_lam)/3600:.2f}h "
                    f"(time-weighted MLE λ={avg_lam:.6f})"
                )
            except ValueError:
                continue

        # Fit domain volatility from per-domain change rates
        domain_changed: Dict[str, int] = defaultdict(int)
        domain_total:   Dict[str, int] = defaultdict(int)
        for c in changes:
            domain_changed[c.get("domain", "")] += 1
        for line in mf_path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("snapshot_available") and r.get("run_id") != "run_00":
                    domain_total[r.get("domain", "")] += 1
            except Exception:
                continue

        for domain, total in domain_total.items():
            if not domain or total < 2:
                continue
            rate = domain_changed.get(domain, 0) / total
            self._domain_vol[domain] = round(rate, 3)
        # Enforce monotonic minimum half-lives per class
        MIN_HALF_LIVES = {
            FreshnessClass.REAL_TIME: 60.0,         # min 1 minute
            FreshnessClass.FAST:      1_800.0,      # min 30 min
            FreshnessClass.MEDIUM:    43_200.0,     # min 12h
            FreshnessClass.SLOW:      172_800.0,    # min 48h
            FreshnessClass.TIMELESS:  2_592_000.0,  # min 30 days
        }
        for fc, min_hl in MIN_HALF_LIVES.items():
            if self._half_life[fc] < min_hl:
                self._half_life[fc] = min_hl
                self._calibration_log.append(
                    f"  → {fc.value}: half_life clamped to minimum {min_hl/3600:.1f}h"
                )
        self._calibrated = True
        self._calibration_log.append(
            f"\nCalibration complete. "
            f"{len(class_lambdas)} classes fitted, "
            f"{len(domain_total)} domains fitted."
        )

    def print_calibration_report(self) -> None:
        print("\n=== Risk Model Calibration Report ===")
        for line in self._calibration_log:
            print(line)
        print("\nFitted half-lives:")
        for fc, hl in self._half_life.items():
            print(f"  {fc.value:<12}: {hl/3600:.2f}h")
        print("=====================================\n")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_query_features(self, query: str) -> QueryFeatures:
        temporal_kws = _detect_temporal_keywords(query)
        answer_type  = _detect_answer_type(query)
        language     = _detect_language(query)
        fc           = _classify_freshness(query, temporal_kws, answer_type)
        return QueryFeatures(
            query=query,
            has_temporal_cue=bool(temporal_kws),
            temporal_keywords=temporal_kws,
            answer_type=answer_type,
            freshness_class=fc,
            language=language,
        )

    # ------------------------------------------------------------------
    # Risk estimation
    # ------------------------------------------------------------------

    def _age_stale_prob(
        self,
        age_seconds: float,
        freshness_class: FreshnessClass,
        tier: CacheTier,
    ) -> float:
        half_life  = self._half_life[freshness_class]
        multiplier = _TIER_MULT[tier]
        lam = math.log(2) / half_life * multiplier
        return 1.0 - math.exp(-lam * age_seconds)

    def _get_domain_volatility(self, url: str) -> float:
        domain = _domain_from_url(url)
        if domain in self._domain_vol:
            return self._domain_vol[domain]
        for known, vol in self._domain_vol.items():
            if domain.endswith("." + known) or domain == known:
                return vol
        return 0.40

    def estimate_risk(
        self,
        query:       str,
        tier:        CacheTier,
        age_seconds: float,
        url:         Optional[str] = None,
        features:    Optional[QueryFeatures] = None,
    ) -> float:
        """Return P(stale) ∈ [0, 1] for a cache entry."""
        if features is None:
            features = self.extract_query_features(query)

        p = self._age_stale_prob(age_seconds, features.freshness_class, tier)

        if url:
            vol = self._get_domain_volatility(url)
            p   = p + vol * 0.15 * (1.0 - p)

        return round(min(1.0, max(0.0, p)), 4)

    def build_risk_profile(
        self,
        query:       str,
        age_seconds: float,
        url:         Optional[str] = None,
    ) -> RiskProfile:
        """Return stale probability for all three tiers in one call."""
        features = self.extract_query_features(query)
        return RiskProfile(
            freshness_class=features.freshness_class,
            p_stale_answer=self.estimate_risk(
                query, CacheTier.ANSWER,   age_seconds, url, features),
            p_stale_url_list=self.estimate_risk(
                query, CacheTier.URL_LIST, age_seconds, url, features),
            p_stale_content=self.estimate_risk(
                query, CacheTier.CONTENT,  age_seconds, url, features),
            reasoning=(
                f"class={features.freshness_class.value}, "
                f"calibrated={self._calibrated}, "
                f"temporal_kws={features.temporal_keywords}, "
                f"answer_type={features.answer_type}"
            ),
        )

# ---------------------------------------------------------------------------
# Learned risk model (trained MLP, replaces rule-based half-life)
# ---------------------------------------------------------------------------

class RiskMLP(torch.nn.Module):
    """Lightweight staleness estimator — same architecture as in training."""
    def __init__(self, input_dim: int = 7,
                 hidden: List[int] = None, dropout: float = 0.3):
        super().__init__()
        import torch.nn as nn
        if hidden is None:
            hidden = [64, 32, 16]
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LearnedRiskModel(RuleBasedRiskModel):
    """
    Drops in for RuleBasedRiskModel.
    Uses trained MLP for P(stale) estimation when a saved model exists.
    Falls back to rule-based decay if no model file is found.
    """

    def __init__(self, model_path: str = "data/risk_model.pt") -> None:
        super().__init__()
        self._mlp:         Optional[RiskMLP] = None
        self._temperature: float             = 1.0
        self._model_path   = Path(model_path)
        self._load_if_available()
        rich_path = Path("data/query_rich_features.json")
        self._rich_features = (
            json.load(open(rich_path)) if rich_path.exists() else {}
        )

    def _load_if_available(self) -> None:
        import torch
        if not self._model_path.exists():
            return
        try:
            ckpt = torch.load(str(self._model_path), map_location="cpu")
            cfg  = ckpt["config"]
            self._mlp = RiskMLP(cfg.get("input_dim", 7), cfg["hidden"], cfg["dropout"])
            self._mlp.load_state_dict(ckpt["model_state"])
            self._mlp.eval()
            self._temperature = float(ckpt["temperature"])
            print(f"LearnedRiskModel: loaded {self._model_path} "
                  f"(T={self._temperature:.4f})")
        except Exception as e:
            print(f"LearnedRiskModel: could not load model ({e}), "
                  f"falling back to rule-based.")
            self._mlp = None

    # Feature vector matching training
    _CLASS_IDX = {"TIMELESS": 0, "SLOW": 1, "MEDIUM": 2, "FAST": 3, "REAL_TIME": 4}
    _MAX_AGE_H = 72.0

    def _mlp_features(self, freshness_class, delta_t, domain, query=None,
                  sim_to_cached=None, n_cached_urls=None,
                  evidence_domains=None, cached_query=None,
                  rich_features_map=None):
        # ── Existing 17 features ─────────────────────────────────────────────
        CLASSES = ["TIMELESS","SLOW","MEDIUM","FAST","REAL_TIME"]
        class_oh = np.zeros(5)
        if freshness_class in CLASSES:
            class_oh[CLASSES.index(freshness_class)] = 1.0

        log_age  = min(math.log(1 + delta_t / 3600 / 72), 1.0)
        dom_vol  = self._domain_vol.get(domain, 0.1)

        TYPES = ["price","score","weather","exchange_rate","stock",
                "definition","biography","news","schedule"]
        answer_oh = np.zeros(9)
        if query and hasattr(self, '_query_meta'):
            qt = self._query_meta.get(query, {}).get("answer_type","")
            if qt in TYPES:
                answer_oh[TYPES.index(qt)] = 1.0

        CUES = {"current","latest","today","now","live","ongoing",
                "this week","this month","right now","real-time"}
        has_cue = 0.0
        if query:
            ql = query.lower()
            has_cue = 1.0 if any(c in ql for c in CUES) else 0.0

        base17 = np.concatenate([class_oh, [log_age], [dom_vol],
                                answer_oh, [has_cue]])

        # ── New 8 features ───────────────────────────────────────────────────
        rf = {}
        if query:
            rf = (rich_features_map or self._rich_features).get(query, {})

        max_ev    = rf.get("max_entity_vol",           0.30)
        mean_ev   = rf.get("mean_entity_vol",          0.30)
        high_ev   = rf.get("has_high_vol_entity",      0.0)

        sim_c     = rf.get("sim_to_cached",            0.80)
        if sim_to_cached is not None:
            sim_c = float(sim_to_cached)   # inference: use actual sim score

        log_n     = rf.get("log_n_urls",               0.3)
        if n_cached_urls is not None:
            log_n = min(math.log(1 + n_cached_urls) / math.log(6), 1.0)

        mean_edom = rf.get("mean_evidence_domain_vol", 0.1)
        if evidence_domains is not None:
            vols = [self._domain_vol.get(d, 0.1) for d in evidence_domains if d]
            mean_edom = float(np.mean(vols)) if vols else 0.1

        lex_j     = rf.get("lexical_jaccard",  1.0)
        ent_ov    = rf.get("entity_overlap",   1.0)
        if cached_query is not None and query is not None:
            ta = set(query.lower().split())
            tb = set(cached_query.lower().split())
            denom = len(ta | tb)
            lex_j  = len(ta & tb) / denom if denom else 1.0
            # entity_overlap: use precomputed if available, else approximate
            ent_ov = rf.get("entity_overlap", 1.0 if lex_j > 0.5 else 0.0)

        new8 = np.array([max_ev, mean_ev, high_ev, sim_c,
                        log_n, mean_edom, lex_j, ent_ov], dtype=float)

        return np.concatenate([base17, new8])

    def estimate_risk(
        self,
        query:       str,
        tier:        CacheTier,
        age_seconds: float,
        url:         Optional[str] = None,
        features:    Optional[QueryFeatures] = None,
    ) -> float:
        if self._mlp is None:
            return super().estimate_risk(query, tier, age_seconds, url, features)

        import torch
        if features is None:
            features = self.extract_query_features(query)

        x = torch.tensor(
            self._mlp_features(features.freshness_class, age_seconds, url, query),
            dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            logit_scaled = self._mlp(x) / self._temperature
        p_content = float(torch.sigmoid(logit_scaled))

        # Tier adjustment: P(tier stale) = 1 − (1 − P_content)^tier_mult
        # This matches the rule-based model's λ × tier_mult scaling.
        mult  = _TIER_MULT.get(tier, 1.0)
        p_adj = 1.0 - (1.0 - p_content) ** mult

        return round(min(1.0, max(0.0, p_adj)), 4)