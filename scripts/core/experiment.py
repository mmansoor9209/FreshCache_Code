"""
experiment.py — FreshCache Tiered Cache Experiment (C2)

Simulates all cache methods on the collected query trace and measures
every metric needed for the paper comparison table and Pareto figure.

Methods:
  NoCache          — always search + fetch all URLs + generate
  ExactTTL         — exact query match + fixed TTL per freshness class
  SemanticTTL      — cosine similarity ≥ 0.85 + fixed TTL (GPTCache-style)
  L3Only           — always search; cache URL content only
  L3_ConditionalGet — L3Only + conditional GET validation signal
  FreshCache_Full  — complete tiered cache (L1 + L2 + L3 + risk model)

Results reported PER FRESHNESS CLASS:
  TIMELESS / SLOW / MEDIUM / FAST / REAL_TIME

Two experimental conditions:
  t=1h   — queries arrive 1 hour after cache was populated
  t=24h  — queries arrive 24 hours after cache was populated

Output:
  Console — per-class comparison tables
  data/experiment_results.json — full metrics
  data/pareto_data.csv         — latency vs stale_error for Pareto figure

Run:
    python experiment.py
"""

from __future__ import annotations
from setproctitle import setproctitle
setproctitle("anon-freshcache")
import json
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# GPU_IDS          = [4, 5, 6, 7]          # all four A6000s
GPU_IDS          = [0, 1]          # 2 gpus A6000s
DATA_DIR      = Path("data")
# # OLD
# EMBEDDINGS_CACHE = DATA_DIR / "query_embeddings.npy"
# SIM_MATRIX_CACHE = DATA_DIR / "sim_matrix.npy"

# # For ablation 
# EMBEDDINGS_CACHE = DATA_DIR / "query_embeddings_minilm.npy"
# SIM_MATRIX_CACHE = DATA_DIR / "sim_matrix_minilm.npy"

# NEW
EMBEDDINGS_CACHE = DATA_DIR / "query_embeddings_bgem3.npy"
SIM_MATRIX_CACHE = DATA_DIR / "sim_matrix_bgem3.npy"

# DATA_DIR      = Path("data")
MANIFEST_FILE = DATA_DIR / "url_manifest.jsonl"
QUERIES_FILE  = DATA_DIR / "queries.jsonl"
CHANGE_FILE   = DATA_DIR / "change_log.jsonl"
RESULTS_FILE  = DATA_DIR / "experiment_results.json"
# # just for ablation of minilm
# RESULTS_FILE  = DATA_DIR / "experiment_results_minilm.json"
PARAPHRASE_FILE = DATA_DIR / "paraphrase_clusters.jsonl"
PARETO_FILE   = DATA_DIR / "pareto_data.csv"
# # just for ablation of minilm
# PARETO_FILE   = DATA_DIR / "pareto_data_minilm.csv"

NEW_RUN_CUTOFF = "2026-05-10"

FRESHNESS_CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]

# ---------------------------------------------------------------------------
# Latency model (milliseconds)
# ---------------------------------------------------------------------------
LATENCY = {
    "search_api":       500,
    "web_fetch":        800,
    "conditional_get":  150,
    "llm_generate":    2000,
    "l1_lookup":          5,
    "l2_lookup":          5,
    "l3_lookup":          5,
}

P95_FETCH_MULT = 3.0

FIXED_TTL = {
    "TIMELESS":  30 * 86_400,
    "SLOW":       7 * 86_400,
    "MEDIUM":     1 * 86_400,
    "FAST":           3_600,
    "REAL_TIME":          0,
}

EPS_ANSWER   = 0.10
EPS_URL_LIST = 0.20
EPS_CONTENT  = 0.35

L1_SIM_THRESHOLD = 0.40   # updated for sentence encoder
L2_SIM_THRESHOLD = 0.35

SIM_AGES = {
    "1h":  3_600,
    "24h": 86_400,
}

# ---------------------------------------------------------------------------
# Calibrated half-lives (from risk_model.calibrate())
# W2 UPDATE: these values are now fit using a per-class temporal holdout
# (calibrate.py, FIT_WINDOWS_BY_CLASS). MEDIUM/FAST are fit on rerun_1h+
# rerun_12h only; TIMELESS/SLOW are fit on rerun_1h+rerun_12h+rerun_24h.
# The holdout is ASYMMETRIC, not uniform. t=24h is held out for MEDIUM and
# FAST only; TIMELESS and SLOW use rerun_24h in their own fit, so for those
# two classes the t=24h evaluation is in-sample. t=7d is later than the
# fitting window for all four. An earlier version of this comment claimed
# 24h was held out for every class; that was wrong and is corrected here.
# This still replaces the pre-W2 values, which were fit on every window
# including the evaluation windows for all four classes.
# Synced manually from data/calibration_report.json — calibrate.py only
# patches freshcache/risk_model.py, not this file.
# ---------------------------------------------------------------------------
HALF_LIFE = {
    "REAL_TIME":         30.0,        # 30s   (kept as prior, unchanged)
    "FAST":          449_280.0,       # 5.2d  (W2 LOCKED: 1h+12h fit, 24h+7d held out)
    "MEDIUM":        457_920.0,       # 5.3d  (W2 LOCKED: 1h+12h fit, 24h+7d held out)
    "SLOW":          993_600.0,       # 11.5d (W2 LOCKED: 1h+12h+24h fit, 7d held out)
    "TIMELESS":    2_592_000.0,       # 30d   (W2 LOCKED: 1h+12h+24h fit, 7d held out)
}
# This is the final W2 calibration. With the C18 semantic-equivalence gate
# in force it produces FreshCache_Full at t=24h: 4.06% hash-based content
# drift, 80.7% search savings (ALL CLASSES, BGE-M3). The 3.3%/98.4% figures
# this comment carried previously predate that gate and match no current
# result; see data/experiment_results.json for the live values.
# See calibrate.py FIT_WINDOWS_BY_CLASS for the fitting methodology.

# Pre-W2 half-lives (fit on all windows including the t=24h/t=7d eval
# windows themselves — the configuration the reviewer flagged as
# fit-on-test). Kept here, unused by default, only for explicit
# side-by-side comparison runs against the W2-corrected HALF_LIFE above.
# Do not use this dict for any number reported as the paper's main result.
HALF_LIFE_PRE_W2_FIT_ON_TEST = {
    "REAL_TIME":         30.0,
    "FAST":          10_800.0,   # 3h
    "MEDIUM":        54_000.0,   # 15h
    "SLOW":       1_382_400.0,   # 16d
    "TIMELESS":   1_900_800.0,   # 22d
}

# Naive prior half-lives — reasonable class-based assumptions before calibration
# Used for the "without calibration" ablation in sim_freshcache_no_calib
HALF_LIFE_UNCAL = {
    "REAL_TIME":      1_800.0,   # 0.5h  — assumed very volatile
    "FAST":           3_600.0,   # 1h    — assumed volatile
    "MEDIUM":        86_400.0,   # 24h   — assumed moderate
    "SLOW":         604_800.0,   # 7d    — assumed slow-changing
    "TIMELESS":   2_592_000.0,   # 30d   — assumed stable
}


# ── MLP risk model — loaded once for sim_freshcache_mlp ablation ──────────
try:
    from freshcache.risk_model import LearnedRiskModel as _LearnedRiskModel
    from freshcache.models   import (CacheTier      as _CacheTier,
                                     FreshnessClass  as _FreshnessClass,
                                     QueryFeatures   as _QueryFeatures)
    _MLP_RISK = _LearnedRiskModel()
    _MLP_RISK.calibrate(str(CHANGE_FILE), str(MANIFEST_FILE))
    _FC_ENUM   = {fc: _FreshnessClass(fc)
                  for fc in ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]}
    _TIER_ENUM = {"answer":   _CacheTier.ANSWER,
                  "url_list": _CacheTier.URL_LIST,
                  "content":  _CacheTier.CONTENT}
    _MLP_AVAILABLE = getattr(_MLP_RISK, "_mlp", None) is not None
    print(f"  MLP risk model: {'loaded' if _MLP_AVAILABLE else 'not found — rule-based fallback'}")
except Exception as _e:
    _MLP_AVAILABLE = False
    print(f"  MLP risk model unavailable: {_e}")

TIER_MULT = {
    "answer":   1.5,
    "url_list": 1.2,
    "content":  1.0,
}

# URL-tier equivalence floor. Fitted and then measured against 590 judged
# L2 hits: 24,021 hits retained at 14.8% mismatch [12.1%, 17.5%], inside
# eps_url_list = 0.20. Lexical and answer-type conditions are not applied
# at L2 because they cost hits without reducing mismatch.
# L2_SIM_THRESHOLD stays at 0.35 because it is shared with the baselines.
_L2_EQ_SIM_FLOOR = 0.75

# ---------------------------------------------------------------------------
# Similarity — sentence encoder with TF-IDF fallback
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9]+|[가-힣]+", text.lower())

def _tf_vec(text: str) -> dict:
    tokens = _tokenize(text)
    total  = len(tokens) or 1
    c      = Counter(tokens)
    return {t: f / total for t, f in c.items()}

def _tfidf_sim(q1: str, q2: str) -> float:
    v1, v2 = _tf_vec(q1), _tf_vec(q2)
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    n1  = math.sqrt(sum(x * x for x in v1.values()))
    n2  = math.sqrt(sum(x * x for x in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0

# ── Entity filter for L1 matching ─────────────────────────────────────────
import spacy as _spacy
_NLP = _spacy.load("en_core_web_sm")

def _entities(text: str) -> set:
    """Extract named entities from query text for L1 entity-match filter."""
    doc = _NLP(text)
    return {ent.text.lower() for ent in doc.ents}

_ENTITY_CACHE: dict = {}

def _entity_match(q1: str, q2: str) -> bool:
    """
    Return True if queries share all named entities or neither has entities.
    Blocks L1 hits where entity sets differ (e.g. iPhone 16 vs iPhone 16 Pro).

    Uses whole-span token matching instead of substring containment to avoid
    false passes like 'iPhone 16' matching inside 'iPhone 16 Pro'.
    An entity span is only accepted if it appears as a complete token sequence
    and is not a prefix of a longer alphanumeric token sequence in the other query.
    """
    if q1 not in _ENTITY_CACHE:
        _ENTITY_CACHE[q1] = _entities(q1)
    if q2 not in _ENTITY_CACHE:
        _ENTITY_CACHE[q2] = _entities(q2)
    e1, e2 = _ENTITY_CACHE[q1], _ENTITY_CACHE[q2]
    if not e1 or not e2:
        return True   # if either has no entities, pass through

    tokens1 = q1.lower().split()
    tokens2 = q2.lower().split()

    def _whole_span_present(ent_tokens: list, query_tokens: list) -> bool:
        """
        True if ent_tokens appears as a complete token sequence in query_tokens
        and is not immediately followed by another alphanumeric token
        (which would mean it is a prefix of a longer entity span).
        """
        n = len(ent_tokens)
        for i in range(len(query_tokens) - n + 1):
            if query_tokens[i:i + n] == ent_tokens:
                if i + n < len(query_tokens):
                    next_tok = query_tokens[i + n]
                    if next_tok.isalnum():
                        continue   # prefix match only — not a complete span
                return True
        return False

    for ent in e1:
        if not _whole_span_present(ent.split(), tokens2):
            return False
    for ent in e2:
        if not _whole_span_present(ent.split(), tokens1):
            return False
    return True

# ── Semantic equivalence gate for L1 (research plan §6.1) ────────────────
# Plan §6.1 specifies:
#     L1 hit allowed iff semantic_equivalent(q_new, q_cached) = true
#                     and P_stale(answer|...) <= eps_answer
# and states "Cosine similarity alone is not used."
#
# Prior versions of this file implemented cosine >= L1_SIM_THRESHOLD plus
# _entity_match() only. Measured consequence at t=24h: 84.7% of realized
# L1 hits (95% CI [81.0%, 88.3%]) served an answer to a different question,
# invisible to the hash-based stale metric because an unrelated but stable
# page never changes.
#
# Operating point fitted against 420 LLM-judged realized L1 hits
# (fit_equivalence_gate.py, data/equivalence_gate_fit.json):
#     sim >= 0.80 AND content-word Jaccard >= 0.30 AND answer types agree
#     -> 9.8% mismatch among admitted, within eps_answer = 0.10
#
# NOTE: the 0.80 floor lives here, NOT in L1_SIM_THRESHOLD, because that
# constant is shared with SemanticTTL, vCache, SCALM, DomainTTL,
# TemporalKeywordTTL and TieredFixedTTL. Changing it would silently move
# every baseline off the operating point it was measured at.

_EQ_SIM_FLOOR   = 0.80   # answer-tier similarity floor
_EQ_JACCARD_MIN = 0.30   # content-word Jaccard floor

_EQ_STOPWORDS = frozenset("""
a an the of in on at to for from by with and or is are was were be been
what which who whom whose when where why how did does do have has had
that this these those it its as than then there their you your i me my
most recent latest current currently now today last next new
""".split())

# Answer-type vocabulary, mirroring _ANSWER_TYPE_PATTERNS in
# freshcache/risk_model.py so the gate and the risk model agree.
_EQ_ANSWER_TYPES = {
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

_EQ_TOKEN_CACHE: dict = {}
_EQ_ATYPE_CACHE: dict = {}


def _eq_content_tokens(text: str) -> set:
    """Content words only, stopwords and temporal cues removed."""
    cached = _EQ_TOKEN_CACHE.get(text)
    if cached is None:
        toks = set(re.findall(r"[a-z0-9]+|[가-힣]+", text.lower()))
        cached = toks - _EQ_STOPWORDS
        _EQ_TOKEN_CACHE[text] = cached
    return cached


def _eq_answer_type(text: str):
    cached = _EQ_ATYPE_CACHE.get(text, "__miss__")
    if cached != "__miss__":
        return cached
    q = text.lower()
    found = None
    for atype, kws in _EQ_ANSWER_TYPES.items():
        if any(kw in q for kw in kws):
            found = atype
            break
    _EQ_ATYPE_CACHE[text] = found
    return found


def semantic_equivalent(q1: str, q2: str, sim: float) -> bool:
    """
    Plan §6.1 semantic_equivalent(). True if serving the stored answer for
    q2 in response to q1 is expected to be correct.

    Three conjunctive conditions, ordered cheapest-first so the common
    rejection path costs one float comparison:
      1. similarity at or above the answer-tier floor
      2. content-word Jaccard at or above the floor
      3. answer types agree when both are detectable
    """
    if sim < _EQ_SIM_FLOOR:
        return False

    t1 = _eq_content_tokens(q1)
    t2 = _eq_content_tokens(q2)
    union = t1 | t2
    if not union:
        return False
    if len(t1 & t2) / len(union) < _EQ_JACCARD_MIN:
        return False

    a1 = _eq_answer_type(q1)
    a2 = _eq_answer_type(q2)
    if a1 is not None and a2 is not None and a1 != a2:
        return False

    # Strict-subset guard. If one query's content tokens are wholly
    # contained in the other's, the extra tokens are a specifier that
    # narrows the question ("iPhone 16" vs "iPhone 16 Pro", "final" vs
    # "semi-final"). Overlap is high precisely because one question is a
    # more specific version of the other, so Jaccard cannot catch it.
    # A genuine paraphrase rewords rather than appends, so it seldom
    # produces an exact subset.
    if t1 != t2 and (t1 <= t2 or t2 <= t1):
        return False

    return True

# ── Encoder (used only as fallback if matrix not yet built) ───────────────



# ── Encoder (used only as fallback if matrix not yet built) ───────────────
try:
    from freshcache.embeddings import query_similarity as _fallback_sim
    ENCODER = "sentence-transformers/precomputed"
except Exception:
    _fallback_sim = _tfidf_sim
    ENCODER = "tfidf-fallback"

# These are populated by precompute_similarity_matrix() before any simulation
_SIM_MATRIX:   np.ndarray | None = None
_QUERY_TO_IDX: dict        | None = None


def cosine_sim(q1: str, q2: str) -> float:
    """
    Fast O(1) similarity lookup using pre-computed matrix.
    Falls back to encoder/tfidf only if matrix is not ready.
    """
    if _SIM_MATRIX is not None and _QUERY_TO_IDX is not None:
        i = _QUERY_TO_IDX.get(q1, -1)
        j = _QUERY_TO_IDX.get(q2, -1)
        if i >= 0 and j >= 0:
            return float(_SIM_MATRIX[i, j])
    return _fallback_sim(q1, q2)

# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------

def p_stale(fc: str, age: float, tier: str = "content") -> float:
    """P(stale) at full floating-point precision.

    PRECISION FIX: this previously returned round(..., 4). The value is used
    directly as a decision boundary (p_stale <= EPS_*), so rounding it moved
    the boundary: a true risk anywhere in (eps, eps + 5e-5) was pulled back
    onto eps and the entry was reused. That made the exponential gate slightly
    more permissive than its own equivalent TTL,
    -h*ln(1-eps)/(m*ln2), which is the exact inverse of the UNROUNDED rule.
    Rounding here is now removed; round only a copy used for display, via
    p_stale_display(). Half-lives, tier multipliers and epsilons are unchanged.
    """
    hl  = HALF_LIFE.get(fc, 86_400.0)
    mul = TIER_MULT.get(tier, 1.0)
    lam = math.log(2) / hl * mul
    return 1.0 - math.exp(-lam * age)

def p_stale_uncal(fc: str, age: float, tier: str = "content") -> float:
    """Identical to p_stale but uses naive prior half-lives (HALF_LIFE_UNCAL).

    PRECISION FIX: rounding removed here for the same reason as p_stale; this
    function also feeds a `<= eps` temporal comparison in the uncalibrated
    variant.
    """
    hl  = HALF_LIFE_UNCAL.get(fc, 86_400.0)
    mul = TIER_MULT.get(tier, 1.0)
    lam = math.log(2) / hl * mul
    return 1.0 - math.exp(-lam * age)


def p_stale_display(p: float, ndigits: int = 4) -> float:
    """Round a risk value for printing or logging ONLY.

    Never call this on a value that is about to be compared against an epsilon.
    """
    return round(p, ndigits)


def p_stale_mlp(fc_str: str, age: float, tier: str,
                url: str = "", query: str = "") -> float:
    """
    P(stale) from the trained MLP (LearnedRiskModel).
    query is used to extract answer_type and has_temporal_cue — the
    features that make the model query-conditioned within a class.
    Falls back to rule-based p_stale if model file is missing.
    """
    if not _MLP_AVAILABLE:
        return p_stale(fc_str, age, tier)
    features = _QueryFeatures(
        query             = query,
        freshness_class   = _FC_ENUM.get(fc_str, _FreshnessClass.MEDIUM),
    )
    return _MLP_RISK.estimate_risk(
        query       = query,
        tier        = _TIER_ENUM.get(tier, _CacheTier.CONTENT),
        age_seconds = age,
        url         = url or "",
        features    = features,
    )

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_query_records(queries: list, manifest: list,
                        paraphrases: list = None) -> list:
    baseline = [m for m in manifest
                if m.get("run_id") == "run_00"
                and m.get("snapshot_available")]

    # Build url index keyed by query_id
    query_urls: dict = defaultdict(list)
    for m in baseline:
        qid = m["query_id"]
        query_urls[qid].append({
            "url_hash": m["url_hash"],
            "url":      m["url"],
            "domain":   m.get("domain", ""),
            "rank":     m.get("rank", 99),
        })
    for qid in query_urls:
        query_urls[qid].sort(key=lambda x: x["rank"])

    # Build base records (queries with real URL coverage)
    records = []
    base_url_by_qid: dict = {}   # query_id → url list for paraphrase lookup

    for q in queries:
        qid  = q["query_id"]
        urls = query_urls.get(qid, [])
        if not urls:
            continue
        rec = {
            "query_id":        qid,
            "query":           q["query"],
            "freshness_class": q["freshness_class"],
            "language":        q.get("language", "en"),
            "cluster_id":      q.get("cluster_id") or qid,
            "is_paraphrase":   False,
            "urls":            urls,
        }
        records.append(rec)
        base_url_by_qid[qid] = urls

    # Add paraphrase records (borrow URLs from base query)
    if paraphrases:
        for p in paraphrases:
            base_id = p.get("base_query_id", "")
            base_urls = base_url_by_qid.get(base_id)
            if not base_urls:
                continue   # base query had no URLs — skip
            records.append({
                "query_id":        p["query_id"],
                "query":           p["query"],
                "freshness_class": p["freshness_class"],
                "language":        p.get("language", "en"),
                "cluster_id":      p.get("cluster_id") or base_id,
                "is_paraphrase":   True,
                "urls":            base_urls,
            })

    # Sort: group by cluster, base query first, then paraphrases
    def sort_key(r):
        cluster = r.get("cluster_id") or r["query_id"]
        is_para = 1 if r.get("is_paraphrase") else 0
        return (cluster, is_para, r["query_id"])

    records.sort(key=sort_key)
    return records


def build_stale_sets(changes: list) -> dict:
    stale = {"rerun_1h": set(), "rerun_24h": set()}
    for c in changes:
        if c.get("noise", False):
            continue
        rid = c.get("run_id")
        if rid == "rerun_1h" and c.get("detected_at", "") > NEW_RUN_CUTOFF:
            stale["rerun_1h"].add(c["url_hash"])
        elif rid == "rerun_24h" and c.get("detected_at", "") > NEW_RUN_CUTOFF:
            stale["rerun_24h"].add(c["url_hash"])
        elif rid == "rerun_12h" and c.get("detected_at", "") > NEW_RUN_CUTOFF:
            # Use 12h changes as proxy for 24h if rerun_24h missing
            stale["rerun_24h"].add(c["url_hash"])
    return stale

# ---------------------------------------------------------------------------
# Method simulators
# ---------------------------------------------------------------------------

def sim_nocache(records: list, stale_urls: set, sim_age: float) -> dict:
    n             = len(records)
    total_search  = n
    total_fetches = sum(len(r["urls"]) for r in records)

    # Build latencies in record order (preserves alignment with records)
    latency_list = [
        LATENCY["search_api"] +
        len(r["urls"]) * LATENCY["web_fetch"] +
        LATENCY["llm_generate"]
        for r in records
    ]

    # Honest p50: base queries only (no paraphrase inflation)
    base_lats = [lat for r, lat in zip(records, latency_list)
                 if not r.get("is_paraphrase")]

    # Base-only counters: every query is a search call in NoCache
    base_n            = sum(1 for r in records if not r.get("is_paraphrase"))
    base_search_calls = base_n
    base_stale_errors = 0

    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50

    return {
        "search_calls":           total_search,
        "fetches":                total_fetches,
        "l1_hits":                0,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p50_base_ms":            p50_base,
        "p95_ms":                 p95,
        "cache_stale_errors":     0,
        "cache_stale_error_rate": 0.0,
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_exact_ttl(records: list, stale_urls: set, sim_age: float) -> dict:
    n            = len(records)
    cache        = {}
    search_calls = 0
    fetches      = 0
    l1_hits      = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        q   = r["query"]
        fc  = r["freshness_class"]
        ttl = FIXED_TTL.get(fc, 0)
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        if q in cache and ttl > 0 and sim_age <= ttl:
            l1_hits += 1
            lat = LATENCY["l1_lookup"] + LATENCY["llm_generate"]
            if any(uh in stale_urls for uh in cache[q]["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(r["urls"])
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            cache[q] = {"freshness_class": fc,
                        "url_hashes": [u["url_hash"] for u in r["urls"]]}
        latency_list.append(lat)

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_semantic_ttl(records: list, stale_urls: set, sim_age: float) -> dict:
    n            = len(records)
    cache        = []
    search_calls = 0
    fetches      = 0
    l1_hits      = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        q   = r["query"]
        fc  = r["freshness_class"]
        ttl = FIXED_TTL.get(fc, 0)
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        hit = None
        if ttl > 0 and sim_age <= ttl:
            best_sim = 0.0
            for entry in cache:
                sim = cosine_sim(q, entry["query"])
                if sim >= L1_SIM_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    hit      = entry

        if hit:
            l1_hits += 1
            lat = LATENCY["l1_lookup"] + LATENCY["llm_generate"]
            if any(uh in stale_urls for uh in hit["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(r["urls"])
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            cache.append({"query": q, "fc": fc,
                          "url_hashes": [u["url_hash"] for u in r["urls"]]})
        latency_list.append(lat)

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_l3only(records: list, stale_urls: set, sim_age: float) -> dict:
    n            = len(records)
    l3_cache     = set()
    search_calls = n
    fetches      = 0
    l3_hits      = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        fc        = r["freshness_class"]
        n_miss    = 0
        hit_stale = False
        is_base   = not r.get("is_paraphrase")
        if is_base:
            base_n += 1
            base_search_calls += 1  # L3Only always searches

        for u in r["urls"]:
            uh = u["url_hash"]
            if uh in l3_cache:
                ps = p_stale(fc, sim_age, "content")
                if ps <= EPS_CONTENT:
                    l3_hits += 1
                    if uh in stale_urls:
                        hit_stale = True
                    continue
            fetches   += 1
            n_miss    += 1
            l3_cache.add(uh)

        lat = (LATENCY["search_api"] +
               n_miss * LATENCY["web_fetch"] +
               (len(r["urls"]) - n_miss) * LATENCY["l3_lookup"] +
               LATENCY["llm_generate"])
        latency_list.append(lat)
        if hit_stale:
            stale_errors += 1
            if is_base:
                base_stale_errors += 1

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                0,
        "l2_hits":                0,
        "l3_hits":                l3_hits,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l3_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_l3_cget(records: list, stale_urls: set, sim_age: float) -> dict:
    n            = len(records)
    l3_cache     = set()
    search_calls = n
    fetches      = 0
    l3_hits      = 0
    validated    = 0
    latency_list = []
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        fc    = r["freshness_class"]
        n_miss = 0
        n_val  = 0
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1
            base_search_calls += 1  # L3+cGET always searches

        for u in r["urls"]:
            uh = u["url_hash"]
            if uh in l3_cache:
                ps = p_stale(fc, sim_age, "content")
                if ps <= EPS_CONTENT:
                    l3_hits += 1
                else:
                    n_val    += 1
                    validated += 1
                    if uh in stale_urls:
                        fetches  += 1
                        n_miss   += 1
                        l3_cache.add(uh)
                    else:
                        l3_hits += 1
                continue
            fetches  += 1
            n_miss   += 1
            l3_cache.add(uh)

        lat = (LATENCY["search_api"] +
               n_miss * LATENCY["web_fetch"] +
               n_val  * LATENCY["conditional_get"] +
               LATENCY["llm_generate"])
        latency_list.append(lat)

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                0,
        "l2_hits":                0,
        "l3_hits":                l3_hits,
        "validated_reuses":       validated,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     0,
        "cache_stale_error_rate": 0.0,
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_freshcache(records: list, stale_urls: set, sim_age: float,
                   dec=None, method="FreshCache_Full", cond="") -> dict:
    n        = len(records)
    l1_cache = []
    l2_cache = []
    l3_cache = set()

    search_calls      = 0
    fetches           = 0
    l1_hits           = 0
    l2_hits           = 0
    l3_hits           = 0
    validated         = 0
    latency_list      = []
    stale_errors      = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    def _jaccard(set_a, set_b):
        u = set_a | set_b
        return len(set_a & set_b) / len(u) if u else 0.0

    for r in records:
        q    = r["query"]
        fc   = r["freshness_class"]
        urls = r["urls"]
        lat  = 0.0
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        url_set_new = {u["url_hash"] for u in urls}

        # L1
        l1_hit = None
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l1_cache:
                sim = cosine_sim(q, entry["query"])
                ps  = p_stale(entry["fc"], sim_age, "answer")
                if sim >= L1_SIM_THRESHOLD and ps <= EPS_ANSWER and sim > best_sim:
                    if (_entity_match(q, entry["query"])
                            and semantic_equivalent(q, entry["query"], sim)):
                        best_sim = sim
                        l1_hit   = entry

        if l1_hit:
            l1_hits += 1
            lat += LATENCY["l1_lookup"]
            if any(uh in stale_urls for uh in l1_hit["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
            latency_list.append(lat)
            continue

        # L2
        l2_hit     = None
        url_hashes = [u["url_hash"] for u in urls]
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l2_cache:
                sim = cosine_sim(q, entry["query"])
                ps  = p_stale(entry["fc"], sim_age, "url_list")
                if (sim >= L2_SIM_THRESHOLD and sim >= _L2_EQ_SIM_FLOOR
                        and ps <= EPS_URL_LIST and sim > best_sim):
                    best_sim = sim
                    l2_hit   = entry

        if l2_hit:
            l2_hits += 1
            lat     += LATENCY["l2_lookup"]
            # Register the served list under this query. Writing only on a
            # miss meant a cluster whose base query was captured by another
            # entry never registered its own URL list, so its paraphrases
            # could not match it: same-cluster L2 reuse was 1.1%.
            l2_cache.append({"query": q, "fc": fc,
                             "url_hashes": l2_hit["url_hashes"]})
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            lat          += LATENCY["search_api"]
            l2_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

        # L3
        hit_stale  = False
        n_validate = 0

        for u in urls:
            uh = u["url_hash"]
            if fc == "REAL_TIME":
                fetches += 1
                lat += LATENCY["web_fetch"]
                l3_cache.add(uh)
                continue
            if uh in l3_cache:
                ps = p_stale(fc, sim_age, "content")
                if ps <= EPS_CONTENT:
                    l3_hits += 1
                    lat     += LATENCY["l3_lookup"]
                    if uh in stale_urls:
                        hit_stale = True
                else:
                    n_validate += 1
                    validated  += 1
                    lat        += LATENCY["conditional_get"]
                    if uh in stale_urls:
                        fetches += 1
                        lat     += LATENCY["web_fetch"]
                        l3_cache.add(uh)
                    else:
                        l3_hits += 1
            else:
                fetches += 1
                lat     += LATENCY["web_fetch"]
                l3_cache.add(uh)

        lat += LATENCY["llm_generate"]
        latency_list.append(lat)

        if hit_stale:
            stale_errors += 1
            if is_base:
                base_stale_errors += 1

        l1_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                l2_hits,
        "l3_hits":                l3_hits,
        "validated_reuses":       validated,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits + l2_hits + l3_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }

# paste _vcache_threshold() and sim_vcache() here
def _vcache_threshold(observations: list, delta: float) -> float:
    if not observations:
        return L1_SIM_THRESHOLD   # ← was: return 1.0
    candidates = sorted(set(s for s, _ in observations))
    best_t = 1.0
    for t in candidates:
        subset = [(s, c) for s, c in observations if s >= t]
        if not subset:
            continue
        err_rate = sum(1 for _, c in subset if c == 0) / len(subset)
        if err_rate <= delta:
            best_t = t
    return best_t


def sim_vcache(records: list, stale_urls: set, sim_age: float,
               delta: float = EPS_ANSWER) -> dict:
    n            = len(records)
    cache        = []
    search_calls = 0
    fetches      = 0
    l1_hits      = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        q          = r["query"]
        fc         = r["freshness_class"]
        urls       = r["urls"]
        url_hashes = [u["url_hash"] for u in urls]
        is_base    = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        if fc == "REAL_TIME":
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(urls)
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            latency_list.append(lat)
            cache.append({"query": q, "fc": fc,
                          "url_hashes": url_hashes, "observations": []})
            continue

        best_sim = 0.0
        nn_entry = None
        for entry in cache:
            sim = cosine_sim(q, entry["query"])
            if sim > best_sim:
                best_sim = sim
                nn_entry = entry

        exploit = False
        if nn_entry is not None and best_sim >= L1_SIM_THRESHOLD:
            t_star = max(_vcache_threshold(nn_entry["observations"], delta),
                         L1_SIM_THRESHOLD)
            if best_sim >= t_star:
                exploit = True

        if exploit:
            l1_hits += 1
            lat = LATENCY["l1_lookup"] + LATENCY["llm_generate"]
            correct = 0 if any(uh in stale_urls for uh in nn_entry["url_hashes"]) else 1
            if correct == 0:
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
            nn_entry["observations"].append((best_sim, correct))
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(urls)
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            cache.append({"query": q, "fc": fc,
                          "url_hashes": url_hashes, "observations": []})

        latency_list.append(lat)

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_tiered_fixed_ttl(records: list, stale_urls: set, sim_age: float) -> dict:
    """
    TieredFixedTTL baseline: identical three-tier architecture to FreshCache
    (L1 answer / L2 URL-list / L3 content, same similarity thresholds and
    latency model), but the staleness gate is a hard TTL check instead of a
    calibrated probability.
    """
    n        = len(records)
    l1_cache = []
    l2_cache = []
    l3_cache = set()

    search_calls      = 0
    fetches           = 0
    l1_hits           = 0
    l2_hits           = 0
    l3_hits           = 0
    latency_list      = []
    stale_errors      = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        q    = r["query"]
        fc   = r["freshness_class"]
        urls = r["urls"]
        lat  = 0.0
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        url_hashes = [u["url_hash"] for u in urls]

        # ── L1: answer cache ──────────────────────────────────────────────
        l1_ttl = HALF_LIFE.get(fc, 86_400.0) * TIER_MULT["answer"]
        l1_hit = None
        if fc != "REAL_TIME" and sim_age <= l1_ttl:
            best_sim = 0.0
            for entry in l1_cache:
                sim = cosine_sim(q, entry["query"])
                if sim >= L1_SIM_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    l1_hit   = entry

        if l1_hit:
            l1_hits += 1
            lat += LATENCY["l1_lookup"]
            if any(uh in stale_urls for uh in l1_hit["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
            latency_list.append(lat)
            continue

        # ── L2: URL-list cache ────────────────────────────────────────────
        l2_ttl = HALF_LIFE.get(fc, 86_400.0) * TIER_MULT["url_list"]
        l2_hit = None
        if fc != "REAL_TIME" and sim_age <= l2_ttl:
            best_sim = 0.0
            for entry in l2_cache:
                sim = cosine_sim(q, entry["query"])
                if sim >= L2_SIM_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    l2_hit   = entry

        if l2_hit:
            l2_hits += 1
            lat += LATENCY["l2_lookup"]
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            lat += LATENCY["search_api"]
            l2_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

        # ── L3: content cache ─────────────────────────────────────────────
        l3_ttl    = HALF_LIFE.get(fc, 86_400.0) * TIER_MULT["content"]
        hit_stale = False

        for u in urls:
            uh = u["url_hash"]
            if fc == "REAL_TIME":
                fetches += 1
                lat     += LATENCY["web_fetch"]
                l3_cache.add(uh)
                continue
            if uh in l3_cache and sim_age <= l3_ttl:
                l3_hits += 1
                lat     += LATENCY["l3_lookup"]
                if uh in stale_urls:
                    hit_stale = True
            else:
                fetches += 1
                lat     += LATENCY["web_fetch"]
                l3_cache.add(uh)

        lat += LATENCY["llm_generate"]
        latency_list.append(lat)

        if hit_stale:
            stale_errors += 1
            if is_base:
                base_stale_errors += 1

        l1_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

    base_lats = [lat for r, lat in zip(records, latency_list)
                 if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50

    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                l2_hits,
        "l3_hits":                l3_hits,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p50_base_ms":            p50_base,
        "p95_ms":                 p95,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits + l2_hits + l3_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_scalm(records: list, stale_urls: set, sim_age: float) -> dict:
    """
    SCALM baseline: frequency-weighted semantic caching with no TTL.
    Li et al., arXiv 2406.00025, 2024.
    """
    n            = len(records)
    cache        = []
    search_calls = 0
    fetches      = 0
    l1_hits      = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        q  = r["query"]
        fc = r["freshness_class"]
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        if fc == "REAL_TIME":
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(r["urls"])
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            latency_list.append(lat)
            continue

        best_entry = None
        best_freq  = -1
        best_sim   = 0.0
        for entry in cache:
            if entry["fc"] != fc:
                continue
            sim = cosine_sim(q, entry["query"])
            if sim >= L1_SIM_THRESHOLD:
                if (entry["freq"] > best_freq or
                        (entry["freq"] == best_freq and sim > best_sim)):
                    best_freq  = entry["freq"]
                    best_sim   = sim
                    best_entry = entry

        if best_entry:
            l1_hits += 1
            lat = LATENCY["l1_lookup"] + LATENCY["llm_generate"]
            best_entry["freq"] += 1
            if any(uh in stale_urls for uh in best_entry["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(r["urls"])
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            cache.append({
                "query":      q,
                "fc":         fc,
                "url_hashes": [u["url_hash"] for u in r["urls"]],
                "freq":       1,
            })

        latency_list.append(lat)

    base_lats = [lat for r, lat in zip(records, latency_list)
                 if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50

    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p50_base_ms":            p50_base,
        "p95_ms":                 p95,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_freshcache_no_calib(records: list, stale_urls: set, sim_age: float) -> dict:
    """
    FreshCache without calibration ablation: identical three-tier architecture
    and EPS thresholds as FreshCache_Full, but p_stale uses naive prior
    half-lives (HALF_LIFE_UNCAL) instead of MLE-calibrated values.
    """
    n        = len(records)
    l1_cache = []
    l2_cache = []
    l3_cache = set()

    search_calls      = 0
    fetches           = 0
    l1_hits           = 0
    l2_hits           = 0
    l3_hits           = 0
    validated         = 0
    latency_list      = []
    stale_errors      = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    def _jaccard(set_a, set_b):
        u = set_a | set_b
        return len(set_a & set_b) / len(u) if u else 0.0

    for r in records:
        q    = r["query"]
        fc   = r["freshness_class"]
        urls = r["urls"]
        lat  = 0.0
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        url_set_new = {u["url_hash"] for u in urls}

        # L1
        l1_hit = None
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l1_cache:
                sim = cosine_sim(q, entry["query"])
                ps  = p_stale_uncal(entry["fc"], sim_age, "answer")
                if sim >= L1_SIM_THRESHOLD and ps <= EPS_ANSWER and sim > best_sim:
                    best_sim = sim
                    l1_hit   = entry

        if l1_hit:
            l1_hits += 1
            lat += LATENCY["l1_lookup"]
            if any(uh in stale_urls for uh in l1_hit["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
            latency_list.append(lat)
            continue

        # L2
        l2_hit     = None
        url_hashes = [u["url_hash"] for u in urls]
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l2_cache:
                sim = cosine_sim(q, entry["query"])
                ps  = p_stale_uncal(entry["fc"], sim_age, "url_list")
                if sim >= L2_SIM_THRESHOLD and ps <= EPS_URL_LIST and sim > best_sim:
                    best_sim = sim
                    l2_hit   = entry

        if l2_hit:
            l2_hits += 1
            lat     += LATENCY["l2_lookup"]
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            lat          += LATENCY["search_api"]
            l2_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

        # L3
        hit_stale  = False
        n_validate = 0

        for u in urls:
            uh = u["url_hash"]
            if fc == "REAL_TIME":
                fetches += 1
                lat += LATENCY["web_fetch"]
                l3_cache.add(uh)
                continue
            if uh in l3_cache:
                ps = p_stale_uncal(fc, sim_age, "content")
                if ps <= EPS_CONTENT:
                    l3_hits += 1
                    lat     += LATENCY["l3_lookup"]
                    if uh in stale_urls:
                        hit_stale = True
                else:
                    n_validate += 1
                    validated  += 1
                    lat        += LATENCY["conditional_get"]
                    if uh in stale_urls:
                        fetches += 1
                        lat     += LATENCY["web_fetch"]
                        l3_cache.add(uh)
                    else:
                        l3_hits += 1
            else:
                fetches += 1
                lat     += LATENCY["web_fetch"]
                l3_cache.add(uh)

        lat += LATENCY["llm_generate"]
        latency_list.append(lat)

        if hit_stale:
            stale_errors += 1
            if is_base:
                base_stale_errors += 1

        l1_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50

    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                l2_hits,
        "l3_hits":                l3_hits,
        "validated_reuses":       validated,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p50_base_ms":            p50_base,
        "p95_ms":                 p95,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits + l2_hits + l3_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_freshcache_mlp(records: list, stale_urls: set, sim_age: float) -> dict:
    """
    FreshCache MLP ablation: identical three-tier architecture and EPS thresholds
    as FreshCache_Full, but p_stale uses the trained LearnedRiskModel (MLP).
    """
    _memo: dict = {}

    def _ps(fc_str: str, tier: str, url: str = "", query: str = "") -> float:
        key = (fc_str, tier, url, query)
        if key not in _memo:
            _memo[key] = p_stale_mlp(fc_str, sim_age, tier, url, query)
        return _memo[key]

    n        = len(records)
    l1_cache = []
    l2_cache = []
    l3_cache = set()

    search_calls      = 0
    fetches           = 0
    l1_hits           = 0
    l2_hits           = 0
    l3_hits           = 0
    validated         = 0
    latency_list      = []
    stale_errors      = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    def _jaccard(set_a, set_b):
        u = set_a | set_b
        return len(set_a & set_b) / len(u) if u else 0.0

    for r in records:
        q    = r["query"]
        fc   = r["freshness_class"]
        urls = r["urls"]
        lat  = 0.0
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        url_set_new = {u["url_hash"] for u in urls}

        # ── L1: answer cache ──────────────────────────────────────────────
        l1_hit = None
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l1_cache:
                sim = cosine_sim(q, entry["query"])
                ps  = _ps(entry["fc"], "answer", query=entry["query"])
                if sim >= L1_SIM_THRESHOLD and ps <= EPS_ANSWER and sim > best_sim:
                    if (_entity_match(q, entry["query"])
                            and semantic_equivalent(q, entry["query"], sim)):
                        best_sim = sim
                        l1_hit   = entry

        if l1_hit:
            l1_hits += 1
            lat += LATENCY["l1_lookup"]
            if any(uh in stale_urls for uh in l1_hit["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
            latency_list.append(lat)
            continue

        # ── L2: URL-list cache ────────────────────────────────────────────
        l2_hit     = None
        url_hashes = [u["url_hash"] for u in urls]
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l2_cache:
                sim = cosine_sim(q, entry["query"])
                ps  = _ps(entry["fc"], "url_list", query=entry["query"])
                if (sim >= L2_SIM_THRESHOLD and sim >= _L2_EQ_SIM_FLOOR
                        and ps <= EPS_URL_LIST and sim > best_sim):
                    best_sim = sim
                    l2_hit   = entry

        if l2_hit:
            l2_hits += 1
            lat     += LATENCY["l2_lookup"]
            l2_cache.append({"query": q, "fc": fc,
                             "url_hashes": l2_hit["url_hashes"]})
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            lat          += LATENCY["search_api"]
            l2_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

        # ── L3: content cache ─────────────────────────────────────────────
        hit_stale  = False
        n_validate = 0

        for u in urls:
            uh      = u["url_hash"]
            url_str = u.get("url", "")
            if fc == "REAL_TIME":
                fetches += 1
                lat += LATENCY["web_fetch"]
                l3_cache.add(uh)
                continue
            if uh in l3_cache:
                ps = _ps(fc, "content", url=url_str, query=q)
                if ps <= EPS_CONTENT:
                    l3_hits += 1
                    lat     += LATENCY["l3_lookup"]
                    if uh in stale_urls:
                        hit_stale = True
                else:
                    n_validate += 1
                    validated  += 1
                    lat        += LATENCY["conditional_get"]
                    if uh in stale_urls:
                        fetches += 1
                        lat     += LATENCY["web_fetch"]
                        l3_cache.add(uh)
                    else:
                        l3_hits += 1
            else:
                fetches += 1
                lat     += LATENCY["web_fetch"]
                l3_cache.add(uh)

        lat += LATENCY["llm_generate"]
        latency_list.append(lat)

        if hit_stale:
            stale_errors += 1
            if is_base:
                base_stale_errors += 1

        l1_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50

    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                l2_hits,
        "l3_hits":                l3_hits,
        "validated_reuses":       validated,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p50_base_ms":            p50_base,
        "p95_ms":                 p95,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits + l2_hits + l3_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_oracle(records: list, stale_urls: set, sim_age: float) -> dict:
    """
    OracleFreshness upper bound: same three-tier architecture as FreshCache
    but uses ground-truth knowledge of which URLs changed. Stale error is
    zero by construction.
    """
    n        = len(records)
    l1_cache = []
    l2_cache = []
    l3_cache = set()

    search_calls      = 0
    fetches           = 0
    l1_hits           = 0
    l2_hits           = 0
    l3_hits           = 0
    latency_list      = []
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        q    = r["query"]
        fc   = r["freshness_class"]
        urls = r["urls"]
        lat  = 0.0
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        url_hashes = [u["url_hash"] for u in urls]

        # ── L1: answer cache ──────────────────────────────────────────────
        l1_hit = None
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l1_cache:
                sim = cosine_sim(q, entry["query"])
                entry_is_fresh = not any(
                    uh in stale_urls for uh in entry["url_hashes"]
                )
                if sim >= L1_SIM_THRESHOLD and entry_is_fresh and sim > best_sim:
                    best_sim = sim
                    l1_hit   = entry

        if l1_hit:
            l1_hits += 1
            lat += LATENCY["l1_lookup"]
            latency_list.append(lat)
            continue

        # ── L2: URL-list cache ────────────────────────────────────────────
        l2_hit = None
        if fc != "REAL_TIME":
            best_sim = 0.0
            for entry in l2_cache:
                sim = cosine_sim(q, entry["query"])
                entry_is_fresh = not any(
                    uh in stale_urls for uh in entry["url_hashes"]
                )
                if sim >= L2_SIM_THRESHOLD and entry_is_fresh and sim > best_sim:
                    best_sim = sim
                    l2_hit   = entry

        if l2_hit:
            l2_hits += 1
            lat += LATENCY["l2_lookup"]
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            lat += LATENCY["search_api"]
            l2_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

        # ── L3: content cache ─────────────────────────────────────────────
        for u in urls:
            uh = u["url_hash"]
            if fc == "REAL_TIME":
                fetches += 1
                lat     += LATENCY["web_fetch"]
                l3_cache.add(uh)
                continue
            if uh in l3_cache and uh not in stale_urls:
                l3_hits += 1
                lat     += LATENCY["l3_lookup"]
            else:
                fetches += 1
                lat     += LATENCY["web_fetch"]
                l3_cache.add(uh)

        lat += LATENCY["llm_generate"]
        latency_list.append(lat)

        l1_cache.append({"query": q, "fc": fc, "url_hashes": url_hashes})

    base_lats = [lat for r, lat in zip(records, latency_list)
             if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50

    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                l2_hits,
        "l3_hits":                l3_hits,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p50_base_ms":            p50_base,
        "p95_ms":                 p95,
        "cache_stale_errors":     0,
        "cache_stale_error_rate": 0.0,
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


def sim_domain_ttl(records: list, stale_urls: set, sim_age: float) -> dict:
    """
    DomainTTL baseline: semantic cache with TTL scaled by observed domain volatility.
    TTL = FIXED_TTL[fc] * (1 - domain_vol), so high-volatility domains get shorter TTL.
    """
    import json as _json
    _rich = _json.load(open("data/query_rich_features.json"))

    n            = len(records)
    cache        = []
    search_calls = 0
    fetches      = 0
    l1_hits      = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    for r in records:
        q   = r["query"]
        fc  = r["freshness_class"]
        qid = r["query_id"]
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1
        base_ttl = FIXED_TTL.get(fc, 0)
        feats = _rich.get(qid, {})
        domain_vol = feats.get("mean_evidence_domain_vol", 0.5)
        ttl = base_ttl * max(0.05, 1.0 - domain_vol)

        hit = None
        if ttl > 0 and sim_age <= ttl:
            best_sim = 0.0
            for entry in cache:
                sim = cosine_sim(q, entry["query"])
                if sim >= L1_SIM_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    hit      = entry

        if hit:
            l1_hits += 1
            lat = LATENCY["l1_lookup"] + LATENCY["llm_generate"]
            if any(uh in stale_urls for uh in hit["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(r["urls"])
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            cache.append({"query": q, "fc": fc,
                          "url_hashes": [u["url_hash"] for u in r["urls"]]})
        latency_list.append(lat)

    base_lats = [lat for r, lat in zip(records, latency_list)
                 if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


TEMPORAL_KEYWORDS = {"current", "latest", "today", "now", "recent",
                     "currently", "this week", "this month", "live", "ongoing"}

def sim_temporal_keyword_ttl(records: list, stale_urls: set, sim_age: float) -> dict:
    """
    TemporalKeywordTTL baseline: semantic cache with TTL halved when the query
    contains temporal keywords (current, latest, today, etc.).
    """
    n            = len(records)
    cache        = []
    search_calls = 0
    fetches      = 0
    l1_hits      = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n            = 0

    def _has_temporal(q: str) -> bool:
        q_lower = q.lower()
        return any(kw in q_lower for kw in TEMPORAL_KEYWORDS)

    for r in records:
        q        = r["query"]
        fc       = r["freshness_class"]
        is_base  = not r.get("is_paraphrase")
        if is_base:
            base_n += 1
        base_ttl = FIXED_TTL.get(fc, 0)
        ttl      = base_ttl * 0.5 if _has_temporal(q) else base_ttl

        hit = None
        if ttl > 0 and sim_age <= ttl:
            best_sim = 0.0
            for entry in cache:
                sim = cosine_sim(q, entry["query"])
                if sim >= L1_SIM_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    hit      = entry

        if hit:
            l1_hits += 1
            lat = LATENCY["l1_lookup"] + LATENCY["llm_generate"]
            if any(uh in stale_urls for uh in hit["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
        else:
            search_calls += 1
            if is_base:
                base_search_calls += 1
            n_fetch = len(r["urls"])
            fetches += n_fetch
            lat = (LATENCY["search_api"] +
                   n_fetch * LATENCY["web_fetch"] +
                   LATENCY["llm_generate"])
            cache.append({"query": q, "fc": fc,
                          "url_hashes": [u["url_hash"] for u in r["urls"]]})
        latency_list.append(lat)

    base_lats = [lat for r, lat in zip(records, latency_list)
                 if not r.get("is_paraphrase")]
    latency_list.sort()
    p50      = latency_list[int(n * 0.50)]
    p95      = latency_list[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50
    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
    }


METHODS = [
    ("NoCache",              sim_nocache),
    ("ExactTTL",             sim_exact_ttl),
    ("SemanticTTL",          sim_semantic_ttl),
    ("DomainTTL",            sim_domain_ttl),
    ("TemporalKeywordTTL",   sim_temporal_keyword_ttl),
    ("TieredFixedTTL",       sim_tiered_fixed_ttl),
    ("L3Only",               sim_l3only),
    ("L3+ConditionalGet",    sim_l3_cget),
    ("vCache",               sim_vcache),
    ("SCALM",                sim_scalm),
    ("FreshCache_Full",      sim_freshcache),
    ("FreshCache_NoCalib",   sim_freshcache_no_calib),
    ("FreshCache_MLP",       sim_freshcache_mlp),
    ("OracleFreshness",      sim_oracle),
]

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: dict, label: str, n_queries: int) -> None:
    nc      = results["NoCache"]
    nc_sc   = nc["search_calls"]
    nc_b_sc = nc.get("base_search_calls", nc_sc)
    nc_bn   = nc.get("base_n", nc["n_queries"])
    nc_lat_base = nc["p50_base_ms"]

    print(f"\n  [{label}]  ({n_queries} queries)")
    print(f"  {'Method':<22} {'Search/q':>9} {'StaleErr':>9} "
          f"{'BaseSearch/q':>13} {'BaseStale':>10} {'Saved(all)':>11} {'Saved(base)':>12}")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*13} {'-'*10} {'-'*11} {'-'*12}")

    for method_name, _ in METHODS:
        m  = results[method_name]
        nq = m["n_queries"]
        bn = m.get("base_n", nq)
        if nq == 0:
            continue
        s_per_q      = m["search_calls"] / nq
        se           = m["cache_stale_error_rate"]
        b_sc         = m.get("base_search_calls", m["search_calls"])
        b_se_n       = m.get("base_stale_errors", m["cache_stale_errors"])
        total_hits   = m["l1_hits"] + m["l2_hits"] + m["l3_hits"]
        # Base stale rate: base stale errors over total hits (same denominator as all-query)
        b_stale_rate = b_se_n / max(total_hits, 1)
        saved_all    = f"{100*(1 - m['search_calls']/max(nc_sc, 1)):.1f}%"
        saved_base   = f"{100*(1 - b_sc/max(nc_b_sc, 1)):.1f}%"
        b_sc_per_q   = b_sc / max(bn, 1)
        print(f"  {method_name:<22} {s_per_q:>9.2f} {se:>9.3f} "
              f"{b_sc_per_q:>13.2f} {b_stale_rate:>10.3f} "
              f"{saved_all:>11} {saved_base:>12}")


def save_results(all_results: dict) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    rows = []
    for key, results in all_results.items():
        for method_name, _ in METHODS:
            if method_name not in results:
                continue
            m = results[method_name]
            if m["n_queries"] == 0:
                continue
            rows.append({
                "condition":            key,
                "method":               method_name,
                "p50_latency_ms":       m["p50_ms"],
                "p50_base_latency_ms":  m["p50_base_ms"],
                "p95_latency_ms":       m["p95_ms"],
                "stale_error_rate":     m["cache_stale_error_rate"],
                "search_per_query":     m["search_calls"] / m["n_queries"],
                "fetch_per_query":      m["fetches"] / m["n_queries"],
                "l1_hits":              m["l1_hits"],
                "l2_hits":              m["l2_hits"],
                "l3_hits":              m["l3_hits"],
                "base_search_calls":    m.get("base_search_calls", ""),
                "base_stale_errors":    m.get("base_stale_errors", ""),
                "base_n":               m.get("base_n", ""),
            })

    if rows:
        with open(PARETO_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def precompute_similarity_matrix(records: list) -> None:
    """
    Encode all query strings across all GPUs, compute the full pairwise
    cosine similarity matrix in one GPU matmul, cache to disk.
    """
    global _SIM_MATRIX, _QUERY_TO_IDX

    queries = [r["query"] for r in records]
    _QUERY_TO_IDX = {q: i for i, q in enumerate(queries)}

    # ── Load cached matrix if available ───────────────────────────────────
    if SIM_MATRIX_CACHE.exists():
        print(f"  Loading cached similarity matrix from {SIM_MATRIX_CACHE}...")
        _SIM_MATRIX = np.load(str(SIM_MATRIX_CACHE))
        print(f"  Matrix shape: {_SIM_MATRIX.shape}  "
              f"({_SIM_MATRIX.nbytes / 1e6:.1f} MB)")
        return

    # ── Encode across all GPUs ────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    n = len(queries)
    print(f"  Encoding {n} queries across GPUs {GPU_IDS} "
          f"(batch_size=1024 per GPU)...")

    model = SentenceTransformer("BAAI/bge-m3")
    # # just for ablation 
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    devices = [f"cuda:{gid}" for gid in GPU_IDS]
    pool    = model.start_multi_process_pool(target_devices=devices)

    embeddings = model.encode_multi_process(
        queries,
        pool,
        batch_size=1024,
        chunk_size=n // len(devices) + 1,
    )
    model.stop_multi_process_pool(pool)

    np.save(str(EMBEDDINGS_CACHE), embeddings)
    print(f"  Embeddings saved → {EMBEDDINGS_CACHE}")

    # ── Compute similarity matrix on GPU 0 ────────────────────────────────
    print(f"  Computing {n}×{n} similarity matrix on cuda:{GPU_IDS[0]}...")
    device = torch.device(f"cuda:{GPU_IDS[0]}")
    E = torch.tensor(embeddings, dtype=torch.float32, device=device)
    E = E / E.norm(dim=1, keepdim=True)
    S = (E @ E.T).cpu().numpy().astype(np.float32)
    del E
    torch.cuda.empty_cache()

    _SIM_MATRIX = S
    np.save(str(SIM_MATRIX_CACHE), S)
    print(f"  Similarity matrix saved → {SIM_MATRIX_CACHE}  "
          f"({S.nbytes / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*70}")
    print(f"  FreshCache Tiered Cache Experiment")
    print(f"  Encoder: {ENCODER}")
    print(f"{'='*70}\n")

    queries    = load_jsonl(QUERIES_FILE)
    manifest   = load_jsonl(MANIFEST_FILE)
    changes    = load_jsonl(CHANGE_FILE)
    paraphrases = load_jsonl(PARAPHRASE_FILE) if PARAPHRASE_FILE.exists() else []

    if not queries or not manifest:
        print("ERROR: Run collect.py and build_queries.py first.")
        sys.exit(1)

    print(f"  Queries  : {len(queries)}")
    print(f"  Manifest : {len(manifest)} records")
    print(f"  Changes  : {len(changes)} records")

    records = build_query_records(queries, manifest, paraphrases)
    print(f"\n  Pre-computing similarity matrix...")
    precompute_similarity_matrix(records)
    print(f"  Matrix ready — all cosine_sim calls are now O(1) lookups.\n")
    n_base  = sum(1 for r in records if not r.get("is_paraphrase"))
    n_para  = sum(1 for r in records if r.get("is_paraphrase"))
    print(f"  Base queries with URLs : {n_base}")
    print(f"  Paraphrase queries     : {n_para}")
    stale_map = build_stale_sets(changes)

    print(f"  Query records with URLs: {len(records)}")
    print(f"  Stale URLs at 1h  : {len(stale_map['rerun_1h'])}")
    print(f"  Stale URLs at 24h : {len(stale_map['rerun_24h'])}")

    all_results = {}

    for cond_name, sim_age in SIM_AGES.items():
        stale = stale_map["rerun_1h" if cond_name == "1h" else "rerun_24h"]

        print(f"\n{'='*70}")
        print(f"  Condition: t = {cond_name}")
        print(f"{'='*70}")

        cond_key = f"t={cond_name}_ALL"
        cond_results = {}
        for method_name, sim_fn in METHODS:
            cond_results[method_name] = sim_fn(records, stale, sim_age)
        all_results[cond_key] = cond_results
        print_table(cond_results, f"t={cond_name} — ALL CLASSES", len(records))

        # Per freshness class breakdown
        for fc in FRESHNESS_CLASSES:
            fc_records = [r for r in records if r["freshness_class"] == fc]
            if not fc_records:
                continue

            fc_key     = f"t={cond_name}_{fc}"
            fc_results = {}
            for method_name, sim_fn in METHODS:
                fc_results[method_name] = sim_fn(fc_records, stale, sim_age)
            all_results[fc_key] = fc_results
            print_table(fc_results, f"t={cond_name} — {fc}", len(fc_records))

    # Pareto summary
    print(f"\n{'='*70}")
    print(f"  Pareto Summary — FreshCache vs L3+cGET per freshness class")
    print(f"{'='*70}")
    for cond_name in SIM_AGES:
        print(f"\n  t={cond_name}:")
        for fc in FRESHNESS_CLASSES:
            key = f"t={cond_name}_{fc}"
            if key not in all_results:
                continue
            r   = all_results[key]
            fc_r  = r["FreshCache_Full"]
            base  = r["L3+ConditionalGet"]
            nc    = r["NoCache"]
            nq    = nc["n_queries"]
            if nq == 0:
                continue
            nc_lat = nc["p50_ms"]
            print(f"    {fc:<10} | "
                  f"L3+cGET stale={base['cache_stale_error_rate']:.3f} "
                  f"search_saved={100*(1-base['search_calls']/max(nc['search_calls'],1)):.1f}% | "
                  f"FreshCache stale={fc_r['cache_stale_error_rate']:.3f} "
                  f"search_saved={100*(1-fc_r['search_calls']/max(nc['search_calls'],1)):.1f}% "
                  f"lat_save={100*(1-fc_r['p50_base_ms']/nc['p50_base_ms']):.1f}%")

    save_results(all_results)
    print(f"\n  Saved: {RESULTS_FILE}")
    print(f"  Saved: {PARETO_FILE}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()