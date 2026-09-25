#!/usr/bin/env python3
"""
v16_exp12/mixed_engine.py — the mixed-age replay, instrumented for Experiments
1 and 2.

Reuses the existing implementation: cache tiers, semantic thresholds, the C18
equivalence gate, the entity guard, the risk model, tier multipliers, epsilons,
the Web-state step function (v9.version_at), the corrected execution path (an L2
hit's served list flows into L3), and three-valued CHANGED/UNCHANGED/
UNOBSERVABLE outcomes. Nothing about any policy is redesigned.

WHAT IS ADDED, AND ONLY THIS
  * per-ENTRY temporal rejection counters at L1, L2 and L3, kept strictly apart
    from per-REQUEST miss counts;
  * cache-hit age distributions per tier (median, p95);
  * two temporal-gate variants for Experiment 2:
      temporal="always"  the temporal test never rejects a semantically eligible
                         entry. Equivalence, entity, similarity, tiers,
                         timestamps and write semantics are untouched.
      temporal="equiv"   the TTL mathematically implied by the exponential rule,
                         age <= -h*ln(1-eps)/(m*ln2), on the same timestamps.

THREE TIMESTAMPS, KEPT DISTINCT (a validation requirement)
  url_discovery_t   an L2 entry's `t`. Propagated unchanged when an alias is
                    registered, so it is the time the URL list was first
                    discovered, not the time it was reused.
  l3_fetch_t        l3_cache[url] — the time that page's content was last
                    fetched or validated.
  l1_evidence_t     an L1 entry's `t` — the time the stored answer was
                    generated, which is the age of the oldest evidence standing
                    behind that answer under the corrected execution path.
  These are never conflated; assert_timestamps() checks they are all present and
  that an alias-propagated L2 entry is strictly older than the request that
  created it.

CONDITIONAL GETs
  The mixed-age replay has no conditional-GET path: on temporal expiry a cached
  page is refetched. That is inherited from v9/mixed_age_v2.py and is NOT
  changed here, because changing it would move the published reference numbers.
  `conditional_gets` is therefore 0 by construction in every mixed-age run and
  is reported as such rather than silently omitted.

SAFETY  Read-only on all previous artifacts.
"""
from __future__ import annotations

import math, statistics, sys, pathlib

import numpy as np
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"v13_corrected"))
sys.path.insert(0, str(ROOT/"v14_baselines"))
sys.path.insert(0, str(ROOT/"v9"))
import experiment as exp                      # noqa: E402
import corrected_engine as ce                 # noqa: E402
import engine_all as ea                       # noqa: E402
import mixed_age_v2 as ma                     # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS

# Experiment-2 variants are FreshCache_Full with one thing swapped.
PAIRS = []   # [Q-16] realized L1 reuse pairs; logging only

VARIANTS = {
    "FreshCache":                   ("FreshCache_Full", "risk"),
    "FreshCache_AlwaysPassTemporal": ("FreshCache_Full", "always"),
    "EquivalentTTL":                ("FreshCache_Full", "equiv"),
    "L2Only":                       ("L2Only", "risk"),
    "FreshCache_MLP":               ("FreshCache_MLP", "risk"),
    "SCALM":                        ("SCALM", "risk"),
    "SemanticTTL":                  ("SemanticTTL", "risk"),
    "TieredFixedTTL":               ("TieredFixedTTL", "risk"),
}


def equiv_ttl(fc, tier):
    """The age at which the exponential gate first refuses. v9.ttl_for, verbatim."""
    eps = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
           "content": exp.EPS_CONTENT}[tier]
    h = exp.HALF_LIFE.get(fc, 86_400.0)
    m = exp.TIER_MULT[tier]
    return -h*math.log(1-eps)/(m*math.log(2))


def outcome(uh, tc, tr, rounds):
    per = rounds.get(uh)
    if not per:
        return UNOBS
    a, b = per.get(ma.version_at(tc)), per.get(ma.version_at(tr))
    if not a or not b or not a["substantive"] or not b["substantive"]:
        return UNOBS
    return CHANGED if a["content_hash"] != b["content_hash"] else UNCHANGED


def replay(stream, rounds, variant, rich=None):
    """
    The similarity scan is vectorised -- one numpy row gather per request instead
    of a Python loop over the whole cache -- and the temporal test and the C18
    gates are then evaluated only on the candidate subset. Selection semantics
    are unchanged: the entry served is the highest-similarity entry that passes
    the temporal test and the gates, ties going to the earliest in cache order.

    Counter definitions, stated so they cannot be confused with request-level
    rates: a CANDIDATE is any cache entry whose similarity clears that tier's
    threshold; a TEMPORAL REJECTION is a candidate the temporal test refuses.
    Both are per-entry-scanned counts and live in entry_counters. Per-request
    miss counts live separately in request_counters.
    """
    name, temporal = VARIANTS[variant]
    PAIRS.clear()   # [Q-16]
    cfg = ea.CONFIG[name]
    l1m, l2m, l3m = cfg["l1"], cfg["l2"], cfg["l3"]
    use_l1, use_l2, use_l3 = l1m != "none", l2m != "none", l3m != "none"
    c18 = l1m in ("c18_risk", "c18_mlp")
    rich = rich or {}
    memo = {}
    n_max = len(stream)
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = exp.L1_SIM_THRESHOLD
    L2FLOOR = (max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
               if l2m.startswith("eq") else exp.L2_SIM_THRESHOLD)

    l1_qi = np.empty(n_max, dtype=np.int64)
    l1_t = np.empty(n_max, dtype=np.float64)
    l1_q, l1_fc, l1_urls, l1_freq = [], [], [], []
    n1 = 0
    l2_qi = np.empty(n_max, dtype=np.int64)
    l2_t = np.empty(n_max, dtype=np.float64)
    l2_fc, l2_urls, l2_q = [], [], []
    n2 = 0
    l3_cache = {}

    l1h = l2h = l3h = search = fetches = cget = 0
    rt_requests = 0
    per_req = []
    ages = defaultdict(list)
    ent = Counter()
    req = Counter()

    def risk(fc, age, tier, url="", query=""):
        if l1m == "c18_mlp" or l2m.endswith("mlp") or l3m.startswith("mlp"):
            k = (fc, round(age), tier, url, query)
            if k not in memo:
                memo[k] = exp.p_stale_mlp(fc, age, tier, url, query)
            return memo[k]
        if "uncal" in l1m or "uncal" in l2m:
            return exp.p_stale_uncal(fc, age, tier)
        return exp.p_stale(fc, age, tier)

    def temporal_ok(fc, age, tier, q="", url=""):
        """The ONLY thing Experiment 2 varies."""
        if temporal == "always":
            return True
        if temporal == "equiv":
            return age <= equiv_ttl(fc, tier)
        t = cfg["ttl"]
        if t == "halflife":
            return age <= exp.HALF_LIFE.get(fc, 86_400.0)*exp.TIER_MULT[tier]
        if t == "fixed":
            v = exp.FIXED_TTL.get(fc, 0)
            return bool(v and age <= v)
        eps = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
               "content": exp.EPS_CONTENT}[tier]
        return risk(fc, age, tier, url, q) <= eps

    for t, r in stream:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        qi = IDX[q]
        row = SIM[qi]
        own = [u["url_hash"] for u in urls]
        uobj = {u["url_hash"]: u for u in urls}
        req["requests"] += 1
        rt_bypass = cfg.get("rt_bypass", True)

        if fc == "REAL_TIME" and rt_bypass:
            rt_requests += 1
            req["realtime_bypass"] += 1
            search += 1
            for h in own:
                fetches += 1
                l3_cache[h] = t
            per_req.append((qid, None))
            continue

        hit1 = -1
        if use_l1 and fc != "REAL_TIME":
            req["reached_l1"] += 1
            if n1:
                sims = row[l1_qi[:n1]]
                if l1m == "scalm":
                    same = np.fromiter((f == fc for f in l1_fc), bool, n1)
                    cand = np.nonzero((sims >= L1T) & same)[0]
                    ent["l1_candidates"] += int(cand.size)
                    if cand.size:
                        fr = np.fromiter((l1_freq[j] for j in cand), np.int64,
                                         cand.size)
                        order = np.lexsort((-sims[cand], -fr))
                        hit1 = int(cand[order[0]])
                else:
                    cand = np.nonzero(sims >= L1T)[0]
                    ent["l1_candidates"] += int(cand.size)
                    if cand.size:
                        okm = np.fromiter(
                            (temporal_ok(l1_fc[j], t-l1_t[j], "answer", l1_q[j])
                             for j in cand), bool, cand.size)
                        ent["l1_temporal_rejections"] += int((~okm).sum())
                        keep = cand[okm]
                        if keep.size:
                            for j in keep[np.argsort(-sims[keep], kind="stable")]:
                                j = int(j)
                                if c18 and not (exp._entity_match(q, l1_q[j])
                                                and exp.semantic_equivalent(
                                                    q, l1_q[j], float(sims[j]))):
                                    ent["l1_semantic_rejections"] += 1
                                    continue
                                hit1 = j
                                break
        if hit1 >= 0:
            PAIRS.append({"query_id": qid, "incoming_query": q,          # [Q-16]
                          "matched_query": l1_q[hit1],
                          "similarity": float(sims[hit1]),
                          "freshness_class": fc,
                          "age_seconds": float(t - l1_t[hit1])})
            l1h += 1
            req["l1_served"] += 1
            ages["l1"].append(t-float(l1_t[hit1]))
            labs = [outcome(x, float(l1_t[hit1]), t, rounds) for x in l1_urls[hit1]]
            o = (CHANGED if CHANGED in labs else
                 (UNCHANGED if UNCHANGED in labs else UNOBS))
            if l1m == "scalm":
                l1_freq[hit1] += 1
            per_req.append((qid, o))
            continue
        if use_l1:
            req["l1_miss"] += 1

        hit2 = -1
        if use_l2 and fc != "REAL_TIME":
            req["reached_l2"] += 1
            if n2:
                sims2 = row[l2_qi[:n2]]
                cand = np.nonzero(sims2 >= L2FLOOR)[0]
                ent["l2_candidates"] += int(cand.size)
                if cand.size:
                    okm = np.fromiter(
                        (temporal_ok(l2_fc[j], t-l2_t[j], "url_list", l2_q[j])
                         for j in cand), bool, cand.size)
                    ent["l2_temporal_rejections"] += int((~okm).sum())
                    keep = cand[okm]
                    if keep.size:
                        hit2 = int(keep[np.argmax(sims2[keep])])

        if use_l2:
            if hit2 >= 0:
                l2h += 1
                req["l2_served"] += 1
                ages["l2"].append(t-float(l2_t[hit2]))
                served = list(l2_urls[hit2])
                disc_t = float(l2_t[hit2])
                if cfg["l2_reg_on_hit"]:
                    l2_qi[n2] = qi; l2_t[n2] = disc_t
                    l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
            else:
                req["l2_miss"] += 1
                search += 1
                served = list(own)
                disc_t = t
                l2_qi[n2] = qi; l2_t[n2] = t
                l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
        else:
            search += 1
            served = list(own)
            disc_t = t

        labs = []
        if use_l3:
            for x in served:
                u = uobj.get(x, {})
                if x in l3_cache:
                    ent["l3_candidates"] += 1
                    if temporal_ok(fc, t-l3_cache[x], "content", q,
                                   u.get("url", "")):
                        l3h += 1
                        ages["l3"].append(t-l3_cache[x])
                        labs.append(outcome(x, l3_cache[x], t, rounds))
                        continue
                    ent["l3_temporal_rejections"] += 1
                fetches += 1
                l3_cache[x] = t
        else:
            fetches += len(served)
        per_req.append((qid, (CHANGED if CHANGED in labs else
                              (UNCHANGED if UNCHANGED in labs else UNOBS))
                        if labs else None))
        if use_l1:
            l1_qi[n1] = qi; l1_t[n1] = t
            l1_q.append(q); l1_fc.append(fc); l1_urls.append(served)
            l1_freq.append(1); n1 += 1

    n = len(stream)
    q50 = lambda k: (statistics.median(ages[k])/3600 if ages[k] else None)
    q95 = lambda k: (sorted(ages[k])[int(.95*len(ages[k]))]/3600 if ages[k] else None)
    allh = ages["l1"]+ages["l2"]+ages["l3"]
    return {
        "variant": variant, "engine_method": name, "temporal_mode": temporal,
        "n_requests": n, "search_calls": search,
        "search_saved_pct": round(100*(1-search/n), 4),
        "search_avoided": n - search,
        "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h,
        "fetches": fetches, "conditional_gets": cget,
        "generations": search + l2h,
        "median_hit_age_h": {k: q50(k) for k in ("l1", "l2", "l3")},
        "p95_hit_age_h": {k: q95(k) for k in ("l1", "l2", "l3")},
        "median_hit_age_all_h": (statistics.median(allh)/3600 if allh else None),
        "p95_hit_age_all_h": (sorted(allh)[int(.95*len(allh))]/3600 if allh else None),
        "entry_counters": dict(ent), "request_counters": dict(req),
        "realtime_requests": rt_requests,
    }, per_req


def assert_run(m, stream):
    """Validation requirements, checked on every run."""
    errs = []
    if m["search_avoided"] != m["l1_hits"] + m["l2_hits"]:
        errs.append(f"search accounting: avoided {m['search_avoided']} != "
                    f"L1 {m['l1_hits']} + L2 {m['l2_hits']}")
    rt = sum(1 for _, r in stream if r["freshness_class"] == "REAL_TIME")
    if m["realtime_requests"] != rt:
        errs.append(f"REAL_TIME bypass: {m['realtime_requests']} != {rt}")
    if m["conditional_gets"] != 0:
        errs.append("mixed-age replay should have no conditional-GET path")
    return errs
