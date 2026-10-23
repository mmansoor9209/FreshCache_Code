#!/usr/bin/env python3
"""
mixed_age_replay.py — the mixed-age replay, instrumented with per-entry
temporal-rejection counters and temporal-gate variants.

Reuses the existing implementation: cache tiers, semantic thresholds, the query
equivalence gate, the entity guard, the risk model, tier multipliers, epsilons,
the Web-state step function (page_version_replay.version_at), the replay execution path (an L2
hit's served list flows into L3), and three-valued CHANGED/UNCHANGED/
UNOBSERVABLE outcomes. Nothing about any policy is redesigned.

WHAT IS ADDED, AND ONLY THIS
  * per-ENTRY temporal rejection counters at L1, L2 and L3, kept strictly apart
    from per-REQUEST miss counts;
  * cache-hit age distributions per tier (median, p95);
  * two temporal-gate variants for the gate comparison:
      temporal="always"  the temporal test never rejects a semantically eligible
                         entry. Equivalence, entity, similarity, tiers,
                         timestamps and write semantics are untouched.
      temporal="equiv"   the TTL mathematically implied by the exponential rule,
                         age <= -h*ln(1-eps)/(m*ln2), on the same timestamps.

THREE TIMESTAMPS, KEPT DISTINCT (checked on every run)
  url_discovery_t   an L2 entry's `t`. Propagated unchanged when an alias is
                    registered, so it is the time the URL list was first
                    discovered, not the time it was reused.
  l3_fetch_t        l3_cache[url] — the time that page's content was last
                    fetched or validated.
  l1_evidence_t     an L1 entry's `t` — the time the stored answer was
                    generated, which is the age of the oldest evidence standing
                    behind that answer under the replay execution path.
  These are never conflated; assert_timestamps() checks they are all present and
  that an alias-propagated L2 entry is strictly older than the request that
  created it.

CONDITIONAL GETs
  The mixed-age replay has no conditional-GET path: on temporal expiry a cached
  page is refetched. That is inherited from freshcache/simulation/page_version_replay.py and is NOT
  changed here, because changing it would move the recorded reference numbers.
  `conditional_gets` is therefore 0 by construction in every mixed-age run and
  is reported as such rather than silently omitted.

SAFETY  Read-only on all previous artifacts.
"""
from __future__ import annotations

import os; import math, statistics, sys, pathlib

import numpy as np
from collections import Counter, defaultdict

ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()   # repository root
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method"))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine"))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation"))
sys.path.insert(0, str(ROOT / "freshcache/simulation"))
sys.path.insert(0, str(ROOT / "freshcache/replay_engine"))   # sibling copies take precedence
import tiered_cache_model as exp                      # noqa: E402
import execution_semantics as ce                 # noqa: E402
import method_replay as ea                       # noqa: E402
import page_version_replay as ma                     # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS

# Experiment-2 variants are FreshCache_Full with one thing swapped.
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
    """The age at which the exponential gate first refuses. page_version_replay.ttl_for, verbatim."""
    eps = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
           "content": exp.EPS_CONTENT}[tier]
    h = exp.HALF_LIFE.get(fc, 86_400.0)
    m = exp.TIER_MULT[tier]
    return -h*math.log(1-eps)/(m*math.log(2))


def round_at(per, t):
    """Latest round of the replay schedule (ma.RUNS) that is not later than t AND has a stored
    snapshot for this page; None if no such round exists. Never returns a future snapshot. A round that is
    missing for this page falls back to the latest earlier stored round instead of being UNOBSERVABLE."""
    best = None
    for name, age in ma.RUNS:
        if age > t:
            break
        if name in per:
            best = name
    return best


def outcome(uh, tc, tr, rounds):
    """Per-page drift between the version held at tc (cached / last validated) and the version available
    at serving time tr. Versions are chosen by round_at."""
    per = rounds.get(uh)
    if not per:
        return UNOBS
    ra, rb = round_at(per, tc), round_at(per, tr)
    if ra is None or rb is None:
        return UNOBS
    a, b = per[ra], per[rb]
    if not a.get("substantive") or not b.get("substantive"):
        return UNOBS
    return CHANGED if a["content_hash"] != b["content_hash"] else UNCHANGED


def aggregate(labs):
    """Request-level drift over the reused pages: CHANGED if any is observably changed; UNCHANGED only
    if every one is observable and unchanged; otherwise UNOBSERVABLE (undeterminable). None when nothing was
    reused. An earlier rule returned UNCHANGED when any page was UNCHANGED and none CHANGED."""
    if not labs:
        return None
    if CHANGED in labs:
        return CHANGED
    return UNCHANGED if all(l == UNCHANGED for l in labs) else UNOBS


# configurations whose L1 freshness time is the oldest supporting-evidence time (manuscript 3.1, 3.4).
# Baselines keep write-time TTL semantics.
def l1_te_oldest_evidence(config_name):
    return config_name.startswith("FreshCache")


# L1 t_e counts supporting PAGE versions only (the URL list's discovery time is excluded).
L1_TE_PAGES_ONLY = True
# FreshCache-family REAL_TIME requests bypass every tier: no L1/L2/L3 reads and no L1/L2/L3 writes.
RT_FULL_BYPASS = True


def request_outcome(row, t, rounds):
    """Request-level drift for a audit_replay.replay row: an L1 hit is scored from the evidence versions the
    stored answer actually used (stored_answer_ev); an L2 hit or miss from the pages reused from L3 (ev entries
    cached before t). Earlier scorers used the L1 entry time for every page."""
    if not row or row.get("tier") == "REAL_TIME":
        return None
    if row["tier"] == "L1":
        pairs = row.get("stored_answer_ev") or []
    else:
        pairs = [(u, tc) for u, tc in (row.get("ev") or []) if tc < t]
    return aggregate([outcome(u, tc, t, rounds) for u, tc in pairs])


def replay(stream, rounds, variant, rich=None, trace=None):   # [INSTR] trace: optional dict qid -> routing record
    """
    The similarity scan is vectorised -- one numpy row gather per request instead
    of a Python loop over the whole cache -- and the temporal test and the semantic
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
    cfg = ea.CONFIG[name]
    l1m, l2m, l3m = cfg["l1"], cfg["l2"], cfg["l3"]
    use_l1, use_l2, use_l3 = l1m != "none", l2m != "none", l3m != "none"
    eqgate = l1m in ("eqgate_risk", "eqgate_mlp")
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
    l1_ev = []            # per L1 entry: (url, time of the page version actually used)
    te_oldest = l1_te_oldest_evidence(name)
    n1 = 0
    l2_qi = np.empty(n_max, dtype=np.int64)
    l2_t = np.empty(n_max, dtype=np.float64)
    l2_fc, l2_urls, l2_q = [], [], []
    n2 = 0
    l3_cache = {}
    _l3src = {}           # [INSTR] class of the request that last fetched each page

    l1h = l2h = l3h = search = fetches = cget = 0
    rt_requests = 0
    per_req = []
    ages = defaultdict(list)
    ent = Counter()
    req = Counter()

    def risk(fc, age, tier, url="", query=""):
        if l1m == "eqgate_mlp" or l2m.endswith("mlp") or l3m.startswith("mlp"):
            k = (fc, round(age), tier, url, query)
            if k not in memo:
                memo[k] = exp.p_stale_mlp(fc, age, tier, url, query)
            return memo[k]
        if "uncal" in l1m or "uncal" in l2m:
            return exp.p_stale_uncal(fc, age, tier)
        return exp.p_stale(fc, age, tier)

    def temporal_ok(fc, age, tier, q="", url=""):
        """The only rule the temporal-gate comparison varies."""
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
        if RT_FULL_BYPASS and te_oldest:
            rt_bypass = True      # defensive no-op: FreshCache configs omit rt_bypass, so the default True already applies

        if fc == "REAL_TIME" and rt_bypass:
            rt_requests += 1
            req["realtime_bypass"] += 1
            search += 1
            for h in own:
                fetches += 1
                if not (RT_FULL_BYPASS and te_oldest):    # no L3 write for FreshCache REAL_TIME
                    l3_cache[h] = t
                    _l3src[h] = fc   # [INSTR]
            if trace is not None: trace[qid] = {"tier": "REAL_TIME_BYPASS", "fc": fc, "reused": []}   # [INSTR]
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
                                if eqgate and not (exp._entity_match(q, l1_q[j])
                                                and exp.semantic_equivalent(
                                                    q, l1_q[j], float(sims[j]))):
                                    ent["l1_semantic_rejections"] += 1
                                    continue
                                hit1 = j
                                break
        if hit1 >= 0:
            l1h += 1
            req["l1_served"] += 1
            ages["l1"].append(t-float(l1_t[hit1]))
            labs = [outcome(x, tc, t, rounds) for x, tc in l1_ev[hit1]]   # per-page provenance
            o = aggregate(labs)                                             # three-valued aggregation
            if o is None:
                o = UNOBS
            if l1m == "scalm":
                l1_freq[hit1] += 1
            if trace is not None: trace[qid] = {"tier": "L1", "fc": fc, "entry": int(hit1), "entry_fc": l1_fc[hit1], "reused": []}   # [INSTR]
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
        _ru = []              # [INSTR]
        evid = []             # (url, time of the version used) for every served page
        if use_l3:
            for x in served:
                u = uobj.get(x, {})
                if x in l3_cache:
                    ent["l3_candidates"] += 1
                    if temporal_ok(fc, t-l3_cache[x], "content", q,
                                   u.get("url", "")):
                        l3h += 1
                        ages["l3"].append(t-l3_cache[x])
                        _ru.append((x, _l3src.get(x)))   # [INSTR]
                        labs.append(outcome(x, l3_cache[x], t, rounds))
                        evid.append((x, float(l3_cache[x])))
                        continue
                    ent["l3_temporal_rejections"] += 1
                fetches += 1
                l3_cache[x] = t
                _l3src[x] = fc   # [INSTR]
                evid.append((x, float(t)))
        else:
            fetches += len(served)
            evid = [(x, float(t)) for x in served]
        if trace is not None: trace[qid] = {"tier": ("L2" if use_l2 and hit2 >= 0 else "miss"), "fc": fc, "entry": (int(hit2) if use_l2 and hit2 >= 0 else None), "entry_fc": (l2_fc[hit2] if use_l2 and hit2 >= 0 else None), "reused": _ru}   # [INSTR]
        per_req.append((qid, aggregate(labs)))                          # three-valued aggregation
        if use_l1:
            # FreshCache: t_e = oldest supporting evidence (page versions used and the URL list's
            # discovery time); generating the answer does not reset it. Baselines: write time, unchanged.
            if te_oldest and (use_l2 or use_l3):
                pages = [tc for _, tc in evid]
                te = min(pages) if L1_TE_PAGES_ONLY and pages else (min(pages + [float(disc_t)]) if pages else t)   # L1 t_e from supporting page versions only
            else:
                te = t
            l1_qi[n1] = qi; l1_t[n1] = te
            l1_q.append(q); l1_fc.append(fc); l1_urls.append(served)
            l1_ev.append(list(evid))
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
