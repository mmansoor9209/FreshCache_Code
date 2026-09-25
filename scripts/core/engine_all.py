#!/usr/bin/env python3
"""
v14_baselines/engine_all.py — every method under one corrected execution path
and one scoring rule, emitting a PER-REQUEST outcome so a common observable
support can be imposed.

Extends v13_corrected/corrected_engine.py to the full method list. Nothing about
any policy is redesigned: each method's tier structure, similarity gate, TTL
rule, registration rule and selection rule is mirrored from its published
implementation in experiment.py, and each is gate-checked against that
implementation before any result is believed.

WHY PER-REQUEST OUTCOMES
  Drift rates measured on different observable supports are not comparable, and
  the corrected 24h run showed supports ranging from 39.7% to 86.1% of a
  method's own outcomes. To compare temporally, every method must be scored on
  the SAME units. This engine therefore records, for each request, one of:
      CHANGED / UNCHANGED / UNOBSERVABLE  -- a reuse happened and was scored
      None                                -- no content was reused
  so any support set can be applied afterwards.

CORRECTED EXECUTION PATH
  After an L2 hit the served URL list flows into L3 and generation. Asserted per
  hit for every method that has both tiers.

SAFETY  Read-only on data/, experiment.py and the corrected manifests.
"""
from __future__ import annotations

import json, pathlib, sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"v13_corrected"))
import experiment as exp                      # noqa: E402
import corrected_engine as ce                 # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
V13 = ROOT/"v13_corrected"
CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS


class AssertionFailure(RuntimeError):
    pass


# l1: none | c18_risk | sim_ttl | scalm | vcache
#   c18_risk  FreshCache: similarity + C18 equivalence + entity + risk bound
#   sim_ttl   TTL families: similarity only, TTL gate on the INCOMING class
#   scalm     frequency-weighted, same-class only, no TTL
#   vcache    per-entry threshold learned online from correctness feedback
# l2: none | eq_risk | raw_risk | raw_ttl
# l3: none | risk | risk_cget | ttl_refetch
CONFIG = {
    "FreshCache_Full":    dict(l1="c18_risk", l2="eq_risk",  l3="risk_cget",
                               l2_reg_on_hit=True,  ttl=None, online=False),
    "FreshCache_MLP":     dict(l1="c18_mlp",  l2="eq_mlp",   l3="mlp_cget",
                               l2_reg_on_hit=True,  ttl=None, online=False),
    "FreshCache_NoCalib": dict(l1="sim_uncal", l2="raw_uncal", l3="uncal_cget",
                               l2_reg_on_hit=False, ttl=None, online=False, rt_bypass=True),
    # A TRUE calibration-only ablation: byte-identical configuration to
    # FreshCache_Full. The only difference is applied at run time, by swapping
    # exp.HALF_LIFE for the naive prior HALF_LIFE_UNCAL, so nothing else can
    # drift. c1_cleanup.py asserts the two config dicts are equal before running.
    "FreshCache_NoCalib_Clean": dict(l1="c18_risk", l2="eq_risk", l3="risk_cget",
                                     l2_reg_on_hit=True, ttl=None, online=False),
    "TieredFixedTTL":     dict(l1="sim_ttl",  l2="raw_ttl",  l3="ttl_refetch",
                               l2_reg_on_hit=False, ttl="halflife", online=False, rt_bypass=True),
    "SemanticTTL":        dict(l1="sim_ttl",  l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl="fixed", online=False, rt_bypass=True),
    # AUDIT: the published sim_domain_ttl looks up its volatility features by
    # query_id, but data/query_rich_features.json is keyed by query TEXT, so the
    # lookup returns {} for every query and domain_vol is always the 0.5 default.
    # The published DomainTTL is therefore a uniform TTL halving, not a
    # domain-adaptive policy. "DomainTTL" mirrors that exactly, for comparability
    # with the published results; "DomainTTL_keyfix" is the same policy with the
    # lookup repaired, reported separately and clearly marked.
    "DomainTTL":          dict(l1="sim_ttl",  l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl="domain", online=False, rt_bypass=True),
    "DomainTTL_keyfix":   dict(l1="sim_ttl",  l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl="domain_fixed", online=False, rt_bypass=True),
    # Sensitivity only, NOT the recommended replacement: the feature file is
    # keyed by BASE query text, so paraphrases still miss even after the key
    # fix. This variant falls back to the cluster's base query, which raises
    # coverage to 100% but is a semantic extension beyond a key repair.
    "DomainTTL_cluster":  dict(l1="sim_ttl",  l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl="domain_cluster",
                               online=False, rt_bypass=True),
    "TemporalKeywordTTL": dict(l1="sim_ttl",  l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl="tempkw", online=False, rt_bypass=True),
    "SCALM":              dict(l1="scalm",    l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl=None, online=False, rt_bypass=True),
    "vCache":             dict(l1="vcache",   l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl=None, online=True, rt_bypass=True),
    "L2Only":             dict(l1="none",     l2="eq_risk",  l3="none",
                               l2_reg_on_hit=True,  ttl=None, online=False),
    # AUDIT: sim_l3only and sim_l3_cget have NO REAL_TIME bypass in their URL
    # loops -- a REAL_TIME page that is already cached goes through the same
    # risk gate as any other, and for L3+cGET that means a conditional GET.
    # Mirrored exactly.
    "L3Only":             dict(l1="none",     l2="none",     l3="risk",
                               l2_reg_on_hit=False, ttl=None, online=False,
                               rt_bypass=False),
    # AUDIT: sim_l3_cget applies the RISK GATE FIRST and only issues a
    # conditional GET when the gate says the content may be stale -- it does not
    # validate every cached page. Its L3 block is identical to FreshCache's.
    # It does consult the change label on the validation path, so it is still
    # marked as requiring online feedback.
    "L3+ConditionalGet":  dict(l1="none",     l2="none",     l3="risk_cget",
                               l2_reg_on_hit=False, ttl=None, online=True,
                               rt_bypass=False),
    "NoCache":            dict(l1="none",     l2="none",     l3="none",
                               l2_reg_on_hit=False, ttl=None, online=False, rt_bypass=True),
}
ONLINE_FEEDBACK = [m for m, c in CONFIG.items() if c["online"]]
# query_id -> the cluster's base query text, for the DomainTTL_cluster variant
_CLUSTER_BASE: dict = {}


def set_cluster_base(records):
    """Map every request to its cluster's base query text."""
    global _CLUSTER_BASE
    base = {}
    for r in records:
        cid = r.get("cluster_id") or r["query_id"]
        if not r.get("is_paraphrase"):
            base[cid] = r["query"]
    _CLUSTER_BASE = {r["query_id"]: base.get(r.get("cluster_id") or r["query_id"], "")
                     for r in records}


def _rich_feats():
    """The same file sim_domain_ttl reads for mean_evidence_domain_vol."""
    p = ROOT/"data"/"query_rich_features.json"
    return json.load(open(p)) if p.exists() else {}


def score(urls, labels):
    seen = False
    for u in urls:
        lab = labels.get(u, UNOBS)
        if lab == CHANGED:
            return CHANGED
        if lab == UNCHANGED:
            seen = True
    return UNCHANGED if seen else UNOBS


def replay(records, labels, sim_age, method, assert_identity=True,
           rich=None, log=None):
    if method not in CONFIG:
        raise ValueError(method)
    cfg = CONFIG[method]
    n = len(records)
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = exp.L1_SIM_THRESHOLD
    L2_EQ = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
    L2_RAW = exp.L2_SIM_THRESHOLD
    EA, EU, EC = exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT
    LAT = exp.LATENCY
    rich = rich if rich is not None else {}
    memo = {}

    def ps(kind, fc, tier, url="", query=""):
        if kind.startswith("mlp") or kind.endswith("mlp"):
            k = (fc, tier, url, query)
            if k not in memo:
                memo[k] = exp.p_stale_mlp(fc, sim_age, tier, url, query)
            return memo[k]
        if "uncal" in kind:
            return exp.p_stale_uncal(fc, sim_age, tier)
        return exp.p_stale(fc, sim_age, tier)

    def ttl_for(fc, tier, q, qid):
        t = cfg["ttl"]
        if t == "halflife":
            return exp.HALF_LIFE.get(fc, 86_400.0)*exp.TIER_MULT[tier]
        if t == "fixed":
            return exp.FIXED_TTL.get(fc, 0)
        if t == "domain":
            # published behaviour: keyed by query_id, which never matches
            base = exp.FIXED_TTL.get(fc, 0)
            vol = rich.get(qid, {}).get("mean_evidence_domain_vol", 0.5)
            return base*max(0.05, 1.0-vol)
        if t == "domain_fixed":
            base = exp.FIXED_TTL.get(fc, 0)
            vol = rich.get(q, {}).get("mean_evidence_domain_vol", 0.5)
            return base*max(0.05, 1.0-vol)
        if t == "domain_cluster":
            base = exp.FIXED_TTL.get(fc, 0)
            f = rich.get(q) or rich.get(_CLUSTER_BASE.get(qid, ""), {})
            vol = f.get("mean_evidence_domain_vol", 0.5)
            return base*max(0.05, 1.0-vol)
        if t == "tempkw":
            base = exp.FIXED_TTL.get(fc, 0)
            ql = q.lower()
            hit = any(kw in ql for kw in exp.TEMPORAL_KEYWORDS)
            return base*0.5 if hit else base
        return None

    l1_qi = np.empty(n, dtype=np.int64); l1_ok = np.zeros(n, dtype=bool)
    l1_q, l1_urls, l1_fc, l1_freq, l1_obs = [], [], [], [], []
    n1 = 0
    l2_qi = np.empty(n, dtype=np.int64); l2_ok = np.zeros(n, dtype=bool)
    l2_urls = []
    n2 = 0
    l3_cache = set()

    search = fetches = l1h = l2h = l3h = validated = 0
    ev = Counter(); outcomes = []
    lat_list = []
    assertions = 0

    for r in records:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        qi = IDX[q]
        row = SIM[qi]
        own = [u["url_hash"] for u in urls]
        uobj = {u["url_hash"]: u for u in urls}
        rt = (fc == "REAL_TIME") and cfg.get("rt_bypass", True)
        rt_true = (fc == "REAL_TIME")
        lat = 0.0

        # ---------------- L1 ----------------
        hit1 = -1
        mode = cfg["l1"]
        if mode != "none" and not rt_true and n1:
            sims = row[l1_qi[:n1]]
            if mode in ("c18_risk", "c18_mlp", "sim_uncal"):
                cand = np.nonzero((sims >= L1T) & l1_ok[:n1])[0]
                if cand.size:
                    order = cand[np.argsort(-sims[cand], kind="stable")]
                    if mode == "sim_uncal":
                        hit1 = int(order[0])
                    else:
                        for j in order:
                            if exp._entity_match(q, l1_q[j]) and \
                                    exp.semantic_equivalent(q, l1_q[j], float(sims[j])):
                                hit1 = int(j); break
            elif mode == "sim_ttl":
                t = ttl_for(fc, "answer", q, qid)
                if t and sim_age <= t:
                    cand = np.nonzero(sims >= L1T)[0]
                    if cand.size:
                        hit1 = int(cand[np.argsort(-sims[cand], kind="stable")][0])
            elif mode == "scalm":
                best_freq, best_sim = -1, 0.0
                for j in range(n1):
                    if l1_fc[j] != fc or sims[j] < L1T:
                        continue
                    if (l1_freq[j] > best_freq
                            or (l1_freq[j] == best_freq and sims[j] > best_sim)):
                        best_freq, best_sim, hit1 = l1_freq[j], float(sims[j]), j
            elif mode == "vcache":
                cand = np.nonzero(sims >= L1T)[0]
                if cand.size:
                    j = int(cand[np.argsort(-sims[cand], kind="stable")][0])
                    # published default: delta = EPS_ANSWER
                    t_star = max(exp._vcache_threshold(l1_obs[j], exp.EPS_ANSWER),
                                 L1T)
                    if float(sims[j]) >= t_star:
                        hit1 = j

        if hit1 >= 0:
            l1h += 1
            lat += LAT["l1_lookup"] + LAT["llm_generate"]
            out = score(l1_urls[hit1], labels)
            ev[out] += 1
            outcomes.append((qid, out))
            if mode == "scalm":
                l1_freq[hit1] += 1
            if mode == "vcache":
                # vCache learns from the realised correctness label
                l1_obs[hit1].append(
                    (float(row[l1_qi[hit1]]),
                     0 if any(labels.get(u) == CHANGED for u in l1_urls[hit1]) else 1))
            if log is not None:
                log.append({"tier": "L1", "query_id": qid, "query": q,
                            "matched_query": l1_q[hit1],
                            "similarity": round(float(row[l1_qi[hit1]]), 4),
                            "fc_incoming": fc, "method": method,
                            "evidence_urls": list(l1_urls[hit1]),
                            "own_urls": list(own), "outcome": out})
            lat_list.append(lat)
            continue

        # ---------------- L2 ----------------
        hit2 = -1
        l2mode = cfg["l2"]
        if l2mode != "none" and not rt_true and n2:
            floor = L2_EQ if l2mode.startswith("eq") else L2_RAW
            allowed = True
            if l2mode == "raw_ttl":
                t = ttl_for(fc, "url_list", q, qid)
                allowed = bool(t and sim_age <= t)
            if allowed:
                sims2 = row[l2_qi[:n2]]
                elig = (sims2 >= floor) & l2_ok[:n2]
                if elig.any():
                    hit2 = int(np.where(elig, sims2, -np.inf).argmax())

        if l2mode != "none":
            if hit2 >= 0:
                l2h += 1; lat += LAT["l2_lookup"]
                served = list(l2_urls[hit2])
            else:
                search += 1; lat += LAT["search_api"]
                served = list(own)
            if cfg["l2_reg_on_hit"] or hit2 < 0:
                l2_qi[n2] = qi
                if l2mode == "raw_ttl":
                    l2_ok[n2] = True
                else:
                    l2_ok[n2] = ps(l2mode, fc, "url_list", query=q) <= EU
                l2_urls.append(served); n2 += 1
        else:
            search += 1; lat += LAT["search_api"]
            served = list(own)

        # ---------------- L3 ----------------
        l3mode = cfg["l3"]
        processed = list(served) if l3mode != "none" else []
        reused = []
        if l3mode == "none":
            fetches += len(served); lat += LAT["web_fetch"]*len(served)
        else:
            for uh in processed:
                u = uobj.get(uh, {})
                if rt:
                    fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh); continue
                if uh in l3_cache:
                    if l3mode == "ttl_refetch":
                        t = ttl_for(fc, "content", q, qid)
                        ok = bool(t and sim_age <= t)
                    else:
                        ok = ps(l3mode, fc, "content", url=u.get("url", ""),
                                query=q) <= EC
                    if ok:
                        l3h += 1; lat += LAT["l3_lookup"]; reused.append(uh)
                    elif l3mode in ("ttl_refetch", "risk"):
                        # L3Only and TieredFixedTTL refetch on expiry; only the
                        # cget variants issue a conditional GET.
                        fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh)
                    else:
                        validated += 1; lat += LAT["conditional_get"]
                        if labels.get(uh) == CHANGED:
                            fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh)
                        else:
                            l3h += 1; reused.append(uh)
                else:
                    fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh)

        gen_evidence = list(processed) if l3mode != "none" else list(served)
        if assert_identity and hit2 >= 0:
            bad = (not (served == processed == gen_evidence) if l3mode != "none"
                   else not (served == gen_evidence and not processed))
            if bad:
                raise AssertionFailure(f"URL identity broken [{method}] {qid}")
            assertions += 1

        lat += LAT["llm_generate"]
        lat_list.append(lat)
        if reused:
            out = score(reused, labels)
            ev[out] += 1
            outcomes.append((qid, out))
        else:
            outcomes.append((qid, None))
        if log is not None and hit2 >= 0:
            log.append({"tier": "L2", "query_id": qid, "query": q,
                        "similarity": round(float(row[l2_qi[hit2]]), 4),
                        "fc_incoming": fc, "method": method,
                        "evidence_urls": list(gen_evidence),
                        "served_urls": list(served), "own_urls": list(own),
                        "outcome": (score(reused, labels) if reused else None)})

        if cfg["l1"] != "none":
            l1_qi[n1] = qi
            m1 = cfg["l1"]
            if m1 in ("c18_risk", "c18_mlp", "sim_uncal"):
                l1_ok[n1] = ps(m1, fc, "answer", query=q) <= EA
            else:
                l1_ok[n1] = True
            l1_q.append(q); l1_urls.append(served); l1_fc.append(fc)
            l1_freq.append(1); l1_obs.append([]); n1 += 1

    stale, clean, unob = ev[CHANGED], ev[UNCHANGED], ev[UNOBS]
    observable = stale + clean
    tier_hits = l1h + l2h + l3h
    lat_list.sort()
    return {
        "method": method, "n_requests": n, "search_calls": search,
        "search_saved_pct": round(100*(1 - search/max(n, 1)), 3),
        "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "tier_hits": tier_hits,
        "fetches": fetches, "generations": search + l2h,
        "conditional_gets": validated,
        "stale_events": stale, "clean_events": clean,
        "unobservable_events": unob, "observable_events": observable,
        "reuse_events": observable + unob,
        "drift_observable": (stale/observable) if observable else None,
        "native_coverage": (observable/(observable+unob)) if (observable+unob) else None,
        "identity_assertions_passed": assertions,
        "requires_online_feedback": cfg["online"],
        "p50_ms": lat_list[int(n*0.50)] if lat_list else 0,
        "p95_ms": lat_list[int(n*0.95)] if lat_list else 0,
    }, outcomes
