#!/usr/bin/env python3
"""
v13_corrected/corrected_engine.py — FreshCache and the baselines with corrected
execution semantics and three-valued outcome labels.

THE POLICY IS NOT CHANGED. Thresholds, epsilons, half-lives, tier multipliers,
the C18 equivalence gate, the entity gate, alias registration and the arrival
order are all the published ones, imported from experiment.py. Two things are
corrected:

1. EXECUTION PATH.
   After an L2 hit the exact served URL list flows into L3 and into generation.
   The published implementation iterated the INCOMING query's own URLs instead,
   on 100% of L2 hits (a2_provenance_audit.json). L3 may refresh or reuse a
   content VERSION but must never change URL IDENTITY, and that is asserted per
   request:
       L2 served URLs == L3 processed URLs == generator evidence URLs
   The assertion runs on every L2 hit unless explicitly disabled, and a failure
   aborts the run rather than being counted.

2. OUTCOME LABELS.
   Staleness is read from the corrected benchmark manifest as one of CHANGED /
   UNCHANGED / UNOBSERVABLE. An outcome with no observable evidence is excluded
   from both the numerator and the denominator of drift, and counted as
   coverage loss. It is never treated as clean.

DRIFT EVENT STRUCTURE, MIRRORING THE PUBLISHED SEMANTICS
  Each request contributes at most one outcome event:
    an L1 hit  -> one event over the cached answer's supporting URLs;
    otherwise  -> one event over the URLs REUSED from L3 without a refetch.
  Refetched URLs are fresh by construction and are not scored, exactly as in
  `sim_freshcache`. The event is stale if any scored URL is CHANGED, clean if
  at least one is UNCHANGED and none CHANGED, and UNOBSERVABLE otherwise.

SAFETY  Read-only on data/, experiment.py and the corrected manifests.
"""
from __future__ import annotations

import json, pathlib, sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import experiment as exp   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CHANGED, UNCHANGED, UNOBS = "CHANGED", "UNCHANGED", "UNOBSERVABLE"


class AssertionFailure(RuntimeError):
    pass


# Which methods this engine is responsible for, and their published tier
# structure and gates. Only methods with an L2 tier need the execution-path
# correction; every other baseline keeps its published implementation and is
# scored through the label-set protocol in a3_corrected_main.py.
#   c18   L1 applies the semantic-equivalence + entity gate (FreshCache only)
#   gate  how the staleness decision is made at each tier
METHOD_CONFIG = {
    # l2_floor          the similarity floor the published L2 tier uses
    # l2_register_on_hit whether an L2 HIT re-registers the served list
    "FreshCache_Full":    dict(l1=True,  l2=True, l3=True, c18=True,  gate="risk",
                               l2_floor="eq", l2_register_on_hit=True,
                               l3_mode="cget"),
    # NOTE: the published sim_freshcache_no_calib does NOT apply the C18
    # equivalence gate or the entity gate at L1 -- only the similarity
    # threshold and the naive staleness bound. So the "no calibration"
    # ablation actually removes two things at once. Mirrored here exactly,
    # and recorded in EXECUTION_PATH_AUDIT.md as an audit finding.
    # ...and its L2 tier uses the bare L2_SIM_THRESHOLD and registers only on
    # a MISS, unlike FreshCache_Full's 0.75 floor with registration on hit.
    "FreshCache_NoCalib": dict(l1=True,  l2=True, l3=True, c18=False, gate="uncal",
                               l2_floor="raw", l2_register_on_hit=False,
                               l3_mode="cget"),
    "FreshCache_MLP":     dict(l1=True,  l2=True, l3=True, c18=True,  gate="mlp",
                               l2_floor="eq", l2_register_on_hit=True,
                               l3_mode="cget"),
    # TieredFixedTTL uses the bare L2_SIM_THRESHOLD, not the 0.75 equivalence
    # floor, and registers into L2 only on a MISS. Both mirrored exactly.
    "TieredFixedTTL":     dict(l1=True,  l2=True, l3=True, c18=False,
                               # Its L3 REFETCHES on TTL expiry; it has no conditional-GET path.
                               gate="halflife_ttl", l2_floor="raw",
                               l2_register_on_hit=False, l3_mode="refetch"),
    "L2Only":             dict(l1=False, l2=True, l3=False, c18=False, gate="risk",
                               l2_floor="eq", l2_register_on_hit=True,
                               l3_mode="none"),
}


def load_labels(window: str) -> dict:
    """url_hash -> CHANGED / UNCHANGED / UNOBSERVABLE for run_00 -> window."""
    out = {}
    for line in open(HERE/"corrected_benchmark_manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["window"] == window:
            out[r["url_hash"]] = r["label"]
    return out


def score(urls, labels):
    """Three-valued outcome over a URL set. Never collapses UNOBSERVABLE."""
    seen_clean = False
    for u in urls:
        lab = labels.get(u, UNOBS)
        if lab == CHANGED:
            return CHANGED
        if lab == UNCHANGED:
            seen_clean = True
    return UNCHANGED if seen_clean else UNOBS


def replay(records, labels, sim_age, method="FreshCache_Full",
           assert_identity=True, use_mlp=False, log=None):
    """
    method: FreshCache_Full | FreshCache_NoCalib | FreshCache_MLP | L2Only |
            L3Only | L3+ConditionalGet | SemanticTTL | TieredFixedTTL | NoCache
    Only FreshCache uses the risk gate; the TTL baselines use the published
    fixed TTLs. L1/L2/L3 participation per method matches the published
    definitions.
    """
    n = len(records)
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = exp.L1_SIM_THRESHOLD
    L2FLOOR_EQ = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
    L2FLOOR_RAW = exp.L2_SIM_THRESHOLD
    EA, EU, EC = exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT
    LAT = exp.LATENCY
    _memo = {}

    if method not in METHOD_CONFIG:
        raise ValueError(
            f"{method} has no L2 tier and does not need the execution-path "
            f"correction; run its published implementation instead.")
    cfg = METHOD_CONFIG[method]
    use_l1, use_l2, use_l3 = cfg["l1"], cfg["l2"], cfg["l3"]
    use_c18 = cfg["c18"]
    gate = cfg["gate"]
    L2FLOOR = L2FLOOR_EQ if cfg["l2_floor"] == "eq" else L2FLOOR_RAW
    register_on_hit = cfg["l2_register_on_hit"]
    l3_mode = cfg.get("l3_mode", "cget")

    # TieredFixedTTL applies its TTL as a REQUEST-level gate keyed on the
    # INCOMING query's freshness class, before the cache is scanned at all --
    # not as a per-entry eligibility flag keyed on the entry's class. The two
    # are not equivalent and the published code does the former.
    request_level_ttl = (gate == "halflife_ttl")

    def ttl_allows(fc_incoming, tier):
        return sim_age <= exp.HALF_LIFE.get(fc_incoming, 86_400.0)*exp.TIER_MULT[tier]

    def tier_ok(fc, tier, url="", query=""):
        """The published staleness decision for this method at this tier."""
        if gate == "none" or request_level_ttl:
            return True
        eps = {"answer": EA, "url_list": EU, "content": EC}[tier]
        if gate == "uncal":
            return exp.p_stale_uncal(fc, sim_age, tier) <= eps
        if gate == "mlp":
            k = (fc, tier, url, query)
            if k not in _memo:
                _memo[k] = exp.p_stale_mlp(fc, sim_age, tier, url, query)
            return _memo[k] <= eps
        return exp.p_stale(fc, sim_age, tier) <= eps

    l1_qi = np.empty(n, dtype=np.int64); l1_ok = np.zeros(n, dtype=bool)
    l1_q, l1_urls = [], []
    n1 = 0
    l2_qi = np.empty(n, dtype=np.int64); l2_ok = np.zeros(n, dtype=bool)
    l2_urls = []
    n2 = 0
    l3_cache = set()

    search = fetches = l1h = l2h = l3h = validated = 0
    ev = Counter()
    lat_list = []
    assertions = 0

    for r in records:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qi = IDX[q]
        row = SIM[qi]
        own = [u["url_hash"] for u in urls]
        url_obj = {u["url_hash"]: u for u in urls}
        real_time = (fc == "REAL_TIME")
        lat = 0.0

        # ---------------- L1 ----------------
        hit1 = -1
        l1_allowed = (not request_level_ttl) or ttl_allows(fc, "answer")
        if use_l1 and not real_time and n1 and l1_allowed:
            sims = row[l1_qi[:n1]]
            cand = np.nonzero((sims >= L1T) & l1_ok[:n1])[0]
            if cand.size:
                order = cand[np.argsort(-sims[cand], kind="stable")]
                if not use_c18:
                    hit1 = int(order[0])
                else:
                    for j in order:
                        eq = l1_q[j]
                        if exp._entity_match(q, eq) and exp.semantic_equivalent(
                                q, eq, float(sims[j])):
                            hit1 = int(j); break
        if hit1 >= 0:
            l1h += 1
            lat += LAT["l1_lookup"]
            ev[score(l1_urls[hit1], labels)] += 1
            if log is not None:
                log.append({"tier": "L1", "query_id": r["query_id"], "query": q,
                            "matched_query": l1_q[hit1],
                            "similarity": round(float(sims[hit1]), 4),
                            "fc_incoming": fc,
                            "evidence_urls": list(l1_urls[hit1]),
                            "own_urls": list(own),
                            "outcome": score(l1_urls[hit1], labels)})
            lat_list.append(lat)
            continue

        # ---------------- L2 ----------------
        hit2 = -1
        l2_allowed = (not request_level_ttl) or ttl_allows(fc, "url_list")
        if use_l2 and not real_time and n2 and l2_allowed:
            sims = row[l2_qi[:n2]]
            elig = (sims >= L2FLOOR) & l2_ok[:n2]
            if elig.any():
                hit2 = int(np.where(elig, sims, -np.inf).argmax())

        if hit2 >= 0:
            l2h += 1
            lat += LAT["l2_lookup"]
            served = list(l2_urls[hit2])
        else:
            search += 1
            lat += LAT["search_api"]
            served = list(own)
        if use_l2 and (register_on_hit or hit2 < 0):
            l2_qi[n2] = qi
            l2_ok[n2] = tier_ok(fc, "url_list", query=q)
            l2_urls.append(served)
            n2 += 1

        # ---------------- L3 ----------------
        # CORRECTION: the list L2 served is what L3 processes and what the
        # generator sees. URL identity is fixed here; only the content VERSION
        # may be refreshed below.
        processed = list(served) if use_l3 else []
        reused = []
        for uh in processed:
            u = url_obj.get(uh, {})
            if real_time:
                fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh); continue
            if uh in l3_cache:
                ok = (ttl_allows(fc, "content") if request_level_ttl
                      else tier_ok(fc, "content", url=u.get("url", ""), query=q))
                if ok:
                    l3h += 1; lat += LAT["l3_lookup"]; reused.append(uh)
                elif l3_mode == "refetch":
                    # TieredFixedTTL: an expired entry is simply refetched.
                    fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh)
                else:
                    validated += 1; lat += LAT["conditional_get"]
                    if labels.get(uh) == CHANGED:
                        fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh)
                    else:
                        l3h += 1; reused.append(uh)
            else:
                fetches += 1; lat += LAT["web_fetch"]; l3_cache.add(uh)
        if not use_l3:
            fetches += len(served)
            lat += LAT["web_fetch"]*len(served)

        generator_evidence = list(processed) if use_l3 else list(served)
        if assert_identity and hit2 >= 0:
            # With an L3 tier the invariant is the full three-way identity.
            # L2Only has no L3 tier by design -- every page is fetched straight
            # from the served list -- so there the invariant is served ==
            # generator evidence, and `processed` is empty by construction.
            bad = (not (served == processed == generator_evidence) if use_l3
                   else not (served == generator_evidence and not processed))
            if bad:
                raise AssertionFailure(
                    f"URL identity broken on {r['query_id']} [{method}]: "
                    f"served={len(served)} processed={len(processed)} "
                    f"generator={len(generator_evidence)}")
            assertions += 1

        lat += LAT["llm_generate"]
        lat_list.append(lat)
        ev[score(reused, labels)] += 1
        if log is not None and hit2 >= 0:
            log.append({"tier": "L2", "query_id": r["query_id"], "query": q,
                        "similarity": round(float(sims[hit2]), 4),
                        "fc_incoming": fc,
                        "evidence_urls": list(generator_evidence),
                        "served_urls": list(served),
                        "own_urls": list(own),
                        "outcome": score(reused, labels)})

        if use_l1:
            l1_qi[n1] = qi
            l1_ok[n1] = tier_ok(fc, "answer", query=q)
            l1_q.append(q); l1_urls.append(served); n1 += 1

    stale, clean, unob = ev[CHANGED], ev[UNCHANGED], ev[UNOBS]
    observable = stale + clean
    total_ev = observable + unob
    tier_hits = l1h + l2h + l3h
    lat_list.sort()
    return {
        "method": method, "n_requests": n,
        "search_calls": search,
        "search_saved_pct": round(100*(1 - search/max(n, 1)), 3),
        "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "tier_hits": tier_hits,
        "fetches": fetches, "generations": search + l2h,
        "conditional_gets": validated,
        "outcome_events": total_ev,
        "stale_events": stale, "clean_events": clean,
        "unobservable_events": unob,
        "observable_events": observable,
        "drift_observable": (stale/observable) if observable else None,
        "observable_coverage": (observable/total_ev) if total_ev else None,
        "drift_if_unobservable_counted_clean": (stale/total_ev) if total_ev else None,
        "identity_assertions_passed": assertions,
        "p50_ms": lat_list[int(n*0.50)] if lat_list else 0,
        "p95_ms": lat_list[int(n*0.95)] if lat_list else 0,
    }
