#!/usr/bin/env python3
"""
scripts/experiments/main/mixed_age_common_support.py -- the common-support comparison on the MIXED-AGE stream, plus an
intersection diagnostic for the fixed horizons.

Mixed-age common support S_mixed
  A request at stream time t is in S_mixed if it is not REAL_TIME and at least one of its OWN urls has a substantive
  body both at run_00 and at the round version_at(t) -- i.e. an outcome for that request could in principle have been
  determined, whatever any cache decided. Policy-independent, as at the fixed horizons.

Outcome for a reuse of url u cached at t_c and read at t_r is CHANGED / UNCHANGED / UNOBSERVABLE from the per-round
content table, never collapsed.

The stream model, version_at and arrival schedule come from freshcache/simulation/page_version_replay.py unchanged.

The intersection diagnostic reads the per-horizon outcome files horizon_outcomes_{1h,24h,7d}_<method>.json and excludes
methods that structurally produce no reuse outcomes (L2Only, NoCache), since including them forces an empty set.

Writes outputs/matched_answers/mixed_age_common_support.json.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import json, math, multiprocessing as mp, os, pathlib, statistics, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
HERE = ((OUTPUT_ROOT / "matched_answers"))
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); sys.path.insert(0, str(ROOT / "freshcache/simulation"))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine"))
import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import page_version_replay as ma             # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS
WORKERS = int(os.environ.get("MIXED_AGE_WORKERS", "8"))
POLICIES = ["FreshCache_Full", "FreshCache_MLP", "SemanticTTL", "TieredFixedTTL",
            "SCALM", "vCache", "L3Only", "L2Only"]


def load_rounds():
    t = {}
    for line in open((DATA_ROOT / "derived/round_table.jsonl"),
                     encoding="utf-8"):
        r = json.loads(line)
        t[r["url_hash"]] = r["rounds"]
    return t


def outcome(uh, tc, tr, rounds):
    per = rounds.get(uh)
    if not per:
        return UNOBS
    a, b = per.get(ma.version_at(tc)), per.get(ma.version_at(tr))
    if not a or not b or not a["substantive"] or not b["substantive"]:
        return UNOBS
    return CHANGED if a["content_hash"] != b["content_hash"] else UNCHANGED


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k/n, 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, (c-m)/d), min(1.0, (c+m)/d))


def replay(stream, rounds, policy):
    """
    Mixed-age replay with the shared execution path and per-request outcomes.
    Each method's L1 selection rule, staleness gate and registration rule are
    dispatched from method_replay.CONFIG, so SCALM's frequency weighting, vCache's
    online per-entry threshold and the MLP risk model are honoured rather than
    collapsed onto the rule-based gate.
    """
    cfg = ea.CONFIG[policy]
    l1mode, l2mode, l3mode = cfg["l1"], cfg["l2"], cfg["l3"]
    use_l1, use_l2, use_l3 = l1mode != "none", l2mode != "none", l3mode != "none"
    eqgate = l1mode in ("eqgate_risk", "eqgate_mlp")
    memo = {}
    l1_cache, l2_cache, l3_cache = [], [], {}
    l1h = l2h = l3h = search = fetches = 0
    per_req = []
    ages = defaultdict(list)

    def risk(fc, age, tier, url="", query=""):
        if l1mode == "eqgate_mlp" or l2mode.endswith("mlp") or l3mode.startswith("mlp"):
            k = (fc, round(age), tier, url, query)
            if k not in memo:
                memo[k] = exp.p_stale_mlp(fc, age, tier, url, query)
            return memo[k]
        if "uncal" in l1mode or "uncal" in l2mode:
            return exp.p_stale_uncal(fc, age, tier)
        return exp.p_stale(fc, age, tier)

    def ok(fc, age, tier, url="", query=""):
        t = cfg["ttl"]
        if t == "halflife":
            return age <= exp.HALF_LIFE.get(fc, 86_400.0)*exp.TIER_MULT[tier]
        if t == "fixed":
            v = exp.FIXED_TTL.get(fc, 0)
            return bool(v and age <= v)
        eps = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
               "content": exp.EPS_CONTENT}[tier]
        return risk(fc, age, tier, url, query) <= eps

    for t, r in stream:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        own = [u["url_hash"] for u in urls]
        uobj = {u["url_hash"]: u for u in urls}
        rt_bypass = cfg.get("rt_bypass", True)
        if fc == "REAL_TIME" and rt_bypass:
            search += 1
            for h in own:
                fetches += 1; l3_cache[h] = t
            per_req.append((r["query_id"], None))
            continue

        # ---- L1, per-method selection ----
        hit1, best = None, 0.0
        if use_l1 and fc != "REAL_TIME":
            if l1mode == "scalm":
                best_freq = -1
                for en in l1_cache:
                    if en["fc"] != fc:
                        continue
                    s_ = exp.cosine_sim(q, en["query"])
                    if s_ < exp.L1_SIM_THRESHOLD:
                        continue
                    if en["freq"] > best_freq or (en["freq"] == best_freq and s_ > best):
                        best_freq, best, hit1 = en["freq"], s_, en
            elif l1mode == "vcache":
                cand, bs = None, 0.0
                for en in l1_cache:
                    s_ = exp.cosine_sim(q, en["query"])
                    if s_ >= exp.L1_SIM_THRESHOLD and s_ > bs:
                        bs, cand = s_, en
                if cand is not None:
                    tstar = max(exp._vcache_threshold(cand["obs"], exp.EPS_ANSWER),
                                exp.L1_SIM_THRESHOLD)
                    if bs >= tstar:
                        hit1, best = cand, bs
            else:
                for en in l1_cache:
                    s_ = exp.cosine_sim(q, en["query"])
                    if s_ < exp.L1_SIM_THRESHOLD or s_ <= best:
                        continue
                    if not ok(en["fc"], t-en["t"], "answer", query=en["query"]):
                        continue
                    if eqgate and not (exp._entity_match(q, en["query"])
                                    and exp.semantic_equivalent(q, en["query"], s_)):
                        continue
                    best, hit1 = s_, en
        if hit1:
            l1h += 1
            ages["l1"].append(t-hit1["t"])
            labs = [outcome(x, hit1["t"], t, rounds) for x in hit1["url_hashes"]]
            o = (CHANGED if CHANGED in labs else
                 (UNCHANGED if UNCHANGED in labs else UNOBS))
            if l1mode == "scalm":
                hit1["freq"] += 1
            if l1mode == "vcache":
                hit1["obs"].append((best, 0 if o == CHANGED else 1))
            per_req.append((r["query_id"], o))
            continue

        # ---- L2 ----
        hit2, best2 = None, 0.0
        if use_l2 and fc != "REAL_TIME":
            floor = (max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
                     if l2mode.startswith("eq") else exp.L2_SIM_THRESHOLD)
            for en in l2_cache:
                s_ = exp.cosine_sim(q, en["query"])
                if s_ < floor or s_ <= best2:
                    continue
                if not ok(en["fc"], t-en["t"], "url_list", query=en["query"]):
                    continue
                best2, hit2 = s_, en
        if use_l2:
            if hit2:
                l2h += 1
                ages["l2"].append(t-hit2["t"])
                served = list(hit2["url_hashes"])
                if cfg["l2_reg_on_hit"]:
                    l2_cache.append({"query": q, "fc": fc, "url_hashes": served,
                                     "t": hit2["t"]})
            else:
                search += 1
                served = list(own)
                l2_cache.append({"query": q, "fc": fc, "url_hashes": own, "t": t})
        else:
            search += 1
            served = list(own)

        # ---- L3 on the SERVED list ----
        labs = []
        if use_l3:
            for x in served:
                u = uobj.get(x, {})
                if fc == "REAL_TIME" and rt_bypass:
                    fetches += 1; l3_cache[x] = t; continue
                if x in l3_cache and ok(fc, t-l3_cache[x], "content",
                                        url=u.get("url", ""), query=q):
                    l3h += 1
                    ages["l3"].append(t-l3_cache[x])
                    labs.append(outcome(x, l3_cache[x], t, rounds))
                else:
                    fetches += 1; l3_cache[x] = t
        else:
            fetches += len(served)
        per_req.append((r["query_id"],
                        (CHANGED if CHANGED in labs else
                         (UNCHANGED if UNCHANGED in labs else UNOBS)) if labs else None))
        if use_l1:
            l1_cache.append({"query": q, "fc": fc, "url_hashes": served, "t": t,
                             "freq": 1, "obs": []})

    n = len(stream)
    med = lambda k: statistics.median(ages[k])/3600 if ages[k] else 0.0
    return {"policy": policy, "n_requests": n, "search_calls": search,
            "search_saved_pct": round(100*(1-search/n), 2),
            "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "fetches": fetches,
            "generations": search + l2h,
            "median_l2_age_h": round(med("l2"), 2)}, per_req


_S = _RD = None


def _init(s, rd):
    global _S, _RD
    _S, _RD = s, rd


def _one(p):
    return p, replay(_S, _RD, p)


def main():
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rounds = load_rounds()
    stream = ma.build_stream(records)

    S = set()
    for t, r in stream:
        if r["freshness_class"] == "REAL_TIME":
            continue
        v = ma.version_at(t)
        for u in r["urls"]:
            per = rounds.get(u["url_hash"])
            if per and per.get("run_00", {}).get("substantive") \
                    and per.get(v, {}).get("substantive"):
                S.add(r["query_id"]); break
    print(f"  mixed-age stream {len(stream):,} requests; common support "
          f"|S_mixed| = {len(S):,} ({100*len(S)/len(stream):.1f}%)", flush=True)

    with mp.Pool(min(WORKERS, len(POLICIES)), initializer=_init,
                 initargs=(stream, rounds)) as pool:
        res = dict(pool.imap_unordered(_one, POLICIES))

    print(f"\n{'='*124}\n  MIXED AGE — COMMON SUPPORT |S| = {len(S):,}\n{'='*124}")
    print(f"  {'policy':<20}{'saved':>8}{'obs hits':>10}{'changed':>9}"
          f"{'drift (common)':>16}{'95% CI':>20}{'S coverage':>12}"
          f"{'fetches':>9}{'gens':>8}{'medL2 age':>11}")
    out = {"support_size": len(S), "n_stream": len(stream), "policies": {}}
    for p in POLICIES:
        m, per_req = res[p]
        insup = [o for qid, o in per_req if o is not None and qid in S]
        ch = sum(1 for o in insup if o == CHANGED)
        un = sum(1 for o in insup if o == UNCHANGED)
        obs = ch + un
        lo, hi = wilson(ch, obs)
        out["policies"][p] = {**m, "observable_hits_in_support": obs,
                              "changed_in_support": ch,
                              "drift_common_support": (ch/obs) if obs else None,
                              "drift_common_ci95": [round(lo, 5), round(hi, 5)],
                              "common_support_coverage": round(obs/len(S), 5)}
        d = (f"{100*ch/obs:.2f}%" if obs else "—")
        ci = (f"[{100*lo:.2f}%, {100*hi:.2f}%]" if obs else "—")
        print(f"  {p:<20}{m['search_saved_pct']:>7.1f}%{obs:>10,}{ch:>9,}{d:>16}"
              f"{ci:>20}{100*obs/len(S):>11.1f}%{m['fetches']:>9,}"
              f"{m['generations']:>8,}{m['median_l2_age_h']:>10.1f}h")

    # ---- intersection diagnostic for the fixed horizons (L2Only and NoCache excluded) ----
    print(f"\n{'='*104}\n  INTERSECTION DIAGNOSTIC (methods that "
          f"structurally produce no reuse\n  outcomes -- L2Only, NoCache -- are "
          f"excluded, since including them forces an empty set)\n{'='*104}")
    inter_out = {}
    for tag in ("1h", "24h", "7d"):
        sets, names = [], []
        for f in sorted(HERE.glob(f"horizon_outcomes_{tag}_*.json")):
            meth = f.stem.replace(f"horizon_outcomes_{tag}_", "")
            if meth in ("L2Only", "NoCache"):
                continue
            oc = json.load(open(f))
            s = {qid for qid, o in oc if o in (CHANGED, UNCHANGED)}
            if s:
                sets.append(s); names.append(meth)
        inter = set.intersection(*sets) if sets else set()
        inter_out[tag] = {"n_methods": len(names), "intersection": len(inter),
                          "methods": names}
        print(f"  {tag:<6}{len(names)} scoring methods -> "
              f"{len(inter):,} requests determinable by all of them")
    out["intersection_diagnostic"] = inter_out
    json.dump(out, open(HERE/"mixed_age_common_support.json", "w"), indent=2)
    print(f"\n  wrote mixed_age_common_support.json")


if __name__ == "__main__":
    main()
