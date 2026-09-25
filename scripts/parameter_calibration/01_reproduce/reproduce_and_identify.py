#!/usr/bin/env python3
"""TASK 1 + the identifiability core.

(a) Reproduce the published FreshCache configuration and its anchors.
(b) Establish, analytically and numerically, which parameters the data can
    possibly identify.

The FreshCache gate admits a cached entry at tier `t` for class `fc` iff
    p_stale(fc, age, t) = 1 - exp(-ln2 * m_t * age / h_fc)  <=  eps_t
which is exactly
    age <= h_fc * k_t,      k_t = -ln(1 - eps_t) / (m_t * ln 2).
`ps(...) <= EPS_*` is the ONLY use of the risk model in the gate (engine_all
lines 318/340/385; mixed_engine equiv_ttl), so the multiplier and the risk
budget enter the policy ONLY through the single ratio k_t. Any (m, eps) pair
on the same k is the same policy. Nothing modified on disk.
"""
from __future__ import annotations
import itertools, json, math, os, pathlib, sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
ROOT = PC.parents[1]
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402
import prep2 as p2                       # noqa: E402

SEED, SCHEDULE = 42, "zipf_uniform"
TIERS = [("answer", "EPS_ANSWER", "L1"), ("url_list", "EPS_URL_LIST", "L2"),
         ("content", "EPS_CONTENT", "L3")]
ANCHOR = {"search_saved_pct": 60.5776, "l1_hits": 806, "l2_hits": 12429}
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def k_of(m, eps):
    return -math.log(1.0 - eps) / (m * math.log(2.0))


def main():
    say("TASK 1 -- published configuration, reproduction and identifiability")

    # ---------------- (a) the published configuration ----------------
    say("\n  == published parameters (read from experiment.py, unmodified) ==")
    say(f"    HALF_LIFE (calibrated, W2 per-class temporal holdout): "
        f"{ {k: round(v) for k, v in exp.HALF_LIFE.items()} }")
    say(f"    TIER_MULT: {exp.TIER_MULT}")
    say(f"    EPS: answer {exp.EPS_ANSWER}, url_list {exp.EPS_URL_LIST}, "
        f"content {exp.EPS_CONTENT}")

    say("\n  == the identified quantity: k_t = -ln(1-eps_t)/(m_t ln2) ==")
    say(f"    {'tier':<10}{'m':>6}{'eps':>7}{'k':>10}   equivalent TTL = h_fc * k")
    kpub = {}
    for tier, epsname, lab in TIERS:
        m, eps = exp.TIER_MULT[tier], getattr(exp, epsname)
        kpub[tier] = k_of(m, eps)
        say(f"    {lab+' '+tier:<10}{m:>6.2f}{eps:>7.2f}{kpub[tier]:>10.6f}")
    say(f"\n    {'class':<11}{'h_fc (s)':>12}" +
        "".join(f"{lab+' TTL':>14}" for _, _, lab in TIERS))
    for fc in ["REAL_TIME", "FAST", "MEDIUM", "SLOW", "TIMELESS"]:
        h = exp.HALF_LIFE[fc]
        cells = "".join(f"{me.equiv_ttl(fc, t)/3600:>11.2f} h"
                        for t, _, _ in TIERS)
        say(f"    {fc:<11}{h:>12,.0f}{cells}")

    # ---------------- reproduce the anchors ----------------
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c, val_c = set(split["test_clusters"]), set(split["validation_clusters"])
    assert not (test_c & val_c), "split is not cluster-disjoint"
    full = sc.build_stream(records, SCHEDULE, SEED)
    test = [(t, r) for t, r in full
            if (r.get("cluster_id") or r["query_id"]) in test_c]
    val = [(t, r) for t, r in full
           if (r.get("cluster_id") or r["query_id"]) in val_c]
    assert len(test) == split["test"]["requests"]
    assert len(val) == split["validation"]["requests"]
    say(f"\n  == cluster-disjoint split ==")
    say(f"    validation {len(val_c):,} clusters / {len(val):,} requests")
    say(f"    held-out   {len(test_c):,} clusters / {len(test):,} requests")

    m0, rows0 = p2.replay(test, {}, "FreshCache")
    bad = [k for k, v in ANCHOR.items()
           if abs(m0[k] - v) > (1e-4 if isinstance(v, float) else 0)]
    say(f"\n  == reproduction gate ==")
    if bad:
        say(f"    FAILED on {bad}: {m0}"); sys.exit(2)
    say(f"    held-out FreshCache reproduces: saved {m0['search_saved_pct']:.4f}%"
        f", L1 {m0['l1_hits']:,}, L2 {m0['l2_hits']:,}  -> PASS")

    # ---------------- (b) numerical non-identifiability ----------------
    say(f"\n  == numerical test: do (m, eps) pairs with equal k act identically? ==")
    say(f"    For scale c, m' = c*m and eps' = 1 - (1-eps)^c leave k unchanged.")
    base_mult = dict(exp.TIER_MULT)
    base_eps = {n: getattr(exp, n) for _, n, _ in TIERS}
    vec0 = [(q, r["tier"], tuple(r.get("served") or []),
             tuple(u for u, _ in (r.get("ev") or []))) for q, r in sorted(rows0.items())]
    results = []
    try:
        for c in (0.5, 2.0, 3.0):
            for tier, epsname, _ in TIERS:
                exp.TIER_MULT[tier] = base_mult[tier] * c
                setattr(exp, epsname, 1.0 - (1.0 - base_eps[epsname]) ** c)
            kk = {t: k_of(exp.TIER_MULT[t], getattr(exp, n)) for t, n, _ in TIERS}
            mc, rowsc = p2.replay(test, {}, "FreshCache")
            vecc = [(q, r["tier"], tuple(r.get("served") or []),
                     tuple(u for u, _ in (r.get("ev") or [])))
                    for q, r in sorted(rowsc.items())]
            same_k = all(abs(kk[t] - kpub[t]) < 1e-12 for t, _, _ in TIERS)
            ident = vecc == vec0
            results.append({"scale": c, "same_k": same_k, "identical": ident,
                            "mult": dict(exp.TIER_MULT),
                            "eps": {n: getattr(exp, n) for _, n, _ in TIERS},
                            "saved": mc["search_saved_pct"],
                            "l1": mc["l1_hits"], "l2": mc["l2_hits"]})
            say(f"    c={c:<4} m={[round(exp.TIER_MULT[t],3) for t,_,_ in TIERS]} "
                f"eps={[round(getattr(exp,n),4) for _,n,_ in TIERS]} "
                f"k unchanged {same_k} | saved {mc['search_saved_pct']:.4f}% "
                f"L1 {mc['l1_hits']:,} L2 {mc['l2_hits']:,} | "
                f"per-request decisions identical: {ident}")
    finally:
        exp.TIER_MULT.update(base_mult)
        for n, v in base_eps.items():
            setattr(exp, n, v)
    allsame = all(r["identical"] for r in results)
    say(f"\n    CONCLUSION: {'every' if allsame else 'NOT every'} rescaled pair "
        f"reproduced the published policy exactly.")
    say(f"    The tier multipliers and risk budgets are NOT separately "
        f"identifiable from any behavioural data. Only k_t is. Reporting")
    say(f"    (1.5, 1.2, 1.0) and (0.10, 0.20, 0.35) as two independent "
        f"parameter triples overstates the number of free parameters:")
    say(f"    FreshCache has THREE free temporal parameters, not six.")

    # sanity: restoration
    m1, _ = p2.replay(test, {}, "FreshCache")
    say(f"    restoration check: published anchor reproduces again: "
        f"{abs(m1['search_saved_pct']-ANCHOR['search_saved_pct'])<1e-4}")

    json.dump({"half_life": exp.HALF_LIFE, "tier_mult": exp.TIER_MULT,
               "eps": {n: getattr(exp, n) for _, n, _ in TIERS},
               "k_published": kpub,
               "equiv_ttl_hours": {fc: {lab: me.equiv_ttl(fc, t)/3600
                                        for t, _, lab in TIERS}
                                   for fc in exp.HALF_LIFE},
               "anchor_reproduced": True,
               "rescaling_tests": results,
               "separately_identifiable": False,
               "split": {"validation_clusters": len(val_c),
                         "validation_requests": len(val),
                         "test_clusters": len(test_c),
                         "test_requests": len(test)}},
              open(HERE / "published_config.json", "w"), indent=2)
    (PC / "logs" / "t1.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote published_config.json")


if __name__ == "__main__":
    main()
