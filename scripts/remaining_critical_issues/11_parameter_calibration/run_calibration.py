#!/usr/bin/env python3
"""
Section 11 -- principled calibration of the tier multipliers and error budgets.

A STRUCTURAL FACT, ESTABLISHED FIRST
  FreshCache's temporal gate is
        p_stale(fc, age, tier) = 1 - exp(-ln2 * m_tier * age / h_fc) <= eps_tier
  which is satisfied exactly when
        age <= h_fc * k_tier,      k_tier = -ln(1 - eps_tier) / (m_tier * ln2).
  The multiplier and the budget therefore enter the decision ONLY through the
  single scalar k_tier: they are not separately identifiable from any replay.
  Sweeping (m, eps) jointly would sweep a 2-D grid over a 1-D family. This
  experiment sweeps the identifiable parameter k_tier per tier and reports, for
  the selected point, the whole (m, eps) family that realises it.

PROTOCOL
  1. Pre-register the objective, the grid and the tie-breaks, hash the file,
     and print the hash BEFORE any cell is evaluated.
  2. Sweep on VALIDATION clusters only.
  3. Freeze ONE configuration.
  4. Evaluate it once on the held-out TEST clusters.
  The published primary FreshCache result is never altered; this is an optional
  validation-selected variant, reported alongside it.

Each k is realised by holding m_tier = 1.0 and setting
        eps_tier = 1 - 2 ** (-k_tier),
which reproduces the published configuration exactly at the published k values
(asserted before the sweep starts).
"""
from __future__ import annotations
import csv, hashlib, itertools, json, math, os, pathlib, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(BASE / "lib"))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402
from e1_robustness import support_of     # noqa: E402

CHANGED, UNCHANGED = me.CHANGED, me.UNCHANGED
SCHEDULE, SEED = "zipf_uniform", 42
LN2 = math.log(2.0)
TIERS = ("answer", "url_list", "content")
GRID = {"answer":   [0.05, 0.075, 0.1013, 0.15, 0.25, 0.40],
        "url_list": [0.10, 0.1750, 0.2682, 0.40, 0.60, 0.90],
        "content":  [0.25, 0.4000, 0.6215, 0.90, 1.30, 1.80]}
WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
         "v14_baselines/engine_all.py", "v16_exp12/schedules.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def k_of(eps, m):
    return -math.log(1.0 - eps) / (m * LN2)


def eps_of(k):
    return 1.0 - 2.0 ** (-k)


def score(m, per, S):
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == CHANGED)
    un = sum(1 for o in ins if o == UNCHANGED)
    det = ch + un
    return {"search_saved_pct": m["search_saved_pct"],
            "drift_pct": (100 * ch / det) if det else None,
            "coverage_pct": (100 * det / len(S)) if S else None,
            "determinable": det, "changed": ch,
            "searches": m["search_calls"], "fetches": m["fetches"],
            "generations": m["generations"], "l1_hits": m["l1_hits"],
            "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
            "n_requests": m["n_requests"]}


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("V3 SECTION 11 -- principled parameter calibration")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    PUB_K = {"answer": k_of(exp.EPS_ANSWER, exp.TIER_MULT["answer"]),
             "url_list": k_of(exp.EPS_URL_LIST, exp.TIER_MULT["url_list"]),
             "content": k_of(exp.EPS_CONTENT, exp.TIER_MULT["content"])}
    say("\n  published hand-set parameters and their identifiable form")
    for t, (e, m) in (("answer", (exp.EPS_ANSWER, exp.TIER_MULT["answer"])),
                      ("url_list", (exp.EPS_URL_LIST, exp.TIER_MULT["url_list"])),
                      ("content", (exp.EPS_CONTENT, exp.TIER_MULT["content"]))):
        say(f"    {t:<9} eps={e:<5} m={m:<4} -> k = {PUB_K[t]:.4f}   "
            f"(reuse while age <= {PUB_K[t]:.4f} x half-life)")

    # ---------------- pre-registration ----------------
    prereg = {
        "written_before_any_cell_was_evaluated": True,
        "identifiability": "the gate depends on (m, eps) only through "
                           "k = -ln(1-eps)/(m ln2); the sweep is over k",
        "realisation": "m_tier = 1.0, eps_tier = 1 - 2**(-k_tier)",
        "grid": GRID, "n_cells": len(GRID['answer'])*len(GRID['url_list'])*len(GRID['content']),
        "data": "VALIDATION clusters only; the test split is not touched "
                "until exactly one configuration has been frozen",
        "objective": "maximise validation search_saved_pct subject to "
                     "validation drift_pct <= DRIFT_BUDGET",
        "drift_budget": "the published FreshCache validation drift, read from "
                        "validation/heldout_baseline_tuning/validation_selection.json",
        "quality_constraint": "a WAI/C-WAI budget is NOT imposed: answer-level "
                              "quality cannot be evaluated for 216 cells "
                              "without 216 generation+judging runs. This is "
                              "declared here, before any result is seen, and "
                              "the selected point's answer quality is reported "
                              "separately rather than used for selection.",
        "tie_breaks": ["higher coverage_pct", "fewer fetches",
                       "smaller k_answer", "smaller k_url_list",
                       "smaller k_content"],
        "published_k": PUB_K,
        "published_point_is_in_grid": all(
            any(abs(v - PUB_K[t]) < 1e-3 for v in GRID[t]) for t in TIERS),
        "frozen_config_count": 1,
        "primary_freshcache_result": "UNCHANGED; this is an optional variant",
    }
    pj = HERE / "prereg.json"
    pj.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    say(f"\n  PRE-REGISTRATION written: {pj.name}")
    say(f"    sha256 {sha(pj)}")
    say(f"    objective   {prereg['objective']}")
    say(f"    grid        {prereg['n_cells']} cells; published point in grid: "
        f"{prereg['published_point_is_in_grid']}")
    say(f"    quality     {prereg['quality_constraint']}")

    # ---------------- data ----------------
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    full = sc.build_stream(records, SCHEDULE, SEED)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])
    val = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in val_c]
    test = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in test_c]
    Sv, St = support_of(val, rounds), support_of(test, rounds)
    say(f"\n  validation {len(val):,} requests |S|={len(Sv):,};  "
        f"test {len(test):,} requests |S|={len(St):,}")

    sel = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                         / "validation_selection.json", encoding="utf-8"))
    fcv = sel["frozen_freshcache_validation"]
    BUDGET = fcv["drift_pct"]
    say(f"  DRIFT BUDGET = published FreshCache validation drift = "
        f"{BUDGET:.4f}%  (saved {fcv['search_saved_pct']:.4f}%, "
        f"cov {fcv['coverage_pct']:.4f}%)")

    ORIG = (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT,
            dict(exp.TIER_MULT))

    def set_k(ka, ku, kc):
        exp.EPS_ANSWER = eps_of(ka)
        exp.EPS_URL_LIST = eps_of(ku)
        exp.EPS_CONTENT = eps_of(kc)
        exp.TIER_MULT.update({"answer": 1.0, "url_list": 1.0, "content": 1.0})

    def restore():
        (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT) = ORIG[:3]
        exp.TIER_MULT.clear(); exp.TIER_MULT.update(ORIG[3])

    # ---- realisation check: the published k must reproduce the published run ----
    set_k(PUB_K["answer"], PUB_K["url_list"], PUB_K["content"])
    try:
        m, per = me.replay(val, rounds, "FreshCache", rich=rich)
        r = score(m, per, Sv)
    finally:
        restore()
    ok = (abs(r["search_saved_pct"] - fcv["search_saved_pct"]) < 1e-4
          and abs(r["drift_pct"] - fcv["drift_pct"]) < 1e-4)
    say(f"\n  REALISATION GATE: re-expressing the published (m, eps) as "
        f"(m=1, eps=1-2^-k) reproduces the published validation result: {ok}")
    say(f"    saved {r['search_saved_pct']:.4f}% vs {fcv['search_saved_pct']:.4f}%   "
        f"drift {r['drift_pct']:.4f}% vs {fcv['drift_pct']:.4f}%")
    if not ok:
        say("  ABORT: the reparameterisation is not exact."); sys.exit(2)

    # ---------------- validation sweep ----------------
    cells = list(itertools.product(GRID["answer"], GRID["url_list"],
                                   GRID["content"]))
    say(f"\n  sweeping {len(cells)} validation cells ...")
    rows, t0 = [], time.time()
    for i, (ka, ku, kc) in enumerate(cells, 1):
        set_k(ka, ku, kc)
        try:
            m, per = me.replay(val, rounds, "FreshCache", rich=rich)
        finally:
            restore()
        d = score(m, per, Sv)
        d.update({"k_answer": ka, "k_url_list": ku, "k_content": kc,
                  "eps_answer_at_m1": eps_of(ka),
                  "eps_url_list_at_m1": eps_of(ku),
                  "eps_content_at_m1": eps_of(kc)})
        rows.append(d)
        if i % 20 == 0 or i == len(cells):
            say(f"    {i}/{len(cells)}  ({time.time()-t0:.0f}s)")

    # ---------------- selection, validation data only ----------------
    def tb(r):
        return (-(r["coverage_pct"] or 0), r["fetches"], r["k_answer"],
                r["k_url_list"], r["k_content"])
    feas = [r for r in rows if r["drift_pct"] is not None
            and r["drift_pct"] <= BUDGET]
    say(f"\n  feasible cells (validation drift <= {BUDGET:.4f}%): "
        f"{len(feas)} / {len(rows)}")
    if not feas:
        say("  NO configuration satisfies the pre-registered drift budget. "
            "Reported as such; the budget is NOT relaxed after seeing results.")
        chosen = None
    else:
        chosen = sorted(feas, key=lambda r: (-r["search_saved_pct"],) + tb(r))[0]
        say(f"  FROZEN configuration (selected on validation only):")
        say(f"    k_answer {chosen['k_answer']}  k_url_list "
            f"{chosen['k_url_list']}  k_content {chosen['k_content']}")
        say(f"    validation saved {chosen['search_saved_pct']:.4f}%  "
            f"drift {chosen['drift_pct']:.4f}%  cov {chosen['coverage_pct']:.4f}%")
        say(f"    realising (m, eps) families (any m gives the same policy):")
        for t, k in (("answer", chosen["k_answer"]),
                     ("url_list", chosen["k_url_list"]),
                     ("content", chosen["k_content"])):
            fam = ", ".join(f"m={mm}: eps={1-2**(-k*mm):.4f}"
                            for mm in (1.0, 1.2, 1.5, 2.0))
            say(f"      {t:<9} k={k:<7} -> {fam}")

    # Pareto frontier and where the published point sits
    def pareto(rs):
        out = []
        for r in rs:
            if r["drift_pct"] is None:
                continue
            dom = any(o["drift_pct"] is not None
                      and o["search_saved_pct"] >= r["search_saved_pct"]
                      and o["drift_pct"] <= r["drift_pct"]
                      and (o["search_saved_pct"] > r["search_saved_pct"]
                           or o["drift_pct"] < r["drift_pct"])
                      for o in rs if o is not r)
            if not dom:
                out.append(r)
        return sorted(out, key=lambda r: -r["search_saved_pct"])
    front = pareto(rows)
    pub = min(rows, key=lambda r: sum(abs(r[f"k_{t}"] - PUB_K[t])
                                      for t in TIERS))
    on_front = any(r is pub for r in front)
    say(f"\n  Pareto frontier: {len(front)} of {len(rows)} cells")
    say(f"  published hand-set point (k = "
        f"{pub['k_answer']}/{pub['k_url_list']}/{pub['k_content']}): "
        f"saved {pub['search_saved_pct']:.4f}%  drift {pub['drift_pct']:.4f}%")
    say(f"  is the published hand-set point ON the validation Pareto frontier? "
        f"{on_front}")
    if not on_front:
        dom = [r for r in front if r["search_saved_pct"] >= pub["search_saved_pct"]
               and r["drift_pct"] <= pub["drift_pct"]]
        for r in dom[:3]:
            say(f"    dominated by k={r['k_answer']}/{r['k_url_list']}/"
                f"{r['k_content']}: saved {r['search_saved_pct']:.4f}%  "
                f"drift {r['drift_pct']:.4f}%")

    # ---------------- ONE held-out evaluation ----------------
    test_out = None
    if chosen:
        say(f"\n  held-out TEST evaluation of the single frozen configuration")
        set_k(chosen["k_answer"], chosen["k_url_list"], chosen["k_content"])
        try:
            m, per = me.replay(test, rounds, "FreshCache", rich=rich)
            test_out = score(m, per, St)
        finally:
            restore()
        set_k(PUB_K["answer"], PUB_K["url_list"], PUB_K["content"])
        try:
            m2, per2 = me.replay(test, rounds, "FreshCache", rich=rich)
            test_pub = score(m2, per2, St)
        finally:
            restore()
        say(f"    published   saved {test_pub['search_saved_pct']:.4f}%  "
            f"drift {test_pub['drift_pct']:.4f}%  cov {test_pub['coverage_pct']:.4f}%  "
            f"L1 {test_pub['l1_hits']:,}  L2 {test_pub['l2_hits']:,}")
        say(f"    calibrated  saved {test_out['search_saved_pct']:.4f}%  "
            f"drift {test_out['drift_pct']:.4f}%  cov {test_out['coverage_pct']:.4f}%  "
            f"L1 {test_out['l1_hits']:,}  L2 {test_out['l2_hits']:,}")
        say(f"    delta       saved "
            f"{test_out['search_saved_pct']-test_pub['search_saved_pct']:+.4f} pp  "
            f"drift {test_out['drift_pct']-test_pub['drift_pct']:+.4f} pp")
        test_out["published_test_reference"] = test_pub
        say(f"    answer-level WAI for this variant was NOT used in selection "
            f"and is not measured here (pre-registered).")

    assert (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT) == ORIG[:3]
    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post
    say("\n  implementation files unchanged: True")

    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "prereg_sha256": sha(pj), "prereg": prereg,
               "drift_budget_pct": BUDGET,
               "published_validation_reference": fcv,
               "published_k": PUB_K, "n_cells": len(rows),
               "feasible_cells": len(feas), "frozen_selection": chosen,
               "pareto_frontier": front,
               "published_point_on_frontier": bool(on_front),
               "published_point_cell": pub,
               "heldout_test": test_out,
               "primary_freshcache_unchanged": True,
               "hashes": pre},
              open(HERE / "calibration_results.json", "w"), indent=2)
    cols = list(rows[0].keys())
    with open(HERE / "validation_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["k_answer"], x["k_url_list"],
                                             x["k_content"])):
            w.writerow(r)
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote prereg.json, validation_grid.csv, calibration_results.json")


if __name__ == "__main__":
    main()
