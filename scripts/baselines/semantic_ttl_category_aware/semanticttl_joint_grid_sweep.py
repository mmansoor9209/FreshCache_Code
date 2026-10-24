#!/usr/bin/env python3
"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.


Baseline fairness check: does SemanticTTL improve when BOTH the L1 similarity
threshold theta and the TTL multiplier kappa are tuned, rather than the TTL
alone?

Validation-only sweep and selection.

  theta in [0.40, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]   (adds 0.60 and 0.85)
  kappa in [1/16, 1/8, 1/4, 1/2, 1, 2]                  (the existing grid)
  = 42 cells, SemanticTTL only, on the VALIDATION clusters only.

Everything else is reused unchanged: the frozen cluster-disjoint split, the
zipf_uniform seed-42 workload, the cached BGE-M3 embeddings, the temporal risk
model, the scripts/evaluation metrics and the validation-split stream construction
(build the full stream, then filter by cluster).

30 of the 42 cells are also produced by
scripts/baselines/semantic_ttl_category_aware/validation_split_tuning.py
(outputs/validation_split/validation_frontier.csv). Those cells are
recomputed here and ASSERTED to reproduce that table exactly, which validates
the harness before any new cell is trusted.

FreshCache is never re-tuned and its implementation is never touched; its frozen
validation reference is read from outputs/validation_split/.

Writes to outputs/semanticttl_joint_grid/.
CPU only, no GPU, no web/API calls, no generation, no judging.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import copy, csv, hashlib, json, os, pathlib, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = ((OUTPUT_ROOT / "semanticttl_joint_grid"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402
from schedule_robustness import support_of  # noqa: E402

CHANGED, UNCHANGED = ce.CHANGED, ce.UNCHANGED
SCHEDULE, SEED = "zipf_uniform", 42
THETAS = [0.40, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
KAPPAS = [1/16, 1/8, 1/4, 1/2, 1.0, 2.0]
STAGE1 = (OUTPUT_ROOT / "validation_split")
WATCH = ["freshcache/method/tiered_cache_model.py", "freshcache/selection_replay_engine/mixed_age_replay.py",
         "freshcache/selection_replay_engine/method_replay.py", "freshcache/simulation/request_schedules.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


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
    say("42-cell STAGE 1 — joint (theta, kappa) sweep on VALIDATION only")
    say(f"  thetas {THETAS}")
    say(f"  kappas {[round(k,4) for k in KAPPAS]}  -> {len(THETAS)*len(KAPPAS)} cells")

    split = json.load(open(STAGE1 / "split.json", encoding="utf-8"))
    val_c = set(split["validation_clusters"])
    sel1 = json.load(open(STAGE1 / "validation_selection.json", encoding="utf-8"))
    fcv = sel1["frozen_freshcache_validation"]
    say(f"  frozen FreshCache validation reference (read, not recomputed): "
        f"saved {fcv['search_saved_pct']:.4f}%  drift {fcv['drift_pct']:.4f}%  "
        f"cov {fcv['coverage_pct']:.4f}%")

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
    with open((DATA_ROOT / "derived/round_table.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    full = sc.build_stream(records, SCHEDULE, SEED)
    assert DS.check("primary_requests", len(full))
    vstream = [(t, r) for t, r in full
               if (r.get("cluster_id") or r["query_id"]) in val_c]
    assert len(vstream) == split["validation"]["requests"]
    S = support_of(vstream, rounds)
    assert len(S) == split["validation"]["support"]
    say(f"  validation stream {len(vstream):,} requests, |S| = {len(S):,} "
        f"(identical construction to Stage 1)")

    ORIG = {"FIXED_TTL": copy.deepcopy(exp.FIXED_TTL),
            "L1T": exp.L1_SIM_THRESHOLD}
    rows = []
    t0 = time.time()
    say(f"\n  {'theta':>6}{'kappa':>8}{'saved%':>10}{'drift%':>9}{'cov%':>9}"
        f"{'det':>7}{'chg':>6}{'L1 hits':>9}")
    for th in THETAS:
        for ka in KAPPAS:
            exp.L1_SIM_THRESHOLD = th
            exp.FIXED_TTL.update({c: v * ka for c, v in ORIG["FIXED_TTL"].items()})
            try:
                m, per = me.replay(vstream, rounds, "SemanticTTL", rich=rich)
            finally:
                exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG["FIXED_TTL"])
                exp.L1_SIM_THRESHOLD = ORIG["L1T"]
            r = score(m, per, S)
            r.update({"theta": th, "kappa": ka})
            rows.append(r)
            say(f"  {th:>6.2f}{ka:>8.4f}{r['search_saved_pct']:>9.4f}%"
                f"{r['drift_pct']:>8.4f}%{r['coverage_pct']:>8.4f}%"
                f"{r['determinable']:>7,}{r['changed']:>6,}{r['l1_hits']:>9,}")
    say(f"  swept {len(rows)} cells in {time.time()-t0:.0f}s")
    assert exp.FIXED_TTL == ORIG["FIXED_TTL"] and exp.L1_SIM_THRESHOLD == ORIG["L1T"]

    # ---- validate against outputs/validation_split on the 30 overlapping cells ----
    s1 = {(round(float(x["theta"]), 4), round(float(x["kappa"]), 4)): x
          for x in csv.DictReader(open(STAGE1 / "validation_frontier.csv",
                                       encoding="utf-8"))
          if x["method"] == "SemanticTTL"}
    checked = 0
    for r in rows:
        k = (round(r["theta"], 4), round(r["kappa"], 4))
        if k not in s1:
            continue
        for f in ("search_saved_pct", "drift_pct", "coverage_pct"):
            a, b = r[f], float(s1[k][f])
            assert abs(a - b) < 1e-4, f"cell {k} {f}: {a} != Stage1 {b}"
        assert r["determinable"] == int(s1[k]["determinable"])
        checked += 1
    say(f"  overlap check: {checked} cells reproduce Stage 1 exactly "
        f"({len(rows)-checked} cells are new to 42-cell)")

    # ---- selection, validation data ONLY ----
    def keys(r):
        return (-(r["coverage_pct"] or 0), r["fetches"], -r["theta"], r["kappa"])
    ok_f = [r for r in rows if r["drift_pct"] is not None
            and r["drift_pct"] <= fcv["drift_pct"]]
    fresh = (sorted(ok_f, key=lambda r: (-r["search_saved_pct"],) + keys(r))[0]
             if ok_f else None)
    ok_s = [r for r in rows if r["search_saved_pct"] >= fcv["search_saved_pct"]]
    save = (sorted(ok_s, key=lambda r: (r["drift_pct"],) + keys(r))[0]
            if ok_s else None)
    best_drift = sorted([r for r in rows if r["drift_pct"] is not None],
                        key=lambda r: (r["drift_pct"],) + keys(r))[0]

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

    say(f"\n  SELECTED ON VALIDATION ONLY (pre-registered Stage 1 rules)")
    for nm, r in (("freshness_matched", fresh), ("savings_matched", save),
                  ("min_drift_overall", best_drift)):
        if r is None:
            say(f"    {nm:<20} NO SUCH POINT"); continue
        say(f"    {nm:<20} theta={r['theta']:.2f} kappa={r['kappa']:.4f}  "
            f"saved {r['search_saved_pct']:.4f}%  drift {r['drift_pct']:.4f}%  "
            f"cov {r['coverage_pct']:.4f}%")
    say(f"    Pareto frontier: {len(front)} of {len(rows)} cells")

    orig = [r for r in rows if abs(r["theta"] - 0.40) < 1e-9
            and abs(r["kappa"] - 1/16) < 1e-9][0]
    say(f"    original baseline  theta=0.40 kappa=0.0625  "
        f"saved {orig['search_saved_pct']:.4f}%  drift {orig['drift_pct']:.4f}%")

    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post, "implementation file changed"
    say(f"\n  implementation files unchanged: True")

    cols = ["theta", "kappa", "search_saved_pct", "drift_pct", "coverage_pct",
            "determinable", "changed", "searches", "fetches", "generations",
            "l1_hits", "l2_hits", "l3_hits", "n_requests"]
    with open(HERE / "joint_validation_grid.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["theta"], x["kappa"])):
            w.writerow(r)
    json.dump({"thetas": THETAS, "kappas": KAPPAS, "n_cells": len(rows),
               "overlap_cells_verified_vs_stage1": checked,
               "freshcache_validation_reference": fcv,
               "selection_rules": sel1["selection_rules"],
               "selected": {"freshness_matched": fresh, "savings_matched": save,
                            "min_drift_overall": best_drift,
                            "original_theta040_kappa1_16": orig},
               "pareto_frontier": front,
               "validation_only": True, "test_touched": False,
               "hashes_pre": pre, "hashes_post": post},
              open(HERE / "joint_validation_selection.json", "w"), indent=2)
    (HERE / "run_stage1.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote joint_validation_grid.csv, joint_validation_selection.json")


if __name__ == "__main__":
    main()
