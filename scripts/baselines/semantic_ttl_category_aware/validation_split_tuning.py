#!/usr/bin/env python3
"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

Validation-split tuning of the SemanticTTL/CategoryAware scripts/baselines (stage 1 only).

Runs frozen FreshCache once and sweeps three scripts/baselines on the validation
clusters only. The held-out test split is never replayed or scored here.

The complete stream is built once with schedule=zipf_uniform and
seed=42, exactly as in the reference replay, then filtered by cluster
assignment. Original timestamps, cache ages and snapshot rounds are preserved.
Each split starts from empty L1/L2/L3 state; only validation is replayed here.

Baseline mechanisms are unchanged. Every swept constant is read dynamically
inside mixed_age_replay.replay, so this script patches `experiment` module globals
per cell and restores them in a finally block. No implementation file is edited.

CPU only: no GPU, no web or API calls, no answer generation, no judging.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import copy; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
import csv
import hashlib
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = ((OUTPUT_ROOT / "validation_split"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))

os.chdir(ROOT)  # project paths in tiered_cache_model.py are relative to the repo root

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402
from schedule_robustness import support_of  # noqa: E402

CHANGED, UNCHANGED = ce.CHANGED, ce.UNCHANGED
SCHEDULE, SEED = "zipf_uniform", 42

THETA_STTL = [0.40, 0.55, 0.70, 0.80, 0.90, 0.95]
KAPPA_STTL = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0]
THETA_SCALM = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
KAPPA_TIERED = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0]

WATCH = ["freshcache/method/tiered_cache_model.py", "freshcache/selection_replay_engine/mixed_age_replay.py",
         "freshcache/selection_replay_engine/method_replay.py", "freshcache/simulation/request_schedules.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def assert_frozen_freshcache():
    """Runtime assertion that the deployed FreshCache setting is untouched."""
    checks = {
        "L1_SIM_THRESHOLD": (exp.L1_SIM_THRESHOLD, 0.40),
        "_L2_EQ_SIM_FLOOR": (exp._L2_EQ_SIM_FLOOR, 0.75),
        "EPS_ANSWER": (exp.EPS_ANSWER, 0.10),
        "EPS_URL_LIST": (exp.EPS_URL_LIST, 0.20),
        "EPS_CONTENT": (exp.EPS_CONTENT, 0.35),
        "TIER_MULT.answer": (exp.TIER_MULT["answer"], 1.5),
        "TIER_MULT.url_list": (exp.TIER_MULT["url_list"], 1.2),
        "TIER_MULT.content": (exp.TIER_MULT["content"], 1.0),
        "HALF_LIFE.TIMELESS": (exp.HALF_LIFE["TIMELESS"], 30 * 86400.0),
        "HALF_LIFE.SLOW": (exp.HALF_LIFE["SLOW"], 993600.0),
        "HALF_LIFE.MEDIUM": (exp.HALF_LIFE["MEDIUM"], 457920.0),
        "HALF_LIFE.FAST": (exp.HALF_LIFE["FAST"], 449280.0),
        "HALF_LIFE.REAL_TIME": (exp.HALF_LIFE["REAL_TIME"], 30.0),
        "_EQ_SIM_FLOOR": (exp._EQ_SIM_FLOOR, 0.80),
        "_EQ_JACCARD_MIN": (exp._EQ_JACCARD_MIN, 0.30),
    }
    bad = {k: v for k, (v, want) in checks.items() if abs(v - want) > 1e-9}
    assert not bad, f"FreshCache configuration is not frozen: {bad}"
    for name in ("_entity_match", "semantic_equivalent"):
        assert callable(getattr(exp, name, None)), f"missing {name}"
    cfg = ea.CONFIG["FreshCache_Full"]
    assert cfg == {"l1": "eqgate_risk", "l2": "eq_risk", "l3": "risk_cget",
                   "l2_reg_on_hit": True, "ttl": None, "online": False}, \
        f"FreshCache_Full config changed: {cfg}"
    return {k: v for k, (v, _) in checks.items()}


def score(m, per, S, n):
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
            "n_requests": m["n_requests"],
            "realtime_requests": m["realtime_requests"]}


def pareto(rows):
    """Nondominated in (maximize savings, minimize drift)."""
    out = []
    for r in rows:
        if r["drift_pct"] is None:
            continue
        dominated = any(
            (o["drift_pct"] is not None
             and o["search_saved_pct"] >= r["search_saved_pct"]
             and o["drift_pct"] <= r["drift_pct"]
             and (o["search_saved_pct"] > r["search_saved_pct"]
                  or o["drift_pct"] < r["drift_pct"]))
            for o in rows if o is not r)
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: -r["search_saved_pct"])


def select(rows, fc_drift, fc_saved):
    """Pre-registered selection rules, fixed before any result was seen.

    Tie-breaks, in order: higher coverage, fewer fetches, stricter semantic
    threshold (higher theta). Where a method's theta is fixed, the
    deterministic fallback is the stricter temporal setting (smaller kappa).
    """
    def keys(r):
        return (-(r["coverage_pct"] or 0), r["fetches"],
                -(r["theta"] if r["theta"] is not None else 0),
                (r["kappa"] if r["kappa"] is not None else 0))

    ok_fresh = [r for r in rows
                if r["drift_pct"] is not None and r["drift_pct"] <= fc_drift]
    fresh = (sorted(ok_fresh, key=lambda r: (-r["search_saved_pct"],) + keys(r))[0]
             if ok_fresh else None)
    ok_save = [r for r in rows
               if r["drift_pct"] is not None and r["search_saved_pct"] >= fc_saved]
    save = (sorted(ok_save, key=lambda r: (r["drift_pct"],) + keys(r))[0]
            if ok_save else None)
    return fresh, save


def main():
    pre = {f: sha(ROOT / f) for f in WATCH}
    print("  pre-run hashes:")
    for f, h in pre.items():
        print(f"    {f:<32}{h[:16]}")

    split = json.load(open(HERE / "split.json", encoding="utf-8"))
    val_clusters = set(split["validation_clusters"])
    test_clusters = set(split["test_clusters"])
    assert not (val_clusters & test_clusters), "splits intersect"
    print(f"\n  split loaded: validation {len(val_clusters):,} clusters, "
          f"test {len(test_clusters):,} clusters, intersection 0")

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
               if (r.get("cluster_id") or r["query_id"]) in val_clusters]
    assert len(vstream) == split["validation"]["requests"]
    S = support_of(vstream, rounds)
    assert len(S) == split["validation"]["support"]
    n = len(vstream)
    print(f"  validation stream: {n:,} requests, support |S| = {len(S):,}")
    print(f"  timestamps preserved from the full stream (Option 1)")

    frozen = assert_frozen_freshcache()
    print(f"\n  frozen FreshCache assertions passed ({len(frozen)} settings)")

    ORIG = {"FIXED_TTL": copy.deepcopy(exp.FIXED_TTL),
            "TIER_MULT": copy.deepcopy(exp.TIER_MULT),
            "L1T": exp.L1_SIM_THRESHOLD}

    def run(method, theta, kappa, patch):
        if patch:
            patch()
        try:
            m, per = me.replay(vstream, rounds, method, rich=rich)
        finally:
            exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG["FIXED_TTL"])
            exp.TIER_MULT.clear(); exp.TIER_MULT.update(ORIG["TIER_MULT"])
            exp.L1_SIM_THRESHOLD = ORIG["L1T"]
        r = score(m, per, S, n)
        r.update({"method": method, "theta": theta, "kappa": kappa})
        return r

    t0 = time.time()
    fc = run("FreshCache", None, None, None)
    print(f"\n  FROZEN FRESHCACHE on validation ({time.time()-t0:.0f}s)")
    print(f"    saved {fc['search_saved_pct']:.4f}%   drift {fc['drift_pct']:.4f}%"
          f"   coverage {fc['coverage_pct']:.4f}%")
    print(f"    determinable {fc['determinable']:,}  changed {fc['changed']:,}"
          f"  searches {fc['searches']:,}  fetches {fc['fetches']:,}"
          f"  generations {fc['generations']:,}")
    print(f"    L1 {fc['l1_hits']:,}  L2 {fc['l2_hits']:,}  L3 {fc['l3_hits']:,}")

    rows = []
    print(f"\n  {'method':<16}{'theta':>7}{'kappa':>8}{'saved%':>10}{'drift%':>9}"
          f"{'cov%':>9}{'det':>7}{'fetch':>8}{'gen':>8}")

    def emit(r):
        th = f"{r['theta']:.2f}" if r["theta"] is not None else "—"
        ka = f"{r['kappa']:.4f}" if r["kappa"] is not None else "—"
        d = r["drift_pct"]
        print(f"  {r['method']:<16}{th:>7}{ka:>8}{r['search_saved_pct']:>9.4f}%"
              f"{(0.0 if d is None else d):>8.4f}%{r['coverage_pct']:>8.4f}%"
              f"{r['determinable']:>7,}{r['fetches']:>8,}{r['generations']:>8,}",
              flush=True)

    for th in THETA_STTL:
        for ka in KAPPA_STTL:
            def p(th=th, ka=ka):
                exp.L1_SIM_THRESHOLD = th
                exp.FIXED_TTL.update({c: v * ka
                                      for c, v in ORIG["FIXED_TTL"].items()})
            r = run("SemanticTTL", th, ka, p)
            rows.append(r); emit(r)
    print(flush=True)
    for th in THETA_SCALM:
        def p(th=th):
            exp.L1_SIM_THRESHOLD = th
        r = run("SCALM", th, None, p)
        rows.append(r); emit(r)
    print(flush=True)
    for ka in KAPPA_TIERED:
        def p(ka=ka):
            exp.TIER_MULT.update({t_: v * ka
                                  for t_, v in ORIG["TIER_MULT"].items()})
        r = run("TieredFixedTTL", None, ka, p)
        rows.append(r); emit(r)

    # thresholds restored?
    restored = (exp.FIXED_TTL == ORIG["FIXED_TTL"]
                and exp.TIER_MULT == ORIG["TIER_MULT"]
                and exp.L1_SIM_THRESHOLD == ORIG["L1T"])
    assert restored, "patched globals were not restored"
    assert_frozen_freshcache()

    with open(HERE / "validation_frontier.csv", "w", newline="",
              encoding="utf-8") as fh:
        cols = ["method", "theta", "kappa", "search_saved_pct", "drift_pct",
                "coverage_pct", "determinable", "changed", "searches",
                "fetches", "generations", "l1_hits", "l2_hits", "l3_hits",
                "n_requests", "realtime_requests"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerow({**{k: fc.get(k) for k in cols}, "method": "FreshCache_FROZEN"})
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items() if k in cols})

    sel, fronts = {}, {}
    for meth in ("SemanticTTL", "SCALM", "TieredFixedTTL"):
        sub = [r for r in rows if r["method"] == meth]
        fronts[meth] = pareto(sub)
        fresh, save = select(sub, fc["drift_pct"], fc["search_saved_pct"])
        sel[meth] = {
            "freshness_matched": fresh or "NO FRESHNESS-MATCHED POINT",
            "savings_matched": save or "NO SAVINGS-MATCHED POINT",
            "n_cells": len(sub), "n_pareto": len(fronts[meth])}

    post = {f: sha(ROOT / f) for f in WATCH}
    unchanged = all(pre[f] == post[f] for f in WATCH)
    print(f"\n  implementation files unchanged: {unchanged}")

    json.dump({
        "stage": 1, "test_split_evaluated": False,
        "frozen_freshcache_validation": fc,
        "frozen_settings_asserted": frozen,
        "selection_rules": {
            "freshness_matched": ("among cells with drift <= frozen FreshCache "
                                  "validation drift, highest search savings"),
            "savings_matched": ("among cells with savings >= frozen FreshCache "
                                "validation savings, lowest drift"),
            "tie_breaks": ["higher observable coverage", "fewer fetches",
                           "stricter semantic threshold (higher theta); where "
                           "theta is frozen, smaller kappa"],
            "pre_registered": True},
        "selection": sel,
        "pareto_frontiers": fronts,
        "split": {"validation_clusters": len(val_clusters),
                  "test_clusters": len(test_clusters),
                  "intersection": 0,
                  "validation_requests": n, "validation_support": len(S)},
        "hashes_pre": pre, "hashes_post": post,
        "implementation_unchanged": unchanged,
        "web_calls": 0, "gpu_used": False, "llm_calls": 0,
    }, open(HERE / "validation_selection.json", "w"), indent=2, default=str)

    write_md(fc, rows, fronts, sel, split, pre, post, unchanged, n, len(S))
    print(f"  wrote validation_frontier.csv, validation_selection.json, "
          f"validation_frontier.md")


def write_md(fc, rows, fronts, sel, split, pre, post, unchanged, n, ns):
    def tbl(rs, show_theta=True, show_kappa=True):
        head = "| " + (" theta |" if show_theta else "") + \
               (" kappa |" if show_kappa else "") + \
               " Saved % | Drift % | Coverage % | Determinable | Changed | Searches | Fetches | Generations | L1 | L2 | L3 |"
        sep = "|" + "---|" * (head.count("|") - 1)
        lines = [head, sep]
        for r in rs:
            cells = []
            if show_theta:
                cells.append(f"{r['theta']:.2f}" if r["theta"] is not None else "—")
            if show_kappa:
                cells.append(f"{r['kappa']:.4f}" if r["kappa"] is not None else "—")
            d = r["drift_pct"]
            cells += [f"{r['search_saved_pct']:.4f}",
                      (f"{d:.4f}" if d is not None else "—"),
                      f"{r['coverage_pct']:.4f}", f"{r['determinable']:,}",
                      f"{r['changed']:,}", f"{r['searches']:,}",
                      f"{r['fetches']:,}", f"{r['generations']:,}",
                      f"{r['l1_hits']:,}", f"{r['l2_hits']:,}", f"{r['l3_hits']:,}"]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def cand(x):
        if isinstance(x, str):
            return f"**{x}**"
        th = f"theta = {x['theta']:.2f}" if x["theta"] is not None else "theta frozen"
        ka = f", kappa = {x['kappa']:.4f}" if x["kappa"] is not None else ""
        return (f"{th}{ka} — saved **{x['search_saved_pct']:.4f}%**, "
                f"drift **{x['drift_pct']:.4f}%**, coverage "
                f"{x['coverage_pct']:.4f}%, determinable {x['determinable']:,}, "
                f"fetches {x['fetches']:,}, generations {x['generations']:,}")

    v, t = split["validation"], split["test"]
    hash_rows = "\n".join(
        f"| `{f}` | `{pre[f][:16]}` | `{post[f][:16]}` | {'unchanged' if pre[f]==post[f] else 'CHANGED'} |"
        for f in pre)

    md = f"""# Stage 1 — validation-split baseline tuning

**Stage 1 only. The held-out test split was NOT evaluated.** No test replay was
executed, no test metric was computed, and no test outcome was inspected.

CPU only, no GPU. **No web or API calls. No LLM answer generation. No LLM
judging.** No implementation or manuscript file was modified.

## 1. Frozen FreshCache on the validation stream

Configuration asserted at runtime before any replay: theta_L1 = 0.40, effective
L2 floor = 0.75, epsilon 0.10 / 0.20 / 0.35, tier multipliers 1.5 / 1.2 / 1.0,
half-lives 30d / 11.5d / 5.3d / 5.2d / 30s, equivalence semantic floor 0.80 with
Jaccard >= 0.30 plus answer-type and entity agreement. Nothing was tuned.

| Metric | Value |
|---|---|
| search savings | **{fc['search_saved_pct']:.4f}%** |
| common-support drift | **{fc['drift_pct']:.4f}%** |
| observable coverage | **{fc['coverage_pct']:.4f}%** |
| determinable | {fc['determinable']:,} |
| changed | {fc['changed']:,} |
| searches | {fc['searches']:,} |
| fetches | {fc['fetches']:,} |
| generations | {fc['generations']:,} |
| L1 / L2 / L3 hits | {fc['l1_hits']:,} / {fc['l2_hits']:,} / {fc['l3_hits']:,} |

These two numbers define the pre-registered selection thresholds:
drift <= {fc['drift_pct']:.4f}% and savings >= {fc['search_saved_pct']:.4f}%.

## 2. SemanticTTL — complete validation frontier

{len([r for r in rows if r['method']=='SemanticTTL'])} cells swept
(theta x kappa). Pareto-nondominated points, maximising savings and minimising
drift:

{tbl(fronts['SemanticTTL'])}

## 3. SCALM style — complete validation frontier

{len([r for r in rows if r['method']=='SCALM'])} cells swept (theta only; SCALM
has no other implemented lever and no temporal test).

{tbl(fronts['SCALM'], show_kappa=False)}

## 4. TieredFixedTTL — complete validation frontier

{len([r for r in rows if r['method']=='TieredFixedTTL'])} cells swept (kappa
only; its similarity thresholds are frozen at the deployed values).

{tbl(fronts['TieredFixedTTL'], show_theta=False)}

Every swept cell, dominated ones included, is in `validation_frontier.csv`.

## 5. Freshness-matched candidates

Rule, fixed before results were seen: among cells with
drift <= {fc['drift_pct']:.4f}%, take the highest search savings.

| Baseline | Candidate |
|---|---|
| SemanticTTL | {cand(sel['SemanticTTL']['freshness_matched'])} |
| SCALM style | {cand(sel['SCALM']['freshness_matched'])} |
| TieredFixedTTL | {cand(sel['TieredFixedTTL']['freshness_matched'])} |

## 6. Savings-matched candidates

Rule, fixed before results were seen: among cells with
savings >= {fc['search_saved_pct']:.4f}%, take the lowest drift.

| Baseline | Candidate |
|---|---|
| SemanticTTL | {cand(sel['SemanticTTL']['savings_matched'])} |
| SCALM style | {cand(sel['SCALM']['savings_matched'])} |
| TieredFixedTTL | {cand(sel['TieredFixedTTL']['savings_matched'])} |

Tie-breaks applied in order: higher observable coverage, fewer fetches, stricter
semantic threshold. Where a method's theta is frozen and constant, the
documented fallback is the stricter temporal setting, the smaller kappa.

**These are selection labels only. No configuration has been evaluated on the
held-out test split.**

## 7. Validation / test split

Cluster-disjoint, stratified by freshness class, seed 42, validation 30%.

| | clusters | requests | support |
|---|---|---|---|
| validation | {v['clusters']:,} | {v['requests']:,} | {v['support']:,} |
| test | {t['clusters']:,} | {t['requests']:,} | {t['support']:,} |

Stream construction is Option 1: the full stream was built once
with schedule `zipf_uniform`, seed 42, then filtered by cluster. Original
timestamps, cache ages and snapshot rounds are preserved. Validation began from
empty L1/L2/L3 state; no test state exists because no test replay was run.

## 8. Test clusters were NOT evaluated

`test_split_evaluated: false` in `validation_selection.json`. The
{t['clusters']:,} test clusters and their {t['requests']:,} requests were
filtered out of the replay stream and never scored. `heldout_split_evaluation.py` was not
written or run.

## 9. No web, API or LLM calls

Every replay read local artifacts only: the cached BGE-M3 similarity matrix, the
reported round table and the benchmark records. No network access, no GPU, no
generation, no judging.

## 10. Implementation files unchanged

| File | SHA256 before | SHA256 after | Status |
|---|---|---|---|
{hash_rows}

All patched globals (`FIXED_TTL`, `TIER_MULT`, `L1_SIM_THRESHOLD`) were restored
in a `finally` block after every cell, and the frozen-FreshCache assertions were
re-run after the sweep and passed.
"""
    (HERE / "validation_frontier.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
