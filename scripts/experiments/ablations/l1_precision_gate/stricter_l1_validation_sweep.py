#!/usr/bin/env python3
"""
Optional stricter-L1 ablation: validation sweep and pre-registered selection.

This does NOT replace primary FreshCache. It asks whether a tighter L1
admission rule removes semantic mismatches without removing most of the L1
benefit.

Tunable knobs (the three that FreshCache's L1 equivalence gate exposes):
  * the answer-tier similarity floor          tiered_cache_model._EQ_SIM_FLOOR
                                              in [0.80, 0.85, 0.90, 0.95]
  * the content-word Jaccard floor            tiered_cache_model._EQ_JACCARD_MIN
                                              in [0.30, 0.45, 0.60, 0.75]
  * whether answer-type agreement is REQUIRED (both types must be detectable
    and equal), rather than only enforced when both happen to be detectable

Protocol: sweep on VALIDATION clusters only (stream zipf_uniform, seed 42),
pre-register the selection rule and hash it before any cell is run, freeze ONE
operating point (the strictest cell retaining at least 50% of the FreshCache
validation L1 hits), then evaluate that single point on the held-out TEST
clusters with scripts/experiments/ablations/l1_precision_gate/strict_l1_gate_tasks.py.

Everything else -- the entity gate, the exponential temporal test, the
freshness classes, L2 and L3 -- is untouched.

Outputs in outputs/l1only_exactttl/l1_hit_analysis/: stricter_prereg.json,
stricter_l1_grid.csv, stricter_l1_results.json.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, hashlib, itertools, json, os, pathlib, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("freshcache-freshcache-finalvalidation")
except Exception:
    pass

HERE = ((OUTPUT_ROOT / "l1only_exactttl/l1_hit_analysis"))
BASE = ((OUTPUT_ROOT / "l1only_exactttl"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "scripts/baselines/exact_ttl_and_l1_only"))
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import tiered_cache_model as exp                 # noqa: E402
import method_replay as ea                  # noqa: E402
import mixed_age_replay as me                # noqa: E402
import request_schedules as sc                   # noqa: E402
from schedule_robustness import support_of     # noqa: E402

CHANGED, UNCHANGED = me.CHANGED, me.UNCHANGED
SCHEDULE, SEED = "zipf_uniform", 42
SIM_FLOORS = [0.80, 0.85, 0.90, 0.95]
JAC_FLOORS = [0.30, 0.45, 0.60, 0.75]
REQ_TYPE = [False, True]
WATCH = ["freshcache/method/tiered_cache_model.py", "freshcache/selection_replay_engine/mixed_age_replay.py",
         "freshcache/selection_replay_engine/method_replay.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def score(m, per, S):
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == CHANGED)
    un = sum(1 for o in ins if o == UNCHANGED)
    det = ch + un
    n = m["n_requests"]
    return {"search_saved_pct": m["search_saved_pct"],
            "drift_pct": (100 * ch / det) if det else None,
            "coverage_pct": (100 * det / len(S)) if S else None,
            "determinable": det, "l1_hits": m["l1_hits"],
            "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
            "searches": m["search_calls"], "fetches": m["fetches"],
            "generations": m["generations"],
            "generation_calls_saved": n - m["generations"],
            "total_ops": m["search_calls"] + m["fetches"] + m["generations"],
            "n_requests": n}


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("stricter-L1 ablation (NOT a replacement for FreshCache)")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    prereg = {
        "written_before_any_cell_was_evaluated": True,
        "knobs": {"_EQ_SIM_FLOOR": SIM_FLOORS, "_EQ_JACCARD_MIN": JAC_FLOORS,
                  "require_answer_type_agreement": REQ_TYPE},
        "n_cells": len(SIM_FLOORS) * len(JAC_FLOORS) * len(REQ_TYPE),
        "data": "VALIDATION clusters only; test is touched once, for one "
                "frozen point",
        "selection_rule": "among cells that RETAIN at least 50% of the "
                          "recorded FreshCache validation L1 hits, choose the "
                          "STRICTEST (largest _EQ_SIM_FLOOR, then largest "
                          "_EQ_JACCARD_MIN, then answer-type REQUIRED). "
                          "Rationale: the ablation exists to reduce mismatch, "
                          "so strictness is the objective and L1 retention is "
                          "the constraint.",
        "retention_floor": 0.50,
        "status_of_primary_freshcache": "UNCHANGED",
    }
    pj = HERE / "stricter_prereg.json"
    pj.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    say(f"  PRE-REGISTRATION {pj.name}  sha256 {sha(pj)}")
    say(f"    {prereg['selection_rule']}")

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
    split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"), encoding="utf-8"))
    val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])
    val = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in val_c]
    test = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in test_c]
    Sv, St = support_of(val, rounds), support_of(test, rounds)

    ORIG_S, ORIG_J = exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN
    ORIG_EQ = exp.semantic_equivalent

    def make_strict(require_type):
        base = ORIG_EQ
        if not require_type:
            return base
        def strict(q1, q2, sim):
            if not base(q1, q2, sim):
                return False
            a1, a2 = exp._eq_answer_type(q1), exp._eq_answer_type(q2)
            return a1 is not None and a2 is not None and a1 == a2
        return strict

    def run(stream, S, s_floor, j_floor, req_type):
        exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN = s_floor, j_floor
        exp.semantic_equivalent = make_strict(req_type)
        me.exp.semantic_equivalent = exp.semantic_equivalent
        try:
            m, per = me.replay(stream, rounds, "FreshCache", rich=rich)
            return score(m, per, S)
        finally:
            exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN = ORIG_S, ORIG_J
            exp.semantic_equivalent = ORIG_EQ
            me.exp.semantic_equivalent = ORIG_EQ

    baseline = run(val, Sv, ORIG_S, ORIG_J, False)
    say(f"\n  recorded gate on validation: L1 {baseline['l1_hits']:,}  "
        f"saved {baseline['search_saved_pct']:.4f}%  "
        f"drift {baseline['drift_pct']:.4f}%")
    FLOOR = prereg["retention_floor"] * baseline["l1_hits"]
    say(f"  L1 retention constraint: >= {FLOOR:.0f} validation L1 hits")

    rows = []
    say(f"\n  {'simF':>6}{'jacF':>7}{'reqType':>9}{'L1':>7}{'saved%':>10}"
        f"{'drift%':>9}{'genSaved':>10}{'totOps':>9}")
    for sf, jf, rt in itertools.product(SIM_FLOORS, JAC_FLOORS, REQ_TYPE):
        r = run(val, Sv, sf, jf, rt)
        r.update({"sim_floor": sf, "jac_floor": jf, "require_answer_type": rt})
        rows.append(r)
        say(f"  {sf:>6.2f}{jf:>7.2f}{str(rt):>9}{r['l1_hits']:>7,}"
            f"{r['search_saved_pct']:>9.4f}%{(r['drift_pct'] or 0):>8.4f}%"
            f"{r['generation_calls_saved']:>10,}{r['total_ops']:>9,}")

    feas = [r for r in rows if r["l1_hits"] >= FLOOR]
    say(f"\n  cells retaining >= 50% of L1 hits: {len(feas)}/{len(rows)}")
    if not feas:
        say("  NO stricter cell satisfies the pre-registered retention floor. "
            "Reported as such; the floor is not relaxed after seeing results.")
        chosen = None
    else:
        chosen = sorted(feas, key=lambda r: (-r["sim_floor"], -r["jac_floor"],
                                             not r["require_answer_type"]))[0]
        say(f"  FROZEN stricter-L1 point (validation only): "
            f"sim_floor {chosen['sim_floor']}  jac_floor {chosen['jac_floor']}  "
            f"require_answer_type {chosen['require_answer_type']}")
        say(f"    validation L1 {chosen['l1_hits']:,} "
            f"({100*chosen['l1_hits']/max(1,baseline['l1_hits']):.1f}% of "
            f"recorded)  saved {chosen['search_saved_pct']:.4f}%  "
            f"drift {chosen['drift_pct']:.4f}%")

    test_out = None
    if chosen:
        say(f"\n  held-out TEST, single frozen point")
        pub_t = run(test, St, ORIG_S, ORIG_J, False)
        test_out = run(test, St, chosen["sim_floor"], chosen["jac_floor"],
                       chosen["require_answer_type"])
        say(f"    {'':<14}{'L1':>7}{'saved%':>10}{'drift%':>9}{'cov%':>9}"
            f"{'genSaved':>10}{'totOps':>10}")
        for nm, r in (("recorded", pub_t), ("stricter-L1", test_out)):
            say(f"    {nm:<14}{r['l1_hits']:>7,}{r['search_saved_pct']:>9.4f}%"
                f"{(r['drift_pct'] or 0):>8.4f}%{(r['coverage_pct'] or 0):>8.4f}%"
                f"{r['generation_calls_saved']:>10,}{r['total_ops']:>10,}")
        say(f"    delta L1 {test_out['l1_hits']-pub_t['l1_hits']:+,}   "
            f"saved {test_out['search_saved_pct']-pub_t['search_saved_pct']:+.4f} pp"
            f"   drift {(test_out['drift_pct'] or 0)-(pub_t['drift_pct'] or 0):+.4f} pp")
        test_out["recorded_test_reference"] = pub_t
        say(f"    semantic mismatch, WAI and C-WAI under the stricter gate are "
            f"reported in l1_analysis.json only for the hits it shares with the "
            f"recorded gate; hits unique to the stricter gate were not "
            f"separately judged, and that limit is stated rather than assumed "
            f"away.")

    assert exp._EQ_SIM_FLOOR == ORIG_S and exp._EQ_JACCARD_MIN == ORIG_J
    assert exp.semantic_equivalent is ORIG_EQ
    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post
    say("\n  implementation files unchanged: True")
    say("  primary FreshCache result: UNCHANGED (this is an ablation)")

    json.dump({"prereg_sha256": sha(pj), "prereg": prereg,
               "validation_baseline": baseline, "grid": rows,
               "frozen": chosen, "heldout_test": test_out,
               "primary_freshcache_unchanged": True},
              open(HERE / "stricter_l1_results.json", "w"), indent=2)
    with open(HERE / "stricter_l1_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (HERE / "stricter_l1_validation_sweep.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote stricter_prereg.json, stricter_l1_grid.csv, "
        "stricter_l1_results.json")


if __name__ == "__main__":
    main()
