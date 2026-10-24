#!/usr/bin/env python3
"""verifier_heldout_replay_selection_engine.py — frozen-gamma sequential replay on the held-out TEST split,
plus the pre-registered gamma+-0.05 sensitivity arms. Operational metrics only."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import numpy as np
HERE = ((OUTPUT_ROOT / "l2_verifier"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine"))
os.chdir(ROOT)
import tiered_cache_model as exp, request_schedules as sc, method_replay as ea            # noqa
import execution_semantics as ce                                          # noqa
from schedule_robustness import support_of                                   # noqa
import verifier_replay as E                                                  # noqa
OUT = HERE / "out"
G = json.load(open(OUT / "gamma_selection.json"))["selected_gamma"]
print(f"  frozen gamma = {G} (selected on validation only)")

L = lambda p: exp.load_jsonl(pathlib.Path(p))
recs = exp.build_query_records(L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
                               L(exp.PARAPHRASE_FILE))
exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}
exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
ea.set_cluster_base(recs)
rounds = {}
with open((DATA_ROOT / "derived/round_table.jsonl")) as fh:
    for line in fh:
        d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
full = sc.build_stream(recs, "zipf_uniform", 42)
split = json.load(open((OUTPUT_ROOT / "validation_split/split.json")))
TEST = set(split["test_clusters"]); VAL = set(split["validation_clusters"])
ts = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in TEST]
assert not any((r.get("cluster_id") or r["query_id"]) in VAL for _, r in ts)
S = support_of(ts, rounds)
print(f"  test stream {len(ts):,}  |S| {len(S):,}  validation clusters present 0")
sup = E.Support(OUT)


def score(m, rows):
    """Drift/coverage over L3-REUSED pages only, matching mixed_age_replay's rule
    that a freshly fetched page yields no drift label. Applied identically to
    every arm. NOTE: absolute values differ from the recorded heldout_split_evaluation
    figures because that scorer uses mixed_age_replay's risk_cget labels; the
    arm-to-arm comparison here is internally consistent."""
    ins = []
    for qid, d in rows.items():
        if qid not in S:
            continue
        ev = ((d.get("stored_answer_ev") or []) if d["tier"] == "L1"
              else [(x, tc) for x, tc in (d.get("ev") or []) if tc != d["t"]])
        labs = [E.ma_outcome(x, tc, d["t"], rounds) for x, tc in ev]
        o = (ce.CHANGED if ce.CHANGED in labs else
             (ce.UNCHANGED if ce.UNCHANGED in labs else None)) if labs else None
        if o is not None:
            ins.append(o)
    ch = sum(1 for o in ins if o == ce.CHANGED); un = len(ins) - ch
    return {**m, "changed": ch, "determinable": ch + un,
            "drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None,
            "coverage_pct": round(100 * (ch + un) / len(S), 4), "support_S": len(S)}


res, allrows = {}, {}
for tag, g in (("FreshCache", None), (f"L2Verify_g{G}", G),
               (f"L2Verify_g{round(G-0.05,2)}", round(G - 0.05, 2)),
               (f"L2Verify_g{round(G+0.05,2)}", round(G + 0.05, 2))):
    m, rows, sc_ = E.replay(ts, gamma=g, support=(sup if g is not None else None))
    res[tag] = score(m, rows)
    allrows[tag] = rows
    if g is not None:
        with open(OUT / f"test_scores_{tag}.jsonl", "w") as f:
            for r in sc_:
                f.write(json.dumps(r) + "\n")
    r = res[tag]
    print(f"    {tag:<20} saved {r['search_saved_pct']:8.4f}%  L1 {r['l1_hits']:>5} "
          f"L2 {r['l2_hits']:>6} L3 {r['l3_hits']:>6}  fetch/1k {r['fetches_per_1k']:7.2f} "
          f"gen/1k {r['generations_per_1k']:7.2f}  drift {r['drift_pct']:7.4f}% "
          f"({r['changed']}/{r['determinable']})  cov {r['coverage_pct']:7.4f}%  "
          f"acc/rej {r['l2_accepted']}/{r['l2_rejected']} unscored {r['l2_unscored']}")
base = res["FreshCache"]
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); assert abs(base["search_saved_pct"] - REF.value("selection_engine/heldout/FreshCache/search_saved_pct")) <= 1e-4, base["search_saved_pct"]
print(f"  anchor: FreshCache reproduces the recorded held-out savings ({base['search_saved_pct']:.4f}%): True")
json.dump({"frozen_gamma": G, "test_requests": len(ts), "support_S": len(S),
           "validation_clusters_in_test_stream": 0, "results": res},
          open(OUT / "test_operational.json", "w"), indent=2)
# per-request serving for the answer audit
import pickle
pickle.dump(allrows, open(OUT / "test_rows.pkl", "wb"))
print("  wrote test_operational.json, test_rows.pkl, test_scores_*.jsonl")
