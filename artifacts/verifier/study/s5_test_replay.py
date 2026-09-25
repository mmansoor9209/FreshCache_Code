#!/usr/bin/env python3
"""s5_test_replay.py — frozen-gamma sequential replay on the held-out TEST split,
plus the pre-registered gamma+-0.05 sensitivity arms. Operational metrics only."""
import json, os, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)
import experiment as exp, schedules as sc, engine_all as ea            # noqa
import corrected_engine as ce                                          # noqa
from e1_robustness import support_of                                   # noqa
import s2_engine as E                                                  # noqa
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
with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl") as fh:
    for line in fh:
        d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
full = sc.build_stream(recs, "zipf_uniform", 42)
split = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json"))
TEST = set(split["test_clusters"]); VAL = set(split["validation_clusters"])
ts = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in TEST]
assert not any((r.get("cluster_id") or r["query_id"]) in VAL for _, r in ts)
S = support_of(ts, rounds)
print(f"  test stream {len(ts):,}  |S| {len(S):,}  validation clusters present 0")
sup = E.Support(OUT)


def score(m, rows):
    """Drift/coverage over L3-REUSED pages only, matching mixed_engine's rule
    that a freshly fetched page yields no drift label. Applied identically to
    every arm. NOTE: absolute values differ from the published evaluate_test
    figures because that scorer uses mixed_engine's risk_cget labels; the
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
assert base["search_saved_pct"] == 60.5776, base["search_saved_pct"]
print(f"  anchor: FreshCache reproduces the published held-out 60.5776% savings: True")
json.dump({"frozen_gamma": G, "test_requests": len(ts), "support_S": len(S),
           "validation_clusters_in_test_stream": 0, "results": res},
          open(OUT / "test_operational.json", "w"), indent=2)
# per-request serving for the answer audit
import pickle
pickle.dump(allrows, open(OUT / "test_rows.pkl", "wb"))
print("  wrote test_operational.json, test_rows.pkl, test_scores_*.jsonl")
