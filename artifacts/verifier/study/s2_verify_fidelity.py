#!/usr/bin/env python3
"""s2_verify_fidelity.py — assert that gamma=None reproduces prep2.replay."""
import json, os, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)
import experiment as exp, schedules as sc, prep2 as P2, s2_engine as E   # noqa

L = lambda p: exp.load_jsonl(pathlib.Path(p))
recs = exp.build_query_records(L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
                               L(exp.PARAPHRASE_FILE))
exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}
exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
import engine_all as ea
ea.set_cluster_base(recs)
rounds = {}
with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl") as fh:
    for line in fh:
        d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
full = sc.build_stream(recs, "zipf_uniform", 42)
split = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json"))
for nm, cl in (("validation", set(split["validation_clusters"])),
               ("test", set(split["test_clusters"]))):
    st = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in cl]
    ref_m, ref_rows = P2.replay(st, rounds, "FreshCache")
    m, rows, _ = E.replay(st, gamma=None, support=None)
    same_agg = all(m[k] == ref_m[k] for k in
                   ("search_calls", "l1_hits", "l2_hits", "l3_hits", "fetches")) \
        and m["search_saved_pct"] == ref_m["search_saved_pct"]
    def evof(d):
        # compare only keys prep2 itself records; L1 rows carry stored_answer_ev
        k = "stored_answer_ev" if d.get("tier") == "L1" else "ev"
        return [tuple(x) for x in (d.get(k) or [])]
    diff = [q for q in ref_rows
            if rows[q]["tier"] != ref_rows[q]["tier"]
            or evof(rows[q]) != evof(ref_rows[q])
            or [x for x in (rows[q].get("served") or [])]
            != [x for x in (ref_rows[q].get("served") or [])]]
    print(f"  {nm:<11} n={len(st):,}  aggregates identical: {same_agg}  "
          f"per-request tier+evidence mismatches: {len(diff)}")
    print(f"    ref  saved {ref_m['search_saved_pct']}  L1 {ref_m['l1_hits']} "
          f"L2 {ref_m['l2_hits']} L3 {ref_m['l3_hits']} fetch {ref_m['fetches']}")
    print(f"    mine saved {m['search_saved_pct']}  L1 {m['l1_hits']} "
          f"L2 {m['l2_hits']} L3 {m['l3_hits']} fetch {m['fetches']}")
    assert same_agg and not diff, f"FIDELITY FAILED on {nm}"
print("  FIDELITY OK — gamma=None reproduces prep2.replay exactly on both splits")
