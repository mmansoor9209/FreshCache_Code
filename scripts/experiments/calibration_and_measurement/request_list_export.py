"""Request-ID lists used by later stages: the primary stream (order, cluster, arrival time, split), the held-out
requests, and the general-audit requests whose references the model majority judges valid.

Extracted from existing outputs without recomputation; run after scripts/experiments/ablations/mixed_age_trace_construction.py,
scripts/experiments/admission_controls/verifier_heldout_replay.py and scripts/evaluation/reference_validation/reference_validation_analysis.py.
Writes outputs/frozen_inputs/generated/request_lists/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import csv, json, os, pathlib, pickle; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
OUT = (OUTPUT_ROOT / "frozen_inputs/generated/request_lists"); OUT.mkdir(parents=True, exist_ok=True)
def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"))); TEST, VAL = set(split["test_clusters"]), set(split["validation_clusters"])
T2 = pickle.load(open((OUTPUT_ROOT / "replay/table2/reported/traces.pkl"), "rb")); SUP = set(T2["support"]); CL = dict(zip(T2["qids"], T2["clusters"]))
write_csv(OUT / "primary_replay.csv",
          [{"order": i, "query_id": q, "cluster_id": c, "arrival_t": t, "split": "test" if c in TEST else "validation" if c in VAL else "neither",
            "in_mixed_age_common_support": int(q in SUP)} for i, (q, c, t) in enumerate(zip(T2["qids"], T2["clusters"], T2["arrival_times"]))],
          ["order", "query_id", "cluster_id", "arrival_t", "split", "in_mixed_age_common_support"])
VR = pickle.load(open((OUTPUT_ROOT / "replay/verifier/out/test_rows.pkl"), "rb"))["FreshCache"]
held = [q for q in T2["qids"] if CL[q] in TEST]; assert set(held) == set(VR), "held-out ids differ"
write_csv(OUT / "heldout_replay.csv", [{"query_id": q, "cluster_id": CL[q], "split": "test"} for q in held], ["query_id", "cluster_id", "split"])
REF = list(csv.DictReader(open((OUTPUT_ROOT / "reference_validation/results/reference_validation_results.csv"))))
write_csv(OUT / "model_majority_valid.csv",
          [{"query_id": r["query_id"], "final_label": r["final_label"], "unanimous": r["unanimous"], "target_time": r["target_time"]} for r in REF if r["final_label"] == "VALID_AT_T"],
          ["query_id", "final_label", "unanimous", "target_time"])
print("wrote", len(T2["qids"]), len(held), sum(1 for r in REF if r["final_label"] == "VALID_AT_T"))
