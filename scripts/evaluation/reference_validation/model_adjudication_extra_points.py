"""Adds the selected standalone operating points to the three-model adjudication (Table 11) after the base policy outputs were judged.
Usage: model_adjudication_extra_points.py {points,addendum}
  points    SemanticTTL (.60,1/4), SemanticTTL (.80,2) and CategoryAware-style (+.075) audit answers from
            outputs/standalone_operating_points/work/stage3.pkl (operating_point_heldout_evaluation.py) -> inputs/extra_points.jsonl
  addendum  savings-matched SemanticTTL (.70,2) audit answers from work/stage3_addendum.pkl (savings_matched_operating_point.py)
            -> inputs/extra_points_addendum.jsonl
Each mode appends its rows to outputs/model_adjudication/inputs/policy_outputs.jsonl (once; a copy of the base file is kept as
policy_outputs_before_extra_points.jsonl); the answer judges then judge only the new pairs and model_adjudication_analysis.py is re-run."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) or len(_sys.argv) < 2 else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pickle, shutil, sys
OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/operating_points"))
import operating_point_helpers as X
MODE = sys.argv[1]; assert MODE in ("points", "addendum"), MODE
T = OUTPUT_ROOT / "model_adjudication/inputs"; POL = T / "policy_outputs.jsonl"
if not (T / "policy_outputs_before_extra_points.jsonl").exists(): shutil.copyfile(POL, T / "policy_outputs_before_extra_points.jsonl")
hl = X.Held(); out = []
if MODE == "points":
    ST = pickle.load(open(X.O / "work/stage3.pkl", "rb"))
    R = {json.loads(l)["query_id"] for l in open(T / "requests.jsonl")}
    print(list(ST["audit"].keys()))
    miss = 0
    for pid in ("sttl_t0.60_k0.25", "sttl_t0.80_k2", "ca_ca_d+0.075_m1.00"):
        A = ST["audit"][pid]; print(pid, len(A), len(set(A) & R))
        for q, a in A.items():
            if q not in R: continue
            ans = hl.answer(a["key"])
            if ans is None: miss += 1; continue
            out.append({"query_id": q, "arm": "op_" + pid, "tier": a.get("tier", ""), "answer_key": a["key"], "answer": ans})
    print(len(out), miss); dst = T / "extra_points.jsonl"
else:
    ST = pickle.load(open(X.O / "work/stage3_addendum.pkl", "rb"))
    for q, a in ST["audit"]["sttl_t0.70_k2"].items():
        ans = hl.answer(a["key"]); assert ans is not None
        out.append({"query_id": q, "arm": "op_sttl_t0.70_k2", "tier": a.get("tier", ""), "answer_key": a["key"], "answer": ans})
    print(len(out)); dst = T / "extra_points_addendum.jsonl"
open(dst, "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
have = {json.loads(l)["arm"] for l in open(POL, encoding="utf-8") if l.strip()}
if not {r["arm"] for r in out} & have:
    with open(POL, "a", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print("policy_outputs rows:", sum(1 for l in open(POL, encoding="utf-8") if l.strip()))
