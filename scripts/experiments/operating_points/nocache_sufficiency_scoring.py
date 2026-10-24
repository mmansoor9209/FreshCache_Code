"""Score NoCache (fresh-pipeline) evidence sufficiency on the 400 audit requests (two-judge agreement rule; disagreement kept
as JUDGE_DISAGREEMENT) and pair it, with an exact McNemar test, against FreshCache per-request sufficiency at theta_L2 0.75 / 0.80
and gamma 0.55 (outputs/l2_operating_points/l2_threshold_request_level.csv, read-only).
Writes outputs/operating_point_summaries/nocache_sufficiency.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import csv, json, math, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = OUTPUT_ROOT; ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
L8 = {r["query_id"]: r["label"] for r in jl(((OUTPUT_ROOT / "nocache_sufficiency/gpu/out/suffv_llama8b.jsonl")))}; Q7 = {r["query_id"]: r["label"] for r in jl(((OUTPUT_ROOT / "nocache_sufficiency/gpu/out/suffv_qwen7b.jsonl")))}
NC = {q: (L8[q] if L8[q] == Q7.get(q) else "JUDGE_DISAGREEMENT") for q in L8}
END_TO_END_ROWS = {}
for r in csv.DictReader(open((OUTPUT_ROOT / "l2_operating_points/l2_threshold_request_level.csv"))): END_TO_END_ROWS.setdefault(r["point"], {})[r["request_id"]] = r["sufficiency"]
def mcn(b, c): n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
R = {"N": len(NC), "NoCache_sufficient": sum(v == "SUFFICIENT" for v in NC.values()), "labels": {k: sum(v == k for v in NC.values()) for k in set(NC.values())}}
for pt in ("theta_0.75", "theta_0.80", "gamma_0.55"):
    F = END_TO_END_ROWS[pt]; b = sum(1 for q in NC if NC[q] == "SUFFICIENT" and F[q] != "SUFFICIENT"); c = sum(1 for q in NC if NC[q] != "SUFFICIENT" and F[q] == "SUFFICIENT")
    R[f"vs_{pt}"] = {"FreshCache_sufficient": sum(F[q] == "SUFFICIENT" for q in NC), "NoCache_only": b, "FreshCache_only": c, "p": mcn(b, c)}
json.dump(R, open(((OUTPUT_ROOT / "operating_point_summaries/nocache_sufficiency.json")), "w"), indent=1); print(json.dumps(R, indent=1))
