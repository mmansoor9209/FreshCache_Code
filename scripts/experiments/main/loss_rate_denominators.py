"""Loss-rate denominators on the common 1,500-request sample restricted to the requests the compared policy itself changes (signature
differs from the baseline), for every main comparison. Reads common-sample outputs read-only; writes outputs/loss_rate_denominators.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json, pickle, sys, pathlib
CS_DIR = ((OUTPUT_ROOT / "common_sample")); O = ((OUTPUT_ROOT / "natural_workload"))
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); sys.argv = [sys.argv[0], "score"]
from common_sample_construction import goldof, REC, Lab   # noqa: E402
C = pickle.load(open(CS_DIR / "work/common.pkl", "rb")); T = pickle.load(open(CS_DIR / "work/traces.pkl", "rb")); rows, S = C["rows"], C["S"]
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query")); lab = Lab()
L = {(q, p): lab.label(REC[q]["query"], goldof(q), a["key"])[0] for (q, p), a in rows.items()}
R = {}
for a, b in (("B", "D"), ("B", "Dp"), ("B", "Bm"), ("Bm", "Dp"), ("D", "Dp")):
    ch = [q for q in S if sig(T[a][q]) != sig(T[b][q])]; bc = [q for q in ch if L[(q, a)] == "CORRECT"]
    lo = [q for q in bc if L[(q, b)] != "CORRECT"]
    R[f"{a}->{b}"] = {"sample_requests_changed_by_this_pair": len(ch), "base_correct_among_them": len(bc), "losses": len(lo), "loss_rate_pct": round(100 * len(lo) / len(bc), 1) if bc else None,
                      "denominator_all_common_sample": sum(L[(q, a)] == "CORRECT" for q in S)}
json.dump(R, open(O / "outputs/loss_rate_denominators.json", "w"), indent=1); print(json.dumps(R, indent=1))
