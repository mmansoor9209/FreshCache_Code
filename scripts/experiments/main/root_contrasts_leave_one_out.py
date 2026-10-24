"""Root-level 2x2 contrasts (evidence reuse vs fresh, write vs no write) with leave-one-root-out for the held-out pre-registered and differing-list
roots and the confirmation-stream roots. Sums with root-bootstrap 95% intervals (2,000 resamples, seed 20261017) and the LOO range of each sum."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, random, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "natural_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample"))
cor = lambda d, a: int(d[f"root_{a}"] == "CORRECT") + d[f"down_correct_{a}"]
C = {"root_evidence (reuse - fresh), write": lambda d: int(d["root_reuse_write"] == "CORRECT") - int(d["root_fresh_write"] == "CORRECT"),
     "downstream write effect | reuse": lambda d: d["down_correct_reuse_write"] - d["down_correct_reuse_nowrite"],
     "downstream write effect | fresh": lambda d: d["down_correct_fresh_write"] - d["down_correct_fresh_nowrite"],
     "downstream evidence effect | write": lambda d: d["down_correct_reuse_write"] - d["down_correct_fresh_write"],
     "total D - (fresh, write)": lambda d: cor(d, "reuse_write") - cor(d, "fresh_write"),
     "total (reuse, no write) - (fresh, no write)": lambda d: cor(d, "reuse_nowrite") - cor(d, "fresh_nowrite")}
SETS = {"heldout_pre (60)": json.load(open(CS_DIR / "outputs/root_clone_2x2_pre.json"))["per_root"], "heldout_differing (40)": json.load(open(CS_DIR / "outputs/root_clone_2x2_diff.json"))["per_root"],
        "confirmation_P (40)": json.load(open(O / "outputs/root_diagnostic_per_root.json"))["per_root"]}
R = {}
for nm, per in SETS.items():
    rng = random.Random(20261017); R[nm] = {"roots": len(per), "root_correct": {a: sum(d[f"root_{a}"] == "CORRECT" for d in per) for a in ("fresh_write", "fresh_nowrite", "reuse_write", "reuse_nowrite")},
                                             "downstream_requests_differing": sum(d["downstream_requests_differing"] for d in per), "contrasts": {}}
    for c, f in C.items():
        v = [f(d) for d in per]; bs = sorted(sum(rng.choice(v) for _ in v) for _ in range(2000)); loo = [sum(v) - x for x in v]
        R[nm]["contrasts"][c] = {"sum": sum(v), "ci": [bs[49], bs[1949]], "loo_range": [min(loo), max(loo)], "roots_nonzero": sum(x != 0 for x in v)}
json.dump(R, open(O / "outputs/root_contrasts_leave_one_out.json", "w"), indent=1); print(json.dumps(R, indent=1))
