"""Holm step-down adjustment over all 24 paired tests of Table 4 (12 WAI + 12 sufficiency exact McNemar p-values).
The raw p-values of the held-out decision tables and the gate, verifier and NoCache comparisons are collected by holm_family_construction.py in outputs/operating_point_summaries/holm_family_24.json;
recomputes p_holm and checks the stored values."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pathlib
OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); d = json.load(open(OUTPUT_ROOT / "operating_point_summaries/holm_family_24.json")); T = d["tests"]; m = len(T); assert m == 24
order = sorted(range(m), key=lambda i: T[i]["p"]); adj = [0.0] * m; run = 0.0
for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * T[i]["p"])); adj[i] = run
for t, a in zip(T, adj): assert abs(round(a, 6) - t["p_holm"]) < 1e-5 or (t["p_holm"] == 1 and a == 1.0), (t["test"], a, t["p_holm"])
print("Holm-24 verified;", sum(a < 0.05 for a in adj), "significant at 0.05")
