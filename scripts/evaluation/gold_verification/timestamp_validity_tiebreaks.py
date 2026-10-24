#!/usr/bin/env python3
"""timestamp_validity_tiebreaks.py -- list the timestamp-validity units on which the primary judge pair (llama8b, qwen7b)
disagrees, for a third judge. Reads r_validity_llama8b.jsonl, r_validity_qwen7b.jsonl and r_validity_tasks.jsonl from the
timestamp-validity working folder under outputs/gold_verification/ and writes r_tiebreak_units.jsonl there."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter
HERE = ((OUTPUT_ROOT / "gold_verification/timestamp_validity"))


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


L = {r["unit_id"]: r["status"] for r in jl(HERE / "r_validity_llama8b.jsonl")}
Q = {r["unit_id"]: r["status"] for r in jl(HERE / "r_validity_qwen7b.jsonl")}
T = jl(HERE / "r_validity_tasks.jsonl")
dis = [t for t in T if t["unit_id"] in L and t["unit_id"] in Q
       and L[t["unit_id"]] != Q[t["unit_id"]]]
with open(HERE / "r_tiebreak_units.jsonl", "w", encoding="utf-8") as f:
    for t in dis:
        f.write(json.dumps({"unit_id": t["unit_id"], "mode": t["mode"],
                            "llama8b": L[t["unit_id"]],
                            "qwen7b": Q[t["unit_id"]]}) + "\n")
print(f"  primary pair judged: {len(L)} / {len(Q)}")
print(f"  disagreements needing a third judge: {len(dis)}")
print(f"  pattern: {dict(Counter((L[t['unit_id']], Q[t['unit_id']]) for t in dis).most_common(8))}")
