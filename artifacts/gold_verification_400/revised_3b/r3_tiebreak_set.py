#!/usr/bin/env python3
"""revised_3b/r3_tiebreak_set.py — units where the primary pair disagrees."""
import json, pathlib
from collections import Counter
HERE = pathlib.Path(__file__).resolve().parent


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
