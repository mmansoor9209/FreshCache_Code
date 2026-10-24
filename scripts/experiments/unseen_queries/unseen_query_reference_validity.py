"""Three-judge reference validity of the 150 live paired cases at the search date (refval/), and the primary A vs B comparison
restricted to majority-valid / unanimous-valid references. Labels and rules are those of unseen_query_scoring.py (copied functions, read-only inputs)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, math, os, pathlib, sys, types; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); src = open((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py")).read(); cut = src.index("PAIR = [")
g = {"__name__": "t2s", "__file__": __file__}; sys.argv = ["unseen_query_scoring.py", "score"]; exec(compile(src[:cut], "<unseen_query_scoring.py functions>", "exec"), g)
C, label, suff, mcn, pdiff = g["C"], g["label"], g["suff"], g["mcn"], g["pdiff"]
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
R = {j: {r["query_id"]: r["label"] for r in jl((OUTPUT_ROOT / f"unseen_queries/refval/outputs/raw/reference_{j}.jsonl"))} for j in ("llama8b", "qwen7b", "mistral7b")}
PAIR = [c for c in C if c["both_evaluable"] and c["reference_available"]]
def maj(q):
    v = collections.Counter(R[j].get(q) for j in R).most_common(1)[0]; return v[0] if v[1] >= 2 else "NO_MAJORITY"
MJ = {c["case_id"]: maj(c["case_id"]) for c in PAIR}; UN = {q for q in MJ if len({R[j].get(q) for j in R}) == 1 and MJ[q] == "VALID_AT_T"}
def fk(rows, cats):
    n, k = len(rows), 3; P = [sum(c * (c - 1) for c in r) / (k * (k - 1)) for r in rows]; pj = [sum(r[i] for r in rows) / (n * k) for i in range(len(cats))]
    Pb, Pe = sum(P) / n, sum(x * x for x in pj); return round((Pb - Pe) / (1 - Pe), 4)
cats = ["VALID_AT_T", "INVALID_AT_T", "TEMPORALLY_AMBIGUOUS", "UNVERIFIABLE", "UNPARSED"]
kap = fk([[sum(1 for j in R if R[j].get(c["case_id"]) == x) for x in cats] for c in PAIR], cats)
def comp(ids, a="A_matched", b="B_cached_live"):
    n = len(ids); A = {c["case_id"]: (label(c, a), suff(c, a)) for c in PAIR}; B = {c["case_id"]: (label(c, b), suff(c, b)) for c in PAIR}
    x = sum(A[i][0] == "CORRECT" and B[i][0] != "CORRECT" for i in ids); y = sum(A[i][0] != "CORRECT" and B[i][0] == "CORRECT" for i in ids)
    s1 = sum(A[i][1] == "SUFFICIENT" and B[i][1] != "SUFFICIENT" for i in ids); s2 = sum(A[i][1] != "SUFFICIENT" and B[i][1] == "SUFFICIENT" for i in ids)
    return {"N": n, "A_correct": sum(A[i][0] == "CORRECT" for i in ids), "B_correct": sum(B[i][0] == "CORRECT" for i in ids), "correct_lost_gained": [x, y], "correct_p": mcn(x, y), "correct_diff_pp_CI": pdiff(x, y, n) if n else None,
            "A_suff": sum(A[i][1] == "SUFFICIENT" for i in ids), "B_suff": sum(B[i][1] == "SUFFICIENT" for i in ids), "suff_lost_gained": [s1, s2], "suff_p": mcn(s1, s2), "suff_diff_pp_CI": pdiff(s1, s2, n) if n else None}
mv = [q for q, v in MJ.items() if v == "VALID_AT_T"]
OUT = {"reference_majority": dict(collections.Counter(MJ.values())), "unanimous_valid": len(UN), "reference_fleiss_kappa": kap,
       "all_cases": comp(list(MJ)), "majority_valid": comp(mv), "unanimous_valid_subset": comp(sorted(UN))}
json.dump(OUT, open((OUTPUT_ROOT / "unseen_queries/refval/results/refval_results.json"), "w"), indent=1); print(json.dumps(OUT, indent=1))
