"""AQ-07 / NEW-357-03: cluster-level paired WAI test for the QUALITY-SELECTED
SemanticTTL arm (theta=0.60, kappa=1/2).

The existing cluster-aware result (54 vs 0) belongs to the DRIFT-selected arm
(kappa=1/16, 64 WAI). The abstract attached it to the quality-selected
comparison (25 vs 7), which is a different arm. This computes the
quality-selected arm's own cluster-level test, reusing the published scoring
path (h4_analyze) verbatim so the request-level numbers must reproduce first.

Read-only; writes only under Reframe_ResearchPaper/artifacts/.
"""
import json, csv, pathlib, sys, collections
from fractions import Fraction

SRC = pathlib.Path("<PROJECT_ROOT>")
HS  = SRC/"validation/quality_aware_baseline/heldout_sensitivity"
OUT = pathlib.Path(__file__).resolve().parent.parent/"artifacts"
sys.path.insert(0, str(HS))
import h4_analyze as H     # constants + helpers, verbatim

jl, h, ABST, A2, GV, RV = H.jl, H.h, H.ABST, H.A2, H.GV, H.RV

sample = json.load(open(A2/"sample.json", encoding="utf-8"))["requests"]
new = {d["query_id"]: d for d in jl(HS/"h_sample.jsonl")}
ans = {tuple(r["key"]): r["answer"] for r in jl(A2/"answers.jsonl")}
for r in jl(HS/"h_answers.jsonl"): ans[tuple(r["key"])] = r["answer"]
j1 = {tuple(r["sig"]): r["label"] for r in jl(A2/"judge_llama8b.jsonl")}
j2 = {tuple(r["sig"]): r["label"] for r in jl(A2/"judge_qwen7b.jsonl")}
for r in jl(HS/"h_judge_llama8b.jsonl"): j1[tuple(r["sig"])] = r["label"]
for r in jl(HS/"h_judge_qwen7b.jsonl"): j2[tuple(r["sig"])] = r["label"]

def outc(d, key):
    a = ans.get(tuple(key))
    if a is None: return None
    if any(p in a.strip().lower() for p in ABST): return "ABSTAIN"
    s = (h(d["query"]), h(d["gold"]), h(a))
    l1, l2 = j1.get(s), j2.get(s)
    if l1 is None or l2 is None: return None
    return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

rows = []
for d in sample:
    r = {"qid": d["query_id"],
         "cluster": new[d["query_id"]]["cluster_id"],
         "fresh": outc(d, d["fresh_key"]), "fc": outc(d, d["fc_key"]),
         "sttl_k1_16": outc(d, d["sttl_key"]),
         "sttl_t060_k1_2": outc(d, new[d["query_id"]]["new_sttl_key"])}
    if any(r[k] is None for k in ("fresh","fc","sttl_k1_16","sttl_t060_k1_2")): continue
    rows.append(r)
print(f"scored {len(rows)} of 400")

def wai(r, arm): return r["fresh"] == "CORRECT" and r[arm] == "WRONG"

print("\n[GATE] request-level WAI must reproduce the published counts")
g = {a: sum(wai(r, a) for r in rows) for a in ("fc","sttl_k1_16","sttl_t060_k1_2")}
for a, want in [("fc",7), ("sttl_k1_16",64), ("sttl_t060_k1_2",25)]:
    print(f"  {'PASS' if g[a]==want else 'FAIL'}  {a}: {g[a]} (published {want})")
if not (g["fc"]==7 and g["sttl_k1_16"]==64 and g["sttl_t060_k1_2"]==25):
    print("  gate failed; not reporting a cluster result"); sys.exit(1)

def exact_mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    tot = Fraction(0)
    from math import comb
    lo = min(b, c)
    for k in range(0, lo+1):
        tot += Fraction(comb(n, k), 2**n)
    p = 2*tot
    return float(min(Fraction(1), p))

def cluster_test(arm):
    cl = collections.defaultdict(lambda: {"a":0,"b":0})
    for r in rows:
        cl[r["cluster"]]["a"] += wai(r,"fc")
        cl[r["cluster"]]["b"] += wai(r,arm)
    only_b = sum(1 for v in cl.values() if v["b"]>0 and v["a"]==0)
    only_a = sum(1 for v in cl.values() if v["a"]>0 and v["b"]==0)
    both   = sum(1 for v in cl.values() if v["a"]>0 and v["b"]>0)
    return {"clusters": len(cl), "baseline_only": only_b, "freshcache_only": only_a,
            "both": both, "exact_mcnemar_p": exact_mcnemar(only_b, only_a)}

res = {}
for arm, label in [("sttl_t060_k1_2","SemanticTTL theta=0.60 kappa=1/2 (quality-selected)"),
                   ("sttl_k1_16","SemanticTTL kappa=1/16 (drift-selected)")]:
    req = {"baseline_only": sum(1 for r in rows if wai(r,arm) and not wai(r,"fc")),
           "freshcache_only": sum(1 for r in rows if wai(r,"fc") and not wai(r,arm)),
           "both": sum(1 for r in rows if wai(r,arm) and wai(r,"fc"))}
    req["exact_mcnemar_p"] = exact_mcnemar(req["baseline_only"], req["freshcache_only"])
    cl = cluster_test(arm)
    res[arm] = {"label": label, "request_level": req, "cluster_level": cl,
                "baseline_wai": g[arm], "freshcache_wai": g["fc"]}
    print(f"\n{label}")
    print(f"  request level: baseline-only {req['baseline_only']}, FC-only {req['freshcache_only']}, "
          f"both {req['both']}, exact McNemar p={req['exact_mcnemar_p']:.3g}")
    print(f"  cluster level: {cl['clusters']} clusters; baseline-only {cl['baseline_only']}, "
          f"FC-only {cl['freshcache_only']}, both {cl['both']}, exact McNemar p={cl['exact_mcnemar_p']:.3g}")

(OUT/"aq07_cluster_quality_selected.json").write_text(json.dumps(
    {"n_scored": len(rows), "gate": {"fc":7,"sttl_k1_16":64,"sttl_t060_k1_2":25,"passed":True},
     "arms": res,
     "note":("The 54-vs-0 cluster result previously quoted belongs to the drift-selected "
             "kappa=1/16 arm. This file gives each arm its own cluster-level test.")}, indent=1))
print("\nwrote artifacts/aq07_cluster_quality_selected.json")
