"""Unseen-query experiment, scoring stage. Modes: items | score.
Correctness: frozen two-judge rule (Llama-3.1-8B and Qwen2.5-7B both CORRECT; frozen held-out abstention patterns first), reference = frozen gold answer.
Sufficiency: frozen SUFFV two-judge rule on each path's evidence block (agree -> label, else JUDGE_DISAGREEMENT).
PRIMARY comparison = A_matched (fresh top-k, k = cached-list length) vs B (cached list, pages fetched in this experiment), paired, on cases where
both paths have usable pages and a reference exists; A_full (top-10) and C (full FreshCache) are reported separately. Exact McNemar; 95% CI for the
paired difference (Newcombe-style Wald on discordant counts). Cost: the concurrency study's live/local component cost model per path."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, hashlib, json, math, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "unseen_queries")); MODE = sys.argv[1]
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
C = jl(O / "work/cases_built.jsonl"); PATHS = ["A_matched", "A_full", "B_cached_live", "C_freshcache"]
def merged(stem):
    out = {}
    for f in [O / "gpu/out" / f"{stem}.jsonl"] + sorted((O / "gpu/out").glob(f"{stem}_shard*.jsonl")):
        for r in jl(f): out.setdefault(tuple(r["key"]) if "key" in r else (tuple(r["sig"]) if "sig" in r else (r["query_id"], r["arm"])), r)
    return out
ANS = {k: r["answer"] for k, r in merged("answers_llama3b").items()}
if MODE == "items":
    miss = [c["case_id"] for c in C for p in PATHS if tuple(c["paths"][p]["key"]) not in ANS]
    if miss: print("STOP: missing answers", len(miss)); sys.exit(1)
    items = {}
    for c in C:
        if not c["reference"]: continue
        for p in PATHS:
            a = ANS[tuple(c["paths"][p]["key"])]
            if any(x in a.strip().lower() for x in HELD): continue
            sig = [h16(c["query"]), h16(c["reference"]), h16(a)]; items.setdefault(tuple(sig), {"sig": sig, "q": c["query"], "gold": c["reference"], "cand": a})
    with open(O / "gpu/in/judge2_items.jsonl", "w", encoding="utf-8") as f:
        for x in items.values(): f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("judge2 items", len(items), "suffv tasks", len(jl(O / "gpu/in/suffv_tasks.jsonl"))); sys.exit(0)
J = {j: {tuple(r["sig"]): r["label"] for r in merged(f"judge_{j}").values()} for j in ("llama8b", "qwen7b")}
S = {j: {(r["query_id"], r["arm"]): r["label"] for r in merged(f"suffv_{j}").values()} for j in ("llama8b", "qwen7b")}
def label(c, p):
    a = ANS[tuple(c["paths"][p]["key"])]
    if any(x in a.strip().lower() for x in HELD): return "ABSTAIN"
    sig = (h16(c["query"]), h16(c["reference"]), h16(a)); l = [J[j].get(sig) for j in J]
    if None in l: return "UNRESOLVED"
    return "CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"
def suff(c, p):
    k = (c["query_id"], c["paths"][p]["evidence_sha"]); l = [S[j].get(k) for j in S]
    if None in l: return "UNRESOLVED"
    return l[0] if l[0] == l[1] else "JUDGE_DISAGREEMENT"
def mcn(b, c):
    n = b + c; return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def pdiff(b, c, n):   # paired difference (second - first) = (c - b)/n, Wald CI on discordant pairs
    d = (c - b) / n; se = math.sqrt(max((b + c) - (c - b) ** 2 / n, 0)) / n; return [round(100 * d, 2), round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)]
CO = json.load(open((OUTPUT_ROOT / "concurrency_cost/component_costs.json")))["coefficients"]
def cost(search, fetch, gen):
    return {sc: round(search * CO["c_search_ms"][sc] + fetch * CO["c_fetch_ms"][sc] + gen * CO["c_gen_ms"]["value"] + CO["c_embed_ms"]["value"], 1) for sc in ("live_p50", "local")}
PAIR = [c for c in C if c["both_evaluable"] and c["reference_available"]]
JUDGE_UNRES = [c["case_id"] for c in PAIR if any(label(c, p) == "UNRESOLVED" for p in ("A_matched", "B_cached_live")) or any(suff(c, p) == "UNRESOLVED" for p in ("A_matched", "B_cached_live"))]
FINAL = [c for c in PAIR if c["case_id"] not in JUDGE_UNRES]
L = {p: {c["case_id"]: (label(c, p), suff(c, p)) for c in FINAL} for p in PATHS}
def compare(a, b, ids):
    n = len(ids); A, B = L[a], L[b]
    t = {"first_SUFF_second_INSUFF": sum(1 for i in ids if A[i][1] == "SUFFICIENT" and B[i][1] != "SUFFICIENT"), "first_INSUFF_second_SUFF": sum(1 for i in ids if A[i][1] != "SUFFICIENT" and B[i][1] == "SUFFICIENT"),
         "first_CORRECT_second_WRONG": sum(1 for i in ids if A[i][0] == "CORRECT" and B[i][0] == "WRONG"), "first_WRONG_or_ABSTAIN_second_CORRECT": sum(1 for i in ids if A[i][0] in ("WRONG", "ABSTAIN") and B[i][0] == "CORRECT"),
         "first_CORRECT_second_not": sum(1 for i in ids if A[i][0] == "CORRECT" and B[i][0] != "CORRECT"), "first_not_second_CORRECT": sum(1 for i in ids if A[i][0] != "CORRECT" and B[i][0] == "CORRECT")}
    sa, sb = sum(1 for i in ids if A[i][1] == "SUFFICIENT"), sum(1 for i in ids if B[i][1] == "SUFFICIENT"); ca, cb = sum(1 for i in ids if A[i][0] == "CORRECT"), sum(1 for i in ids if B[i][0] == "CORRECT")
    return {"N": n, f"{a}_sufficient": sa, f"{b}_sufficient": sb, "sufficiency_diff_pp_CI95": pdiff(t["first_SUFF_second_INSUFF"], t["first_INSUFF_second_SUFF"], n), "sufficiency_mcnemar_p": mcn(t["first_SUFF_second_INSUFF"], t["first_INSUFF_second_SUFF"]),
            f"{a}_correct": ca, f"{b}_correct": cb, "correctness_diff_pp_CI95": pdiff(t["first_CORRECT_second_not"], t["first_not_second_CORRECT"], n), "correctness_mcnemar_p": mcn(t["first_CORRECT_second_not"], t["first_not_second_CORRECT"]),
            f"{a}_abstain": sum(1 for i in ids if A[i][0] == "ABSTAIN"), f"{b}_abstain": sum(1 for i in ids if B[i][0] == "ABSTAIN"),
            "WAI_second_vs_first": t["first_CORRECT_second_WRONG"], "WAI_first_vs_second": sum(1 for i in ids if B[i][0] == "CORRECT" and A[i][0] == "WRONG"), "transitions": t}
ids = [c["case_id"] for c in FINAL]; byid = {c["case_id"]: c for c in FINAL}
OPS = {}
for p in PATHS:
    s = 1 if p.startswith("A") else 0
    f = [c["k"] if p in ("A_matched", "B_cached_live") else (len(c["fresh_links_top10"]) if p == "A_full" else c["k"] - c["l3_served_pages_C"]) for c in FINAL]
    OPS[p] = {"search_calls_per_request": s, "page_fetches_per_request": round(sum(f) / len(f), 3), "generations_per_request": 1, **{f"cost_ms_{k}": v for k, v in cost(s, sum(f) / len(f), 1).items()}}
RES = {"flow": {**json.load(open(O / "work/build_summary.json"))["flow"], "judge_unresolved_in_primary_pair": len(JUDGE_UNRES), "final_paired_set": len(FINAL)},
       "exclusion_reasons": json.load(open(O / "work/build_summary.json"))["exclusion_reasons"],
       "overlap_on_final_set": {"mean_url_overlap_k": round(sum(byid[i]["url_overlap_k"] for i in ids) / len(ids), 3), "share_any_url_overlap_k": round(sum(1 for i in ids if byid[i]["url_overlap_k"] > 0) / len(ids), 4),
                                "share_any_domain_overlap_k": round(sum(1 for i in ids if byid[i]["domain_overlap_k"] > 0) / len(ids), 4), "share_any_url_overlap_top10": round(sum(1 for i in ids if byid[i]["url_overlap_top10"] > 0) / len(ids), 4),
                                "share_any_domain_overlap_top10": round(sum(1 for i in ids if byid[i]["domain_overlap_top10"] > 0) / len(ids), 4), "k_distribution": dict(collections.Counter(byid[i]["k"] for i in ids))},
       "PRIMARY_matched_depth_A_vs_B": compare("A_matched", "B_cached_live", ids), "C_full_freshcache_vs_A_matched": compare("A_matched", "C_freshcache", ids),
       "B_vs_C_page_reuse_effect": compare("B_cached_live", "C_freshcache", ids), "SECONDARY_full_search_A_full_vs_B": compare("A_full", "B_cached_live", ids), "operations_and_cost": OPS}
a, b = OPS["A_matched"], OPS["B_cached_live"]
RES["headline"] = {"sufficiency_change_pp_B_minus_A_matched": RES["PRIMARY_matched_depth_A_vs_B"]["sufficiency_diff_pp_CI95"], "retrieval_cost_reduction_pct_live": round(100 * (1 - (b["cost_ms_live_p50"] - CO["c_gen_ms"]["value"]) / (a["cost_ms_live_p50"] - CO["c_gen_ms"]["value"])), 2),
                   "total_cost_reduction_pct_live": round(100 * (1 - b["cost_ms_live_p50"] / a["cost_ms_live_p50"]), 2)}
json.dump(RES, open(O / "unseen_query_results.json", "w"), indent=1)
print(json.dumps(RES, indent=1)[:6000])
