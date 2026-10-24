"""Aggregate judgments and compute both diagnostics. Reads only this folder plus the read-only validated-reference table."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import sys, json, csv, math, re, pathlib, collections; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ((OUTPUT_ROOT / "tier_quality")); RES = W/"results"   # optional workspace (used by reference_and_tier_recomputation.py)
AP = json.load(open(ROOT / "configs/protocols/reference_validity_protocol.json")); J = list(AP["judges"]); ABST = AP["answer_stage"]["abstain_patterns"]
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def parse_ans(raw):
    u = re.sub(r"\bINCORRECT\b", "WRONG", raw.upper()); u = re.sub(r"^(WROONG|WONG)\b", "WRONG", u)
    hits = [(u.find(l), l) for l in ("CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE") if u.find(l) >= 0]; return min(hits)[1] if hits else "UNPARSED"
def parse_suf(raw):
    u = raw.upper()
    for l in ("INSUFFICIENT", "UNCLEAR", "SUFFICIENT"):
        if l in u: return l
    return "UNPARSED"
def maj(ls, fb):
    c = collections.Counter(l for l in ls if l not in ("UNPARSED", "MISSING")).most_common(1); return c[0][0] if c and c[0][1] >= 2 else fb
def mcn(b, c):
    n = b + c; return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c)+1))*2, 2**n)))
def fleiss(rows, cats):
    rows = [r for r in rows if all(x in cats for x in r)]; n, N = 3, len(rows)
    if not N: return None
    p = {c: sum(r.count(c) for r in rows)/(N*n) for c in cats}; Pb = sum((sum(r.count(c)**2 for c in cats) - n)/(n*(n-1)) for r in rows)/N; Pe = sum(v*v for v in p.values())
    return round((Pb-Pe)/(1-Pe), 4)
S = {}
# ---- parse and aggregate ----
AJ = {j: {r["task_id"]: r for r in jl(W/"outputs/raw"/f"answer_{j}.jsonl")} for j in J}
SJ = {j: {r["task_id"]: r for r in jl(W/"outputs/raw"/f"sufficiency_{j}.jsonl")} for j in J}
for kind, D, fn in (("answer", AJ, parse_ans), ("sufficiency", SJ, parse_suf)):
    for j in J:
        with open(W/"outputs/parsed"/f"{kind}_{j}.jsonl", "w", encoding="utf-8") as f:
            for r in D[j].values(): r["label"] = fn(r["raw"]); f.write(json.dumps(r, ensure_ascii=False) + "\n")
AT = {t["task_id"]: t for t in jl(W/"inputs/answer_tasks.jsonl")}
alab = {}
for tid, t in AT.items():
    ls = [AJ[j][tid]["label"] if tid in AJ[j] else "MISSING" for j in J]
    alab[tid] = ("ABSTAIN" if any(p in t["candidate"].strip().lower() for p in ABST) else maj(ls, "UNRESOLVABLE"), ls)
slab = {tid: (maj([SJ[j][tid]["label"] if tid in SJ[j] else "MISSING" for j in J], "NO_MAJORITY"), [SJ[j][tid]["label"] if tid in SJ[j] else "MISSING" for j in J]) for tid in {r["task_id"] for r in jl(W/"inputs/sufficiency_tasks.jsonl")}}
S["judging"] = {"answer_tasks": len(AT), "sufficiency_tasks": len(slab), "answer_fleiss": fleiss([v[1] for v in alab.values()], ["CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE"]),
                "sufficiency_fleiss": fleiss([v[1] for v in slab.values()], ["SUFFICIENT", "INSUFFICIENT", "UNCLEAR"]),
                "unparsed": {f"{k}_{j}": sum(1 for r in D[j].values() if r["label"] == "UNPARSED") for k, D in (("answer", AJ), ("sufficiency", SJ)) for j in J}}
PR = jl(W/"inputs/path_rows.jsonl")
out = []
for r in PR:
    fl, ls = alab[r["answer_task_id"]]; row = {**{k: r[k] for k in ("task", "query_id", "path", "target_date", "context_empty", "answer")}, **{f"answer_{j}": l for j, l in zip(J, ls)}, "answer_final": fl}
    if r["sufficiency_task_id"]:
        sf, ss = slab[r["sufficiency_task_id"]]; row.update({**{f"suff_{j}": l for j, l in zip(J, ss)}, "suff_final": sf})
    out.append(row)
L = {(r["task"], r["query_id"], r["path"]): r for r in out}
VALID = {r["query_id"] for r in csv.DictReader(open((OUTPUT_ROOT / "reference_validation/results/reference_validation_results.csv"))) if r["final_label"] == "VALID_AT_T"}
# ---- consistency with the earlier study (same answers, same protocol) ----
prev = {(r["query_id"], r["arm"]): r["final_label"] for r in csv.DictReader(open((OUTPUT_ROOT / "reference_validation/results/answer_judgments_per_output.csv")))}
cmp = [(L[("l2_reuse", q, p)]["answer_final"], prev[(q, arm)]) for (t, q, p) in L if t == "l2_reuse" for (pp, arm) in (("A_fresh_retrieval", "fresh"), ("C_actual_freshcache", "freshcache")) if p == pp]
S["consistency_with_model_based_reference_validation"] = {"n": len(cmp), "identical_final_label": sum(a == b for a, b in cmp)}
def per_path(task, paths, ids, label="answer_final"):
    res = {}
    for p in paths:
        c = collections.Counter(L[(task, q, p)][label] for q in ids); res[p] = {"N": len(ids), **dict(c)}
        if label == "answer_final": res[p]["accuracy_pct"] = round(100*c["CORRECT"]/len(ids), 2) if ids else None
        else: res[p]["sufficient_pct"] = round(100*c["SUFFICIENT"]/len(ids), 2) if ids else None
    return res
def pair(task, a, b, ids, good="CORRECT", label="answer_final"):
    ga = lambda q, p: L[(task, q, p)][label] == good
    x = sum(1 for q in ids if ga(q, a) and not ga(q, b)); y = sum(1 for q in ids if ga(q, b) and not ga(q, a))
    w = sum(1 for q in ids if L[(task, q, b)][label] == good and L[(task, q, a)][label] == "WRONG") if label == "answer_final" else None
    return {"contrast": f"{a} vs {b}", "N": len(ids), f"{good.lower()}_{a}_only": x, f"{good.lower()}_{b}_only": y, "mcnemar_p": mcn(x, y), **({f"{b}_correct_{a}_wrong": w} if w is not None else {})}
tables, ptables = [], []
# ---------- L1 reuse study ----------
T1 = {x["query_id"]: x for x in jl(W/"inputs/l1_reuse_hits.jsonl")}; P1 = ["A_stored_answer", "B_regen_stored_evidence", "C_fresh"]
pops1 = {"all": sorted(T1), "validated_reference": sorted(q for q in T1 if q in VALID), "stored_context_nonempty": sorted(q for q in T1 if not T1[q]["stored_context_empty"]),
         "stored_context_empty": sorted(q for q in T1 if T1[q]["stored_context_empty"]), "both_contexts_nonempty": sorted(q for q in T1 if not T1[q]["stored_context_empty"] and not T1[q]["fresh_context_empty"]),
         "query_equivalent_SAME": sorted(q for q in T1 if T1[q]["equivalence_4juror"] == "SAME"), "query_mismatch_DIFFERENT_or_TIE": sorted(q for q in T1 if T1[q]["equivalence_4juror"] != "SAME")}
for name, ids in pops1.items():
    for p, v in per_path("l1_reuse", P1, ids).items(): tables.append({"task": "l1_reuse", "population": name, "path": p, **v})
    for a, b in (("A_stored_answer", "C_fresh"), ("B_regen_stored_evidence", "C_fresh"), ("A_stored_answer", "B_regen_stored_evidence")): ptables.append({"task": "l1_reuse", "population": name, **pair("l1_reuse", a, b, ids)})
mm = pops1["query_mismatch_DIFFERENT_or_TIE"]
S["l1_reuse_mismatch_but_correct"] = {"mismatch_hits": len(mm), "A_correct": sum(1 for q in mm if L[("l1_reuse", q, "A_stored_answer")]["answer_final"] == "CORRECT"),
    "A_correct_and_C_correct": sum(1 for q in mm if L[("l1_reuse", q, "A_stored_answer")]["answer_final"] == "CORRECT" and L[("l1_reuse", q, "C_fresh")]["answer_final"] == "CORRECT"),
    "A_correct_C_not": sum(1 for q in mm if L[("l1_reuse", q, "A_stored_answer")]["answer_final"] == "CORRECT" and L[("l1_reuse", q, "C_fresh")]["answer_final"] != "CORRECT"),
    "by_equivalence_label": {e: sum(1 for q in mm if T1[q]["equivalence_4juror"] == e and L[("l1_reuse", q, "A_stored_answer")]["answer_final"] == "CORRECT") for e in ("DIFFERENT", "JURY_TIE")}}
S["l1_reuse_populations"] = {k: len(v) for k, v in pops1.items()}
# ---------- L2 reuse study ----------
T2 = {x["query_id"]: x for x in jl(W/"inputs/l2_reuse_hits.jsonl")}; P2 = ["A_fresh_retrieval", "B_cached_list_current_pages", "C_actual_freshcache"]
diffBC = sorted(q for q in T2 if T2[q]["paths"]["B_cached_list_current_pages"]["context"] != T2[q]["paths"]["C_actual_freshcache"]["context"])
sameAB = sorted(q for q in T2 if T2[q]["paths"]["A_fresh_retrieval"]["context"] == T2[q]["paths"]["B_cached_list_current_pages"]["context"])
pops2 = {"all": sorted(T2), "validated_reference": sorted(q for q in T2 if q in VALID), "B_context_differs_from_C": diffBC, "A_context_differs_from_B": sorted(set(T2) - set(sameAB))}
for name, ids in pops2.items():
    for p, v in per_path("l2_reuse", P2, ids).items(): tables.append({"task": "l2_reuse_answer", "population": name, "path": p, **v})
    for p, v in per_path("l2_reuse", P2, ids, "suff_final").items(): tables.append({"task": "l2_reuse_sufficiency", "population": name, "path": p, **v})
    for a, b in (("A_fresh_retrieval", "B_cached_list_current_pages"), ("B_cached_list_current_pages", "C_actual_freshcache"), ("A_fresh_retrieval", "C_actual_freshcache")):
        ptables.append({"task": "l2_reuse_answer", "population": name, **pair("l2_reuse", a, b, ids)})
        ptables.append({"task": "l2_reuse_sufficiency", "population": name, **pair("l2_reuse", a, b, ids, "SUFFICIENT", "suff_final")})
S["l2_reuse_populations"] = {k: len(v) for k, v in pops2.items()}
S["l2_reuse_context_identity"] = {"A_equals_B": len(sameAB), "B_equals_C": len(T2) - len(diffBC), "N": len(T2)}
with open(RES/"per_request_path_judgments.csv", "w", newline="", encoding="utf-8") as f:
    keys = list({k: 1 for r in out for k in r}); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
for fn, rows in (("path_results.csv", tables), ("paired_tests.csv", ptables)):
    with open(RES/fn, "w", newline="") as f:
        keys = list({k: 1 for r in rows for k in r}); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump(S, open(RES/"summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
for r in tables: print(r)
for r in ptables: print(r)
