"""Reference validation, stage 3: aggregate judge labels and compute all tables. Reads only this folder."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, csv, hashlib, math, pathlib, collections; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
from fractions import Fraction
W = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ((OUTPUT_ROOT / "reference_validation")); RES = W/"results"   # optional workspace (used by reference_and_tier_recomputation.py)
P = json.load(open(W/"config/protocol.json")); J = list(P["judges"])
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
REQ = [json.loads(l) for l in open(W/"inputs/requests.jsonl")]; EV = {r["query_id"]: r for r in jl(W/"inputs/evidence.jsonl")}
OUTS = jl(W/"inputs/policy_outputs.jsonl"); ARMS = ["fresh", "freshcache", "semanticttl_quality_0.60_1-2", "semanticttl_drift_0.40_1-16"]
def majority(labels, fallback):
    c = collections.Counter(l for l in labels if l != "UNPARSED"); top = c.most_common(1)
    return top[0][0] if top and top[0][1] >= 2 else fallback
def fleiss(rows, cats):
    rows = [r for r in rows if all(x in cats for x in r)]
    if not rows: return None
    n, N = len(rows[0]), len(rows); p = {c: sum(r.count(c) for r in rows)/(N*n) for c in cats}
    Pi = [(sum(r.count(c)**2 for c in cats) - n)/(n*(n-1)) for r in rows]; Pb, Pe = sum(Pi)/N, sum(v*v for v in p.values())
    return round((Pb - Pe)/(1 - Pe), 4) if Pe < 1 else None
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    return float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c)+1))*2, 2**n)))
def wilson(k, n, z=1.96):
    if not n: return (None, None)
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n); m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return (round(100*(c-m)/d, 2), round(100*(c+m)/d, 2))
summ = {"protocol_sha256": hashlib.sha256((W/"config/protocol.json").read_bytes()).hexdigest()}
# ---------- reference stage ----------
R = {j: {r["query_id"]: r for r in jl(W/"outputs/parsed"/f"reference_{j}.jsonl")} for j in J}
LAB = P["reference_stage"]["labels"]; ref_rows = []
for r in REQ:
    q = r["query_id"]; e = EV[q]; used = [p for p in e["pages"] if p.get("used")]
    if not used: labs = ["UNVERIFIABLE"]*3; final = "UNVERIFIABLE"; src = "no_evidence_rule"
    else:
        labs = [R[j][q]["label"] if q in R[j] else "MISSING" for j in J]; final = majority(labs, "NO_MAJORITY"); src = "majority"
    ref_rows.append({"query_id": q, "cluster_id": r["cluster_id"], "freshness_class": r["freshness_class"], "question": r["question"], "reference_answer": r["reference_answer"],
        "replay_time_s": r["replay_time_s"], "snapshot_round": r["snapshot_round"], "target_time": (e["evidence_time_max"] or "")[:10],
        "evidence_captured_min": e["evidence_time_min"], "evidence_captured_max": e["evidence_time_max"], "n_pages_used": e["n_pages_used"], "evidence_chars": e["evidence_chars"],
        "evidence_url_hashes": "|".join(p["url_hash"] for p in used),
        **{f"{j}_raw": (R[j][q]["raw"] if q in R[j] else "") for j in J}, **{f"{j}_label": l for j, l in zip(J, labs)},
        "input_truncated_any": any(R[j].get(q, {}).get("truncated", False) for j in J),
        "n_agree_final": sum(1 for l in labs if l == final), "unanimous": len(set(labs)) == 1, "final_label": final, "final_source": src,
        "in_validated_subset": final == "VALID_AT_T"})
with open(RES/"reference_validation_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(ref_rows[0])); w.writeheader(); w.writerows(ref_rows)
with open(RES/"reference_disagreement_cases.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(ref_rows[0])); w.writeheader(); w.writerows([x for x in ref_rows if not x["unanimous"]])
VAL = {x["query_id"] for x in ref_rows if x["in_validated_subset"]}; REF.record("reference_validation/validated_subset_size", len(VAL))
summ["reference"] = {"n": len(ref_rows), "final": dict(collections.Counter(x["final_label"] for x in ref_rows)),
    "per_judge": {j: dict(collections.Counter(x[f"{j}_label"] for x in ref_rows)) for j in J},
    "unanimous": sum(x["unanimous"] for x in ref_rows), "exactly_two_agree": sum(1 for x in ref_rows if x["n_agree_final"] == 2 and x["final_label"] != "NO_MAJORITY"),
    "no_majority": sum(1 for x in ref_rows if x["final_label"] == "NO_MAJORITY"), "fleiss_kappa": fleiss([[x[f"{j}_label"] for j in J] for x in ref_rows], LAB),
    "pairwise_agreement_pct": {f"{a}|{b}": round(100*sum(1 for x in ref_rows if x[f"{a}_label"] == x[f"{b}_label"])/len(ref_rows), 2) for i, a in enumerate(J) for b in J[i+1:]},
    "unparsed": {j: sum(1 for x in ref_rows if x[f"{j}_label"] == "UNPARSED") for j in J}, "inputs_truncated": sum(x["input_truncated_any"] for x in ref_rows),
    "validated_subset_n": len(VAL), "validated_by_class": dict(collections.Counter(x["freshness_class"] for x in ref_rows if x["in_validated_subset"])),
    "valid_unanimous": sum(1 for x in ref_rows if x["in_validated_subset"] and x["unanimous"])}
# ---------- answer stage ----------
A = {j: {r["task_id"]: r for r in jl(W/"outputs/parsed"/f"answer_{j}.jsonl")} for j in J}
ABST = P["answer_stage"]["abstain_patterns"]; lab = {}; arows = []
for o in OUTS:
    tid = hashlib.sha256(f"{o['query_id']}\x00{o['answer']}".encode()).hexdigest()[:16]
    labs = [A[j][tid]["label"] if tid in A[j] else "MISSING" for j in J]
    pat = any(p in (o["answer"] or "").strip().lower() for p in ABST)
    final = "ABSTAIN" if pat else majority(labs, "UNRESOLVABLE")
    lab[(o["query_id"], o["arm"])] = final
    arows.append({"query_id": o["query_id"], "arm": o["arm"], "tier": o["tier"], "task_id": tid, "answer": o["answer"], **{f"{j}_raw": A[j].get(tid, {}).get("raw", "") for j in J},
                  **{f"{j}_label": l for j, l in zip(J, labs)}, "abstain_pattern": pat, "final_label": final, "in_validated_subset": o["query_id"] in VAL})
with open(RES/"answer_judgments_per_output.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(arows[0])); w.writeheader(); w.writerows(arows)
summ["answer_judging"] = {"unique_tasks": len({x["task_id"] for x in arows}), "fleiss_kappa_on_judged": fleiss([[x[f"{j}_label"] for j in J] for x in {x["task_id"]: x for x in arows}.values()], P["answer_stage"]["labels"]),
    "unparsed": {j: sum(1 for x in {x["task_id"]: x for x in arows}.values() if x[f"{j}_label"] == "UNPARSED") for j in J}}
def metrics(ids, name, lab=lab):
    ids = sorted(ids); n = len(ids); fc = [q for q in ids if lab[(q, "fresh")] == "CORRECT"]; out = []
    for arm in ARMS:
        L = collections.Counter(lab[(q, arm)] for q in ids)
        row = {"population": name, "arm": arm, "N": n, "correct": L["CORRECT"], "wrong": L["WRONG"], "abstain": L["ABSTAIN"], "unresolvable": L["UNRESOLVABLE"],
               "accuracy_pct": round(100*L["CORRECT"]/n, 2) if n else None, "accuracy_ci95": wilson(L["CORRECT"], n), "abstention_rate_pct": round(100*L["ABSTAIN"]/n, 2) if n else None,
               "fresh_correct": len(fc)}
        if arm != "fresh":
            wai = sum(1 for q in fc if lab[(q, arm)] == "WRONG")
            cw = sum(1 for q in ids if lab[(q, "fresh")] == "CORRECT" and lab[(q, arm)] != "CORRECT"); wc = sum(1 for q in ids if lab[(q, "fresh")] != "CORRECT" and lab[(q, arm)] == "CORRECT")
            row.update({"wai": wai, "wai_pct": round(100*wai/n, 2) if n else None, "wai_ci95": wilson(wai, n), "conditional_wai_pct": round(100*wai/len(fc), 2) if fc else None,
                        "fresh_correct_to_abstain": sum(1 for q in fc if lab[(q, arm)] == "ABSTAIN"), "fresh_correct_to_unresolvable": sum(1 for q in fc if lab[(q, arm)] == "UNRESOLVABLE"),
                        "vs_fresh_correct_lost": cw, "vs_fresh_correct_gained": wc, "vs_fresh_mcnemar_p": mcnemar(cw, wc)})
        out.append(row)
    return out
def paired(ids, name, lab=lab):
    out = []
    for b in ("semanticttl_quality_0.60_1-2", "semanticttl_drift_0.40_1-16"):
        fcok = lambda q: lab[(q, "freshcache")] == "CORRECT"; bok = lambda q: lab[(q, b)] == "CORRECT"
        w = lambda q, a: lab[(q, "fresh")] == "CORRECT" and lab[(q, a)] == "WRONG"
        c1 = sum(1 for q in ids if fcok(q) and not bok(q)); c2 = sum(1 for q in ids if bok(q) and not fcok(q))
        w1 = sum(1 for q in ids if w(q, "freshcache") and not w(q, b)); w2 = sum(1 for q in ids if w(q, b) and not w(q, "freshcache")); wb = sum(1 for q in ids if w(q, b) and w(q, "freshcache"))
        out.append({"population": name, "contrast": f"freshcache vs {b}", "N": len(ids), "correct_freshcache_only": c1, "correct_baseline_only": c2, "correctness_mcnemar_p": mcnemar(c1, c2),
                    "wai_freshcache_only": w1, "wai_baseline_only": w2, "wai_both": wb, "wai_mcnemar_p": mcnemar(w1, w2)})
    return out
ALL = [r["query_id"] for r in REQ]
PAT = {(x["query_id"], x["arm"]): x["abstain_pattern"] for x in arows}
labS = {k: ("WRONG" if v == "ABSTAIN" and not PAT[k] else v) for k, v in lab.items()}   # sensitivity: preserved pattern-only abstention
summ["judge_only_abstain_reclassified"] = sum(1 for k, v in lab.items() if v == "ABSTAIN" and not PAT[k])
M = metrics(ALL, "all_audit") + metrics(VAL, "validated_subset") + metrics(ALL, "all_audit__abstain_pattern_only", labS) + metrics(VAL, "validated_subset__abstain_pattern_only", labS)
PT = paired(ALL, "all_audit") + paired(sorted(VAL), "validated_subset") + paired(ALL, "all_audit__abstain_pattern_only", labS) + paired(sorted(VAL), "validated_subset__abstain_pattern_only", labS)
for fn, rows in (("answer_quality_results.csv", M), ("paired_tests.csv", PT)):
    with open(RES/fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list({k: 1 for r in rows for k in r})); w.writeheader(); w.writerows(rows)
summ["answer_quality"] = M; summ["paired"] = PT
json.dump(summ, open(RES/"summary.json", "w"), indent=1, default=str); print(json.dumps({k: v for k, v in summ.items() if k not in ("answer_quality", "paired")}, indent=1))
for r in M: print(r["population"], r["arm"], r["N"], r["accuracy_pct"], r.get("wai"), r.get("conditional_wai_pct"), r["abstain"], r["unresolvable"])
for r in PT: print(r)
