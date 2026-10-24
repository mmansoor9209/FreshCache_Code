"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

SemanticTTL / CategoryAware stage 4: judging and scoring.
  items : judge items for held-out and combined-validation answers
  score : held-out audit, validated subset, pairing, combined-validation check, outputs and report"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, glob, hashlib, json, os, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_category_aware"))
import baseline_selection_helpers as X
ST3 = json.load(open(X.O / "work/stage3.json")); KEYS, OPS = ST3["keys"], ST3["ops"]; FZ = json.load(open(X.O / "frozen_selection.json"))
SAMPLE = {d["query_id"]: d for d in json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]}
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
VK = ST3["ca_validation_combined"]["keys"]; QK = {r["query_id"]: r for r in json.load(open(X.CRR / "results/quality_selection_keys_reported.json"))["records"]}
hl, vl = X.Held(), X.Val()
if sys.argv[1] == "items":
    a, m1 = X.judge_items(hl, [(SAMPLE[q]["query"], SAMPLE[q]["gold"], v["key"]) for arm in KEYS.values() for q, v in arm.items()])
    b, m2 = X.judge_items(vl, [(QS[q]["query"], QS[q]["gold"], v["key"]) for q, v in VK.items()])
    if m1 or m2: X.log(f"STOP: missing answers {len(m1)} held-out, {len(m2)} validation"); sys.exit(1)
    seen, items = set(), []
    for it in a + b:
        if tuple(it["sig"]) not in seen: seen.add(tuple(it["sig"])); items.append(it)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("judge2 items:", len(items)); sys.exit(0)
ARMS = ["FreshCache", "SemanticTTL_0.60_k0.5_fixed", "A1", "A2", "CategoryAware_style"]
L = {}
for arm in ["fresh"] + ARMS:
    for q, v in KEYS[arm].items():
        l, a, _ = hl.labels(SAMPLE[q]["query"], SAMPLE[q]["gold"], v["key"]); L[(arm, q)] = (l if l is not None else "UNRESOLVED", a)
VALID_SUB = {r["query_id"] for r in csv.DictReader(open((OUTPUT_ROOT / "frozen_inputs/generated/request_lists/model_majority_valid.csv")))}
def stats(arm, ids):
    fc = [q for q in ids if L[("fresh", q)][0] == "CORRECT"]; c = collections.Counter(L[(arm, q)][0] for q in ids); w = sum(1 for q in fc if L[(arm, q)][0] == "WRONG")
    out = {"N": len(ids), "fresh_correct": len(fc), "CORRECT": c["CORRECT"], "WRONG": c["WRONG"], "ABSTAIN": c["ABSTAIN"], "UNRESOLVED": c["UNRESOLVED"], "accuracy_pct": round(100 * c["CORRECT"] / len(ids), 3),
           "WAI": w, "C_WAI_pct": round(100 * w / len(fc), 3) if fc else None, "fresh_correct_to_abstain": sum(1 for q in fc if L[(arm, q)][0] == "ABSTAIN")}
    if arm != "FreshCache":
        wa = lambda a_, q: L[("fresh", q)][0] == "CORRECT" and L[(a_, q)][0] == "WRONG"
        b_ = sum(1 for q in ids if wa(arm, q) and not wa("FreshCache", q)); c_ = sum(1 for q in ids if wa("FreshCache", q) and not wa(arm, q))
        x_ = sum(1 for q in ids if L[(arm, q)][0] == "CORRECT" and L[("FreshCache", q)][0] != "CORRECT"); y_ = sum(1 for q in ids if L[("FreshCache", q)][0] == "CORRECT" and L[(arm, q)][0] != "CORRECT")
        out.update({"WAI_arm_only_vs_FC": b_, "WAI_FC_only": c_, "WAI_both": sum(1 for q in ids if wa(arm, q) and wa("FreshCache", q)), "WAI_mcnemar_p": X.mcnemar(b_, c_),
                    "correct_arm_only_vs_FC": x_, "correct_FC_only": y_, "correct_mcnemar_p": X.mcnemar(x_, y_)})
    return out
ALL = sorted(SAMPLE); SUB = sorted(VALID_SUB)
QA = {a: stats(a, ALL) for a in ARMS}; QV = {a: stats(a, SUB) for a in ARMS}
G = {"FreshCache fresh-correct / correct / WAI (run-time reference)": REF.check("heldout_audit/fresh_correct", QA["FreshCache"]["fresh_correct"]) and REF.check("heldout_audit/FreshCache_primary/CORRECT", QA["FreshCache"]["CORRECT"]) and REF.check("heldout_audit/FreshCache_primary/WAI", QA["FreshCache"]["WAI"]),
     "fixed SemanticTTL (0.60,1/2) correct / WAI (run-time reference)": REF.check("heldout_audit/SemanticTTL_0.60_k0.5_fixed/CORRECT", QA["SemanticTTL_0.60_k0.5_fixed"]["CORRECT"]) and REF.check("heldout_audit/SemanticTTL_0.60_k0.5_fixed/WAI", QA["SemanticTTL_0.60_k0.5_fixed"]["WAI"]),
     "no unresolved held-out labels": all(QA[a]["UNRESOLVED"] == 0 for a in ARMS), "validated subset size (run-time reference)": REF.check("reference_validation/validated_subset_size", len(SUB)) and VALID_SUB <= set(ALL)}
# frozen 3-judge validated-subset protocol: reusable only if every (query_id, answer) task already has three judgments
import answer_pools as P
J3 = P.judge3(extra=[X.CRR / "shadow/reported/model_based/outputs/parsed"])
cov = {a: sum(1 for q in SUB if P.tid(q, L[(a, q)][1] or "") in J3 and len(J3[P.tid(q, L[(a, q)][1] or "")]) == 3) for a in ARMS}
# combined CategoryAware-style validation check (report only)
fresh_v = {q: vl.labels(QS[q]["query"], QS[q]["gold"], QK[q]["fresh_key"])[0] for q in VK}; ca_v = {q: vl.labels(QS[q]["query"], QS[q]["gold"], VK[q]["key"])[0] for q in VK}
fcn_v = sum(v == "CORRECT" for v in fresh_v.values()); ca_w = sum(1 for q in VK if fresh_v[q] == "CORRECT" and ca_v[q] == "WRONG")
REF.record("validation_quality/CategoryAware_original_WAI", ca_w)
VCHK = {"validation_fresh_correct": fcn_v, "CategoryAware_style_validation_WAI": ca_w, "C_WAI_pct": round(100 * ca_w / fcn_v, 4), "FreshCache_validation_WAI": REF.value("validation_quality/FreshCache_WAI"),
        "validation_savings_pct": ST3["ca_validation_combined"]["ops"]["search_saved_pct"], "cross_category_hits": ST3["ca_validation_combined"]["ops"]["cross_category_hits"],
        "validation_l1_hits": ST3["ca_validation_combined"]["ops"]["l1_hits"], "unresolved": sum(v is None for v in ca_v.values())}
for k, v in G.items(): X.log("GATE", k, v)
if not all(G.values()): X.log("STOP: scoring gate failed"); sys.exit(1)
PTS = []
for a in ARMS:
    o = OPS[a]; e1 = o.get("audit_secondary", {})
    PTS.append({"arm": a, "method": o["method"], "search_saved_pct": o["search_saved_pct"], "search_calls": o.get("search_calls"), "cache_hits_L1": o["l1_hits"], "fetches": o["fetches"],
                "fetch_per_1k": o["fetch_per_1k"], "gen_per_1k": o["gen_per_1k"], "generations": o.get("generations"), **{f"audit_{k}": v for k, v in QA[a].items()}, **{f"subset118_{k}": v for k, v in QV[a].items()},
                "validated_subset_frozen_3judge_reusable": f"{cov[a]}/{len(SUB)}", "drift_pct_secondary": o.get("drift_pct"), "coverage_pct_secondary": o.get("coverage_pct"),
                "audit_cross_round_drift_pct_secondary": e1.get("cross_round_drift_pct", "not computed (frozen arm; see the cross-round audit for FreshCache)"), "audit_determinate": e1.get("cross_round_determinate")})
with open(X.O / "baseline_selection_points.csv", "w", newline="") as f:
    cols = list(dict.fromkeys(k for r in PTS for k in r)); w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(PTS)
with open(X.O / "baseline_selection_request_level.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["request_id", "in_model_majority_valid", "fresh_label"] + [x for a in ARMS for x in (f"{a}_tier", f"{a}_label", f"{a}_answer_key")])
    for q in ALL: w.writerow([q, q in VALID_SUB, L[("fresh", q)][0]] + [x for a in ARMS for x in (KEYS[a][q]["tier"], L[(a, q)][0], "|".join(KEYS[a][q]["key"]))])
BEF = json.load(open(X.O / "work/read_only_before.json")); AFT = {p: X.fstat(inpath(p)) for p in BEF}; UNCH = {p: BEF[p] == AFT[p] for p in BEF}
gpu = [json.loads(l) for l in open(X.O / "work/gpu_runs.jsonl")]
VS = json.load(open(X.O / "work/validation_scores.json"))
SUM = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "design_sha256": json.load(open(X.O / "design_frozen.json"))["sha256"], "selection": FZ,
       "gates": {**json.load(open(X.O / "work/gates_stage1.json"))["gates"], **ST3["gates"], **VS["gates"], **G}, "validation_partA": VS["partA"], "validation_partB_selected_cells": {c: VS["partB_cells"][v["cell"]] for c, v in FZ["CategoryAware_style"]["per_category"].items()},
       "ca_validation_combined_check": VCHK, "heldout": PTS, "gpu_runs": gpu, "new_outputs": {"answers": len(X.jl(X.O / "gpu/out/answers_llama3b.jsonl")), "judge_llama8b": len(X.jl(X.O / "gpu/out/judge_llama8b.jsonl")), "judge_qwen7b": len(X.jl(X.O / "gpu/out/judge_qwen7b.jsonl"))},
       "read_only_files": {p: {**BEF[p], "unchanged": UNCH[p]} for p in BEF}, "all_read_only_unchanged": all(UNCH.values())}
json.dump(SUM, open(X.O / "baseline_selection_summary.json", "w"), indent=1, default=str)
X.log("held-out", json.dumps([{k: r[k] for k in ("arm", "search_saved_pct", "audit_CORRECT", "audit_WAI", "audit_C_WAI_pct", "audit_WAI_arm_only_vs_FC", "audit_WAI_FC_only", "audit_WAI_mcnemar_p") if k in r} for r in PTS]))
X.log("CA validation combined check", json.dumps(VCHK)); X.log("3-judge validated-subset coverage", cov); X.log("all read-only unchanged", SUM["all_read_only_unchanged"])
