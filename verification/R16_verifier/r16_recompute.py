"""R16 audit recompute. Read-only over validation_V2/l2_evidence_verification/. Writes only into this folder."""
import json, csv, pickle, hashlib, math, collections, pathlib, sys, os, datetime
from fractions import Fraction
ROOT = pathlib.Path("<PROJECT_ROOT>"); S = ROOT/"validation_V2/l2_evidence_verification"; O = S/"out"; OUT = pathlib.Path(__file__).parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12", "validation/mixed_age_full_policy_audit"):
    sys.path.insert(0, str(ROOT/p) if p else str(ROOT))
sys.path.insert(0, str(S)); os.chdir(ROOT)
R = {"utc": datetime.datetime.utcnow().isoformat()+"Z"}
def say(*a): print(*a, flush=True)
# ---- 1. prereg hash
P = json.load(open(O/"prereg_gamma.json")); emb = P.pop("prereg_sha256")
R["prereg"] = {"embedded_hash": emb, "recomputed_content_hash": hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest(), "file_sha256": hashlib.sha256((O/"prereg_gamma.json").read_bytes()).hexdigest(),
    "mtime_utc": datetime.datetime.utcfromtimestamp((O/"prereg_gamma.json").stat().st_mtime).isoformat()}
R["prereg"]["content_hash_matches"] = R["prereg"]["embedded_hash"] == R["prereg"]["recomputed_content_hash"]
R["timeline_mtime_utc"] = {f: datetime.datetime.utcfromtimestamp((O/f).stat().st_mtime).isoformat() for f in ("prereg_gamma.json","val_label_tasks.jsonl","val_support_llama8b.jsonl","val_support_qwen7b.jsonl","val_labels.jsonl","gamma_selection.json","test_scores_L2Verify_g0.55.jsonl","test_operational.json","audit_sample.jsonl","audit_results.json")}
say("PREREG", R["prereg"])
# ---- 1b. labels, exclusion, split membership, gamma sweep
rows = [json.loads(l) for l in open(O/"val_labels.jsonl") if l.strip()]
lab = collections.Counter(r["label"] for r in rows)
use = [r for r in rows if r["label"] in ("SUFFICIENT","INSUFFICIENT") and r["support_score"] is not None]
pos = [r for r in use if r["label"]=="INSUFFICIENT"]; neg = [r for r in use if r["label"]=="SUFFICIENT"]
audited = {c["incoming_query_id"] for c in csv.DictReader(open(ROOT/"validation/l2_independent_audit_300/selected_cases.csv"))}
import experiment as exp
recs = exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))
cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in recs}
split = json.load(open(ROOT/"validation/heldout_baseline_tuning/split.json")); VAL = set(split["validation_clusters"]); TEST = set(split["test_clusters"])
ids = [r["query_id"] for r in rows]
R["labels"] = {"rows": len(rows), "label_counts": dict(lab), "usable": len(use), "n_insufficient": len(pos), "n_sufficient": len(neg),
    "usable_with_null_score": sum(1 for r in rows if r["label"] in ("SUFFICIENT","INSUFFICIENT") and r["support_score"] is None),
    "ids_in_300_audit": len(set(ids) & audited), "audit_300_incoming_ids": len(audited), "ids_in_validation_clusters": sum(1 for q in ids if cid[q] in VAL), "ids_in_test_clusters": sum(1 for q in ids if cid[q] in TEST),
    "distinct_validation_clusters": len({cid[q] for q in ids}), "s3b_build": json.load(open(O/"s3b_build.json"))}
say("LABELS", R["labels"])
tab = []; best = None
for i in range(91):
    g = round(i*0.01, 2)
    tpr = sum(1 for r in pos if r["support_score"] < g)/len(pos); fpr = sum(1 for r in neg if r["support_score"] < g)/len(neg); j = tpr-fpr
    rej = sum(1 for r in use if r["support_score"] < g)/len(use)
    row = {"gamma": g, "tpr_reject_insufficient": round(tpr,4), "fpr_reject_sufficient": round(fpr,4), "youden_j": round(j,4), "reject_rate": round(rej,4), "tpr_raw": tpr, "fpr_raw": fpr, "j_raw": j}
    tab.append(row)
    if best is None or (j, -g, -fpr) > (best["j_raw"], -best["gamma"], -best["fpr_raw"]): best = row
jmax = max(r["j_raw"] for r in tab); ties = [r["gamma"] for r in tab if abs(r["j_raw"]-jmax) < 1e-12]
auc = sum((1 if a["support_score"] > b["support_score"] else 0.5 if a["support_score"] == b["support_score"] else 0) for a in neg for b in pos)/(len(neg)*len(pos))
auc_rej = sum((1 if -b["support_score"] > -a["support_score"] else 0.5 if a["support_score"] == b["support_score"] else 0) for a in neg for b in pos)/(len(neg)*len(pos))
rec = json.load(open(O/"gamma_selection.json"))
R["gamma"] = {"grid_size": len(tab), "selected": {k: best[k] for k in ("gamma","tpr_reject_insufficient","fpr_reject_sufficient","youden_j","reject_rate")}, "gammas_tied_at_max_J": ties, "auc_sufficient_scores_higher": round(auc,4), "auc_rejection_score_insufficient_positive": round(auc_rej,4),
    "recorded": {"selected_gamma": rec["selected_gamma"], "selected": rec["selected"], "auc": rec["auc"]}, "sweep_identical_to_record": all(abs(a["youden_j"]-b["youden_j"])<1e-9 and a["gamma"]==b["gamma"] for a,b in zip(tab, rec["sweep"]))}
say("GAMMA", R["gamma"])
with open(OUT/"gamma_recompute.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["gamma","tpr_reject_insufficient","fpr_reject_sufficient","youden_j","reject_rate"]); w.writeheader()
    for r in tab: w.writerow({k: r[k] for k in w.fieldnames})
# ---- 2/3. request accounting from test_rows.pkl
allrows = pickle.load(open(O/"test_rows.pkl","rb"))
FC, LV = allrows["FreshCache"], allrows["L2Verify_g0.55"]
scores = [json.loads(l) for l in open(O/"test_scores_L2Verify_g0.55.jsonl")]
def account(rows):
    N = len(rows); tiers = collections.Counter(d["tier"] for d in rows.values())
    search = tiers["miss"] + tiers["miss_rejected"] + tiers["REAL_TIME"]
    fetch = 0
    for d in rows.values():
        if d["tier"] == "L1": continue
        fetch += len({x for x, tc in d["ev"] if tc == d["t"]})
    l2 = tiers["L2"]; gen = search + l2
    return {"N": N, "tiers": dict(tiers), "search_calls": search, "search_saved_pct": round(100*(1-search/N),4), "l2_hits": l2, "l1_hits": tiers["L1"], "fetches": fetch, "fetches_per_1k": round(1000*fetch/N,2), "generations": gen, "generations_per_1k": round(1000*gen/N,2),
            "verdicts": dict(collections.Counter(d.get("verified") for d in rows.values()))}
acc = {"FreshCache": account(FC), "L2Verify_g0.55": account(LV)}
vc = collections.Counter(s["verdict"] for s in scores)
reach = sum(1 for d in LV.values() if d["tier"] in ("L2","miss_rejected"))
acc["scores_file"] = {"rows": len(scores), "verdicts": dict(vc), "scored": vc["ACCEPT"]+vc["REJECT"], "unique_query_ids": len({s["query_id"] for s in scores}), "L2_hits_reaching_verification_point_in_rows": reach, "every_reaching_hit_has_score_row": reach == len(scores)}
trans = collections.Counter((FC[q]["tier"], LV[q]["tier"]) for q in FC)
acc["tier_transition_FC_to_LV"] = {f"{a}->{b}": n for (a,b), n in sorted(trans.items())}
to = json.load(open(O/"test_operational.json"))["results"]
acc["record"] = {k: {kk: to[k][kk] for kk in ("search_calls","search_saved_pct","l1_hits","l2_hits","l3_hits","fetches","generations","fetches_per_1k","generations_per_1k","l2_accepted","l2_rejected","l2_unscored")} for k in ("FreshCache","L2Verify_g0.55")}
acc["savings_arithmetic"] = {"delta_pp": round(acc["FreshCache"]["search_saved_pct"]-acc["L2Verify_g0.55"]["search_saved_pct"],4), "extra_search_calls": acc["L2Verify_g0.55"]["search_calls"]-acc["FreshCache"]["search_calls"], "rejections": vc["REJECT"], "rejections_over_N_pp": round(100*vc["REJECT"]/len(FC),4),
    "FC_L2_hits_that_become_L2_in_LV": trans[("L2","L2")], "FC_miss_that_become_L2_in_LV": trans[("miss","L2")], "FC_L2_that_become_rejected": trans[("L2","miss_rejected")], "FC_miss_that_become_rejected": trans[("miss","miss_rejected")]}
R["accounting"] = acc; say("ACCOUNTING", json.dumps(acc, indent=1))
with open(OUT/"request_accounting.csv","w",newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm","quantity","recomputed","record"])
    for arm in ("FreshCache","L2Verify_g0.55"):
        a = acc[arm]; r = acc["record"][arm]
        for k in ("N","search_calls","search_saved_pct","l1_hits","l2_hits","fetches","fetches_per_1k","generations","generations_per_1k"): w.writerow([arm,k,a.get(k,""),r.get(k,"") if k!="N" else 21848])
        for t,n in a["tiers"].items(): w.writerow([arm,f"tier:{t}",n,""])
    for k,n in vc.items(): w.writerow(["L2Verify_g0.55",f"verdict:{k}",n,{"ACCEPT":to["L2Verify_g0.55"]["l2_accepted"],"REJECT":to["L2Verify_g0.55"]["l2_rejected"],"ACCEPT_UNSCORED":to["L2Verify_g0.55"]["l2_unscored"]}[k]])
    for k,n in acc["tier_transition_FC_to_LV"].items(): w.writerow(["FC->LV",k,n,""])
# ---- 4. 400 audit
import s6_answer_audit as s6
samp = s6.jl(s6.OUT/"audit_sample.jsonl"); ref = json.load(open(s6.A2/"sample.json"))["requests"]
R["audit400"] = {"n": len(samp), "ids_identical_in_order_to_answer_audit_k1_16": [d["query_id"] for d in samp] == [d["query_id"] for d in ref]}
ans = s6._ans(); jd = {}
for name in s6.JUDGES:
    jd[name] = {tuple(r["sig"]): r["label"] for r in s6.jl(s6.A2/f"judge_{name}.jsonl")}
    for r in s6.jl(s6.OUT/f"audit_judge_{name}.jsonl"): jd[name][tuple(r["sig"])] = r["label"]
def outc(d, key):
    v = ans.get(tuple(key))
    if v is None: return None
    if any(p in v.strip().lower() for p in s6.ABST): return "ABSTAIN"
    sig = (s6.h(d["query"]), s6.h(d["gold"]), s6.h(v)); a, b = jd["llama8b"].get(sig), jd["qwen7b"].get(sig)
    if a is None or b is None: return None
    return "CORRECT" if (a=="CORRECT" and b=="CORRECT") else "WRONG"
sl = {(r["query_id"], r["arm"]): r["label"] for r in s6.jl(O/"audit_suff_llama8b.jsonl")}; sq = {(r["query_id"], r["arm"]): r["label"] for r in s6.jl(O/"audit_suff_qwen7b.jsonl")}
out = []; missing = 0
for d in samp:
    fr, fc, lv = outc(d, d["fresh_key"]), outc(d, d["fc_key"]), outc(d, d["lv_key"])
    if None in (fr, fc, lv): missing += 1; continue
    def agreed(arm):
        a, b = sl.get((d["query_id"], arm)), sq.get((d["query_id"], arm)); return a if a == b else "JUDGE_DISAGREEMENT"
    out.append({"query_id": d["query_id"], "fc_tier": d["fc_tier"], "lv_tier": d["lv_tier"], "lv_verdict": d["lv_verdict"], "serving_changed": 1-d["serving_unchanged"], "fresh": fr, "fc": fc, "lv": lv,
                "fc_wai": int(fr=="CORRECT" and fc=="WRONG"), "lv_wai": int(fr=="CORRECT" and lv=="WRONG"), "fc_suff_llama8b": sl.get((d["query_id"],"fc")), "fc_suff_qwen7b": sq.get((d["query_id"],"fc")), "lv_suff_llama8b": sl.get((d["query_id"],"lv")), "lv_suff_qwen7b": sq.get((d["query_id"],"lv")), "fc_suff_agreed": agreed("fc"), "lv_suff_agreed": agreed("lv")})
def mcn(b, c):
    n = b+c
    if n == 0: return 1.0
    k = min(b,c); return float(min(Fraction(1), Fraction(sum(math.comb(n,i) for i in range(k+1))*2, 2**n)))
a4 = R["audit400"]; a4["rows_with_all_three_labels"] = len(out); a4["rows_dropped_missing_label"] = missing
a4["fresh_correct"] = sum(1 for r in out if r["fresh"]=="CORRECT")
for arm in ("fc","lv"):
    c = collections.Counter(r[arm+"_suff_agreed"] for r in out)
    a4[arm] = {"correct": sum(1 for r in out if r[arm]=="CORRECT"), "wai": sum(r[arm+"_wai"] for r in out), "wai_ids": [r["query_id"] for r in out if r[arm+"_wai"]], "suff_SUFFICIENT": c["SUFFICIENT"], "suff_INSUFFICIENT": c["INSUFFICIENT"], "suff_JUDGE_DISAGREEMENT": c["JUDGE_DISAGREEMENT"], "suff_other": {k: v for k, v in c.items() if k not in ("SUFFICIENT","INSUFFICIENT","JUDGE_DISAGREEMENT")}}
a4["paired_wai"] = {"both": sum(1 for r in out if r["fc_wai"] and r["lv_wai"]), "fc_only": sum(1 for r in out if r["fc_wai"] and not r["lv_wai"]), "lv_only": sum(1 for r in out if r["lv_wai"] and not r["fc_wai"])}; a4["paired_wai"]["exact_mcnemar_p"] = mcn(a4["paired_wai"]["fc_only"], a4["paired_wai"]["lv_only"])
cw = sum(1 for r in out if r["fc"]=="CORRECT" and r["lv"]!="CORRECT"); wc = sum(1 for r in out if r["fc"]!="CORRECT" and r["lv"]=="CORRECT"); a4["paired_correctness"] = {"fc_only": cw, "lv_only": wc, "exact_mcnemar_p": mcn(cw, wc)}
tr = collections.Counter((r["fc_suff_agreed"], r["lv_suff_agreed"]) for r in out)
a4["sufficiency_transition_table"] = {f"{a}->{b}": n for (a,b), n in sorted(tr.items())}
a_only = sum(1 for r in out if r["fc_suff_agreed"]=="SUFFICIENT" and r["lv_suff_agreed"]!="SUFFICIENT"); b_only = sum(1 for r in out if r["lv_suff_agreed"]=="SUFFICIENT" and r["fc_suff_agreed"]!="SUFFICIENT")
a4["paired_sufficiency_binary_rule"] = {"rule": "SUFFICIENT (both judges) vs everything else (INSUFFICIENT or JUDGE_DISAGREEMENT); all 400 rows enter", "fc_sufficient_only": a_only, "lv_sufficient_only": b_only, "exact_mcnemar_p": mcn(a_only, b_only),
    "of_which_disagreement_involved": {"fc_SUFF->lv_DISAGREE": tr[("SUFFICIENT","JUDGE_DISAGREEMENT")], "fc_DISAGREE->lv_SUFF": tr[("JUDGE_DISAGREEMENT","SUFFICIENT")], "fc_SUFF->lv_INSUFF": tr[("SUFFICIENT","INSUFFICIENT")], "fc_INSUFF->lv_SUFF": tr[("INSUFFICIENT","SUFFICIENT")]}}
ag = [r for r in out if r["fc_suff_agreed"] in ("SUFFICIENT","INSUFFICIENT") and r["lv_suff_agreed"] in ("SUFFICIENT","INSUFFICIENT")]
a4["paired_sufficiency_agreed_both_arms_only"] = {"n": len(ag), "fc_only": sum(1 for r in ag if r["fc_suff_agreed"]=="SUFFICIENT" and r["lv_suff_agreed"]=="INSUFFICIENT"), "lv_only": sum(1 for r in ag if r["lv_suff_agreed"]=="SUFFICIENT" and r["fc_suff_agreed"]=="INSUFFICIENT")}
a4["paired_sufficiency_agreed_both_arms_only"]["exact_mcnemar_p"] = mcn(a4["paired_sufficiency_agreed_both_arms_only"]["fc_only"], a4["paired_sufficiency_agreed_both_arms_only"]["lv_only"])
a4["serving_changed"] = sum(r["serving_changed"] for r in out); a4["lv_verdicts_in_400"] = dict(collections.Counter(r["lv_verdict"] for r in out)); a4["record"] = json.load(open(O/"audit_results.json"))
say("AUDIT400", json.dumps({k: v for k, v in a4.items() if k != "record"}, indent=1))
with open(OUT/"paired_400_transitions.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
json.dump(R, open(OUT/"R16_results.json","w"), indent=1, default=str)
