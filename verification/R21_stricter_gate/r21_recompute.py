"""R21 audit: recompute the stricter-gate study's headline numbers from row-level records.
Read-only over validation_V3/l1_precision_gate/. Writes R21_results.json next to this file."""
import csv, json, math, collections, hashlib, pathlib, datetime
S = pathlib.Path("<PROJECT_ROOT>/validation_V3/l1_precision_gate")
OUT = pathlib.Path(__file__).parent
ARMS = ("published", "strict090", "precision")
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(100*(c-h), 2), round(100*(c+h), 2)]
R = {"utc": datetime.datetime.utcnow().isoformat()+"Z", "inputs": {}}
for f in ("heldout_l1_hits_judged.csv","jury_results.jsonl","heldout_per_request.csv","answer_audit.csv",
          "heldout_audit_sample.json","frozen_l1_precision_gate.json","l1_precision_prereg.json","l1_guards.py","l1_gate.py"):
    R["inputs"][f] = hashlib.sha256((S/f).read_bytes()).hexdigest()
# 1. mismatch from item-level jury verdicts
J = list(csv.DictReader(open(S/"heldout_l1_hits_judged.csv")))
jr = {r["pair_id"]: r for r in map(json.loads, open(S/"jury_results.jsonl"))}
R["jury"] = {"distinct_pairs": len(jr), "pairs_missing_a_label": sum(1 for r in jr.values() if len(r["labels"]) < 4),
             "verdicts": dict(collections.Counter(r["verdict"] for r in jr.values()))}
mm = {}
for a in ARMS:
    P = [r for r in J if r["arm"] == a]; n = len(P)
    c = collections.Counter(r["equivalence"] for r in P); d, t, s = c["DIFFERENT"], c["JURY_TIE"], c["SAME"]
    mm[a] = {"l1_hits": n, "unique_request_ids": len({r["request_id"] for r in P}),
             "unique_pairs": len({(r["incoming_query"], r["cached_query"]) for r in P}),
             "SAME": s, "DIFFERENT": d, "JURY_TIE": t,
             "ties_to_DIFFERENT_pct": round(100*(d+t)/n, 2), "ties_to_DIFFERENT_wilson95": wilson(d+t, n),
             "ties_excluded_pct": round(100*d/(d+s), 2), "ties_to_SAME_pct": round(100*d/n, 2),
             "min_similarity": min(float(r["similarity"]) for r in P), "min_jaccard": min(float(r["jaccard"]) for r in P),
             "entity_match_all_true": all(r["entity_match"] == "True" for r in P),
             "guard_all_pass": all(r["numeric_guard_pass"] == r["negation_guard_pass"] == r["comparative_guard_pass"] == "True" for r in P),
             "cross_cluster_hits": sum(1 for r in P if r["cross_cluster"] == "1")}
R["mismatch"] = mm
ids = {a: {r["request_id"] for r in J if r["arm"] == a} for a in ARMS}
R["hit_overlap"] = {"strict090_and_published": len(ids["strict090"] & ids["published"]),
                    "precision_and_published": len(ids["precision"] & ids["published"]),
                    "precision_and_strict090": len(ids["precision"] & ids["strict090"])}
lost = [r for r in J if r["arm"] == "strict090" and r["request_id"] not in ids["precision"]]
R["strict090_hits_not_in_precision"] = {"n": len(lost), "guard_pattern_num_neg_comp": {
    str(k): v for k, v in collections.Counter((r["numeric_guard_pass"], r["negation_guard_pass"], r["comparative_guard_pass"]) for r in lost).items()}}
lost2 = [r for r in J if r["arm"] == "published" and r["request_id"] not in ids["strict090"]]
R["published_hits_not_in_strict090"] = {"n": len(lost2), "with_similarity_below_0.90": sum(1 for r in lost2 if float(r["similarity"]) < 0.9)}
# 2. hits / savings / displacement from per-request tiers
Q = list(csv.DictReader(open(S/"heldout_per_request.csv"))); N = len(Q)
cost = {}
for a in ARMS:
    c = collections.Counter(r["tier_"+a] for r in Q)
    nonhit = c["miss"] + c["REAL_TIME"]
    cost[a] = {"tiers": dict(c), "search_calls_miss_plus_realtime": nonhit,
               "search_saved_pct_full_precision": 100*(1-nonhit/N), "search_saved_pct_4dp": round(100*(1-nonhit/N), 4)}
R["n_requests"] = N; R["cost"] = cost
R["savings_delta_pp_published_minus_precision"] = round(cost["published"]["search_saved_pct_full_precision"]-cost["precision"]["search_saved_pct_full_precision"], 4)
for a in ("strict090", "precision"):
    c = collections.Counter((r["tier_published"], r["tier_"+a]) for r in Q if r["tier_published"] == "L1" or r["tier_"+a] == "L1")
    R["displacement_published_to_"+a] = {f"{k[0]}->{k[1]}": v for k, v in c.items()}
# 3. WAI from the 400-request audit
A = list(csv.DictReader(open(S/"answer_audit.csv")))
w = {}
for a in ARMS:
    wai = [x["query_id"] for x in A if x["fresh"] == "CORRECT" and x[a] == "WRONG"]
    rev = [x["query_id"] for x in A if x["fresh"] == "WRONG" and x[a] == "CORRECT"]
    w[a] = {"n": len(A), "wai": len(wai), "wai_ids": wai, "reverse": len(rev), "reverse_ids": rev,
            "labels": dict(collections.Counter(x[a] for x in A)), "tier_composition": dict(collections.Counter(x[a+"_tier"] for x in A)),
            "tier_of_wai_events": dict(collections.Counter(x[a+"_tier"] for x in A if x["query_id"] in wai))}
R["wai"] = w
R["wai_ids_identical_across_arms"] = w["published"]["wai_ids"] == w["strict090"]["wai_ids"] == w["precision"]["wai_ids"]
R["audit_rows_with_any_label_difference_across_arms"] = [
    {k: x[k] for k in ("query_id","fresh","published","published_tier","strict090","strict090_tier","precision","precision_tier")}
    for x in A if not (x["published"] == x["strict090"] == x["precision"])]
smp = json.load(open(S/"heldout_audit_sample.json"))
ref = json.load(open("<PROJECT_ROOT>/validation/heldout_baseline_tuning/answer_audit_k1_16/sample.json"))
rq = ref.get("requests") if isinstance(ref, dict) else ref
def qid(r): return r["query_id"] if isinstance(r, dict) else r
R["audit_sample"] = {"n": len(smp["requests"]), "sample_gate_recorded": smp["sample_gate"],
                     "ids_identical_in_order_to_answer_audit_k1_16": [qid(r) for r in rq] == [r["query_id"] for r in smp["requests"]] if rq else "reference sample.json has no request list",
                     "answer_key_equality_(pub==strict,pub==prec)": {str(k): v for k, v in collections.Counter(
                         (r["published_key"] == r["strict090_key"], r["published_key"] == r["precision_key"]) for r in smp["requests"]).items()}}
# 4. earlier experiment cross-run jury comparison
old = {}
for j in ("llama3b","llama8b","mistral7b","qwen7b"):
    for l in open(S.parent/"final_three_issues"/"02_l1_strict_gate"/f"equiv_{j}.jsonl"):
        d = json.loads(l); old.setdefault(d.get("pair_id") or d.get("key"), {})[j] = d.get("label") or d.get("verdict") or d.get("answer")
shared = set(old) & set(jr)
R["earlier_experiment_02_l1_strict_gate"] = {"pairs": len(old), "shared_pair_ids": len(shared),
    "juror_label_changes_on_shared_pairs": sum(1 for p in shared for j in jr[p]["labels"] if old[p].get(j) != jr[p]["labels"][j]),
    "juror_labels_compared": 4*len(shared)}
json.dump(R, open(OUT/"R21_results.json", "w"), indent=2)
print(json.dumps(R, indent=1))
