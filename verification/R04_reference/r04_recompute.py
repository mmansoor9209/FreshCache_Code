"""R04 / V35-N03 recompute from existing records. Read-only; writes only into this folder."""
import json, csv, re, hashlib, pathlib, sys, os, collections, datetime
ROOT = pathlib.Path("<PROJECT_ROOT>"); OUT = pathlib.Path(__file__).parent
A2 = ROOT/"validation/heldout_baseline_tuning/answer_audit_k1_16"; GV = ROOT/"validation/gold_verification_400"; RV = GV/"revised_3b"
HS = ROOT/"validation/quality_aware_baseline/heldout_sensitivity"; SRE = ROOT/"validation_V3/strong_reference_evaluation/02_reference_verification"
sys.path.insert(0, str(ROOT/"v9")); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import mixed_age_v2 as ma
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def h16(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
ABST = ["i don't know","i do not know","cannot be determined","not enough information","insufficient","unable to determine","no information"]
YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
R = {"utc": datetime.datetime.utcnow().isoformat()+"Z"}
sample = json.load(open(A2/"sample.json"))["requests"]; ids = [d["query_id"] for d in sample]
R["ids"] = {"n": len(ids), "unique": len(set(ids)), "duplicates": [k for k, v in collections.Counter(ids).items() if v > 1]}
# ---- 1. classification recompute from the gold-verification records ----
gv = {r["query_id"]: r for r in csv.DictReader(open(GV/"combined_request_status.csv"))}
orig_units = {r["unit_id"]: r for r in csv.DictReader(open(GV/"gold_validity_at_t.csv"))}
rev_units = {r["unit_id"]: r for r in csv.DictReader(open(RV/"r_gold_validity.csv"))}
unit_of = {}
for u, r in orig_units.items():
    for q in r["request_ids"].split("|"): unit_of.setdefault(q, []).append(u)
VMAP = {"VALID_AT_T":"SUPPORTED","SUPERSEDED_AT_T":"CONTRADICTED","INDETERMINATE_AT_T":"UNVERIFIABLE","JUDGE_DISAGREEMENT":"UNVERIFIABLE"}
diag = {r["query_id"]: r for r in csv.DictReader(open(SRE/"reference_and_diagnosis.csv"))}
recl = {}; mism = 0; cov = 0; unit_round_mismatch = 0; multi_unit = 0
for d in sample:
    q = d["query_id"]; g = gv.get(q)
    if g is None: recl[q] = "NO_GV_RECORD"; continue
    cov += 1
    base = VMAP.get(g["validity"], "UNVERIFIABLE"); vol = g["volatility"]; qy = bool(YEAR.search(d["query"]))
    lab = "TEMPORALLY_AMBIGUOUS" if (base == "UNVERIFIABLE" and vol == "VOLATILE" and not qy) else base
    recl[q] = lab
    if diag[q]["reference_status_final"] != lab: mism += 1
    if g["snapshot_round"] != ma.version_at(float(d["t"])): unit_round_mismatch += 1
    if len(unit_of.get(q, [])) > 1: multi_unit += 1
R["classification"] = {"gv_coverage": cov, "recomputed": dict(collections.Counter(recl.values())), "recorded_in_reference_and_diagnosis": dict(collections.Counter(r["reference_status_final"] for r in diag.values())),
    "mismatches_vs_recorded": mism, "raw_stage3b_pair_status_on_400": dict(collections.Counter(gv[q]["validity"] for q in ids if q in gv)),
    "ambiguity_rule": "UNVERIFIABLE (INDETERMINATE or JUDGE_DISAGREEMENT) relabelled TEMPORALLY_AMBIGUOUS when volatility==VOLATILE and the question has no 4-digit year (verify_and_diagnose.py:128-129,153-156)",
    "unit_round_equals_version_at_request_t": unit_round_mismatch == 0, "requests_mapped_to_more_than_one_unit": multi_unit}
# ---- 2. sets ----
fid = {r["query_id"]: r["final_label"] for r in csv.DictReader(open(GV/"paraphrase_fidelity.csv"))}
runit = {}
for u, r in rev_units.items():
    for q in r["request_ids"].split("|"): runit[q] = r
EV = {q for q in runit if fid.get(q, "BASE_QUERY") in ("SAME","BASE_QUERY") and runit[q]["status_majority"] == "VALID_AT_T"}
held = set(ids); screened = EV & held
supported = {q for q in ids if recl.get(q) == "SUPPORTED"}
cw = json.load(open(ROOT/"Reframe_Paper_Pro/artifacts/crosswalk_57_107.json"))
R["sets"] = {"screened_107": len(screened), "supported_57": len(supported), "intersection": len(screened & supported), "supported_only": len(supported - screened), "screened_only": len(screened - supported),
    "matches_crosswalk_file": (sorted(screened) == cw["screened_107"] and sorted(supported) == cw["supported_57"] and sorted(screened & supported) == cw["intersection"]),
    "fidelity_labels_on_400": dict(collections.Counter(fid.get(q, "BASE_QUERY") for q in ids)), "revised_majority_on_400": dict(collections.Counter(runit[q]["status_majority"] for q in ids if q in runit)),
    "revised_pair_on_400": dict(collections.Counter(runit[q]["status_pair"] for q in ids if q in runit)),
    "why_supported_not_screened": {f"{k[0]}|{k[1]}": v for k, v in collections.Counter((fid.get(q,"BASE_QUERY"), runit[q]["status_majority"]) for q in supported - screened).items()},
    "why_screened_not_supported": dict(collections.Counter(gv[q]["validity"] for q in screened - supported))}
# ---- 3. evidence / judges / times for the 400 ----
tasks = {t["unit_id"]: t for t in jl(GV/"validity_tasks.jsonl")}; rtasks = {t["unit_id"]: t for t in jl(RV/"r_validity_tasks.jsonl")}
L = {r["unit_id"]: r for r in jl(GV/"validity_labels_llama8b.jsonl")}; Q = {r["unit_id"]: r for r in jl(GV/"validity_labels_qwen7b.jsonl")}
rL = {r["unit_id"]: r["status"] for r in jl(RV/"r_validity_llama8b.jsonl")}; rQ = {r["unit_id"]: r["status"] for r in jl(RV/"r_validity_qwen7b.jsonl")}; rM = {r["unit_id"]: r["status"] for r in jl(RV/"r_validity_mistral7b.jsonl")}
rounds = {}
for line in open(ROOT/"v13_corrected/corrected_round_table.jsonl"):
    d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
ev_rows = []; fetched = collections.defaultdict(list); missing_snap = 0
for d in sample:
    q = d["query_id"]; u = unit_of.get(q, [None])[0]; g = gv.get(q, {}); t = tasks.get(u, {}); rt = rtasks.get(u, {})
    rd = g.get("snapshot_round", "")
    fa = []
    for uh, tv in d.get("fresh_versions", []):
        e = rounds.get(uh, {}).get(ma.version_at(tv))
        if e: fa.append(e.get("fetched_at", "")); fetched[ma.version_at(tv)].append(e.get("fetched_at", ""))
        else: missing_snap += 1
    lu = L.get(u, {}); qu = Q.get(u, {})
    ev_rows.append({"query_id": q, "unit_id": u, "class": recl.get(q), "gold": d["gold"], "gold_year": g.get("gold_year",""), "volatility": g.get("volatility",""), "request_t_s": d["t"], "version_at_t": ma.version_at(float(d["t"])), "unit_snapshot_round": rd,
        "orig_n_pages": t.get("n_pages_kept",""), "orig_evidence_chars": t.get("evidence_chars",""), "orig_llama8b": lu.get("status",""), "orig_llama8b_raw": lu.get("raw",""), "orig_qwen7b": qu.get("status",""), "orig_qwen7b_raw": qu.get("raw",""), "orig_final": orig_units.get(u, {}).get("final_status",""),
        "rev_mode": rt.get("mode",""), "rev_evidence_chars": rt.get("evidence_chars",""), "rev_llama8b": rL.get(u,""), "rev_qwen7b": rQ.get(u,""), "rev_mistral7b": rM.get(u,""), "rev_majority": rev_units.get(u, {}).get("status_majority",""), "fidelity": fid.get(q, "BASE_QUERY"),
        "in_107": int(q in screened), "in_57": int(q in supported), "evidence_pages_fetched_at": "|".join(fa)})
R["evidence"] = {"requests_with_unit": sum(1 for r in ev_rows if r["unit_id"]), "orig_task_present": sum(1 for r in ev_rows if r["orig_n_pages"] != ""), "orig_labels_present_both": sum(1 for r in ev_rows if r["orig_llama8b"] and r["orig_qwen7b"]),
    "supported57_all_both_VALID_raw": all(r["orig_llama8b"] == "VALID_AT_T" and r["orig_qwen7b"] == "VALID_AT_T" for r in ev_rows if r["in_57"]),
    "supported57_orig_evidence_chars": sorted(int(r["orig_evidence_chars"]) for r in ev_rows if r["in_57"] and r["orig_evidence_chars"] != "")[:5],
    "evidence_fetched_at_range_by_round": {k: (min(v), max(v)) for k, v in fetched.items()}, "fresh_versions_without_snapshot_entry": missing_snap,
    "rounds_on_400": dict(collections.Counter(r["unit_snapshot_round"] for r in ev_rows)), "orig_prompt_evidence_cap": "2000 chars/page, 2800 budget (stage3b_validity.py:29-31)", "revised_prompt": "r_validity_tasks: per-unit mode + expanded evidence (see r1_build.py)"}
with open(OUT/"heldout_400_reference_records.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ev_rows[0])); w.writeheader(); w.writerows(ev_rows)
# ---- 4. Table 15 recompute (published scoring path) ----
new = {d["query_id"]: d for d in jl(HS/"h_sample.jsonl")}
ans = {tuple(r["key"]): r["answer"] for r in jl(A2/"answers.jsonl")}
for r in jl(HS/"h_answers.jsonl"): ans[tuple(r["key"])] = r["answer"]
j1 = {tuple(r["sig"]): r["label"] for r in jl(A2/"judge_llama8b.jsonl")}; j2 = {tuple(r["sig"]): r["label"] for r in jl(A2/"judge_qwen7b.jsonl")}
for r in jl(HS/"h_judge_llama8b.jsonl"): j1[tuple(r["sig"])] = r["label"]
for r in jl(HS/"h_judge_qwen7b.jsonl"): j2[tuple(r["sig"])] = r["label"]
def outc(d, key):
    a = ans.get(tuple(key))
    if a is None: return None
    if any(p in a.strip().lower() for p in ABST): return "ABSTAIN"
    s = (h16(d["query"]), h16(d["gold"]), h16(a)); l1, l2 = j1.get(s), j2.get(s)
    if l1 is None or l2 is None: return None
    return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"
rows = {}; dropped = []
for d in sample:
    r = {"fresh": outc(d, d["fresh_key"]), "fc": outc(d, d["fc_key"]), "sttl": outc(d, new[d["query_id"]]["new_sttl_key"])}
    if any(v is None for v in r.values()): dropped.append(d["query_id"]); continue
    rows[d["query_id"]] = r
def score(sub_ids):
    sub = [rows[q] for q in sub_ids]; n = len(sub); fcn = sum(1 for r in sub if r["fresh"] == "CORRECT"); o = {"N": n, "fresh_correct": fcn}
    for a in ("fc","sttl"):
        c = sum(1 for r in sub if r[a] == "CORRECT"); w = sum(1 for r in sub if r["fresh"] == "CORRECT" and r[a] == "WRONG"); ab = sum(1 for r in sub if r[a] == "ABSTAIN")
        o[a] = {"correct": c, "acc_pct": round(100*c/n, 2), "wai": w, "abstain": ab}
    return o
T15 = {"all_400": score(rows), "screened_107": score(screened), "supported_57": score(supported), "intersection_41": score(screened & supported)}
R["table15"] = {"rows_scored": len(rows), "dropped_missing_label": dropped, "recomputed": T15,
    "manuscript": {"all_400": (400,81,"18.75",7,"16.00",25), "screened_107": (107,49,"42.06",4,"35.51",15), "supported_57": (57,40,"64.91",3,"49.12",13), "intersection_41": (41,30,"65.85",3,"53.66",9)},
    "rules": "abstention pattern -> ABSTAIN (excluded from correct, not a WAI event); CORRECT only if both judges say CORRECT; WAI = fresh CORRECT and policy WRONG (crosswalk_57_107.py:24-30, 49-52)"}
json.dump(R, open(OUT/"R04_results.json","w"), indent=1, default=str); print(json.dumps(R, indent=1, default=str))
