"""Build GPU inputs from the reported CPU outputs, removing everything already present in the read-only pools.
  gen   : every generation key needed by the reported answer audits, budget audit, stricter audits, tier paths and verifier audit
  judge : 2-judge items, 4-juror pairs, 3-judge answer tasks, tier sufficiency tasks and verifier sufficiency tasks (after gen)"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, hashlib, random, pathlib, csv; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import answer_pools as P, engine_loader as C
R, W = C.ROOT, C.W; I = W/"gpu/in"; STAGE = sys.argv[1]
def jl(p): return P.jl(p)
def wj(name, rows):
    with open(I/name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), flush=True)
AK = json.load(open(W/"results/audit_keys_reported.json")); BK = json.load(open(W/"results/budget_keys_reported.json"))
SC = json.load(open(W/"results/stricter_cpu_reported.json")); TR = json.load(open(W/"results/tier_cpu_reported.json"))
VA = json.load(open((OUTPUT_ROOT / "l1_precision/validation_audit_sample.json")))["requests"]
VF = json.load(open(W/"results/stricter_validation_fresh_reported.json"))   # reported fresh keys of the validation audit
def gen_tasks():
    T = {}
    def add(t): T.setdefault(tuple(t["key"][1:]), t)
    for t in json.load(open(W/"results/audit_gen_tasks_reported.json")): add(t)
    for t in json.load(open(W/"results/budget_gen_tasks_reported.json")): add(t)
    for part in ("validation", "heldout"):
        for cell in SC[part].values():
            for v in cell["audit"].values(): add({"key": v["key"], "question": v["question"], "context": v["context"]})
    for t in VF.values(): add(t)
    for x in TR["l1_reuse"] + TR["l2_reuse"]:
        for v in x["paths"].values(): add({"key": v["key"], "question": v["question"], "context": v["context"]})
    for t in json.load(open(W/"verifier/audit/gen_tasks.json")): add(t)
    return T
if STAGE == "gen":
    A = P.answers(); T = gen_tasks()
    wj("gen_tasks.jsonl", [t for k, t in T.items() if k not in A])
    json.dump({"distinct_generation_keys": len(T), "reused_from_pools": sum(1 for k in T if k in A)}, open(I/"gen_manifest.json", "w"), indent=1)
    sys.exit()
A = P.AnswerLookup(extra=[W/"gpu/out/answers_llama3b.jsonl"]); J2 = P.judge2(); JY = P.jury(); J3 = P.judge3(); S3 = P.suff3()
h = P.h16; items, pairs, t3, s3, miss = {}, {}, {}, {}, []
def ans(key):
    a = A.get(key)   # exact (role, ctx, q) first, then content
    if a is None: miss.append(key)
    return a
def j2(q, gold, key):
    a = ans(key)
    if a is None or not gold: return
    sig = (h(q), h(gold), h(a))
    if sig not in J2["llama8b"] or sig not in J2["qwen7b"]: items.setdefault(sig, {"sig": list(sig), "q": q, "gold": gold, "cand": a})
def pair(pid, inc, cached):
    have = JY.get(pid, {})
    if len([j for j in ("llama3b", "llama8b", "qwen7b", "mistral7b") if j in have]) < 4: pairs.setdefault(pid, {"pair_id": pid, "incoming": inc, "cached": cached})
# --- held-out 400 audit (Table 6 blocks, subsets, model-based study) and primary 400 audit ---
HS = {d["query_id"]: d for d in json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]}
for q, rec in AK["keys"]["heldout"].items():
    for arm, (k, tier) in rec.items(): j2(HS[q]["query"], HS[q]["gold"], k)
PS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "mixed_age_audit/mixed_age_sample.jsonl"))}
for q, rec in AK["keys"]["primary"].items():
    for arm, (k, tier) in rec.items(): j2(PS[q]["query"], PS[q]["gold"], k)
# --- budget audit A-D ---
for r in BK["records"]:
    for n in ("fresh", "A_original", "B_mult_only", "C_budget_only", "D_full"): j2(r["query"], r["gold"], r[f"{n}_key"])
# --- stricter: validation audit (all cells), held-out audit arms, jury on realised hit pairs ---
VAd = {d["query_id"]: d for d in VA}; HA = {d["query_id"]: d for d in json.load(open((OUTPUT_ROOT / "l1_precision/heldout_audit_sample.json")))["requests"]}
for q, t in VF.items(): j2(VAd[q]["query"], VAd[q]["gold"], t["key"])
for part, aud in (("validation", VAd), ("heldout", HA)):
    for cell in SC[part].values():
        for q, v in cell["audit"].items(): j2(aud[q]["query"], aud[q]["gold"], v["key"])
        for x in cell["hits"]: pair(x["pair_id"], x["incoming"], x["cached"])
# --- tier diagnostic (every reported held-out L1 hit): jury, 2-judge on A and C (hit-level harm), 3-judge on A/B/C ---
REFV = {r["query_id"]: r for r in csv.DictReader(open((OUTPUT_ROOT / "reference_validation/results/reference_validation_results.csv"), encoding="utf-8"))}
TQ = {r["query_id"]: r["target_date"] for r in jl((OUTPUT_ROOT / "tier_quality/inputs/path_rows.jsonl"))}
def tier_target(qid, t):   # tier study rule (tier_quality_judging_tasks.target_date), computed once per request
    if qid in TQ: return TQ[qid]
    import os; os.chdir(R); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "")); sys.path.insert(0, str(REPO_ROOT / "scripts/data")); sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation")); sys.path.insert(0, str(REPO_ROOT / "freshcache/method")); sys.path.insert(0, str(REPO_ROOT / "freshcache/simulation"))
    import tiered_cache_model as exp, page_version_replay as ma
    global OWN
    if "OWN" not in globals(): OWN = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))}
    ds = []
    for uh in dict.fromkeys(OWN[qid]):
        p = (DATA_ROOT / "snapshots")/ma.version_at(float(t))/f"{uh}.json"
        if p.exists():
            s = json.load(open(p, encoding="utf-8"))
            if len(" ".join((s.get("extracted_text") or "").split())) >= 200: ds.append((s.get("fetched_at") or "")[:10])
    return max(ds) if ds else "unknown"
def add3(qid, question, target, reference, a):
    tid = P.tid(qid, a)
    if len([j for j in ("llama8b", "qwen7b", "mistral7b") if j in J3.get(tid, {})]) < 3:
        t3.setdefault(tid, {"task_id": tid, "query_id": qid, "question": question, "target": target, "reference": reference, "candidate": a})
    return tid
tier_rows = []
for task, qf in (("l1_reuse", "incoming_query"), ("l2_reuse", "question")):
    for x in TR[task]:
        td = tier_target(x["query_id"], x["t"])
        if task == "l1_reuse": pair(x["pair_id"], x["incoming_query"], x["matched_query"])
        for path, v in x["paths"].items():
            a = ans(v["key"])
            if a is None: continue
            if task == "l1_reuse" and path in ("A_stored_answer", "C_fresh"): j2(x["incoming_query"], x["reference_answer"], v["key"])
            aid = add3(x["query_id"], x[qf], td, x["reference_answer"], a); sid = None
            if "evidence_block" in v:
                sid = P.tid(x["query_id"], v["evidence_block"])
                if len([j for j in ("llama8b", "qwen7b", "mistral7b") if j in S3.get(sid, {})]) < 3: s3.setdefault(sid, {"task_id": sid, "query_id": x["query_id"], "question": x[qf], "evidence": v["evidence_block"]})
            tier_rows.append({"task": task, "query_id": x["query_id"], "path": path, "answer": a, "answer_task_id": aid, "sufficiency_task_id": sid, "target_date": td, "context_empty": not v["context"]})
with open(W/"results/tier_path_rows_reported.jsonl", "w", encoding="utf-8") as f:
    for r in tier_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
# --- model-based validated-subset study: 4 arms on the held-out 400, frozen targets from the reference stage ---
ARM = {"fresh": "fresh", "freshcache": "freshcache", "semanticttl_quality_0.60_1-2": "sttl_q060_k1_2", "semanticttl_drift_0.40_1-16": "sttl_k1_16"}
mb = []
for q, rec in AK["keys"]["heldout"].items():
    for arm, src in ARM.items():
        k, tier = rec[src]; a = ans(k)
        if a is None: continue
        tid = add3(q, REFV[q]["question"], (REFV[q]["target_time"] or "unknown") if REFV[q]["target_time"] else "unknown", REFV[q]["reference_answer"], a)
        mb.append({"query_id": q, "arm": arm, "tier": tier, "answer_key": k, "answer": a, "task_id": tid})
with open(W/"results/model_based_policy_outputs_reported.jsonl", "w", encoding="utf-8") as f:
    for r in mb: f.write(json.dumps(r, ensure_ascii=False) + "\n")
# --- verifier audit: 2-judge on lv answers; sufficiency reused only when the evidence text is byte-identical ---
VS = jl(W/"verifier/audit/audit_sample.jsonl")
for d in VS: j2(d["query"], d["gold"], d["lv_key"]); j2(d["query"], d["gold"], d["fc_key"]); j2(d["query"], d["gold"], d["fresh_key"])
OV = (OUTPUT_ROOT / "l2_verifier/out")
old_ev = {(t["query_id"], t["arm"]): t["evidence"] for t in jl(OV/"audit_suff_tasks.jsonl")}
old_lab = {j: {(r["query_id"], r["arm"]): r["label"] for r in jl(OV/f"audit_suff_{j}.jsonl")} for j in ("llama8b", "qwen7b")}
sv, reuse = [], {}
for t in jl(W/"verifier/audit/suff_tasks.jsonl"):
    k = (t["query_id"], t["arm"])
    if old_ev.get(k) == t["evidence"] and all(k in old_lab[j] for j in old_lab): reuse[f"{k[0]}|{k[1]}"] = {j: old_lab[j][k] for j in old_lab}
    else: sv.append(t)
json.dump(reuse, open(W/"verifier/audit/suff_reused.json", "w"))
# the model-based targets: REFV target_time is the reference-stage evidence date (reference_validation_analysis: (evidence_time_max)[:10])
wj("judge2_items.jsonl", list(items.values())); wj("jury_pairs.jsonl", list(pairs.values()))
L3 = list(t3.values()); random.Random(42).shuffle(L3); wj("judge3_tasks.jsonl", L3)
L4 = list(s3.values()); random.Random(42).shuffle(L4); wj("suff3_tasks.jsonl", L4); wj("suffv_tasks.jsonl", sv)
print("missing answers", len(miss), "verifier suff reused", len(reuse), flush=True)
