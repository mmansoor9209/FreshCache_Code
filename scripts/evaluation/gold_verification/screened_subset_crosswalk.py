"""Request-ID crosswalk between the SUPPORTED references and the screened held-out requests, with FreshCache and
both SemanticTTL arms scored on each subset and on their intersection.
Reuses the scoring path of scripts/evaluation/answer_audits/heldout_sensitivity_analysis.py verbatim and is gated on the
request-level WAI counts reproducing (FreshCache 7, SemanticTTL kappa=1/16 64, SemanticTTL theta=0.60 kappa=1/2 25).
Writes outputs/audit_recounts/artifacts/screened_subset_crosswalk.json; read-only elsewhere."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import json, csv, pathlib, sys, collections; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
SRC=pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
HS=(OUTPUT_ROOT / "quality_selection/heldout_sensitivity")
OUT=((OUTPUT_ROOT / "audit_recounts/artifacts")); OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation/answer_audits")); import heldout_sensitivity_analysis as H
jl,h,ABST,A2,GV,RV=H.jl,H.h,H.ABST,H.A2,H.GV,H.RV
sample=json.load(open(A2/"sample.json"))["requests"]
new={d["query_id"]:d for d in jl(HS/"h_sample.jsonl")}
ans={tuple(r["key"]):r["answer"] for r in jl(A2/"answers.jsonl")}
for r in jl(HS/"h_answers.jsonl"): ans[tuple(r["key"])]=r["answer"]
j1={tuple(r["sig"]):r["label"] for r in jl(A2/"judge_llama8b.jsonl")}
j2={tuple(r["sig"]):r["label"] for r in jl(A2/"judge_qwen7b.jsonl")}
for r in jl(HS/"h_judge_llama8b.jsonl"): j1[tuple(r["sig"])]=r["label"]
for r in jl(HS/"h_judge_qwen7b.jsonl"): j2[tuple(r["sig"])]=r["label"]
fid={r["query_id"]:r["final_label"] for r in csv.DictReader(open(GV/"paraphrase_fidelity.csv"))}
unit_of={}
for r in csv.DictReader(open(RV/"r_gold_validity.csv")):
    for q in r["request_ids"].split("|"): unit_of[q]=r
EV={q for q in unit_of if fid.get(q,"BASE_QUERY") in ("SAME","BASE_QUERY") and unit_of[q]["status_majority"]=="VALID_AT_T"}
def outc(d,key):
    a=ans.get(tuple(key))
    if a is None: return None
    if any(p in a.strip().lower() for p in ABST): return "ABSTAIN"
    s=(h(d["query"]),h(d["gold"]),h(a)); l1,l2=j1.get(s),j2.get(s)
    if l1 is None or l2 is None: return None
    return "CORRECT" if (l1=="CORRECT" and l2=="CORRECT") else "WRONG"
rows={}
for d in sample:
    r={"fresh":outc(d,d["fresh_key"]),"fc":outc(d,d["fc_key"]),
       "sttl_k1_16":outc(d,d["sttl_key"]),"sttl_t060_k1_2":outc(d,new[d["query_id"]]["new_sttl_key"])}
    if any(v is None for v in r.values()): continue
    rows[d["query_id"]]=r
held=set(rows); screened_ids=EV&held
sup=csv.DictReader(open((OUTPUT_ROOT / "strong_reference_evaluation/reference_verification/reference_and_diagnosis.csv")))
supported_ids={r["query_id"] for r in sup if r["reference_status"]=="SUPPORTED"}
print(f"held-out scored: {len(rows)}   screened (fidelity+temporal-gold majority): {len(screened_ids)}   supported (two-judge VALID_AT_T): {len(supported_ids)}")
g={a:sum(1 for r in rows.values() if r['fresh']=='CORRECT' and r[a]=='WRONG') for a in ('fc','sttl_k1_16','sttl_t060_k1_2')}
print("gate request-level WAI fc/k1_16/t060_k1_2 =",g, "(run-time reference)")
assert REF.check("heldout_audit/FreshCache_primary/WAI", g['fc']) and REF.check("heldout_audit/SemanticTTL_k1_16/WAI", g['sttl_k1_16']) and REF.check("heldout_audit/SemanticTTL_0.60_k0.5_fixed/WAI", g['sttl_t060_k1_2']), "gate failed"
I=screened_ids&supported_ids
print(f"\ncrosswalk: |supported ∩ screened| = {len(I)}   supported-only = {len(supported_ids-screened_ids)}   screened-only = {len(screened_ids-supported_ids)}")
def score(ids,label):
    sub=[rows[q] for q in ids]; n=len(sub); fcn=sum(1 for r in sub if r["fresh"]=="CORRECT")
    out={"n":n,"fresh_correct":fcn}
    for a in ("fc","sttl_t060_k1_2","sttl_k1_16"):
        c=sum(1 for r in sub if r[a]=="CORRECT"); w=sum(1 for r in sub if r["fresh"]=="CORRECT" and r[a]=="WRONG")
        out[a]={"correct":c,"acc_pct":round(100*c/n,2),"wai":w,"wai_pct":round(100*w/n,2),
                "cwai_pct":round(100*w/fcn,2) if fcn else None}
    print(f"{label}: n={n} fresh-correct={fcn}  FC acc {out['fc']['acc_pct']}% WAI {out['fc']['wai']}  |  "
          f"STTL(.60,1/2) acc {out['sttl_t060_k1_2']['acc_pct']}% WAI {out['sttl_t060_k1_2']['wai']}  |  "
          f"STTL(1/16) acc {out['sttl_k1_16']['acc_pct']}% WAI {out['sttl_k1_16']['wai']}")
    return out
res={"n_heldout":len(rows),"screened":sorted(screened_ids),"supported":sorted(supported_ids),
     "intersection":sorted(I),
     "criteria":{"screened":"paraphrase fidelity SAME or base query, AND temporal-gold validity VALID_AT_T by THREE-model majority (expanded_evidence_3b)",
                 "supported":"reference_status SUPPORTED: two-judge (Llama-3.1-8B + Qwen2.5-7B) agreement VALID_AT_T, disagreement excluded"},
     "scores":{"all_audit":score(rows,"all audit requests"),"screened":score(screened_ids,"screened"),
               "supported":score(supported_ids,"supported"),"intersection":score(I,"supported ∩ screened")}}
json.dump(res,open(OUT/"screened_subset_crosswalk.json","w"),indent=1)
print("\nwrote artifacts/screened_subset_crosswalk.json")
