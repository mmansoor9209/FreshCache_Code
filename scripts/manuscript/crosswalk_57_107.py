"""V35-N03: request-ID crosswalk between the 57 SUPPORTED references and the
107 screened held-out requests, and both policies scored on the same subsets.
Reuses the published scoring path (h4_analyze) verbatim; gated on the
request-level counts reproducing. Read-only outside this folder."""
import json, csv, pathlib, sys, collections
SRC=pathlib.Path("<PROJECT_ROOT>")
HS=SRC/"validation/quality_aware_baseline/heldout_sensitivity"
OUT=pathlib.Path(__file__).resolve().parent.parent/"artifacts"; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HS)); import h4_analyze as H
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
held=set(rows); screened107=EV&held
sup=csv.DictReader(open(SRC/"validation_V3/strong_reference_evaluation/02_reference_verification/reference_and_diagnosis.csv"))
supported57={r["query_id"] for r in sup if r["reference_status"]=="SUPPORTED"}
print(f"held-out scored: {len(rows)}   screened (fidelity+temporal-gold majority): {len(screened107)}   supported (two-judge VALID_AT_T): {len(supported57)}")
g={a:sum(1 for r in rows.values() if r['fresh']=='CORRECT' and r[a]=='WRONG') for a in ('fc','sttl_k1_16','sttl_t060_k1_2')}
print("gate request-level WAI fc/k1_16/t060_k1_2 =",g, "(published 7/64/25)")
assert g=={'fc':7,'sttl_k1_16':64,'sttl_t060_k1_2':25}, "gate failed"
I=screened107&supported57
print(f"\ncrosswalk: |57 ∩ 107| = {len(I)}   57-only = {len(supported57-screened107)}   107-only = {len(screened107-supported57)}")
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
res={"n_heldout":len(rows),"screened_107":sorted(screened107),"supported_57":sorted(supported57),
     "intersection":sorted(I),
     "criteria":{"107":"paraphrase fidelity SAME or base query, AND temporal-gold validity VALID_AT_T by THREE-model majority (revised_3b)",
                 "57":"reference_status SUPPORTED: two-judge (Llama-3.1-8B + Qwen2.5-7B) agreement VALID_AT_T, disagreement excluded"},
     "scores":{"all_400":score(rows,"all 400"),"screened_107":score(screened107,"screened 107"),
               "supported_57":score(supported57,"supported 57"),"intersection":score(I,"57∩107")}}
json.dump(res,open(OUT/"crosswalk_57_107.json","w"),indent=1)
print("\nwrote artifacts/crosswalk_57_107.json")
