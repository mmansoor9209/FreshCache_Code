"""Task 5 control: replay FreshCache under alternative half-life sets on the full
31,201-request stream. Read-only: exp.HALF_LIFE is patched in memory and
restored; nothing under data/ is written."""
import json, sys, os, copy, pathlib, collections
ROOT=pathlib.Path("<PROJECT_ROOT>"); os.chdir(ROOT)
for p in ("","v16_exp12","v14_baselines","v13_corrected","v9"): sys.path.insert(0,str(ROOT/p) if p else str(ROOT))
import numpy as np, experiment as exp, engine_all as ea, schedules as sc, mixed_engine as me
OUT=pathlib.Path(__file__).resolve().parent
records=exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE),exp.load_jsonl(exp.MANIFEST_FILE),exp.load_jsonl(exp.PARAPHRASE_FILE))
exp._QUERY_TO_IDX={r["query"]:i for i,r in enumerate(records)}; exp._SIM_MATRIX=np.load(str(exp.SIM_MATRIX_CACHE),mmap_mode="r")
ea._rich_feats(); ea.set_cluster_base(records)
rounds={}
for line in open(ROOT/"v13_corrected/corrected_round_table.jsonl"): d=json.loads(line); rounds[d["url_hash"]]=d["rounds"]
stream=sc.build_stream(records,"zipf_uniform",42); assert len(stream)==31201
audit_ids=set(r["query_id"] for r in json.load(open(ROOT/"validation/heldout_baseline_tuning/answer_audit_k1_16/sample.json"))["requests"])
H=lambda h: h*3600.0
VARIANTS={
 "deployed (experiment.py HALF_LIFE)": None,
 "recomputed, hand-dict filter (713.81->720 clamp/276.47/126.06/124.94)": {"TIMELESS":H(720),"SLOW":H(276.47),"MEDIUM":H(126.06),"FAST":H(124.94)},
 "observed 1h+24h lookup (calibrate.py patch intent)": {"TIMELESS":H(805.94),"SLOW":H(289.52),"MEDIUM":H(128.80),"FAST":H(122.03)},
 "fit-rounds-only 1h+12h lookup (control)": None,   # filled from results.json
 "no domain filter": {"TIMELESS":H(817.21),"SLOW":H(301.11),"MEDIUM":H(133.93),"FAST":H(125.64)},
}
r=json.load(open(OUT/"results.json"))["filter_variants"]["fit_rounds_only_1h+12h (control)"]
VARIANTS["fit-rounds-only 1h+12h lookup (control)"]={c:H(r[c]["half_life_h"]) for c in ("TIMELESS","SLOW","MEDIUM","FAST")}
ORIG=dict(exp.HALF_LIFE); CH,UN=me.CHANGED,me.UNCHANGED
res={}; per_by={}
print("setup complete; starting replays",flush=True)
for name,hl in VARIANTS.items():
    try:
        if hl: exp.HALF_LIFE.update(hl)
        m,per=me.replay(stream,rounds,"FreshCache")
    finally:
        exp.HALF_LIFE.clear(); exp.HALF_LIFE.update(ORIG)
    outs=[o for q,o in per if o is not None]; ch=sum(1 for o in outs if o==CH); un=sum(1 for o in outs if o==UN)
    res[name]={"half_life_h":{c:round(exp.HALF_LIFE[c]/3600,2) for c in ("TIMELESS","SLOW","MEDIUM","FAST")} if hl is None else {c:round(v/3600,2) for c,v in hl.items()},
               "search_saved_pct":round(m["search_saved_pct"],4),"drift_pct":round(100*ch/(ch+un),4),"determinable":ch+un,
               "coverage_pct":round(100*(ch+un)/31201,4),"l1_hits":m["l1_hits"],"l2_hits":m["l2_hits"],"l3_hits":m["l3_hits"],"fetches":m["fetches"]}
    per_by[name]={q:o for q,o in per}
    print("  done:",name,res[name]["search_saved_pct"],res[name]["drift_pct"],flush=True)
    json.dump(res,open(OUT/"replay_control.partial.json","w"),indent=1)
base=per_by["deployed (experiment.py HALF_LIFE)"]
for name in VARIANTS:
    d=sum(1 for q in base if base[q]!=per_by[name].get(q)); da=sum(1 for q in audit_ids if base.get(q)!=per_by[name].get(q))
    res[name]["requests_with_different_outcome_vs_deployed"]=d; res[name]["audited_400_requests_with_different_outcome"]=da
json.dump(res,open(OUT/"replay_control.json","w"),indent=1)
for n,v in res.items():
    print(f"{n}\n   HL {v['half_life_h']}  saved {v['search_saved_pct']}%  drift {v['drift_pct']}%  cov {v['coverage_pct']}%  L1/L2/L3 {v['l1_hits']}/{v['l2_hits']}/{v['l3_hits']}  fetches {v['fetches']}\n   outcome differs vs deployed: {v['requests_with_different_outcome_vs_deployed']} requests; among the 400 audited: {v['audited_400_requests_with_different_outcome']}")
