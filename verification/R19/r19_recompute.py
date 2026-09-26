import os,sys,json,pathlib,hashlib,collections
os.environ["HF_DATASETS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
ROOT=pathlib.Path("<PROJECT_ROOT>"); E=ROOT/"validation_V3/remaining_critical_issues/09_external_generalization"
sys.path.insert(0,str(E)); os.chdir(ROOT)
OUT=pathlib.Path(__file__).resolve().parent; L=open(OUT/"run.log","w")
def say(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); L.write(s+"\n")
import run_external as X
stored=json.load(open(E/"external_results.json"))
say("module constants: AGE",X.AGE,"NATURAL_AGE",X.NATURAL_AGE,"POLICIES",X.POLICIES)
say("consumer params: HALF_LIFE",dict(X.exp.HALF_LIFE),"FIXED_TTL",dict(X.exp.FIXED_TTL),"EPS_ANSWER",X.exp.EPS_ANSWER,"TIER_MULT",dict(X.exp.TIER_MULT))
import inspect; say("reuse_decision signature:",inspect.signature(X.reuse_decision),"| evaluate calls reuse_decision(pol, fc, AGE) -> theta_ok always True")
res={"params":{"AGE_s":X.AGE,"NATURAL_AGE_s":X.NATURAL_AGE,"HALF_LIFE":dict(X.exp.HALF_LIFE),"FIXED_TTL":dict(X.exp.FIXED_TTL),"EPS_ANSWER":X.exp.EPS_ANSWER,"TIER_MULT":dict(X.exp.TIER_MULT)}}
# effective TTLs
say("\nSemanticTTL/ExactTTL effective TTL per class (age <= FIXED_TTL, kappa absent =1):",{k:f"{v/3600:.2f} h" for k,v in X.exp.FIXED_TTL.items()})
import math
eff={k:(-h*math.log(1-X.exp.EPS_ANSWER)/(X.exp.TIER_MULT['answer']*math.log(2))) for k,h in X.exp.HALF_LIFE.items()}
say("FreshCache-L1 effective admission age T = -h ln(1-eps)/(m ln2):",{k:f"{v/3600:.2f} h" for k,v in eff.items()},"| at 24 h admits:",{k:bool(X.exp.p_stale(k,X.AGE,'answer')<=X.exp.EPS_ANSWER) for k in X.exp.HALF_LIFE})
res["effective"]={"fixed_ttl_h":{k:v/3600 for k,v in X.exp.FIXED_TTL.items()},"freshcache_L1_admit_age_h":{k:v/3600 for k,v in eff.items()},"admits_at_24h":{k:bool(X.exp.p_stale(k,X.AGE,'answer')<=X.exp.EPS_ANSWER) for k in X.exp.HALF_LIFE}}
def recount(name,rows):
    uq=len({r["question"] for r in rows}); return {"pairs":len(rows),"unique_questions":uq}
def compare(name,rows):
    log=[]; r=X.evaluate(rows,name,log,X.AGE,"  [audit recompute, common 24 h]")
    cmp={}
    for pol,v in r["policies"].items():
        s=stored["results"][name]["policies"][pol]
        cmp[pol]={"reuse":(v["reuse"],s["reuse"]),"error_n":(v["cache_induced_error_n"],s["cache_induced_error_n"]),"acc_pct":(v["accuracy_pct"],s["accuracy_pct"]),"n":(v["n_pairs"],s["n_pairs"]),
                  "match":v["reuse"]==s["reuse"] and v["cache_induced_error_n"]==s["cache_induced_error_n"] and v["n_pairs"]==s["n_pairs"]}
    return r,cmp
# DailyQA (local)
dq=X.load_dailyqa(); rc=recount("DailyQA",dq); say("\nDailyQA recount:",rc,"| stored pairs",stored["results"]["DailyQA"]["n_pairs"])
dup=collections.Counter(); 
for p in sorted(X.DQ.glob("qa_*.jsonl")):
    qs=[ (json.loads(l).get("question") or json.loads(l).get("query") or "").strip() for l in open(p) if l.strip()]; dup[p.stem]=(len(qs),len(set(qs)))
say("per-day rows vs unique questions (duplicates within a day: last answer wins, run_external.py:171):",dict(list(dup.items())[:6]),"... days",len(dup))
r_dq,c_dq=compare("DailyQA",dq); say("DailyQA policy integers (recomputed, stored):",json.dumps(c_dq))
res["DailyQA"]={"recount":rc,"per_day_rows_unique":dup,"compare":c_dq,"recomputed":{p:{k:v[k] for k in ("n_pairs","reuse","cache_induced_error_n","accuracy_pct")} for p,v in r_dq["policies"].items()}}
# EvolvingQA (HF cache, offline)
try:
    eq=X.load_evolvingqa(); rc=recount("EvolvingQA",eq); say("\nEvolvingQA recount (offline cache):",rc,"| stored",stored["results"]["EvolvingQA"]["n_pairs"])
    r_eq,c_eq=compare("EvolvingQA",eq); say("EvolvingQA policy integers (recomputed, stored):",json.dumps(c_eq))
    res["EvolvingQA"]={"recount":rc,"compare":c_eq,"recomputed":{p:{k:v[k] for k in ("n_pairs","reuse","cache_induced_error_n","accuracy_pct")} for p,v in r_eq["policies"].items()}}
except Exception as e:
    say("\nEvolvingQA: could not load offline ->",type(e).__name__,str(e)[:200]); res["EvolvingQA"]={"blocker":f"{type(e).__name__}: {str(e)[:200]}"}
json.dump(res,open(OUT/"R19_recompute.json","w"),indent=1,default=str); say("wrote R19_recompute.json")
