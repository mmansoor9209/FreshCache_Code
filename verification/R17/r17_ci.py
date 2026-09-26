import json,math,random,collections,pathlib,sys,os
ROOT=pathlib.Path("<PROJECT_ROOT>"); os.chdir(ROOT); DATA=ROOT/"data"; JURY=ROOT/"v8/jury"
OUT=pathlib.Path(__file__).resolve().parent; log=open(OUT/"run.log","w")
def say(*a):
    s=" ".join(str(x) for x in a); print(s); log.write(s+"\n")
def jl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
def row_key(r): return r.get("row_id") or f"{r.get('query_id','')}|{r.get('matched_query','')[:80]}"
BANDS6=[(0.35,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.95),(0.95,1.01)]
def band_of(s,bands):
    for a,b in bands:
        if a<=s<b: return f"{a:.2f}-{b:.2f}"
    return None
# cluster map for the quality-selected sample
qc={}
for r in jl(DATA/"paraphrase_clusters.jsonl"): qc[r["query_id"]]=r.get("cluster_id") or r.get("base_query_id")
for r in jl(DATA/"queries.jsonl"): qc.setdefault(r["query_id"],r.get("cluster_id") or r["query_id"])
def published_row(key,fname,label,popfile,Nfield):
    base=[r for r in jl(DATA/fname) if r.get("verdict") in ("SAME","DIFFERENT")]
    verd={"llama3b":{row_key(r):r["verdict"] for r in base}}
    for j in ("llama8b","qwen7b","mistral7b"):
        verd[j]={r["key"]:r["verdict"] for r in jl(JURY/f"{j}__{key}.jsonl") if r.get("verdict") in ("SAME","DIFFERENT")}
    common=set.intersection(*(set(v) for v in verd.values()))
    info={row_key(r):r for r in base}
    items=[]
    for k in sorted(common):
        v=[verd[j][k] for j in ("llama3b","llama8b","qwen7b","mistral7b")]; nd=sum(1 for x in v if x=="DIFFERENT")
        items.append({"band":info[k]["band"],"v":"DIFFERENT" if nd>=2 else "SAME","cluster":info[k].get("cluster_incoming") or qc.get(info[k].get("query_id"))})
    # population strata
    src=None; band_pop=None; N=None
    stem=fname.replace("judgments","judgment_summary").replace(".jsonl",".json")
    if (DATA/stem).exists():
        s=json.load(open(DATA/stem)); band_pop={b:v["population"] for b,v in s.get("by_band",{}).items()}; N=s.get(Nfield) or s.get("n_realized_hits"); src=stem
    if (not band_pop) and popfile and (DATA/popfile).exists():
        pop=jl(DATA/popfile); bands=sorted({it["band"] for it in items}); bp=collections.Counter()
        for p in pop:
            b=next((bb for bb in bands if float(bb.split("-")[0])<=p["similarity"]<float(bb.split("-")[1])),None)
            if b: bp[b]+=1
        band_pop=dict(bp); N=len(pop); src=popfile
    return label,items,band_pop,N,src
def analyse(label,items,band_pop,N,src,old_ci_method):
    strata={}
    for b,Nh in sorted(band_pop.items()):
        rows=[it for it in items if it["band"]==b]; nh=len(rows); yh=sum(1 for it in rows if it["v"]=="DIFFERENT")
        if nh: strata[b]={"N_h":Nh,"n_h":nh,"y_h":yh,"p_h":yh/nh,"weight_N_over_n":Nh/nh}
    Ntot=sum(s["N_h"] for s in strata.values())
    est=sum(s["N_h"]/Ntot*s["p_h"] for s in strata.values())
    # old CI as implemented
    if old_ci_method=="stratified_wald_fpc":
        var=0.0
        for s in strata.values():
            if s["n_h"]>1: var+=(s["N_h"]/Ntot)**2*(s["p_h"]*(1-s["p_h"])/s["n_h"])*max(0.0,(s["N_h"]-s["n_h"])/max(s["N_h"]-1,1))
        se=math.sqrt(var); old=[max(0,est-1.96*se),min(1,est+1.96*se)]
    else:  # wilson on total judged count, ignoring strata (aggregate_selected_jury.py)
        n=sum(s["n_h"] for s in strata.values()); z=1.959963985; d0=1+z*z/n; c=(est+z*z/(2*n))/d0; hw=z*math.sqrt(est*(1-est)/n+z*z/(4*n*n))/d0; old=[c-hw,c+hw]
    # stratified bootstrap
    rng=random.Random(42); by_band={b:[it for it in items if it["band"]==b] for b in strata}; reps=[]
    for _ in range(10000):
        e=0.0
        for b,s in strata.items():
            rows=by_band[b]; k=len(rows); d=sum(1 for _ in range(k) if rows[rng.randrange(k)]["v"]=="DIFFERENT"); e+=s["N_h"]/Ntot*d/k
        reps.append(e)
    reps.sort(); boot=[reps[int(0.025*10000)],reps[int(0.975*10000)-1]]
    # cluster-aware stratified bootstrap: resample clusters within band
    n_cl=len({it["cluster"] for it in items if it["cluster"]}); multi=sum(1 for c,k in collections.Counter(it["cluster"] for it in items if it["cluster"]).items() if k>1)
    creps=[]
    if n_cl and multi:
        rng=random.Random(42)
        for _ in range(10000):
            e=0.0
            for b,s in strata.items():
                groups=collections.defaultdict(list)
                for it in by_band[b]: groups[it["cluster"] or id(it)].append(it["v"])
                cl=list(groups.values()); draws=[cl[rng.randrange(len(cl))] for _ in range(len(cl))]
                tot=sum(len(g) for g in draws); d=sum(v=="DIFFERENT" for g in draws for v in g); e+=s["N_h"]/Ntot*(d/tot if tot else 0)
            creps.append(e)
        creps.sort(); cboot=[creps[250],creps[9749]]
    else: cboot=None
    n=sum(s["n_h"] for s in strata.values())
    say(f"\n{label}   source of N_h: {src}   N={Ntot:,}  judged n={n}")
    for b,s in strata.items(): say(f"   {b}  N_h {s['N_h']:>6,}  n_h {s['n_h']:>3}  y_h {s['y_h']:>3}  p_h {100*s['p_h']:6.1f}%  w=N_h/n_h {s['weight_N_over_n']:8.2f}")
    say(f"   weighted mismatch {100*est:.2f}%   old CI ({old_ci_method}) [{100*old[0]:.2f}, {100*old[1]:.2f}]   stratified bootstrap [{100*boot[0]:.2f}, {100*boot[1]:.2f}]   cluster-aware {'[%.2f, %.2f]'%(100*cboot[0],100*cboot[1]) if cboot else 'n/a'}   clusters {n_cl} ({multi} with >1 judged item)")
    return {"weighted_mismatch_pct":round(100*est,3),"old_ci_pct":[round(100*x,3) for x in old],"old_ci_method":old_ci_method,
            "stratified_bootstrap_ci_pct":[round(100*x,3) for x in boot],"cluster_aware_ci_pct":[round(100*x,3) for x in cboot] if cboot else None,
            "judged_n":n,"population_N":Ntot,"N_source":src,"strata":strata,"n_clusters":n_cl,"clusters_with_multiple_items":multi}
res={}
for key,fname,label,popfile,Nf in [("scalm","scalm_hit_judgments_24h.jsonl","SCALM-style","scalm_hits_t24h.jsonl","n_realized_hits"),
    ("semanticttl","semanticttl_hit_judgments_24h.jsonl","SemanticTTL permissive (theta=0.40)","semanticttl_hits_t24h.jsonl","n_realized_hits"),
    ("fc_l1","l1_hit_judgments_24h.jsonl","FreshCache L1",None,"n_realized_l1_hits"),
    ("fc_l2","l2_hit_judgments_24h.jsonl","FreshCache L2","l2_hits_t24h_floor075.jsonl","n_realized_l2_hits")]:
    lab,items,bp,N,src=published_row(key,fname,label,popfile,Nf); res[lab]=analyse(lab,items,bp,N,src,"stratified_wald_fpc")
# quality-selected row (own artifacts)
Q=ROOT/"Reframe_ResearchPaper/artifacts/q16"; sample={(r["query_id"],r["matched_query"]):r for r in jl(Q/"jury_selected/sample.jsonl")}
verd={j:{(r["query_id"],r["matched_query"]):r["verdict"] for r in jl(Q/f"jury_selected/{j}.jsonl")} for j in ("llama3b","llama8b","qwen7b","mistral7b")}
items=[]
for k in sorted(set.intersection(*(set(v) for v in verd.values()))):
    nd=sum(1 for j in verd if verd[j][k]=="DIFFERENT"); items.append({"band":sample[k]["band"],"v":"DIFFERENT" if nd>=2 else "SAME","cluster":qc.get(k[0])})
pop=jl(Q/"semanticttl_selected_hits_t24h.jsonl"); bp=collections.Counter()
for p in pop:
    b=band_of(p["similarity"],[(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)]); 
    if b: bp[b]+=1
res["SemanticTTL quality-selected (0.60, 1/2)"]=analyse("SemanticTTL quality-selected (0.60, 1/2)",items,dict(bp),len(pop),"artifacts/q16/semanticttl_selected_hits_t24h.jsonl","wilson_on_total_judged")
say("\nACCOUNTING: judged items per row:",{k:v["judged_n"] for k,v in res.items()},"| four published rows total",sum(v["judged_n"] for k,v in res.items() if "quality" not in k))
json.dump(res,open(OUT/"R17_results.json","w"),indent=1); say("wrote R17_results.json")
