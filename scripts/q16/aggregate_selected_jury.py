"""Aggregate the four-juror audit of SemanticTTL at (theta=0.60, kappa=1/2).

Published protocol: majority of four, 2-2 ties resolve to DIFFERENT,
band-stratified sample reweighted to the realized-hit population, Fleiss kappa
across the four jurors.
"""
import json, math, pathlib, collections

ART=pathlib.Path(__file__).resolve().parent.parent.parent/"artifacts/q16"
JD=ART/"jury_selected"
JURORS=["llama3b","llama8b","qwen7b","mistral7b"]

pop=[json.loads(l) for l in open(ART/"semanticttl_selected_hits_t24h.jsonl",encoding="utf-8")]
per={j:{} for j in JURORS}
for j in JURORS:
    for r in (json.loads(l) for l in open(JD/f"{j}.jsonl",encoding="utf-8")):
        per[j][(r["query_id"],r["matched_query"])]=r["verdict"]
keys=sorted(set.intersection(*(set(per[j]) for j in JURORS)))
sample={ (r["query_id"],r["matched_query"]):r for r in
         (json.loads(l) for l in open(JD/"sample.jsonl",encoding="utf-8")) }
print(f"judged pairs with all four jurors: {len(keys)}")

rows=[]
for k in keys:
    v=[per[j][k] for j in JURORS]
    nd=sum(1 for x in v if x=="DIFFERENT")
    rows.append({"band":sample[k]["band"],"n_different":nd,
                 "majority_different": nd>=2 if nd==2 else nd>len(JURORS)/2})
# published rule: 2-2 ties resolve to DIFFERENT  => DIFFERENT iff nd>=2
for r in rows: r["majority_different"]= r["n_different"]>=2

BANDS=sorted({r["band"] for r in rows})
def in_band(s,b):
    a,c=b.split("-"); return float(a)<=s<float(c)
Nh=collections.Counter()
for h in pop:
    for b in BANDS:
        if in_band(h["similarity"],b): Nh[b]+=1; break

num=den=0.0; detail={}
for b in BANDS:
    sj=[r for r in rows if r["band"]==b]
    if not sj or not Nh[b]: continue
    rate=sum(r["majority_different"] for r in sj)/len(sj)
    num+=rate*Nh[b]; den+=Nh[b]
    detail[b]={"judged":len(sj),"population":Nh[b],"mismatch_rate":round(rate,4)}
est=num/den
n_eff=sum(d["judged"] for d in detail.values())
z=1.959963985; d0=1+z*z/n_eff
c=(est+z*z/(2*n_eff))/d0
hw=z*math.sqrt(est*(1-est)/n_eff+z*z/(4*n_eff*n_eff))/d0

def fleiss(rs,n=4):
    N=len(rs); Pi=[]
    for r in rs:
        d=r["n_different"]; s=n-d
        Pi.append((d*(d-1)+s*(s-1))/(n*(n-1)))
    Pbar=sum(Pi)/N
    pj=sum(r["n_different"] for r in rs)/(N*n)
    Pe=pj**2+(1-pj)**2
    return (Pbar-Pe)/(1-Pe)

k=fleiss(rows)
print(f"\npopulation {int(den):,} realized hits   judged {n_eff}")
print(f"mismatch {100*est:.1f}%   Wilson 95% CI [{100*(c-hw):.1f}, {100*(c+hw):.1f}]   Fleiss kappa {k:.3f}")
for b,d in sorted(detail.items()):
    print(f"   {b}  judged {d['judged']:>3}  pop {d['population']:>6,}  rate {100*d['mismatch_rate']:.1f}%")

out={"policy":"SemanticTTL","theta":0.60,"kappa":0.5,
     "operating_point":"fixed-24h simulated age, published generator",
     "population_realized_hits":int(den),"judged":n_eff,"jurors":JURORS,
     "tie_rule":"2-2 resolves to DIFFERENT",
     "mismatch_pct":round(100*est,2),
     "ci95":[round(100*(c-hw),2),round(100*(c+hw),2)],
     "fleiss_kappa":round(k,3),"by_band":detail}
(ART/"q16_selected_point_mismatch.json").write_text(json.dumps(out,indent=1))
print("\nwrote artifacts/q16/q16_selected_point_mismatch.json")
