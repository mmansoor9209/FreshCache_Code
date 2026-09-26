import json,sys,os,copy,math,collections,pathlib
ROOT=pathlib.Path("<PROJECT_ROOT>"); sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
import calibrate as C, freshcache.risk_model as RM
OUT=pathlib.Path(__file__).resolve().parent; log=open(OUT/"run.log","a"); res={}
def say(*a):
    s=" ".join(str(x) for x in a); print(s); log.write(s+"\n")
say("\n[addendum_tests.py]")
# ---- (1) day-rounding reconciliation ----
full={"TIMELESS":713.81,"SLOW":276.47,"MEDIUM":126.06,"FAST":124.94}; dep={"TIMELESS":720,"SLOW":276,"MEDIUM":127.2,"FAST":124.8}
say("(1) hours -> floor -> days -> round 1dp -> hours:")
rec={}
for fc,h in full.items():
    hf=max(h,720.0) if fc=="TIMELESS" else h; d=round(hf/24,1); back=round(d*24,2)
    rec[fc]={"full_h":h,"after_floor_h":hf,"days_1dp":d,"back_h":back,"deployed_h":dep[fc],"match":abs(back-dep[fc])<1e-9,"deployed_seconds":dep[fc]*3600}
    say(f"   {fc:<9} {h} h -> {hf} h -> {d} d -> {back} h   deployed {dep[fc]} h   {'MATCH' if rec[fc]['match'] else 'no'}")
say("   calibrate._fmt_hl prints half-lives >= 2 days as f'{seconds/86400:.1f}d'; experiment.py HALF_LIFE comments read '5.2d','5.3d','11.5d','30d' and its seconds equal those day values x 86400")
res["day_rounding"]=rec
# ---- (2) TIMELESS/SLOW isolation: keep 1h/12h/24h, remove or perturb ONLY 7d and 48h ----
mf=C.load_jsonl(C.MANIFEST_FILE); ch=C.load_jsonl(C.CHANGE_FILE); AGES={"rerun_1h":3600,"rerun_12h":43200,"rerun_24h":86400,"rerun_7d":604800}
def counts(mf,ch,filt):
    tr=collections.Counter(); rm=collections.Counter(); cg=collections.Counter()
    for m in mf:
        if m.get("snapshot_available") and m.get("run_id","run_00")!="run_00":
            fc=m.get("freshness_class",""); d=m.get("domain","")
            if fc and filt(d,fc): rm[(m["run_id"],fc)]+=1; continue
            tr[(m["run_id"],fc)]+=1
    for c in ch:
        if c.get("noise") or c.get("detected_at","")<=C.NEW_RUN_CUTOFF: continue
        cg[(c["run_id"],c["freshness_class"])]+=1
    return tr,rm,cg
def fit(tr,cg,fc):
    pairs=[(cg.get((w,fc),0)/tr[(w,fc)] if tr.get((w,fc)) else None,AGES[w]) for w in C.FIT_WINDOWS_BY_CLASS[fc]]
    hl,_=C.fit_half_life_multi_window(pairs); return None if hl is None else round(hl/3600,2)
later={"rerun_7d","rerun_48h"}
base=counts(mf,ch,RM.is_noise_url)
mf2=[m for m in mf if m.get("run_id") not in later]; ch2=[c for c in ch if c.get("run_id") not in later]; t2=counts(mf2,ch2,RM.is_noise_url)
mf3=copy.deepcopy(mf); ch3=[c for c in ch if c.get("run_id") not in later]
for m in mf3:
    if m.get("run_id") in later: m["snapshot_available"]=bool(m.get("snapshot_available")) and (hash(m["url_hash"])%2==0)
t3=counts(mf3,ch3,RM.is_noise_url)
say("(2) TIMELESS/SLOW: 1h/12h/24h retained; 7d and 48h removed, then perturbed (7d/48h change events deleted, denominators halved)")
iso2={}
for fc in ("TIMELESS","SLOW"):
    row={}
    for lab,(tr,rm,cg) in (("baseline",base),("7d_48h_removed",t2),("7d_48h_perturbed",t3)):
        u=fit(tr,cg,fc); op=max(u,720.0) if (fc=="TIMELESS" and u is not None) else u
        row[lab]={"unclamped_h":u,"operational_h_after_floor":op,
                  "mask_removed_in_window":sum(rm[(w,fc)] for w in C.FIT_WINDOWS_BY_CLASS[fc]),
                  "tracked_in_window":sum(tr[(w,fc)] for w in C.FIT_WINDOWS_BY_CLASS[fc]),
                  "weights_window_ages_s":[AGES[w] for w in C.FIT_WINDOWS_BY_CLASS[fc]],
                  "equiv_ttl_L3_h":None if op is None else round(-op*math.log(1-0.35)/(1.0*math.log(2)),2)}
    iso2[fc]=row
    for lab,v in row.items(): say(f"   {fc:<9} {lab:<17} unclamped {v['unclamped_h']} h  operational {v['operational_h_after_floor']} h  mask removed {v['mask_removed_in_window']}/{v['tracked_in_window']}  L3 equiv-TTL {v['equiv_ttl_L3_h']} h")
    say(f"   {fc:<9} lookup: hand dictionary, identical across conditions (constant); weights: window ages, identical")
res["timeless_slow_isolation"]=iso2
# ---- (2b) class-specific vs common-cutoff lookup ----
def vol_from(mf,ch,rounds):
    obs=collections.defaultdict(list); cs={(c["url_hash"],c["run_id"]) for c in ch if c.get("detected_at","")>C.NEW_RUN_CUTOFF and not c.get("noise")}
    for m in mf:
        if m.get("run_id") in rounds and m.get("snapshot_available"): obs[m.get("domain","").lower().removeprefix("www.")].append(int((m["url_hash"],m["run_id"]) in cs))
    return {d:round(sum(o)/len(o),3) for d,o in obs.items() if len(o)>=2}
def mkf(lk):
    def f(d,fc):
        d=(d or "").lower().removeprefix("www."); v=lk.get(d,RM._DEFAULT_DOMAIN_VOL)
        if v==RM._DEFAULT_DOMAIN_VOL:
            for k,vv in lk.items():
                if d.endswith("."+k): v=vv; break
        return v>RM.CLASS_VOL_CEILING.get(fc,0.5)
    return f
common=vol_from(mf,ch,{"rerun_1h","rerun_12h"}); spec={fc:vol_from(mf,ch,set(C.FIT_WINDOWS_BY_CLASS[fc])) for fc in ("TIMELESS","SLOW","MEDIUM","FAST")}
say("(2b) lookup scope: existing 'fitting-rounds-only' control used a COMMON 1h+12h cutoff for every class; class-specific variant uses each class's own fit rounds:")
cmp={}
for fc in ("TIMELESS","SLOW","MEDIUM","FAST"):
    trc,rmc,cgc=counts(mf,ch,mkf(common)); trs,rms,cgs=counts(mf,ch,mkf(spec[fc]))
    cmp[fc]={"common_1h12h":{"h":fit(trc,cgc,fc),"removed":sum(rmc[(w,fc)] for w in C.FIT_WINDOWS_BY_CLASS[fc]),"lookup_domains":len(common)},
             "class_specific":{"h":fit(trs,cgs,fc),"removed":sum(rms[(w,fc)] for w in C.FIT_WINDOWS_BY_CLASS[fc]),"lookup_domains":len(spec[fc]),"rounds":C.FIT_WINDOWS_BY_CLASS[fc]}}
    say(f"   {fc:<9} common {cmp[fc]['common_1h12h']['h']} h (removed {cmp[fc]['common_1h12h']['removed']}, {len(common)} domains)   class-specific {cmp[fc]['class_specific']['h']} h (removed {cmp[fc]['class_specific']['removed']}, {len(spec[fc])} domains, rounds {C.FIT_WINDOWS_BY_CLASS[fc]})")
res["lookup_scope"]=cmp
json.dump(res,open(OUT/"addendum_results.json","w"),indent=1); say("wrote addendum_results.json")
