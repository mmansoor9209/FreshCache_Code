"""R13/R14 isolation test. Read-only on the repository; writes only here.
Uses calibrate.py's OWN functions (imported), on temporary in-memory copies of
the inputs. calibrate.main() is never called (it would patch risk_model.py and
overwrite data/calibration_report.json)."""
import json, sys, copy, math, pathlib, importlib, types, collections
ROOT=pathlib.Path("<PROJECT_ROOT>"); sys.path.insert(0,str(ROOT))
import os; os.chdir(ROOT)
import calibrate as C
import freshcache.risk_model as RM
OUT=pathlib.Path(__file__).resolve().parent
log=open(OUT/"run.log","w")
def say(*a):
    s=" ".join(str(x) for x in a); print(s); log.write(s+"\n")
manifest=C.load_jsonl(C.MANIFEST_FILE); changes=C.load_jsonl(C.CHANGE_FILE)
say("inputs: manifest",len(manifest),"change_log",len(changes))
RUN_AGES={"rerun_1h":C.T_1H,"rerun_12h":12*3600,"rerun_24h":C.T_24H,"rerun_7d":C.T_7D}
CLASSES=["TIMELESS","SLOW","MEDIUM","FAST"]

def tracked_counts(mf, filt):
    """replicates calibrate.compute_run_class_rates denominators with an explicit filter fn,
    returning tracked, removed counts per (run,class)"""
    tracked=collections.Counter(); removed=collections.Counter()
    for m in mf:
        if m.get("snapshot_available") and m.get("run_id","run_00")!="run_00":
            fc=m.get("freshness_class",""); dom=m.get("domain","")
            if fc and filt(dom,fc): removed[(m["run_id"],fc)]+=1; continue
            tracked[(m["run_id"],fc)]+=1
    return tracked,removed
def rates_from(mf,ch,filt):
    tracked,removed=tracked_counts(mf,filt)
    changed=collections.Counter()
    for c in ch:
        if c.get("noise",False): continue
        if c.get("detected_at","")<=C.NEW_RUN_CUTOFF: continue
        changed[(c.get("run_id"),c.get("freshness_class",""))]+=1
    rates={k:changed.get(k,0)/t for k,t in tracked.items() if t}
    return rates,tracked,removed,changed
def fit(rates):
    out={}
    for fc in CLASSES:
        pairs=[(rates.get((r,fc)),RUN_AGES[r]) for r in C.FIT_WINDOWS_BY_CLASS[fc]]
        hl,note=C.fit_half_life_multi_window(pairs); out[fc]=(hl,note)
    return out
def hours(hl): return None if hl is None else round(hl/3600,2)

# --- 0. sanity: my replication of compute_run_class_rates == calibrate's own ---
own=C.compute_run_class_rates(manifest,changes)
mine,tr,rm,chg=rates_from(manifest,changes,RM.is_noise_url)
assert all(abs(own[k]-mine[k])<1e-12 for k in own) and set(own)==set(mine)
say("\n[0] replication of calibrate.compute_run_class_rates: identical on",len(own),"(run,class) cells")

# --- 1. baseline: original inputs, hand-dict filter (what calibrate.py does today) ---
base=fit(own)
say("\n[1] BASELINE fit with current code + current inputs + is_noise_url (hand dict):")
for fc in CLASSES: say(f"   {fc:<9} windows {C.FIT_WINDOWS_BY_CLASS[fc]}  rates {[round(own.get((r,fc)),6) for r in C.FIT_WINDOWS_BY_CLASS[fc]]}  -> {hours(base[fc][0])} h")
say("   deployed HALF_LIFE (experiment.py): TIMELESS 720 h, SLOW 276 h, MEDIUM 127.2 h, FAST 124.8 h")
say("   retained (tracked) per (run,class) and removed by domain filter:")
for fc in CLASSES:
    for r in ["rerun_1h","rerun_12h","rerun_24h","rerun_7d"]:
        say(f"     {fc:<9} {r:<9} tracked {tr[(r,fc)]:>6}  removed {rm[(r,fc)]:>5}  changed {chg[(r,fc)]:>5}")

# --- 2. remove ALL later-round records (rerun_24h, rerun_7d, rerun_48h) from both inputs ---
later={"rerun_24h","rerun_7d","rerun_48h"}
mf2=[m for m in manifest if m.get("run_id") not in later]; ch2=[c for c in changes if c.get("run_id") not in later]
r2,_,_,_=rates_from(mf2,ch2,RM.is_noise_url); f2=fit(r2)
say("\n[2] LATER ROUNDS REMOVED (24h, 7d, 48h dropped from manifest+change_log; filter = same hand dict):")
for fc in CLASSES: say(f"   {fc:<9} {hours(base[fc][0])} h -> {hours(f2[fc][0])} h   delta {None if f2[fc][0] is None or base[fc][0] is None else round((f2[fc][0]-base[fc][0])/3600,3)} h")

# --- 3. perturb only later-round records: flip every 24h/7d change record to unchanged (drop them) and double the rest ---
ch3=[c for c in changes if c.get("run_id") not in later]           # remove all later-round change events
mf3=copy.deepcopy(manifest)
for m in mf3:
    if m.get("run_id") in later: m["snapshot_available"]=bool(m.get("snapshot_available")) and (hash(m["url_hash"])%2==0)  # halve later-round denominators
r3,_,_,_=rates_from(mf3,ch3,RM.is_noise_url); f3=fit(r3)
say("\n[3] LATER ROUNDS PERTURBED (24h/7d change events removed, 24h/7d denominators halved; fitting rounds untouched):")
for fc in CLASSES: say(f"   {fc:<9} {hours(base[fc][0])} h -> {hours(f3[fc][0])} h")

# --- 4. positive control: perturb an IN-WINDOW round (drop rerun_12h) to prove the fit is live, not cached ---
mf4=[m for m in manifest if m.get("run_id")!="rerun_12h"]; ch4=[c for c in changes if c.get("run_id")!="rerun_12h"]
r4,_,_,_=rates_from(mf4,ch4,RM.is_noise_url); f4=fit(r4)
say("\n[4] POSITIVE CONTROL: rerun_12h removed (an in-window round) -> fits must move:")
for fc in CLASSES: say(f"   {fc:<9} {hours(base[fc][0])} h -> {hours(f4[fc][0])} h")

# --- 5. the OBSERVED volatility lookup (calibrate.compute_observed_domain_volatility) with/without 24h ---
gt=C.build_ground_truth(manifest,changes); vol_all=C.compute_observed_domain_volatility(gt)
gt2=C.build_ground_truth(mf2,ch2); vol_no24=C.compute_observed_domain_volatility(gt2)
changed_dom=sum(1 for d in vol_all if d in vol_no24 and vol_all[d]!=vol_no24[d]); only_all=len(set(vol_all)-set(vol_no24))
say(f"\n[5] OBSERVED volatility lookup (1h+24h in code): {len(vol_all)} domains; without 24h: {len(vol_no24)}; values changed {changed_dom}, domains lost {only_all}")
say("    -> this lookup DOES depend on the 24h round. But is it what the filter reads?")
# --- 6. what the filter actually reads: the hand dict; compare to observed values for the same domains ---
hd=RM._DEFAULT_DOMAIN_VOLATILITY
overlap=[(d,hd[d],vol_all.get(d)) for d in hd if d in vol_all]
same=sum(1 for d,h,o in overlap if o is not None and abs(h-o)<1e-9)
say(f"[6] FILTER dict (_DEFAULT_DOMAIN_VOLATILITY): {len(hd)} hand-set entries; {len(overlap)} also have observed values; {same} equal the observed value")
for d,h,o in overlap[:12]: say(f"     {d:<24} dict {h:<5} observed(1h+24h) {o}")
# --- 7. what if the filter used the OBSERVED lookup (as calibrate.py's patch intends)? and fit-rounds-only lookup? ---
def make_filter(lookup):
    def f(domain,fc):
        domain=(domain or "").lower().removeprefix("www."); vol=lookup.get(domain,RM._DEFAULT_DOMAIN_VOL)
        if vol==RM._DEFAULT_DOMAIN_VOL:
            for known,v in lookup.items():
                if domain.endswith("."+known): vol=v; break
        return vol>RM.CLASS_VOL_CEILING.get(fc,0.50)
    return f
def vol_from_rounds(mf,ch,rounds):
    obs=collections.defaultdict(list); chset={(c["url_hash"],c["run_id"]) for c in ch if c.get("detected_at","")>C.NEW_RUN_CUTOFF and not c.get("noise")}
    for m in mf:
        if m.get("run_id") in rounds and m.get("snapshot_available"):
            obs[m.get("domain","").lower().removeprefix("www.")].append(int((m["url_hash"],m["run_id"]) in chset))
    return {d:round(sum(o)/len(o),3) for d,o in obs.items() if len(o)>=2}
vol_fit_only=vol_from_rounds(manifest,changes,{"rerun_1h","rerun_12h"})
vol_four=vol_from_rounds(manifest,changes,{"rerun_1h","rerun_12h","rerun_24h","rerun_7d"})
res={}
for label,filt in [("hand_dict (in force)",RM.is_noise_url),("no_filter",lambda d,fc:False),
                   ("observed_1h+24h (calibrate.py patch intent)",make_filter(vol_all)),
                   ("fit_rounds_only_1h+12h (control)",make_filter(vol_fit_only)),
                   ("all_four_rounds (manuscript wording)",make_filter(vol_four))]:
    r,t,rmv,_=rates_from(manifest,changes,filt); f=fit(r)
    res[label]={fc:{"half_life_h":hours(f[fc][0]),"removed_fit_window":sum(rmv[(w,fc)] for w in C.FIT_WINDOWS_BY_CLASS[fc]),
                    "tracked_fit_window":sum(t[(w,fc)] for w in C.FIT_WINDOWS_BY_CLASS[fc])} for fc in CLASSES}
say("\n[7] FILTER VARIANTS, same estimator, same inputs (half-life h; removed/tracked over the class's fit windows):")
for label,d in res.items():
    say(f"   {label}")
    for fc in CLASSES: say(f"      {fc:<9} {d[fc]['half_life_h']!s:>9} h   removed {d[fc]['removed_fit_window']:>5} / tracked {d[fc]['tracked_fit_window']}")
json.dump({"baseline_hand_dict":{fc:hours(base[fc][0]) for fc in CLASSES},
           "later_rounds_removed":{fc:hours(f2[fc][0]) for fc in CLASSES},
           "later_rounds_perturbed":{fc:hours(f3[fc][0]) for fc in CLASSES},
           "positive_control_12h_removed":{fc:hours(f4[fc][0]) for fc in CLASSES},
           "observed_lookup":{"n_domains_1h24h":len(vol_all),"n_domains_without_24h":len(vol_no24),"values_changed":changed_dom,"domains_lost":only_all},
           "filter_dict_vs_observed":{"n_dict":len(hd),"n_overlap":len(overlap),"n_equal":same},
           "filter_variants":res,
           "deployed_half_life_h":{"TIMELESS":720,"SLOW":276,"MEDIUM":127.2,"FAST":124.8},
           "fit_windows":C.FIT_WINDOWS_BY_CLASS,"run_ages_s":RUN_AGES},open(OUT/"results.json","w"),indent=1)
say("\nwrote results.json")
