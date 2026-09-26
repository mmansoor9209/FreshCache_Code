"""Observability audit: recompute every population from row-level records. Read-only; writes only into audit/observability/."""
import json, csv, re, math, pathlib, sys, os, collections, glob, datetime
ROOT = pathlib.Path("<PROJECT_ROOT>"); OUT = pathlib.Path(__file__).parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import experiment as exp
from freshcache.risk_model import is_noise_url
try:
    from scipy.stats import chi2_contingency, chi2 as chi2d; HAVE_SCIPY = True
except Exception: HAVE_SCIPY = False
RUNS = ["run_00","rerun_1h","rerun_12h","rerun_24h","rerun_7d"]; MIN_BODY = 400
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|access denied|are you a robot|captcha|cloudflare|403 forbidden|404 not found|page not found", re.I)
CLASSES = ["TIMELESS","SLOW","MEDIUM","FAST","REAL_TIME"]
R = {"utc": datetime.datetime.utcnow().isoformat()+"Z"}
def say(*a): print(*a, flush=True)

# ---- 1. rebuild substantive flags straight from the snapshot files ----
snap = collections.defaultdict(dict); status = collections.Counter(); reasons = collections.Counter()
for run in RUNS:
    for f in glob.glob(f"data/snapshots/{run}/*.json"):
        try: s = json.load(open(f))
        except Exception: continue
        t = s.get("extracted_text") or ""; sc = s.get("status_code")
        sub = bool(t and len(t.strip()) >= MIN_BODY and not BLOCK.search(t[:400]))
        snap[pathlib.Path(f).stem][run] = {"sub": sub, "status": sc, "chars": len(t)}
        status[(run, sc)] += 1
        if sub: reasons["substantive"] += 1
        elif sc != 200: reasons[f"non_200 ({sc})"] += 1
        elif len(t.strip()) == 0: reasons["empty_extraction"] += 1
        elif len(t.strip()) < MIN_BODY: reasons["short_text_under_400"] += 1
        else: reasons["block_page_pattern"] += 1
tl = {json.loads(l)["url_hash"]: json.loads(l) for l in open("v9/version_timeline.jsonl")}
mism = sum(1 for u in tl for r in tl[u]["runs"] if tl[u]["runs"][r]["sub"] != snap[u][r]["sub"])
n_obs_total = sum(len(v) for v in snap.values())
common24 = {u for u in snap if snap[u].get("run_00",{}).get("sub") and snap[u].get("rerun_24h",{}).get("sub")}
sub_status_not200 = sum(1 for u in snap for r in snap[u] if snap[u][r]["sub"] and snap[u][r]["status"] != 200)
R["url_level"] = {"unique_urls_with_any_snapshot": len(snap), "v9_timeline_urls": len(tl),
    "url_round_observations": n_obs_total, "v9_flag_mismatches_vs_rebuild": mism,
    "reason_taxonomy": dict(reasons), "status_codes_by_run": {f"{k[0]}:{k[1]}": v for k, v in sorted(status.items(), key=str)},
    "substantive_but_non200": sub_status_not200,
    "urls_with_run00_and_rerun24h_snapshot": sum(1 for u in snap if "run_00" in snap[u] and "rerun_24h" in snap[u]),
    "body_observable_24h_urls": len(common24), "reported": "4,382 of 8,635",
    "rule_actually_coded": "len(strip(extracted_text)) >= 400 AND no block-page regex in first 400 chars; HTTP status is NOT tested in v9/build_version_timeline.py:46-49"}
say("URL level:", json.dumps(R["url_level"], indent=1))
# v13 round table agreement
rt = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open("v13_corrected/corrected_round_table.jsonl")}
common24_rt = {u for u, rr in rt.items() if rr.get("run_00",{}).get("substantive") and rr.get("rerun_24h",{}).get("substantive")}
R["v13_round_table"] = {"urls": len(rt), "url_round_entries": sum(len(v) for v in rt.values()),
    "body_observable_24h_urls": len(common24_rt), "identical_set_to_v9": common24_rt == common24,
    "entries_disagreeing_with_v9_sub": sum(1 for u in rt for r in rt[u] if u in tl and r in tl[u]["runs"] and bool(rt[u][r].get("substantive")) != tl[u]["runs"][r]["sub"])}
say("v13:", R["v13_round_table"])

# ---- 2. request pool ----
manifest = exp.load_jsonl(exp.MANIFEST_FILE); queries = exp.load_jsonl(exp.QUERIES_FILE)
paras = exp.load_jsonl(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else []
records = exp.build_query_records(queries, manifest, paras)
kept_any = [r for r in records if any(u["url_hash"] in common24 for u in r["urls"])]
kept_all = [r for r in records if r["urls"] and all(u["url_hash"] in common24 for u in r["urls"])]
no_url = sum(1 for r in records if not r["urls"])
missing_snap = sum(1 for r in records for u in r["urls"] if u["url_hash"] not in snap)
bycls = {}
for fc in CLASSES:
    rs = [r for r in records if r["freshness_class"] == fc]
    bycls[fc] = {"requests": len(rs), "kept_any_url": sum(1 for r in rs if any(u["url_hash"] in common24 for u in r["urls"])),
                 "kept_all_urls": sum(1 for r in rs if r["urls"] and all(u["url_hash"] in common24 for u in r["urls"]))}
    bycls[fc]["kept_any_pct"] = round(100*bycls[fc]["kept_any_url"]/len(rs), 2)
R["request_level"] = {"records": len(records), "reported": "18,216 of 31,201", "kept_if_ANY_url_observable": len(kept_any),
    "kept_if_ALL_urls_observable": len(kept_all), "records_with_no_url": no_url, "url_refs_with_no_snapshot_at_all": missing_snap,
    "urls_per_record": dict(collections.Counter(len(r["urls"]) for r in records)), "includes_REAL_TIME": True,
    "by_class": bycls, "rule": "record kept when at least ONE of its URLs is body observable at both run_00 and rerun_24h (v10_remaining_feedback/e1_c1_observable_refit.py:194; validation/observability_by_class/analyze.py:259)"}
say("requests:", json.dumps(R["request_level"], indent=1))

# ---- 3. class table at three units ----
def chi(tab):
    N = sum(map(sum, tab)); rs = [sum(r) for r in tab]; cs = [sum(t[j] for t in tab) for j in (0,1)]
    E = [[rs[i]*cs[j]/N for j in (0,1)] for i in range(len(tab))]
    x = sum((tab[i][j]-E[i][j])**2/E[i][j] for i in range(len(tab)) for j in (0,1)); df = len(tab)-1
    p = chi2d.sf(x, df) if HAVE_SCIPY else None
    return {"N": N, "chi2": round(x, 4), "df": df, "p": p, "cramers_v": round(math.sqrt(x/N), 4), "min_expected": round(min(map(min, E)), 1)}
tracked = [m for m in manifest if m.get("snapshot_available") and m.get("run_id") == "rerun_24h"]
filt = [m for m in tracked if m["freshness_class"] in CLASSES and not is_noise_url(m.get("domain",""), m["freshness_class"])]
def obs(u): return u in common24
units = {}
# (a) manifest rows (query x URL) as the study did
rows_tab = []; per = {}
for fc in CLASSES:
    g = [m for m in filt if m["freshness_class"] == fc]; o = sum(1 for m in g if obs(m["url_hash"]))
    per[fc] = {"tracked_rows": len(g), "observable": o, "not_observable": len(g)-o, "pct": round(100*o/len(g), 2),
               "unique_urls_in_rows": len({m["url_hash"] for m in g}), "unique_queries": len({m["query_id"] for m in g})}
    rows_tab.append([o, len(g)-o])
units["a_manifest_rows_query_x_url"] = {"per_class": per, "chi": chi(rows_tab), "reported": "chi2 121.53 df 4 p 2.5e-25 V 0.084; TIMELESS 58.19 SLOW 57.09 MEDIUM 60.82 FAST 58.53 REAL_TIME 73.27"}
# (b) unique (url, class)
uc = {}; 
for m in filt: uc[(m["url_hash"], m["freshness_class"])] = 1
per = {}; tab = []
for fc in CLASSES:
    us = {u for (u, c) in uc if c == fc}; o = sum(1 for u in us if obs(u))
    per[fc] = {"unique_urls": len(us), "observable": o, "not_observable": len(us)-o, "pct": round(100*o/len(us), 2)}; tab.append([o, len(us)-o])
units["b_unique_url_class_pairs"] = {"per_class": per, "chi": chi(tab), "sum_of_class_urls": sum(p["unique_urls"] for p in per.values())}
# (c) unique URL, single-class only (multi-class URLs excluded)
cls_of = collections.defaultdict(set)
for (u, c) in uc: cls_of[u].add(c)
single = {u: next(iter(c)) for u, c in cls_of.items() if len(c) == 1}; multi = {u for u, c in cls_of.items() if len(c) > 1}
per = {}; tab = []
for fc in CLASSES:
    us = [u for u, c in single.items() if c == fc]; o = sum(1 for u in us if obs(u))
    per[fc] = {"unique_urls": len(us), "observable": o, "not_observable": len(us)-o, "pct": round(100*o/len(us), 2)}; tab.append([o, len(us)-o])
units["c_unique_url_single_class"] = {"per_class": per, "chi": chi(tab), "multi_class_urls_excluded": len(multi),
    "multi_class_observable": sum(1 for u in multi if obs(u))}
R["partition"] = {"urls_tracked_at_rerun_24h_after_noise_filter": len(cls_of), "of_8635_timeline_urls": len(tl),
    "timeline_urls_not_in_any_class_row": len(set(tl) - set(cls_of)), "urls_in_more_than_one_class": len(multi),
    "class_sets_partition_url_pool": len(multi) == 0 and len(set(tl) - set(cls_of)) == 0,
    "observable_urls_covered_by_class_rows": len(common24 & set(cls_of)), "of_4382": len(common24)}
R["class_units"] = units
say("class units:", json.dumps(units, indent=1)); say("partition:", R["partition"])

# ---- 4. paired missingness ----
pr = list(csv.DictReader(open("validation_V3/observability_resolution/02_recovery/per_request_recovered.csv")))
def score(on, off):
    pp = [r for r in pr if r[on] and r[off]]; jo = [r for r in pp if r[on] != "UNOBSERVABLE" and r[off] != "UNOBSERVABLE"]
    return {"stream_rows": len(pr), "paired": len(pp), "jointly_observable": len(jo), "coverage_pct": round(100*len(jo)/len(pp), 2),
            "unobs_on": sum(1 for r in pp if r[on] == "UNOBSERVABLE"), "unobs_off": sum(1 for r in pp if r[off] == "UNOBSERVABLE"),
            "drift_on_pct": round(100*sum(1 for r in jo if r[on] == "CHANGED")/len(jo), 4), "drift_off_pct": round(100*sum(1 for r in jo if r[off] == "CHANGED")/len(jo), 4)}
R["paired"] = {"thr400": score("outcome_on_thr400", "outcome_off_thr400"), "thr50": score("outcome_on_thr50", "outcome_off_thr50"),
    "reported": "54.62% -> 79.68%", "unit": "held-out replay request (21,848 stream; paired = both arms emit an outcome, i.e. non-REAL_TIME with a comparison snapshot)"}
say("paired:", json.dumps(R["paired"], indent=1))
json.dump(R, open(OUT/"observability_results.json", "w"), indent=2, default=str)
# count tables
with open(OUT/"class_count_tables.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["unit","class","n","observable","not_observable","observable_pct"])
    for k, v in units.items():
        for fc, p in v["per_class"].items(): w.writerow([k, fc, p.get("tracked_rows", p.get("unique_urls")), p["observable"], p["not_observable"], p["pct"]])
    for k, v in units.items(): w.writerow([k, "CHI2", v["chi"]["N"], v["chi"]["chi2"], v["chi"]["p"], v["chi"]["cramers_v"]])
say("HAVE_SCIPY", HAVE_SCIPY)
