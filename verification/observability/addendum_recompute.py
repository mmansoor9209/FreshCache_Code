"""Addendum to the observability audit. Read-only; writes only into audit/observability/."""
import json, csv, re, glob, os, sys, pathlib, collections, math, datetime
ROOT = pathlib.Path("<PROJECT_ROOT>"); OUT = pathlib.Path(__file__).parent
os.chdir(ROOT)
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12", "validation/mixed_age_full_policy_audit"):
    sys.path.insert(0, str(ROOT/p) if p else str(ROOT))
import experiment as exp, schedules as sc
from e1_robustness import support_of
R = {"utc": datetime.datetime.utcnow().isoformat()+"Z"}
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|access denied|are you a robot|captcha|cloudflare|403 forbidden|404 not found|page not found", re.I)
# ---- 1. mutually exclusive snapshot taxonomy per round ----
ROUNDS = ["run_00","rerun_1h","rerun_12h","rerun_24h","rerun_7d","rerun_48h"]
tax = {}
for run in ROUNDS:
    c = collections.Counter()
    for f in glob.glob(f"data/snapshots/{run}/*.json"):
        s = json.load(open(f)); t = (s.get("extracted_text") or ""); ts = t.strip()
        if s.get("status_code") != 200: c["non_200"] += 1
        elif len(ts) >= 400 and not BLOCK.search(t[:400]): c["substantive"] += 1
        elif len(ts) >= 400: c["block_page_400plus"] += 1
        elif len(ts) == 0: c["empty"] += 1
        else: c["short_text_under_400"] += 1   # block regex not applied here by v9 substantive() (it returns False on length first)
    n = sum(c.values()); tax[run] = {"snapshots": n, **dict(c), "non_substantive_pct": round(100*(n-c["substantive"])/n, 4), "short_pct": round(100*c["short_text_under_400"]/n, 4)}
def agg(runs):
    c = collections.Counter()
    for r in runs:
        for k, v in tax[r].items():
            if k not in ("non_substantive_pct","short_pct"): c[k] += v
    n = c["snapshots"]; return {**dict(c), "non_substantive_pct": round(100*(n-c["substantive"])/n, 4), "short_pct": round(100*c["short_text_under_400"]/n, 4), "block_pct": round(100*c.get("block_page_400plus",0)/n, 4)}
R["snapshot_taxonomy"] = {"per_round": tax, "five_named_rounds": agg(ROUNDS[:5]), "six_rounds_incl_48h": agg(ROUNDS),
    "claims_to_verify": {"baseline_non_sub_47.20": tax["run_00"]["non_substantive_pct"], "five_round_non_sub_47.63": agg(ROUNDS[:5])["non_substantive_pct"],
                         "six_round_non_sub_47.62": agg(ROUNDS)["non_substantive_pct"], "six_round_short_only_47.24": agg(ROUNDS)["short_pct"]}}
print("TAX", json.dumps(R["snapshot_taxonomy"]["claims_to_verify"]), json.dumps(R["snapshot_taxonomy"]["five_named_rounds"]))
# ---- 2. URLs per request ----
manifest = exp.load_jsonl(exp.MANIFEST_FILE); queries = exp.load_jsonl(exp.QUERIES_FILE)
paras = exp.load_jsonl(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else []
records = exp.build_query_records(queries, manifest, paras)
base = [m for m in manifest if m.get("run_id") == "run_00"]
rows_per_q = collections.Counter(m["query_id"] for m in base)
avail_per_q = collections.Counter(m["query_id"] for m in base if m.get("snapshot_available"))
rank_dist = collections.Counter(m.get("rank") for m in base)
dup_hash = sum(1 for q in avail_per_q if len({m["url_hash"] for m in base if m["query_id"] == q and m.get("snapshot_available")}) < avail_per_q[q])
sizes = collections.Counter(len(r["urls"]) for r in records if not r["is_paraphrase"])
sizes_all = collections.Counter(len(r["urls"]) for r in records)
ex = []
for r in records:
    if not r["is_paraphrase"] and len(r["urls"]) >= 3 and len(ex) < 4:
        rows = [m for m in base if m["query_id"] == r["query_id"]]
        ex.append({"query_id": r["query_id"], "query": r["query"][:80], "n_urls": len(r["urls"]),
                   "run_00_rows": [f"rank {m.get('rank')} hash {m['url_hash'][:8]} at {m.get('retrieval_timestamp','')[:19]} avail={bool(m.get('snapshot_available'))}" for m in sorted(rows, key=lambda m: (m.get("retrieval_timestamp",""), m.get("rank")))]})
ts_per_q = collections.Counter(len({m.get("retrieval_timestamp") for m in base if m["query_id"] == q}) for q in rows_per_q)
R["urls_per_request"] = {"manuscript": "freshcache_pro.tex:672 'one Serper search returns at most two URLs'",
    "run_00_manifest_rows_per_query": dict(rows_per_q.__class__(collections.Counter(rows_per_q.values()))),
    "run_00_available_rows_per_query": dict(collections.Counter(avail_per_q.values())), "rank_values_at_run_00": dict(rank_dist),
    "queries_with_duplicate_url_hash_among_available_rows": dup_hash, "distinct_retrieval_timestamps_per_query_at_run_00": dict(ts_per_q),
    "record_url_list_size_base_queries": dict(sizes), "record_url_list_size_all_requests": dict(sizes_all),
    "records_whose_url_list_repeats_a_url_hash": sum(1 for r in records if not r["is_paraphrase"] and len({u["url_hash"] for u in r["urls"]}) < len(r["urls"])),
    "distinct_hash_list_size_base_queries": dict(collections.Counter(len({u["url_hash"] for u in r["urls"]}) for r in records if not r["is_paraphrase"])),
    "code_path": "experiment.build_query_records:503-518 takes EVERY run_00 manifest row with snapshot_available for the query, sorted by rank; no cap; paraphrases inherit the base list (:541-556)", "examples_3plus": ex}
print("URLS", json.dumps({k: v for k, v in R["urls_per_request"].items() if k != "examples_3plus"})); print(json.dumps(ex, indent=1))
# ---- 3. paired 16,813 vs common support 12,568 ----
rounds = {}
for line in open("v13_corrected/corrected_round_table.jsonl"):
    d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
split = json.load(open("validation/heldout_baseline_tuning/split.json")); test_c = set(split["test_clusters"])
full = sc.build_stream(records, "zipf_uniform", 42)
stream = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in test_c]
S = support_of(stream, rounds)
pr = list(csv.DictReader(open("validation_V3/observability_resolution/02_recovery/per_request_recovered.csv")))
paired = {r["query_id"] for r in pr if r["outcome_on_thr400"] and r["outcome_off_thr400"]}
jo = {r["query_id"] for r in pr if r["outcome_on_thr400"] and r["outcome_off_thr400"] and "UNOBSERVABLE" not in (r["outcome_on_thr400"], r["outcome_off_thr400"])}
rec = {r["query_id"]: r for r in records}
def cls(ids): return dict(collections.Counter(rec[q]["freshness_class"] for q in ids))
onlyP = paired - S; onlyS = S - paired
R["paired_vs_support"] = {"stream_requests": len(stream), "csv_rows": len(pr), "support_S": len(S), "paired": len(paired), "intersection": len(paired & S),
    "paired_only": len(onlyP), "support_only": len(onlyS), "jointly_observable_400": len(jo), "jo_subset_of_S": jo <= S, "S_subset_of_paired": S <= paired,
    "paired_only_by_class": cls(onlyP), "support_only_by_class": cls(onlyS), "paired_by_class": cls(paired), "S_by_class": cls(S),
    "paired_only_outcomes": {f"{k[0]}|{k[1]}": v for k, v in collections.Counter((r["outcome_on_thr400"], r["outcome_off_thr400"]) for r in pr if r["query_id"] in onlyP).items()},
    "paired_only_has_run00_and_arrival_substantive_url": sum(1 for t, r in stream if r["query_id"] in onlyP and any(rounds.get(u["url_hash"],{}).get("run_00",{}).get("substantive") and rounds.get(u["url_hash"],{}).get(__import__("mixed_age_v2").version_at(t),{}).get("substantive") for u in r["urls"])),
    "rules": {"S": "e1_robustness.support_of:50-61: non-REAL_TIME request with >=1 URL substantive at run_00 AND at version_at(arrival t); same zipf_uniform seed-42 stream, same corrected_round_table",
              "paired": "recover_and_recompute.py:152-153: both arms' mixed_engine.replay per-request outcome is not None (CHANGED/UNCHANGED/UNOBSERVABLE)"}}
print("PAIRED", json.dumps(R["paired_vs_support"], indent=1))
# ---- 4. corrected deduplicated arithmetic ----
prev = json.load(open(OUT/"observability_results.json"))["class_units"]
fix = {}
for k, v in prev.items():
    pc = {c: 100*p["observable"]/p.get("tracked_rows", p.get("unique_urls")) for c, p in v["per_class"].items()}
    N = v["chi"]["N"]; fix[k] = {"chi2": v["chi"]["chi2"], "N": N, "cramers_v_6dp": round(math.sqrt(v["chi"]["chi2"]/N), 6),
        "mean_FAST_REALTIME_minus_mean_TIMELESS_SLOW_pp": round((pc["FAST"]+pc["REAL_TIME"])/2 - (pc["TIMELESS"]+pc["SLOW"])/2, 6), "pct_unrounded": {c: round(x, 6) for c, x in pc.items()}}
R["corrected_arithmetic"] = fix; print("FIX", json.dumps(fix, indent=1))
json.dump(R, open(OUT/"addendum_results.json", "w"), indent=2, default=str)
