"""Benchmark-construction sensitivity. (1) same- vs cross-cluster decomposition of the frozen FreshCache reuse; (2) base-only replays
(one original request per cluster) with the frozen engine and settings; (3) comparison + pre-registered verdict."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, glob, hashlib, json, os, pathlib, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); CRR = (OUTPUT_ROOT / "replay"); W = (OUTPUT_ROOT / "benchmark_locality")
for d in ("results", "logs", "frozen"): (W / d).mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); sys.path.insert(0, str(ROOT / "freshcache/simulation"))
import engine_loader as C, provenance_replay as PR   # noqa: E402
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
PRE = {"written_utc": now(), "base_only_definition": "requests with is_paraphrase == False (exactly one per retained cluster), original arrival times/order, own URL list, class; fresh cache",
       "verdict_rule": {"metric": "search savings of the base-only replay relative to the corresponding full replay (ratio base/full), primary stream; held-out reported alongside",
                        "a_largely_retained": "ratio >= 0.80", "b_reduced_but_substantial": "0.40 <= ratio < 0.80", "c_strongly_dependent": "ratio < 0.40"},
       "caveat": "base-only replays have ~5x fewer requests over the same time horizon (lower density); diagnostic only, no claim about production traffic"}
PRE["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in PRE.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (W / "frozen/prereg.json").exists(): json.dump(PRE, open(W / "frozen/prereg.json", "w"), indent=1)
PRE = json.load(open(W / "frozen/prereg.json"))
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
INP = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/method_replay.py",
       "freshcache/replay_engine/execution_semantics.py", "freshcache/replay_engine/engine_loader.py", "freshcache/method/tiered_cache_model.py", "freshcache/simulation/page_version_replay.py", "freshcache/simulation/schedule_robustness.py",
       "freshcache/simulation/request_schedules.py", "data/derived/round_table.jsonl", "data/queries.jsonl", "data/url_manifest.jsonl", "data/paraphrase_clusters.jsonl", "data/sim_matrix_bgem3.npy",
       "outputs/validation_split/split.json", "outputs/replay/results/replays_reported.json", "outputs/replay/results/replays_reported_per_request.pkl",
       "outputs/replay/table2/reported/traces.pkl"]
INT = {p: {"sha256": sha(inpath(p)), "freeze": FH.get(p)} for p in INP}
for v in INT.values(): v["match"] = (v["sha256"] == v["freeze"]) if v["freeze"] else None
if any(v["match"] is False for v in INT.values()): json.dump(INT, open(W / "logs/STOPPED.json", "w"), indent=1); sys.exit("STOP integrity")
E = C.load("reported"); ST = C.streams(E); FR = json.load(open(CRR / "results/replays_reported.json")); PK = pickle.load(open(CRR / "results/replays_reported_per_request.pkl", "rb"))
G = {}
full = {}
for s in ("primary", "heldout"):
    S = E["support_of"](ST[s], E["rounds"]); row, per, tr = C.metrics(E, ST[s], "FreshCache", S); full[s] = row
    G[f"{s}_full_FreshCache_equals_frozen"] = row == FR[f"{s}/FreshCache"]
G["primary_saved_and_drift_equal_reported_replay"] = (full["primary"]["search_saved_pct"], full["primary"]["drift_pct"]) == (FR["primary/FreshCache"]["search_saved_pct"], FR["primary/FreshCache"]["drift_pct"])
PROV = {}
for s in ("primary", "heldout"):
    rows = PR.run(E, ST[s]); PROV[s] = rows
    G[f"{s}_provenance_transcription_tiers_equal_frozen"] = all(rows[r["query_id"]]["tier"] == PK[f"{s}/FreshCache"]["tier"][r["query_id"]] for _, r in ST[s])
    G[f"{s}_provenance_fetches_equal_frozen"] = sum(x["fetched"] for x in rows.values()) == FR[f"{s}/FreshCache"]["fetches"]
    G[f"{s}_provenance_l3_reuses_equal_frozen"] = sum(len(x.get("l3_reuse_writers", [])) for x in rows.values()) == FR[f"{s}/FreshCache"]["l3_hits"]
json.dump({"gates": G, "integrity": INT, "prereg_sha256": PRE["sha256"]}, open(W / "logs/gates.json", "w"), indent=1)
if not all(G.values()): sys.exit(f"STOP gates {[k for k, v in G.items() if not v]}")
# ---------------- 1. decomposition
REC = {r["query_id"]: r for r in E["recs"]}; CL = lambda q: REC[q].get("cluster_id") or q
DEC = []
for s in ("primary", "heldout"):
    rows = PROV[s]; n = len(ST[s]); c = collections.Counter()
    for _, r in ST[s]:
        q = r["query_id"]; x = rows[q]; me_ = CL(q)
        if x["tier"] == "L1":
            rel = "same" if CL(x["l1_entry_creator"]) == me_ else "cross"; c[f"L1_hits_{rel}"] += 1; c[f"search_avoided_{rel}"] += 1; c[f"generation_avoided_{rel}"] += 1
            c[f"fetch_reduction_L1_{rel}"] += x["own"]
        elif x["tier"] == "L2":
            rel = "same" if CL(x["l2_entry_creator"]) == me_ else "cross"; relo = "same" if CL(x["l2_list_origin"]) == me_ else "cross"
            c[f"L2_hits_{rel}"] += 1; c[f"search_avoided_{rel}"] += 1; c[f"L2_list_origin_{relo}"] += 1; c[f"fetch_reduction_L2_list_length_delta_{rel}"] += x["own"] - x["served"]
        if x["tier"] in ("L2", "miss"):
            for w in x["l3_reuse_writers"]:
                rel = "same_request_duplicate" if w == q else ("same" if CL(w) == me_ else "cross"); c[f"L3_reuses_{rel}"] += 1; c[f"fetch_reduction_L3_{rel}"] += 1
    nocache_fetches = sum(len(r["urls"]) for _, r in ST[s]); fc = FR[f"{s}/FreshCache"]
    red = nocache_fetches - fc["fetches"]; parts = sum(v for k, v in c.items() if k.startswith("fetch_reduction"))
    G[f"{s}_fetch_reduction_components_sum_exactly"] = parts == red
    sa = n - fc["search_calls"] if "search_calls" in fc else n - round(n * (1 - fc["search_saved_pct"] / 100))
    for metric, keys, total in (("search avoidance (L1+L2 hits)", ("search_avoided",), fc["l1_hits"] + fc["l2_hits"]), ("L1 hits", ("L1_hits",), fc["l1_hits"]),
                                ("L2 hits (entry used)", ("L2_hits",), fc["l2_hits"]), ("L2 hits (original list discoverer)", ("L2_list_origin",), fc["l2_hits"]),
                                ("L3 reuses (page writer)", ("L3_reuses",), fc["l3_hits"]), ("generation reduction (L1 hits)", ("generation_avoided",), fc["l1_hits"]),
                                ("fetch reduction vs NoCache", ("fetch_reduction_L1", "fetch_reduction_L2_list_length_delta", "fetch_reduction_L3"), red)):
        same = sum(c[f"{k}_same"] for k in keys); cross = sum(c[f"{k}_cross"] for k in keys); dup = sum(c.get(f"{k}_same_request_duplicate", 0) for k in keys)
        DEC.append({"stream": s, "metric": metric, "total": total, "same_cluster": same, "cross_cluster": cross, "same_request_duplicate": dup,
                    "same_cluster_pct": round(100 * same / total, 2) if total else None, "cross_cluster_pct": round(100 * cross / total, 2) if total else None,
                    "components_sum_to_total": same + cross + dup == total})
    DEC.append({"stream": s, "metric": "fetch reduction detail", "total": red, "same_cluster": json.dumps({k: v for k, v in c.items() if k.startswith("fetch_reduction")}), "cross_cluster": "", "same_request_duplicate": "",
                "same_cluster_pct": "", "cross_cluster_pct": "", "components_sum_to_total": parts == red})
G["decomposition_components_sum"] = all(r["components_sum_to_total"] for r in DEC)
# ---------------- 2. base-only replays (frozen engine, fresh cache)
BASE = {}
for s in ("primary", "heldout"):
    bs = [(t, r) for t, r in ST[s] if not r.get("is_paraphrase")]
    assert DS.check(f"{s}_base_requests", len(bs)) and len({CL(r["query_id"]) for _, r in bs}) == len(bs)
    Sb = E["support_of"](bs, E["rounds"]); row, per, tr = C.metrics(E, bs, "FreshCache", Sb)
    row["search_calls"] = sum(1 for q in tr if tr[q]["tier"] in ("miss", "REAL_TIME_BYPASS")); row["determinable"] = row["changed"] + row["unchanged"]
    rowsb = PR.run(E, bs); G[f"{s}_base_transcription_tiers_equal_engine"] = all(rowsb[q]["tier"] == tr[q]["tier"] for q in tr)
    G[f"{s}_base_all_reuse_cross_cluster"] = all((x["tier"] != "L1" or CL(x["l1_entry_creator"]) != CL(q)) and (x["tier"] != "L2" or CL(x["l2_entry_creator"]) != CL(q)) for q, x in rowsb.items())
    BASE[s] = row
    with open(W / "results" / f"base_only_{s}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "cluster_id", "t", "freshness_class", "tier", "drift_outcome", "in_support", "url_count"])
        for t, r in bs: w.writerow([r["query_id"], CL(r["query_id"]), t, r["freshness_class"], tr[r["query_id"]]["tier"], per.get(r["query_id"]), r["query_id"] in Sb, len(r["urls"])])
    json.dump(row, open(W / "results" / f"base_only_{s}_metrics.json", "w"), indent=1)
json.dump({"gates": G, "integrity": INT, "prereg_sha256": PRE["sha256"]}, open(W / "logs/gates.json", "w"), indent=1)
if not all(G.values()): sys.exit(f"STOP gates {[k for k, v in G.items() if not v]}")
def wcsv(n, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(W / "results" / n, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
wcsv("full_trace_cluster_decomposition.csv", DEC)
# ---------------- 3. comparison + verdict
CMP = []
for name, m, s in (("original full replay", full["primary"], "primary"), ("base-only replay", BASE["primary"], "primary"), ("held-out full replay", full["heldout"], "heldout"),
                   ("held-out base-only replay", BASE["heldout"], "heldout")):
    CMP.append({"replay": name, "requests": m["n"], "search_saved_pct": m["search_saved_pct"], "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
                "fetch_per_1k": m["fetch_per_1k"], "gen_per_1k": m["gen_per_1k"], "drift_pct": m["drift_pct"], "coverage_pct": m["coverage_pct"],
                "determinable": m["changed"] + m["unchanged"], "changed": m["changed"], "support": m["support"]})
wcsv("comparison.csv", CMP)
ratio = {s: BASE[s]["search_saved_pct"] / full[s]["search_saved_pct"] for s in ("primary", "heldout")}
v = "a" if ratio["primary"] >= .8 else ("b" if ratio["primary"] >= .4 else "c")
json.dump({"gates": G, "prereg_sha256": PRE["sha256"], "search_savings_ratio_base_over_full": ratio, "verdict": v, "decomposition": DEC, "comparison": CMP, "finished": now()},
          open(W / "results/verdict.json", "w"), indent=1, default=str)
print("GATES", all(G.values())); print("ratio", ratio, "verdict", v)
for r in CMP: print(r)
for r in DEC: print(r)
