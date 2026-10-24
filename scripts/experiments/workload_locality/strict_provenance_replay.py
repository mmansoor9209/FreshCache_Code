"""Strict-provenance base-only replay (Sec. 5.1: standalone FreshCache savings on held-out originals, including the part via L2 lists
sharing no URL with the question's own list). Frozen FreshCache primary configuration (theta_L2 = 0.75; eps L1/L2/L3 = 0.10/0.20/0.35), frozen
engine, frozen order/timestamps; the primary and held-out ORIGINAL (is_paraphrase == False) requests, exactly as the base-only
replay of cluster_locality_decomposition.py and condition A of paraphrase_share_sensitivity.py.
  * Official operation metrics come from the frozen engine (engine_loader.metrics -> mixed_age_replay.replay), unchanged.
  * Provenance comes from an IN-MEMORY copy of freshcache/simulation/provenance_replay.py (file not modified) with bookkeeping-only
    additions: the URL list each request served/stored (L2 entries, L1 entries). No decision reads the additions; the copy's per-request
    tiers, fetches and L3 reuses are gated against the frozen engine.
  * Reproduction gate against the frozen base-only values; on failure writes REPRODUCTION_FAILURE.md and stops before any analysis.
Outputs in outputs/strict_provenance/ (results.json, provenance_hits.csv, run.log). No web data, no new queries/URL lists/labels, no tuning."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, hashlib, json, os, pathlib, statistics, subprocess, sys, types; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "strict_provenance"); O.mkdir(parents=True, exist_ok=True)
BASE_ONLY_DIR = (OUTPUT_ROOT / "benchmark_locality"); now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
def loc(p): return DATA_ROOT / p[5:] if p.startswith("data/") else OUTPUT_ROOT / p[8:] if p.startswith("outputs/") else ROOT / p
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
INPUTS = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/method_replay.py",
          "freshcache/replay_engine/execution_semantics.py", "freshcache/replay_engine/engine_loader.py", "freshcache/method/tiered_cache_model.py", "freshcache/simulation/page_version_replay.py", "freshcache/simulation/schedule_robustness.py",
          "freshcache/simulation/request_schedules.py", "data/derived/round_table.jsonl", "data/queries.jsonl", "data/url_manifest.jsonl", "data/paraphrase_clusters.jsonl", "data/sim_matrix_bgem3.npy",
          "outputs/validation_split/split.json", "freshcache/simulation/provenance_replay.py",
          "outputs/benchmark_locality/results/base_only_primary.csv", "outputs/benchmark_locality/results/base_only_primary_metrics.json",
          "outputs/benchmark_locality/results/base_only_heldout.csv", "outputs/benchmark_locality/results/base_only_heldout_metrics.json",
          "outputs/paraphrase_share/work/runs.json", "configs/integrity/FILE_HASHES.csv"]
BEFORE = {p: fstat(loc(p)) for p in INPUTS}
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
LOG = open(O / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
log(f"[{now()}] strict-provenance base-only replay")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
E = C.load("reported"); exp = E["exp"]; ST = C.streams(E)
CFG = {"theta_L2 (_L2_EQ_SIM_FLOOR)": exp._L2_EQ_SIM_FLOOR, "L2 floor = max(L2_SIM_THRESHOLD, theta_L2)": max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR), "eps_L1 (EPS_ANSWER)": exp.EPS_ANSWER,
       "eps_L2 (EPS_URL_LIST)": exp.EPS_URL_LIST, "eps_L3 (EPS_CONTENT)": exp.EPS_CONTENT, "L1_SIM_THRESHOLD": exp.L1_SIM_THRESHOLD, "_EQ_SIM_FLOOR": exp._EQ_SIM_FLOOR}
# ---------------- provenance transcription: in-memory copy with bookkeeping-only additions
src = open(ROOT / "freshcache/simulation/provenance_replay.py", encoding="utf-8").read()
def rep(old, new):
    global src; assert src.count(old) == 1, old; src = src.replace(old, new)
rep('L1 = {"qi": np.empty(n, np.int64), "t": np.empty(n, np.float64), "fc": [], "q": [], "creator": [], "n": 0}',
    'L1 = {"qi": np.empty(n, np.int64), "t": np.empty(n, np.float64), "fc": [], "q": [], "creator": [], "urls": [], "n": 0}')
rep('rows[qid] = {"tier": "L1", "own": len(own), "fetched": 0, "l1_entry_creator": L1["creator"][hit]}; continue',
    'rows[qid] = {"tier": "L1", "own": len(own), "fetched": 0, "l1_entry_creator": L1["creator"][hit], "l1_entry_urls": list(L1["urls"][hit])}; continue')
rep('"l3_reuse_writers": reuse,', '"l3_reuse_writers": reuse, "served_urls": list(served),')
rep('L1["fc"].append(fc); L1["q"].append(q); L1["creator"].append(qid); L1["n"] += 1', 'L1["fc"].append(fc); L1["q"].append(q); L1["creator"].append(qid); L1["urls"].append(list(served)); L1["n"] += 1')
PR = types.ModuleType("provenance_replay_bookkeeping"); exec(compile(src, "<in-memory copy of provenance_replay.py + list bookkeeping>", "exec"), PR.__dict__)
REC = {r["query_id"]: r for r in E["recs"]}; CL = lambda q: REC[q].get("cluster_id") or q
SRC = {json.loads(l)["query_id"]: json.loads(l)["source"] for l in open(DATA_ROOT / "queries.jsonl", encoding="utf-8")}
MAN = collections.defaultdict(list)
for l in open(DATA_ROOT / "url_manifest.jsonl", encoding="utf-8"):
    x = json.loads(l)
    if x["run_id"] == "run_00": MAN[x["query_id"]].append((x.get("rank", 99), x["url_hash"], bool(x.get("snapshot_available"))))
EXPECT = {s: json.load(open(BASE_ONLY_DIR / f"results/base_only_{s}_metrics.json")) for s in ("primary", "heldout")}   # headline values of the base-only replay

G, RUN = {}, {}
FHOK = {p: (BEFORE[p]["sha256"] == FH[p]) for p in INPUTS if p in FH}
G["frozen engine/data inputs match configs/integrity/FILE_HASHES.csv"] = all(FHOK.values())
for s in ("primary", "heldout"):
    bs = [(t, r) for t, r in ST[s] if not r.get("is_paraphrase")]; n = len(bs)
    Sb = E["support_of"](bs, E["rounds"]); row, per, tr = C.metrics(E, bs, "FreshCache", Sb)
    row["search_calls"] = sum(1 for q in tr if tr[q]["tier"] in ("miss", "REAL_TIME_BYPASS")); row["determinable"] = row["changed"] + row["unchanged"]   # the two keys the base-only replay adds after the engine call
    prov = PR.run(E, bs)
    BASE_ONLY_METRICS = json.load(open(BASE_ONLY_DIR / f"results/base_only_{s}_metrics.json")); BASE_ONLY_TIERS = {r["query_id"]: r["tier"] for r in csv.DictReader(open(BASE_ONLY_DIR / f"results/base_only_{s}.csv"))}
    G[f"{s}: {n} original requests, zero paraphrases, one per cluster"] = n == EXPECT[s]["n"] and not any(r.get("is_paraphrase") for _, r in bs) and len({CL(r["query_id"]) for _, r in bs}) == n
    G[f"{s}: frozen order/timestamps (non-decreasing t; identical id order to the base-only file)"] = all(bs[i][0] <= bs[i + 1][0] for i in range(n - 1)) and [r["query_id"] for _, r in bs] == list(BASE_ONLY_TIERS)
    G[f"{s}: engine metrics equal the frozen base-only metrics file exactly"] = row == BASE_ONLY_METRICS
    G[f"{s}: per-request engine tiers equal the frozen base-only tiers (all {n})"] = all(tr[q]["tier"] == BASE_ONLY_TIERS[q] for q in BASE_ONLY_TIERS)
    G[f"{s}: provenance copy tiers = engine tiers (all {n})"] = all(prov[q]["tier"] == tr[q]["tier"] for q in tr)
    G[f"{s}: provenance copy fetches and L3 reuses = engine"] = sum(x["fetched"] for x in prov.values()) == row["fetches"] and sum(len(x.get("l3_reuse_writers", [])) for x in prov.values()) == row["l3_hits"]
    G[f"{s}: replay URL lists equal the engine's stored-list definition: run_00 rows with snapshot_available, stable-sorted by rank (all {n})"] = all([u["url_hash"] for u in r["urls"]] == [u for _, u, a in sorted(MAN[r["query_id"]], key=lambda z: z[0]) if a] for _, r in bs)
    exp_ = EXPECT[s]
    G[f"{s}: reproduces the base-only headline values"] = all(row[k] == v for k, v in exp_.items() if k != "n") and row["n"] == exp_["n"]
    RUN[s] = {"bs": bs, "row": row, "prov": prov, "tr": tr}
G["primary / held-out savings equal the base-only values after 2-dp rounding"] = all(round(RUN[s]["row"]["search_saved_pct"], 2) == round(EXPECT[s]["search_saved_pct"], 2) for s in ("primary", "heldout"))
SHARE_RUNS = json.load(open(OUTPUT_ROOT / "paraphrase_share/work/runs.json")); A0 = next(r for r in SHARE_RUNS["runs"] if r.get("condition", "").startswith("A"))
G["paraphrase-share condition A agrees (requests, saved, L1, L2, L3)"] = (A0["requests"], A0["search_saved_pct"], A0["l1_hits"], A0["l2_hits"], A0["l3_hits"]) == tuple(EXPECT["primary"][k] for k in ("n", "search_saved_pct", "l1_hits", "l2_hits", "l3_hits"))
for k, v in G.items(): log("GATE", k, v)
if not all(G.values()):
    bad = [k for k, v in G.items() if not v]
    open(O / "REPRODUCTION_FAILURE.md", "w").write("# REPRODUCTION FAILURE\n\nThe base-only replay did not reproduce the frozen values; no analysis was produced or interpreted.\n\nFailed gates:\n\n" +
        "\n".join(f"- {k}" for k in bad) + "\n\nObserved: " + json.dumps({s: RUN[s]["row"] for s in RUN}, default=str) + "\n")
    log("STOP: reproduction failed", bad); sys.exit(1)
# ---------------- strict analysis (bookkeeping only; nothing above is re-decided)
def rel(own, used):
    a, b = set(own), set(used); i, u = len(a & b), len(a | b)
    cls = "IDENTICAL_LIST" if a == b else ("OVERLAPPING_LIST" if i else "DISJOINT_LIST")
    return i, u, (i / u if u else None), cls
HITS, RES = [], {}
for s in ("primary", "heldout"):
    R = RUN[s]; bs, row, prov = R["bs"], R["row"], R["prov"]; n = len(bs); c = collections.Counter(); jac = []; l3rel = collections.Counter()
    for t, r in bs:
        q = r["query_id"]; x = prov[q]; own = [u["url_hash"] for u in r["urls"]]
        if x["tier"] in ("L2", "miss"):
            for w in x["l3_reuse_writers"]: l3rel["same_request_duplicate" if w == q else ("cross_cluster" if CL(w) != CL(q) else "same_cluster")] += 1
        if x["tier"] not in ("L1", "L2"): continue
        if x["tier"] == "L2":
            src_req, origin, used = x["l2_entry_creator"], x["l2_list_origin"], x["served_urls"]
            assert used == [u["url_hash"] for u in REC[origin]["urls"]], "reused list must equal its discoverer's own stored list"
        else:
            src_req, origin, used = x["l1_entry_creator"], x["l1_entry_creator"], x["l1_entry_urls"]
        i, u, j, cls = rel(own, used)
        own_full = [v for _, v, _ in sorted(MAN[q], key=lambda z: z[0])]; used_full = [v for _, v, _ in sorted(MAN[origin], key=lambda z: z[0])]
        i_f, u_f, j_f, cls_f = rel(own_full, used_full)
        h = {"stream": s, "tier": x["tier"], "incoming_request_id": q, "incoming_cluster_id": CL(q), "incoming_source": SRC.get(q, ""), "incoming_freshness_class": r["freshness_class"], "t": t,
             "source_request_id": src_req, "source_cluster_id": CL(src_req), "list_origin_request_id": origin, "list_origin_cluster_id": CL(origin), "list_origin_source": SRC.get(origin, ""),
             "different_cluster_entry": CL(src_req) != CL(q), "different_cluster_list_origin": CL(origin) != CL(q),
             "dataset_relation": ("same_dataset" if SRC.get(origin) == SRC.get(q) else "cross_dataset") if SRC.get(origin) and SRC.get(q) and "reconstructed" not in (SRC.get(origin) + SRC.get(q)) else "unknown_source_involved",
             "incoming_own_urls": "|".join(own), "reused_urls": "|".join(used), "own_list_len": len(own), "reused_list_len": len(used), "url_intersection": i, "url_union": u,
             "url_jaccard": round(j, 6) if j is not None else "", "classification": cls, "sensitivity_full_run00_classification": cls_f, "sensitivity_full_run00_jaccard": round(j_f, 6) if j_f is not None else "", "l3_reused_pages": len(x.get("l3_reuse_writers", [])) if x["tier"] == "L2" else 0, "pages_fetched": x["fetched"]}
        HITS.append(h); c[(x["tier"], cls)] += 1; c[("full_" + x["tier"], cls_f)] += 1
        if x["tier"] == "L2": jac.append(j)
    l2 = [h for h in HITS if h["stream"] == s and h["tier"] == "L2"]; l1 = [h for h in HITS if h["stream"] == s and h["tier"] == "L1"]; nl2 = len(l2)
    assert nl2 == row["l2_hits"] and len(l1) == row["l1_hits"]
    pp = lambda k: round(100 * k / n, 4)
    cat = {k: {"L2_hits": c[("L2", k)], "pct_of_L2_hits": round(100 * c[("L2", k)] / nl2, 2), "search_savings_pp": pp(c[("L2", k)])} for k in ("IDENTICAL_LIST", "OVERLAPPING_LIST", "DISJOINT_LIST")}
    qs = statistics.quantiles(jac, n=20) if len(jac) > 1 else []
    RES[s] = {"requests": n, "fresh_searches": row["search_calls"], "total_search_savings_pct": row["search_saved_pct"], "L1_hits": row["l1_hits"], "L1_only_avoided_searches": row["l1_hits"],
              "L1_search_savings_pp": pp(row["l1_hits"]), "L2_hits": nl2, "A_all_L2_search_savings_pp": pp(nl2), "B_IDENTICAL_LIST_L2_savings_pp": cat["IDENTICAL_LIST"]["search_savings_pp"],
              "C_OVERLAPPING_LIST_L2_savings_pp": cat["OVERLAPPING_LIST"]["search_savings_pp"], "D_DISJOINT_LIST_L2_savings_pp": cat["DISJOINT_LIST"]["search_savings_pp"],
              "strict_disjoint_L2_search_savings_pct": pp(c[("L2", "DISJOINT_LIST")]), "strict_disjoint_L2_hits": c[("L2", "DISJOINT_LIST")], "L2_overlap_categories": cat,
              "L2_url_jaccard": {"median": statistics.median(jac), "mean": round(statistics.fmean(jac), 6), "share_exactly_0": round(sum(1 for v in jac if v == 0) / nl2, 4),
                                 "share_exactly_1": round(sum(1 for v in jac if v == 1) / nl2, 4), "p05_to_p95_by_5": [round(v, 4) for v in qs],
                                 "histogram": dict(collections.Counter(("0" if v == 0 else "1" if v == 1 else f"({b / 10:.1f},{(b + 1) / 10:.1f}]") for v in jac for b in [min(int((v - 1e-12) * 10), 9)]))},
              "L2_hits_from_different_cluster_entry": sum(1 for h in l2 if h["different_cluster_entry"]), "L2_hits_from_different_cluster_list_origin": sum(1 for h in l2 if h["different_cluster_list_origin"]),
              "L2_dataset_relation_diagnostic": dict(collections.Counter(h["dataset_relation"] for h in l2)), "L2_by_incoming_source_and_class_diagnostic": dict(collections.Counter(f'{h["incoming_source"]}|{h["classification"]}' for h in l2)),
              "sensitivity_full_run00_lists_L2_categories": {k: c[("full_L2", k)] for k in ("IDENTICAL_LIST", "OVERLAPPING_LIST", "DISJOINT_LIST")},
              "sensitivity_full_run00_strict_disjoint_L2_search_savings_pct": pp(c[("full_L2", "DISJOINT_LIST")]),
              "L1_overlap_categories_separate": {k: c[("L1", k)] for k in ("IDENTICAL_LIST", "OVERLAPPING_LIST", "DISJOINT_LIST")},
              "L3_reused_pages": row["l3_hits"], "L3_reuse_by_page_writer": dict(l3rel), "fetches": row["fetches"], "fetch_per_1k": row["fetch_per_1k"], "generations": row["generations"], "gen_per_1k": row["gen_per_1k"],
              "real_time_bypass_requests": sum(1 for q in RUN[s]["tr"] if RUN[s]["tr"][q]["tier"] == "REAL_TIME_BYPASS")}
    log(s, json.dumps({k: RES[s][k] for k in ("total_search_savings_pct", "A_all_L2_search_savings_pp", "B_IDENTICAL_LIST_L2_savings_pp", "C_OVERLAPPING_LIST_L2_savings_pp", "D_DISJOINT_LIST_L2_savings_pp", "L1_search_savings_pp")}))
cols = list(HITS[0])
with open(O / "provenance_hits.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(HITS)
AFTER = {p: fstat(loc(p)) for p in INPUTS}; UNCH = {p: AFTER[p] == BEFORE[p] for p in INPUTS}
json.dump({"generated_utc": now(), "configuration": CFG, "gates": G, "results": RES, "inputs_unchanged": UNCH, "freeze_hash_match": FHOK,
           "notes": ["benchmark-derived replay; not a natural or production workload", "no drift/WAI/sufficiency/correctness/equivalence reported (no new labels)",
                     "classification compares the reused cached URL list with the incoming question's own stored run_00 list (set comparison)"]},
          open(O / "results.json", "w"), indent=1, default=str)
log("done; inputs unchanged:", all(UNCH.values()))
