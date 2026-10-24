"""Offline preparation (no Web) for the unseen-query live-retrieval experiment: for every untouched base query, simulate its arrival at a seeded
scripts/evaluation time inside the frozen held-out replay timeline and apply FreshCache's frozen L2 rule (cos >= 0.75 against entries created before t;
calibrated url_list window p_stale(entry class, t - t_entry) <= 0.20) to the held-out L2 cache state at that time. Output: which untouched queries
would get an L2 hit, the cached list they would reuse, its origin cluster/source, similarity, and the L3 cache state for those URLs. This defines the
cross-cluster candidate pool. Cache state comes from an in-memory copy of the frozen provenance replay (file untouched)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, datetime, json, os, pathlib, random, sys, types; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "unseen_queries")); BASE_ONLY_DIR = (OUTPUT_ROOT / "benchmark_locality")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
import numpy as np   # noqa: E402
E = C.load("reported"); exp = E["exp"]; H = C.streams(E)["heldout"]; RT = E["rounds"]
SEED = 20261004; rng = random.Random(SEED)
src = open((ROOT / "freshcache/simulation/provenance_replay.py"), encoding="utf-8").read()
old = "    return rows"; assert src.count(old) == 1
src = src.replace(old, '    return rows, {"L2": L2, "L3": L3, "L3w": L3w, "L1": L1}')
PR = types.ModuleType("provenance_copy"); exec(compile(src, "<in-memory copy of provenance_replay.py + state return>", "exec"), PR.__dict__)
rows, STATE = PR.run(E, H); L2 = STATE["L2"]; n2 = L2["n"]
FR = json.load(open((OUTPUT_ROOT / "replay/results/replays_reported.json")))["heldout/FreshCache"]
tiers = collections.Counter(r["tier"] for r in rows.values()); assert tiers["L1"] == FR["l1_hits"] and tiers["L2"] == FR["l2_hits"], tiers
REC = {r["query_id"]: r for r in E["recs"]}; CL = lambda q: REC[q].get("cluster_id") or q
SRC = {json.loads(l)["query_id"]: json.loads(l)["source"] for l in open((DATA_ROOT / "queries.jsonl"))}
Z = np.load(O / "emb_untouched.npz", allow_pickle=True); UQ = list(Z["keys"]); UID = list(Z["ids"]); UE = Z["emb"]
F = np.load((OUTPUT_ROOT / "l2_verifier/out/emb_queries.npz"), allow_pickle=True); FK = {k: i for i, k in enumerate(F["keys"])}; FE = F["emb"] / np.linalg.norm(F["emb"], axis=1, keepdims=True)
l2_vec = FE[[FK[REC[q]["query"]] for q in L2["creator"][:n2]]]   # entry query embeddings (frozen file)
l2_t = L2["t"][:n2]; l2_fc = L2["fc"][:n2]; l2_urls = L2["urls"][:n2]; l2_creator = L2["creator"][:n2]; l2_origin = L2["origin"][:n2]
t_min, t_max = H[0][0], H[-1][0]; EPS = exp.EPS_URL_LIST; FLOOR = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
UNQ = {json.loads(l)["query_id"]: json.loads(l) for l in open(O / "untouched_queries.jsonl")}
out = []; agg = collections.Counter()
for i, (qid, qtext) in enumerate(zip(UID, UQ)):
    t = rng.uniform(t_min, t_max); fc = UNQ[qid]["freshness_class"]
    rec = {"query_id": qid, "query": qtext, "freshness_class": fc, "source": SRC.get(qid), "eval_t": t, "own_run00_urls": [u["url_hash"] for u in UNQ[qid]["run00_urls"]]}
    if fc == "REAL_TIME": rec["route"] = "REAL_TIME_BYPASS"; agg["REAL_TIME"] += 1; out.append(rec); continue
    before = np.nonzero(l2_t < t)[0]
    if before.size == 0: rec["route"] = "miss"; agg["miss_empty"] += 1; out.append(rec); continue
    sims = l2_vec[before] @ UE[i]; cand = before[sims >= FLOOR]
    ok = [int(j) for j in cand if exp.p_stale(l2_fc[int(j)], t - l2_t[j], "url_list") <= EPS]
    rec["n_sim_candidates"] = int(cand.size); rec["n_temporally_eligible"] = len(ok)
    if not ok: rec["route"] = "miss"; agg["miss_" + ("no_sim" if cand.size == 0 else "temporal_reject")] += 1; out.append(rec); continue
    j = max(ok, key=lambda j: float(l2_vec[j] @ UE[i])); o = l2_origin[j]
    rec.update({"route": "L2", "similarity": float(l2_vec[j] @ UE[i]), "entry_age_s": t - float(l2_t[j]), "entry_class": l2_fc[j], "cached_urls": list(l2_urls[j]), "entry_creator": l2_creator[j], "entry_creator_query": REC[l2_creator[j]]["query"],
                "list_origin": o, "list_origin_query": REC[o]["query"], "list_origin_cluster": CL(o), "list_origin_source": SRC.get(o), "relation": "CROSS (untouched cluster)",
                "url_overlap_with_own_run00": len(set(l2_urls[j]) & set(rec["own_run00_urls"])), "l3_cached_at_t": [u for u in l2_urls[j] if u in STATE["L3"] and STATE["L3"][u] < t], "l3_cache_time": {u: float(STATE["L3"][u]) for u in l2_urls[j] if u in STATE["L3"] and STATE["L3"][u] < t}})
    agg["L2"] += 1; out.append(rec)
with open(O / "cache_state_l2_matching.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
hits = [r for r in out if r["route"] == "L2"]
summ = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seed": SEED, "eval_time": "uniform over the held-out timeline (seeded), one per query", "rule": f"cos >= {FLOOR}, p_stale(entry class, age, url_list) <= {EPS}, entries created before t", "routes": dict(agg),
        "l2_hits": len(hits), "hits_by_class": dict(collections.Counter(r["freshness_class"] for r in hits)), "hits_by_source": dict(collections.Counter(r["source"] for r in hits)),
        "hits_by_origin_source": dict(collections.Counter(r["list_origin_source"] for r in hits)), "sim_bands": dict(collections.Counter(f"{0.05 * int(r['similarity'] / 0.05):.2f}" for r in hits)),
        "cached_list_len": dict(collections.Counter(len(r["cached_urls"]) for r in hits)), "hits_with_zero_url_overlap_vs_own_run00": sum(1 for r in hits if r["url_overlap_with_own_run00"] == 0),
        "hits_with_all_cached_urls_in_L3_at_t": sum(1 for r in hits if len(r["l3_cached_at_t"]) == len(r["cached_urls"]))}
json.dump(summ, open(O / "work/cache_state_matching_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
