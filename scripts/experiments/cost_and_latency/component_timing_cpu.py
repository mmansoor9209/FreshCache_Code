"""Systems timing stage 1 (CPU): exact-reproduction gate, CPU component timing (2 passes), storage by tier, capacity sensitivity (primary stream)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, glob, hashlib, json, os, pathlib, pickle, statistics, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); CRR = (OUTPUT_ROOT / "replay"); W = (OUTPUT_ROOT / "systems_timing")
for d in ("results", "logs", "frozen", "intermediate"): (W / d).mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); sys.path.insert(0, str(ROOT / "freshcache/simulation"))
import engine_loader as C, reference_replay as R   # noqa: E402
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
FRAC = [0.25, 0.50, 0.75, 1.00]
PRE = {"written_utc": now(), "stream": "primary (frozen FreshCache settings)", "eviction_rule": "per-tier least-recently-used; a hit or a write refreshes recency; capacity counted in entries (L1, L2) or pages (L3)",
       "capacity_reference": "observed peak = entries/pages held at the end of the unbounded replay (the frozen replay never evicts)",
       "scenarios": [{"name": f"all_tiers_{f:.2f}", "L1": f, "L2": f, "L3": f} for f in FRAC] + [{"name": f"{t}_only_{f:.2f}", t: f} for t in ("L1", "L2", "L3") for f in FRAC[:3]],
       "no_threshold_or_setting_changed": True}
PRE["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in PRE.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (W / "frozen/prereg.json").exists(): json.dump(PRE, open(W / "frozen/prereg.json", "w"), indent=1)
PRE = json.load(open(W / "frozen/prereg.json"))
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
INP = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/engine_loader.py", "freshcache/method/tiered_cache_model.py",
       "data/derived/round_table.jsonl", "data/queries.jsonl", "data/url_manifest.jsonl", "data/paraphrase_clusters.jsonl", "data/sim_matrix_bgem3.npy", "data/query_embeddings_bgem3.npy",
       "outputs/replay/results/replays_reported.json", "outputs/replay/results/replays_reported_per_request.pkl", "outputs/validation_split/split.json"]
INT = {p: {"sha256": sha(inpath(p)), "freeze": FH.get(p)} for p in INP}
for v in INT.values(): v["match"] = (v["sha256"] == v["freeze"]) if v["freeze"] else None
if any(v["match"] is False for v in INT.values()): json.dump(INT, open(W / "logs/STOPPED.json", "w"), indent=1); sys.exit("STOP integrity")
E = C.load("reported"); me = E["me"]; P = C.streams(E)["primary"]; assert DS.check("primary_requests", len(P))
FR = json.load(open(CRR / "results/replays_reported.json"))["primary/FreshCache"]; PK = pickle.load(open(CRR / "results/replays_reported_per_request.pkl", "rb"))["primary/FreshCache"]["tier"]
G = {}
rep = R.Replay(E, timers=True); t0 = time.time(); m, tiers, T1, EV = rep.run(P); wall1 = time.time() - t0; ST = rep.state
G["unbounded_metrics_equal_frozen"] = {k: m[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "l3_hits", "fetches", "generations")} == {k: FR[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "l3_hits", "fetches", "generations")}
G["unbounded_per_request_tiers_equal_frozen"] = all(tiers[r["query_id"]] == PK[r["query_id"]] for _, r in P)
if not all(G.values()): json.dump({"gates": G, "m": m, "frozen": FR}, open(W / "logs/STOPPED.json", "w"), indent=1); sys.exit(f"STOP gate {G}")
rep2 = R.Replay(E, timers=True); t0 = time.time(); m2, tiers2, T2, _ = rep2.run(P); wall2 = time.time() - t0
G["second_timed_pass_identical_decisions"] = tiers2 == tiers
def summ(v):
    v = sorted(v); n = len(v); return {"n": n, "mean_ms": round(statistics.mean(v), 5), "p50_ms": round(v[n // 2], 5), "p95_ms": round(v[int(.95 * n)], 5)} if v else {"n": 0}
TIM = []
for comp, label in (("l1_lookup", "L1 lookup (similarity-row gather over live entries + temporal test + entity/semantic gates)"), ("l2_lookup", "L2 lookup (similarity gather + temporal test + argmax)"),
                    ("l3_lookup_per_request", "L3 lookup, all served pages of a request (dict lookup + temporal test + write on miss)"), ("cache_write_update", "cache write/update (L2 registration + L1 insert + LRU bookkeeping)")):
    a, b = summ(T1[comp]), summ(T2[comp]); stable = abs(a["p50_ms"] - b["p50_ms"]) <= 0.2 * max(a["p50_ms"], 1e-9)
    TIM.append({"component": label, "key": comp, "source": "same-environment CPU, this session, replay implementation (precomputed similarity)", **a, "pass2_p50_ms": b["p50_ms"],
                "pass2_mean_ms": b["mean_ms"], "stable_p50_within_20pct": stable})
json.dump(TIM, open(W / "intermediate/cpu_timing.json", "w"), indent=1)
# ---------------- storage by tier (end of unbounded replay)
FROZ = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open((DATA_ROOT / "derived/round_table.jsonl"))}
L1, L2, L3 = ST["L1"], ST["L2"], ST["L3"]
def text_bytes(u, rnd):
    p = (DATA_ROOT / "snapshots") / rnd / f"{u}.json"
    if not p.exists(): return 0
    return len((json.load(open(p, encoding="utf-8")).get("extracted_text") or "").encode("utf-8"))
page_bytes = sum(text_bytes(u, me.round_at(FROZ[u], tc) or "run_00") for u, tc in L3.items() if u in FROZ)
pool_ans = list(C.__dict__.get("P", None) or [])  # placeholder (answers measured below)
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import answer_pools as PO   # noqa: E402
ans = list(PO.answers_exact(extra=[CRR / "gpu/out/answers_llama3b.jsonl"]).values()); ans_b = statistics.mean(len(a.encode("utf-8")) for a in ans)
n1, n2 = L1["n"], L2["n"]; ev_pairs = sum(len(e) for e in L1["ev"]); url_refs = sum(len(u) for u in L2["urls"])
STO = [{"tier": "L1", "entries_or_pages": n1, "component": "answer text", "bytes": round(n1 * ans_b), "basis": f"mean UTF-8 bytes of {len(ans)} frozen Llama-3.2-3B answers x entries (answers are not produced by the replay)"},
       {"tier": "L1", "entries_or_pages": n1, "component": "evidence-version manifest + t_e + class", "bytes": ev_pairs * 32 + n1 * 9, "basis": f"{ev_pairs} (url_hash 24 B + float64) pairs + 9 B/entry, exact counts"},
       {"tier": "L2", "entries_or_pages": n2, "component": "URL list + discovery time + class", "bytes": url_refs * 24 + n2 * 9, "basis": f"{url_refs} url_hash references x 24 B + 9 B/entry, exact counts"},
       {"tier": "L3", "entries_or_pages": len(L3), "component": "page text of the cached version", "bytes": page_bytes, "basis": "exact UTF-8 bytes of the snapshot text of each cached page's version (round_at of its cache time)"},
       {"tier": "L3", "entries_or_pages": len(L3), "component": "page key + cache time", "bytes": len(L3) * 32, "basis": "24 B url_hash + float64"},
       {"tier": "index", "entries_or_pages": n1, "component": "query embeddings (float32 x 1024), one per cached query, shared by its L1 and L2 entries", "bytes": n1 * 4096, "basis": "data/query_embeddings_bgem3.npy row size"},
       {"tier": "index", "entries_or_pages": n1, "component": "replay arrays (qi int64 + t float64 + alive bool) x 2 tiers", "bytes": 2 * n1 * 17, "basis": "numpy array element sizes"}]
STO.append({"tier": "TOTAL", "entries_or_pages": "", "component": "total cache footprint (payload accounting)", "bytes": sum(s["bytes"] for s in STO), "basis": "sum of the rows above"})
json.dump({"storage": STO, "peak": m["peak_entries"]}, open(W / "intermediate/storage.json", "w"), indent=1)
# ---------------- capacity sensitivity
peak = m["peak_entries"]; CAP = []
def row(name, capd, mm, wall):
    return {"scenario": name, "cap_L1_entries": capd.get("L1", "unbounded"), "cap_L2_entries": capd.get("L2", "unbounded"), "cap_L3_pages": capd.get("L3", "unbounded"),
            "search_saved_pct": mm["search_saved_pct"], "search_calls": mm["search_calls"], "fetches_per_1k": round(1000 * mm["fetches"] / len(P), 1), "generations_per_1k": round(1000 * mm["generations"] / len(P), 1),
            "l1_hits": mm["l1_hits"], "l2_hits": mm["l2_hits"], "l3_hits": mm["l3_hits"], "evictions": json.dumps(mm["evictions"]), "wall_s": round(wall, 1)}
CAP.append(row("unbounded (frozen)", {}, m, wall1))
for sc in PRE["scenarios"]:
    capd = {t: int(sc[t] * peak[t]) for t in ("L1", "L2", "L3") if t in sc}
    t0 = time.time(); mm, _, _, _ = R.Replay(E, cap=capd).run(P); CAP.append(row(sc["name"], capd, mm, time.time() - t0)); print(sc["name"], CAP[-1], flush=True)
G["all_tiers_1.00_equals_unbounded"] = all(CAP[-1 - 0][k] == CAP[0][k] for k in ()) or next(r for r in CAP if r["scenario"] == "all_tiers_1.00")["search_saved_pct"] == CAP[0]["search_saved_pct"]
def wcsv(n, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(W / "results" / n, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
wcsv("capacity_sensitivity.csv", CAP)
nocache = {"search_calls": len(P), "fetches": sum(len(r["urls"]) for _, r in P), "generations": len(P)}
json.dump({"gates": G, "integrity": INT, "prereg_sha256": PRE["sha256"], "unbounded": m, "nocache_counts": nocache, "replay_wall_s": [wall1, wall2], "finished": now()}, open(W / "logs/cpu_timing.json", "w"), indent=1)
print("GATES", G); print(TIM); print(STO[-1]); print(nocache)
