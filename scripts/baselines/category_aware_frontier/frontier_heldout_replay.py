"""CategoryAware-style comparator: our reimplementation of the per-category similarity threshold and TTL policy of
category-aware semantic caching (Wang et al., 2025), not the complete published system; it omits quotas, the storage tier,
indexing and load adaptation.

CategoryAware frontier stage 3: held-out replays and operation counts for the selected policies (runs only after frontier_selection_manifest.json is frozen read-only).
Also builds answer keys for the audit sample, served-evidence blocks (request-time-retained provenance only), and GPU task lists for anything missing."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, hashlib, json, os, pickle, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/category_aware_frontier"))
import frontier_helpers as X
MAN = X.O / "frontier_selection_manifest.json"; assert MAN.exists() and not (os.stat(MAN).st_mode & stat.S_IWUSR)
M = json.load(open(MAN)); assert X.sha_obj({k: v for k, v in M.items() if k not in ("sha256", "frozen_utc")}) == M["sha256"]
X.log(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] held-out stage; manifest sha256 {M['sha256']}")
E = X.engine(); exp, p2 = E["exp"], E["p2"]; H = X.C.streams(E)["heldout"]; n = len(H); assert DS.check("heldout_requests", n); nonrt = sum(1 for _, r in H if r["freshness_class"] != "REAL_TIME")
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}
GRID = {g["id"]: g for g in json.load(open(X.O / "grid_frozen.json"))["grid"]}
FR = json.load(open(X.CRR / "results/replays_reported.json"))["heldout/FreshCache"]; L2_POINTS = {r["point"]: r for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_points.csv"))}
VO = json.load(open(X.CRR / "verifier/out/test_operational.json"))["results"]["L2Verify_g0.55"]; BASELINE_POINTS = {r["arm"]: r for r in csv.DictReader(open(X.BASELINE_SEL_DIR / "baseline_selection_points.csv"))}
G, OPS, CA = {}, {}, {}
OPS["FreshCache primary (theta_L2=0.75)"] = {"search": int(L2_POINTS["theta_0.75"]["fresh_searches"]), "fetch": FR["fetches"], "gen": FR["generations"], "lookup": X.LOOKUP["FreshCache"], "note": "frozen replays_reported heldout"}
o = exp._L2_EQ_SIM_FLOOR; exp._L2_EQ_SIM_FLOOR = 0.80
try: m80, _, t80 = X.C.metrics(E, H, "FreshCache")
finally: exp._L2_EQ_SIM_FLOOR = o
assert exp._L2_EQ_SIM_FLOOR == 0.75
OPS["FreshCache theta_L2=0.80 (sensitivity)"] = {"search": sum(1 for q in t80 if t80[q]["tier"] in ("miss", "REAL_TIME_BYPASS")), "fetch": m80["fetches"], "gen": m80["generations"], "lookup": X.LOOKUP["FreshCache"], "note": "mixed_age_replay replay, floor 0.80 in memory (gated to L2-sweep)"}
G["theta 0.80 replay equals L2-sweep theta_0.80 point"] = abs(m80["search_saved_pct"] - float(L2_POINTS["theta_0.80"]["search_saved_pct"])) < 1e-9 and m80["fetches"] == int(L2_POINTS["theta_0.80"]["fetches"])
OPS["FreshCache verifier gamma=0.55 (sensitivity)"] = {"search": VO["search_calls"], "fetch": VO["fetches"], "gen": VO["generations"], "lookup": X.LOOKUP["FreshCache"], "note": "frozen verifier replay; cost EXCLUDES verifier scoring (unmeasured)"}
POL_CA = {"CategoryAware-style (original)": "ca_d+0.000_m1.00"}
for k in ("CA_Strict", "CA_Relaxed"):
    v = M["selected"][k]
    if v != "NONE": POL_CA.setdefault(f"{k.replace('_', '-')} ({v})" if v != "ca_d+0.000_m1.00" else None, v)
POL_CA.pop(None, None)
for name, cid in POL_CA.items():
    g = GRID[cid]; m, rows = X.ca_replay(H, g["theta"], g["kappa"]); CA[name] = {"cid": cid, "rows": {q: rows[q] for q in SQ}}
    OPS[name] = {"search": m["search_calls"], "fetch": m["fetches"], "gen": m["generations"], "lookup": X.LOOKUP["CategoryAware"], "note": f"baseline-selection ca_replay, {cid}"}
    if cid == "ca_d+0.000_m1.00": G["original CategoryAware held-out replay equals baseline-selection point"] = abs(m["search_saved_pct"] - float(BASELINE_POINTS["CategoryAware_style"]["search_saved_pct"])) < 1e-9 and m["fetches"] == int(BASELINE_POINTS["CategoryAware_style"]["fetches"])
for k_, v in OPS.items(): v.update(X.cost(v["search"], v["fetch"], v["gen"], n, nonrt, v["lookup"])); v.update({"search_saved_pct": round(100 * (1 - v["search"] / n), 4), "search_per_1k": round(1000 * v["search"] / n, 1), "fetch_per_1k": round(1000 * v["fetch"] / n, 1), "gen_per_1k": round(1000 * v["gen"] / n, 1)})
# 400-audit answers + served evidence for CategoryAware policies
BASELINE_KEYS = json.load(open(X.BASELINE_SEL_DIR / "work/stage3.json"))["keys"]["CategoryAware_style"]
TASK_G, TASK_S, AUD = {}, {}, {}
for name, d in CA.items():
    AUD[name] = {}
    for q, dd in SQ.items():
        p = d["rows"][q]; k, ctx, gq = X.key_for(p, dd["query"]); ev = [tuple(x) for x in (p.get("ev") or [])]
        blk = X.evblock(ev) if ev else None
        AUD[name][q] = {"tier": p["tier"], "key": k, "gen_question": gq, "context": ctx, "ev": ev, "evidence_block": blk, "evidence_sha": hashlib.sha256(blk.encode()).hexdigest() if blk else None,
                        "evidence_source": ("stored entry evidence (pages at the entry's creation time)" if p["tier"] == "L1" else "evidence given to generation (own pages at request time)") if ev else "EVIDENCE_UNAVAILABLE"}
        TASK_G.setdefault(tuple(k), {"key": k, "question": gq, "context": ctx})
        if blk: TASK_S.setdefault((q, AUD[name][q]["evidence_sha"]), {"query_id": q, "arm": AUD[name][q]["evidence_sha"][:16], "query": dd["query"], "evidence": blk})
    if d["cid"] == "ca_d+0.000_m1.00": G["original CategoryAware 400 keys equal frozen baseline-selection keys"] = all(AUD[name][q]["key"] == BASELINE_KEYS[q]["key"] and AUD[name][q]["tier"] == BASELINE_KEYS[q]["tier"] for q in SQ)
# sufficiency-label reuse pools (earlier sufficiency outputs)
SL = {}
def addsuff(tasks, l8, q7):
    for t in tasks:
        a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
        if a is not None and b is not None: SL.setdefault((t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()), (a, b))
jl = X.jl
addsuff(jl(X.OV / "audit_suff_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(X.OV / "audit_suff_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(X.OV / "audit_suff_qwen7b.jsonl")})
RU = json.load(open(X.CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
for r in jl(X.CRR / "gpu/out/suffv_llama8b.jsonl"): cl8[(r["query_id"], r["arm"])] = r["label"]
for r in jl(X.CRR / "gpu/out/suffv_qwen7b.jsonl"): cq7[(r["query_id"], r["arm"])] = r["label"]
addsuff(jl(X.CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7)
for D in (X.L2_POINTS_DIR, X.CALIBRATION_AUDIT_DIR, X.O):
    addsuff(jl(D / "gpu/in/suffv_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(D / "gpu/out/suffv_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(D / "gpu/out/suffv_qwen7b.jsonl")})
new_s = [t for k, t in TASK_S.items() if k not in SL]
hl = X.Held(); new_g = [t for t in TASK_G.values() if hl.answer(t["key"]) is None]
old = {(t["query_id"], t["arm"]) for t in jl(X.O / "gpu/in/suffv_tasks.jsonl")}
X.wjl(X.O / "gpu/in/suffv_tasks.jsonl", jl(X.O / "gpu/in/suffv_tasks.jsonl") + [t for t in new_s if (t["query_id"], t["arm"]) not in old]); X.wjl(X.O / "gpu/in/gen_tasks.jsonl", new_g)
for k_, v in G.items(): X.log("GATE", k_, v)
if not all(G.values()): X.log("STOP: held-out gate failed"); sys.exit(1)
pickle.dump({"ops": OPS, "audit": {k: {q: {kk: vv for kk, vv in a.items() if kk not in ("context", "evidence_block")} for q, a in v.items()} for k, v in AUD.items()}, "gates": G, "policies_ca": {k: v["cid"] for k, v in CA.items()}}, open(X.O / "work/stage3.pkl", "wb"))
X.log(f"held-out ops: {json.dumps({k: {x: v[x] for x in ('search_saved_pct', 'fetch_per_1k', 'gen_per_1k', 'local_ms_per_request', 'live_p50_ms_per_request')} for k, v in OPS.items()})}")
X.log(f"CategoryAware policies evaluated: {json.dumps({k: v['cid'] for k, v in CA.items()})}; new answers {len(new_g)}; new sufficiency tasks {len(new_s)} (of {len(TASK_S)}); evidence unavailable {sum(1 for a in AUD.values() for x in a.values() if x['evidence_sha'] is None)}")
