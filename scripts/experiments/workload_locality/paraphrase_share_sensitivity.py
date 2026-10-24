"""FreshCache savings vs paraphrase share. Frozen replay engine, frozen settings, PRIMARY stream (zipf_uniform,
seed 42). Conditions: A 0% (base-only replay, reproduced), B 10% and C 30% paraphrase share (subsamples of existing paraphrase requests, 10 seeds
each), D original full replay (frozen). Every replay starts from an empty cache. Design frozen before any replay; sampling uses no metric."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, hashlib, importlib.util, json, math, os, pathlib, random, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "paraphrase_share"); CRR = (OUTPUT_ROOT / "replay"); BASE_ONLY_DIR = (OUTPUT_ROOT / "benchmark_locality")
(O / "work").mkdir(parents=True, exist_ok=True); LOGF = open(O / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
sha_obj = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
SEEDS = list(range(42, 52)); SHARES = {"B_10pct": 0.10, "C_30pct": 0.30}
DESIGN = {"status": "workload-sensitivity diagnostic; no FreshCache parameter tuned; no configuration selected; not a production-traffic claim",
  "engine": "frozen (outputs/replay/engine), FreshCache, frozen settings; each replay starts from an empty cache",
  "stream": "PRIMARY (full) stream: sc.build_stream(recs, 'zipf_uniform', 42); the held-out split is not used here",
  "A_0pct": "base-only replay reproduced: requests with is_paraphrase == False (one per cluster), original timestamps/order; gated to outputs/benchmark_locality/results/base_only_primary_metrics.json",
  "D_full": "frozen primary replay; gated to outputs/replay/results/replays_reported.json primary/FreshCache and the cross-round FreshCache audit (outputs/cross_round_audit/cross_round_summary.json)",
  "B_C_construction": "keep ALL base requests (N_base); add P paraphrase requests drawn from the existing paraphrase requests so that P / (N_base + P) equals the target share: P = round(N_base * s / (1 - s)). Paraphrase quota per freshness class proportional to the class's share of the paraphrase pool (largest remainder). Within a class, strata = query clusters: stratum h (N_h paraphrases) receives floor(Q_c N_h / N_c) plus one extra with probability equal to the fractional part (systematic PPS on fractional parts, seeded), so cluster popularity is preserved in expectation; simple random sample within the stratum. Retained requests keep their original timestamps; the replay is the original stream filtered to base + retained paraphrases, in original chronological order. No synthesis. No metric is consulted.",
  "seeds": {"values": SEEDS, "representative": 42, "rng": "random.Random(seed) for class quota ties, fractional selection and within-stratum draws"},
  "metrics": "requests, base, paraphrase, share, unique clusters; search savings; L1/L2/L3 hits and rates (per request; L3 page-reuse rate = L3 reuses / pages served on L2+miss routes); fresh searches; fetches/1k; generations/1k; avoided searches split within-cluster vs cross-cluster by the cluster of the request that created the reused L1/L2 entry (locality decomposition via the gated provenance transcription); step-function drift/coverage (SECONDARY); cross-round cross-round drift/coverage (outcome() observation wrapper, cross-round rule)",
  "aggregation": "per share over 10 seeds: mean, SD, min, max, seed-level 95% CI mean +/- t(0.975, 9) SD/sqrt(10) (sampling variability of the subsample only), seed-42 representative",
  "trend": "monotonicity of savings (and other metrics) across A -> B -> C -> D using seed means; also whether every B seed < every C seed"}
DESIGN["sha256"] = sha_obj(DESIGN)
if not (O / "design_frozen.json").exists(): json.dump({**DESIGN, "written_utc": now()}, open(O / "design_frozen.json", "w"), indent=1)
assert json.load(open(O / "design_frozen.json"))["sha256"] == DESIGN["sha256"], "design changed after freezing"
log(f"[{now()}] paraphrase-share start; design sha256 {DESIGN['sha256']}")
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/method_replay.py",
        "freshcache/replay_engine/execution_semantics.py", "freshcache/replay_engine/engine_loader.py", "outputs/replay/results/replays_reported.json",
        "data/derived/round_table.jsonl", "freshcache/simulation/page_version_replay.py", "freshcache/simulation/request_schedules.py", "freshcache/method/tiered_cache_model.py", "data/queries.jsonl", "data/url_manifest.jsonl",
        "data/paraphrase_clusters.jsonl", "data/sim_matrix_bgem3.npy", "outputs/validation_split/split.json", "configs/integrity/FILE_HASHES.csv",
        "freshcache/simulation/provenance_replay.py", "scripts/experiments/workload_locality/cluster_locality_decomposition.py",
        "outputs/benchmark_locality/results/base_only_primary_metrics.json", "outputs/benchmark_locality/results/full_trace_cluster_decomposition.csv",
        "outputs/cross_round_audit/cross_round_summary.json"]
BEF = {p: fstat(inpath(p)) for p in READ}
if not (O / "work/read_only_before.json").exists(): json.dump(BEF, open(O / "work/read_only_before.json", "w"), indent=1)
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
for p, v in BEF.items(): log(f"  read {p} sha256={v['sha256']} freeze={'match' if FH.get(p) == v['sha256'] else ('MISMATCH' if p in FH else 'not in freeze')}")
if any(p in FH and FH[p] != v["sha256"] for p, v in BEF.items()): log("STOP: frozen input differs"); sys.exit(1)
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
E = C.load("reported"); me, RT = E["me"], E["rounds"]
spec = importlib.util.spec_from_file_location("provenance_replay", (ROOT / "freshcache/simulation/provenance_replay.py")); PR = importlib.util.module_from_spec(spec); spec.loader.exec_module(PR)
PRI = C.streams(E)["primary"]; assert DS.check("primary_requests", len(PRI))
REC = {r["query_id"]: r for r in E["recs"]}; CL = lambda q: REC[q].get("cluster_id") or q
BASE = [(t, r) for t, r in PRI if not r.get("is_paraphrase")]; PARA = [(t, r) for t, r in PRI if r.get("is_paraphrase")]
assert DS.check("primary_base_requests", len(BASE)) and len({CL(r["query_id"]) for _, r in BASE}) == len(BASE) and DS.check("paraphrase_requests", len(PARA))
assert all(CL(r["query_id"]) in {CL(b["query_id"]) for _, b in BASE} for _, r in PARA)
def sample(share, seed):
    rng = random.Random(seed); P = round(len(BASE) * share / (1 - share)); byc = collections.defaultdict(list)
    for t, r in PARA: byc[r["freshness_class"]].append(r["query_id"])
    Nc = {c: len(v) for c, v in byc.items()}; tot = sum(Nc.values()); raw = {c: P * n / tot for c, n in Nc.items()}; Q = {c: int(math.floor(v)) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: (-(raw[c] - Q[c]), rng.random())):
        if sum(Q.values()) >= P: break
        Q[c] += 1
    keep = set()
    for c in sorted(byc):
        cl = collections.defaultdict(list)
        for q in byc[c]: cl[CL(q)].append(q)
        keys = sorted(cl); e = {k: Q[c] * len(cl[k]) / Nc[c] for k in keys}; k_ = {k: int(math.floor(e[k])) for k in keys}; rem = Q[c] - sum(k_.values())
        frac = [(k, e[k] - k_[k]) for k in keys if e[k] - k_[k] > 0]; rng.shuffle(frac)   # systematic PPS on fractional parts, random order
        if rem > 0:
            tf = sum(f for _, f in frac); step = tf / rem; u = rng.random() * step; cum = 0.0; marks = [u + i * step for i in range(rem)]; mi = 0
            for k, f in frac:
                cum += f
                while mi < len(marks) and marks[mi] < cum: k_[k] += 1; mi += 1
        for k in keys:
            if k_[k]: keep.update(rng.sample(sorted(cl[k]), min(k_[k], len(cl[k]))))
    assert len(keep) == sum(Q.values()) == P, (len(keep), P)
    return [(t, r) for t, r in PRI if (not r.get("is_paraphrase")) or r["query_id"] in keep], Q
def w4(S, per, byq):
    ch = un = 0
    for q in S:
        if per.get(q) is None: continue
        pages = []
        for u, tc, tr, lab in byq.get(q, []):
            p_ = RT.get(u) or {}; rc, rs = me.round_at(p_, tc), me.round_at(p_, tr); pages.append((rc is not None and rc == rs, lab))
        if any((not s) and l == me.CHANGED for s, l in pages): ch += 1
        elif pages and all((not s) and l == me.UNCHANGED for s, l in pages): un += 1
    return {"cross_round_determinate": ch + un, "cross_round_changed": ch, "cross_round_drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None, "cross_round_coverage_pct": round(100 * (ch + un) / len(S), 4) if S else None}
_o = me.outcome
def run(cond, seed, stream, quota=None):
    evs = []
    def rec(uh, tc, tr, rounds): l = _o(uh, tc, tr, rounds); evs.append((uh, tc, tr, l)); return l
    S = E["support_of"](stream, RT); me.outcome = rec
    try: m, per, tr = C.metrics(E, stream, "FreshCache", S)
    finally: me.outcome = _o
    t2q = {t: r["query_id"] for t, r in stream}; byq = collections.defaultdict(list)
    for uh, tc, trq, l in evs: byq[t2q[trq]].append((uh, tc, trq, l))
    prov = PR.run(E, stream); g = all(prov[q]["tier"] == tr[q]["tier"] for q in tr) and sum(x["fetched"] for x in prov.values()) == m["fetches"] and sum(len(x.get("l3_reuse_writers", [])) for x in prov.values()) == m["l3_hits"]
    within = cross = 0
    for t, r in stream:
        q = r["query_id"]; x = prov[q]
        if x["tier"] == "L1": within += CL(x["l1_entry_creator"]) == CL(q); cross += CL(x["l1_entry_creator"]) != CL(q)
        elif x["tier"] == "L2": within += CL(x["l2_entry_creator"]) == CL(q); cross += CL(x["l2_entry_creator"]) != CL(q)
    n = len(stream); npara = sum(1 for _, r in stream if r.get("is_paraphrase")); sc = sum(1 for q in tr if tr[q]["tier"] in ("miss", "REAL_TIME_BYPASS"))
    served = sum(x.get("served", 0) for x in prov.values() if x["tier"] in ("L2", "miss"))
    fcd = collections.Counter(r["freshness_class"] for _, r in stream)
    return {"condition": cond, "seed": seed, "requests": n, "base_requests": n - npara, "paraphrase_requests": npara, "paraphrase_share_pct": round(100 * npara / n, 4), "unique_clusters": len({CL(r["query_id"]) for _, r in stream}),
            "search_saved_pct": m["search_saved_pct"], "fresh_searches": sc, "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
            "l1_hit_rate_pct": round(100 * m["l1_hits"] / n, 4), "l2_hit_rate_pct": round(100 * m["l2_hits"] / n, 4), "l3_hits_per_request": round(m["l3_hits"] / n, 4), "l3_page_reuse_rate_pct": round(100 * m["l3_hits"] / served, 4) if served else None,
            "fetches": m["fetches"], "fetch_per_1k": m["fetch_per_1k"], "generations": m["generations"], "gen_per_1k": m["gen_per_1k"],
            "avoided_searches": m["l1_hits"] + m["l2_hits"], "avoided_within_cluster": within, "avoided_cross_cluster": cross,
            "within_cluster_share_of_avoided_pct": round(100 * within / (within + cross), 4) if within + cross else None,
            "savings_within_cluster_pp": round(100 * within / n, 4), "savings_cross_cluster_pp": round(100 * cross / n, 4),
            "step_drift_pct_SECONDARY": m["drift_pct"], "step_coverage_pct_SECONDARY": m["coverage_pct"], "step_changed": m["changed"], "step_unchanged": m["unchanged"], "support_S": len(S),
            **{f"audit_{k}": v for k, v in w4(S, dict(per), byq).items()}, "provenance_transcription_matches_engine": g,
            "class_distribution": json.dumps(dict(sorted(fcd.items()))), "paraphrase_quota_by_class": json.dumps(quota) if quota else None}
RUNS, G = [], {}
rA = run("A_0pct", None, BASE); RUNS.append(rA); BASE_ONLY_A = json.load(open(BASE_ONLY_DIR / "results/base_only_primary_metrics.json"))
G["A reproduces the base-only primary replay exactly"] = all(rA[k] == BASE_ONLY_A[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "l3_hits", "fetches", "generations", "fetch_per_1k", "gen_per_1k")) and (rA["step_drift_pct_SECONDARY"], rA["step_coverage_pct_SECONDARY"], rA["fresh_searches"]) == (BASE_ONLY_A["drift_pct"], BASE_ONLY_A["coverage_pct"], BASE_ONLY_A["search_calls"])
G["A: all reuse cross-cluster (one request per cluster)"] = rA["avoided_within_cluster"] == 0
log("A", json.dumps({k: rA[k] for k in ("requests", "search_saved_pct", "avoided_cross_cluster", "audit_cross_round_drift_pct")}))
rD = run("D_full", None, PRI); RUNS.append(rD); FR = json.load(open(CRR / "results/replays_reported.json"))["primary/FreshCache"]; CROSS_ROUND_DIR = json.load(open((OUTPUT_ROOT / "cross_round_audit/cross_round_summary.json")))["per_policy"]["FreshCache"]
G["D reproduces frozen primary FreshCache"] = all(rD[k] == FR[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "l3_hits", "fetches", "generations", "fetch_per_1k", "gen_per_1k")) and (rD["step_drift_pct_SECONDARY"], rD["step_coverage_pct_SECONDARY"]) == (FR["drift_pct"], FR["coverage_pct"])
G["D cross-round drift equals the cross-round audit (drift, determinate N)"] = (rD["audit_cross_round_drift_pct"], rD["audit_cross_round_determinate"]) == (CROSS_ROUND_DIR["reported_drift_pct"], CROSS_ROUND_DIR["reported_cross_round_determinate_N"])
DEC = {r["metric"]: r for r in csv.DictReader(open(BASE_ONLY_DIR / "results/full_trace_cluster_decomposition.csv")) if r["stream"] == "primary"}
G["D within/cross split equals the locality decomposition"] = (rD["avoided_within_cluster"], rD["avoided_cross_cluster"]) == (int(DEC["search avoidance (L1+L2 hits)"]["same_cluster"]), int(DEC["search avoidance (L1+L2 hits)"]["cross_cluster"]))
log("D", json.dumps({k: rD[k] for k in ("requests", "search_saved_pct", "avoided_within_cluster", "avoided_cross_cluster", "audit_cross_round_drift_pct")}))
for k, v in G.items(): log("GATE", k, v)
if not all(G.values()): json.dump({"gates": G, "runs": RUNS}, open(O / "work/STOPPED.json", "w"), indent=1); log("STOP: reproduction gate failed"); sys.exit(1)
SAMPLES = {}
for cond, s in SHARES.items():
    for seed in SEEDS:
        st, Q = sample(s, seed); SAMPLES[f"{cond}|{seed}"] = sorted(r["query_id"] for _, r in st if r.get("is_paraphrase"))
        assert [t for t, _ in st] == sorted(t for t, _ in st)
        r_ = run(cond, seed, st, Q); RUNS.append(r_); log(cond, seed, json.dumps({k: r_[k] for k in ("requests", "paraphrase_share_pct", "search_saved_pct", "avoided_within_cluster", "avoided_cross_cluster", "audit_cross_round_drift_pct", "provenance_transcription_matches_engine")}))
json.dump(SAMPLES, open(O / "work/retained_paraphrase_ids.json", "w"))
G["provenance transcription matches engine in every run"] = all(r["provenance_transcription_matches_engine"] for r in RUNS)
G["B/C: all base requests kept in every run"] = all(r["base_requests"] == len(BASE) for r in RUNS)
json.dump({"gates": G, "runs": RUNS}, open(O / "work/runs.json", "w"), indent=1)
cols = list(RUNS[0])
with open(O / "paraphrase_share_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(RUNS)
log("runs written", len(RUNS), "gates", json.dumps(G))
