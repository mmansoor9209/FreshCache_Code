"""Systems cost summary: Part A (existing same-environment concurrency runs, exported; no new timing), Part B (exact held-out operation counts
for six frozen methods by replay, gated to frozen outputs), Part C (one common component-cost model; coefficients recovered from the systems
timing and end-to-end latency runs; method-specific local overhead measured now with one fixed protocol), Part D (outputs). Read-only on
every pre-existing file."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import ast, collections, copy, csv, datetime, hashlib, importlib.util, json, math, os, pathlib, platform, statistics as stt, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "concurrency_cost"); CRR = (OUTPUT_ROOT / "replay"); SYS_DIR = (OUTPUT_ROOT / "systems_timing")
BASELINE_SEL_DIR = (OUTPUT_ROOT / "baseline_selection")
LOGF = open(O / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/method_replay.py",
        "freshcache/replay_engine/engine_loader.py", "outputs/replay/results/replays_reported.json", "outputs/replay/results/extras_reported.json", "freshcache/method/tiered_cache_model.py",
        "data/derived/round_table.jsonl", "freshcache/simulation/request_schedules.py", "data/queries.jsonl", "data/url_manifest.jsonl", "data/sim_matrix_bgem3.npy", "outputs/validation_split/split.json",
        "scripts/evaluation/answer_audits/quality_selection_sample.py", "scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py", "outputs/baseline_selection/baseline_selection_points.csv", "outputs/baseline_selection/frozen_selection.json",
        "outputs/baseline_selection/baseline_selection_summary.json", "outputs/l2_operating_points/l2_threshold_points.csv", "outputs/systems_timing/results/break_even.json",
        "outputs/systems_timing/results/end_to_end_requests.csv", "outputs/systems_timing/README.md",
        "outputs/systems_timing/intermediate/generation_judging_stage.json", "freshcache/simulation/stepwise_replay.py", "configs/integrity/FILE_HASHES.csv"]
BEF = {p: fstat(inpath(p)) for p in READ}; json.dump(BEF, open(O / "work/read_only_before_final.json", "w"), indent=1)
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
if any(p in FH and FH[p] != v["sha256"] for p, v in BEF.items()): log("STOP: frozen input differs"); sys.exit(1)
log(f"[{now()}] cost-model final (Parts A-D) start")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
import numpy as np   # noqa: E402
E = C.load("reported"); exp, me, p2, RT = E["exp"], E["me"], E["p2"], E["rounds"]
ST = C.streams(E); H = ST["heldout"]; n = len(H); assert DS.check("heldout_requests", n); SH = E["support_of"](H, RT)
spec = importlib.util.spec_from_file_location("quality_selection_sample", ROOT / "scripts/evaluation/answer_audits/quality_selection_sample.py"); Q2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q2); assert Q2.P2 is p2
# baseline_selection_helpers definitions extracted read-only (importing baseline_selection_helpers would append to its run log)
src = open((ROOT / "scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py")).read(); NS = {"np": np, "copy": copy, "engine": lambda: E}
for node in ast.parse(src).body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("patched", "ca_replay"): exec(compile(ast.Module([node], []), "baseline_selection_helpers.py", "exec"), NS)
patched, ca_replay = NS["patched"], NS["ca_replay"]
BASELINE_POINTS = {r["arm"]: r for r in csv.DictReader(open(BASELINE_SEL_DIR / "baseline_selection_points.csv"))}; FZ3 = json.load(open(BASELINE_SEL_DIR / "frozen_selection.json"))
FR = json.load(open(CRR / "results/replays_reported.json"))["heldout/FreshCache"]
G = {}; OPS = {}
own_total = sum(len(r["urls"]) for _, r in H); nonrt = sum(1 for _, r in H if r["freshness_class"] != "REAL_TIME")
OPS["NoCache"] = {"search": n, "fetch": own_total, "gen": n, "l1_hits": 0, "l2_hits": 0, "l3_hits": 0, "embeds": 0, "source": "direct: every request searches, fetches its own URL list and generates"}
def tiers_search(tr): return sum(1 for q in tr if tr[q]["tier"] in ("miss", "REAL_TIME_BYPASS"))
m, per, tr = C.metrics(E, H, "FreshCache", SH)
OPS["FreshCache"] = {"search": tiers_search(tr), "fetch": m["fetches"], "gen": m["generations"], "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"], "embeds": nonrt, "saved_check": m["search_saved_pct"],
                     "source": "mixed_age_replay.replay (frozen reported), held-out"}
G["FreshCache counts equal frozen replays_reported heldout"] = all(m[k] == FR[k] for k in ("search_saved_pct", "fetches", "generations", "l1_hits", "l2_hits", "l3_hits"))
STTL = {"SemanticTTL (.60,1/2)": (0.60, 0.5, "SemanticTTL_0.60_k0.5_fixed"), "SemanticTTL (.80,1/4)": (0.80, 0.25, "A1"), "SemanticTTL (.80,2)": (0.80, 2.0, "A2")}
for name, (th, ka, arm) in STTL.items():
    with patched(th, ka): mm, pp, tt = C.metrics(E, H, "SemanticTTL", SH)
    with patched(th, ka) as pt: mt, rows = Q2.sttl_replay(H, th, ka, pt.ottl)
    OPS[name] = {"search": tiers_search(tt), "fetch": mm["fetches"], "gen": mm["generations"], "l1_hits": mm["l1_hits"], "l2_hits": 0, "l3_hits": 0, "embeds": nonrt, "saved_check": mm["search_saved_pct"],
                 "source": f"mixed_age_replay.replay SemanticTTL theta={th} kappa={ka} (frozen baseline-selection arm {arm})"}
    e3 = BASELINE_POINTS[arm]
    G[f"{name}: engine equals sttl_replay (savings, search calls)"] = mm["search_saved_pct"] == mt["search_saved_pct"] and OPS[name]["search"] == mt["search_calls"]
    G[f"{name}: equals frozen baseline-selection point (savings, fetch/1k, gen/1k)"] = abs(mm["search_saved_pct"] - float(e3["search_saved_pct"])) < 1e-9 and abs(round(1000 * mm["fetches"] / n, 1) - float(e3["fetch_per_1k"])) < 1e-9 and abs(round(1000 * mm["generations"] / n, 1) - float(e3["gen_per_1k"])) < 1e-9
CA = FZ3["CategoryAware_style"]["per_category"]; TH = {c: v["theta"] for c, v in CA.items()}; KA = {c: v["kappa"] for c, v in CA.items()}; TH["REAL_TIME"] = 1.1; KA["REAL_TIME"] = 0.0
mc, rc = ca_replay(H, TH, KA)
OPS["CategoryAware-style"] = {"search": mc["search_calls"], "fetch": mc["fetches"], "gen": mc["generations"], "l1_hits": mc["l1_hits"], "l2_hits": 0, "l3_hits": 0, "embeds": nonrt, "saved_check": mc["search_saved_pct"],
                              "source": "baseline-selection ca_replay transcription with the frozen per-category selection"}
G["CategoryAware-style equals frozen baseline-selection point"] = abs(mc["search_saved_pct"] - float(BASELINE_POINTS["CategoryAware_style"]["search_saved_pct"])) < 1e-9 and mc["fetches"] == int(BASELINE_POINTS["CategoryAware_style"]["fetches"])
QUAL = {"NoCache": (0, 0.0, "fresh reference arm of the frozen 400 audit (WAI = 0 by definition)"), "FreshCache": (int(BASELINE_POINTS["FreshCache"]["audit_WAI"]), float(BASELINE_POINTS["FreshCache"]["audit_C_WAI_pct"]), "frozen 400 audit"),
        "SemanticTTL (.60,1/2)": (int(BASELINE_POINTS["SemanticTTL_0.60_k0.5_fixed"]["audit_WAI"]), float(BASELINE_POINTS["SemanticTTL_0.60_k0.5_fixed"]["audit_C_WAI_pct"]), "frozen 400 audit (baseline-selection)"),
        "SemanticTTL (.80,1/4)": (int(BASELINE_POINTS["A1"]["audit_WAI"]), float(BASELINE_POINTS["A1"]["audit_C_WAI_pct"]), "frozen 400 audit (baseline-selection A1)"),
        "SemanticTTL (.80,2)": (int(BASELINE_POINTS["A2"]["audit_WAI"]), float(BASELINE_POINTS["A2"]["audit_C_WAI_pct"]), "frozen 400 audit (baseline-selection A2)"),
        "CategoryAware-style": (int(BASELINE_POINTS["CategoryAware_style"]["audit_WAI"]), float(BASELINE_POINTS["CategoryAware_style"]["audit_C_WAI_pct"]), "frozen 400 audit (baseline-selection)")}
# ---------------- C_local: one fixed protocol (decision-only replay over the same held-out stream, single thread, 3 reps; rep 1 cold)
_out = me.outcome
def stub(uh, tc, tr_, rounds): return me.UNOBS   # drift labelling is not cache work and does not affect decisions
def tm(fn, reps=3):
    v = []
    for _ in range(reps):
        me.outcome = stub
        try: t0 = time.perf_counter(); fn(); v.append((time.perf_counter() - t0) * 1e3)
        finally: me.outcome = _out
    return v
LOCAL = {"NoCache": {"total_ms": [0.0, 0.0, 0.0], "protocol": "no cache: no lookup or write"}}
LOCAL["FreshCache"] = {"total_ms": tm(lambda: me.replay(H, RT, "FreshCache")), "protocol": "mixed_age_replay.replay FreshCache, outcome() stubbed"}
for name, (th, ka, arm) in STTL.items():
    def f(th=th, ka=ka):
        with patched(th, ka): me.replay(H, RT, "SemanticTTL")
    LOCAL[name] = {"total_ms": tm(f), "protocol": f"mixed_age_replay.replay SemanticTTL (theta={th}, kappa={ka}), outcome() stubbed"}
    def g(th=th, ka=ka):
        with patched(th, ka) as pt: Q2.sttl_replay(H, th, ka, pt.ottl)
    LOCAL[name]["cross_check_sttl_replay_total_ms"] = tm(g)
LOCAL["CategoryAware-style"] = {"total_ms": tm(lambda: ca_replay(H, TH, KA)), "protocol": "baseline-selection ca_replay (decision-only transcription)"}
G["engine restored"] = me.outcome is _out
for k, v in LOCAL.items():
    v["in_process_after_count_replays_first_pass_ms_per_request"] = round(v["total_ms"][0] / n, 4); v["warm_ms_per_request_mean"] = round(stt.mean(v["total_ms"][1:]) / n, 4)
    fp = O / f"work/local_cold_{k.replace(' ', '_').replace('/', '-')}.json"
    if k == "NoCache": v["cold_ms_per_request"] = 0.0
    else:
        fr = json.load(open(fp)); v["cold_ms_per_request"] = fr["cold_ms_per_request"]; v["fresh_process_warm_ms_per_request"] = fr["warm_ms_per_request_mean"]; v["protocol"] += "; C_local = fresh-process cold pass (cold_lookup_timing.py)"
# ---------------- coefficients (one set for every method)
BE = json.load(open(SYS_DIR / "results/break_even.json")); COST_RUNS = json.load(open(O / "work/runs.json"))
fc1 = [r for r in COST_RUNS["runs"] if r["policy"] == "FreshCache" and r["concurrency"] == 1 and not r.get("sensitivity_io_sleep")]; nc1 = [r for r in COST_RUNS["runs"] if r["policy"] == "NoCache" and r["concurrency"] == 1 and not r.get("sensitivity_io_sleep")]
LIVE_T = json.load(open(OUTPUT_ROOT / "latency_e2e/stage_d_live_timings.json")); EMB_T = json.load(open(OUTPUT_ROOT / "latency_e2e/stage_b_gpu_timings.json"))["embed"]
MIC_T = json.load(open(SYS_DIR / "intermediate/cpu_micro_stability.json"))["local_page_read"]["run2"]
COEF = {"c_gen_ms": {"value": round(BE["scenarios"][0]["generation_ms"], 1), "provenance": "in-session generation (Llama-3.2-3B batch 1, FORCED, 80 tokens): outputs/systems_timing/results/break_even.json (component_cost_break_even.py)"},
        "c_fetch_ms": {"local": round(MIC_T["mean_ms"], 3), "live_p50": round(LIVE_T["web_fetch"]["p50_ms"], 2), "provenance": "local: snapshot read (outputs/systems_timing/intermediate/cpu_micro_stability.json); live: p50 of the live GETs in outputs/latency_e2e/stage_d_live_timings.json (latency_network_measurement.py; includes its fetch error rate)"},
        "c_search_ms": {"local": 0.0, "live_p50": round(LIVE_T["search_api"]["p50_ms"], 2), "provenance": "local: replayed URL lists (no external search executed); live: p50 of the live Serper calls in outputs/latency_e2e/stage_d_live_timings.json"},
        "c_embed_ms": {"value": round(EMB_T["mean_ms"], 3), "provenance": "outputs/latency_e2e/stage_b_gpu_timings.json BGE-M3 batch 1 mean (latency_gpu_measurement.py); applied to every non-REAL_TIME request of every similarity-based method",
                       "corroboration_in_session_ms": round(stt.mean(r["embed_ms_total_ms"] for r in fc1) / 512, 3) if fc1 else None},
        "C_local_lookup": {"protocol": "measured now: decision-only replay of each frozen method over the held-out requests, single thread; USED = first pass in a fresh process (cold, cold_lookup_timing.py); in-process and warm passes reported", "per_method_ms_per_request_cold": {k: v["cold_ms_per_request"] for k, v in LOCAL.items()}},
        "not_used_context": {"in_session_generation_c1_NoCache_ms": round(stt.mean(r["generation_ms_total_ms"] for r in nc1) / 512, 2) if nc1 else None,
                             "systems_FreshCache_lookup_overhead_cold_ms_per_request_primary": BE["overhead_ms_per_1k_excluding_embedding"]["cold"] / 1000}}
def model(o, lk, scen):
    k = 1000 / n; cs = COEF["c_search_ms"][scen]; cf = COEF["c_fetch_ms"][scen]
    parts = {"search": o["search"] * k * cs, "fetch": o["fetch"] * k * cf, "gen": o["gen"] * k * COEF["c_gen_ms"]["value"], "local_lookup": lk * 1000, "embedding": o["embeds"] * k * COEF["c_embed_ms"]["value"]}
    return {**{f"{p}_ms_per_1k": round(v, 1) for p, v in parts.items()}, "total_ms_per_1k": round(sum(parts.values()), 1), "ms_per_request": round(sum(parts.values()) / 1000, 2)}
METH = ["NoCache", "FreshCache", "SemanticTTL (.60,1/2)", "SemanticTTL (.80,1/4)", "SemanticTTL (.80,2)", "CategoryAware-style"]
TAB = []
for mth in METH:
    o = OPS[mth]; lk = LOCAL[mth]["cold_ms_per_request"]; rr = {"method": mth}
    rr.update({"search_saved_pct": round(100 * (1 - o["search"] / n), 4), "search_per_1k": round(1000 * o["search"] / n, 1), "fetch_per_1k": round(1000 * o["fetch"] / n, 1), "gen_per_1k": round(1000 * o["gen"] / n, 1),
               "WAI": QUAL[mth][0], "C_WAI_pct": QUAL[mth][1], "local_lookup_ms_per_request_cold": lk, "embedding_requests_per_1k": round(1000 * o["embeds"] / n, 1)})
    for sc in ("local", "live_p50"):
        mm_ = model(o, lk, sc); rr.update({f"{sc}_{k}": v for k, v in mm_.items()})
    TAB.append(rr)
nc = {sc: next(r for r in TAB if r["method"] == "NoCache")[f"{sc}_total_ms_per_1k"] for sc in ("local", "live_p50")}
for sc in ("local", "live_p50"): REF.record(f"cost/FreshCache_heldout/{sc}_ms_per_request_2dp", next(r for r in TAB if r["method"] == "FreshCache")[f"{sc}_ms_per_request"])
for r in TAB:
    for sc in ("local", "live_p50"): r[f"{sc}_relative_to_NoCache"] = round(r[f"{sc}_total_ms_per_1k"] / nc[sc], 4)
    o = OPS[r["method"]]; ncO = OPS["NoCache"]; k = 1000 / n
    ds = (ncO["search"] - o["search"]) * k
    other = (o["fetch"] - ncO["fetch"]) * k * COEF["c_fetch_ms"]["local"] + (o["gen"] - ncO["gen"]) * k * COEF["c_gen_ms"]["value"] + r["local_lookup_ms_per_request_cold"] * 1000 + o["embeds"] * k * COEF["c_embed_ms"]["value"]
    r["break_even_search_ms_local_fetch"] = (round(max(0.0, other / ds), 3) if ds > 0 else None)
# reproduce the systems-timing ~17 ms break-even from its own (primary-stream) inputs
PB = BE["per_1000_primary_requests"]; sc_ = next(s for s in BE["scenarios"] if s["lookup_timing"] == "cold" and s["embedding_assumption"].startswith("worst") and s["scenario"].startswith("local"))
rep = (sc_["overhead_ms_per_1k"] - sc_["saved_ms_per_1k"]) / sc_["searches_avoided_per_1k"]
G["break-even reproduced from its stored (0.1-ms-rounded) inputs within rounding (tol 0.01 ms)"] = abs(rep - sc_["break_even_search_ms_given_fetch_and_generation"]) < 0.01
SYS_E2E = list(csv.DictReader(open(SYS_DIR / "results/end_to_end_requests.csv")))
def e2e(pol):
    v = [x for x in SYS_E2E if x["policy"] == pol]; tot = lambda k: sum(float(x[k]) for x in v if x.get(k) not in (None, ""))
    return {"requests": len(v), "ms_per_request": round(sum(float(x.get("embed_ms") or 0) + float(x.get("vector_search_ms") or 0) + sum(float(x.get(k) or 0) for k in ("l1_lookup", "l2_lookup", "l3_lookup", "cache_write_update")) + float(x.get("page_fetch_local_ms") or 0) + float(x.get("context_build_ms") or 0) + float(x.get("generation_ms") or 0) + float(x.get("stored_answer_read_ms") or 0) for x in v) / len(v), 1)}
OLD = {"FreshCache": e2e("FreshCache"), "NoCache": e2e("NoCache")}
for k, v in G.items(): log("GATE", k, v)
if not all(G.values()): json.dump({"gates": G, "ops": OPS}, open(O / "work/STOPPED_final.json", "w"), indent=1, default=str); log("STOP: gate failed"); sys.exit(1)
# ---------------- Part A export
RUNS = COST_RUNS["runs"]; CON = []
for r in RUNS:
    CON.append({"policy": r["policy"], "concurrency": r["concurrency"], "rep": r["rep"], "sensitivity_io_sleep": r.get("sensitivity_io_sleep"), "n_requests": r.get("n_requests"), "wall_s": r.get("wall_s"),
                "throughput_rps": r.get("throughput_rps"), "latency_mean_ms": r.get("latency_mean_ms"), "latency_p50_ms": r.get("latency_median_ms"), "latency_p95_ms": r.get("latency_p95_ms"), "latency_p99_ms": r.get("latency_p99_ms"),
                "searches": r.get("searches"), "fetches": r.get("fetches"), "generations": r.get("generations"), "cache_lookups_reads": r.get("cache_reads"), "cache_writes": r.get("cache_writes"),
                "cpu_util_process_pct_of_one_core": r.get("proc_cpu_util_pct_of_one_core"), "cpu_util_system_pct": r.get("system_cpu_util_pct_mean"), "gpu_util_pct": r.get("gpu_util_pct_mean"),
                "gpu_mem_peak_MiB": r.get("gpu_mem_used_MiB_peak"), "rss_peak_MiB": r.get("rss_MiB_peak"), "failures_timeouts": r.get("errors"), "other_gpu_processes": r.get("other_gpu_processes"),
                "semantic_equivalence": bool(r.get("gate_tiers_equal_logical") and r.get("gate_contexts_equal_ctx_for") and r.get("gate_answers_equal_c1_rep1"))})
with open(O / "concurrency_results.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(CON[0])); w.writeheader(); w.writerows(CON)
AGG = {}
for pol in ("NoCache", "FreshCache"):
    for c in (1, 2, 4, 8, 16, 32):
        rs = [r for r in CON if r["policy"] == pol and r["concurrency"] == c and not r["sensitivity_io_sleep"]]
        AGG[f"{pol}|{c}"] = {k: {"mean": round(stt.mean(r[k] for r in rs), 3), "sd": round(stt.stdev(r[k] for r in rs), 3)} for k in ("wall_s", "throughput_rps", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "cpu_util_process_pct_of_one_core", "gpu_util_pct", "rss_peak_MiB")}
        AGG[f"{pol}|{c}"].update({"reps": len(rs), "failures": sum(r["failures_timeouts"] for r in rs), "searches": rs[0]["searches"], "fetches": rs[0]["fetches"], "generations": rs[0]["generations"],
                                  "cache_reads": rs[0]["cache_lookups_reads"], "cache_writes": rs[0]["cache_writes"], "all_semantically_equivalent": all(r["semantic_equivalence"] for r in rs)})
split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"))); TC = set(split["test_clusters"]); PRI = ST["primary"]; W = COST_RUNS["env"]["window"]
win = PRI[W["start_index"]:W["start_index"] + W["size"]]; held_share = sum(1 for _, r in win if (r.get("cluster_id") or r["query_id"]) in TC)
# ---------------- outputs
with open(O / "baseline_serving_cost.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(TAB[0])); w.writeheader(); w.writerows(TAB)
AFT = {p: fstat(inpath(p)) for p in BEF}; UNCH = {p: BEF[p] == AFT[p] for p in BEF}
B0 = json.load(open(O / "work/read_only_before.json")); UNCH0 = {p: fstat(inpath(p)) == v for p, v in B0.items()}
COMP = {"coefficients": COEF, "local_overhead_measurements": LOCAL, "model": "C_1000 = N_search*c_search + N_fetch*c_fetch + N_gen*c_gen + C_local; C_local = lookup (measured) + embedding (non-REAL_TIME requests x c_embed)",
        "scenarios": {"local": f"c_search 0 ms, c_fetch {COEF['c_fetch_ms']['local']} ms (no external search, disk-resident pages)", "live_p50": f"c_search {COEF['c_search_ms']['live_p50']} ms, c_fetch {COEF['c_fetch_ms']['live_p50']} ms (same-host live medians)"},
        "break_even_reproduction": {"inputs": {k: sc_[k] for k in ("overhead_ms_per_1k", "saved_ms_per_1k", "searches_avoided_per_1k", "embedding_ms", "lookup_timing", "scenario")}, "break_even_ms": round(rep, 4), "reported_by_systems_ms": sc_["break_even_search_ms_given_fetch_and_generation"], "note": "difference is rounding of the stored inputs (0.1 ms/1k); no implementation error"}}
json.dump(COMP, open(O / "component_costs.json", "w"), indent=1)
ENV = COST_RUNS["env"]
SUM = {"generated_utc": now(), "partA": {"source": "existing cost-model concurrency runs (work/runs.json; design work/../design_frozen.json); no new timing", "aggregates": AGG, "window": W, "window_heldout_requests": held_share,
       "env": ENV, "concurrency_only_report_archived": "work/concurrency_report_archived.md"}, "partB_ops": OPS, "partB_quality": QUAL, "partC_table": TAB, "component_costs": COMP,
       "end_to_end_diagnostic_ms_per_request": OLD, "gates": G,
       "freshcache_or_baseline_configuration_changed": False, "read_only_files_final": {p: {**BEF[p], "unchanged": UNCH[p]} for p in BEF}, "read_only_files_partA": {p: UNCH0[p] for p in B0},
       "all_read_only_unchanged": all(UNCH.values()) and all(UNCH0.values())}
json.dump(SUM, open(O / "serving_cost_summary.json", "w"), indent=1, default=str)
log(json.dumps({r["method"]: {k: r[k] for k in ("search_saved_pct", "fetch_per_1k", "gen_per_1k", "local_ms_per_request", "live_p50_ms_per_request", "break_even_search_ms_local_fetch", "WAI", "C_WAI_pct")} for r in TAB}))
log("LOCAL", json.dumps({k: (v["cold_ms_per_request"], v["warm_ms_per_request_mean"]) for k, v in LOCAL.items()}), "| break-even reproduced", round(rep, 3), "| end-to-end diagnostic", json.dumps(OLD), "| held-out share of cost-model window", held_share)
log("all read-only unchanged", SUM["all_read_only_unchanged"])
