"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

SemanticTTL / CategoryAware-style scripts/baselines, stage 1 (validation only): design freeze and validation sweeps.
Part A: SemanticTTL at theta=0.80 over the recovered kappa grid. Part B: CategoryAware-style per-class sweeps. Also evaluates the gates,
builds answer keys for the frozen validation quality set and writes generation tasks for missing answers."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, datetime, hashlib, json, sys, os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_category_aware"))
import baseline_selection_helpers as X
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
for d in ("work", "gpu/in", "gpu/out"): (X.O / d).mkdir(parents=True, exist_ok=True)
KG = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]; TG = [0.40, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]; CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST"]
DESIGN = {"written_utc": now(), "status": "baseline diagnostics; FreshCache is not retuned and no new primary FreshCache setting is selected",
    "partA": {"method": "SemanticTTL (existing semantics: l1 sim_ttl, no L2/L3; theta = L1 similarity threshold, kappa multiplies the class TTLs)", "theta": 0.80,
              "kappa_grid": KG, "kappa_grid_source": "recovered: outputs/semanticttl_joint_grid/joint_validation_grid.csv (42-cell grid, 1/16..2)",
              "objective": "documented quality-aware selection (outputs/quality_selection, Table 16; selection_and_8b.json quality_selection_reported): screened validation requests, 2-judge rule, C-WAI limit = FreshCache WAI / fresh-correct (primary) and +5 pp (relaxed)",
              "A1": "among theta=0.80 cells: C-WAI <= FreshCache -> max validation savings; else C-WAI <= FreshCache+5pp (relaxed) -> max savings; else lowest WAI (ties: higher savings)",
              "A2": "validation search savings closest to FreshCache validation savings; ties: lower WAI, then higher savings"},
    "partB": {"name": "CategoryAware-style (not a faithful reproduction of arXiv:2510.26835)", "verified_source": "arXiv:2510.26835 (Wang, Liu, Zhu, Youssef, Nagpurkar, Chen): per-category similarity threshold, TTL and quota; shared HNSW index; category from request metadata; response cache; no empirical scripts/evaluation",
              "implemented": "answer-only shared cache; threshold from the request's category, TTL (frozen base class TTL x kappa) from the entry's category; category = frozen FreshCache freshness class (available at request time); REAL_TIME bypass as SemanticTTL",
              "not_implemented": "HNSW index, external storage tier, per-category quotas/eviction, load-adaptive threshold/TTL policies",
              "grid_per_category": {"theta": TG, "kappa": KG, "source": "42-cell 42-cell (theta, kappa) grid"},
              "selection": "validation only. Stage 1: for each category c, replay the category-c validation sub-stream (other categories absent) for every grid cell; choose the cell with maximum category-c search savings subject to category-c WAI <= FreshCache's category-c WAI on the screened set (fresh-correct -> WRONG). If none is feasible, the cell with the lowest category-c WAI (ties: higher savings). Stage 2: combine the four selections into one configuration (frozen), replay it once on validation with the shared cache and report; no re-tuning.",
              "note": "the separable per-category selection is a tractable approximation; cross-category hits in the shared cache are measured in the combined validation replay"}}
DESIGN["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in DESIGN.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (X.O / "design_frozen.json").exists(): json.dump(DESIGN, open(X.O / "design_frozen.json", "w"), indent=1)
DESIGN = json.load(open(X.O / "design_frozen.json")); X.log(f"[{now()}] design sha256 {DESIGN['sha256']}")
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/method_replay.py",
        "freshcache/replay_engine/engine_loader.py", "freshcache/replay_engine/answer_pools.py", "scripts/evaluation/generation_judging_stage.py", "freshcache/method/tiered_cache_model.py", "freshcache/simulation/page_version_replay.py",
        "scripts/evaluation/answer_audits/quality_selection_sample.py", "outputs/quality_selection/q_sample.jsonl", "outputs/quality_selection/q_verification.csv",
        "outputs/quality_selection/q_answers.jsonl", "outputs/quality_selection/q_judge_llama8b.jsonl", "outputs/quality_selection/q_judge_qwen7b.jsonl",
        "outputs/semanticttl_joint_grid/joint_validation_grid.csv", "outputs/replay/results/selection_and_8b.json", "outputs/replay/results/quality_selection_keys_reported.json",
        "outputs/replay/results/audit_keys_reported.json", "outputs/replay/results/answer_quality.json", "outputs/replay/results/replays_reported.json",
        "outputs/replay/results/extras_reported.json", "outputs/general_audit/sample.json", "outputs/validation_split/split.json",
        "outputs/frozen_inputs/generated/request_lists/model_majority_valid.csv", "data/derived/round_table.jsonl", "data/queries.jsonl", "data/sim_matrix_bgem3.npy",
        "outputs/semanticttl_frontier/validation/quality_keys.json", "outputs/semanticttl_frontier/validation/request_level_quality.csv",
        "outputs/semanticttl_frontier/validation/replay_metrics.json", "outputs/semanticttl_frontier/gpu/out/val_answers_llama3b.jsonl",
        "outputs/semanticttl_frontier/gpu/out/ho_answers_llama3b.jsonl"]
if not (X.O / "work/read_only_before.json").exists(): json.dump({p: X.fstat(inpath(p)) for p in READ}, open(X.O / "work/read_only_before.json", "w"), indent=1)
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(X.ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
BEF = json.load(open(X.O / "work/read_only_before.json"))
if any(p in FH and FH[p] != v["sha256"] for p, v in BEF.items()): X.log("STOP: frozen input differs"); sys.exit(1)
E = X.engine(); exp, p2 = E["exp"], E["p2"]; h, ctx_for = p2.h, p2.ctx_for; Q2 = X.q2()
V = X.C.streams(E)["validation"]; assert DS.check("validation_requests", len(V)); S = E["support_of"](V, E["rounds"])
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
EV = sorted(r["query_id"] for r in csv.DictReader(open((OUTPUT_ROOT / "quality_selection/q_verification.csv"))) if r.get("evidence_valid") == "1")
STTL_FRONTIER_KEYS = json.load(open(X.STTL_FRONTIER_DIR / "validation/quality_keys.json")); G = {}
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); fcm, _, _ = X.C.metrics(E, V, "FreshCache", S); G["FreshCache validation savings equals the recorded reference"] = REF.check("reported_engine/validation/FreshCache/search_saved_pct", fcm["search_saved_pct"])
def key_for(p, query):
    ctx = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", query); return ["gen", h(ctx), h(gq)], ctx, gq
GRID42 = {(float(r["theta"]), float(r["kappa"])): r for r in csv.DictReader(open((OUTPUT_ROOT / "semanticttl_joint_grid/joint_validation_grid.csv")))}
A, KEYS, TASKS = {}, {}, {}
for k in KG:
    with X.patched(0.80, k): m, per, tr = X.C.metrics(E, V, "SemanticTTL", S)
    with X.patched(0.80, k) as pt: mt, rows = Q2.sttl_replay(V, 0.80, k, pt.ottl)
    ca_m, ca_rows = X.ca_replay(V, {c: 0.80 for c in CLASSES}, {c: k for c in CLASSES})
    G[f"A theta0.80 k{k}: engine == sttl_replay savings"] = m["search_saved_pct"] == mt["search_saved_pct"]
    G[f"A theta0.80 k{k}: savings == 42-cell grid"] = abs(m["search_saved_pct"] - float(GRID42[(0.8, k)]["search_saved_pct"])) < 1e-9
    G[f"CA transcription (uniform theta0.80 k{k}) == sttl_replay rows"] = all(ca_rows[q]["tier"] == rows[q]["tier"] and [tuple(x) for x in ca_rows[q]["ev"]] == [tuple(x) for x in rows[q]["ev"]]
                                                                              and ca_rows[q]["gen_query"] == rows[q]["gen_query"] for q in rows) and ca_m["search_saved_pct"] == m["search_saved_pct"]
    cid = f"sttl_t0.80_k{k:g}"; A[cid] = {"theta": 0.80, "kappa": k, **{x: m[x] for x in ("search_saved_pct", "l1_hits", "fetches", "generations", "fetch_per_1k", "gen_per_1k", "drift_pct", "coverage_pct")},
                                          "search_calls": mt["search_calls"]}
    KEYS[cid] = {}
    for q in EV:
        kk, ctx, gq = key_for(rows[q], QS[q]["query"]); KEYS[cid][q] = {"key": kk, "tier": rows[q]["tier"], "class": QS[q]["fc"]}; TASKS.setdefault(tuple(kk), {"key": kk, "question": gq, "context": ctx})
    frontier_id = f"sttl_t0.80_k{k:g}"
    if frontier_id in STTL_FRONTIER_KEYS["configs"]: G[f"A {cid}: keys equal STTL_FRONTIER_DIR existing keys"] = all(KEYS[cid][q]["key"] == STTL_FRONTIER_KEYS["configs"][frontier_id][q]["key"] for q in EV)
X.log("A validation", json.dumps(A))
# ---------------- Part B stage 1: per-category sub-stream sweeps
B1, BK = {}, {}
for c in CLASSES:
    sub = [(t, r) for t, r in V if r["freshness_class"] == c]; evc = [q for q in EV if QS[q]["fc"] == c]
    for th in TG:
        for k in KG:
            m, rows = X.ca_replay(sub, {c: th}, {c: k}); cid = f"{c}|t{th:.2f}|k{k:g}"
            B1[cid] = {"class": c, "theta": th, "kappa": k, "class_requests": len(sub), "search_saved_pct_class": m["search_saved_pct"], "l1_hits": m["l1_hits"], "searches": m["search_calls"]}
            BK[cid] = {}
            for q in evc:
                kk, ctx, gq = key_for(rows[q], QS[q]["query"]); BK[cid][q] = {"key": kk, "tier": rows[q]["tier"]}; TASKS.setdefault(tuple(kk), {"key": kk, "question": gq, "context": ctx})
X.log("B stage1 cells", len(B1))
json.dump({"gates": G}, open(X.O / "work/gates_stage1.json", "w"), indent=1)
for k_, v in G.items(): X.log("GATE", k_, v)
if not all(G.values()): X.log("STOP: gate failed"); sys.exit(1)
lab = X.Val(); todo = [t for k_, t in TASKS.items() if lab.answer(t["key"]) is None]
X.wjl(X.O / "gpu/in/val_gen_tasks.jsonl", todo)
json.dump({"A_ops": A, "A_keys": KEYS, "B1_ops": B1, "B1_keys": BK, "EV": EV, "freshcache_validation": fcm}, open(X.O / "work/stage1.json", "w"))
X.log(f"validation keys: {len(TASKS)} distinct; new answers needed: {len(todo)}")
