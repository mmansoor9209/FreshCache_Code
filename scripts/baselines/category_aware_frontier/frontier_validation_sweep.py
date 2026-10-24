"""CategoryAware-style comparator: our reimplementation of the per-category similarity threshold and TTL policy of
category-aware semantic caching (Wang et al., 2025), not the complete published system; it omits quotas, the storage tier,
indexing and load adaptation.

CategoryAware frontier, stage 1 (validation only; no held-out request is replayed or labelled here).
Freezes and hashes the design and candidate grid first, then runs validation replays of all candidates and of FreshCache, builds answer
keys for the frozen validation quality set, and writes generation tasks for missing answers."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, json, os, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/category_aware_frontier"))
import frontier_helpers as X
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
DTH = [0.0, 0.025, 0.05, 0.075, 0.10]; MT = [1.0, 0.75, 0.50, 0.25]
GRID = [{"id": f"ca_d{d:+.3f}_m{m:.2f}", "dtheta": d, "m_ttl": m, **dict(zip(("theta", "kappa"), X.ca_params(d, m)))} for d in DTH for m in MT]
DESIGN = {"status": "sensitivity/frontier study added after the primary experiment; CategoryAware-frontier-selected configurations are NOT new primary scripts/baselines and were not selected before the original held-out study; selection uses validation only and is frozen before CategoryAware-frontier held-out scoring",
  "freshcache": "primary theta_L2 = 0.75 unchanged; theta_L2 = 0.80 and verifier gamma = 0.55 are sensitivities (frozen L2-sweep / reported verifier replays and labels)",
  "categoryaware_recovered": {"implementation": "baseline-selection ca_replay (scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py): answer-only shared cache; similarity threshold from the REQUEST's class, TTL = frozen FIXED_TTL[class] x kappa[class] from the cached ENTRY's class; REAL_TIME bypass; L1 hit serves the stored answer generated from the entry's own evidence (gen_query = entry query, ev = entry's URLs at its time)",
                              "current_per_class": {c: {"theta": X.BASE_CA[c]["theta"], "kappa": X.BASE_CA[c]["kappa"]} for c in X.CLASSES}, "base_fixed_ttl_s": {"TIMELESS": 2592000, "SLOW": 604800, "MEDIUM": 86400, "FAST": 3600},
                              "original_validation_selection": "baseline-selection per-category sub-stream sweep over the 42-cell (theta, kappa) grid: max class savings s.t. class WAI <= FreshCache class WAI on the screened validation set; combined once",
                              "provenance_retained": "yes in the baseline-selection transcription (each entry keeps its creating request's URL hashes and version time); a response-only cache would not retain it, so the CategoryAware-frontier evidence assessment is favourable to CategoryAware"},
  "existing_grid_reusable": "no: the baseline-selection grid is a per-category SUB-STREAM sweep selected by per-class WAI; it does not enumerate combined configurations scorable against a combined C-WAI constraint. The predeclared stricter-only grid is used.",
  "grid": {"dtheta": DTH, "m_ttl": MT, "rule": "theta_c + dtheta (capped at 0.99), kappa_c x m_ttl, same for every class, REAL_TIME bypass", "n": len(GRID)},
  "validation": "validation quality set (q_sample, evidence_valid); two-judge rule; WAI = fresh-correct -> WRONG; C-WAI = WAI / fresh-correct",
  "constraints": {"STRICT": "C-WAI <= FreshCache validation C-WAI (WAI / fresh-correct)", "RELAXED": "C-WAI <= FreshCache validation C-WAI + 5 pp"},
  "selection": "CA-Strict / CA-Relaxed = minimum validation modeled cost (cost-model coefficients, live-median regime primary; local regime reported) among candidates meeting the constraint; ties -> lower C-WAI, then grid order; NONE if no candidate meets it; no further relaxation",
  "cost_model": "cost-model common model: search, fetch, generation, measured lookup (CategoryAware code path for every CA candidate; FreshCache; both from component_costs.json), embedding c_embed per non-REAL_TIME request",
  "heldout": "frozen held-out replay and frozen audit; FreshCache arms reuse frozen L2-sweep / verifier labels exactly",
  "evidence": "served evidence only from request-time-retained provenance: generated answer -> evidence given to generation; cached-answer hit -> the entry's stored evidence (pages at the entry's creation time, local snapshot store); empty provenance -> EVIDENCE_UNAVAILABLE; SUFFV 2-judge protocol (L2-sweep / recorded audit)",
  "multiple_testing": "one family: FreshCache primary vs each SELECTED CategoryAware configuration (CA-Strict, CA-Relaxed if distinct) on (a) WAI and (b) evidence sufficiency; exact McNemar; Holm adjustment"}
DESIGN["sha256"] = X.sha_obj(DESIGN); GF = X.O / "grid_frozen.json"
if not GF.exists():
    G0 = {"written_utc": now(), "design": DESIGN, "grid": GRID}; G0["sha256"] = X.sha_obj({"design": DESIGN, "grid": GRID})
    json.dump(G0, open(GF, "w"), indent=1); os.chmod(GF, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
G0 = json.load(open(GF)); assert G0["sha256"] == X.sha_obj({"design": G0["design"], "grid": G0["grid"]}) and G0["grid"] == GRID
X.log(f"[{now()}] CategoryAware-frontier stage 1 (validation only); grid + design frozen sha256 {G0['sha256']} ({len(GRID)} candidates)")
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/engine_loader.py", "freshcache/replay_engine/answer_pools.py",
        "scripts/evaluation/generation_judging_stage.py", "outputs/replay/results/replays_reported.json", "outputs/replay/results/quality_selection_keys_reported.json",
        "outputs/replay/results/audit_keys_reported.json", "outputs/replay/verifier/out/test_operational.json", "outputs/replay/verifier/out/test_rows.pkl",
        "outputs/quality_selection/q_sample.jsonl", "outputs/quality_selection/q_verification.csv", "outputs/quality_selection/q_answers.jsonl",
        "outputs/general_audit/sample.json", "outputs/validation_split/split.json", "freshcache/method/tiered_cache_model.py", "data/sim_matrix_bgem3.npy", "data/derived/round_table.jsonl",
        "scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py", "outputs/baseline_selection/frozen_selection.json", "outputs/baseline_selection/baseline_selection_summary.json", "outputs/baseline_selection/baseline_selection_points.csv",
        "outputs/baseline_selection/baseline_selection_request_level.csv", "outputs/baseline_selection/work/stage3.json", "outputs/l2_operating_points/l2_threshold_points.csv", "outputs/l2_operating_points/l2_threshold_request_level.csv",
        "outputs/concurrency_cost/component_costs.json", "outputs/concurrency_cost/baseline_serving_cost.csv", "configs/integrity/FILE_HASHES.csv"]
if not (X.O / "work/read_only_before.json").exists(): json.dump({p: X.fstat(inpath(p)) for p in READ}, open(X.O / "work/read_only_before.json", "w"), indent=1)
E = X.engine(); V = X.C.streams(E)["validation"]; nV = len(V); assert DS.check("validation_requests", nV)
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
EV = sorted(r["query_id"] for r in csv.DictReader(open((OUTPUT_ROOT / "quality_selection/q_verification.csv"))) if r.get("evidence_valid") == "1"); assert REF.check("validation_quality/N", len(EV))
QK = {r["query_id"]: r for r in json.load(open(X.CRR / "results/quality_selection_keys_reported.json"))["records"]}
BASELINE_SUMMARY = json.load(open(X.BASELINE_SEL_DIR / "baseline_selection_summary.json"))["ca_validation_combined_check"]
nonrt = sum(1 for _, r in V if r["freshness_class"] != "REAL_TIME")
G = {}; OPS, KEYS, TASKS = {}, {}, {}
for g in GRID:
    m, rows = X.ca_replay(V, g["theta"], g["kappa"]); REQV = {r["query_id"]: r for _, r in V}
    cross = sum(1 for q, rr in rows.items() if rr["tier"] == "L1" and rr["entry_class"] != REQV[q]["freshness_class"])
    OPS[g["id"]] = {"search": m["search_calls"], "fetch": m["fetches"], "gen": m["generations"], "l1_hits": m["l1_hits"], "cross_category_hits": cross, "search_saved_pct": m["search_saved_pct"],
                    "fetch_per_1k": round(1000 * m["fetches"] / nV, 1), "gen_per_1k": round(1000 * m["generations"] / nV, 1), **X.cost(m["search_calls"], m["fetches"], m["generations"], nV, nonrt, X.LOOKUP["CategoryAware"])}
    KEYS[g["id"]] = {}
    for q in EV:
        k, ctx, gq = X.key_for(rows[q], QS[q]["query"]); KEYS[g["id"]][q] = {"key": k, "tier": rows[q]["tier"]}; TASKS.setdefault(tuple(k), {"key": k, "question": gq, "context": ctx})
    if g["dtheta"] == 0.0 and g["m_ttl"] == 1.0:
        G["candidate (0, 1.0) = original CategoryAware reproduces baseline-selection combined validation (savings, L1 hits, cross-category hits)"] = (m["search_saved_pct"], m["l1_hits"], cross) == (BASELINE_SUMMARY["validation_savings_pct"], BASELINE_SUMMARY["validation_l1_hits"], BASELINE_SUMMARY["cross_category_hits"])
fm, _, ftr = X.C.metrics(E, V, "FreshCache")
fsearch = sum(1 for q in ftr if ftr[q]["tier"] in ("miss", "REAL_TIME_BYPASS"))
OPS["FreshCache_primary"] = {"search": fsearch, "fetch": fm["fetches"], "gen": fm["generations"], "l1_hits": fm["l1_hits"], "search_saved_pct": fm["search_saved_pct"], "fetch_per_1k": fm["fetch_per_1k"], "gen_per_1k": fm["gen_per_1k"],
                             **X.cost(fsearch, fm["fetches"], fm["generations"], nV, nonrt, X.LOOKUP["FreshCache"])}
G["FreshCache validation savings equals the recorded reference"] = REF.check("reported_engine/validation/FreshCache/search_saved_pct", fm["search_saved_pct"])
for k_, v in G.items(): X.log("GATE", k_, v)
if not all(G.values()): X.log("STOP: validation gate failed"); sys.exit(1)
lab = X.Val(); todo = [t for t in TASKS.values() if lab.answer(t["key"]) is None]
X.wjl(X.O / "gpu/in/gen_tasks.jsonl", todo)
json.dump({"gates": G, "ops": OPS, "keys": KEYS, "EV": EV, "n_validation": nV, "nonrt": nonrt}, open(X.O / "work/stage1.json", "w"))
X.log(f"validation: {len(GRID)} candidates replayed; distinct answer keys {len(TASKS)}; new answers needed {len(todo)}")
X.log(json.dumps({k: {x: v[x] for x in ("search_saved_pct", "live_p50_ms_per_request", "local_ms_per_request")} for k, v in OPS.items()}))
