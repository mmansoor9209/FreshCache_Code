"""CategoryAware-style comparator: our reimplementation of the per-category similarity threshold and TTL policy of
category-aware semantic caching (Wang et al., 2025), not the complete published system; it omits quotas, the storage tier,
indexing and load adaptation.

Composition of a CategoryAware-style answer cache (frozen original frontier configuration) with FreshCache L2/L3. Modes: val | val_items | select | heldout | items | score.
Primary FreshCache (theta_L2 0.75) is unchanged; variants H80 / H55 use the existing FreshCache sensitivities (theta_L2 0.80; verifier gamma 0.55)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, hashlib, json, os, pickle, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); RV = _ilu.module_from_spec(_rs); _rs.loader.exec_module(RV)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import categoryaware_composition_helpers as X, categoryaware_freshcache_engine as HE
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); MODE = sys.argv[1]
VARS = {"H75": {"floor": 0.75, "gamma": None, "desc": "CategoryAware L1 + FreshCache L2 theta 0.75 + L3"}, "H80": {"floor": 0.80, "gamma": None, "desc": "CategoryAware L1 + FreshCache L2 theta 0.80 + L3"},
        "H55": {"floor": 0.75, "gamma": 0.55, "desc": "CategoryAware L1 + FreshCache L2 theta 0.75 + verifier gamma 0.55 + L3"}}
DESIGN = {"status": "new composition experiment motivated by CategoryAware-frontier; not the original primary FreshCache; primary FreshCache theta_L2 = 0.75 unchanged; H80/H55 use existing FreshCache sensitivities",
  "system": "L1 = frozen original CategoryAware-style answer cache (CategoryAware-frontier ca_d+0.000_m1.00: per-class theta/kappa, threshold by request class, TTL by entry class, write-time age); on L1 hit return its answer; on L1 miss run FreshCache L2 (+ verifier for H55) and L3 under their normal rules, generate from the served evidence, store the answer in the CategoryAware cache (entry keeps the evidence pages used) and update L2/L3 normally; REAL_TIME full bypass; FreshCache's own L1 not run; empty caches per replay",
  "engine": "categoryaware_freshcache_engine.py (transcription of reported verifier verifier_replay.replay), gated in work/engine_gates.json to reproduce frozen FreshCache, verifier gamma 0.55, theta 0.80 and baseline-selection ca_replay exactly",
  "variants": VARS, "validation": "validation quality set, two-judge rule; served-evidence sufficiency (SUFFV) where computable",
  "selection": {"Hybrid-Strict": "min validation modeled cost (cost-model live-median regime) with validation C-WAI <= FreshCache primary validation C-WAI", "Hybrid-CA-Matched": "min validation modeled cost with validation C-WAI <= original CategoryAware validation C-WAI", "ties": "lower C-WAI, then H75 < H80 < H55", "none": "NONE; no relaxation"},
  "cost": "cost-model coefficients; lookup ms/request: hybrids measured (fresh-process cold pass; validation timing for selection, held-out timing for held-out), CategoryAware and FreshCache from component_costs.json (cost-model); embedding c_embed per non-REAL_TIME request; H55 excludes verifier page-embedding cost (unmeasured)",
  "family": "composition family: each selected hybrid vs original CategoryAware on WAI and on evidence sufficiency (common evaluable cases), exact McNemar, Holm"}
DESIGN["sha256"] = X.sha_obj(DESIGN); MAN = X.O / "ca_composition_selection_manifest.json"
if not (X.O / "design_frozen.json").exists(): json.dump({**DESIGN, "written_utc": now()}, open(X.O / "design_frozen.json", "w"), indent=1)
assert json.load(open(X.O / "design_frozen.json"))["sha256"] == DESIGN["sha256"]
READ = ["freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/verifier_replay.py", "outputs/replay/verifier/out/test_rows.pkl",
        "outputs/replay/verifier/out/test_operational.json", "outputs/replay/results/replays_reported.json", "outputs/replay/results/quality_selection_keys_reported.json",
        "outputs/l2_verifier/out/emb_queries.npz", "outputs/l2_verifier/out/emb_pages.npz", "outputs/quality_selection/q_sample.jsonl", "outputs/quality_selection/q_verification.csv",
        "outputs/general_audit/sample.json", "freshcache/method/tiered_cache_model.py", "data/sim_matrix_bgem3.npy", "scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py", "outputs/baseline_selection/frozen_selection.json",
        "outputs/l2_operating_points/l2_threshold_request_level.csv", "outputs/l2_operating_points/l2_threshold_points.csv", "outputs/concurrency_cost/component_costs.json", "outputs/concurrency_cost/baseline_serving_cost.csv",
        "outputs/categoryaware_frontier/frontier_validation_grid.csv", "outputs/categoryaware_frontier/frontier_heldout_results.csv", "outputs/categoryaware_frontier/frontier_quality_audit.csv",
        "outputs/categoryaware_frontier/frontier_evidence_sufficiency.csv", "scripts/baselines/category_aware_frontier/frontier_helpers.py", "outputs/categoryaware_frontier/run.log", "configs/integrity/FILE_HASHES.csv"]
if not (X.O / "work/read_only_before.json").exists(): json.dump({p: X.fstat(inpath(p)) for p in READ}, open(X.O / "work/read_only_before.json", "w"), indent=1)
jl = X.jl
def suff_pool():
    SL = {}
    def addsuff(tasks, l8, q7):
        for t in tasks:
            a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
            if a is not None and b is not None: SL.setdefault((t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()), (a, b))
    addsuff(jl(X.OV / "audit_suff_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(X.OV / "audit_suff_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(X.OV / "audit_suff_qwen7b.jsonl")})
    RU = json.load(open(X.CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
    for r in jl(X.CRR / "gpu/out/suffv_llama8b.jsonl"): cl8[(r["query_id"], r["arm"])] = r["label"]
    for r in jl(X.CRR / "gpu/out/suffv_qwen7b.jsonl"): cq7[(r["query_id"], r["arm"])] = r["label"]
    addsuff(jl(X.CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7)
    for D in (X.L2_POINTS_DIR, X.CALIBRATION_AUDIT_DIR, X.CA_FRONTIER_DIR, X.O):
        addsuff(jl(D / "gpu/in/suffv_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(D / "gpu/out/suffv_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(D / "gpu/out/suffv_qwen7b.jsonl")})
    return SL
def add_suff_tasks(tasks):
    old = jl(X.O / "gpu/in/suffv_tasks.jsonl"); have = {(t["query_id"], t["arm"]) for t in old}
    X.wjl(X.O / "gpu/in/suffv_tasks.jsonl", old + [t for t in tasks if (t["query_id"], t["arm"]) not in have])
def build(stream, ids, qtext):
    E = X.engine(); ma = sys.modules["page_version_replay"]; SUP = HE.Support(X.OV, ma); TH, KA = X.ca_params(0.0, 1.0); OUT = {}
    for v, c in VARS.items():
        m, rows = HE.replay(E, stream, l1="ca", floor=c["floor"], gamma=c["gamma"], support=SUP if c["gamma"] else None, theta_by_class=TH, kappa_by_class=KA)
        misses = [q for q, r in rows.items() if r["tier"] != "L1" and r["tier"] != "REAL_TIME"]
        mix = {"l1_misses_nonRT": len(misses), "L2_reuse": sum(rows[q]["tier"] == "L2" for q in misses), "fresh_search": sum(rows[q]["tier"] in ("miss", "miss_rejected") for q in misses),
               "verifier_rejected": sum(rows[q]["tier"] == "miss_rejected" for q in misses), "any_L3_reuse": sum(rows[q]["l3_reused_pages"] > 0 for q in misses)}
        aud = {}
        for q in ids:
            p = rows[q]; k, ctx, gq = X.key_for(p, qtext[q]); ev = [tuple(x) for x in (p.get("ev") or [])]; blk = X.evblock(ev) if ev else None
            aud[q] = {"tier": p["tier"], "key": k, "gen_question": gq, "context": ctx, "evidence_block": blk, "evidence_sha": hashlib.sha256(blk.encode()).hexdigest() if blk else None,
                      "evidence_source": ("stored CategoryAware entry evidence (pages used to generate it)" if p["tier"] == "L1" else "evidence given to generation (after L2/L3)") if ev else "EVIDENCE_UNAVAILABLE"}
        OUT[v] = {"metrics": m, "miss_mix": mix, "audit": aud}
    return OUT
def tasks_from(OUT, lab, qtext):
    G, S = {}, {}
    for v, d in OUT.items():
        for q, a in d["audit"].items():
            if lab.answer(a["key"]) is None: G.setdefault(tuple(a["key"]), {"key": a["key"], "question": a["gen_question"], "context": a["context"]})
            if a["evidence_block"]: S.setdefault((q, a["evidence_sha"]), {"query_id": q, "arm": a["evidence_sha"][:16], "query": qtext[q], "evidence": a["evidence_block"]})
    SL = suff_pool(); return list(G.values()), [t for k, t in S.items() if k not in SL]
def ops_row(m, n, nonrt, lookup):
    return {"search": m["search_calls"], "fetch": m["fetches"], "gen": m["generations"], "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"], "search_saved_pct": round(100 * (1 - m["search_calls"] / n), 4),
            "search_per_1k": round(1000 * m["search_calls"] / n, 1), "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1), "lookup_ms": lookup, **X.cost(m["search_calls"], m["fetches"], m["generations"], n, nonrt, lookup)}
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
EV = sorted(r["query_id"] for r in csv.DictReader(open((OUTPUT_ROOT / "quality_selection/q_verification.csv"))) if r.get("evidence_valid") == "1")
QK = {r["query_id"]: r for r in json.load(open(X.CRR / "results/quality_selection_keys_reported.json"))["records"]}
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
if MODE == "val":
    assert not MAN.exists(), "selection already frozen"
    X.log(f"[{now()}] composition validation stage; design sha256 {DESIGN['sha256']}")
    E = X.engine(); V = X.C.streams(E)["validation"]; OUT = build(V, EV, {q: QS[q]["query"] for q in EV})
    gt, st = tasks_from(OUT, X.Val(), {q: QS[q]["query"] for q in EV}); X.wjl(X.O / "gpu/in/gen_tasks.jsonl", gt); add_suff_tasks(st)
    pickle.dump({v: {"metrics": d["metrics"], "miss_mix": d["miss_mix"], "audit": {q: {k: x for k, x in a.items() if k not in ("context", "evidence_block")} for q, a in d["audit"].items()}} for v, d in OUT.items()}, open(X.O / "work/val.pkl", "wb"))
    X.log("validation replays:", json.dumps({v: {k: d["metrics"][k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "l3_hits", "fetches")} for v, d in OUT.items()}), "| new answers", len(gt), "| new sufficiency tasks", len(st)); sys.exit(0)
if MODE in ("val_items", "select"):
    VAL = pickle.load(open(X.O / "work/val.pkl", "rb")); lab = X.Val()
    ALL = [(q, QK[q]["fresh_key"]) for q in EV] + [(q, a["key"]) for d in VAL.values() for q, a in d["audit"].items()]
    if MODE == "val_items":
        items, miss = X.judge_items(lab, [(QS[q]["query"], QS[q]["gold"], k) for q, k in ALL])
        if miss: X.log(f"STOP: {len(miss)} validation answers missing"); sys.exit(1)
        X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("validation judge2 items:", len(items)); sys.exit(0)
    assert not MAN.exists(), "selection already frozen"
    fresh = {q: lab.labels(QS[q]["query"], QS[q]["gold"], QK[q]["fresh_key"])[0] for q in EV}; FC = sum(v == "CORRECT" for v in fresh.values()); assert RV.check("validation_quality/fresh_correct", FC)
    E = X.engine(); V = X.C.streams(E)["validation"]; nV = len(V); nonrt = sum(1 for _, r in V if r["freshness_class"] != "REAL_TIME"); SL = suff_pool()
    REF = {r["config_id"]: r for r in csv.DictReader(open(X.CA_FRONTIER_DIR / "frontier_validation_grid.csv"))}
    ROWS = []
    for v, d in VAL.items():
        labs = {q: lab.labels(QS[q]["query"], QS[q]["gold"], d["audit"][q]["key"])[0] for q in EV}; assert all(labs.values())
        w = sum(1 for q in EV if fresh[q] == "CORRECT" and labs[q] == "WRONG"); c = collections.Counter(labs.values())
        sf = collections.Counter(("EVIDENCE_UNAVAILABLE" if d["audit"][q]["evidence_sha"] is None else ("UNRESOLVED" if (q, d["audit"][q]["evidence_sha"]) not in SL else (SL[(q, d["audit"][q]["evidence_sha"])][0] if SL[(q, d["audit"][q]["evidence_sha"])][0] == SL[(q, d["audit"][q]["evidence_sha"])][1] else "JUDGE_DISAGREEMENT"))) for q in EV)
        tm = json.load(open(X.O / f"work/time_{v}_validation.json"))["cold_ms_per_request"]
        ROWS.append({"method": v, "desc": VARS[v]["desc"], **ops_row(d["metrics"], nV, nonrt, tm), "correct": c["CORRECT"], "WAI": w, "C_WAI_pct": round(100 * w / FC, 4),
                     "sufficient": sf["SUFFICIENT"], "insufficient": sf["INSUFFICIENT"], "unresolved": sf["JUDGE_DISAGREEMENT"] + sf["UNRESOLVED"], "unavailable": sf["EVIDENCE_UNAVAILABLE"], **{f"miss_{k}": x for k, x in d["miss_mix"].items()}})
    for nm, cid in (("CategoryAware-style (original)", "ca_d+0.000_m1.00"), ("FreshCache primary", "FreshCache_primary")):
        r = REF[cid]; ROWS.append({"method": nm, "desc": "reference (CategoryAware-frontier validation)", "search_saved_pct": float(r["search_saved_pct"]), "fetch_per_1k": float(r["fetch_per_1k"]), "gen_per_1k": float(r["gen_per_1k"]),
                                   "l1_hits": int(r["l1_hits"]), "local_ms_per_request": float(r["local_ms_per_request"]), "live_p50_ms_per_request": float(r["live_p50_ms_per_request"]), "correct": int(r["CORRECT"]), "WAI": int(r["WAI"]), "C_WAI_pct": float(r["C_WAI_pct"])})
    cols = list(dict.fromkeys(k for r in ROWS for k in r))
    with open(X.O / "ca_composition_validation_results.csv", "w", newline="") as f: w_ = csv.DictWriter(f, fieldnames=cols); w_.writeheader(); w_.writerows(ROWS)
    H = [r for r in ROWS if r["method"] in VARS]; order = list(VARS)
    def pick(lim):
        f_ = [r for r in H if r["C_WAI_pct"] <= lim + 1e-9]; return min(f_, key=lambda r: (r["live_p50_ms_per_request"], r["C_WAI_pct"], order.index(r["method"])))["method"] if f_ else "NONE"
    LIM = {"Hybrid-Strict": 100 * int(REF["FreshCache_primary"]["WAI"]) / FC, "Hybrid-CA-Matched": 100 * int(REF["ca_d+0.000_m1.00"]["WAI"]) / FC}; SEL = {k: pick(v) for k, v in LIM.items()}
    M = {"frozen_before_heldout_composition": True, "heldout_outcomes_consulted": False, "design_sha256": DESIGN["sha256"], "limits_pct": {k: round(v, 4) for k, v in LIM.items()},
         "selected": SEL, "validation_rows": ROWS}
    M["sha256"] = X.sha_obj(M); M["frozen_utc"] = now(); json.dump(M, open(MAN, "w"), indent=1); os.chmod(MAN, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    X.log(f"[{M['frozen_utc']}] selection manifest frozen sha256 {M['sha256']} BEFORE held-out: {json.dumps(SEL)}"); sys.exit(0)
assert MAN.exists() and not (os.stat(MAN).st_mode & stat.S_IWUSR); M = json.load(open(MAN)); assert X.sha_obj({k: v for k, v in M.items() if k not in ("sha256", "frozen_utc")}) == M["sha256"]
if MODE == "heldout":
    X.log(f"[{now()}] held-out stage; manifest sha256 {M['sha256']}")
    E = X.engine(); H = X.C.streams(E)["heldout"]; assert DS.check("heldout_requests", len(H)); OUT = build(H, IDS, {q: SQ[q]["query"] for q in IDS})
    gt, st = tasks_from(OUT, X.Held(), {q: SQ[q]["query"] for q in IDS}); X.wjl(X.O / "gpu/in/gen_tasks.jsonl", gt); add_suff_tasks(st)
    pickle.dump({v: {"metrics": d["metrics"], "miss_mix": d["miss_mix"], "audit": {q: {k: x for k, x in a.items() if k not in ("context", "evidence_block")} for q, a in d["audit"].items()}} for v, d in OUT.items()}, open(X.O / "work/heldout.pkl", "wb"))
    X.log("held-out replays:", json.dumps({v: {k: d["metrics"][k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "l3_hits", "fetches")} for v, d in OUT.items()}), "| new answers", len(gt), "| new sufficiency tasks", len(st)); sys.exit(0)
HO = pickle.load(open(X.O / "work/heldout.pkl", "rb")); hl = X.Held()
if MODE == "items":
    items, miss = X.judge_items(hl, [(SQ[q]["query"], SQ[q]["gold"], d["audit"][q]["key"]) for d in HO.values() for q in IDS])
    if miss: X.log(f"STOP: {len(miss)} held-out answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("held-out judge2 items:", len(items)); sys.exit(0)
# ================================ score
L2_REQUESTS = collections.defaultdict(dict)
for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")): L2_REQUESTS[r["point"]][r["request_id"]] = r
FRESH = {q: L2_REQUESTS["theta_0.75"][q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values())
QA = {r["request_id"]: r for r in csv.DictReader(open(X.CA_FRONTIER_DIR / "frontier_quality_audit.csv", encoding="utf-8"))}; QE = {r["request_id"]: r for r in csv.DictReader(open(X.CA_FRONTIER_DIR / "frontier_evidence_sufficiency.csv", encoding="utf-8"))}
CAO = "CategoryAware-style (original)"; FCP = "FreshCache primary (theta_L2=0.75)"
LAB = {FCP: {q: {"answer": L2_REQUESTS["theta_0.75"][q]["answer_label"], "suff": L2_REQUESTS["theta_0.75"][q]["sufficiency"], "tier": L2_REQUESTS["theta_0.75"][q]["tier"]} for q in IDS},
       CAO: {q: {"answer": QA[q][f"{CAO} | answer"], "suff": QE[q][f"{CAO} | sufficiency"], "tier": QA[q][f"{CAO} | tier"]} for q in IDS}}
SL = suff_pool()
for v, d in HO.items():
    LAB[v] = {}
    for q in IDS:
        a = d["audit"][q]; l, _, _ = hl.labels(SQ[q]["query"], SQ[q]["gold"], a["key"])
        s = "EVIDENCE_UNAVAILABLE" if a["evidence_sha"] is None else ("UNRESOLVED" if (q, a["evidence_sha"]) not in SL else (SL[(q, a["evidence_sha"])][0] if SL[(q, a["evidence_sha"])][0] == SL[(q, a["evidence_sha"])][1] else "JUDGE_DISAGREEMENT"))
        LAB[v][q] = {"answer": l or "UNRESOLVED", "suff": s, "tier": a["tier"]}
G = {"fresh-correct (run-time reference)": RV.check("heldout_audit/fresh_correct", FCN), "no unresolved labels": all(x["answer"] != "UNRESOLVED" and x["suff"] != "UNRESOLVED" for L_ in LAB.values() for x in L_.values()), **json.load(open(X.O / "work/engine_gates.json"))}
E = X.engine(); Hs = X.C.streams(E)["heldout"]; n = len(Hs); nonrt = sum(1 for _, r in Hs if r["freshness_class"] != "REAL_TIME")
COST_TABLE = {r["method"]: r for r in csv.DictReader(open(X.COST_DIR / "baseline_serving_cost.csv"))}; FRONTIER_HELDOUT = {r["method"]: r for r in csv.DictReader(open(X.CA_FRONTIER_DIR / "frontier_heldout_results.csv"))}
FR = json.load(open(X.CRR / "results/replays_reported.json"))["heldout/FreshCache"]; L2_POINTS = {r["point"]: r for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_points.csv"))}
OPS = {"NoCache": {"search_saved_pct": 0.0, "search_per_1k": 1000.0, "fetch_per_1k": float(COST_TABLE["NoCache"]["fetch_per_1k"]), "gen_per_1k": 1000.0, "l1_hits": 0, "l2_hits": 0, "l3_hits": 0,
                   "local_ms_per_request": float(COST_TABLE["NoCache"]["local_ms_per_request"]), "live_p50_ms_per_request": float(COST_TABLE["NoCache"]["live_p50_ms_per_request"])},
       FCP: {"search_saved_pct": FR["search_saved_pct"], "search_per_1k": round(1000 * int(L2_POINTS["theta_0.75"]["fresh_searches"]) / n, 1), "fetch_per_1k": FR["fetch_per_1k"], "gen_per_1k": FR["gen_per_1k"], "l1_hits": FR["l1_hits"], "l2_hits": FR["l2_hits"], "l3_hits": FR["l3_hits"],
             "local_ms_per_request": float(FRONTIER_HELDOUT[FCP]["local_ms_per_request"]), "live_p50_ms_per_request": float(FRONTIER_HELDOUT[FCP]["live_p50_ms_per_request"])},
       CAO: {"search_saved_pct": float(FRONTIER_HELDOUT[CAO]["search_saved_pct"]), "search_per_1k": float(FRONTIER_HELDOUT[CAO]["search_per_1k"]), "fetch_per_1k": float(FRONTIER_HELDOUT[CAO]["fetch_per_1k"]), "gen_per_1k": float(FRONTIER_HELDOUT[CAO]["gen_per_1k"]),
             "l1_hits": None, "l2_hits": 0, "l3_hits": 0, "local_ms_per_request": float(FRONTIER_HELDOUT[CAO]["local_ms_per_request"]), "live_p50_ms_per_request": float(FRONTIER_HELDOUT[CAO]["live_p50_ms_per_request"])}}
for v, d in HO.items(): OPS[v] = ops_row(d["metrics"], n, nonrt, json.load(open(X.O / f"work/time_{v}_heldout.json"))["cold_ms_per_request"]); OPS[v]["miss_mix"] = d["miss_mix"]
wa = lambda L_, q: FRESH[q] == "CORRECT" and L_[q]["answer"] == "WRONG"
def qual(L_):
    c = collections.Counter(x["answer"] for x in L_.values()); s = collections.Counter(x["suff"] for x in L_.values()); w = sum(1 for q in IDS if wa(L_, q))
    return {"correct": c["CORRECT"], "WAI": w, "WAI_pct": round(100 * w / len(IDS), 2), "C_WAI_pct": round(100 * w / FCN, 3), "sufficient": s["SUFFICIENT"], "insufficient": s["INSUFFICIENT"], "unresolved": s["JUDGE_DISAGREEMENT"] + s["UNRESOLVED"], "unavailable": s["EVIDENCE_UNAVAILABLE"]}
POL = ["NoCache", FCP, CAO] + list(VARS); Q = {p: qual(LAB[p]) for p in POL if p in LAB}
def paired(a, b):
    A, B = LAB[a], LAB[b]; w1 = sum(1 for q in IDS if wa(B, q) and not wa(A, q)); w0 = sum(1 for q in IDS if wa(A, q) and not wa(B, q))
    com = [q for q in IDS if A[q]["suff"] in ("SUFFICIENT", "INSUFFICIENT") and B[q]["suff"] in ("SUFFICIENT", "INSUFFICIENT")]
    s1 = sum(1 for q in com if B[q]["suff"] == "SUFFICIENT" and A[q]["suff"] != "SUFFICIENT"); s0 = sum(1 for q in com if A[q]["suff"] == "SUFFICIENT" and B[q]["suff"] != "SUFFICIENT")
    return {"WAI_hybrid_only": w1, "WAI_CA_only": w0, "WAI_p": X.mcnemar(w1, w0), "suff_common_N": len(com), "suff_hybrid": sum(B[q]["suff"] == "SUFFICIENT" for q in com), "suff_CA": sum(A[q]["suff"] == "SUFFICIENT" for q in com),
            "suff_hybrid_only": s1, "suff_CA_only": s0, "suff_p": X.mcnemar(s1, s0)}
SELV = sorted({v for v in M["selected"].values() if v != "NONE"}, key=list(VARS).index)
PAIR = {v: paired(CAO, v) for v in VARS}; FAM = []
for v in SELV:
    FAM.append({"test": f"{v} vs CategoryAware: WAI", "p_raw": PAIR[v]["WAI_p"], "hybrid_better": PAIR[v]["WAI_hybrid_only"] < PAIR[v]["WAI_CA_only"]})
    FAM.append({"test": f"{v} vs CategoryAware: evidence sufficiency", "p_raw": PAIR[v]["suff_p"], "hybrid_better": PAIR[v]["suff_hybrid_only"] > PAIR[v]["suff_CA_only"]})
for f_, a_ in zip(FAM, X.holm([f["p_raw"] for f in FAM])): f_["p_holm"] = a_
INC = []
for v in VARS:
    o, c = OPS[v], OPS[CAO]; mm = o["miss_mix"]
    INC.append({"from": CAO, "to": v, "selected": v in SELV, "additional_searches_avoided_per_1k": round(c["search_per_1k"] - o["search_per_1k"], 1), "additional_fetches_avoided_per_1k": round(c["fetch_per_1k"] - o["fetch_per_1k"], 1),
                "generation_change_per_1k": round(o["gen_per_1k"] - c["gen_per_1k"], 1), "search_saved_change_pp": round(o["search_saved_pct"] - c["search_saved_pct"], 4),
                "cost_change_local_ms_per_request": round(o["local_ms_per_request"] - c["local_ms_per_request"], 2), "cost_change_live_ms_per_request": round(o["live_p50_ms_per_request"] - c["live_p50_ms_per_request"], 2),
                "WAI_change": Q[v]["WAI"] - Q[CAO]["WAI"], "sufficient_change": Q[v]["sufficient"] - Q[CAO]["sufficient"],
                "L1_misses_nonRT": mm["l1_misses_nonRT"], "frac_misses_L2_reuse": round(mm["L2_reuse"] / mm["l1_misses_nonRT"], 4), "frac_misses_any_L3_reuse": round(mm["any_L3_reuse"] / mm["l1_misses_nonRT"], 4),
                "frac_misses_fresh_search": round(mm["fresh_search"] / mm["l1_misses_nonRT"], 4), "verifier_rejections": mm["verifier_rejected"]})
with open(X.O / "ca_composition_incremental_value.csv", "w", newline="") as f: w_ = csv.DictWriter(f, fieldnames=list(INC[0])); w_.writeheader(); w_.writerows(INC)
with open(X.O / "ca_composition_heldout_results.csv", "w", newline="") as f:
    cols = ["method", "search_saved_pct", "search_per_1k", "fetch_per_1k", "gen_per_1k", "l1_hits", "l2_hits", "l3_hits", "local_ms_per_request", "live_p50_ms_per_request"]
    w_ = csv.DictWriter(f, fieldnames=cols); w_.writeheader()
    for p in POL: w_.writerow({"method": p, **{k: OPS[p].get(k) for k in cols[1:]}})
with open(X.O / "ca_composition_quality_audit.csv", "w", newline="") as f:
    w_ = csv.writer(f); keys = [p for p in POL if p in LAB]; w_.writerow(["request_id", "fresh_label"] + [x for p in keys for x in (f"{p} | tier", f"{p} | answer", f"{p} | sufficiency")])
    for q in IDS: w_.writerow([q, FRESH[q]] + [x for p in keys for x in (LAB[p][q]["tier"], LAB[p][q]["answer"], LAB[p][q]["suff"])])
BEF = json.load(open(X.O / "work/read_only_before.json")); UNCH = {p: X.fstat(inpath(p)) == v for p, v in BEF.items()}
best = SELV[0] if SELV else None
SUM = {"generated_utc": now(), "design_sha256": DESIGN["sha256"], "selection_manifest_sha256": M["sha256"], "selected": M["selected"], "validation": M["validation_rows"], "heldout_ops": OPS, "quality": Q,
       "paired_vs_CategoryAware": PAIR, "family": FAM, "incremental": INC, "gates": G, "read_only_files": {p: {**BEF[p], "unchanged": UNCH[p]} for p in BEF}, "all_read_only_unchanged": all(UNCH.values())}
json.dump(SUM, open(X.O / "ca_composition_summary.json", "w"), indent=1, default=str)
X.log("held-out", json.dumps({p: {k: OPS[p].get(k) for k in ("search_saved_pct", "fetch_per_1k", "gen_per_1k", "live_p50_ms_per_request")} for p in POL})); X.log("quality", json.dumps(Q)); X.log("family", json.dumps(FAM)); X.log("unchanged", SUM["all_read_only_unchanged"])
