"""Operating-point comparison, stage 1 (validation only): design freeze and replay of every candidate on the validation stream.
Also builds answer keys for the frozen validation quality set and lists missing answers for generation.
Held-out is not touched. Candidates: SemanticTTL (theta x kappa grid incl. stricter thetas), CategoryAware-style (the 22 validation-only
configurations of the CategoryAware frontier, reused), FreshCache primary, FreshCache theta_L2 = 0.80, NoCache. No FreshCache parameter is tuned."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, copy, csv, datetime, hashlib, json, os, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/operating_points"))
import operating_point_helpers as X
import numpy as np
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
TG_S = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]; KG = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]; CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST"]
COSTJ = json.load(open(X.COST_DIR / "component_costs.json"))["coefficients"]
LOOKUP = {"FreshCache": COSTJ["C_local_lookup"]["per_method_ms_per_request_cold"]["FreshCache"], "CategoryAware": COSTJ["C_local_lookup"]["per_method_ms_per_request_cold"]["CategoryAware-style"],
          "SemanticTTL": round(float(np.mean([v for k, v in COSTJ["C_local_lookup"]["per_method_ms_per_request_cold"].items() if k.startswith("SemanticTTL")])), 4), "NoCache": 0.0}
DESIGN = {"written_utc": now(), "purpose": "quality-matched and cost-matched comparison of FreshCache against strong answer caches; FreshCache is NOT retuned; held-out scored once after this freeze",
  "split": "outputs/validation_split/split.json (validation and held-out clusters, cluster-disjoint)", "seed": 42,
  "validation_quality_set": "outputs/quality_selection: screened requests with evidence_valid == 1; 2-judge rule (Llama-3.1-8B + Qwen2.5-7B both CORRECT); abstain patterns PRIM; UNRESOLVED excluded from C-WAI numerator",
  "candidates": {"SemanticTTL": {"theta": TG_S, "kappa": KG, "cells": len(TG_S) * len(KG), "semantics": "answer-only, l1 sim_ttl, FIXED_TTL[class] x kappa, REAL_TIME bypass (baseline_selection_helpers.ca_replay uniform == engine SemanticTTL, gated)"},
                 "CategoryAware-style": {"source": "outputs/categoryaware_frontier/frontier_validation_grid.csv (22 validation-only configurations around the baseline-selection per-class selection; stricter-or-equal thetas)", "reused": True},
                 "FreshCache": ["primary theta_L2 0.75 (frozen)", "theta_L2 0.80 (sensitivity; L2-sweep frozen point)"], "NoCache": "search + fetch all + generate every request"},
  "answer_quality_constraint": f"validation C-WAI (fresh-correct -> WRONG over {REF.value('validation_quality/fresh_correct')}) <= FreshCache primary validation C-WAI (WAI / fresh-correct); relaxed tier +5 pp reported separately",
  "total_cost_definition": "cost-model component model, live p50 regime: searches x c_s + fetches x c_f + generations x c_g + embeddings x c_embed (one per non-REAL_TIME request for similarity methods; coefficients of component_costs.json) per request + measured cold lookup ms (component_costs.json; SemanticTTL mean of cost-model values; NoCache 0); local regime reported alongside",
  "selection_rules": {"quality_matched": "among a family's validation cells meeting the constraint: maximise validation search savings; ties -> lower live cost, then lower WAI",
                      "cost_matched": "among a family's validation cells with live cost within +-5% of FreshCache primary validation live cost: minimise C-WAI; ties -> lower cost; if no cell within band: nearest cost (reported as such)"},
  "exclusions": "REAL_TIME requests bypass every cache (all methods); validation requests outside the screened set carry no quality label; UNRESOLVED labels counted neither CORRECT nor WRONG",
  "heldout_protocol": "after freezing selected points: held-out replay for operations + cost; frozen held-out audit for accuracy / WAI / C-WAI / abstention; evidence sufficiency on served evidence with the frozen SUFFV two-judge protocol; exact McNemar vs FreshCache primary", "lookup_ms": LOOKUP}
DESIGN["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in DESIGN.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (X.O / "design_frozen.json").exists(): json.dump(DESIGN, open(X.O / "design_frozen.json", "w"), indent=1)
DESIGN = json.load(open(X.O / "design_frozen.json")); X.log(f"[{now()}] T1 stage 1; design sha256 {DESIGN['sha256']}")
# ---- baseline_selection_helpers helpers, extracted read-only (importing baseline_selection_helpers would append to its run log)
NS = {"np": np, "copy": copy, "engine": lambda: X.engine(), "importlib": __import__("importlib"), "ROOT": X.ROOT}
for node in ast.parse(open((REPO_ROOT / "scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py")).read()).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("ca_replay", "drift_from_rows", "patched"): exec(compile(ast.Module([node], []), "baseline_selection_helpers.py", "exec"), NS)
E = X.engine(); exp, p2 = E["exp"], E["p2"]; h, ctx_for = p2.h, p2.ctx_for
V = X.C.streams(E)["validation"]; assert DS.check("validation_requests", len(V)); S = E["support_of"](V, E["rounds"]); n = len(V); nonrt = sum(1 for _, r in V if r["freshness_class"] != "REAL_TIME")
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
EV = sorted(r["query_id"] for r in csv.DictReader(open((OUTPUT_ROOT / "quality_selection/q_verification.csv"))) if r.get("evidence_valid") == "1"); assert REF.check("validation_quality/N", len(EV))
G = {}; CELLS, KEYS, TASKS = {}, {}, {}
def key_for(p, query):
    if p["tier"] == "L1" and "stored_answer_ev" in p:   # audit_replay FreshCache rows: stored answer served
        ev = [tuple(x) for x in p["stored_answer_ev"]]; sctx = ctx_for(tuple(ev)); return ["stored", h(sctx), h(p["stored_answer_query"])], sctx, p["stored_answer_query"]
    ctx = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", query); return ["gen", h(ctx), h(gq)], ctx, gq
def cost(search, fetch, gen, emb, lk):
    out = {}
    for sc in ("local", "live_p50"):
        tot = (search * COSTJ["c_search_ms"][sc] + fetch * COSTJ["c_fetch_ms"][sc] + gen * COSTJ["c_gen_ms"]["value"] + emb * COSTJ["c_embed_ms"]["value"]) / n + lk
        out[f"{sc}_ms_per_request"] = round(tot, 3)
    return out
def add(cid, fam, m, rows, extra):
    if fam == "SemanticTTL": per, _, dr = NS["drift_from_rows"](E, V, rows, S)   # answer-only rows (ca_replay format)
    else: dr = {"drift_pct": m.get("drift_pct"), "coverage_pct": m.get("coverage_pct")}   # FreshCache: frozen engine metrics (Table-2 definition); NoCache: none
    CELLS[cid] = {"family": fam, **extra, "search_saved_pct": round(m["search_saved_pct"], 4), "search_calls": m["search_calls"], "l1_hits": m["l1_hits"], "fetches": m["fetches"], "generations": m["generations"],
                  "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1), **dr, **cost(m["search_calls"], m["fetches"], m["generations"], nonrt if fam != "NoCache" else 0, LOOKUP[fam])}
    if rows is not None:
        KEYS[cid] = {}
        for q in EV:
            kk, ctx, gq = key_for(rows[q], QS[q]["query"]); KEYS[cid][q] = {"key": kk, "tier": rows[q]["tier"]}; TASKS.setdefault(tuple(kk), {"key": kk, "question": gq, "context": ctx})
# ---- FreshCache primary and theta_L2 = 0.80 (metrics from the frozen engine; rows from audit_replay for keys)
fcm, _, fctr = X.C.metrics(E, V, "FreshCache", S); G["FreshCache validation savings equals the recorded reference"] = REF.check("reported_engine/validation/FreshCache/search_saved_pct", fcm["search_saved_pct"])
_, fcrows = p2.replay(V, {}, "FreshCache"); G["audit_replay FreshCache tiers == engine tiers (validation)"] = all({"L1": "L1", "L2": "L2", "miss": "miss", "REAL_TIME": "REAL_TIME_BYPASS"}[fcrows[q]["tier"]] == fctr[q]["tier"] for q in fcrows)
fcm["search_calls"] = sum(1 for q in fcrows if fcrows[q]["tier"] in ("miss", "REAL_TIME")); add("FreshCache_primary", "FreshCache", fcm, fcrows, {"theta_L2": 0.75})
o = exp._L2_EQ_SIM_FLOOR; exp._L2_EQ_SIM_FLOOR = 0.80
try:
    m80, _, _ = X.C.metrics(E, V, "FreshCache", S); _, rows80 = p2.replay(V, {}, "FreshCache"); m80["search_calls"] = sum(1 for q in rows80 if rows80[q]["tier"] in ("miss", "REAL_TIME"))
finally: exp._L2_EQ_SIM_FLOOR = o; assert exp._L2_EQ_SIM_FLOOR == 0.75
add("FreshCache_thetaL2_0.80", "FreshCache", m80, rows80, {"theta_L2": 0.80}); X.log(f"[{now()}] FreshCache primary {fcm['search_saved_pct']} / thetaL2 0.80 {m80['search_saved_pct']}")
# ---- NoCache
nc = {"search_saved_pct": 0.0, "search_calls": n, "l1_hits": 0, "fetches": sum(len(r["urls"]) for _, r in V), "generations": n}
add("NoCache", "NoCache", nc, None, {}); KEYS["NoCache"] = {q: {"key": ["fresh", "", ""], "tier": "fresh"} for q in EV}   # quality = fresh labels (WAI = 0 by definition)
# ---- SemanticTTL grid (ca_replay with uniform parameters == engine SemanticTTL; gated at theta 0.80)
for th in TG_S:
    for k in KG:
        m, rows = NS["ca_replay"](V, {c: th for c in CLASSES}, {c: k for c in CLASSES}); cid = f"sttl_t{th:.2f}_k{k:g}"
        if th == 0.80:
            with NS["patched"](0.80, k): me_, _, _ = X.C.metrics(E, V, "SemanticTTL", S)
            G[f"{cid}: ca_replay uniform == engine SemanticTTL (savings, fetches, generations)"] = (m["search_saved_pct"], m["fetches"], m["generations"]) == (me_["search_saved_pct"], me_["fetches"], me_["generations"])
        add(cid, "SemanticTTL", m, rows, {"theta": th, "kappa": k}); X.log(f"[{now()}] cell {cid} saved {m['search_saved_pct']}")
# ---- CategoryAware-style: the validation-only frontier grid, reused as-is (operations, cost and screened-set quality already computed on validation only)
FRONTIER_GRID = list(csv.DictReader(open(X.CA_FRONTIER_DIR / "frontier_validation_grid.csv")))
fnum = lambda v, t=float: t(v) if v not in ("", None) else None
REQ_CA = ("search_saved_pct", "fetch_per_1k", "gen_per_1k", "live_p50_ms_per_request", "local_ms_per_request", "N", "fresh_correct", "CORRECT", "WAI", "C_WAI_pct")
CA_REASON = {}
for r in FRONTIER_GRID:
    if any(r.get(k) in ("", None) for k in REQ_CA): CA_REASON[r["config_id"]] = "missing required field(s): " + ",".join(k for k in REQ_CA if r.get(k) in ("", None))
    elif (int(r["N"]), int(r["fresh_correct"])) != (len(EV), REF.value("validation_quality/fresh_correct")): CA_REASON[r["config_id"]] = f"different quality set (N={r['N']}, fresh_correct={r['fresh_correct']})"
CA_EXCLUDED = sorted(CA_REASON); X.log("CategoryAware-frontier rows", len(FRONTIER_GRID), "CA cells excluded:", len(CA_EXCLUDED), json.dumps(CA_REASON))
for r in [r for r in FRONTIER_GRID if r["config_id"] not in CA_EXCLUDED]:
    cid = f"ca_{r['config_id']}"; CELLS[cid] = {"family": "CategoryAware", "config_id": r["config_id"], "theta_by_class": {c: fnum(r[f"theta_{c}"]) for c in CLASSES}, "kappa_by_class": {c: fnum(r[f"kappa_{c}"]) for c in CLASSES}, "dtheta": fnum(r.get("dtheta")), "m_ttl": fnum(r.get("m_ttl")),
        "search_saved_pct": float(r["search_saved_pct"]), "l1_hits": int(r["l1_hits"]), "fetch_per_1k": float(r["fetch_per_1k"]), "gen_per_1k": float(r["gen_per_1k"]), "live_p50_ms_per_request": float(r["live_p50_ms_per_request"]), "local_ms_per_request": float(r["local_ms_per_request"]),
        "quality_source": "CategoryAware-frontier validation grid (screened-set, 2-judge)", "N": int(r["N"]), "fresh_correct": int(r["fresh_correct"]), "CORRECT": int(r["CORRECT"]), "WRONG": fnum(r["WRONG"], int), "ABSTAIN": fnum(r["ABSTAIN"], int) or 0, "accuracy_pct": fnum(r["accuracy_pct"]), "WAI": int(r["WAI"]), "C_WAI_pct": float(r["C_WAI_pct"])}
G["CategoryAware-frontier CA grid: original baseline-selection configuration (ca_d+0.000_m1.00) is eligible on the validation quality set"] = "ca_ca_d+0.000_m1.00" in CELLS
for k_, v in G.items(): X.log("GATE", k_, v)
if not all(G.values()): X.log("STOP: gate failed"); sys.exit(1)
lab = X.Val(); todo = [t for k_, t in TASKS.items() if lab.answer(t["key"]) is None]
X.wjl(X.O / "gpu/in/gen_tasks.jsonl", todo)
pickle.dump({"ca_excluded": CA_REASON, "cells": CELLS, "keys": KEYS, "gates": G, "EV": EV, "n": n, "nonrt": nonrt}, open(X.O / "work/stage1.pkl", "wb"))
X.log("cells", len(CELLS), "distinct keys", len(TASKS), "new validation answers needed:", len(todo))
