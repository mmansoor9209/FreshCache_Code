"""Primary and held-out replays for every mixed_age_replay variant, on one engine.
Drift = CHANGED / (CHANGED + UNCHANGED) among the policy's own outcomes inside the common support S (support_of);
coverage = determinate / |S|. Saves per-request outcomes and routes for paired statistics."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, pathlib, pickle; import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); ST = C.streams(E)
MODE = sys.argv[2] if len(sys.argv) > 2 else "main"   # "main": Table 2 variants + NoL1; "mlp": learned-risk variant with rich features (mixed_age_trace_construction.py:112)
VARS = ["FreshCache", "FreshCache_AlwaysPassTemporal", "EquivalentTTL", "FreshCache_NoL1", "L2Only", "SCALM", "SemanticTTL", "TieredFixedTTL"] if MODE == "main" else ["FreshCache_MLP"]
RICH = E["ea"]._rich_feats() if MODE == "mlp" else None
SUF = "" if MODE == "main" else "_mlp"
res, keep = {}, {}
for sname in ("primary", "heldout"):
    s = ST[sname]; S = E["support_of"](s, E["rounds"])
    for v in VARS:
        row, per, tr = C.metrics(E, s, v, S, rich=RICH); res[f"{sname}/{v}"] = row
        keep[f"{sname}/{v}"] = {"per": per, "tier": {q: x["tier"] for q, x in tr.items()}, "reused": {q: sorted(u for u, _ in x["reused"]) for q, x in tr.items()}}
        print(sname, v, {k: row[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "fetch_per_1k", "gen_per_1k", "drift_pct", "coverage_pct")}, flush=True)
json.dump(res, open(C.W/"results"/f"replays_{ENG}{SUF}.json", "w"), indent=1)
if ENG == "reported" and MODE == "main":   # run-time references read by later stages (scripts/analysis/reference_values.py)
    for k in [f"{s}/{v}" for s in ("primary", "heldout") for v in ("FreshCache", "FreshCache_AlwaysPassTemporal")]:
        for mk in ("search_saved_pct", "l1_hits", "l2_hits"): REF.record(f"reported_engine/{k}/{mk}", res[k][mk])
pickle.dump(keep, open(C.W/"results"/f"replays_{ENG}{SUF}_per_request.pkl", "wb"))
