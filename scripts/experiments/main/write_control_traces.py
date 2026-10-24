"""Held-out traces of B, B-minus (matched mask), B-minus (eligibility rule), D and D' on the held-out stream with the resumable engine.
Gates: B and D signatures equal the frozen composition traces (composition_ablation_replay.pkl); D' equals composition_engine_reuse_write_suppression; B-minus (mask) equals composition_engine_write_mask.
Outputs: work/traces.pkl (rows per policy), outputs/write_control_traces.json (operation counts, cost, mask-matching log)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, pickle, sys, pathlib, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("common_sample"); import composition_study_helpers as X, composition_engine_resumable as RV, composition_engine_reuse_write_suppression as RD, composition_engine_write_mask as RT   # noqa: E402
E = X.engine(); TH, KA = X.ca_params(0.0, 1.0); CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # component_costs.json (serving_cost_summary.py)
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
FZ = pickle.load(open(X.RCD / "work/composition_ablation_replay.pkl", "rb"))["policies"]
fz = lambda P, q: (FZ[P]["sig"][q][0], tuple(tuple(x) for x in FZ[P]["sig"][q][1]), FZ[P]["sig"][q][2])
H = X.C.streams(E)["heldout"]; n = len(H); nonrt = sum(1 for _, r in H if r["freshness_class"] != "REAL_TIME") / n
kw = dict(floor=0.80, theta_by_class=TH, kappa_by_class=KA)
T = {}; M = {}
t0 = time.time(); M["B"], T["B"] = RV.replay(E, H, l2_on=False, **kw); print("B", round(time.time() - t0, 1), "s", flush=True)
M["D"], T["D"] = RV.replay(E, H, **kw)
M["Dp"], T["Dp"] = RV.replay(E, H, write_rule="no_l2", **kw)
mask = frozenset(q for q, r in T["Dp"].items() if r["tier"] == "L2")
M["Bm"], T["Bm"] = RV.replay(E, H, l2_on=False, write_rule="mask", mask=mask, **kw)
M["Be"], T["Be"] = RV.replay(E, H, l2_on=False, write_rule="elig", **kw)
G = {"B_equals_frozen": all(sig(T["B"][q]) == fz("B", q) for q in T["B"]), "D_equals_frozen": all(sig(T["D"][q]) == fz("D", q) for q in T["D"])}
_, rd = RD.replay(E, H, l1="ca", l1_write_from_l2=False, **kw); G["Dp_equals_rc_engine_dprime"] = all(sig(T["Dp"][q]) == sig(rd[q]) for q in rd)
_, rt = RT.replay(E, H, l1="ca", l2_on=False, skip_l1_write=mask, **kw); G["Bm_equals_rc_engine_thin"] = all(sig(T["Bm"][q]) == sig(rt[q]) for q in rt)
assert all(G.values()), G
cost = lambda m: round((m["search_calls"] * CS + m["fetches"] * CF + m["generations"] * CG) / n + nonrt * CE, 1)
R = {"gate": G, "n": n, "mask_size": len(mask), "policies": {}}
for p, m in M.items():
    R["policies"][p] = dict(m, cost_ms=cost(m), search_per_1k=round(1000 * m["search_calls"] / n, 1), fetch_per_1k=round(1000 * m["fetches"] / n, 1), gen_per_1k=round(1000 * m["generations"] / n, 1),
                            writes=sum(r["written"] for r in T[p].values()), write_opps=sum(r["write_opp"] for r in T[p].values()), withheld=sum(r["write_opp"] and not r["written"] for r in T[p].values()))
# mask matching log: does B-minus(mask) have a write opportunity at each masked request? (it has none when it serves the request from its answer cache)
mm = collections.Counter(("opportunity" if T["Bm"][q]["write_opp"] else "no_opportunity:" + T["Bm"][q]["tier"]) for q in mask)
R["mask_matching"] = {"masked_ids": len(mask), "withheld_at_actual_opportunity": mm["opportunity"], "no_opportunity": {k: v for k, v in mm.items() if k != "opportunity"},
                      "Dp_withheld": R["policies"]["Dp"]["withheld"], "Bm_withheld": R["policies"]["Bm"]["withheld"]}
# eligibility rule vs D' withholding: overlap of withheld request ids
wE = {q for q, r in T["Be"].items() if r["write_opp"] and not r["written"]}
R["elig_vs_Dp"] = {"Be_withheld": len(wE), "Dp_withheld": len(mask), "both": len(wE & mask), "Be_only": len(wE - mask), "Dp_only": len(mask - wE)}
pickle.dump(T, open(O / "work/traces.pkl", "wb")); json.dump(R, open(O / "outputs/write_control_traces.json", "w"), indent=1); print(json.dumps(R, indent=1))
