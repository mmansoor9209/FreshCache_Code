"""Unified cost/timing table and break-even for composition D vs B (per request, ms).
saved = dS*c_s + dF*c_f + dG*c_g - dO; search break-even c_s* = max(0, (dO - dF*c_f - dG*c_g) / dS).
Operation counts and the decision-path overhead from outputs/composition/outputs/composition_cost.json (frozen trace); c_s, c_f (live p50) and c_g from
outputs/concurrency_cost/component_costs.json (dG = 0 for A-D).
Overhead dO scenarios: (a) measured decision path with a precomputed similarity matrix (D-B, conservative, `dec` below; below the 0.05 ms noise
floor); (b) exact / HNSW FAISS lookup-and-insert timing (vector_index_timing.json; lookup timing only: top-16 candidate retrieval was not checked for
parity with the frozen admission decisions); this is ADDED to (a) because the index replaces only the matrix-row read, so (a)+(b) slightly
double-counts that read (at most the `dec` of (a)); (c) hypothetical 1 ms and 10 ms index overheads. Writes outputs/break_even_analysis.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "changed_population")); RC = ((OUTPUT_ROOT / "composition/outputs")); CHECKS_OUT = ((OUTPUT_ROOT / "composition_checks/outputs"))
T = json.load(open(RC / "composition_cost.json"))["policies"]; ANN = json.load(open(CHECKS_OUT / "vector_index_timing.json")); LAT = json.load(open(((OUTPUT_ROOT / "composition_summary/outputs/live_latency_summary.json"))))
CO = json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]; cf, cs_live, cg = CO["c_fetch_ms"]["live_p50"], CO["c_search_ms"]["live_p50"], CO["c_gen_ms"]["value"]
dS = T["B"]["n_s"] - T["D"]["n_s"]; dF = T["B"]["n_f"] - T["D"]["n_f"]; dG = T["B"]["n_g"] - T["D"]["n_g"]
dec = json.load(open(RC / "composition_cost.json"))["B_vs_D"]["d_overhead_ms_conservative"]   # measured D-B decision path (composition_cost_model.py)
SC = {"measured decision path (matrix)": dec, "+ exact index (measured)": dec + ANN["flat"]["D_minus_B_ms_per_request"],
      "+ HNSW index (measured)": dec + ANN["hnsw"]["D_minus_B_ms_per_request"], "hypothetical 1 ms index": 1.0, "hypothetical 10 ms index": 10.0}
rows = {}
for k, dO in SC.items():
    rows[k] = {"dO_ms": round(dO, 3), "break_even_search_ms_with_fetch_benefit": round(max(0.0, (dO - dF * cf - dG * cg) / dS), 2),
               "break_even_search_ms_search_only": round(dO / dS, 2), "saving_ms_at_live_costs": round(dS * cs_live + dF * cf - dO, 1)}
R = {"dS_per_request": round(dS, 5), "dF_per_request": round(dF, 5), "dG_per_request": round(dG, 5), "fetch_benefit_ms_at_live_cf": round(dF * cf, 2),
     "modeled_cost_s": {p: round(T[p]["cost_ms_live"] / 1000, 3) for p in "ABCD"}, "scenarios": rows,
     "index_timing_ms_per_request": {"exact": {"B": ANN["flat"]["B_answer_index_only"]["mean_ms_per_request"], "D": ANN["flat"]["D_answer_and_url_list_index"]["mean_ms_per_request"]},
                                      "hnsw": {"B": ANN["hnsw"]["B_answer_index_only"]["mean_ms_per_request"], "D": ANN["hnsw"]["D_answer_and_url_list_index"]["mean_ms_per_request"]}},
     "index_timing_status": "top-16 parity checked (outputs/vector_index_parity.json): exact and HNSW top-16 differ from the frozen trace on a few requests (counts in that file)",
     "live_latency": LAT}
json.dump(R, open(O / "outputs/break_even_analysis.json", "w"), indent=1)
print("dS", R["dS_per_request"], "dF", R["dF_per_request"], "fetch benefit ms", R["fetch_benefit_ms_at_live_cf"])
for k, v in rows.items(): print(f"{k:34s}", v)
