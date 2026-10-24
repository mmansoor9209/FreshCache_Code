"""Cross-workload cost-and-quality summary and loss lineage with exposure, computed read-only from the producing runs.
Producing runs: the common sample (outputs/common_sample/outputs/common_sample_construction.json; costs from outputs/write_control_traces.json), the paraphrase live
stream (outputs/common_sample/outputs/paraphrase_stream_results.json) and the natural QReCC workload (outputs/natural_workload/outputs/qrecc_results.json).
Denominators are explicit: benchmark costs are over the operational held-out requests; benchmark quality is the finite-stream estimate
over the reference-bearing requests (frame U, common sample); the live streams are censuses (all requests reference-bearing).
Per-1,000 rates: benchmark = sample-expanded totals / (N_ref / 1,000); streams = census counts / stream length * 1,000.
No non-inferiority margin. Writes outputs/main_tables.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, pickle, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
O = ((OUTPUT_ROOT / "cross_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); NW_DIR = ((OUTPUT_ROOT / "natural_workload"))
CM = json.load(open(CS_DIR / "outputs/common_sample_construction.json")); C = json.load(open(CS_DIR / "outputs/paraphrase_stream_results.json")); Q = json.load(open(NW_DIR / "outputs/qrecc_results.json"))
TR = json.load(open(CS_DIR / "outputs/write_control_traces.json"))["policies"]; Nref, U, n = CM["design"]["N_ref"], CM["design"]["frame_U"], CM["design"]["n"]
rows = []
def bench(cmp, base, pol, label):
    v = CM["comparisons"][cmp]; st = v["stream_accuracy"]; cb, cp = TR[base]["cost_ms"], TR[pol]["cost_ms"]
    exp_l, exp_g = v["expanded_losses_in_U"], v["expanded_gains_in_U"]
    rows.append({"workload": "Benchmark common sample", "config": "P (3B, 2,800 ch.)", "comparison": label, "design": f"sample {n:,} of frame {U:,}; N_ref {Nref:,}",
                 "cost_base_s": round(cb / 1000, 3), "cost_pol_s": round(cp / 1000, 3), "cost_change_pct": round(100 * (cp / cb - 1), 1), "cost_population": "operational",
                 "losses_raw": v["total_losses"], "losses_wrong": v["newly_wrong"], "losses_abstain": v["abstention_losses"], "gains_raw": v["gains"],
                 "losses_expanded": round(exp_l[0] if isinstance(exp_l, list) else exp_l, 1) if not isinstance(exp_l, dict) else exp_l, "gains_expanded": exp_g,
                 "losses_per_1k": round(1000 * v["total_losses"] * U / n / Nref, 2), "gains_per_1k": round(1000 * v["gains"] * U / n / Nref, 2),
                 "net_pp": st["pp"], "net_ci_pp": st["ci_pp"], "suff_full_pp": v["suff_full_evidence"]["pp"], "suff_ctx_pp": v["suff_generator_context"]["pp"], "producing_run": "common_sample"})
bench("B->D", "B", "D", "D vs. B"); bench("B->Dp", "B", "Dp", "D' vs. B"); bench("Bm->Dp", "Bm", "Dp", "D' vs. B-"); bench("B->Bm", "B", "Bm", "B- vs. B"); bench("D->Dp", "D", "Dp", "D' vs. D")
def stream(src, name, N, run, cfgname):
    for cfg in ("P", "S"):
        c = src["configs"][cfg]; pol = c["policies"]
        for cmp, base, p, label in (("B->D", "B", "D", "D vs. B"), ("Be->Dp", "Be", "Dp", "D' vs. B-"), ("B->Be", "B", "Be", "B- vs. B"), ("D->Dp", "D", "Dp", "D' vs. D")):
            v = c["comparisons"][cmp]
            rows.append({"workload": name, "config": cfgname[cfg], "comparison": label, "design": f"census of {N:,}", "cost_base_s": round(pol[base]["cost_ms"] / 1000, 3),
                         "cost_pol_s": round(pol[p]["cost_ms"] / 1000, 3), "cost_change_pct": round(100 * (pol[p]["cost_ms"] / pol[base]["cost_ms"] - 1), 1), "cost_population": f"{N:,} operational",
                         "losses_raw": v["total_losses"], "losses_wrong": v["newly_wrong"], "losses_abstain": v["abstention_losses"], "gains_raw": v["gains"],
                         "losses_per_1k": round(1000 * v["total_losses"] / N, 2), "gains_per_1k": round(1000 * v["gains"] / N, 2), "net_pp": v["stream_pp"], "net_ci_pp": v["ci_pp_cluster"],
                         "suff_full_pp": v["suff_full_change_pp"], "suff_ctx_pp": v["suff_ctx_change_pp"], "producing_run": run, "changed": v["changed"]})
CFGN = {"P": "P (3B, depth 2, 2,800 ch.)", "S": "S (14B, depth 5, 8,000 tok.)"}
stream(C, "Paraphrase stream", C["n_requests"], "paraphrase_stream", CFGN); stream(Q, "Natural QReCC workload", Q["n_requests"], "qrecc_stream", CFGN)
# ---- lineage with exposure
def cat3(cats):
    im = sum(v for k, v in cats.items() if k.startswith("immediate")); pr = sum(v for k, v in cats.items() if k.startswith("propagation") or k.startswith("served_wrong"))
    return im, pr, sum(cats.values()) - im - pr
lin = []
v = CM["comparisons"]["B->D"]; im, pr, ot = cat3(v["loss_categories"]); ex = CM["exposure"]["D"]
lin.append({"workload": "Benchmark common sample", "config": "P", "immediate": im, "propagated": pr, "other": ot, "total": v["total_losses"],
            "exposure_list_reuses": ex["L2_reuse_events"], "exposure_L2_rooted_answer_hits": ex["answer_hits_from_L2_root"], "exposure_population": "reference-bearing (traces)",
            "note": f"sample counts ({n:,} of U); exposures over the full trace", "producing_run": "common_sample"})
def exposure(name, k):
    sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); import live_stream_library as S
    st = S.Stream(name); V = st.common_sample_rows(); D = {q: V[(k, "D", q)] for q in st.BYQ}
    return sum(r["tier"] == "L2" for r in D.values()), sum(r["tier"] == "L1" and r.get("root_route") == "L2" for r in D.values())
for src, name, key, N, run in ((C, "Paraphrase stream", "para", C["n_requests"], "paraphrase_stream"), (Q, "Natural QReCC workload", "nat", Q["n_requests"], "qrecc_stream")):
    for cfg, k in (("P", 2), ("S", 5)):
        v = src["configs"][cfg]["comparisons"]["B->D"]; im, pr, ot = cat3(v["loss_categories"]); e1, e2 = exposure(key, k)
        lin.append({"workload": name, "config": cfg, "immediate": im, "propagated": pr, "other": ot, "total": v["total_losses"], "exposure_list_reuses": e1,
                    "exposure_L2_rooted_answer_hits": e2, "exposure_population": f"{N:,} census", "categories": v["loss_categories"], "producing_run": run})
for l in lin:
    f = U / n if l["producing_run"] == "common_sample" else 1.0   # benchmark sample counts expanded to the frame before dividing by full-trace exposure
    l["expansion_factor"] = round(f, 4); l["immediate_expanded"] = round(l["immediate"] * f, 1); l["propagated_expanded"] = round(l["propagated"] * f, 1)
    l["immediate_per_100_reuses"] = round(100 * l["immediate_expanded"] / l["exposure_list_reuses"], 2) if l["exposure_list_reuses"] else None
    l["propagated_per_100_L2_rooted_hits"] = round(100 * l["propagated_expanded"] / l["exposure_L2_rooted_answer_hits"], 2) if l["exposure_L2_rooted_answer_hits"] else None
    continue
    l["immediate_per_100_reuses"] = round(100 * l["immediate"] / l["exposure_list_reuses"], 2) if l["exposure_list_reuses"] else None
    l["propagated_per_100_L2_rooted_hits"] = round(100 * l["propagated"] / l["exposure_L2_rooted_answer_hits"], 2) if l["exposure_L2_rooted_answer_hits"] else None
dec = {"quality_pp": CM["decomposition_quality_pp"], "cost_ms": CM["decomposition_cost_ms"], "producing_run": "common_sample"}
json.dump({"cross_workload": rows, "lineage": lin, "decomposition": dec}, open(O / "outputs/main_tables.json", "w"), indent=1)
for r in rows: print(r["workload"][:22], r["config"][:2], r["comparison"], r["cost_change_pct"], r["losses_raw"], f"({r['losses_wrong']}/{r['losses_abstain']})", r["gains_raw"], r["losses_per_1k"], r["gains_per_1k"], r["net_pp"], r["net_ci_pp"], r["suff_ctx_pp"])
for l in lin: print(l["workload"][:22], l["config"], l["immediate"], l["propagated"], l["other"], l["exposure_list_reuses"], l["exposure_L2_rooted_answer_hits"], l["immediate_per_100_reuses"], l["propagated_per_100_L2_rooted_hits"])
