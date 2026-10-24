"""Complete cost accounting and break-even conditions. C = n_s c_s + n_f c_f + n_g c_g + c_overhead per request, with
c_overhead = query embedding (c_embed per non-REAL_TIME request, identical for A-D; L2 reuses that embedding; no page embeddings in A-D)
+ measured decision-path cost (fresh-process cold pass; also the larger of the sequential and the concurrent timing run, as a conservative bound) + measured page-store writes.
Break-even: C_X - C_D = (n_s^X-n_s^D) c_s + (n_f^X-n_f^D) c_f + (n_g^X-n_g^D) c_g - (o_D - o_X). Writes outputs/composition_cost.json and figures."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json, pickle, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = ((OUTPUT_ROOT / "composition"))
CO = json.load(open(((OUTPUT_ROOT / "concurrency_cost/component_costs.json"))))["coefficients"]
ST = pickle.load(open(O / "work/composition_ablation_replay.pkl", "rb")); P = ST["policies"]; N = ST["n"]
PAY = json.load(open(O / "outputs/payload_writes.json"))["policies"]
TIM = {p: json.load(open(O / f"work/time_{p}.json")) for p in "ABCD"}
PAR = {p: json.load(open(O / f"work/time_parallel_{p}.json"))["cold_ms_per_request"] for p in "ABCD"}   # concurrent cold timing run (decision_path_timing.py <p> parallel), the conservative alternative
nonrt = sum(1 for r in P["A"]["sig"].values() if r[0] != "REAL_TIME") / N
cs, cf, cg, ce = CO["c_search_ms"]["live_p50"], CO["c_fetch_ms"]["live_p50"], CO["c_gen_ms"]["value"], CO["c_embed_ms"]["value"]
R = {"coefficients_ms": {"search_live": cs, "fetch_live": cf, "search_local": CO["c_search_ms"]["local"], "fetch_local": CO["c_fetch_ms"]["local"], "generation": cg, "query_embedding": ce}}
ops = {}
for p in "ABCD":
    m = P[p]["metrics"]; ov_cold = TIM[p]["cold_ms_per_request"]; ov_max = max(ov_cold, PAR[p])
    ops[p] = {"n_s": m["search_calls"] / N, "n_f": m["fetches"] / N, "n_g": m["generations"] / N, "embed_ms": nonrt * ce, "decision_ms_cold": ov_cold, "decision_ms_conservative": ov_max,
              "write_ms": PAY[p]["page_write_ms_per_request"]}
    for reg, (S, F) in {"live": (cs, cf), "local": (CO["c_search_ms"]["local"], CO["c_fetch_ms"]["local"])}.items():
        ops[p][f"cost_ms_{reg}"] = round(ops[p]["n_s"] * S + ops[p]["n_f"] * F + ops[p]["n_g"] * cg + ops[p]["embed_ms"] + ov_cold + ops[p]["write_ms"], 3)
R["policies"] = ops
def diff(x, y="D"):
    a, b = ops[x], ops[y]; do_c = (b["decision_ms_cold"] + b["write_ms"]) - (a["decision_ms_cold"] + a["write_ms"]); do_m = (b["decision_ms_conservative"] + b["write_ms"]) - (a["decision_ms_cold"] + a["write_ms"])
    return {"d_search": round(a["n_s"] - b["n_s"], 5), "d_fetch": round(a["n_f"] - b["n_f"], 5), "d_gen": round(a["n_g"] - b["n_g"], 5), "d_overhead_ms_measured": round(do_c, 4), "d_overhead_ms_conservative": round(max(do_m, do_c, 0), 4),
            "saving_ms_live": round((a["n_s"] - b["n_s"]) * cs + (a["n_f"] - b["n_f"]) * cf + (a["n_g"] - b["n_g"]) * cg - do_c, 2),
            "saving_pct_live": round(100 * ((a["n_s"] - b["n_s"]) * cs + (a["n_f"] - b["n_f"]) * cf - do_c) / a["cost_ms_live"], 2),
            "saving_ms_local": round((a["n_s"] - b["n_s"]) * CO["c_search_ms"]["local"] + (a["n_f"] - b["n_f"]) * CO["c_fetch_ms"]["local"] - do_c, 4)}
R["A_vs_D"] = diff("A"); R["B_vs_D"] = diff("B"); R["A_vs_B"] = diff("A", "B")
for k in ("A_vs_D", "B_vs_D"):
    d = R[k]; R[k]["break_even_search_cost_ms_if_fetch_free"] = {f"overhead_{o}ms": round(o / d["d_search"], 3) for o in (d["d_overhead_ms_conservative"], 1.0, 10.0)}
json.dump(R, open(O / "outputs/composition_cost.json", "w"), indent=1); print(json.dumps(R, indent=1))
# break-even map (log-scaled component costs so the break-even region is visible)
S = np.logspace(-2, np.log10(3000), 300); F = np.logspace(-2, np.log10(3000), 300); SS, FF = np.meshgrid(S, F)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a_, (k, title) in zip(ax, (("A_vs_D", "A (CategoryAware) vs D (+L2/L3)"), ("B_vs_D", "B (+L3 page cache) vs D (+L2/L3)"))):
    d = R[k]; Z = d["d_search"] * SS + d["d_fetch"] * FF
    im = a_.contourf(SS, FF, np.log10(np.maximum(Z, 1e-4)), levels=20, cmap="Greens"); cb = plt.colorbar(im, ax=a_); cb.set_label("log10 external-operation saving (ms/request)", fontsize=8)
    for o, ls in ((d["d_overhead_ms_conservative"], "-"), (1.0, "--"), (10.0, ":")):
        a_.contour(SS, FF, Z - o, levels=[0], colors="k", linestyles=ls, linewidths=1.2); a_.plot([], [], color="k", ls=ls, label=f"break-even, added overhead {o:.2g} ms")
    a_.scatter([cs], [cf], marker="*", s=200, c="#c0392b", edgecolor="k", zorder=5, label="measured live costs")
    a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("search cost c_s (ms)"); a_.set_ylabel("fetch cost c_f (ms)"); a_.set_title(title + ": beneficial above/right of each line", fontsize=9); a_.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(O / "outputs/break_even_map.pdf"); fig.savefig(O / "outputs/break_even_map.png", dpi=150); print("figure saved")
