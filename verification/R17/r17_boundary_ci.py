"""Boundary-aware stratified interval for Table 3: independent per-stratum Jeffreys posteriors
Beta(y_h+1/2, n_h-y_h+1/2), combined with the design weights W_h=N_h/N by Monte Carlo (200k draws, seed 42).
Also Clopper-Pearson-based sensitivity (Beta(y+1,n-y+1) i.e. uniform prior) and the naive Wald/FPC for reference."""
import json, numpy as np
d = json.load(open("R17_results.json")); rng = np.random.default_rng(42); B = 200_000
out = {}
for row, v in d.items():
    S = v["strata"]; N = sum(s["N_h"] for s in S.values())
    draws_j = np.zeros(B); draws_u = np.zeros(B)
    for s in S.values():
        W = s["N_h"]/N; y, n = s["y_h"], s["n_h"]
        draws_j += W*rng.beta(y+0.5, n-y+0.5, B)
        draws_u += W*rng.beta(y+1.0, n-y+1.0, B)
    est = 100*sum(s["N_h"]/N*s["y_h"]/s["n_h"] for s in S.values())
    out[row] = {"estimate": round(est,2), "jeffreys_stratified_95": [round(100*x,2) for x in np.quantile(draws_j,[.025,.975])],
                "uniform_prior_stratified_95": [round(100*x,2) for x in np.quantile(draws_u,[.025,.975])],
                "old_printed": v["old_ci_pct"], "bootstrap": v["stratified_bootstrap_ci_pct"], "sum_wilson_bound": v["conservative_bound_pct_sum_of_stratum_wilson"],
                "saturated_strata": v["saturated_strata"], "weight_in_saturated_strata_pct": round(100*sum(s["N_h"] for s in S.values() if s["y_h"] in (0,s["n_h"]))/N,1)}
    print(f"{row:42s} est {est:6.2f}  old {v['old_ci_pct']}  boot {v['stratified_bootstrap_ci_pct']}  JEFFREYS {out[row]['jeffreys_stratified_95']}  uniform {out[row]['uniform_prior_stratified_95']}  sat={v['saturated_strata']} ({out[row]['weight_in_saturated_strata_pct']}% weight)")
json.dump(out, open("R17_boundary_ci.json","w"), indent=1)
