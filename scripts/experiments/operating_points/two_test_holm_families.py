"""Holm adjustment within the two two-test families of Appendix B and C (two tests, so the smaller p-value is doubled):
  * primary FreshCache vs theta_L2 = 0.80 (WAI and evidence sufficiency; App. C), raw p-values from
    outputs/l2_operating_points/l2_threshold_summary.json (l2_threshold_sweep_scoring.py);
  * L2 inherited list (A) vs cached list (B) in the tier diagnostic (answer accuracy and sufficiency; App. B), raw p-values from
    outputs/replay/shadow/reported/tier/results/paired_tests.csv (tier_diagnostic_analysis.py).
Writes outputs/operating_point_summaries/two_test_holm_families.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import csv, json, os, pathlib; OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
def holm(T):
    m = len(T); order = sorted(range(m), key=lambda i: T[i][1]); adj = [None] * m; run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * T[i][1])); adj[i] = run
    return [{"test": t, "p": p, "p_holm": a} for (t, p), a in zip(T, adj)]
P80 = next(p for p in json.load(open(OUTPUT_ROOT / "l2_operating_points/l2_threshold_summary.json"))["points"] if p["point"] == "theta_0.80")
F1 = [("primary vs theta_L2 0.80: WAI", P80["WAI_mcnemar_p"]), ("primary vs theta_L2 0.80: sufficiency", P80["sufficiency_mcnemar_p"])]
PT = {(r["task"], r["population"], r["contrast"]): float(r["mcnemar_p"]) for r in csv.DictReader(open(OUTPUT_ROOT / "replay/shadow/reported/tier/results/paired_tests.csv"))}
AB = "A_fresh_retrieval vs B_cached_list_current_pages"
F2 = [("L2 A vs B: answer accuracy", PT[("l2_reuse_answer", "all", AB)]), ("L2 A vs B: sufficiency", PT[("l2_reuse_sufficiency", "all", AB)])]
OUT = {"theta_L2_0.80_family": holm(F1), "L2_inherited_vs_cached_family": holm(F2)}
(OUTPUT_ROOT / "operating_point_summaries").mkdir(parents=True, exist_ok=True)
json.dump(OUT, open(OUTPUT_ROOT / "operating_point_summaries/two_test_holm_families.json", "w"), indent=1); print(json.dumps(OUT, indent=1))
