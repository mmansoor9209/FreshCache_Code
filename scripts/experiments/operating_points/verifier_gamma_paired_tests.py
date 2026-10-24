"""Paired tests of the page-support verifier points against the primary FreshCache point (theta_L2 = 0.75) on the held-out audit
(Table 6 row "Verifier gamma = 0.55": WAI and sufficiency). Reads the request-level labels written by
l2_threshold_sweep_scoring.py (outputs/l2_operating_points/l2_threshold_request_level.csv); exact two-sided McNemar tests.
Writes outputs/operating_point_summaries/verifier_gamma_paired_tests.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, json, math, os, pathlib; OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
R = collections.defaultdict(dict)
for r in csv.DictReader(open(OUTPUT_ROOT / "l2_operating_points/l2_threshold_request_level.csv")): R[r["point"]][r["request_id"]] = r
print(sorted(R))
def mcn(b, c):
    n = b + c; return 1.0 if n == 0 else min(1, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
fc = R["theta_0.75"]; OUT = {}
for p in R:
    if "gamma" not in p and "0.55" not in p: continue
    g = R[p]; a = collections.Counter(x["answer_label"] for x in g.values()); s = sum(x["sufficiency"] == "SUFFICIENT" for x in g.values())
    wb = sum(fc[q]["fresh_label"] == "CORRECT" and fc[q]["answer_label"] == "WRONG" and g[q]["answer_label"] != "WRONG" for q in fc)
    wc = sum(fc[q]["fresh_label"] == "CORRECT" and g[q]["answer_label"] == "WRONG" and fc[q]["answer_label"] != "WRONG" for q in fc)
    sb = sum(fc[q]["sufficiency"] == "SUFFICIENT" and g[q]["sufficiency"] != "SUFFICIENT" for q in fc); sc = sum(g[q]["sufficiency"] == "SUFFICIENT" and fc[q]["sufficiency"] != "SUFFICIENT" for q in fc)
    print(p, dict(a), s, "WAI", wb, wc, round(mcn(wb, wc), 3), "SUFF", sb, sc, round(mcn(sb, sc), 4))
    OUT[p] = {"answer_labels": dict(a), "sufficient": s, "WAI_primary_only": wb, "WAI_point_only": wc, "WAI_p": mcn(wb, wc),
              "suff_primary_only": sb, "suff_point_only": sc, "suff_p": mcn(sb, sc)}
(OUTPUT_ROOT / "operating_point_summaries").mkdir(parents=True, exist_ok=True)
json.dump(OUT, open(OUTPUT_ROOT / "operating_point_summaries/verifier_gamma_paired_tests.json", "w"), indent=1)
