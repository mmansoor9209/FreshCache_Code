"""Paired tests of the selected CategoryAware + FreshCache L2/L3 composition (H80) against standalone FreshCache and against
CategoryAware-style on the held-out audit (Table 6 row "CA + FC L2/L3": WAI and sufficiency).
Reads outputs/categoryaware_freshcache/ca_composition_quality_audit.csv (categoryaware_composition.py score); exact two-sided McNemar tests.
Writes outputs/operating_point_summaries/categoryaware_composition_paired_tests.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, json, math, os, pathlib; OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
R = list(csv.DictReader(open(OUTPUT_ROOT / "categoryaware_freshcache/ca_composition_quality_audit.csv")))
print(len(R))
def mcn(b, c):
    n = b + c; return 1.0 if n == 0 else min(1, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
OUT = {"N": len(R), "arms": {}, "paired_vs_H80": {}}
for arm in ["FreshCache primary (theta_L2=0.75)", "CategoryAware-style (original)", "H80"]:
    a = collections.Counter(r[f"{arm} | answer"] for r in R); s = collections.Counter(r[f"{arm} | sufficiency"] for r in R)
    w = sum(r["fresh_label"] == "CORRECT" and r[f"{arm} | answer"] == "WRONG" for r in R)
    print(arm, dict(a), "suff", s["SUFFICIENT"], dict(s), "WAI", w)
    OUT["arms"][arm] = {"answer_labels": dict(a), "sufficiency_labels": dict(s), "WAI": w}
fc = "FreshCache primary (theta_L2=0.75)"; h = "H80"; ca = "CategoryAware-style (original)"
for o in (fc, ca):
    b = sum(r[f"{o} | sufficiency"] == "SUFFICIENT" and r[f"{h} | sufficiency"] != "SUFFICIENT" for r in R); c = sum(r[f"{h} | sufficiency"] == "SUFFICIENT" and r[f"{o} | sufficiency"] != "SUFFICIENT" for r in R)
    wb = sum(r["fresh_label"] == "CORRECT" and r[f"{o} | answer"] == "WRONG" and r[f"{h} | answer"] != "WRONG" for r in R); wc = sum(r["fresh_label"] == "CORRECT" and r[f"{h} | answer"] == "WRONG" and r[f"{o} | answer"] != "WRONG" for r in R)
    print("vs", o, "suff", b, c, round(mcn(b, c), 4), "WAI", wb, wc, round(mcn(wb, wc), 4))
    OUT["paired_vs_H80"][o] = {"suff_other_only": b, "suff_H80_only": c, "suff_p": mcn(b, c), "WAI_other_only": wb, "WAI_H80_only": wc, "WAI_p": mcn(wb, wc)}
(OUTPUT_ROOT / "operating_point_summaries").mkdir(parents=True, exist_ok=True)
json.dump(OUT, open(OUTPUT_ROOT / "operating_point_summaries/categoryaware_composition_paired_tests.json", "w"), indent=1)
