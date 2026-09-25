#!/usr/bin/env python3
"""Paired contrasts between frozen configurations on the JOINTLY OBSERVABLE
held-out requests, so that a drift difference cannot be produced merely by one
configuration observing fewer requests than another."""
from __future__ import annotations
import csv, json, math, pathlib, random
from collections import defaultdict
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
SEED, B = 42, 10_000
CFG = ["A_original", "B_mult_only", "C_budget_only", "D_full",
       "E_equivalent_ttl"]
OBS = ("CHANGED", "UNCHANGED")
rows = list(csv.DictReader(open(HERE / "heldout_per_request.csv", encoding="utf-8")))
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def mcnemar(b01, b10):
    n = b01 + b10
    if not n:
        return 1.0
    return min(1.0, float(sum(Fraction(math.comb(n, i), 2 ** n)
                              for i in range(0, min(b01, b10) + 1)) * 2))


say("PAIRED CONTRASTS -- jointly observable held-out requests only")
say(f"  {'contrast':<28}{'n':>8}{'drift X':>10}{'drift Y':>10}{'delta pp':>11}"
    f"{'95% CI':>22}{'X->Y drift':>12}{'Y->X drift':>12}{'McNemar p':>13}")
out = {}
for y in CFG[1:]:
    pr = [r for r in rows if r["A_original"] in OBS and r[y] in OBS]
    n = len(pr)
    cx = sum(1 for r in pr if r["A_original"] == "CHANGED")
    cy = sum(1 for r in pr if r[y] == "CHANGED")
    b01 = sum(1 for r in pr if r["A_original"] == "UNCHANGED" and r[y] == "CHANGED")
    b10 = sum(1 for r in pr if r["A_original"] == "CHANGED" and r[y] == "UNCHANGED")
    byc = defaultdict(list)
    for r in pr:
        byc[r["cluster_id"]].append(r)
    ks = list(byc); rng = random.Random(SEED); ds = []
    for _ in range(B):
        fl = [x for _ in range(len(ks)) for x in byc[ks[rng.randrange(len(ks))]]]
        ds.append(100 * sum(1 for x in fl if x[y] == "CHANGED") / len(fl)
                  - 100 * sum(1 for x in fl if x["A_original"] == "CHANGED") / len(fl))
    ds.sort()
    ci = [round(ds[int(.025 * B)], 4), round(ds[int(.975 * B)], 4)]
    p = mcnemar(b01, b10)
    d = 100 * cy / n - 100 * cx / n
    out[f"A_vs_{y}"] = {"n": n, "drift_A_pct": round(100 * cx / n, 4),
                        "drift_other_pct": round(100 * cy / n, 4),
                        "delta_pp": round(d, 4), "ci95": ci,
                        "A_clean_other_drift": b01, "A_drift_other_clean": b10,
                        "mcnemar_p": p}
    say(f"  {'A vs '+y:<28}{n:>8,}{100*cx/n:>9.4f}%{100*cy/n:>9.4f}%"
        f"{d:>+11.4f}{str(ci):>22}{b01:>12}{b10:>12}{p:>13.4g}")

say(f"\n  Positive delta = the alternative drifts MORE than the published "
    f"configuration on the same requests.")
json.dump(out, open(HERE / "paired_contrasts.json", "w"), indent=2)
(PC / "logs" / "t6_paired.log").write_text("\n".join(log) + "\n", encoding="utf-8")
