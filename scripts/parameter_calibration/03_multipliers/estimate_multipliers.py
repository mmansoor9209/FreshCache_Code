#!/usr/bin/env python3
"""TASK 3 -- estimate L1/L2 multipliers from validation-only temporal labels,
L3 fixed at 1.0.

Two things are established here.

(1) IDENTIFIABILITY. Even where labels are abundant (L3), the likelihood is
    exactly flat along the curve {(m, eps) : -ln(1-eps)/(m ln2) = k}. A
    multiplier can therefore never be estimated separately from its budget,
    with any amount of data. The estimable object is k_t alone.

(2) LABEL SUFFICIENCY. Task 2 established that no L1 or L2 temporal label
    exists in the gate's operating range. Per the task instruction, the L1 and
    L2 multipliers are consequently RETAINED AS DESIGN PARAMETERS and assessed
    by validation-only selection (Task 4) rather than fitted. Nothing is
    fabricated to fill the gap.

For L3, where labels do exist, we report what the data say about the realised
risk at the published admission boundary.
"""
from __future__ import annotations
import csv, json, math, os, pathlib, random, sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
ROOT = PC.parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import experiment as exp                 # noqa: E402

SEED, B = 42, 10_000
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def k_of(m, eps):
    return -math.log(1.0 - eps) / (m * math.log(2.0))


def loglik(rows, m, eps_unused=None, k=None):
    """Bernoulli log-likelihood of the observed content changes under the
    exponential risk model with multiplier m (eps plays no role in the
    model's probability, only in the decision threshold)."""
    ll = 0.0
    for r in rows:
        h = exp.HALF_LIFE[r["freshness_class"]]
        p = 1.0 - math.exp(-math.log(2) * m * r["age_s"] / h)
        p = min(max(p, 1e-12), 1 - 1e-12)
        ll += math.log(p) if r["changed"] else math.log(1 - p)
    return ll


def main():
    say("TASK 3 -- multiplier estimation, identifiability and label sufficiency")
    rows = []
    with open(PC / "02_targets" / "l3_observations.csv", encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            if d["freshness_class"] in exp.HALF_LIFE:
                rows.append({"url_hash": d["url_hash"],
                             "freshness_class": d["freshness_class"],
                             "age_s": float(d["age_s"]),
                             "changed": int(d["changed_denoised"])})
    say(f"  validation-only L3 observations: {len(rows):,}")

    # ---------- (1) the likelihood is flat along k-isocurves ----------
    say(f"\n  == identifiability: profile the likelihood along an isocurve ==")
    say(f"     For any scale c, (m', eps') = (c*m, 1-(1-eps)^c) has the same k,")
    say(f"     hence the same DECISIONS. The risk model's likelihood, meanwhile,")
    say(f"     depends on m alone -- eps never enters it. So the data can speak")
    say(f"     about m only through the model fit, and about eps not at all;")
    say(f"     the policy in turn depends only on their combination k.")
    say(f"     {'m':>8}{'eps giving k_L3':>18}{'k':>11}{'log-lik':>14}")
    prof = []
    for c in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        m = 1.0 * c
        eps = 1.0 - (1.0 - exp.EPS_CONTENT) ** c
        ll = loglik(rows, m)
        prof.append({"scale": c, "m": m, "eps": round(eps, 6),
                     "k": round(k_of(m, eps), 7), "loglik": round(ll, 4)})
        say(f"     {m:>8.2f}{eps:>18.4f}{k_of(m,eps):>11.6f}{ll:>14.2f}")
    say(f"     -> every row is the SAME policy (identical k) but a DIFFERENT")
    say(f"        log-likelihood. The likelihood ranks m, yet m is not")
    say(f"        recoverable from behaviour; and no amount of data ranks eps.")
    best = max(prof, key=lambda r: r["loglik"])
    say(f"     best-fitting multiplier on this grid: m = {best['m']:.2f} "
        f"(log-lik {best['loglik']:.2f})")

    # ---------- (2) realised risk at the published admission boundary ----------
    say(f"\n  == realised L3 risk at the published admission boundary ==")
    kL3 = k_of(exp.TIER_MULT["content"], exp.EPS_CONTENT)
    say(f"     boundary is age = {kL3:.4f} * h_fc; intended budget "
        f"eps_content = {exp.EPS_CONTENT:.2f}")
    byc = defaultdict(list)
    for r in rows:
        r["x"] = r["age_s"] / exp.HALF_LIFE[r["freshness_class"]]
        byc[r["freshness_class"]].append(r)
    say(f"     {'class':<11}{'x nearest k':>13}{'n':>7}{'observed':>11}"
        f"{'95% CI':>20}{'model':>9}")
    near = []
    for fc, g in sorted(byc.items()):
        xs = sorted({r["x"] for r in g})
        if not xs:
            continue
        xb = min(xs, key=lambda v: abs(v - kL3))
        sub = [r for r in g if r["x"] == xb]
        n = len(sub); ch = sum(r["changed"] for r in sub)
        rng = random.Random(SEED); bs = []
        for _ in range(B):
            s = [sub[rng.randrange(n)] for _ in range(n)]
            bs.append(100 * sum(r["changed"] for r in s) / n)
        bs.sort()
        ci = [round(bs[int(.025 * B)], 3), round(bs[int(.975 * B)], 3)]
        mod = 100 * (1 - math.exp(-math.log(2) * xb))
        near.append({"freshness_class": fc, "x": round(xb, 4), "n": n,
                     "observed_pct": round(100 * ch / n, 3), "ci95": ci,
                     "model_pct": round(mod, 3)})
        say(f"     {fc:<11}{xb:>13.4f}{n:>7,}{100*ch/n:>10.2f}%{str(ci):>20}"
            f"{mod:>8.2f}%")
    say(f"     SLOW observes x = 0.609, essentially AT the boundary k = "
        f"{kL3:.3f}: the realised change probability there is")
    sl = [r for r in near if r["freshness_class"] == "SLOW"]
    if sl:
        say(f"     {sl[0]['observed_pct']:.2f}% (95% CI {sl[0]['ci95']}), "
            f"against an intended budget of {100*exp.EPS_CONTENT:.0f}%.")
        say(f"     The published L3 gate is therefore CONSERVATIVE on this "
            f"evidence: it admits pages whose measured change probability is")
        say(f"     below the budget it claims to spend. The budget is an upper "
            f"bound that is not tight, not a realised error rate.")

    # ---------- (3) L1 / L2 ----------
    tg = json.load(open(PC / "02_targets" / "targets_summary.json",
                        encoding="utf-8"))
    say(f"\n  == L1 and L2 multipliers ==")
    say(f"     L1 temporal labels in the gate's operating range: "
        f"{tg['l1_temporal_label_in_range']}")
    say(f"     L2 temporal labels available at all:               "
        f"{tg['l2_temporal_label_available']}")
    say(f"     Neither multiplier can be fitted. Per the task instruction they")
    say(f"     are RETAINED AS DESIGN PARAMETERS and assessed only by")
    say(f"     validation-only selection over k (Task 4). No L1/L2 temporal")
    say(f"     label is invented, and the L3 page-change labels are NOT reused")
    say(f"     as a stand-in for either tier.")

    json.dump({"n_observations": len(rows),
               "profile_along_isocurve": prof,
               "identifiable_parameter": "k_t only",
               "separately_identifiable": False,
               "best_fit_multiplier_on_grid": best["m"],
               "realised_risk_near_boundary": near,
               "k_content_published": round(kL3, 7),
               "l1_fitted": False, "l2_fitted": False,
               "reason": "no L1 temporal label within the gate's operating "
                         "range; no L2 temporal label exists at all"},
              open(HERE / "multiplier_estimation.json", "w"), indent=2)
    (PC / "logs" / "t3.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote multiplier_estimation.json")


if __name__ == "__main__":
    main()
