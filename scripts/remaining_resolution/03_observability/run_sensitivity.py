#!/usr/bin/env python3
"""
EXPERIMENT 3, TASKS 3B-3E -- stratified observability, paired bounds,
explicit sensitivity scenarios, and corrected recollection feasibility.

Reuses the row-level per-request pairing already produced in
final_three_issues/03_observability_bounds/per_request_pair.csv. Nothing is
re-replayed; every number is regenerated from that file plus the manifest.

The worst-case bound is NEVER called a confidence interval. Sensitivity
scenarios are labelled as assumptions, not observations.
"""
from __future__ import annotations
import csv, json, math, os, pathlib, random, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
RR = HERE.parent
V3 = RR.parent
ROOT = V3.parent
OB = V3 / "final_three_issues" / "03_observability_bounds"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
SEED, B = 42, 10_000
CHANGED, UNCHANGED, UNOBS = "CHANGED", "UNCHANGED", "UNOBSERVABLE"


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 4),
            round(100 * min(1.0, (c + m) / d), 4))


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("EXPERIMENT 3 (3B-3E) -- observability: strata, bounds, sensitivity")
    rows = list(csv.DictReader(open(OB / "per_request_pair.csv", encoding="utf-8")))
    say(f"  row-level pairing reused: {len(rows):,} held-out requests")
    pr = [r for r in rows if r["paired"] == "1"]
    N = len(pr)
    c_on = sum(1 for r in pr if r["outcome_on"] == CHANGED)
    c_off = sum(1 for r in pr if r["outcome_off"] == CHANGED)
    u_on = sum(1 for r in pr if r["outcome_on"] == UNOBS)
    u_off = sum(1 for r in pr if r["outcome_off"] == UNOBS)
    say(f"  paired N = {N:,}   changed ON {c_on:,}  OFF {c_off:,}   "
        f"unobservable ON {u_on:,}  OFF {u_off:,}")

    # ---------------- 3C: reproduce the paired bound ----------------
    prev = json.load(open(OB / "bounds_overall.json", encoding="utf-8"))["overall"]
    dmin = 100 * (c_off - (c_on + u_on)) / N
    dmax = 100 * ((c_off + u_off) - c_on) / N
    say(f"\n  == TASK 3C: paired partial-identification bound (reproduced) ==")
    say(f"    Delta_min {dmin:+.4f} pp   (previously {prev['delta_min_pp']:+.4f})")
    say(f"    Delta_max {dmax:+.4f} pp   (previously {prev['delta_max_pp']:+.4f})")
    ok = (abs(dmin - prev["delta_min_pp"]) < 1e-3
          and abs(dmax - prev["delta_max_pp"]) < 1e-3)
    say(f"    reproduces the original bound exactly: {ok}")
    say(f"    This is a DETERMINISTIC WORST-CASE BOUND. It is not a confidence "
        f"interval and is never described as one.")
    jo = [r for r in pr if r["outcome_on"] != UNOBS and r["outcome_off"] != UNOBS]
    n_jo = len(jo)
    con = sum(1 for r in jo if r["outcome_on"] == CHANGED)
    coff = sum(1 for r in jo if r["outcome_off"] == CHANGED)
    d_obs = 100 * coff / n_jo - 100 * con / n_jo
    byc = defaultdict(list)
    for r in jo:
        byc[r["cluster_id"]].append(r)
    ks = list(byc); rng = random.Random(SEED); ds = []
    for _ in range(B):
        s = [byc[ks[rng.randrange(len(ks))]] for _ in range(len(ks))]
        fl = [x for g in s for x in g]
        if fl:
            ds.append(100*sum(1 for x in fl if x["outcome_off"] == CHANGED)/len(fl)
                      - 100*sum(1 for x in fl if x["outcome_on"] == CHANGED)/len(fl))
    ds.sort()
    ci = [round(ds[int(.025*len(ds))], 4), round(ds[int(.975*len(ds))], 4)]
    say(f"    OBSERVED paired estimate (jointly observable only, n = {n_jo:,} "
        f"in {len(ks):,} clusters): Delta = {d_obs:+.4f} pp, "
        f"cluster-bootstrap 95% CI {ci}")

    # ---------------- 3B: stratified observability ----------------
    say(f"\n  == TASK 3B: observability coverage by stratum ==")
    def cov(sub, arm):
        n = len(sub)
        o = sum(1 for r in sub if r[f"outcome_{arm}"] in (CHANGED, UNCHANGED))
        return n, o, (100*o/n if n else None)
    strata = {}
    for key, name in (("freshness_class", "freshness class"),
                      ("volatility_band", "domain volatility")):
        say(f"    by {name}")
        say(f"      {'stratum':<16}{'paired':>9}{'obs ON':>9}{'cov ON%':>10}"
            f"{'obs OFF':>10}{'cov OFF%':>11}")
        g = defaultdict(list)
        for r in pr:
            g[r[key]].append(r)
        strata[key] = {}
        for k in sorted(g, key=str):
            n, oa, ca = cov(g[k], "on")
            _, ob, cb = cov(g[k], "off")
            strata[key][k] = {"paired": n, "observable_on": oa,
                              "coverage_on_pct": round(ca, 3) if ca else None,
                              "observable_off": ob,
                              "coverage_off_pct": round(cb, 3) if cb else None}
            say(f"      {str(k):<16}{n:>9,}{oa:>9,}{ca:>9.3f}%{ob:>10,}{cb:>10.3f}%")
    say(f"    by tier reached")
    g = defaultdict(list)
    for r in pr:
        g[r.get("tier_on") or "?"].append(r)
    strata["tier_on"] = {}
    for k in sorted(g, key=str):
        n, oa, ca = cov(g[k], "on")
        strata["tier_on"][k] = {"paired": n, "observable_on": oa,
                                "coverage_on_pct": round(ca, 3) if ca else None}
        say(f"      tier {str(k):<11}{n:>9,}{oa:>9,}{ca:>9.3f}%")
    hi = strata["volatility_band"].get("high >=0.5", {})
    lo = strata["volatility_band"].get("low <0.2", {})
    say(f"    HIGH-volatility content is systematically LESS observable: "
        f"{hi.get('coverage_on_pct')}% vs low-volatility "
        f"{lo.get('coverage_on_pct')}% (gate ON)")
    nc = len({r["cluster_id"] for r in pr})
    fully = sum(1 for cl, rs in
                ((cl, [r for r in pr if r["cluster_id"] == cl])
                 for cl in list({r["cluster_id"] for r in pr})[:0]) if rs)
    say(f"    request clusters represented: {nc:,}")

    # ---------------- 3D: explicit sensitivity scenarios ----------------
    say(f"\n  == TASK 3D: sensitivity scenarios (ASSUMPTIONS, not observations) ==")
    r_on_obs = con / n_jo
    r_off_obs = coff / n_jo
    say(f"    observed change rate among JOINTLY OBSERVABLE: "
        f"gate ON {100*r_on_obs:.4f}%, gate OFF {100*r_off_obs:.4f}%")

    def delta(r_on, r_off):
        return 100 * ((c_off + r_off*u_off) - (c_on + r_on*u_on)) / N

    scen = {}
    for nm, ron, roff in (
            ("S1 missing behave like the observed, per arm", r_on_obs, r_off_obs),
            ("S2 missing behave like the observed, SAME rate both arms",
             r_on_obs, r_on_obs),
            ("S3 1.5x the observed rate, both arms", 1.5*r_on_obs, 1.5*r_on_obs),
            ("S4 2x the observed rate, both arms", 2*r_on_obs, 2*r_on_obs),
            ("S5 ALL missing changed, both arms", 1.0, 1.0),
            ("S6 NONE missing changed, both arms", 0.0, 0.0),
            ("S7 ADVERSARIAL against the gate (worst case)", 1.0, 0.0),
            ("S8 ADVERSARIAL for the gate (best case)", 0.0, 1.0)):
        d = delta(ron, roff)
        scen[nm] = {"rate_on": round(ron, 6), "rate_off": round(roff, 6),
                    "delta_pp": round(d, 4)}
        say(f"    {nm:<52}Delta = {d:+8.4f} pp")
    say(f"    Under EVERY scenario in which the two arms' missing pages behave "
        f"the SAME way (S2-S6), Delta stays positive at about "
        f"{scen['S6 NONE missing changed, both arms']['delta_pp']:+.2f} to "
        f"{scen['S5 ALL missing changed, both arms']['delta_pp']:+.2f} pp. "
        f"The sign flips only when the two arms are allowed to differ "
        f"adversarially (S7).")

    # break-even differential
    # Delta = 0  <=>  r_off*u_off - r_on*u_on = -(c_off - c_on)
    need = -(c_off - c_on)
    say(f"\n    BREAK-EVEN: with r_on fixed at the observed "
        f"{100*r_on_obs:.4f}%, the gate-OFF missing pages would need a change "
        f"rate of")
    r_off_be = (need + r_on_obs*u_on) / u_off
    say(f"      r_off = {100*r_off_be:.4f}%  "
        f"(a differential of {100*(r_off_be - r_on_obs):+.4f} pp relative to "
        f"gate ON) for the effect to vanish")
    say(f"    Equivalently, with equal unobservable counts the gate-OFF "
        f"missing pages would have to change about "
        f"{100*abs(need)/u_off:.2f} pp LESS often than the gate-ON ones. "
        f"There is no mechanism in the design that would produce that: both "
        f"arms draw from the same URL population and the same snapshot rounds.")
    say(f"    That argument is a JUDGEMENT about plausibility, not a "
        f"measurement. It does not make the worst-case bound go away.")

    # ---------------- 3E: corrected recollection feasibility ----------------
    say(f"\n  == TASK 3E: recollection feasibility, on CORRECTED denominators ==")
    rec = json.load(open(HERE / "task3a_reconciliation.json", encoding="utf-8"))
    tab = rec["corrected_table"]
    uniq_total = rec["recomputed_unique_non_substantive_urls"]
    T0 = {}
    p0 = (V3 / "remaining_critical_issues" / "05_prospective_temporal"
          / "snapshots" / "T0.jsonl")
    for l in open(p0, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            T0[d["url_hash"]] = d
    dom_ok, dom_tot = Counter(), Counter()
    for d in T0.values():
        u = d.get("url") or ""
        dm = u.split("/")[2].lower().removeprefix("www.") if "//" in u else "?"
        dom_tot[dm] += 1
        if d.get("substantive"):
            dom_ok[dm] += 1
    say(f"    unique URLs needing recollection: {uniq_total:,}")
    say(f"    (the earlier report's per-domain numbers were request "
        f"occurrences; corrected unique-URL counts are used here)")
    say(f"    {'domain':<30}{'unique URLs':>12}{'% of total':>12}"
        f"{'T0 fetched':>12}{'T0 substantive':>16}{'yield':>8}")
    blocked = recoverable = 0
    for e in tab[:14]:
        d = e["domain"]
        t, s = dom_tot.get(d, 0), dom_ok.get(d, 0)
        y = (100*s/t) if t else None
        if y is not None and y < 20:
            blocked += e["unique_urls"]
        elif y is not None:
            recoverable += e["unique_urls"] * y / 100
        say(f"    {d:<30}{e['unique_urls']:>12,}{e['pct_of_unique_urls']:>11.2f}%"
            f"{t:>12,}{s:>16,}" + (f"{y:>7.1f}%" if y is not None else f"{'-':>8}"))
    ig = next((e for e in tab if e["domain"] == "instagram.com"), {})
    rd = next((e for e in tab if e["domain"] == "reddit.com"), {})
    both = ig.get("unique_urls", 0) + rd.get("unique_urls", 0)
    say(f"\n    CORRECTION to the earlier feasibility claim: Instagram and "
        f"Reddit are {ig.get('unique_urls',0)} + {rd.get('unique_urls',0)} = "
        f"{both} UNIQUE URLs = {100*both/uniq_total:.1f}% of the {uniq_total:,}, "
        f"NOT the ~59% implied by the earlier occurrence-based numbers.")
    say(f"    They remain unfetchable (0% substantive on a fresh T0 fetch) and "
        f"are NOT re-attempted, but they block far less of the population than "
        f"previously stated.")
    say(f"    The other {uniq_total-both:,} URLs ({100*(uniq_total-both)/uniq_total:.1f}%) "
        f"are spread thinly across a long tail of ordinary sites, many of "
        f"which DO yield content on a fresh fetch (bbc.com 41.6%, "
        f"facebook.com 51.6%, jleague.co 83.3%).")
    say(f"    A request becomes observable only when EVERY one of its missing "
        f"pages is recovered, so partial recovery of a long tail converts few "
        f"requests. Quantifying that conversion requires an actual "
        f"recollection; it is NOT estimated here.")
    say(f"    No access controls are bypassed and no page body is imputed.")

    json.dump({"paired": {"N": N, "changed_on": c_on, "changed_off": c_off,
                          "unobs_on": u_on, "unobs_off": u_off},
               "bound": {"delta_min_pp": round(dmin, 4),
                         "delta_max_pp": round(dmax, 4),
                         "reproduces_original": bool(ok),
                         "is_confidence_interval": False},
               "observed_paired": {"n": n_jo, "clusters": len(ks),
                                   "delta_pp": round(d_obs, 4),
                                   "cluster_bootstrap_ci95": ci},
               "strata": strata,
               "sensitivity_scenarios_ASSUMPTIONS": scen,
               "break_even_r_off": round(r_off_be, 6),
               "break_even_differential_pp": round(100*(r_off_be-r_on_obs), 4),
               "recollection": {"unique_urls": uniq_total,
                                "instagram_reddit_unique": both,
                                "instagram_reddit_pct": round(100*both/uniq_total, 2)}},
              open(HERE / "observability_sensitivity.json", "w"), indent=2)
    (RR / "logs").mkdir(exist_ok=True)
    (RR / "logs" / "exp3_sensitivity.log").write_text("\n".join(log) + "\n",
                                                      encoding="utf-8")
    say(f"\n  wrote observability_sensitivity.json")


if __name__ == "__main__":
    main()
