#!/usr/bin/env python3
"""
Issue 3 -- CORRECTED paired observability bounds.

WHAT WAS WRONG BEFORE
  The earlier report compared two MARGINAL intervals,
      gate ON  = [3.0660%, 10.0869%]
      gate OFF = [6.4855%, 16.2001%]
  and claimed they do not overlap. They DO overlap, on [6.4855%, 10.0869%].
  That claim is withdrawn. Comparing marginal bounds is also the wrong
  procedure: the two arms are evaluated on the SAME requests, so the bound
  must be computed on the PAIRED difference, where an unobservable outcome
  constrains both arms simultaneously.

WHAT THIS COMPUTES INSTEAD
  Delta = Drift(gate off) - Drift(gate on), on the requests where BOTH arms
  reused content, so both drifts share one denominator N.

  Every request in that paired set falls in exactly one cell:
      both observable / on-only observable / off-only observable / neither.

  Observed outcomes are kept exactly as observed. Unobservable outcomes are
  then assigned ADVERSARIALLY:

      Delta_min  makes the gate look as BAD as arithmetically possible:
                 every unobservable gate-ON outcome becomes CHANGED,
                 every unobservable gate-OFF outcome becomes UNCHANGED.
      Delta_max  makes the gate look as GOOD as possible: the mirror image.

  [Delta_min, Delta_max] is a DETERMINISTIC partial-identification
  (worst-case) bound. It is NOT a confidence interval and is never described
  as one. A paired cluster bootstrap is computed separately, on the jointly
  observable subset only, and the two are never combined.

  Delta_min > 0  =>  the gate reduces drift even under the worst possible
                     assignment of every unobservable paired outcome.
  Delta_min <= 0 =>  the observability issue is NOT fully resolved.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, pathlib, random, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-final3")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
V3 = BASE.parent
ROOT = V3.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402
import prep2 as p2                       # noqa: E402
from freshcache.risk_model import (      # noqa: E402
    _DEFAULT_DOMAIN_VOLATILITY, _DEFAULT_DOMAIN_VOL)

CHANGED, UNCHANGED, UNOBS = me.CHANGED, me.UNCHANGED, me.UNOBS
SCHEDULE, SEED, B = "zipf_uniform", 42, 10_000
WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
         "v14_baselines/engine_all.py"]
ANCHOR = {"search_saved_pct": 60.5776, "l1_hits": 806, "l2_hits": 12429}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def domvol(domain):
    d = (domain or "").lower().removeprefix("www.")
    v = _DEFAULT_DOMAIN_VOLATILITY.get(d, _DEFAULT_DOMAIN_VOL)
    if v == _DEFAULT_DOMAIN_VOL:
        for k, vv in _DEFAULT_DOMAIN_VOLATILITY.items():
            if d.endswith("." + k) or k in d:
                return vv
    return v


def volband(v):
    return ("low <0.2" if v < 0.2 else "mid 0.2-0.5" if v < 0.5 else "high >=0.5")


def bounds(pairs):
    """pairs: list of (outcome_on, outcome_off). Returns the deterministic
    paired partial-identification bound on Delta = drift_off - drift_on."""
    n = len(pairs)
    if not n:
        return None
    c_on = sum(1 for a, _ in pairs if a == CHANGED)
    c_off = sum(1 for _, b in pairs if b == CHANGED)
    u_on = sum(1 for a, _ in pairs if a == UNOBS)
    u_off = sum(1 for _, b in pairs if b == UNOBS)
    # gate as BAD as possible: every unknown ON becomes CHANGED,
    #                          every unknown OFF becomes UNCHANGED
    dmin = 100.0 * ((c_off) - (c_on + u_on)) / n
    # gate as GOOD as possible: the mirror image
    dmax = 100.0 * ((c_off + u_off) - (c_on)) / n
    obs = [(a, b) for a, b in pairs if a != UNOBS and b != UNOBS]
    d_obs = (100.0 * (sum(1 for _, b in obs if b == CHANGED)
                      - sum(1 for a, _ in obs if a == CHANGED)) / len(obs)
             if obs else None)
    return {"n_paired": n, "changed_on_observed": c_on,
            "changed_off_observed": c_off,
            "unobservable_on": u_on, "unobservable_off": u_off,
            "delta_min_pp": round(dmin, 4), "delta_max_pp": round(dmax, 4),
            "n_jointly_observable": len(obs),
            "observed_delta_pp": (round(d_obs, 4) if d_obs is not None else None),
            "observed_drift_on_pct": (round(100 * sum(1 for a, _ in obs if a == CHANGED)
                                            / len(obs), 4) if obs else None),
            "observed_drift_off_pct": (round(100 * sum(1 for _, b in obs if b == CHANGED)
                                             / len(obs), 4) if obs else None)}


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("ISSUE 3 -- corrected PAIRED observability bounds")
    say("  WITHDRAWN: the earlier claim that gate-ON [3.0660, 10.0869] and")
    say("  gate-OFF [6.4855, 16.2001] do not overlap. They overlap on")
    say("  [6.4855, 10.0869]. Marginal bounds are also the wrong comparison")
    say("  for two arms evaluated on the same requests.")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    dom = {}
    for m in manifest:
        if m.get("url_hash") and m.get("domain"):
            dom.setdefault(m["url_hash"], m["domain"])

    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c, val_c = set(split["test_clusters"]), set(split["validation_clusters"])
    assert not (test_c & val_c)
    full = sc.build_stream(records, SCHEDULE, SEED)
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(stream) == split["test"]["requests"]
    assert not [1 for _, r in stream
                if (r.get("cluster_id") or r["query_id"]) in val_c]
    say(f"\n  held-out stream {len(stream):,} requests, validation leakage 0")

    m_on, per_on = me.replay(stream, rounds, "FreshCache", rich=rich)
    bad = [k for k, v in ANCHOR.items()
           if abs(m_on[k] - v) > (1e-4 if isinstance(v, float) else 0)]
    if bad:
        say(f"  ANCHOR GATE FAILED on {bad}"); sys.exit(2)
    say(f"  gate ON  saved {m_on['search_saved_pct']:.4f}%  L1 {m_on['l1_hits']:,}"
        f"  L2 {m_on['l2_hits']:,}  L3 {m_on['l3_hits']:,}  -> anchor reproduces")
    m_off, per_off = me.replay(stream, rounds, "FreshCache_AlwaysPassTemporal",
                               rich=rich)
    say(f"  gate OFF saved {m_off['search_saved_pct']:.4f}%  "
        f"L1 {m_off['l1_hits']:,}  L2 {m_off['l2_hits']:,}  L3 {m_off['l3_hits']:,}")

    ON = dict(per_on)
    OFF = dict(per_off)

    # served URLs per request per arm, for volatility and recollection analysis
    _, rowsA = p2.replay(stream, {}, "FreshCache")
    _, rowsB = p2.replay(stream, {}, "FreshCache_AlwaysPassTemporal")

    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    fco = {r["query_id"]: r["freshness_class"] for r in records}

    def reused_urls(row):
        if not row:
            return []
        if row.get("tier") == "L1":
            return list(row.get("served") or [])
        return [u for u, _ in (row.get("ev") or [])]

    rows, pairs = [], []
    cells = Counter()
    for t, r in stream:
        q = r["query_id"]
        a, b = ON.get(q), OFF.get(q)
        both = a is not None and b is not None
        us = reused_urls(rowsA.get(q)) or reused_urls(rowsB.get(q)) \
            or [u["url_hash"] for u in r["urls"]]
        vols = [domvol(dom.get(u, "")) for u in us] or [_DEFAULT_DOMAIN_VOL]
        mv = sum(vols) / len(vols)
        cell = ("not_paired" if not both else
                "both_observable" if a != UNOBS and b != UNOBS else
                "on_only_observable" if a != UNOBS else
                "off_only_observable" if b != UNOBS else "neither_observable")
        cells[cell] += 1
        rows.append({"query_id": q, "cluster_id": cid[q],
                     "freshness_class": fco.get(q),
                     "tier_on": (rowsA.get(q) or {}).get("tier"),
                     "tier_off": (rowsB.get(q) or {}).get("tier"),
                     "outcome_on": a, "outcome_off": b,
                     "paired": int(both), "cell": cell,
                     "n_reused_urls": len(us),
                     "mean_domain_volatility": round(mv, 4),
                     "volatility_band": volband(mv)})
        if both:
            pairs.append((a, b))

    say(f"\n  == paired observability cells ==")
    tot = len(stream)
    for k in ("both_observable", "on_only_observable", "off_only_observable",
              "neither_observable", "not_paired"):
        say(f"    {k:<22}{cells[k]:>8,}  {100*cells[k]/tot:>6.2f}% of requests")
    npair = len(pairs)
    obs_on = sum(1 for a, _ in pairs if a != UNOBS)
    obs_off = sum(1 for _, b in pairs if b != UNOBS)
    say(f"    paired requests (both arms reused): {npair:,}")
    say(f"      observable in gate-ON  {obs_on:,} ({100*obs_on/npair:.2f}%)")
    say(f"      observable in gate-OFF {obs_off:,} ({100*obs_off/npair:.2f}%)")
    say(f"      jointly observable     {cells['both_observable']:,} "
        f"({100*cells['both_observable']/npair:.2f}%)")
    say(f"      unobservable ONLY in gate-ON  {cells['on_only_observable'] and 0 or 0}"
        f"")
    say(f"      unobservable only in gate-ON : {cells['off_only_observable']:,}")
    say(f"      unobservable only in gate-OFF: {cells['on_only_observable']:,}")
    say(f"      unobservable in BOTH         : {cells['neither_observable']:,}")

    # ---------------- overall bound ----------------
    ov = bounds(pairs)
    say(f"\n  == PAIRED PARTIAL-IDENTIFICATION BOUND on "
        f"Delta = drift(gate off) - drift(gate on) ==")
    say(f"    denominator N = {ov['n_paired']:,} paired requests")
    say(f"    observed CHANGED: gate-on {ov['changed_on_observed']:,}, "
        f"gate-off {ov['changed_off_observed']:,}")
    say(f"    unobservable: gate-on {ov['unobservable_on']:,}, "
        f"gate-off {ov['unobservable_off']:,}")
    say(f"    Delta_min = {ov['delta_min_pp']:+.4f} pp   "
        f"(every unknown gate-ON outcome forced to CHANGED, every unknown "
        f"gate-OFF outcome forced to UNCHANGED)")
    say(f"    Delta_max = {ov['delta_max_pp']:+.4f} pp   (mirror image)")
    say(f"    BOUND [{ov['delta_min_pp']:+.4f}, {ov['delta_max_pp']:+.4f}] pp "
        f"-- a deterministic worst-case partial-identification bound, "
        f"NOT a confidence interval")
    resolved = ov["delta_min_pp"] > 0
    say(f"    Delta_min > 0 : {resolved}  -> "
        + ("the temporal gate reduces drift even under the worst possible "
           "assignment of every unobservable paired outcome"
           if resolved else
           "the observability issue is NOT fully resolved: an adversarial "
           "assignment of the unobservable outcomes can erase or reverse the "
           "gate's benefit"))

    # ---------------- bootstrap on the OBSERVED paired effect ----------------
    jo = [(q, ON[q], OFF[q]) for q in ON
          if ON.get(q) is not None and OFF.get(q) is not None
          and ON[q] != UNOBS and OFF[q] != UNOBS]
    byc = defaultdict(list)
    for q, a, b in jo:
        byc[cid[q]].append((a, b))
    ks = list(byc)
    rng = random.Random(SEED)
    ds = []
    for _ in range(B):
        s = [byc[ks[rng.randrange(len(ks))]] for _ in range(len(ks))]
        fl = [x for g in s for x in g]
        if not fl:
            continue
        ds.append(100 * sum(1 for _, b in fl if b == CHANGED) / len(fl)
                  - 100 * sum(1 for a, _ in fl if a == CHANGED) / len(fl))
    ds.sort()
    boot = {"n_jointly_observable": len(jo), "clusters": len(ks),
            "resamples": len(ds), "seed": SEED,
            "observed_delta_pp": ov["observed_delta_pp"],
            "ci95_pp": [round(ds[int(.025 * len(ds))], 4),
                        round(ds[int(.975 * len(ds))], 4)]}
    say(f"\n  == SEPARATE: paired cluster bootstrap on the JOINTLY OBSERVABLE "
        f"subset only ==")
    say(f"    n = {len(jo):,} requests in {len(ks):,} clusters, "
        f"{len(ds):,} resamples, seed {SEED}")
    say(f"    observed drift gate-on {ov['observed_drift_on_pct']:.4f}%  "
        f"gate-off {ov['observed_drift_off_pct']:.4f}%")
    say(f"    observed Delta = {ov['observed_delta_pp']:+.4f} pp   "
        f"95% CI [{boot['ci95_pp'][0]:+.4f}, {boot['ci95_pp'][1]:+.4f}] pp")
    say(f"    This CI is a sampling-uncertainty statement about the OBSERVED "
        f"subset only. It is NOT combined with the worst-case bound above, "
        f"and it does not address missingness.")

    # ---------------- by class and by volatility ----------------
    def grouped(key):
        g = defaultdict(list)
        idx = {r["query_id"]: r for r in rows}
        for t, r in stream:
            q = r["query_id"]
            rr = idx[q]
            if rr["paired"]:
                g[rr[key]].append((rr["outcome_on"], rr["outcome_off"]))
        return {k: bounds(v) for k, v in g.items() if v}

    bycl = grouped("freshness_class")
    byvo = grouped("volatility_band")
    for name, blk in (("freshness class", bycl), ("domain-volatility band", byvo)):
        say(f"\n  == paired bound by {name} ==")
        say(f"    {'group':<16}{'N':>8}{'unobs_on':>10}{'unobs_off':>11}"
            f"{'Delta_min':>11}{'Delta_max':>11}{'observed':>10}")
        for k in sorted(blk, key=str):
            r = blk[k]
            od = (f"{r['observed_delta_pp']:+.3f}"
                  if r["observed_delta_pp"] is not None else "-")
            say(f"    {str(k):<16}{r['n_paired']:>8,}{r['unobservable_on']:>10,}"
                f"{r['unobservable_off']:>11,}{r['delta_min_pp']:>+11.3f}"
                f"{r['delta_max_pp']:>+11.3f}{od:>10}")

    # ---------------- what would targeted recollection need? ----------------
    recol = None
    if not resolved:
        say(f"\n  == what drives the remaining uncertainty? ==")
        miss = Counter()
        missurl = Counter()
        urls_needed = set()
        idx = {r["query_id"]: r for r in rows}
        for t, r in stream:
            q = r["query_id"]
            rr = idx[q]
            if not rr["paired"]:
                continue
            if rr["outcome_on"] == UNOBS or rr["outcome_off"] == UNOBS:
                miss[(rr["freshness_class"], rr["volatility_band"])] += 1
                for u in set(reused_urls(rowsA.get(q))
                             + reused_urls(rowsB.get(q))):
                    per = rounds.get(u) or {}
                    if not (per.get("run_00", {}).get("substantive")):
                        urls_needed.add(u)
                        missurl[dom.get(u, "?")] += 1
        say(f"    paired requests with at least one unobservable arm: "
            f"{sum(miss.values()):,}")
        say(f"    distinct URLs that would need recollection: {len(urls_needed):,}")
        say(f"    top domains by contribution")
        for d_, c_ in missurl.most_common(15):
            say(f"      {d_:<40}{c_:>6,}")
        say(f"    by (class, volatility band)")
        for k, c_ in miss.most_common(12):
            say(f"      {str(k):<40}{c_:>6,}")
        say(f"    NOTE: targeted recollection is NOT performed here. These are "
            f"the exact counts a recollection would have to cover.")
        recol = {"paired_requests_with_unobservable_arm": sum(miss.values()),
                 "distinct_urls_to_recollect": len(urls_needed),
                 "top_domains": missurl.most_common(25),
                 "by_class_and_volatility": [[list(k), v]
                                             for k, v in miss.most_common()]}

    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post
    say(f"\n  implementation files unchanged: True")

    with open(HERE / "per_request_pair.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump({"withdrawn_claim":
               "gate-ON [3.0660,10.0869] and gate-OFF [6.4855,16.2001] were "
               "wrongly described as non-overlapping; they overlap on "
               "[6.4855,10.0869], and marginal bounds are the wrong comparison",
               "bound_type": "deterministic paired partial-identification "
                             "(worst-case) bound, NOT a confidence interval",
               "seed": SEED, "cells": dict(cells),
               "gate_on_metrics": m_on, "gate_off_metrics": m_off,
               "overall": ov, "delta_min_positive": resolved,
               "recollection_requirement": recol},
              open(HERE / "bounds_overall.json", "w"), indent=2)
    json.dump(boot, open(HERE / "bootstrap_results.json", "w"), indent=2)
    for fn, blk, lab in (("bounds_by_class.csv", bycl, "freshness_class"),
                         ("bounds_by_volatility.csv", byvo, "volatility_band")):
        with open(HERE / fn, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([lab, "n_paired", "changed_on_observed",
                        "changed_off_observed", "unobservable_on",
                        "unobservable_off", "n_jointly_observable",
                        "observed_delta_pp", "delta_min_pp", "delta_max_pp"])
            for k in sorted(blk, key=str):
                r = blk[k]
                w.writerow([k, r["n_paired"], r["changed_on_observed"],
                            r["changed_off_observed"], r["unobservable_on"],
                            r["unobservable_off"], r["n_jointly_observable"],
                            r["observed_delta_pp"], r["delta_min_pp"],
                            r["delta_max_pp"]])
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote per_request_pair.csv, bounds_overall.json, "
        "bounds_by_class.csv, bounds_by_volatility.csv, bootstrap_results.json")


if __name__ == "__main__":
    main()
