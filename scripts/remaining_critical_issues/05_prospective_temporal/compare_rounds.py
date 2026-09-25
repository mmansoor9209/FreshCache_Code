#!/usr/bin/env python3
"""
Section 5 -- prospective evaluation, to be run ONCE a post-freeze snapshot PAIR
exists.

Compares two rounds collected after the freeze and reports, per freshness
class, the observed change rate and the drift a frozen FreshCache would have
incurred at that age. It refuses to run if either round is missing, and it
never refits a half-life, threshold or epsilon -- the freeze manifest's
parameters are asserted unchanged before anything is computed.

  python compare_rounds.py --a T0 --b T0_plus_24h --age-hours 24
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, pathlib, sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import experiment as exp   # noqa: E402

SNAP = HERE / "snapshots"


def load(rd):
    p = SNAP / f"{rd}.jsonl"
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist. Collect it first with "
                 f"collect_t0.py / recollect.py. Nothing is estimated in its "
                 f"absence.")
    return {json.loads(l)["url_hash"]: json.loads(l)
            for l in open(p, encoding="utf-8") if l.strip()}


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 3),
            round(100 * min(1.0, (c + m) / d), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--age-hours", type=float, required=True)
    a = ap.parse_args()

    man = json.load(open(HERE / "freeze_manifest.json", encoding="utf-8"))
    fp = man["frozen_parameters"]
    bad = {k: (getattr(exp, k, None), v) for k, v in fp.items()
           if k not in ("HALF_LIFE", "TIER_MULT", "FIXED_TTL")
           and getattr(exp, k, None) != v}
    for k in ("HALF_LIFE", "TIER_MULT", "FIXED_TTL"):
        if dict(getattr(exp, k)) != fp[k]:
            bad[k] = (dict(getattr(exp, k)), fp[k])
    if bad:
        sys.exit(f"ERROR: frozen parameters have changed since the freeze: "
                 f"{bad}. A prospective evaluation is only valid against the "
                 f"frozen configuration.")
    print(f"  frozen parameters unchanged since "
          f"{man['freeze_time_utc']}: True", flush=True)

    A, B = load(a.a), load(a.b)
    age = a.age_hours * 3600.0
    shared = [u for u in A if u in B
              and A[u].get("substantive") and B[u].get("substantive")]
    print(f"  {a.a}: {len(A):,} rows   {a.b}: {len(B):,} rows   "
          f"both substantive: {len(shared):,}")

    by = {}
    for u in shared:
        fc = A[u].get("freshness_class") or "?"
        ch = A[u].get("content_hash") != B[u].get("content_hash")
        d = by.setdefault(fc, [0, 0])
        d[0] += 1
        d[1] += int(ch)
    print(f"\n  {'class':<12}{'observed':>10}{'changed':>9}{'rate%':>9}"
          f"{'95% CI':>18}{'model P(stale)':>16}{'reuse?':>9}")
    out = {}
    for fc in sorted(by):
        n, k = by[fc]
        ps = exp.p_stale(fc, age, "content")
        reuse = ps <= exp.EPS_CONTENT
        out[fc] = {"observed": n, "changed": k,
                   "change_rate_pct": round(100 * k / n, 4),
                   "ci": wilson(k, n), "model_p_stale": round(ps, 6),
                   "frozen_gate_would_reuse": bool(reuse)}
        print(f"  {fc:<12}{n:>10,}{k:>9,}{100*k/n:>8.3f}%"
              f"{str(wilson(k, n)):>18}{ps:>16.4f}{str(reuse):>9}")
    tot_n = sum(v[0] for v in by.values())
    tot_k = sum(v[1] for v in by.values())
    print(f"  {'ALL':<12}{tot_n:>10,}{tot_k:>9,}{100*tot_k/max(1,tot_n):>8.3f}%"
          f"{str(wilson(tot_k, tot_n)):>18}")
    json.dump({"round_a": a.a, "round_b": a.b, "age_hours": a.age_hours,
               "freeze_time_utc": man["freeze_time_utc"],
               "parameters_refit_after_freeze": False,
               "observed": tot_n, "changed": tot_k,
               "change_rate_pct": round(100 * tot_k / max(1, tot_n), 4),
               "ci": wilson(tot_k, tot_n), "by_class": out},
              open(HERE / f"prospective_{a.a}_vs_{a.b}.json", "w"), indent=2)
    print(f"\n  wrote prospective_{a.a}_vs_{a.b}.json")


if __name__ == "__main__":
    main()
