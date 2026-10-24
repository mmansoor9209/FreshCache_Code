#!/usr/bin/env python3
"""
scripts/data/version_timeline.py — exact per-URL version timeline across all five
snapshot rounds, plus a body-observability flag.

WHAT THIS PROVIDES
  A Web-state model for arbitrary replay times.
     Snapshots exist at run_00, +1h, +12h, +24h and +7d. This builds, for every
     URL, the observed content hash at each of those five times. A replay can
     then answer exactly: what version was live at time t, and did the version
     cached at t_c differ from the version live at t_r. That replaces the
     cumulative-from-run_00 approximation, which is only correct when the cached
     version is the baseline one.

  Body observability (title-only extractions are not counted as unchanged).
     Change detection hashes extracted text. If extraction yielded only a page
     title, the hash is stable no matter what the body did, so the URL is
     UNOBSERVABLE rather than unchanged. This marks each URL body-observable or
     not, so every method can be re-scored on a common observable set.

  Raw HTML was not retained, so re-extraction is impossible. Observability is
  therefore defined on what was stored: a URL is body-observable when its
  extracted text is substantive (>= MIN_BODY chars after stripping boilerplate
  markers) in the baseline AND in at least one rerun, so a body change could in
  principle have been seen.

SAFETY  Read-only on data/. Writes only outputs/mixed_age/.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os; import json, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter, defaultdict

ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
DATA, OUT = DATA_ROOT, ((OUTPUT_ROOT / "mixed_age"))
RUNS = ["run_00", "rerun_1h", "rerun_12h", "rerun_24h", "rerun_7d"]
RUN_AGE = {"run_00": 0.0, "rerun_1h": 3_600.0, "rerun_12h": 43_200.0,
           "rerun_24h": 86_400.0, "rerun_7d": 604_800.0}
MIN_BODY = 400            # chars of extracted text for a body change to be visible
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|"
                   r"access denied|are you a robot|captcha|cloudflare|"
                   r"403 forbidden|404 not found|page not found", re.I)


def substantive(t: str) -> bool:
    if not t or len(t.strip()) < MIN_BODY:
        return False
    return not BLOCK.search(t[:400])


def main():
    tl = defaultdict(dict)       # url_hash -> run -> {hash, chars, substantive}
    for run in RUNS:
        d = DATA/"snapshots"/run
        if not d.exists():
            print(f"  {run}: MISSING"); continue
        n = 0
        for f in d.glob("*.json"):
            try:
                s = json.load(open(f))
            except Exception:
                continue
            t = s.get("extracted_text") or ""
            tl[f.stem][run] = {"hash": s.get("content_hash"),
                               "chars": len(t), "sub": substantive(t)}
            n += 1
        print(f"  {run:<10} {n:>7,} snapshots")

    rows = []
    for uh, per in tl.items():
        base = per.get("run_00")
        reruns = [r for r in RUNS[1:] if r in per]
        observable = bool(base and base["sub"] and
                          any(per[r]["sub"] for r in reruns))
        # exact change relative to the baseline, per window
        chg = {r: (bool(base and per[r]["hash"] != base["hash"]))
               for r in reruns} if base else {}
        rows.append({"url_hash": uh,
                     "runs": {r: {"chars": per[r]["chars"], "sub": per[r]["sub"]}
                              for r in per},
                     "hashes": {r: per[r]["hash"] for r in per},
                     "observable": observable,
                     "changed_vs_baseline": chg,
                     "n_runs": len(per)})

    with open(OUT/"version_timeline.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")

    n = len(rows)
    obs = sum(1 for r in rows if r["observable"])
    print(f"\n{'='*84}\n  OBSERVABILITY\n{'='*84}")
    print(f"  URLs with any snapshot                {n:>8,}")
    print(f"  body-observable (>= {MIN_BODY} chars in run_00 and >=1 rerun)"
          f"   {obs:>8,}  ({100*obs/n:.1f}%)")
    print(f"  UNOBSERVABLE                          {n-obs:>8,}  "
          f"({100*(n-obs)/n:.1f}%)  <- change is undetectable by construction")

    print(f"\n{'='*84}\n  CHANGE RATES: ALL URLs vs BODY-OBSERVABLE ONLY\n{'='*84}")
    print(f"  {'window':<12}{'all: chg/tot':>20}{'rate':>8}"
          f"{'observable: chg/tot':>24}{'rate':>8}{'lift':>8}")
    summary = {}
    for run in RUNS[1:]:
        A = [r for r in rows if run in r["changed_vs_baseline"]]
        O = [r for r in A if r["observable"]]
        ca = sum(1 for r in A if r["changed_vs_baseline"][run])
        co = sum(1 for r in O if r["changed_vs_baseline"][run])
        ra = ca/len(A) if A else 0.0
        ro = co/len(O) if O else 0.0
        summary[run] = {"all_changed": ca, "all_total": len(A), "all_rate": round(ra, 5),
                        "obs_changed": co, "obs_total": len(O), "obs_rate": round(ro, 5)}
        print(f"  {run:<12}{f'{ca:,}/{len(A):,}':>20}{100*ra:>7.2f}%"
              f"{f'{co:,}/{len(O):,}':>24}{100*ro:>7.2f}%{ro/ra if ra else 0:>7.2f}x")

    # exact pairwise transitions, needed for the arbitrary-time web-state model
    print(f"\n{'='*84}\n  EXACT PAIRWISE CHANGE (version at A vs version at B)\n{'='*84}")
    print(f"  {'from -> to':<26}{'n':>9}{'changed':>10}{'rate':>9}")
    pair = {}
    for i, a in enumerate(RUNS):
        for b in RUNS[i+1:]:
            both = [r for r in rows if a in r["hashes"] and b in r["hashes"]]
            c = sum(1 for r in both if r["hashes"][a] != r["hashes"][b])
            pair[f"{a}->{b}"] = {"n": len(both), "changed": c,
                                 "rate": round(c/len(both), 5) if both else 0}
            print(f"  {a+' -> '+b:<26}{len(both):>9,}{c:>10,}"
                  f"{100*c/max(len(both),1):>8.2f}%")

    json.dump({"min_body_chars": MIN_BODY, "n_urls": n, "n_observable": obs,
               "observable_pct": round(100*obs/n, 2),
               "window_change_rates": summary, "pairwise": pair,
               "run_age_seconds": RUN_AGE},
              open(OUT/"version_timeline_summary.json", "w"), indent=2)
    print(f"\n  wrote {OUT/'version_timeline.jsonl'}")
    print(f"  wrote {OUT/'version_timeline_summary.json'}")


if __name__ == "__main__":
    main()
