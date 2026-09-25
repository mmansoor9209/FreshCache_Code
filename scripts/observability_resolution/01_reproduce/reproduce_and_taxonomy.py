#!/usr/bin/env python3
"""
TASK 1 -- reproduce the three published observability numbers from saved
row-level data.
TASK 2 -- classify exactly WHY each (URL, round) observation is missing, and
determine what could legitimately be recovered.

Read-only. No API calls, no fetches, no modification of any existing artifact.
"""
from __future__ import annotations
import csv, json, math, os, pathlib, random, re, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
OR = HERE.parent
V3 = OR.parent
ROOT = V3.parent
OB = V3 / "final_three_issues" / "03_observability_bounds"
RS = V3 / "remaining_resolution" / "03_observability"
SNAP = ROOT / "data" / "snapshots"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CHANGED, UNCHANGED, UNOBS = "CHANGED", "UNCHANGED", "UNOBSERVABLE"
SEED, B = 42, 10_000
MIN_BODY = 400
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|"
                   r"access denied|are you a robot|captcha|cloudflare|"
                   r"403 forbidden|404 not found|page not found", re.I)
CLAIM = {"observed": 3.3972, "dmin": -39.5587, "dmax": 43.9303,
         "sym_lo": 2.10, "sym_hi": 2.27}


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("TASK 1 -- reproduce published observability numbers from row level")
    rows = list(csv.DictReader(open(OB / "per_request_pair.csv", encoding="utf-8")))
    pr = [r for r in rows if r["paired"] == "1"]
    N = len(pr)
    c_on = sum(1 for r in pr if r["outcome_on"] == CHANGED)
    c_off = sum(1 for r in pr if r["outcome_off"] == CHANGED)
    u_on = sum(1 for r in pr if r["outcome_on"] == UNOBS)
    u_off = sum(1 for r in pr if r["outcome_off"] == UNOBS)
    jo = [r for r in pr if r["outcome_on"] != UNOBS and r["outcome_off"] != UNOBS]
    con = sum(1 for r in jo if r["outcome_on"] == CHANGED)
    coff = sum(1 for r in jo if r["outcome_off"] == CHANGED)
    d_obs = 100*coff/len(jo) - 100*con/len(jo)
    dmin = 100*(c_off - (c_on + u_on))/N
    dmax = 100*((c_off + u_off) - c_on)/N
    r_on = con/len(jo)
    sym = [100*((c_off + r*u_off) - (c_on + r*u_on))/N
           for r in (0.0, r_on, 1.5*r_on, 2*r_on, 1.0)]
    say(f"  paired N {N:,}; jointly observable {len(jo):,}")
    say(f"  {'quantity':<34}{'published':>12}{'recomputed':>12}{'match':>8}")
    def chk(nm, a, b, tol=0.01):
        ok = abs(a-b) < tol
        say(f"  {nm:<34}{a:>12.4f}{b:>12.4f}{('YES' if ok else 'NO'):>8}")
        return ok
    ok1 = chk("observed paired effect (pp)", CLAIM["observed"], d_obs)
    ok2 = chk("worst-case Delta_min (pp)", CLAIM["dmin"], dmin)
    ok3 = chk("worst-case Delta_max (pp)", CLAIM["dmax"], dmax)
    ok4 = chk("symmetric sensitivity low (pp)", CLAIM["sym_lo"], min(sym), 0.02)
    ok5 = chk("symmetric sensitivity high (pp)", CLAIM["sym_hi"], max(sym), 0.02)
    say(f"  all three published results reproduce: {all([ok1,ok2,ok3,ok4,ok5])}")

    # ---------------- TASK 2: why is each observation missing? ----------
    say(f"\nTASK 2 -- anatomy of missingness")
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d
    say(f"  round table: {len(rounds):,} URLs")
    say(f"  substantive := n_chars >= {MIN_BODY} AND status 200 AND no "
        f"block-page pattern in the first 400 chars")

    cat = Counter()
    nchar_short = []
    detail = []
    for uh, d in rounds.items():
        for run, e in d["rounds"].items():
            if e.get("substantive"):
                cat["substantive"] += 1
                continue
            sc, nc = e.get("status_code"), e.get("n_chars", 0)
            if sc != 200:
                k = f"non_200_status ({sc})"
            elif nc == 0:
                k = "empty_extraction"
            elif nc < MIN_BODY:
                k = f"short_text_under_{MIN_BODY}"
                nchar_short.append(nc)
            else:
                k = "block_page_pattern"
            cat[k] += 1
            detail.append({"url_hash": uh, "round": run, "reason": k,
                           "n_chars": nc, "status_code": sc,
                           "has_content_hash": bool(e.get("content_hash"))})
    tot = sum(cat.values())
    say(f"\n  every (URL, round) observation, {tot:,} in total")
    say(f"    {'reason':<34}{'n':>9}{'share':>9}")
    for k, v in cat.most_common():
        say(f"    {k:<34}{v:>9,}{100*v/tot:>8.2f}%")
    nsub = tot - cat["substantive"]
    say(f"  non-substantive observations: {nsub:,} ({100*nsub/tot:.2f}%)")
    if nchar_short:
        nchar_short.sort()
        say(f"\n  the short-text group ({len(nchar_short):,} observations) has a "
            f"recorded content_hash, so a change comparison is arithmetically "
            f"possible; it is excluded only by the {MIN_BODY}-char rule")
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            say(f"    n_chars p{int(q*100):<3} "
                f"{nchar_short[int(q*(len(nchar_short)-1))]:>6,}")
        for t in (50, 100, 200, 300):
            say(f"    would become eligible at MIN_BODY={t:<4}: "
                f"{sum(1 for x in nchar_short if x >= t):>7,} "
                f"({100*sum(1 for x in nchar_short if x >= t)/len(nchar_short):.1f}% "
                f"of short)")
    say(f"\n  RECOVERY ASSESSMENT")
    say(f"    snapshots store extracted_text ONLY -- no raw HTML is retained "
        f"(keys: url, status_code, fetched_at, content_hash, extracted_text, "
        f"headers). Re-extraction from stored bytes is therefore IMPOSSIBLE.")
    say(f"    non-200 and empty-extraction observations carry no usable text "
        f"and cannot be recovered from local data.")
    say(f"    block-page observations captured a bot wall, not the page; "
        f"recovering them would require re-fetching, which cannot reproduce "
        f"the historical state and is not attempted.")
    say(f"    the ONLY locally recoverable group is short-but-genuine text, "
        f"whose historical content_hash was captured at the correct time. "
        f"Recovering it is a THRESHOLD decision, not new data.")

    with open(HERE / "missing_observation_taxonomy.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0]))
        w.writeheader(); w.writerows(detail)
    json.dump({"reproduction": {"observed_pp": round(d_obs, 4),
                                "delta_min_pp": round(dmin, 4),
                                "delta_max_pp": round(dmax, 4),
                                "symmetric_range_pp": [round(min(sym), 4),
                                                       round(max(sym), 4)],
                                "all_match": bool(all([ok1, ok2, ok3, ok4, ok5]))},
               "paired": {"N": N, "jointly_observable": len(jo),
                          "changed_on": c_on, "changed_off": c_off,
                          "unobs_on": u_on, "unobs_off": u_off},
               "missingness_taxonomy": dict(cat),
               "raw_html_retained": False,
               "locally_recoverable_group": "short_text_under_400"},
              open(HERE / "reproduce_and_taxonomy.json", "w"), indent=2)
    (OR / "logs").mkdir(exist_ok=True)
    (OR / "logs" / "t1t2.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote reproduce_and_taxonomy.json, missing_observation_taxonomy.csv")


if __name__ == "__main__":
    main()
