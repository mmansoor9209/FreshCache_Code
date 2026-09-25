#!/usr/bin/env python3
"""
Issue 1 -- prospective +24h temporal validation.

This script does exactly three things, in order:

  A. verifies that the T0 live-Web snapshot was collected AFTER the parameter
     freeze, and that every frozen parameter is still at its frozen value;
  B. determines whether >= 24 h have elapsed since the T0 collection;
  C. if and only if (B) is true, runs the frozen recollection over the SAME
     frozen URL set and evaluates.

If (B) is false it writes the exact earliest timestamp at which the +24h
collection may run and stops. It does not fabricate, extrapolate or
substitute a +24h result, and it never replaces a failed URL with a different
URL.

Read-only on everything outside validation_V3/final_three_issues/.
"""
from __future__ import annotations
import datetime as dt
import hashlib, json, os, pathlib, subprocess, sys

try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-final3")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
V3 = BASE.parent
ROOT = V3.parent
PROS = V3 / "remaining_critical_issues" / "05_prospective_temporal"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import experiment as exp   # noqa: E402

REQUIRED_HOURS = 24.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    now = dt.datetime.now(dt.timezone.utc)
    say("ISSUE 1 -- prospective +24h temporal validation")
    say(f"  now (UTC) {now.isoformat(timespec='seconds')}")

    man = json.load(open(PROS / "freeze_manifest.json", encoding="utf-8"))
    freeze = parse(man["freeze_time_utc"])
    say(f"\n  == A. freeze and T0 provenance ==")
    say(f"    freeze_time_utc            {freeze.isoformat(timespec='seconds')}")
    say(f"    freeze manifest sha256     {man['sha256_of_this_manifest']}")
    say(f"    frozen query set           {man['frozen_query_set']['n']:,} queries")
    say(f"    frozen URL set             {man['frozen_url_set']['n']:,} URLs")

    # frozen parameters must still be frozen
    fp = man["frozen_parameters"]
    drift = {}
    for k, v in fp.items():
        cur = getattr(exp, k, None)
        cur = dict(cur) if isinstance(cur, dict) else cur
        if cur != v:
            drift[k] = {"frozen": v, "current": cur}
    say(f"    frozen parameters unchanged: {not drift}"
        + (f"  DRIFT -> {drift}" if drift else ""))
    # frozen artifact checksums must still match
    bad = {f: h for f, h in man["frozen_artifact_sha256"].items()
           if not (ROOT / f).exists() or sha(ROOT / f) != h}
    say(f"    frozen artifact checksums unchanged: {not bad} "
        f"({len(man['frozen_artifact_sha256'])} files checked)"
        + (f"  MISMATCH -> {list(bad)}" if bad else ""))

    t0p = PROS / "snapshots" / "T0.jsonl"
    if not t0p.exists():
        say("\n    T0 SNAPSHOT MISSING -- issue 1 cannot proceed.")
        sys.exit(2)
    rows = [json.loads(l) for l in open(t0p, encoding="utf-8") if l.strip()]
    stamps = sorted(parse(r["fetched_at"]) for r in rows if r.get("fetched_at"))
    t0_start, t0_end = stamps[0], stamps[-1]
    say(f"    T0 file                    {t0p.relative_to(ROOT)}")
    say(f"    T0 sha256                  {sha(t0p)}")
    say(f"    T0 rows                    {len(rows):,}")
    say(f"    T0 fetch window            {t0_start.isoformat(timespec='seconds')}"
        f"  ..  {t0_end.isoformat(timespec='seconds')}")

    post_freeze = t0_start >= freeze
    say(f"\n    A. T0 collected AFTER the parameter freeze: {post_freeze}"
        f"   (T0 start - freeze = "
        f"{(t0_start - freeze).total_seconds():,.0f} s)")
    if not post_freeze:
        say("    T0 is NOT post-freeze; the prospective claim is not available.")
        sys.exit(2)

    # ---------------- B. has 24h elapsed? ----------------
    earliest_start = t0_start + dt.timedelta(hours=REQUIRED_HOURS)
    earliest_full = t0_end + dt.timedelta(hours=REQUIRED_HOURS)
    elapsed_h = (now - t0_end).total_seconds() / 3600.0
    ready = now >= earliest_full
    say(f"\n  == B. has {REQUIRED_HOURS:.0f} h elapsed since T0? ==")
    say(f"    elapsed since T0 END       {elapsed_h:.3f} h")
    say(f"    earliest +24h START        "
        f"{earliest_start.isoformat(timespec='seconds')}  "
        f"(24 h after the first T0 fetch)")
    say(f"    earliest +24h COMPLETE     "
        f"{earliest_full.isoformat(timespec='seconds')}  "
        f"(24 h after the last T0 fetch -- required so that EVERY URL has a "
        f"true 24 h gap)")
    say(f"    ready to collect now:      {ready}")

    summary = {
        "utc_now": now.isoformat(timespec="seconds"),
        "freeze_time_utc": man["freeze_time_utc"],
        "freeze_manifest_sha256": man["sha256_of_this_manifest"],
        "frozen_parameters_unchanged": not drift,
        "frozen_parameter_drift": drift,
        "frozen_artifact_checksums_unchanged": not bad,
        "t0_file": str(t0p.relative_to(ROOT)),
        "t0_sha256": sha(t0p),
        "t0_rows": len(rows),
        "t0_fetch_window_utc": [t0_start.isoformat(timespec="seconds"),
                                t0_end.isoformat(timespec="seconds")],
        "t0_is_post_freeze": post_freeze,
        "required_gap_hours": REQUIRED_HOURS,
        "elapsed_hours_since_t0_end": round(elapsed_h, 4),
        "earliest_plus24h_start_utc": earliest_start.isoformat(timespec="seconds"),
        "earliest_plus24h_complete_utc": earliest_full.isoformat(timespec="seconds"),
        "plus24h_collection_ran": False,
        "plus24h_evaluated": False,
        "verdict": None,
        "prospective_claim_allowed": False,
    }

    if not ready:
        wait_h = (earliest_full - now).total_seconds() / 3600.0
        say(f"\n  == C. NOT RUN ==")
        say(f"    The +24h Web state does not exist yet. No +24h number is")
        say(f"    produced, estimated or extrapolated.")
        say(f"    Wait {wait_h:.2f} h. The collection may run from "
            f"{earliest_full.isoformat(timespec='seconds')}:")
        say(f"      python {PROS.relative_to(ROOT)}/recollect.py "
            f"--round T0_plus_24h --workers 12")
        say(f"      python {PROS.relative_to(ROOT)}/compare_rounds.py "
            f"--a T0 --b T0_plus_24h --age-hours 24")
        say(f"      python {HERE.relative_to(ROOT)}/verify_and_collect.py  "
            f"# re-run this script to evaluate")
        say(f"\n    VERDICT: UNRESOLVED")
        summary["verdict"] = "UNRESOLVED"
        summary["hours_to_wait"] = round(wait_h, 4)
        summary["reason"] = (
            "Fewer than 24 h have elapsed since the T0 collection, so the +24h "
            "Web state does not exist. No prospective result is fabricated.")
        json.dump(summary, open(HERE / "summary.json", "w"), indent=2)
        (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
        say(f"  wrote summary.json, run.log")
        return

    # ---------------- C. collect and evaluate ----------------
    say(f"\n  == C. running the frozen +24h recollection ==")
    r = subprocess.run(
        [sys.executable, str(PROS / "recollect.py"), "--round", "T0_plus_24h",
         "--workers", "12"], capture_output=True, text=True)
    say(r.stdout[-4000:])
    if r.returncode != 0:
        say(f"    RECOLLECTION FAILED: {r.stderr[-2000:]}")
        sys.exit(2)
    summary["plus24h_collection_ran"] = True
    json.dump(summary, open(HERE / "summary.json", "w"), indent=2)
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  +24h snapshot collected. Run evaluate_24h.py next.")


if __name__ == "__main__":
    main()
