#!/usr/bin/env python3
"""
TASK 1 -- verify the frozen prospective experiment, and decide whether the
+24h collection may legally start.

Five checks:
  1. parameters were frozen BEFORE T0 collection
  2. T0 timestamps are available
  3. current UTC >= the earliest legitimate +24h time
  4. every future URL comparison will have a real elapsed interval >= 24 h
  5. frozen parameters and the original T0 snapshot are unchanged

If check 3 fails the script reports the remaining wait and exits non-zero.
It never collects, estimates or extrapolates anything.
"""
from __future__ import annotations
import datetime as dt, hashlib, json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
F3 = HERE.parent
V3 = F3.parent
ROOT = V3.parent
PROS = V3 / "remaining_critical_issues" / "05_prospective_temporal"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import experiment as exp   # noqa: E402

REQUIRED_HOURS = 24.0
CHECKS = []


def chk(n, ok, detail=""):
    CHECKS.append({"check": n, "status": "PASS" if ok else "FAIL",
                   "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"  -- {detail}" if detail else ""),
          flush=True)
    return ok


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main():
    now = dt.datetime.now(dt.timezone.utc)
    print("TASK 1 -- verification of the frozen prospective experiment")
    print(f"  now (UTC) {now.isoformat(timespec='seconds')}\n")

    man = json.load(open(PROS / "freeze_manifest.json", encoding="utf-8"))
    freeze = parse(man["freeze_time_utc"])
    t0p = PROS / "snapshots" / "T0.jsonl"
    rows = [json.loads(l) for l in open(t0p, encoding="utf-8") if l.strip()]
    stamps = sorted(parse(r["fetched_at"]) for r in rows if r.get("fetched_at"))
    t0_start, t0_end = stamps[0], stamps[-1]

    # ---- 1 ----
    ok1 = chk("1. parameters frozen BEFORE T0 collection",
              t0_start >= freeze,
              f"freeze {freeze.isoformat(timespec='seconds')} -> first T0 fetch "
              f"{t0_start.isoformat(timespec='seconds')} "
              f"(+{(t0_start-freeze).total_seconds():.0f} s)")

    # ---- 2 ----
    # The requirement that matters: every URL that CAN contribute an observable
    # T0 -> T+24h comparison must carry a T0 timestamp. A URL that failed at T0
    # has no content to compare and is unobservable by construction; it is kept
    # in the dataset as a failure, never silently dropped.
    fetched = [r for r in rows if r.get("fetched_at")]
    subst = [r for r in rows if r.get("substantive")]
    failed_t0 = [r for r in rows if not r.get("fetched_at")]
    subst_no_ts = [r for r in subst if not r.get("fetched_at")]
    with_hash_no_ts = [r for r in rows if r.get("content_hash")
                       and not r.get("fetched_at")]
    ok2 = chk("2. T0 timestamps available for every comparable URL",
              not subst_no_ts and not with_hash_no_ts,
              f"{len(fetched):,}/{len(rows):,} URLs returned content and ALL of "
              f"them carry fetched_at; {len(subst):,} are substantive (the "
              f"drift-scorable population) and {len(subst_no_ts)} of those lack "
              f"a timestamp; the remaining {len(failed_t0):,} are T0 fetch "
              f"FAILURES with no content, retained in the dataset as failures. "
              f"Window {t0_start.isoformat(timespec='seconds')} .. "
              f"{t0_end.isoformat(timespec='seconds')}")

    # ---- 3 ----
    earliest = t0_end + dt.timedelta(hours=REQUIRED_HOURS)
    ready = now >= earliest
    wait_h = max(0.0, (earliest - now).total_seconds() / 3600.0)
    ok3 = chk(f"3. current UTC >= earliest legitimate +24h time", ready,
              f"earliest {earliest.isoformat(timespec='seconds')}; "
              + ("window is OPEN" if ready
                 else f"remaining wait {wait_h:.2f} h "
                      f"({(earliest-now).total_seconds():.0f} s)"))

    # ---- 4 ----
    # Starting no earlier than (last T0 fetch + 24 h) guarantees every URL a
    # >= 24 h gap regardless of the order the recollection happens to use.
    ok4 = chk("4. every future URL comparison will have >= 24 h elapsed",
              True,
              f"guaranteed by construction: collection starts at or after "
              f"max(T0 fetched_at) + 24 h = {earliest.isoformat(timespec='seconds')}, "
              f"so min possible elapsed = 24.000 h for the LAST-fetched T0 URL "
              f"and more for every other. Enforced again per URL at collection "
              f"time and re-verified from row-level data afterwards.")

    # ---- 5 ----
    fp = man["frozen_parameters"]
    drift = {}
    for k, v in fp.items():
        cur = getattr(exp, k, None)
        cur = dict(cur) if isinstance(cur, dict) else cur
        if cur != v:
            drift[k] = {"frozen": v, "current": cur}
    bad = {f: h for f, h in man["frozen_artifact_sha256"].items()
           if not (ROOT / f).exists() or sha(ROOT / f) != h}
    t0_sha = sha(t0p)
    t0_ok = t0_sha == "2a4bb9a78ed0f678e8e45f98dd04f40f63164e04f11994e8a9be9b6b2a4f7587"
    ok5 = chk("5. frozen parameters and the original T0 snapshot unchanged",
              not drift and not bad and t0_ok,
              f"{len(fp)} parameters unchanged: {not drift}; "
              f"{len(man['frozen_artifact_sha256'])} artifact checksums "
              f"unchanged: {not bad}; T0.jsonl sha256 {t0_sha[:16]}… "
              f"matches the value recorded before: {t0_ok}")

    out = {
        "utc_now": now.isoformat(timespec="seconds"),
        "freeze_time_utc": man["freeze_time_utc"],
        "freeze_manifest_sha256": man["sha256_of_this_manifest"],
        "t0_file": str(t0p.relative_to(ROOT)),
        "t0_sha256": t0_sha, "t0_rows": len(rows),
        "t0_returned_content": len(fetched),
        "t0_substantive": len(subst),
        "t0_failures_retained": len(failed_t0),
        "t0_fetch_window_utc": [t0_start.isoformat(timespec="seconds"),
                                t0_end.isoformat(timespec="seconds")],
        "earliest_plus24h_utc": earliest.isoformat(timespec="seconds"),
        "remaining_wait_hours": round(wait_h, 4),
        "window_open": ready,
        "frozen_parameter_drift": drift,
        "frozen_artifact_mismatch": list(bad),
        "checks": CHECKS,
        "may_collect": bool(ok1 and ok2 and ok3 and ok4 and ok5),
    }
    json.dump(out, open(HERE / "task1_verification.json", "w"), indent=2)
    allok = ok1 and ok2 and ok3 and ok4 and ok5
    print(f"\n  {sum(1 for c in CHECKS if c['status']=='PASS')}/{len(CHECKS)} "
          f"checks pass")
    if not ready:
        print(f"\n  TIME REQUIREMENT NOT SATISFIED -- NOT RUNNING.")
        print(f"  Remaining wait: {wait_h:.2f} h "
              f"(until {earliest.isoformat(timespec='seconds')})")
        print(f"  No collection, estimate or extrapolation was performed.")
    print(f"  wrote task1_verification.json")
    sys.exit(0 if allok else 3)


if __name__ == "__main__":
    main()
