#!/usr/bin/env python3
"""
TASK 2 -- collect the real +24h Web snapshot for the FROZEN URL set.

HARD TIME GATE: refuses to run before max(T0.fetched_at) + 24 h, so that every
URL is guaranteed a real elapsed interval of at least 24 hours regardless of
the order this collection happens to visit them in.

FETCH PATH. collect.fetch_url() returns None for every failure mode, which
discards the HTTP status and the reason -- both required here. fetch_verbose()
below is a faithful transcription of collect.fetch_url with the SAME
requests.get arguments, the SAME accepted status codes, the SAME content-type
rule, the SAME encoding handling, and the SAME extract_text / content_hash
calls, so a successful fetch produces a byte-identical content hash. The only
difference is that failures return a reason instead of None. assert_transcription()
checks the constants it depends on against the live collect module before any
URL is fetched.

Per URL it records: url id, T0 timestamp, T+24h timestamp, actual elapsed
seconds, HTTP status, observable/unobservable status, T0 content hash, new
content hash, observed change, and the failure reason if any.

Failures are KEPT in the dataset. No URL is replaced, dropped or invented, and
no retrospective snapshot is substituted.
"""
from __future__ import annotations
import argparse, datetime as dt, gzip, hashlib, json, os, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
F3 = HERE.parent
V3 = F3.parent
ROOT = V3.parent
PROS = V3 / "remaining_critical_issues" / "05_prospective_temporal"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import requests                      # noqa: E402
import urllib3                       # noqa: E402
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import collect as C                  # noqa: E402

ROUND = "T0_plus_24h"
REQUIRED_HOURS = 24.0
OUT = HERE / "snapshots"
BODIES = HERE / "bodies"
MIN_SUBSTANTIVE = 200     # the T0 rule, unchanged


def parse(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def assert_transcription():
    """The transcription depends on these; fail loudly if collect.py changed."""
    assert hasattr(C, "fetch_url") and hasattr(C, "extract_text")
    assert hasattr(C, "content_hash") and hasattr(C, "normalize_for_hashing")
    assert hasattr(C, "HEADERS") and hasattr(C, "FETCH_TIMEOUT")
    src = __import__("inspect").getsource(C.fetch_url)
    for token in ("(200, 203, 206)", "text/html", "text/plain",
                  "allow_redirects=True", "verify=False",
                  "extract_text(resp.text, url)", "content_hash(hash_text)"):
        assert token in src, f"collect.fetch_url no longer contains {token!r}; " \
                             f"the transcription is stale"
    return hashlib.sha256(src.encode()).hexdigest()


def fetch_verbose(url):
    """collect.fetch_url, transcribed, returning the reason instead of None."""
    try:
        resp = requests.get(url, headers=C.HEADERS, timeout=C.FETCH_TIMEOUT,
                            allow_redirects=True, verify=False)
    except requests.exceptions.Timeout:
        return {"ok": False, "status": None, "reason": "TIMEOUT"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "status": None, "reason": "CONNECTION_ERROR"}
    except Exception as e:
        return {"ok": False, "status": None, "reason": f"ERROR_{type(e).__name__}"}
    st = resp.status_code
    if st not in (200, 203, 206):
        return {"ok": False, "status": st, "reason": f"HTTP_{st}"}
    ct = resp.headers.get("Content-Type", "")
    if "text/html" not in ct and "text/plain" not in ct:
        return {"ok": False, "status": st, "reason": f"NON_HTML:{ct[:40]}"}
    if resp.encoding and resp.encoding.upper() not in ("UTF-8", "UTF8"):
        try:
            resp.content.decode("utf-8", errors="replace")
        except Exception:
            pass
    display_text, hash_text = C.extract_text(resp.text, url)
    if not display_text:
        return {"ok": False, "status": st, "reason": "EMPTY_EXTRACTION"}
    h = {k.lower(): v for k, v in resp.headers.items()}
    return {"ok": True, "status": st, "reason": None,
            "content_hash": C.content_hash(hash_text),
            "content_length": len(display_text),
            "final_url": resp.url, "etag": h.get("etag"),
            "last_modified": h.get("last-modified"),
            "content_type": h.get("content-type", ""),
            "server": h.get("server", ""),
            "text": display_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-bodies", action="store_true", default=True)
    ap.add_argument("--force-after-gate", action="store_true",
                    help="only for resuming AFTER the gate has already opened")
    a = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    t0rows = [json.loads(l) for l in
              open(PROS / "snapshots" / "T0.jsonl", encoding="utf-8") if l.strip()]
    T0 = {r["url_hash"]: r for r in t0rows}
    stamps = sorted(parse(r["fetched_at"]) for r in t0rows if r.get("fetched_at"))
    t0_end = stamps[-1]
    earliest = t0_end + dt.timedelta(hours=REQUIRED_HOURS)
    print(f"  now {now.isoformat(timespec='seconds')}")
    print(f"  last T0 fetch {t0_end.isoformat(timespec='seconds')}")
    print(f"  earliest legal start {earliest.isoformat(timespec='seconds')}")
    if now < earliest:
        wait = (earliest - now).total_seconds()
        sys.exit(f"REFUSING TO COLLECT: the +24h window is not open. "
                 f"Wait {wait/3600:.2f} h ({wait:.0f} s). Nothing was fetched, "
                 f"estimated or extrapolated.")
    print(f"  TIME GATE OPEN")

    src_sha = assert_transcription()
    print(f"  collect.fetch_url source sha256 {src_sha[:16]}… "
          f"(transcription tokens verified)")

    urls = [json.loads(l) for l in
            open(PROS / "frozen_url_set.jsonl", encoding="utf-8") if l.strip()]
    assert len(urls) == 8120, f"frozen URL set is {len(urls)}, expected 8120"
    OUT.mkdir(exist_ok=True)
    if a.save_bodies:
        BODIES.mkdir(exist_ok=True)
    out = OUT / f"{ROUND}.jsonl"
    done = ({json.loads(l)["url_hash"] for l in open(out, encoding="utf-8")}
            if out.exists() else set())
    todo = [u for u in urls if u["url_hash"] not in done]
    print(f"  frozen URL set {len(urls):,}; already collected {len(done):,}; "
          f"to fetch {len(todo):,}")

    def one(rec):
        uh, url = rec["url_hash"], rec.get("url")
        t0 = T0.get(uh, {})
        t0_ts = t0.get("fetched_at")
        t0_hash = t0.get("content_hash")
        t0_subst = bool(t0.get("substantive"))
        if not url:
            return {"url_hash": uh, "round": ROUND, "url": None,
                    "freshness_class": rec.get("freshness_class"),
                    "t0_fetched_at": t0_ts, "t24_fetched_at": None,
                    "elapsed_seconds": None, "elapsed_hours": None,
                    "http_status": None, "snapshot_available": False,
                    "substantive": False, "t0_content_hash": t0_hash,
                    "t24_content_hash": None, "t0_substantive": t0_subst,
                    "observable": False, "content_changed": None,
                    "failure_reason": "NO_URL_IN_FROZEN_SET"}
        r = fetch_verbose(url)
        ts = dt.datetime.now(dt.timezone.utc)
        el = ((ts - parse(t0_ts)).total_seconds() if t0_ts else None)
        row = {"url_hash": uh, "round": ROUND, "url": url,
               "freshness_class": rec.get("freshness_class"),
               "t0_fetched_at": t0_ts,
               "t24_fetched_at": ts.isoformat(timespec="seconds"),
               "elapsed_seconds": (round(el, 1) if el is not None else None),
               "elapsed_hours": (round(el / 3600.0, 4) if el is not None else None),
               "http_status": r.get("status"),
               "snapshot_available": bool(r["ok"]),
               "t0_content_hash": t0_hash, "t0_substantive": t0_subst,
               "failure_reason": r.get("reason")}
        if r["ok"]:
            subst = r["content_length"] >= MIN_SUBSTANTIVE
            row.update({"substantive": subst,
                        "t24_content_hash": r["content_hash"],
                        "content_length": r["content_length"],
                        "final_url": r["final_url"], "etag": r["etag"],
                        "last_modified": r["last_modified"],
                        "content_type": r["content_type"], "server": r["server"]})
            if a.save_bodies and subst:
                with gzip.open(BODIES / f"{uh}.txt.gz", "wt",
                               encoding="utf-8") as fh:
                    fh.write(r["text"])
        else:
            row.update({"substantive": False, "t24_content_hash": None,
                        "content_length": 0})
        # observable ONLY when both rounds produced a substantive body
        row["observable"] = bool(t0_subst and row["substantive"])
        row["content_changed"] = (None if not row["observable"]
                                  else (t0_hash != row["t24_content_hash"]))
        return row

    t0c = time.time()
    n = 0
    with open(out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one, todo):
            fh.write(json.dumps(r) + "\n")
            n += 1
            if n % 250 == 0:
                fh.flush()
                print(f"    {n:,}/{len(todo):,}  ({time.time()-t0c:.0f}s)",
                      flush=True)
    print(f"  wrote {out}")

    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    el = [r["elapsed_hours"] for r in rows if r["elapsed_hours"] is not None]
    obs = [r for r in rows if r["observable"]]
    chg = sum(1 for r in obs if r["content_changed"])
    from collections import Counter
    print(f"\n  rows {len(rows):,}")
    print(f"  min elapsed {min(el):.4f} h   max {max(el):.4f} h   "
          f"(>= 24 h for all: {min(el) >= 24.0})")
    print(f"  reached at T+24h {sum(1 for r in rows if r['snapshot_available']):,}")
    print(f"  substantive at T+24h {sum(1 for r in rows if r['substantive']):,}")
    print(f"  OBSERVABLE (substantive in BOTH rounds) {len(obs):,}")
    print(f"  observed changed {chg:,} = "
          f"{100*chg/max(1,len(obs)):.4f}% of observable")
    print(f"  failure reasons: "
          f"{dict(Counter(r['failure_reason'] for r in rows if r['failure_reason']).most_common(10))}")
    json.dump({"round": ROUND, "utc_started": now.isoformat(timespec="seconds"),
               "earliest_legal_start": earliest.isoformat(timespec="seconds"),
               "frozen_url_set": len(urls), "rows": len(rows),
               "min_elapsed_hours": min(el), "max_elapsed_hours": max(el),
               "all_at_least_24h": bool(min(el) >= 24.0),
               "reached": sum(1 for r in rows if r["snapshot_available"]),
               "substantive": sum(1 for r in rows if r["substantive"]),
               "observable": len(obs), "changed": chg,
               "fetch_url_source_sha256": src_sha,
               "bodies_saved": bool(a.save_bodies)},
              open(HERE / "collection_summary.json", "w"), indent=2)
    print(f"  wrote collection_summary.json")


if __name__ == "__main__":
    main()
