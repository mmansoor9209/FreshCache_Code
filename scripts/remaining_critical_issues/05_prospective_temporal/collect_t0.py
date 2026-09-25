#!/usr/bin/env python3
"""
Section 5 -- live-Web T0 collection for the prospective study.

Re-collects the FROZEN URL set using collect.py's own fetch and extraction
path, so the content hashes are comparable with the existing rounds. Writes
ONLY into 05_prospective_temporal/snapshots/ and appends to a manifest in this
directory; data/ is never touched.

  python collect_t0.py --round T0            [--limit N] [--workers 8]
  python collect_t0.py --round T0_plus_24h   (same command, later)
  python collect_t0.py --round T0_plus_7d

Then compare rounds with compare_rounds.py.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import collect as C   # noqa: E402

OUT = HERE / "snapshots"


def one(rec):
    u = rec.get("url")
    if not u:
        return None
    try:
        r = C.fetch_url(u)
    except Exception as e:
        return {"url_hash": rec["url_hash"], "url": u, "error": str(e),
                "snapshot_available": False}
    if not r:
        return {"url_hash": rec["url_hash"], "url": u,
                "snapshot_available": False}
    text = r.get("extracted_text") or ""
    return {"url_hash": rec["url_hash"], "url": u,
            "freshness_class": rec.get("freshness_class"),
            "snapshot_available": bool(text),
            "substantive": len(text) >= 200,
            "content_hash": (C.content_hash(C.normalize_for_hashing(text))
                             if text else None),
            "content_length": len(text),
            "etag": r.get("etag"), "last_modified": r.get("last_modified"),
            "fetched_at": C.now_iso()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    recs = [json.loads(l) for l in
            open(HERE / "frozen_url_set.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    out = OUT / f"{a.round}.jsonl"
    done = ({json.loads(l)["url_hash"] for l in open(out, encoding="utf-8")}
            if out.exists() else set())
    todo = [r for r in recs if r["url_hash"] not in done]
    print(f"  round {a.round}: {len(done):,} done, {len(todo):,} to fetch",
          flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            if r:
                r["round"] = a.round
                fh.write(json.dumps(r) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(f"    {i:,}/{len(todo):,}  ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
