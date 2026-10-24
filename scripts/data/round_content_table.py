#!/usr/bin/env python3
"""
scripts/data/round_content_table.py — per-URL, per-round content table.

The transition manifest covers run_00 -> W. The mixed-age replay needs
arbitrary round pairs (an entry cached at one round and read at another), so it
needs the raw per-round facts. Same scan, same observability rule, written once
and set read-only.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import hashlib, json, os, pathlib, re, stat; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()

ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
OUT = ((DATA_ROOT / "derived"))
SNAP = (DATA_ROOT / "snapshots")
RUNS = ["run_00", "rerun_1h", "rerun_12h", "rerun_24h", "rerun_48h", "rerun_7d"]
MIN_BODY = 400
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|"
                   r"access denied|are you a robot|captcha|cloudflare|"
                   r"403 forbidden|404 not found|page not found", re.I)
TABLE = OUT/"round_table.jsonl"


def substantive(t):
    if not t or len(t.strip()) < MIN_BODY:
        return False
    return not BLOCK.search(t[:400])


def main():
    if TABLE.exists():
        print(f"  {TABLE.name} exists and is immutable; refusing to overwrite.")
        return
    per = {}
    for run in RUNS:
        d = SNAP/run
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            try:
                s = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            t = s.get("extracted_text") or ""
            u = s.get("url_hash") or p.stem
            per.setdefault(u, {"url_hash": u, "url": s.get("url", ""), "rounds": {}})
            per[u]["rounds"][run] = {
                "content_hash": s.get("content_hash"),
                "text_hash": hashlib.md5(t.encode("utf-8")).hexdigest(),
                "n_chars": len(t), "status_code": s.get("status_code"),
                "fetched_at": s.get("fetched_at"),
                "substantive": bool(substantive(t) and s.get("status_code") == 200)}
    with open(TABLE, "w", encoding="utf-8") as f:
        for v in per.values():
            f.write(json.dumps(v, ensure_ascii=False)+"\n")
    os.chmod(TABLE, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"  wrote {TABLE.name}: {len(per):,} URLs x up to {len(RUNS)} rounds (read-only)")
    for run in RUNS:
        n = sum(1 for v in per.values() if run in v["rounds"])
        s = sum(1 for v in per.values() if v["rounds"].get(run, {}).get("substantive"))
        print(f"    {run:<12}{n:>8,} snapshots{s:>9,} substantive")


if __name__ == "__main__":
    main()
