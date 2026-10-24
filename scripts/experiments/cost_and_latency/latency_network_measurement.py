#!/usr/bin/env python3
"""
Live measurement of the two network legs of the latency benchmark: search API and page fetch.

This is the only stage that touches the network, and it exists solely to time
those two operations.

  * The live responses are NOT used to recompute any accuracy, drift, freshness
    or other saved FreshCache result, and are NOT written into DATA_DIR/snapshots or
    any existing file. Everything lands in this directory.
  * A modest stratified subset is used to keep Serper spend small.
  * Search reuses web_collection.py's exact Serper call shape (same endpoint, same
    num=MAX_URLS_PER_QUERY, same 10s timeout); fetch reuses web_collection.py's HEADERS
    and 20s timeout, matching how the corpus was originally collected.

Usage:
  python latency_network_measurement.py --searches 40 --fetches 60          # live
  python latency_network_measurement.py --dry-run                            # no network
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, json, os, pathlib, random, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import defaultdict

HERE = ((OUTPUT_ROOT / "latency_e2e"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import requests                       # noqa: E402
import urllib3                        # noqa: E402
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import tiered_cache_model as exp              # noqa: E402
import web_collection                        # noqa: E402  (HEADERS / timeouts only)

CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]
SEED = 42
SERPER_URL = "https://google.serper.dev/search"


def key():
    if os.environ.get("SERPER_API_KEY"):
        return os.environ["SERPER_API_KEY"].strip()
    for line in open(ROOT / ".env", encoding="utf-8"):
        line = line.strip()
        if line.startswith("SERPER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("SERPER_API_KEY not found in .env")


def stats(v):
    if not v:
        return None
    s = sorted(v)
    p = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
    return {"n": len(s), "mean_ms": sum(s) / len(s), "p50_ms": p(0.50),
            "p90_ms": p(0.90), "p95_ms": p(0.95), "min_ms": s[0], "max_ms": s[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--searches", type=int, default=40)
    ap.add_argument("--fetches", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(SEED)
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)

    by_q = defaultdict(list)
    for r in records:
        by_q[r["freshness_class"]].append(r)
    per_q = max(1, a.searches // len(CLASSES))
    qs = []
    for c in CLASSES:
        pool = sorted(by_q[c], key=lambda r: r["query_id"])
        rng.shuffle(pool)
        qs.extend(pool[:per_q])
    qs = qs[:a.searches]

    by_u = defaultdict(list)
    seen = set()
    for m in manifest:
        if m.get("run_id") == "run_00" and m.get("snapshot_available") \
                and m.get("url") and m["url_hash"] not in seen:
            seen.add(m["url_hash"])
            by_u[m.get("freshness_class", "")].append(m)
    per_u = max(1, a.fetches // len(CLASSES))
    us = []
    for c in CLASSES:
        pool = sorted(by_u.get(c, []), key=lambda m: m["url_hash"])
        rng.shuffle(pool)
        us.extend(pool[:per_u])
    us = us[:a.fetches]

    print(f"  sample: {len(qs)} search queries, {len(us)} URLs to fetch "
          f"(stratified by freshness class, seed {SEED})", flush=True)
    print(f"  search : {SERPER_URL}, num={web_collection.MAX_URLS_PER_QUERY}, "
          f"timeout 10s   [web_collection.py's call shape]", flush=True)
    print(f"  fetch  : web_collection.py HEADERS, timeout {web_collection.FETCH_TIMEOUT}s",
          flush=True)
    if a.dry_run:
        print("  --dry-run: no network call issued.", flush=True)
        for c in CLASSES:
            print(f"    {c:<11}{sum(1 for r in qs if r['freshness_class']==c):>4}"
                  f" queries {sum(1 for m in us if m.get('freshness_class')==c):>4}"
                  f" urls", flush=True)
        return

    # ---------------- live search ----------------
    K = key()
    hdr = {"X-API-KEY": K, "Content-Type": "application/json"}
    srows = []
    print("\n  [search] live Serper calls ...", flush=True)
    for i, r in enumerate(qs):
        t0 = time.perf_counter()
        ok, n_org, err = False, 0, ""
        try:
            resp = requests.post(SERPER_URL, headers=hdr,
                                 json={"q": r["query"],
                                       "num": web_collection.MAX_URLS_PER_QUERY},
                                 timeout=10)
            ms = (time.perf_counter() - t0) * 1000
            ok = resp.status_code == 200
            if ok:
                n_org = len(resp.json().get("organic", []))
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            err = f"{type(e).__name__}: {e}"[:120]
        srows.append({"query_id": r["query_id"], "fc": r["freshness_class"],
                      "ms": round(ms, 2), "ok": ok, "n_organic": n_org,
                      "error": err})
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(qs)}", flush=True)
        time.sleep(0.2)

    # ---------------- live fetch ----------------
    frows = []
    print("\n  [fetch] live cold GETs ...", flush=True)
    for i, m in enumerate(us):
        t0 = time.perf_counter()
        ok, code, nbytes, err = False, 0, 0, ""
        try:
            resp = requests.get(m["url"], headers=web_collection.HEADERS,
                                timeout=web_collection.FETCH_TIMEOUT, verify=False)
            ms = (time.perf_counter() - t0) * 1000
            code, ok = resp.status_code, resp.status_code == 200
            nbytes = len(resp.content)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            err = f"{type(e).__name__}: {e}"[:120]
        frows.append({"url_hash": m["url_hash"], "domain": m.get("domain", ""),
                      "fc": m.get("freshness_class", ""), "ms": round(ms, 2),
                      "ok": ok, "status": code, "bytes": nbytes, "error": err})
        if (i + 1) % 15 == 0:
            print(f"    {i+1}/{len(us)}", flush=True)
        time.sleep(0.3)

    with open(HERE / "stage_d_live_raw.jsonl", "w", encoding="utf-8") as fh:
        for r in srows:
            fh.write(json.dumps({"kind": "search", **r}) + "\n")
        for r in frows:
            fh.write(json.dumps({"kind": "fetch", **r}) + "\n")

    sok = [r["ms"] for r in srows if r["ok"]]
    fok = [r["ms"] for r in frows if r["ok"]]
    out = {"seed": SEED, "n_search_attempted": len(srows),
           "n_search_ok": len(sok), "n_fetch_attempted": len(frows),
           "n_fetch_ok": len(fok),
           "search_api": stats(sok), "web_fetch": stats(fok),
           "search_error_rate_pct": 100 * (len(srows) - len(sok)) / max(len(srows), 1),
           "fetch_error_rate_pct": 100 * (len(frows) - len(fok)) / max(len(frows), 1),
           "fetch_by_class": {c: stats([r["ms"] for r in frows
                                        if r["ok"] and r["fc"] == c])
                              for c in CLASSES},
           "assumed_search_ms": exp.LATENCY["search_api"],
           "assumed_fetch_ms": exp.LATENCY["web_fetch"],
           "note": ("live timings for the latency benchmark only; not used to "
                    "recompute any accuracy/drift/freshness result and not "
                    "written into data/")}
    json.dump(out, open(HERE / "stage_d_live_timings.json", "w"), indent=2)
    s, f = out["search_api"], out["web_fetch"]
    print(f"\n  search_api  n={s['n']}  mean {s['mean_ms']:.1f}  p50 {s['p50_ms']:.1f}"
          f"  p95 {s['p95_ms']:.1f} ms   (assumed {exp.LATENCY['search_api']})",
          flush=True)
    print(f"  web_fetch   n={f['n']}  mean {f['mean_ms']:.1f}  p50 {f['p50_ms']:.1f}"
          f"  p95 {f['p95_ms']:.1f} ms   (assumed {exp.LATENCY['web_fetch']})",
          flush=True)
    print("  wrote stage_d_live_raw.jsonl, stage_d_live_timings.json", flush=True)


if __name__ == "__main__":
    main()
