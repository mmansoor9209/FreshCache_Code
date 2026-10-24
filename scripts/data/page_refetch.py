"""Re-fetch the benchmark's web pages by URL and compare them with the stored content hashes.

The dataset (and data/sample) references pages only by URL, URL hash and content hash. This script downloads the pages of one collection
round of DATA_DIR (default run_00) with the fetcher and text extraction of web_collection.py (same request headers, timeout, extraction and
content hash), and reports for every URL whether the page is UNCHANGED, CHANGED, UNAVAILABLE (no usable HTML response) or NO_STORED_HASH
(the page had no snapshot in the original round). No search API key is needed and DATA_DIR is never written.

Outputs (default OUTPUT_DIR/page_refetch/):
  snapshots/<round>/<url_hash>.json   re-fetched snapshot, in the snapshot format of scripts/data/web_collection.py
  refetch_report.csv                  one row per URL: stored and new content hash, status, HTTP status, extracted characters, substantive
  refetch_summary.json                counts per status
With --build-data DIR, DIR becomes a complete data directory for the pipeline: every file of DATA_DIR plus the
re-fetched pages as snapshots/<round>/. Run any stage on it with DATA_DIR=DIR (scripts/run_sample_quickstart.sh does this with REFETCH=1).
Exact reproduction of the paper needs the original snapshots; re-fetched pages reflect the web at run time and may differ.

Usage: python scripts/data/page_refetch.py [--round run_00] [--limit N] [--delay S] [--out DIR] [--build-data DIR] [--skip-fetch] [--dry-run]
  --dry-run     list what would be fetched and where results would go; no network access, nothing written
  --skip-fetch  do not download; rebuild the report (and --build-data) from snapshots already in --out
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, collections, csv, datetime, json, os, pathlib, re, shutil, sys, time
REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
MIN_BODY = 400   # observability rule of scripts/data/round_content_table.py and snapshot_audit.py
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|access denied|are you a robot|captcha|cloudflare|"
                   r"403 forbidden|404 not found|page not found", re.I)
MARKER = "refetched_data.json"

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--round", default="run_00"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--delay", type=float, default=None)
ap.add_argument("--out", default=str(OUTPUT_ROOT / "page_refetch")); ap.add_argument("--build-data", default=None)
ap.add_argument("--skip-fetch", action="store_true"); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
OUT = pathlib.Path(a.out).resolve(); SNAP = OUT / "snapshots" / a.round

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def substantive(t, status):
    return bool(t) and len(t.strip()) >= MIN_BODY and not BLOCK.search(t[:400]) and status == 200

# ---- URLs of the round and their stored content hashes (manifest first, then the per-round table)
man = [r for r in jl(DATA_ROOT / "url_manifest.jsonl") if r.get("run_id") == a.round]
if not man: sys.exit(f"no url_manifest.jsonl rows for round {a.round} in {DATA_ROOT}")
URLS = {}
for r in man:
    u = URLS.setdefault(r["url_hash"], {"url_hash": r["url_hash"], "url": r["url"], "domain": r.get("domain", ""), "stored_content_hash": None})
    if r.get("snapshot_available") and r.get("content_hash"): u["stored_content_hash"] = r["content_hash"]
for r in jl(DATA_ROOT / "derived/round_table.jsonl"):
    u = URLS.get(r["url_hash"]); rd = r.get("rounds", {}).get(a.round) or {}
    if u is not None and u["stored_content_hash"] is None and rd.get("content_hash"): u["stored_content_hash"] = rd["content_hash"]
todo = sorted(URLS.values(), key=lambda u: u["url_hash"])
if a.limit: todo = todo[:a.limit]
print(f"{len(todo)} URLs of round {a.round} from {DATA_ROOT} ({sum(1 for u in todo if u['stored_content_hash'])} with a stored content hash)")
if a.build_data:
    BD = pathlib.Path(a.build_data).resolve()
    if BD == DATA_ROOT or DATA_ROOT in BD.parents or BD in DATA_ROOT.parents: sys.exit("--build-data must be outside DATA_DIR")
if a.dry_run:
    print(f"dry run: would write snapshots to {SNAP}, the report to {OUT}" + (f", and a data directory to {BD}" if a.build_data else "")); sys.exit(0)

# ---- fetch (same fetcher, extraction and content hash as the collection)
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))
import web_collection as WC   # noqa: E402
delay = WC.FETCH_DELAY if a.delay is None else a.delay
SNAP.mkdir(parents=True, exist_ok=True)
attempted = set()
if not a.skip_fetch:
    for i, u in enumerate(todo, 1):
        attempted.add(u["url_hash"]); print(f"  [{i}/{len(todo)}] {u['url'][:90]}", flush=True)
        r = WC.fetch_url(u["url"])
        if r is not None:
            r["run_id"] = a.round; r["url_hash"] = u["url_hash"]; r["refetched"] = True
            with open(SNAP / f"{u['url_hash']}.json", "w", encoding="utf-8") as f: json.dump(r, f, ensure_ascii=False, indent=2)
        elif (SNAP / f"{u['url_hash']}.json").exists():
            (SNAP / f"{u['url_hash']}.json").unlink()   # a page that is unavailable now must not keep an older re-fetch
        if i < len(todo): time.sleep(delay)

# ---- report
rows = []
for u in todo:
    p = SNAP / f"{u['url_hash']}.json"; s = json.load(open(p, encoding="utf-8")) if p.exists() else None
    if s is None: status = "UNAVAILABLE"
    elif not u["stored_content_hash"]: status = "NO_STORED_HASH"
    else: status = "UNCHANGED" if s.get("content_hash") == u["stored_content_hash"] else "CHANGED"
    t = (s or {}).get("extracted_text") or ""
    rows.append({"url_hash": u["url_hash"], "url": u["url"], "domain": u["domain"], "stored_content_hash": u["stored_content_hash"] or "",
                 "refetched_content_hash": (s or {}).get("content_hash", ""), "status": status, "http_status": (s or {}).get("status_code", ""),
                 "n_chars": len(t), "substantive": substantive(t, (s or {}).get("status_code")) if s else False, "fetched_at": (s or {}).get("fetched_at", "")})
with open(OUT / "refetch_report.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cnt = collections.Counter(r["status"] for r in rows)
SUM = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "data_dir": str(DATA_ROOT), "round": a.round, "urls": len(rows),
       "fetched_this_run": len(attempted), "status_counts": {k: cnt.get(k, 0) for k in ("UNCHANGED", "CHANGED", "UNAVAILABLE", "NO_STORED_HASH")},
       "substantive": sum(1 for r in rows if r["substantive"]), "snapshots_dir": str(SNAP),
       "note": "content hashes follow web_collection.py; CHANGED means the extracted page content differs from the stored observation"}
json.dump(SUM, open(OUT / "refetch_summary.json", "w"), indent=1); print(json.dumps(SUM["status_counts"]), "substantive:", SUM["substantive"])

# ---- optional: a data directory = DATA_DIR files + the re-fetched pages as snapshots/<round>/
if a.build_data:
    if BD.exists():
        if not (BD / MARKER).exists(): sys.exit(f"{BD} exists and was not created by page_refetch.py; choose a new directory")
        shutil.rmtree(BD)
    shutil.copytree(DATA_ROOT, BD, ignore=shutil.ignore_patterns("snapshots"))
    shutil.copytree(SNAP, BD / "snapshots" / a.round)
    json.dump({"source_data_dir": str(DATA_ROOT), "refetched_round": a.round, "refetch_summary": str(OUT / "refetch_summary.json"),
               "snapshots": sum(1 for _ in (BD / "snapshots" / a.round).glob("*.json"))}, open(BD / MARKER, "w"), indent=1)
    print(f"data directory with re-fetched pages: {BD} (use DATA_DIR={BD})")
