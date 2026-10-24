"""Refetch stage A2 for unseen queries (documented deviation from the frozen collector). On current Wikipedia markup the collector's
_is_dynamic_tag() matches a class on the <html> element and decomposes the whole document, so web_collection.extract_text returns '' and fetch_url
returns None with no error ("silent extraction failure"). This hits most Wikipedia top-1 fresh results and every Wikipedia URL of either path.
Fix: refetch ONCE every URL whose pass-1 record is ok=False with an empty error (silent extraction failure), for BOTH fresh and cached paths
identically, with the same collector except that the dynamic-tag filter may not remove <html> or <body>. 4 threads. No Serper calls.
Pass-1 records are kept unchanged; pass-2 records go to raw/pages_pass2/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import contextlib, datetime, hashlib, io, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "unseen_queries")); PG, P2 = O / "raw/pages", O / "raw/pages_pass2"; P2.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); import web_collection   # noqa: E402
_orig = web_collection._is_dynamic_tag
web_collection._is_dynamic_tag = lambda tag: False if getattr(tag, "name", None) in ("html", "body") else _orig(tag)   # the only change
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
todo = []
for p in sorted(PG.glob("*.json")):
    d = json.load(open(p, encoding="utf-8"))
    if not d.get("ok") and not (d.get("error") or "").strip(): todo.append(d["url"])
uid = lambda u: hashlib.sha1(u.encode()).hexdigest()
def fetch(u):
    p = P2 / f"{uid(u)}.json"
    if p.exists(): return
    ts = now(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: d = web_collection.fetch_url(u); err = None
        except Exception as ex: d, err = None, type(ex).__name__
    json.dump({"url": u, "fetched_utc": ts, "pass": 2, "ok": d is not None, "error": err or (buf.getvalue().strip()[:300] if d is None else None), "final_url": (d or {}).get("final_url"),
               "status_code": (d or {}).get("status_code"), "content_type": (d or {}).get("content_type"), "extracted_text": (d or {}).get("extracted_text", "")}, open(p, "w", encoding="utf-8"), ensure_ascii=False)
t0 = now(); print(f"[{t0}] pass-2 refetch of {len(todo)} silently failed URLs", flush=True)
with ThreadPoolExecutor(4) as ex: list(ex.map(fetch, todo))
ok = sum(1 for u in todo if json.load(open(P2 / f"{uid(u)}.json")).get("ok"))
json.dump({"pass2_window_utc": [t0, now()], "silent_failures_pass1": len(todo), "recovered_pass2": ok, "rule": "dynamic-tag filter may not remove <html>/<body>; everything else frozen"}, open(O / "work/refetch_summary.json", "w"), indent=1)
print(f"[{now()}] pass-2 done: {ok}/{len(todo)} recovered", flush=True)
