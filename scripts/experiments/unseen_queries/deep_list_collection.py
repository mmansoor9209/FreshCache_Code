"""Live collection for the 150 paired cross-cluster cases. Stage t0: one Serper search (num=10, gl=us, hl=en) of each distinct SOURCE query,
then fetch of the source top-10 pages. Stage t1 (refused unless after the last t0 search; no minimum gap): one Serper search of each
incoming query (num=10), then fetch of the incoming top-10 AND the source top-10 pages in the same window. Frozen collector (web_collection.fetch_url)
with the pass-2 extractor rule. The API key is read from SERPER_API_KEY only; it is never printed, logged or stored (logs pass a redactor).
Hard cap 320 calls including the 20 pilot calls, counted BEFORE each call; no retries. Usage: deep_list_collection.py t0|t1"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import contextlib, datetime, hashlib, io, json, os, pathlib, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
O = ((OUTPUT_ROOT / "common_sample")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); T2 = (OUTPUT_ROOT / "unseen_queries")
PH = sys.argv[1]; R = O / "raw/deep_lists"; SR = R / f"search_{PH}"; PG = R / f"pages_{PH}"
for d in (SR, PG): d.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("SERPER_API_KEY"); assert KEY, "SERPER_API_KEY not set"
CAP = 320; CNT = O / "work/serper_counter_key2.json"   # combined key-2 cap: 20 pilot + 300 deep-list
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); LOG = open(O / "work/deep_list_collect.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a).replace(KEY, "<redacted>"); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
CASES = [c for c in jl(T2 / "work/cases_built.jsonl") if c["both_evaluable"] and c["reference_available"]]; assert DS.check("deep_list_cases", len(CASES))
FZ = {c["case_id"]: c for c in json.load(open(T2 / "sample_frozen.json"))["cases"]}
qid = lambda q: hashlib.sha1(q.encode()).hexdigest()[:16]
if PH == "t0": Q = sorted({FZ[c["case_id"]]["list_origin_query"] for c in CASES})
else:
    last = max(json.load(open(p))["searched_utc"] for p in (R / "search_t0").glob("*.json"))
    gap = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(last)).total_seconds()
    assert gap > 0, "t1 refused: incoming searches must follow every source search"   # no minimum gap
    Q = sorted({c["query"] for c in CASES})
import requests   # noqa: E402
def counter(): return json.load(open(CNT)) if CNT.exists() else {"calls": 0, "pilot_calls": 0, "cap": CAP}
def search(q):
    c = counter()
    if c["calls"] >= CAP: raise RuntimeError("Serper call cap reached")
    c["calls"] += 1; json.dump(c, open(CNT, "w"))
    try:
        r = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": KEY, "Content-Type": "application/json"}, json={"q": q, "num": 10, "gl": "us", "hl": "en"}, timeout=30)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None), None
    except Exception as ex: return None, None, type(ex).__name__
log(f"[{now()}] deep-list {PH}: {len(Q)} distinct queries; calls so far {counter()['calls']} / cap {CAP}"); fails = 0
for q in Q:
    p = SR / f"{qid(q)}.json"
    if p.exists(): continue
    ts = now(); code, body, err = search(q)
    json.dump({"query": q, "searched_utc": ts, "http_status": code, "error": err if body is None else None, "request_body": {"q": q, "num": 10, "gl": "us", "hl": "en"}, "response": body},
              open(p, "w", encoding="utf-8"), ensure_ascii=False); fails += code != 200; time.sleep(0.15)
log(f"[{now()}] searches done; non-200 {fails}; calls {counter()['calls']}")
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); import web_collection   # noqa: E402  frozen fetch/extraction (read-only import)
_orig = web_collection._is_dynamic_tag; web_collection._is_dynamic_tag = lambda tag: False if getattr(tag, "name", None) in ("html", "body") else _orig(tag)   # pass-2 extractor rule
links = lambda p: [o["link"] for o in (((json.load(open(p, encoding="utf-8")).get("response") or {}).get("organic")) or []) if o.get("link")][:10]
URLS = set()
for p in SR.glob("*.json"): URLS.update(links(p))
if PH == "t1":
    for p in (R / "search_t0").glob("*.json"): URLS.update(links(p))
uid = lambda u: hashlib.sha1(u.encode()).hexdigest()
def fetch(u):
    p = PG / f"{uid(u)}.json"
    if p.exists(): return
    ts = now(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: d = web_collection.fetch_url(u); err = None
        except Exception as ex: d, err = None, type(ex).__name__
    json.dump({"url": u, "fetched_utc": ts, "ok": d is not None, "error": err or (buf.getvalue().strip()[:300] if d is None else None), "final_url": (d or {}).get("final_url"),
               "status_code": (d or {}).get("status_code"), "content_type": (d or {}).get("content_type"), "extracted_text": (d or {}).get("extracted_text", "")}, open(p, "w", encoding="utf-8"), ensure_ascii=False)
f0 = now(); log(f"[{f0}] fetching {len(URLS)} distinct URLs")
with ThreadPoolExecutor(16) as ex: list(ex.map(fetch, sorted(URLS)))
ok = sum(1 for u in URLS if json.load(open(PG / f"{uid(u)}.json")).get("ok"))
json.dump({"phase": PH, "queries": len(Q), "search_failures": fails, "calls_total_key2": counter()["calls"], "cap": CAP, "fetch_window_utc": [f0, now()], "urls": len(URLS), "fetched_ok": ok},
          open(O / f"work/deep_list_{PH}_summary.json", "w"), indent=1)
log(f"[{now()}] fetch done {ok}/{len(URLS)} ok")
