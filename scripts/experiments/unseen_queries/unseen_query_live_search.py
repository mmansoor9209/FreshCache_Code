"""Unseen-query live search and fetch (web access): a contemporary independent-search experiment (not a reconstruction of historical rankings).
Population: every untouched-cluster query (cluster never used for development, tuning, validation selection, verifier selection or test
diagnostics) that FreshCache's frozen L2 rule would serve from a cached URL list (unseen_query_candidate_matching.py): cross-cluster cases. A within-cluster stratum
cannot be drawn from untouched clusters (they have no paraphrases); this is documented, not filled with previously used clusters.
Per case: ONE Serper search of the incoming query itself (num=10, gl=us, hl=en), raw JSON stored (request body, timestamp, rank order; the API key
is read from the SERPER_API_KEY environment variable and is never printed, logged or stored). HARD CAP: CAP search calls (CAP below), no retries. Then live page
fetches (frozen collector web_collection.fetch_url) for the fresh top-10 and for the cached-list URLs, all in the same experimental window."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import contextlib, datetime, hashlib, io, json, os, pathlib, random, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "unseen_queries"))
SR, PG = O / "raw/search", O / "raw/pages"; SR.mkdir(parents=True, exist_ok=True); PG.mkdir(parents=True, exist_ok=True)
CNT = O / "work/serper_call_counter.json"; CAP = 358; KEY = os.environ.get("SERPER_API_KEY"); assert KEY, "SERPER_API_KEY not set"
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); LOG = open(O / "work/search_fetch.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a).replace(KEY, "<redacted>"); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
# ---------------- freeze the sample BEFORE any call
FZP = O / "sample_frozen.json"
if not FZP.exists():
    M = [json.loads(l) for l in open(O / "untouched_l2_matching.jsonl", encoding="utf-8")]; L2 = [m for m in M if m["route"] == "L2"]
    url_of = {}
    for l in open((DATA_ROOT / "url_manifest.jsonl"), encoding="utf-8"):
        x = json.loads(l)
        if x["run_id"] == "run_00": url_of.setdefault(x["url_hash"], x["url"])
    gold = json.load(open((OUTPUT_ROOT / "gold_answers/reference_answers.json"), encoding="utf-8"))
    cases = []
    for i, m in enumerate(sorted(L2, key=lambda m: m["query_id"])):
        g = gold.get(m["query_id"]) or gold.get(m["query"]) or gold.get(m["query"].strip().lower())
        cases.append({"case_id": f"t2_{i:04d}", "query_id": m["query_id"], "incoming_query": m["query"], "freshness_class": m["freshness_class"], "source": m["source"], "stratum": "CROSS",
                      "similarity": m["similarity"], "entry_class": m["entry_class"], "list_origin": m["list_origin"], "list_origin_query": m["list_origin_query"], "list_origin_cluster": m["list_origin_cluster"],
                      "cached_url_hashes": m["cached_urls"], "cached_urls": [url_of.get(u) for u in m["cached_urls"]], "l3_cached_url_hashes_at_eval_t": m["l3_cached_at_t"], "eval_t": m["eval_t"],
                      "reference_answer": (g or {}).get("answer"), "reference_aliases": (g or {}).get("aliases", []), "reference_source": (g or {}).get("gold_source")})
    FZ = {"frozen_utc": now(), "population": "all untouched-cluster queries that receive an L2 hit under the frozen FreshCache rule (unseen_query_candidate_matching.py, seed 20261004)", "n": len(cases),
          "strata": {"CROSS": len(cases), "WITHIN": 0}, "within_cluster_note": "untouched clusters contain no paraphrases, so no within-cluster case exists among never-used clusters",
          "search": {"provider": "Serper google.serper.dev/search", "num": 10, "gl": "us", "hl": "en", "calls_cap": CAP, "retries": 0}, "cases": cases}
    FZ["sha256"] = hashlib.sha256(json.dumps(FZ["cases"], sort_keys=True).encode()).hexdigest(); json.dump(FZ, open(FZP, "w"), indent=1)
FZ = json.load(open(FZP)); log(f"[{now()}] frozen sample {FZ['n']} cases sha256 {FZ['sha256'][:16]}")
import requests   # noqa: E402
def counter():
    return json.load(open(CNT)) if CNT.exists() else {"calls": 0}
def search(q):
    c = counter()
    if c["calls"] >= CAP: raise RuntimeError("Serper call cap reached")
    c["calls"] += 1; json.dump(c, open(CNT, "w"))   # counted BEFORE the call, so a crash can never exceed the cap
    try:
        r = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": KEY, "Content-Type": "application/json"}, json={"q": q, "num": 10, "gl": "us", "hl": "en"}, timeout=30)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None), None
    except Exception as ex: return None, None, type(ex).__name__
t_s0 = now(); log(f"[{t_s0}] search stage start; calls so far {counter()['calls']} / cap {CAP}")
fails = 0
for c in FZ["cases"]:
    p = SR / f"{c['case_id']}.json"
    if p.exists(): continue
    ts = now(); code, body, err = search(c["incoming_query"])
    rec = {"case_id": c["case_id"], "query_id": c["query_id"], "query": c["incoming_query"], "searched_utc": ts, "http_status": code, "error": err if body is None else None,
           "request_body": {"q": c["incoming_query"], "num": 10, "gl": "us", "hl": "en"}, "endpoint": "https://google.serper.dev/search", "response": body}
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); fails += body is None or code != 200
    if code != 200: log(f"  {c['case_id']} search status {code} err {err}")
    time.sleep(0.2)
t_s1 = now(); log(f"[{t_s1}] searches done; non-200/failed {fails}; calls {counter()['calls']}")
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); import web_collection   # noqa: E402  frozen fetch/extraction
URLS = set()
for c in FZ["cases"]:
    URLS.update(u for u in c["cached_urls"] if u)
    s = json.load(open(SR / f"{c['case_id']}.json", encoding="utf-8"))
    for o in ((s.get("response") or {}).get("organic") or [])[:10]:
        if o.get("link"): URLS.add(o["link"])
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
from concurrent.futures import ThreadPoolExecutor   # noqa: E402
t_f0 = now(); log(f"[{t_f0}] fetching {len(URLS)} distinct URLs")
with ThreadPoolExecutor(16) as ex: list(ex.map(fetch, sorted(URLS)))
t_f1 = now(); okn = sum(1 for u in URLS if json.load(open(PG / f"{uid(u)}.json")).get("ok"))
json.dump({"serper_calls_total": counter()["calls"], "cap": CAP, "search_window_utc": [t_s0, t_s1], "fetch_window_utc": [t_f0, t_f1], "distinct_urls": len(URLS), "fetched_ok": okn, "search_failures": fails},
          open(O / "work/search_fetch_summary.json", "w"), indent=1)
log(f"[{t_f1}] fetch done: {okn}/{len(URLS)} ok; Serper calls {counter()['calls']}")
