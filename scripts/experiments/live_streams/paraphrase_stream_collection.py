"""Online live-search collection for the frozen request block work/c_block.json. Requests are processed in arrival order (virtual schedule time;
no compressed clock, no waiting). Per request: ONE Serper search of the request itself (num=10, gl=us, hl=en; key 1; hard cap 2,400
counted BEFORE each call; no retries; a failed search gives an empty own list), then every policy steps ONLINE in its own independent state
(B, B-minus(elig), B-minus(mask: D' decision at the same request), D, D') for each frozen configuration (own-list depth k), and every page URL any
policy fetches at this request is fetched once by the evaluator store (frozen collector, pass-2 rule) and stored as version (url, request order).
Policies never see another request's own results except through their own L2 registry, and never see future requests. Similarities: frozen BGE-M3,
computed for the block queries before the run (prior entries only are ever consulted). The API key is read from SERPER_API_KEY only; it is never
printed, logged or stored. Resumable: state is rebuilt by replaying stored searches before continuing."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import contextlib, datetime, hashlib, io, json, os, pathlib, pickle, sys, time, types; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from concurrent.futures import ThreadPoolExecutor
O = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
RC = O / "raw/c"; SR, PG = RC / "search", RC / "pages"
for d in (SR, PG): d.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("SERPER_API_KEY"); assert KEY, "SERPER_API_KEY not set"; CAP = 2400; CNT = O / "work/serper_counter_key1.json"
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); LOG = open(O / "work/c_collect.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a).replace(KEY, "<redacted>"); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
F = json.load(open(O / "work/c_block.json")); PR = json.load(open(REPO_ROOT / "configs/protocols/paraphrase_stream_protocol.json")); assert F["sha256"] == PR["block_sha256"]
KS = PR["own_list_depths"]; REQ = F["requests"]
import numpy as np   # noqa: E402
import output_workspace; output_workspace.use("common_sample"); import composition_study_helpers as X, composition_engine_resumable as RV   # noqa: E402
E0 = X.engine(); exp0 = E0["exp"]
SIMF = O / "work/c_sim.npy"
if not SIMF.exists():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-m3", device="cuda:0"); V = m.encode([r["query"] for r in REQ], batch_size=64, normalize_embeddings=True, convert_to_numpy=True); np.save(SIMF, (V @ V.T).astype(np.float32)); del m
SIM = np.load(SIMF); IDX = {}
for i, r in enumerate(REQ): IDX.setdefault(r["query"], i)
class Exp(types.SimpleNamespace):
    def __getattr__(self, k): return getattr(exp0, k)
E = {"exp": Exp(_SIM_MATRIX=SIM, _QUERY_TO_IDX=IDX)}
TH, KA = X.ca_params(0.0, 1.0); kw = dict(floor=0.80, theta_by_class=TH, kappa_by_class=KA)
POLS = ["B", "Be", "D", "Dp", "Bm"]   # Dp steps before Bm (Bm's mask uses D''s decision at the same request)
ST = {(k, p): RV.State(E, len(REQ), l2_on=p in ("D", "Dp"), write_rule={"B": "all", "D": "all", "Dp": "no_l2", "Bm": "mask", "Be": "elig"}[p], mask=set(), **kw) for k in KS for p in POLS}
import requests   # noqa: E402
def counter(): return json.load(open(CNT)) if CNT.exists() else {"calls": 0, "cap": CAP}
def search(q):
    c = counter()
    if c["calls"] >= CAP: raise RuntimeError("Serper call cap reached")
    c["calls"] += 1; json.dump(c, open(CNT, "w"))
    try:
        r = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": KEY, "Content-Type": "application/json"}, json={"q": q, "num": 10, "gl": "us", "hl": "en"}, timeout=30)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None), None
    except Exception as ex: return None, None, type(ex).__name__
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "")); sys.path.insert(0, str(REPO_ROOT / "scripts/data")); sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation")); sys.path.insert(0, str(REPO_ROOT / "freshcache/method")); import web_collection   # noqa: E402  frozen collector (read-only import)
_orig = web_collection._is_dynamic_tag; web_collection._is_dynamic_tag = lambda tag: False if getattr(tag, "name", None) in ("html", "body") else _orig(tag)
vid = lambda u, i: f"{hashlib.sha1(u.encode()).hexdigest()}_{i:05d}"
def fetch(u, i):
    p = PG / f"{vid(u, i)}.json"
    if p.exists(): return
    ts = now(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: d = web_collection.fetch_url(u); err = None
        except Exception as ex: d, err = None, type(ex).__name__
    json.dump({"url": u, "order": i, "fetched_utc": ts, "ok": d is not None, "error": err, "status_code": (d or {}).get("status_code"), "extracted_text": (d or {}).get("extracted_text", "")},
              open(p, "w", encoding="utf-8"), ensure_ascii=False)
POOL = ThreadPoolExecutor(16); ROWS = {}; t0 = now(); fails = 0
log(f"[{t0}] C collection: {len(REQ)} requests; key-1 calls so far {counter()['calls']} / cap {CAP}; depths {KS}")
for i, r in enumerate(REQ):
    sp = SR / f"{i:05d}_{r['query_id']}.json"
    if sp.exists(): s = json.load(open(sp, encoding="utf-8"))
    else:
        ts = now(); code, body, err = search(r["query"])
        s = {"order": i, "query_id": r["query_id"], "query": r["query"], "t_virtual": r["t_virtual"], "searched_utc": ts, "http_status": code, "error": err if body is None else None,
             "request_body": {"q": r["query"], "num": 10, "gl": "us", "hl": "en"}, "response": body}
        json.dump(s, open(sp, "w", encoding="utf-8"), ensure_ascii=False); fails += code != 200
    links = [o["link"] for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")][:10]
    need = set()
    for k in KS:
        rec = {"query": r["query"], "query_id": r["query_id"], "freshness_class": r["freshness_class"], "urls": [{"url_hash": u} for u in links[:k]]}
        for p in POLS:
            st = ST[(k, p)]
            if p == "Bm" and ROWS[(k, "Dp", r["query_id"])]["tier"] == "L2": st.mask.add(r["query_id"])
            row = st.step(r["t_virtual"], rec); ROWS[(k, p, r["query_id"])] = row
            need.update(u for u, tv in row["ev"] if tv == r["t_virtual"] and row["tier"] != "L1")
    for u in sorted(need): POOL.submit(fetch, u, i)
    if i % 100 == 0: log(f"[{now()}] {i}/{len(REQ)} calls {counter()['calls']} failures {fails}")
POOL.shutdown(wait=True)
pickle.dump({"rows": ROWS, "metrics": {f"{k}|{p}": ST[(k, p)].metrics() for k in KS for p in POLS}}, open(O / "work/c_rows.pkl", "wb"))
json.dump({"requests": len(REQ), "search_failures": fails, "calls_key1": counter()["calls"], "cap": CAP, "window_utc": [t0, now()], "pages": len(list(PG.glob("*.json")))},
          open(O / "work/c_collect_summary.json", "w"), indent=1)
log(f"[{now()}] C collection done; calls {counter()['calls']}; failures {fails}")
