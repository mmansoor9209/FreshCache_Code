"""Live collection for the natural QReCC workload, using the frozen live-stream collector with unchanged policy logic.
Additions: (a) the QReCC block (work/qr_block.json, checked against work/qr_protocol_frozen.json), (b) two Serper keys with separate hard caps
(key 1: 2,050 calls, then key 2: 350; read from SERPER_KEY_1 / SERPER_KEY_2 only, never printed, logged or stored), and (c) measured durations
of every search call and page fetch (for measured cost). Requests are processed in arrival order (virtual schedule time; no compressed clock,
no waiting). Per request: ONE Serper search of the request itself (num=10, gl=us, hl=en; cap counted BEFORE each call; no retries; a failed
search gives an empty own list), then every policy steps ONLINE in its own independent state (B, B-minus(elig), B-minus(mask: D' decision at the
same request), D, D') for each frozen configuration (own-list depth k), and every page URL any policy fetches at this request is fetched once by
the evaluator store and stored as version (url, request order). Policies never see another request's own results except through their own L2
registry, and never see future requests. Similarities: frozen BGE-M3, computed for the block queries before the run (only prior entries are
ever consulted). Resumable: state is rebuilt by replaying stored searches before continuing."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import contextlib, datetime, hashlib, io, json, os, pathlib, pickle, sys, time, types; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from concurrent.futures import ThreadPoolExecutor
O = ((OUTPUT_ROOT / "natural_workload")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
RC = O / "raw/qr"; SR, PG = RC / "search", RC / "pages"
for d in (SR, PG): d.mkdir(parents=True, exist_ok=True)
KEYS = [(os.environ.get("SERPER_KEY_1"), 2050, O / "work/serper_counter_qr_key1.json"), (os.environ.get("SERPER_KEY_2"), 350, O / "work/serper_counter_qr_key2.json")]; assert all(k for k, _, _ in KEYS), "keys not set"
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); LOG = open(O / "work/qr_collect.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a)
    for k, _, _ in KEYS: s = s.replace(k, "<redacted>")
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
F = json.load(open(O / "work/qr_block.json")); PR = json.load(open(REPO_ROOT / "configs/protocols/qrecc_stream_protocol.json")); assert F["sha256"] == PR["block_sha256"]
KS = PR["own_list_depths"]; REQ = F["requests"]
import numpy as np   # noqa: E402
import output_workspace; output_workspace.use("natural_workload"); import composition_study_helpers as X, composition_engine_resumable as RV   # noqa: E402
E0 = X.engine(); exp0 = E0["exp"]
SIMF = O / "work/qr_sim.npy"
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
def cnt(f, cap): return json.load(open(f)) if f.exists() else {"calls": 0, "cap": cap}
def counter(): return {"calls": sum(cnt(f, c)["calls"] for _, c, f in KEYS)}
def search(q):
    for key, cap, f in KEYS:
        c = cnt(f, cap)
        if c["calls"] < cap: break
    else: raise RuntimeError("Serper call caps reached")
    c["calls"] += 1; json.dump(c, open(f, "w"))   # counted BEFORE the call
    t0 = time.perf_counter()
    try:
        r = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": key, "Content-Type": "application/json"}, json={"q": q, "num": 10, "gl": "us", "hl": "en"}, timeout=30)
        out = r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None), None
    except Exception as ex: out = None, None, type(ex).__name__
    search.last_s = time.perf_counter() - t0; search.key_index = [k for k, _, _ in KEYS].index(key) + 1; return out
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "")); sys.path.insert(0, str(REPO_ROOT / "scripts/data")); sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation")); sys.path.insert(0, str(REPO_ROOT / "freshcache/method")); import web_collection   # noqa: E402  frozen collector (read-only import)
_orig = web_collection._is_dynamic_tag; web_collection._is_dynamic_tag = lambda tag: False if getattr(tag, "name", None) in ("html", "body") else _orig(tag)
vid = lambda u, i: f"{hashlib.sha1(u.encode()).hexdigest()}_{i:05d}"
def fetch(u, i):
    p = PG / f"{vid(u, i)}.json"
    if p.exists(): return
    ts = now(); buf = io.StringIO(); t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        try: d = web_collection.fetch_url(u); err = None
        except Exception as ex: d, err = None, type(ex).__name__
    json.dump({"url": u, "order": i, "fetched_utc": ts, "duration_s": round(time.perf_counter() - t0, 4), "ok": d is not None, "error": err, "status_code": (d or {}).get("status_code"), "extracted_text": (d or {}).get("extracted_text", "")},
              open(p, "w", encoding="utf-8"), ensure_ascii=False)
POOL = ThreadPoolExecutor(16); ROWS = {}; t0 = now(); fails = 0
log(f"[{t0}] QR collection: {len(REQ)} requests; calls so far {counter()['calls']} / caps 2050+350; depths {KS}")
for i, r in enumerate(REQ):
    sp = SR / f"{i:05d}_{r['query_id']}.json"
    if sp.exists(): s = json.load(open(sp, encoding="utf-8"))
    else:
        ts = now(); code, body, err = search(r["query"])
        s = {"order": i, "query_id": r["query_id"], "query": r["query"], "t_virtual": r["t_virtual"], "searched_utc": ts, "duration_s": round(search.last_s, 4), "key": search.key_index, "http_status": code, "error": err if body is None else None,
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
pickle.dump({"rows": ROWS, "metrics": {f"{k}|{p}": ST[(k, p)].metrics() for k in KS for p in POLS}}, open(O / "work/qr_rows.pkl", "wb"))
json.dump({"requests": len(REQ), "search_failures": fails, "calls_total": counter()["calls"], "caps": [2050, 350], "window_utc": [t0, now()], "pages": len(list(PG.glob("*.json")))},
          open(O / "work/qr_collect_summary.json", "w"), indent=1)
log(f"[{now()}] QR collection done; calls {counter()['calls']}; failures {fails}")
