"""Live-network runtime check for A, B, D on the same fixed window (held-out positions 12,000-12,299) as local_runtime_check.py.
Cache state: each policy's own state evolves through the preceding stream (composition_engine replay); window routes are the policy's own.
Executed for real per request and policy (policy order rotated per request): BGE-M3 query embedding (GPU); LIVE Serper search
(google.serper.dev, num=2 as the frozen collector, no retries) for every request the policy searches, issued once per unique query and its
measured time attributed to each policy issuing the same query (shared read-only store of identical external responses); LIVE HTTP GET of
every page the policy fetches (frozen collector headers, 20 s timeout), sequential; page reuse = local snapshot read; Llama-3.2-3B
generation for every answer-cache miss. Generation context = the replay evidence (replay-routed, as in the earlier live diagnostic).
The API key is read from the SERPER_API_KEY environment variable only and is never printed, logged or written. HARD CAP: 100 calls.
Writes outputs/runtime_live.json and raw/live_search/*.json (responses only)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, json, os, random, statistics, sys, time, pathlib, hashlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import requests
O = ((OUTPUT_ROOT / "composition")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
KEY = os.environ.get("SERPER_API_KEY"); assert KEY, "SERPER_API_KEY not set"
CAP = 100; CNT = O / "work/serper_call_counter.json"; calls = json.load(open(CNT))["calls"] if CNT.exists() else 0
import composition_helpers as X, composition_engine as RE   # noqa: E402
import answer_grading as e2e   # noqa: E402
import torch   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from sentence_transformers import SentenceTransformer   # noqa: E402
E = X.engine(); p2 = E["p2"]; H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0)
URL = {}
for l in open((DATA_ROOT / "url_manifest.jsonl"), encoding="utf-8"):
    m = json.loads(l)
    if m.get("url_hash") and m.get("url"): URL.setdefault(m["url_hash"], m["url"])
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9,ko;q=0.8", "Accept-Encoding": "gzip, deflate, br"}
K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(X.ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("FORCED", "MODELS")}
tok = AutoTokenizer.from_pretrained(K["MODELS"]["llama3b"]); tok.pad_token = tok.pad_token or tok.eos_token
gen = AutoModelForCausalLM.from_pretrained(K["MODELS"]["llama3b"], dtype=torch.bfloat16, device_map="cuda:0"); gen.eval()
enc = SentenceTransformer("BAAI/bge-m3", device="cuda:0"); enc.encode(["warm"], normalize_embeddings=True)
def generate(q, ctx):
    x = tok([K["FORCED"].format(question=q, context=ctx)], return_tensors="pt", truncation=True, max_length=1024).to(gen.device)
    with torch.no_grad(): gen.generate(**x, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
generate("warm up", "warm up")
W0, W1 = 12000, 12300; WIN = H[W0:W1]; POL = {"A": dict(l2l3=False), "B": dict(l2_on=False), "D": dict()}
ROWS = {p: RE.replay(E, H[:W1], l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **kw)[1] for p, kw in POL.items()}
SEARCH = {}   # query -> {"ms", "ok"}
def live_search(q):
    global calls
    if q in SEARCH: return {**SEARCH[q], "now": False}
    if calls >= CAP: raise SystemExit(f"cap {CAP} reached")
    calls += 1; json.dump({"calls": calls, "cap": CAP}, open(CNT, "w"))
    t0 = time.perf_counter()
    try:
        r = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": KEY, "Content-Type": "application/json"}, json={"q": q, "num": 2}, timeout=20)
        ms = (time.perf_counter() - t0) * 1e3; ok = r.status_code == 200; body = r.json() if ok else {"status": r.status_code}
    except Exception as ex:
        ms = (time.perf_counter() - t0) * 1e3; ok = False; body = {"error": type(ex).__name__}
    hq = hashlib.sha256(q.encode()).hexdigest()[:16]
    json.dump({"query": q, "ok": ok, "ms": ms, "response": body}, open(O / f"raw/live_search/{hq}.json", "w"), ensure_ascii=False)
    SEARCH[q] = {"ms": ms, "ok": ok}; return {**SEARCH[q], "now": True}
def live_fetch(uh):
    u = URL.get(uh); t0 = time.perf_counter()
    if not u: return 0.0, "no_url"
    try: r = requests.get(u, headers=HDR, timeout=20); st = r.status_code; _ = r.content
    except Exception as ex: st = type(ex).__name__
    return (time.perf_counter() - t0) * 1e3, st
LAT = {p: [] for p in POL}; COMP = {p: {"search_ms": 0.0, "fetch_ms": 0.0, "gen_ms": 0.0, "embed_ms": 0.0, "fetches": 0, "fetch_fail": 0, "searches": 0} for p in POL}
rng = random.Random(7)
for _, r in WIN:
    order = list(POL); rng.shuffle(order)
    for p in order:
        row = ROWS[p][r["query_id"]]; torch.cuda.synchronize(); t0 = time.perf_counter()
        enc.encode([r["query"]], normalize_embeddings=True); te = time.perf_counter(); s_ms = f_ms = 0.0; add = 0.0
        if row["tier"] in ("miss", "REAL_TIME", "miss_rejected"):
            s = live_search(r["query"]); s_ms = s["ms"]; COMP[p]["searches"] += 1; add = 0.0 if s["now"] else s_ms   # shared response: attribute the measured search time
        if row["tier"] != "L1":
            for uh, tv in (row.get("ev") or []):
                if tv == row["t"]:
                    ms, st = live_fetch(uh); f_ms += ms; COMP[p]["fetches"] += 1; COMP[p]["fetch_fail"] += 0 if st == 200 else 1
                else: e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv))
            ctx = p2.ctx_for(tuple(tuple(x) for x in row["ev"])) if row.get("ev") else ""
            tg = time.perf_counter(); generate(row.get("gen_query", r["query"]), ctx); torch.cuda.synchronize(); COMP[p]["gen_ms"] += (time.perf_counter() - tg) * 1e3
        wall = (time.perf_counter() - t0) * 1e3   # measured wall-clock (embedding + fetches + generation); search time added below
        COMP[p]["embed_ms"] += (te - t0) * 1e3; COMP[p]["search_ms"] += s_ms; COMP[p]["fetch_ms"] += f_ms
        LAT[p].append(wall + add)
    print(f"\r{len(LAT['A'])}/{len(WIN)} calls={calls}", end="", flush=True)
R = {"window": [W0, W1], "serper_calls_total": calls, "cap": CAP, "unique_searches": len(SEARCH), "search_failures": sum(1 for v in SEARCH.values() if not v["ok"]), "policies": {}}
for p in POL:
    L = LAT[p]; n = len(L)
    R["policies"][p] = {"mean_ms": round(statistics.mean(L), 1), "p50_ms": round(statistics.median(L), 1), "p95_ms": round(sorted(L)[int(.95 * n) - 1], 1),
                        "mean_search_ms_per_request": round(COMP[p]["search_ms"] / n, 1), "mean_fetch_ms_per_request": round(COMP[p]["fetch_ms"] / n, 1), "mean_gen_ms_per_request": round(COMP[p]["gen_ms"] / n, 1),
                        "mean_embed_ms_per_request": round(COMP[p]["embed_ms"] / n, 1), "searches": COMP[p]["searches"], "fetches": COMP[p]["fetches"], "fetch_failures": COMP[p]["fetch_fail"]}
json.dump(R, open(O / "outputs/runtime_live.json", "w"), indent=1); print("\n" + json.dumps(R, indent=1))
