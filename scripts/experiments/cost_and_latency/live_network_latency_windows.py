"""Live-network latency of A, B, D on several additional held-out windows (same harness as
scripts/experiments/cost_and_latency/live_network_latency.py, which covered positions 12,000-12,299).
Modes:
  plan  -- replay A, B, D once through the latest window end (each policy's own cache state), store window routes in
           work/live_rows.pkl, and count the unique live searches each candidate window needs. No API calls.
  run W0 [W0 ...] -- execute the windows for real: BGE-M3 embedding, LIVE Serper search (num=2, no retries; one call per unique
           query, shared across policies), LIVE HTTP fetches, local page reads for reused pages, Llama-3.2-3B generation per
           answer-cache miss. Per-request latencies and routes are written to outputs/live_latency_W0.json.
The API key is read from SERPER_API_KEY only and is never printed, logged or written. HARD CAP: 400 calls (counter
work/serper_call_counter_windows.json); a window is started only if its planned unique searches fit under the remaining cap."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, json, os, pickle, random, statistics, sys, time, pathlib, hashlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "composition_checks")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("composition_checks"); import composition_study_helpers as X, composition_engine as RE   # noqa: E402
MODE = sys.argv[1]; WINDOWS = [3000, 7000, 16000, 20000]; WL = 300
E = X.engine(); p2 = E["p2"]; H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0)
POL = {"A": dict(l2l3=False), "B": dict(l2_on=False), "D": dict()}
MISS = ("miss", "REAL_TIME", "miss_rejected")
if MODE == "plan":
    end = max(WINDOWS) + WL
    ROWS = {p: RE.replay(E, H[:end], l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **kw)[1] for p, kw in POL.items()}
    keep = {w: [r["query_id"] for _, r in H[w:w + WL]] for w in WINDOWS}
    ROWS = {p: {q: ROWS[p][q] for w in WINDOWS for q in keep[w]} for p in POL}
    pickle.dump({"windows": WINDOWS, "ids": keep, "query": {r["query_id"]: r["query"] for w in WINDOWS for _, r in H[w:w + WL]}, "rows": ROWS}, open(O / "work/live_rows.pkl", "wb"))
    Q = pickle.load(open(O / "work/live_rows.pkl", "rb"))["query"]
    plan = {w: len({Q[q] for q in keep[w] for p in POL if ROWS[p][q]["tier"] in MISS}) for w in WINDOWS}
    json.dump(plan, open(O / "work/live_plan.json", "w"), indent=1); print(plan, "total", sum(plan.values())); sys.exit(0)

import requests, torch   # noqa: E402
import answer_grading as e2e   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from sentence_transformers import SentenceTransformer   # noqa: E402
KEY = os.environ.get("SERPER_API_KEY"); assert KEY, "SERPER_API_KEY not set"
CAP = 400; CNT = O / "work/serper_call_counter_windows.json"; calls = json.load(open(CNT))["calls"] if CNT.exists() else 0
P = pickle.load(open(O / "work/live_rows.pkl", "rb")); PLAN = json.load(open(O / "work/live_plan.json"))
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
def live_fetch(uh):
    u = URL.get(uh); t0 = time.perf_counter()
    if not u: return 0.0, "no_url"
    try: r = requests.get(u, headers=HDR, timeout=20); st = r.status_code; _ = r.content
    except Exception as ex: st = type(ex).__name__
    return (time.perf_counter() - t0) * 1e3, st
for W0 in [int(x) for x in sys.argv[2:]]:
    need = PLAN[str(W0)]
    if calls + need > CAP: print(f"window {W0}: needs {need}, remaining {CAP - calls}; skipped"); continue
    SEARCH = {}
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
    REQ = []; rng = random.Random(7 + W0)
    for qid in P["ids"][W0]:
        q = P["query"][qid]; order = list(POL); rng.shuffle(order); rec = {"query_id": qid}
        for p in order:
            row = P["rows"][p][qid]; torch.cuda.synchronize(); t0 = time.perf_counter()
            enc.encode([q], normalize_embeddings=True); te = time.perf_counter(); s_ms = f_ms = g_ms = 0.0; add = 0.0; nf = ff = 0
            if row["tier"] in MISS:
                s = live_search(q); s_ms = s["ms"]; add = 0.0 if s["now"] else s_ms
            if row["tier"] != "L1":
                for uh, tv in (row.get("ev") or []):
                    if tv == row["t"]:
                        ms, st = live_fetch(uh); f_ms += ms; nf += 1; ff += 0 if st == 200 else 1
                    else: e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv))
                ctx = p2.ctx_for(tuple(tuple(x) for x in row["ev"])) if row.get("ev") else ""
                tg = time.perf_counter(); generate(row.get("gen_query", q), ctx); torch.cuda.synchronize(); g_ms = (time.perf_counter() - tg) * 1e3
            wall = (time.perf_counter() - t0) * 1e3
            rec[p] = {"tier": row["tier"], "ms": round(wall + add, 2), "search_ms": round(s_ms, 2), "fetch_ms": round(f_ms, 2), "gen_ms": round(g_ms, 2), "embed_ms": round((te - t0) * 1e3, 2), "fetches": nf, "fetch_fail": ff}
        REQ.append(rec); print(f"\rW{W0} {len(REQ)}/{WL} calls={calls}", end="", flush=True)
    json.dump({"window": [W0, W0 + WL], "serper_calls_total_windows": calls, "cap": CAP, "unique_searches": len(SEARCH), "search_failures": sum(1 for v in SEARCH.values() if not v["ok"]), "requests": REQ},
              open(O / f"outputs/live_latency_window_{W0}.json", "w"), indent=0)
    print()
