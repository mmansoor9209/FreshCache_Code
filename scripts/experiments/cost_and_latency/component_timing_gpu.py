"""Systems timing, stage 2 (same environment, GPU 4 + CPU): embedding, vector search, local page read, generation timing, and the end-to-end
sequential diagnostic (NoCache vs FreshCache) on a fixed sample. No network call is made: live search/fetch are NOT measured here."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import ast, collections, csv, datetime, json, os, pathlib, random, re, statistics, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
GPU = sys.argv[1] if len(sys.argv) > 1 else "4"; os.environ["CUDA_VISIBLE_DEVICES"] = GPU
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); CRR = (OUTPUT_ROOT / "replay"); W = (OUTPUT_ROOT / "systems_timing")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); sys.path.insert(0, str(ROOT / "freshcache/simulation"))
import numpy as np, torch   # noqa: E402
import engine_loader as C, reference_replay as R, stepwise_replay as ON   # noqa: E402
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
E = C.load("reported"); exp, p2, me = E["exp"], E["p2"], E["me"]; P = C.streams(E)["primary"]
EMB = np.load((DATA_ROOT / "query_embeddings_bgem3.npy"), mmap_mode="r"); IDX = exp._QUERY_TO_IDX
def summ(v):
    v = sorted(v); n = len(v); return {"n": n, "mean_ms": round(statistics.mean(v), 4), "p50_ms": round(v[n // 2], 4), "p95_ms": round(v[int(.95 * n)], 4)} if v else {"n": 0}
def sync(): torch.cuda.synchronize()
OUT = {"started": now(), "gpu": torch.cuda.get_device_name(0), "host": os.environ.get("FRESHCACHE_HOST_LABEL", "local")}
rng = random.Random(42)
# ---------------- models
from sentence_transformers import SentenceTransformer   # noqa: E402
emb = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
RUN = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open((ROOT / "scripts/evaluation/generation_judging_stage.py")).read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("FORCED", "MODELS")}
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
tok = AutoTokenizer.from_pretrained(RUN["MODELS"]["llama3b"]); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
gm = AutoModelForCausalLM.from_pretrained(RUN["MODELS"]["llama3b"], dtype=torch.bfloat16, device_map="cuda:0"); gm.eval()
def generate(question, context):
    enc = tok([RUN["FORCED"].format(question=question, context=context)], return_tensors="pt", truncation=True, max_length=1024).to(gm.device)
    sync(); t0 = time.perf_counter()
    with torch.no_grad(): g = gm.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    sync(); return (time.perf_counter() - t0) * 1000, int(enc["input_ids"].shape[1]), int(g.shape[1] - enc["input_ids"].shape[1]), tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
def embed(q):
    sync(); t0 = time.perf_counter(); v = emb.encode([q], show_progress_bar=False, normalize_embeddings=True); sync(); return (time.perf_counter() - t0) * 1000, v[0]
def read_page(u, tv):
    t0 = time.perf_counter(); rnd = p2.snapshot_round_on_disk(u, tv); pth = (DATA_ROOT / "snapshots") / rnd / f"{u}.json"
    txt = (json.load(open(pth, encoding="utf-8")).get("extracted_text") or "") if pth.exists() else ""; txt = re.sub(r"\s+", " ", txt).strip()
    return (time.perf_counter() - t0) * 1000, txt
# ---------------- component micro-benchmarks
nonrt = [r for _, r in P if r["freshness_class"] != "REAL_TIME"]; qs = [r["query"] for r in rng.sample(nonrt, 200)]
for q in qs[:5]: embed(q)
emb_ms = [embed(q)[0] for q in qs]
live_vs_stored = [float(np.dot(embed(q)[1], EMB[IDX[q]])) for q in qs[:50]]
V = np.asarray(EMB, dtype=np.float32); ann = {}
for nidx in (1000, 5000, 10000, 20000, 31000):
    M = V[:nidx]; qv = V[rng.sample(range(len(V)), 200)]; ts = []
    for x in qv:
        t0 = time.perf_counter_ns(); s = M @ x; np.argpartition(-s, 5)[:5]; ts.append((time.perf_counter_ns() - t0) / 1e6)
    ann[nidx] = summ(ts)
pages = rng.sample([(u["url_hash"], t) for t, r in P for u in r["urls"]], 500); page_ms = [read_page(u, t)[0] for u, t in pages]
gen_items = rng.sample([(t, r) for t, r in P if r["freshness_class"] != "REAL_TIME"], 100); gen_ms, gin, gout = [], [], []
for i, (t, r) in enumerate(gen_items[:2] + gen_items):
    ctx = p2.ctx_for(tuple((u["url_hash"], t) for u in r["urls"])); ms, a, b, _ = generate(r["query"], ctx)
    if i >= 2: gen_ms.append(ms); gin.append(a); gout.append(b)
OUT["micro"] = {"embedding_bge_m3_batch1": summ(emb_ms), "live_vs_stored_embedding_cosine_min": round(min(live_vs_stored), 6),
                "vector_search_numpy_bruteforce_by_index_size": ann, "local_snapshot_page_read": summ(page_ms),
                "generation_llama3b_batch1": {**summ(gen_ms), "mean_input_tokens": round(statistics.mean(gin), 1), "mean_output_tokens": round(statistics.mean(gout), 1)}}
print(json.dumps(OUT["micro"], indent=1), flush=True)
# ---------------- end-to-end sequential diagnostic
cl = collections.defaultdict(list)
for t, r in P: cl[r.get("cluster_id") or r["query_id"]].append((t, r))
elig = sorted(c for c, v in cl.items() if len(v) >= 3); pick = set(random.Random(42).sample(elig, 40))
SUB = [(t, r) for t, r in P if (r.get("cluster_id") or r["query_id"]) in pick]
ref_m, ref_tiers, _, _ = R.Replay(E).run(SUB)
rows, online, answers, store = [], ON.Online(E), {}, {}
for t, r in SUB:                                           # ---- FreshCache
    q = r["query"]; rec = {"policy": "FreshCache", "query_id": r["query_id"], "t": t}
    if r["freshness_class"] != "REAL_TIME":
        rec["embed_ms"], v = embed(q)
        n1 = len(online.L1["qi"]); t0 = time.perf_counter_ns()
        if n1: s = V[np.asarray(online.L1["qi"])] @ v; np.argpartition(-s, min(5, n1 - 1))
        rec["vector_search_ms"] = (time.perf_counter_ns() - t0) / 1e6
    d = online.decide(t, r); rec["tier"] = d["tier"]; rec.update({k: round(v_, 4) for k, v_ in d["timers_ms"].items()})
    fm = 0.0
    for u in d["fetch"]:
        ms, txt = read_page(u, t); fm += ms; store[u] = txt
    rec["pages_fetched_local"] = len(d["fetch"]); rec["page_fetch_local_ms"] = round(fm, 3)
    if d["tier"] == "L1":
        t0 = time.perf_counter_ns(); a = answers[d["l1_entry"]]; rec["stored_answer_read_ms"] = (time.perf_counter_ns() - t0) / 1e6; rec["generation_ms"] = 0.0
    else:
        t0 = time.perf_counter(); ctx = p2.ctx_for(tuple(tuple(x) for x in d["ev"])); rec["context_build_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        rec["generation_ms"], rec["input_tokens"], rec["output_tokens"], a = generate(q, ctx)
        if d.get("new_l1_entry") is not None: answers[d["new_l1_entry"]] = a
    rows.append(rec)
G = {"online_decisions_equal_replay_transcription_on_sample": [x["tier"] for x in rows] == [ref_tiers[r["query_id"]] for _, r in SUB]}
for t, r in SUB:                                           # ---- NoCache
    rec = {"policy": "NoCache", "query_id": r["query_id"], "t": t, "tier": "none"}; own = [u["url_hash"] for u in r["urls"]]
    fm = sum(read_page(u, t)[0] for u in own); rec["pages_fetched_local"] = len(own); rec["page_fetch_local_ms"] = round(fm, 3)
    t0 = time.perf_counter(); ctx = p2.ctx_for(tuple((u, t) for u in own)); rec["context_build_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    rec["generation_ms"], rec["input_tokens"], rec["output_tokens"], _ = generate(r["query"], ctx); rows.append(rec)
OUT.update({"e2e_sample": {"clusters": 40, "requests": len(SUB), "seed": 42, "tiers": dict(collections.Counter(x["tier"] for x in rows if x["policy"] == "FreshCache")),
                           "reference_replay_on_sample": ref_m}, "gates": G, "finished": now()})
cols = list(dict.fromkeys(k for x in rows for k in x))
with open(W / "results/end_to_end_requests.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
json.dump(OUT, open(W / "intermediate/generation_judging_stage.json", "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "micro"}, indent=1))
