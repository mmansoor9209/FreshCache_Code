"""Targeted runtime check (local-retrieval regime). Same fixed window of 300 consecutive held-out requests (positions 12,000-12,299) for
A, B and D. Each policy's cache state evolves through the preceding stream by its own admission and update rules (composition_engine replay); within the
window every request is EXECUTED for real and timed end to end: BGE-M3 query embedding (GPU), the policy's route, page reads for every
served page from the local snapshot store (fetched or L3-reused alike are read from disk; no network), context assembly, and Llama-3.2-3B
generation (frozen FORCED prompt, greedy, 80 tokens, batch 1) for every answer-cache miss; answer-cache hits return the stored answer.
Live search and live fetch are NOT executed (no API budget): this measures wall-clock latency when retrieval is local, separately from the
cost model. Writes outputs/runtime_local.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import ast, json, statistics, sys, time, pathlib
O = ((OUTPUT_ROOT / "composition")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import composition_helpers as X, composition_engine as RE   # noqa: E402
import answer_grading as e2e   # noqa: E402
import torch   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from sentence_transformers import SentenceTransformer   # noqa: E402
E = X.engine(); p2 = E["p2"]; H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0)
K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(X.ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("FORCED", "MODELS")}
tok = AutoTokenizer.from_pretrained(K["MODELS"]["llama3b"]); tok.pad_token = tok.pad_token or tok.eos_token
gen = AutoModelForCausalLM.from_pretrained(K["MODELS"]["llama3b"], dtype=torch.bfloat16, device_map="cuda:0"); gen.eval()
enc = SentenceTransformer("BAAI/bge-m3", device="cuda:0"); enc.encode(["warm"], normalize_embeddings=True)
W0, W1 = 12000, 12300; WIN = H[W0:W1]
def generate(q, ctx):
    x = tok([K["FORCED"].format(question=q, context=ctx)], return_tensors="pt", truncation=True, max_length=1024).to(gen.device)
    with torch.no_grad(): gen.generate(**x, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
generate("warm up", "warm up")
R = {}
for p, kw in {"A": dict(l2l3=False), "B": dict(l2_on=False), "D": dict()}.items():
    m, rows = RE.replay(E, H[:W1], l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **kw)   # state evolves through the prefix; window routes are the policy's own
    lat, gens = [], 0
    for _, r in WIN:
        row = rows[r["query_id"]]; torch.cuda.synchronize(); t0 = time.perf_counter()
        enc.encode([r["query"]], normalize_embeddings=True)
        if row["tier"] != "L1":
            for uh, tv in (row.get("ev") or []): e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv))
            ctx = p2.ctx_for(tuple(tuple(x) for x in row["ev"])) if row.get("ev") else ""
            generate(row.get("gen_query", r["query"]), ctx); gens += 1
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1e3)
    tiers = {t: sum(1 for _, r in WIN if rows[r["query_id"]]["tier"] == t) for t in ("L1", "L2", "miss", "REAL_TIME")}
    R[p] = {"requests": len(WIN), "tiers": tiers, "generations": gens, "mean_ms": round(statistics.mean(lat), 1), "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(sorted(lat)[int(.95 * len(lat)) - 1], 1),
            "fetched_pages_in_window": sum(rows[r["query_id"]].get("fetched", 0) for _, r in WIN)}
    print(p, R[p], flush=True)
json.dump({"window": [W0, W1], "regime": "local retrieval (snapshot reads); live search/fetch not executed", "policies": R}, open(O / "outputs/runtime_local.json", "w"), indent=1)
