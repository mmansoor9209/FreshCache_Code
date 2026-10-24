"""GPU timing stability: embedding (200 queries) and generation (100 prompts), same fixed inputs as component_timing_gpu.py (seed 42), run twice.
STABLE if the two runs' p50 and mean agree within 20%; otherwise the component is reported as MISSING (unstable)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import ast, json, os, pathlib, random, statistics, subprocess, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True; os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); CRR = (OUTPUT_ROOT / "replay"); W = (OUTPUT_ROOT / "systems_timing")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C, torch
E = C.load("reported"); p2 = E["p2"]; P = C.streams(E)["primary"]; rng = random.Random(42)
nonrt = [r for _, r in P if r["freshness_class"] != "REAL_TIME"]; qs = [r["query"] for r in rng.sample(nonrt, 200)]
import numpy as np; V = np.load((DATA_ROOT / "query_embeddings_bgem3.npy"), mmap_mode="r")
for nidx in (1000, 5000, 10000, 20000, 31000): rng.sample(range(len(V)), 200)          # advance RNG exactly as in component_timing_gpu.py
rng.sample([(u["url_hash"], t) for t, r in P for u in r["urls"]], 500)
gen_items = rng.sample([(t, r) for t, r in P if r["freshness_class"] != "REAL_TIME"], 100)
from sentence_transformers import SentenceTransformer; emb = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
RUN = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open((ROOT / "scripts/evaluation/generation_judging_stage.py")).read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("FORCED", "MODELS")}
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(RUN["MODELS"]["llama3b"]); tok.pad_token = tok.pad_token or tok.eos_token
gm = AutoModelForCausalLM.from_pretrained(RUN["MODELS"]["llama3b"], dtype=torch.bfloat16, device_map="cuda:0"); gm.eval()
ctxs = [p2.ctx_for(tuple((u["url_hash"], t) for u in r["urls"])) for t, r in gen_items]
def summ(v): v = sorted(v); return {"n": len(v), "mean_ms": statistics.mean(v), "p50_ms": v[len(v) // 2], "p95_ms": v[int(.95 * len(v))]}
def gpu_state(): return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
runs = {"embedding": [], "generation": []}; states = []
for rep in range(2):
    states.append(gpu_state())
    for q in qs[:5]: emb.encode([q], show_progress_bar=False)
    e = []
    for q in qs:
        torch.cuda.synchronize(); t0 = time.perf_counter(); emb.encode([q], show_progress_bar=False, normalize_embeddings=True); torch.cuda.synchronize(); e.append((time.perf_counter() - t0) * 1000)
    g = []
    for i, ((t, r), ctx) in enumerate(zip(gen_items[:2] + gen_items, ctxs[:2] + ctxs)):
        enc = tok([RUN["FORCED"].format(question=r["query"], context=ctx)], return_tensors="pt", truncation=True, max_length=1024).to(gm.device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): gm.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        if i >= 2: g.append((time.perf_counter() - t0) * 1000)
    runs["embedding"].append(summ(e)); runs["generation"].append(summ(g)); print(rep, runs["embedding"][-1]["p50_ms"], runs["generation"][-1]["p50_ms"], flush=True)
states.append(gpu_state())
res = {k: {"run1": v[0], "run2": v[1], "stable": abs(v[0]["p50_ms"] - v[1]["p50_ms"]) <= .2 * max(v[0]["p50_ms"], v[1]["p50_ms"]) and abs(v[0]["mean_ms"] - v[1]["mean_ms"]) <= .2 * max(v[0]["mean_ms"], v[1]["mean_ms"])} for k, v in runs.items()}
json.dump({"components": res, "nvidia_smi_snapshots": states}, open(W / "intermediate/gpu_timing_stability.json", "w"), indent=1)
print({k: v["stable"] for k, v in res.items()}); print(states)
