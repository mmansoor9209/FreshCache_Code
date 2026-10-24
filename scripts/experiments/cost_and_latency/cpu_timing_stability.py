"""Repeat (2 runs) of the CPU micro-benchmarks: brute-force vector search at live index sizes and local snapshot page read; 20% stability rule."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import json, pathlib, random, re, statistics, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = (OUTPUT_ROOT / "systems_timing"); sys.path.insert(0, str(ROOT / "freshcache/replay_engine"))
import numpy as np, engine_loader as C
E = C.load("reported"); p2 = E["p2"]; P = C.streams(E)["primary"]; V = np.asarray(np.load((DATA_ROOT / "query_embeddings_bgem3.npy"), mmap_mode="r"), dtype=np.float32)
def summ(v): v = sorted(v); return {"n": len(v), "mean_ms": statistics.mean(v), "p50_ms": v[len(v) // 2], "p95_ms": v[int(.95 * len(v))]}
pages = random.Random(7).sample([(u["url_hash"], t) for t, r in P for u in r["urls"]], 500)
def read_page(u, tv):
    t0 = time.perf_counter(); pth = (DATA_ROOT / "snapshots") / p2.snapshot_round_on_disk(u, tv) / f"{u}.json"
    txt = (json.load(open(pth, encoding="utf-8")).get("extracted_text") or "") if pth.exists() else ""; re.sub(r"\s+", " ", txt).strip(); return (time.perf_counter() - t0) * 1000
res = {}
for rep in range(2):
    rng = random.Random(42 + rep); d = {}
    for n in (1000, 5000, 10000, 20000, 31000):
        M = V[:n]; ts = []
        for i in rng.sample(range(len(V)), 200):
            t0 = time.perf_counter_ns(); s = M @ V[i]; np.argpartition(-s, 5)[:5]; ts.append((time.perf_counter_ns() - t0) / 1e6)
        d[f"vector_search_{n}"] = summ(ts)
    d["local_page_read"] = summ([read_page(u, t) for u, t in pages]); res[rep] = d
out = {k: {"run1": res[0][k], "run2": res[1][k], "stable": abs(res[0][k]["p50_ms"] - res[1][k]["p50_ms"]) <= .2 * max(res[0][k]["p50_ms"], res[1][k]["p50_ms"])
           and abs(res[0][k]["mean_ms"] - res[1][k]["mean_ms"]) <= .2 * max(res[0][k]["mean_ms"], res[1][k]["mean_ms"])} for k in res[0]}
json.dump(out, open(W / "intermediate/cpu_micro_stability.json", "w"), indent=1); print({k: (round(v["run1"]["p50_ms"], 4), round(v["run2"]["p50_ms"], 4), v["stable"]) for k, v in out.items()})
