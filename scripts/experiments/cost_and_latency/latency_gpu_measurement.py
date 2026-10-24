#!/usr/bin/env python3
"""
Stage B (GPU half): embedding and LLM generation, real wall-clock.

Run in its own process: FAISS and torch-CUDA in one process segfault on the
measurement machine, so the CPU-side measurements live in a separate stage.

  embed       BAAI/bge-m3, batch size 1 (the serving path), GPU
  ann         FAISS inner-product search over a normalised index, at several
              index sizes, so each request can be charged at its own live
              index size instead of one average
  entity      spaCy NER + whole-span comparison (FreshCache's L1 entity guard)
  lexical     content-token Jaccard, answer-type agreement, subset guard
  l3_lookup   snapshot read + decode + exponential temporal gate, per page
  generate    Llama-3.2-3B-Instruct, batch size 1, the audit's FORCED prompt on
              real stored contexts, greedy, max_new_tokens=80

Methodology: warm up, then time each call individually with
time.perf_counter and report p50/p95.

No web calls here (stage D does those). Writes only into this directory.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, os, pathlib, random, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()

# CUDA_VISIBLE_DEVICES must be set BEFORE importing experiment/torch: importing
# experiment loads LearnedRiskModel onto CUDA, which initialises the device
# ordering. Setting it inside main() is too late and silently lands on GPU 0.
_gpu = None
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _gpu = sys.argv[_i + 1]
if _gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

HERE = ((OUTPUT_ROOT / "latency_e2e"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402

N_TRIALS, N_WARM = 200, 20
ANN_SIZES = [1000, 5000, 10000, 20000, 31000]
SEED = 42


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] * 1000.0


def stats(v):
    return {"n": len(v), "mean_ms": float(np.mean(v)) * 1000,
            "p50_ms": pct(v, 0.50), "p95_ms": pct(v, 0.95),
            "min_ms": min(v) * 1000, "max_ms": max(v) * 1000,
            "samples_ms": [round(x * 1000, 4) for x in v]}


def main(gpu):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    rng = random.Random(SEED)
    out = {"n_trials": N_TRIALS, "seed": SEED, "gpu": gpu,
           "host": os.environ.get("FRESHCACHE_HOST_LABEL", "local")}

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    texts = [r["query"] for r in records]
    idx = list(range(len(texts))); rng.shuffle(idx)
    trial = [texts[i] for i in idx[:N_TRIALS]]
    warm = [texts[i] for i in idx[N_TRIALS:N_TRIALS + N_WARM]]
    print(f"  {len(records):,} records; {N_TRIALS} timed trials", flush=True)

    # ---------- 1. query embedding ----------
    print("  [1] embedding, BAAI/bge-m3, batch size 1 ...", flush=True)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-m3")
    for t in warm:
        emb.encode([t], show_progress_bar=False)
    et = []
    for t in trial:
        t0 = time.perf_counter()
        emb.encode([t], show_progress_bar=False, normalize_embeddings=True)
        et.append(time.perf_counter() - t0)
    out["embed"] = stats(et)
    print(f"      p50 {out['embed']['p50_ms']:.2f} ms  "
          f"p95 {out['embed']['p95_ms']:.2f} ms", flush=True)

    # ---------- 6. LLM generation ----------
    print("  [6] generation, Llama-3.2-3B-Instruct, batch size 1 ...",
          flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import matched_answer_quality as MAQ
    samp = [json.loads(l) for l in
            open((OUTPUT_ROOT / "mixed_age_audit/mixed_age_sample.jsonl"), encoding="utf-8") if l.strip()]
    rng.shuffle(samp)
    gsub = samp[:120]
    tok = AutoTokenizer.from_pretrained(MAQ.GEN)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    gm = AutoModelForCausalLM.from_pretrained(MAQ.GEN, dtype=torch.bfloat16,
                                              device_map="cuda:0")
    gm.eval()
    prompts = [MAQ.FORCED.format(question=d["query"],
                                context=d["fresh_ctx"][:MAQ.MAXCTX])
               for d in gsub]
    for p in prompts[:5]:
        enc = tok([p], return_tensors="pt", truncation=True,
                  max_length=1024).to(gm.device)
        with torch.no_grad():
            gm.generate(**enc, max_new_tokens=80, do_sample=False,
                        pad_token_id=tok.pad_token_id)
    gen, ntok = [], []
    for p in prompts:
        enc = tok([p], return_tensors="pt", truncation=True,
                  max_length=1024).to(gm.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o = gm.generate(**enc, max_new_tokens=80, do_sample=False,
                            pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        gen.append(time.perf_counter() - t0)
        ntok.append(int(o.shape[1] - enc["input_ids"].shape[1]))
    out["generate"] = {**stats(gen), "mean_new_tokens": float(np.mean(ntok)),
                       "model": MAQ.GEN, "max_new_tokens": 80,
                       "batch_size": 1, "decoding": "greedy"}
    print(f"      p50 {out['generate']['p50_ms']:.1f} ms  "
          f"p95 {out['generate']['p95_ms']:.1f} ms  "
          f"mean new tokens {out['generate']['mean_new_tokens']:.1f}", flush=True)

    json.dump(out, open(HERE / "stage_b_gpu_timings.json", "w"), indent=2)
    print("  wrote stage_b_gpu_timings.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    main(ap.parse_args().gpu)
