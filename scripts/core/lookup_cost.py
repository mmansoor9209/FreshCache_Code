#!/usr/bin/env python3
"""
lookup_cost.py — what an L1 lookup costs in a deployment (reviewer C12)

The simulator charges 5ms for an L1 lookup because it reads a precomputed
31,201 x 31,201 similarity matrix. No deployment has that matrix. The real
path is: embed the incoming query, search an approximate nearest-neighbour
index over the cache, extract named entities for the entity guard, and
evaluate the lexical conditions of semantic_equivalent(). This measures
each stage.

WHAT IS MEASURED
  encode    one query through BAAI/bge-m3, batch size 1, which is what a
            serving path does; a batched figure is reported alongside for
            reference, since throughput-oriented deployments would batch
  ann       FAISS inner-product search over a normalised index, at three
            cache sizes, since the deployed cache holds 1,164 entries at
            t=24h and that is too small to characterise scaling
  entity    spaCy NER on the incoming query, cached per query text, plus
            the whole-span comparison against the top candidate
  lexical   content-token Jaccard, answer-type agreement and the subset
            guard against the top candidate

WHAT THIS IS NOT
  A single number for "the" lookup cost. It is one hardware configuration
  (one A6000, local model, in-process index). A deployment using CPU
  inference or a hosted embedding endpoint would differ, and the paper
  should present this as a lower bound on this configuration rather than
  as a general figure.

Reads existing embeddings and query text. Writes only
data/lookup_cost_results.json.

Run:
    cd ~/freshcache && CUDA_VISIBLE_DEVICES=4 python3 lookup_cost.py
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import experiment as exp

DATA_DIR     = Path("data")
EMB_FILE     = DATA_DIR / "query_embeddings_bgem3.npy"
RESULTS_FILE = DATA_DIR / "lookup_cost_results.json"

INDEX_SIZES = [1_000, 10_000, 30_000]
N_TRIALS    = 300      # queries timed per configuration
N_WARMUP    = 20
TOP_K       = 5
SEED        = 42


def pct(xs: list, p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))] * 1000.0   # to ms


def main() -> None:
    rng = random.Random(SEED)
    print(f"\n{'='*76}")
    print(f"  C12: L1 lookup cost, per stage")
    print(f"{'='*76}\n", flush=True)

    records = exp.build_query_records(
        exp.load_jsonl(exp.QUERIES_FILE),
        exp.load_jsonl(exp.MANIFEST_FILE),
        exp.load_jsonl(exp.PARAPHRASE_FILE)
        if exp.PARAPHRASE_FILE.exists() else [])
    texts = [r["query"] for r in records]

    E = np.load(str(EMB_FILE)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    print(f"  records    : {len(records):,}")
    print(f"  embeddings : {E.shape}\n", flush=True)

    results = {"n_trials": N_TRIALS, "top_k": TOP_K}

    # ── Stage 1: encoding ─────────────────────────────────────────────
    print("  [1] encoding a single query with BAAI/bge-m3 ...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")

    sample_idx = rng.sample(range(len(texts)), N_TRIALS + N_WARMUP)
    warm = [texts[i] for i in sample_idx[:N_WARMUP]]
    trial = [texts[i] for i in sample_idx[N_WARMUP:]]

    for t in warm:
        model.encode([t], show_progress_bar=False)

    enc_times = []
    for t in trial:
        t0 = time.perf_counter()
        q = model.encode([t], show_progress_bar=False,
                         convert_to_numpy=True).astype(np.float32)
        enc_times.append(time.perf_counter() - t0)
    q_vecs = model.encode(trial, batch_size=64, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)

    t0 = time.perf_counter()
    model.encode(trial, batch_size=64, show_progress_bar=False)
    batched_per_query = (time.perf_counter() - t0) / len(trial)

    print(f"      p50 {pct(enc_times,0.50):>7.2f} ms   "
          f"p95 {pct(enc_times,0.95):>7.2f} ms   "
          f"(batched: {batched_per_query*1000:.2f} ms/query)")
    results["encode"] = {
        "p50_ms": round(pct(enc_times, 0.50), 3),
        "p95_ms": round(pct(enc_times, 0.95), 3),
        "batched_ms_per_query": round(batched_per_query * 1000, 3),
    }

    # ── Stage 2: ANN search ───────────────────────────────────────────
    print("\n  [2] FAISS inner-product search over the cache", flush=True)
    import faiss

    results["ann"] = {}
    print(f"      {'entries':>9} {'p50 (ms)':>10} {'p95 (ms)':>10} "
          f"{'build (s)':>10}")
    top_ids = {}
    for n in INDEX_SIZES:
        if n > len(E):
            continue
        sub = E[:n]
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(sub.shape[1])
        index.add(sub)
        build = time.perf_counter() - t0

        for i in range(min(N_WARMUP, len(q_vecs))):
            index.search(q_vecs[i:i+1], TOP_K)

        ann_times, ids = [], []
        for i in range(len(q_vecs)):
            t0 = time.perf_counter()
            _, I = index.search(q_vecs[i:i+1], TOP_K)
            ann_times.append(time.perf_counter() - t0)
            ids.append(int(I[0][0]))
        top_ids[n] = ids

        print(f"      {n:>9,} {pct(ann_times,0.50):>10.3f} "
              f"{pct(ann_times,0.95):>10.3f} {build:>10.2f}")
        results["ann"][str(n)] = {
            "p50_ms": round(pct(ann_times, 0.50), 4),
            "p95_ms": round(pct(ann_times, 0.95), 4),
            "build_s": round(build, 3),
        }
        del index

    # ── Stage 3: entity extraction and guard ──────────────────────────
    print("\n  [3] spaCy NER and the entity guard, against the top "
          "candidate", flush=True)
    cand = [texts[i] for i in top_ids[max(top_ids)]]

    ent_times = []
    for a, b in zip(trial[:N_WARMUP], cand[:N_WARMUP]):
        exp._entity_match(a, b)
    exp._ENTITY_CACHE.clear()       # time a cold path, as a first request is

    for a, b in zip(trial, cand):
        t0 = time.perf_counter()
        exp._entity_match(a, b)
        ent_times.append(time.perf_counter() - t0)
        exp._ENTITY_CACHE.pop(a, None)   # the incoming query is always new

    print(f"      p50 {pct(ent_times,0.50):>7.2f} ms   "
          f"p95 {pct(ent_times,0.95):>7.2f} ms")
    results["entity"] = {"p50_ms": round(pct(ent_times, 0.50), 3),
                         "p95_ms": round(pct(ent_times, 0.95), 3)}

    # ── Stage 4: lexical conditions ───────────────────────────────────
    print("\n  [4] lexical conditions of semantic_equivalent()", flush=True)
    lex_times = []
    for a, b in zip(trial, cand):
        exp._EQ_TOKEN_CACHE.pop(a, None)
        exp._EQ_ATYPE_CACHE.pop(a, None)
        t0 = time.perf_counter()
        exp.semantic_equivalent(a, b, 0.85)   # above the floor, so all
        lex_times.append(time.perf_counter() - t0)  # conditions execute

    print(f"      p50 {pct(lex_times,0.50):>7.3f} ms   "
          f"p95 {pct(lex_times,0.95):>7.3f} ms")
    results["lexical"] = {"p50_ms": round(pct(lex_times, 0.50), 4),
                          "p95_ms": round(pct(lex_times, 0.95), 4)}

    # ── Total ─────────────────────────────────────────────────────────
    n_big = max(top_ids)
    tot_p50 = (results["encode"]["p50_ms"]
               + results["ann"][str(n_big)]["p50_ms"]
               + results["entity"]["p50_ms"]
               + results["lexical"]["p50_ms"])
    tot_p95 = (results["encode"]["p95_ms"]
               + results["ann"][str(n_big)]["p95_ms"]
               + results["entity"]["p95_ms"]
               + results["lexical"]["p95_ms"])

    print(f"\n{'-'*76}")
    print(f"  End-to-end L1 lookup, {n_big:,}-entry index")
    print(f"    p50 {tot_p50:>7.2f} ms      p95 {tot_p95:>7.2f} ms")
    print(f"    simulator charges {exp.LATENCY['l1_lookup']:.0f} ms")
    print(f"\n  Against the {exp.LATENCY['search_api']:.0f} ms search call and "
          f"{exp.LATENCY['llm_generate']:.0f} ms generation an L1 hit avoids,")
    print(f"  the lookup is {100*tot_p50/(exp.LATENCY['search_api']+exp.LATENCY['llm_generate']):.1f}% "
          f"of what it saves at the median.")
    print(f"{'-'*76}")

    results["total"] = {"index_entries": n_big,
                        "p50_ms": round(tot_p50, 2),
                        "p95_ms": round(tot_p95, 2),
                        "simulator_charge_ms": exp.LATENCY["l1_lookup"]}

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
