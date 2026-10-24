"""
answer_grading.py: end-to-end answer grading of cache hits.

For a stratified sample of FreshCache-Bench queries (up to 150 per
freshness class), generates answers from:
  1. Cached evidence (run_00 snapshot content), simulating a cache hit
  2. Fresh evidence (rerun_24h snapshot content), simulating NoCache

Then grades whether the cached answer matches the fresh answer using
exact/substring match and Llama-3.2-3B-Instruct as judge.

Reports:
  - Per-class correlation between hash-diff staleness and answer-level staleness
  - Cache-induced answer error rate per class
  - Overall answer accuracy of cache hits

Run:
    CUDA_VISIBLE_DEVICES=0 python3 answer_grading.py
"""

from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_DIR    = (DATA_ROOT)
SNAP_DIR    = DATA_DIR / "snapshots"
OUT_RESULTS = DATA_DIR / "e2e_grading_results.json"
OUT_SUMMARY = DATA_DIR / "e2e_grading_summary.json"

SAMPLES_PER_CLASS = 50   # 50 per class = 250 total, manageable runtime
CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]
MAX_CONTEXT_CHARS = 2000
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SEED = 42

ANSWER_PROMPT = """You are a factual question answering assistant.
Answer the following question in one sentence using only the provided context.
If the context does not contain enough information, say "I don't know."

Question: {question}

Context:
{context}

Answer:"""

JUDGE_PROMPT = """You are evaluating whether two answers to the same question are factually equivalent.

Question: {question}
Answer A (cached): {answer_a}
Answer B (fresh): {answer_b}

Are these two answers factually equivalent? Reply with YES if they convey the same factual information, or NO if they differ in a meaningful way. Reply with only YES or NO.

Response:"""


def load_snapshot_text(url_hash: str, run_id: str) -> str:
    snap_path = SNAP_DIR / run_id / f"{url_hash}.json"
    if not snap_path.exists():
        return ""
    try:
        data = json.load(open(snap_path))
        return (data.get("extracted_text") or "")[:MAX_CONTEXT_CHARS]
    except Exception:
        return ""


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answers_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if len(nb) <= 40 and nb in na:
        return True
    if len(na) <= 40 and na in nb:
        return True
    return False


def generate_answer(model, tokenizer, question: str, context: str,
                    device: str, max_new_tokens: int = 80) -> str:
    if not context.strip():
        return "I don't know."
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Take only first sentence
    for sep in [".", "\n"]:
        if sep in text:
            text = text[:text.index(sep)+1].strip()
            break
    return text or "I don't know."


def judge_answers(model, tokenizer, question: str,
                  ans_a: str, ans_b: str, device: str) -> bool:
    prompt = JUDGE_PROMPT.format(
        question=question, answer_a=ans_a, answer_b=ans_b)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip().upper()
    return text.startswith("YES")


def main():
    random.seed(SEED)

    print("Loading manifest and change log...")
    manifest = [json.loads(l) for l in open(DATA_DIR / "url_manifest.jsonl")]
    changes  = [json.loads(l) for l in open(DATA_DIR / "change_log.jsonl")]
    queries  = {json.loads(l)["query_id"]: json.loads(l)
                for l in open(DATA_DIR / "queries.jsonl")}

    # Build stale set at 24h
    stale_24h = {c["url_hash"] for c in changes
                 if c.get("run_id") == "rerun_24h"
                 and not c.get("noise", False)}

    # Build URL map: url_hash -> {run_00: text, rerun_24h: text}
    run00_map  = {}
    run24_map  = {}
    for m in manifest:
        if m.get("run_id") == "run_00" and m.get("snapshot_available"):
            run00_map[m["url_hash"]] = m
        elif m.get("run_id") == "rerun_24h" and m.get("snapshot_available"):
            run24_map[m["url_hash"]] = m

    # Build per-class samples: need queries with both snapshots available
    # Group by query_id -> list of url_hashes with both snapshots
    query_urls = defaultdict(list)
    for m in manifest:
        if m.get("run_id") == "run_00" and m.get("snapshot_available"):
            uh = m["url_hash"]
            if uh in run24_map:
                query_urls[m["query_id"]].append(uh)

    # Stratify by freshness class
    by_class = defaultdict(list)
    for qid, url_hashes in query_urls.items():
        if qid not in queries or not url_hashes:
            continue
        fc = queries[qid]["freshness_class"]
        by_class[fc].append((qid, url_hashes))

    # Sample
    sample = []
    for fc in CLASSES:
        pool = by_class[fc]
        random.shuffle(pool)
        selected = pool[:SAMPLES_PER_CLASS]
        for qid, url_hashes in selected:
            sample.append({
                "query_id":        qid,
                "query":           queries[qid]["query"],
                "freshness_class": fc,
                "url_hashes":      url_hashes,
            })

    print(f"Total samples: {len(sample)}")
    for fc in CLASSES:
        n = sum(1 for s in sample if s["freshness_class"] == fc)
        print(f"  {fc}: {n}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()
    print(f"Model loaded on {device}")

    # Run scripts/evaluation
    results = []
    n_total = len(sample)

    print(f"\nGenerating answers for {n_total} queries...")

    for i, s in enumerate(sample):
        q         = s["query"]
        fc        = s["freshness_class"]
        url_hashes = s["url_hashes"]

        # Aggregate context from available URLs
        cached_ctx = " ".join(
            load_snapshot_text(uh, "run_00") for uh in url_hashes
        )[:MAX_CONTEXT_CHARS]
        fresh_ctx = " ".join(
            load_snapshot_text(uh, "rerun_24h") for uh in url_hashes
        )[:MAX_CONTEXT_CHARS]

        # Check if any URL changed (hash-based staleness)
        hash_stale = any(uh in stale_24h for uh in url_hashes)

        # Generate answers
        cached_ans = generate_answer(model, tokenizer, q, cached_ctx, device)
        fresh_ans  = generate_answer(model, tokenizer, q, fresh_ctx,  device)

        # Grade: exact/substring match first, then judge
        exact_match = answers_match(cached_ans, fresh_ans)
        if exact_match:
            answers_equiv = True
        else:
            answers_equiv = judge_answers(
                model, tokenizer, q, cached_ans, fresh_ans, device)

        # Cache-induced error: cache answer differs from fresh (ground truth)
        cache_induced_error = not answers_equiv

        results.append({
            "query_id":           s["query_id"],
            "query":              q,
            "freshness_class":    fc,
            "hash_stale":         hash_stale,
            "cached_answer":      cached_ans,
            "fresh_answer":       fresh_ans,
            "exact_match":        exact_match,
            "answers_equivalent": answers_equiv,
            "cache_induced_error": cache_induced_error,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_total}] {fc}: hash_stale={hash_stale} "
                  f"cache_error={cache_induced_error} | "
                  f"cached='{cached_ans[:50]}' fresh='{fresh_ans[:50]}'")

    # Compute summary
    print("\n" + "="*75)
    print("  End-to-End Answer Grading Results")
    print("  Cache hit = cached evidence; Fresh = rerun_24h evidence")
    print("="*75)

    print(f"\n  {'Class':<12} {'N':>5} {'Hash%':>7} {'CacheErr%':>10} "
          f"{'AnsAcc%':>9} {'Correlation':>12}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*10} {'-'*9} {'-'*12}")

    summary_rows = []
    for fc in CLASSES:
        fc_rows = [r for r in results if r["freshness_class"] == fc]
        if not fc_rows:
            continue
        n            = len(fc_rows)
        n_hash       = sum(1 for r in fc_rows if r["hash_stale"])
        n_err        = sum(1 for r in fc_rows if r["cache_induced_error"])
        n_correct    = n - n_err
        hash_rate    = n_hash / n * 100
        err_rate     = n_err  / n * 100
        acc_rate     = n_correct / n * 100

        # Correlation: among hash-stale, what fraction has cache error
        hash_stale_rows = [r for r in fc_rows if r["hash_stale"]]
        corr = (sum(1 for r in hash_stale_rows if r["cache_induced_error"])
                / max(len(hash_stale_rows), 1) * 100)

        print(f"  {fc:<12} {n:>5} {hash_rate:>6.1f}% {err_rate:>9.1f}% "
              f"{acc_rate:>8.1f}% {corr:>11.1f}%")

        summary_rows.append({
            "freshness_class":     fc,
            "n":                   n,
            "hash_stale_rate":     round(hash_rate/100, 4),
            "cache_induced_error_rate": round(err_rate/100, 4),
            "answer_accuracy":     round(acc_rate/100, 4),
            "hash_to_answer_corr": round(corr/100, 4),
        })

    # Overall
    n_total_r  = len(results)
    n_hash_all = sum(1 for r in results if r["hash_stale"])
    n_err_all  = sum(1 for r in results if r["cache_induced_error"])
    hash_stale_all = [r for r in results if r["hash_stale"]]
    corr_all = (sum(1 for r in hash_stale_all if r["cache_induced_error"])
                / max(len(hash_stale_all), 1) * 100)

    print(f"  {'ALL':<12} {n_total_r:>5} "
          f"{n_hash_all/n_total_r*100:>6.1f}% "
          f"{n_err_all/n_total_r*100:>9.1f}% "
          f"{(n_total_r-n_err_all)/n_total_r*100:>8.1f}% "
          f"{corr_all:>11.1f}%")
    print("="*75)

    # Save
    with open(OUT_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_SUMMARY, "w") as f:
        json.dump({
            "per_class": summary_rows,
            "overall": {
                "n": n_total_r,
                "hash_stale_rate":     round(n_hash_all/n_total_r, 4),
                "cache_induced_error_rate": round(n_err_all/n_total_r, 4),
                "answer_accuracy":     round((n_total_r-n_err_all)/n_total_r, 4),
                "hash_to_answer_corr": round(corr_all/100, 4),
            }
        }, f, indent=2)

    print(f"\n  Saved: {OUT_RESULTS}")
    print(f"  Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
