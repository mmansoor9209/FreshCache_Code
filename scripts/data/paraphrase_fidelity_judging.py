#!/usr/bin/env python3
"""
paraphrase_fidelity_judging.py — does each paraphrase still ask its base
                               query's question?

WHY
  Most realized L1 hits serve an answer to a different question, and most
  of those hits have a PARAPHRASE as the incoming query. Two
  very different causes are currently entangled:

    (1) the cache genuinely matches unrelated queries   -> system failure
    (2) the paraphrase no longer asks its base question -> benchmark failure

  paraphrase_generation.py generates English paraphrases with
  temperature=1.5, top_k=120, repetition_penalty=2.5 and no fidelity
  filter, and Korean paraphrases with templates whose fallbacks
  ("...에 대한 정보를 알려주세요", "현재 {lstrip}", "최신 {lstrip}") can change
  the question. So (2) is plausible and must be measured before deciding
  whether to regenerate.

  This measures (2) directly. High fidelity means the L1 mismatch is mostly a
  real system finding. Low fidelity means much of it is benchmark noise
  and regeneration is justified.

METHOD
  Sample paraphrases, pair each with its base query via base_query_id,
  and ask the SAME judge, prompt and decoding used in
  judge_semantic_pairs.py and judge_l1_hits.py.

  Sampling is restricted to paraphrases that actually appear in the
  experiment record set (base query has run_00 URL coverage), so the
  measured fidelity applies to the population that produced the hits.

  Two strata that fail differently, sampled separately:
    en  -> T5 model path
    ko  -> template path

SAFETY
  Reads queries.jsonl, url_manifest.jsonl, paraphrase_clusters.jsonl and
  (optionally) l1_hits_rulebased_t24h.jsonl. Writes only:
      data/paraphrase_fidelity_judgments.jsonl   (resume-safe)
      outputs/benchmark/paraphrase_fidelity_summary.json

Run:
    cd "$FRESHCACHE_ROOT" && CUDA_VISIBLE_DEVICES=0 python3 paraphrase_fidelity_judging.py --dry-run
    cd "$FRESHCACHE_ROOT" && CUDA_VISIBLE_DEVICES=0 python3 paraphrase_fidelity_judging.py
"""

from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import pipeline

DATA_DIR      = (DATA_ROOT)
QUERIES_FILE  = DATA_DIR / "queries.jsonl"
MANIFEST_FILE = DATA_DIR / "url_manifest.jsonl"
PARA_FILE     = DATA_DIR / "paraphrase_clusters.jsonl"
L1_HITS_FILE  = DATA_DIR / "l1_hits_rulebased_t24h.jsonl"
RESULTS_FILE  = DATA_DIR / "paraphrase_fidelity_judgments.jsonl"
SUMMARY_FILE  = OUTPUT_ROOT / "benchmark/paraphrase_fidelity_summary.json"

MODEL_ID   = "meta-llama/Llama-3.2-3B-Instruct"
GPU_DEVICE = "cuda:0"      # first visible GPU (set CUDA_VISIBLE_DEVICES)
SEED       = 42

N_EN = 250
N_KO = 150

# Identical wording to the other two judge scripts so all three are
# directly comparable. Only the framing sentence differs.
SYSTEM_PROMPT = (
    "You are evaluating a semantic cache for a question-answering system. "
    "You will be given two questions. Decide whether a single correct answer "
    "would satisfy BOTH questions, so that serving the stored answer for "
    "Question A in response to Question B would be correct.\n\n"
    "Rules:\n"
    "- VERDICT: SAME       if both questions ask for the same fact about the "
    "same entity, even if worded very differently or in different languages.\n"
    "- VERDICT: DIFFERENT  if they ask about different entities (e.g. iPhone 16 "
    "vs iPhone 16 Pro), different attributes of the same entity (e.g. the "
    "trainer vs the jockey), different time references, or different events.\n"
    "- VERDICT: DIFFERENT  if Question B is garbled, incoherent, or so vague "
    "that it no longer asks for the specific fact Question A asks for.\n"
    "- When the questions are merely phrased differently but seek the same "
    "fact, answer SAME.\n"
    "- When in doubt, answer DIFFERENT.\n\n"
    "Respond with exactly one line starting with VERDICT: SAME or "
    "VERDICT: DIFFERENT, followed by a single sentence of reasoning "
    "(max 20 words)."
)

USER_TEMPLATE = "Question A: {q1}\nQuestion B: {q2}"


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def parse_verdict(text: str) -> tuple[str, str]:
    verdict, reasoning = "UNKNOWN", ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if "VERDICT:" in upper:
            after = line.split(":", 1)[1].strip() if ":" in line else ""
            au = after.upper()
            if au.startswith("DIFFERENT"):
                verdict = "DIFFERENT"
            elif au.startswith("SAME"):
                verdict = "SAME"
            parts = after.split(None, 1)
            if len(parts) > 1:
                reasoning = parts[1].strip()
        elif verdict != "UNKNOWN" and not reasoning:
            reasoning = line
    return verdict, reasoning


def main(dry_run: bool) -> None:
    print("\n=== Paraphrase fidelity against base query ===\n")

    queries  = load_jsonl(QUERIES_FILE)
    paras    = load_jsonl(PARA_FILE)
    manifest = load_jsonl(MANIFEST_FILE)

    base_by_id = {q["query_id"]: q for q in queries}

    covered = set()
    for m in manifest:
        if m.get("run_id") == "run_00" and m.get("snapshot_available"):
            covered.add(m["query_id"])

    # Restrict to paraphrases actually present in the experiment records.
    pool = []
    for p in paras:
        bid = p.get("base_query_id", "")
        if bid not in covered or bid not in base_by_id:
            continue
        pool.append({
            "para_id":    p["query_id"],
            "base_id":    bid,
            "paraphrase": p["query"],
            "base_query": base_by_id[bid]["query"],
            "language":   p.get("language", "en"),
            "fc":         p.get("freshness_class", ""),
            "source":     p.get("source", ""),
        })

    # Which paraphrases actually produced an L1 hit at t=24h, if available.
    hit_ids = set()
    for h in load_jsonl(L1_HITS_FILE):
        if h.get("is_paraphrase"):
            hit_ids.add(h.get("query_id"))
    for r in pool:
        r["was_l1_hit"] = r["para_id"] in hit_ids

    en = [r for r in pool if r["language"] == "en"]
    ko = [r for r in pool if r["language"] == "ko"]

    print(f"  paraphrases in experiment set : {len(pool):,}")
    print(f"    English (T5 path)           : {len(en):,}")
    print(f"    Korean  (template path)     : {len(ko):,}")
    if hit_ids:
        print(f"    of these, produced an L1 hit: "
              f"{sum(1 for r in pool if r['was_l1_hit']):,}")

    rng = random.Random(SEED)
    sample = (en if len(en) <= N_EN else rng.sample(en, N_EN)) + \
             (ko if len(ko) <= N_KO else rng.sample(ko, N_KO))

    print(f"\n  sampling {len(sample)} "
          f"({min(N_EN, len(en))} en + {min(N_KO, len(ko))} ko)")

    if dry_run:
        print(f"\n  [DRY RUN] Would judge {len(sample)} pairs with {MODEL_ID}.")
        print("\n  Sample pairs:\n")
        for r in sample[:4] + sample[-3:]:
            print(f"    [{r['language']}] {r['source']}")
            print(f"      base : {r['base_query'][:66]}")
            print(f"      para : {r['paraphrase'][:66]}")
            print()
        return

    token_path = Path("~/.cache/huggingface/token").expanduser()
    hf_token   = token_path.read_text().strip() if token_path.exists() else ""
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    print(f"\n  Loading {MODEL_ID} on {GPU_DEVICE}...\n")
    judge = pipeline("text-generation", model=MODEL_ID,
                     torch_dtype=torch.bfloat16,
                     device_map={"": GPU_DEVICE},
                     token=hf_token or None)
    judge.tokenizer.pad_token_id = judge.tokenizer.eos_token_id
    judge.model.generation_config.max_length = None
    print("  Model loaded.\n")

    done      = {r["para_id"] for r in load_jsonl(RESULTS_FILE)}
    remaining = [r for r in sample if r["para_id"] not in done]
    print(f"  Already judged : {len(done)}")
    print(f"  Remaining      : {len(remaining)}\n")

    for i, r in enumerate(remaining, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                q1=r["base_query"], q2=r["paraphrase"])},
        ]
        t0 = time.time()
        try:
            out = judge(messages, max_new_tokens=60, do_sample=False,
                        temperature=None,
                        pad_token_id=judge.tokenizer.eos_token_id)
            raw = out[0]["generated_text"][-1]["content"]
        except Exception as e:
            raw = f"VERDICT: UNKNOWN\nError: {e}"

        verdict, reasoning = parse_verdict(raw)
        rec = dict(r)
        rec.update({"verdict": verdict, "reasoning": reasoning,
                    "raw_output": raw[:300]})
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        mark = "FAITHFUL" if verdict == "SAME" else \
               "CORRUPT " if verdict == "DIFFERENT" else "???     "
        print(f"  [{i:3d}/{len(remaining)}] [{r['language']}] {mark} "
              f"({time.time()-t0:.1f}s)")
        print(f"        base : {r['base_query'][:58]}")
        print(f"        para : {r['paraphrase'][:58]}")

    # ── Summary ───────────────────────────────────────────────────────────
    judged = [r for r in load_jsonl(RESULTS_FILE)
              if r.get("verdict") in ("SAME", "DIFFERENT")]

    def fid(rows: list) -> tuple[int, int, float, float]:
        n = len(rows)
        k = sum(1 for r in rows if r["verdict"] == "SAME")
        p = k / n if n else 0.0
        se = math.sqrt(p * (1 - p) / n) if n else 0.0
        return n, k, p, se

    print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
    print(f"\n  {'stratum':>12} {'n':>5} {'faithful':>9} {'fidelity':>9} "
          f"{'95% CI':>18}")

    summary = {"model": MODEL_ID, "strata": {}}
    for name, rows in (("english", [r for r in judged if r["language"] == "en"]),
                       ("korean",  [r for r in judged if r["language"] == "ko"]),
                       ("ALL",     judged)):
        n, k, p, se = fid(rows)
        if n == 0:
            continue
        lo, hi = max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
        print(f"  {name:>12} {n:>5} {k:>9} {p:>9.3f} "
              f"  [{lo:.3f}, {hi:.3f}]")
        summary["strata"][name] = {
            "n": n, "n_faithful": k, "fidelity": round(p, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    # Fidelity among paraphrases that actually produced an L1 hit.
    hit_rows = [r for r in judged if r.get("was_l1_hit")]
    if hit_rows:
        n, k, p, se = fid(hit_rows)
        print(f"\n  Among paraphrases that produced an L1 hit at t=24h:")
        print(f"    n={n}  faithful={k}  fidelity={p:.3f}")
        summary["among_l1_hits"] = {
            "n": n, "n_faithful": k, "fidelity": round(p, 4)}

    print(f"\n  By freshness class:")
    for fc in ("TIMELESS", "SLOW", "MEDIUM", "FAST"):
        rows = [r for r in judged if r["fc"] == fc]
        if rows:
            n, k, p, _ = fid(rows)
            print(f"    {fc:<10} n={n:<4} fidelity={p:.3f}")
            summary.setdefault("by_class", {})[fc] = {
                "n": n, "fidelity": round(p, 4)}

    print(f"\n  By generator:")
    for src, rows in sorted(
            ((s, [r for r in judged if r["source"] == s])
             for s in {r["source"] for r in judged})):
        if rows:
            n, k, p, _ = fid(rows)
            print(f"    {src:<28} n={n:<4} fidelity={p:.3f}")
            summary.setdefault("by_source", {})[src] = {
                "n": n, "fidelity": round(p, 4)}

    _, _, p_all, _ = fid(judged)
    print(f"\n{'-'*70}")
    print(f"  INTERPRETATION")
    print(f"    Measured paraphrase fidelity : {p_all:.1%}")
    print(f"    Realized L1 mismatch         : see l1_hit_judgment_summary_24h.json")
    if p_all >= 0.85:
        print(f"    -> Paraphrases are largely faithful. The L1 mismatch is")
        print(f"       mostly the cache matching strangers: a system finding.")
        print(f"       Regeneration is NOT indicated.")
    elif p_all >= 0.60:
        print(f"    -> Mixed. A meaningful share of the L1 mismatch traces to")
        print(f"       corrupted paraphrases. Regeneration would change the")
        print(f"       numbers but would not remove the system effect.")
    else:
        print(f"    -> Paraphrases are largely corrupted. Much of the L1 mismatch")
        print(f"       is benchmark noise, not cache behaviour. Regeneration")
        print(f"       with sane decoding is indicated before any conclusion.")
    print(f"{'-'*70}")

    summary["overall_fidelity"] = round(p_all, 4)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_FILE}")
    print(f"  Saved: {SUMMARY_FILE}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(p.parse_args().dry_run)
