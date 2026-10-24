#!/usr/bin/env python3
"""
scripts/evaluation/answer_audits/mixed_age_audit_generation_judging.py

Stage 2 and 3. Protocol is byte-identical to the previous mixed-age audit:
FORCED prompt, Llama-3.2-3B-Instruct, greedy, max_new_tokens=80, 1024-token
truncation, no system prompt, no chat template, answer cut at the first "." or
newline. Judges are Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, greedy,
max_new_tokens=6, chat template, both must agree.

The only change is bookkeeping: generation is keyed by (mode, context hash,
question hash) so an answer shared by several requests or arms is produced once.
Identical inputs give identical greedy output, so deduplication cannot change a
single result.

Run:  python mixed_age_audit_generation_judging.py gen   --gpu 4
      python mixed_age_audit_generation_judging.py judge --judge llama8b --gpu 4
      python mixed_age_audit_generation_judging.py judge --judge qwen7b  --gpu 4
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import hashlib
import json
import os
import pathlib

HERE = ((OUTPUT_ROOT / "mixed_age_audit"))

GEN = "meta-llama/Llama-3.2-3B-Instruct"
JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}

FORCED = """You are a factual question answering assistant.
Answer the following question in one sentence using the provided context.
You must commit to a single best answer. Never reply that you do not know and
never state that the context is insufficient. If the context is incomplete,
give the most likely specific answer that the context points toward.

Question: {question}

Context:
{context}

Answer:"""

JUDGE = """You are grading a short answer against the reference answer.

QUESTION: {q}
REFERENCE ANSWER: {gold}
CANDIDATE ANSWER: {cand}

Does the candidate convey the same factual answer as the reference? Minor
wording, extra context, or added detail do not matter. Reply with exactly one
word: CORRECT or INCORRECT."""


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def hf_token():
    p = pathlib.Path("~/.cache/huggingface/token").expanduser()
    return p.read_text().strip() if p.exists() else None


def load(model_id, gpu):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0", token=hf_token())
    m.eval()
    return m, tok


def gen(args):
    import torch
    out = HERE / "answers2.jsonl"
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / "gen_tasks.jsonl")
            if tuple(t["key"]) not in done]
    print(f"  {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    model, tok = load(GEN, args.gpu)
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), args.batch_size):
            ch = todo[s:s + args.batch_size]
            prompts = [FORCED.format(question=c["question"],
                                     context=c["context"]) for c in ch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                t = t.strip()
                for sep in (".", "\n"):
                    if sep in t:
                        t = t[:t.index(sep) + 1].strip()
                        break
                fh.write(json.dumps({"key": c["key"],
                                     "answer": t or "I don't know."}) + "\n")
            fh.flush()
            if (s // args.batch_size) % 5 == 0:
                print(f"    {min(s + args.batch_size, len(todo))}/{len(todo)}",
                      flush=True)
    print("  generation done", flush=True)


def judge_items():
    """One judge item per (request, arm), deduplicated by content."""
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "answers2.jsonl")}
    items, seen = [], {}
    for d in jl(HERE / "mixed_age_sample.jsonl"):
        fk = ("fresh", d["fresh_ctx_sha"], h(d["query"]))
        for arm, key in (("fresh", fk), ("on", tuple(d["on_key"])),
                         ("off", tuple(d["off_key"]))):
            a = ans.get(tuple(key))
            if a is None:
                continue
            sig = (h(d["query"]), h(d["gold"]), h(a))
            if sig not in seen:
                seen[sig] = {"sig": list(sig), "q": d["query"],
                             "gold": d["gold"], "cand": a}
                items.append(seen[sig])
    return items


def judge(args):
    import torch
    name = args.judge
    out = HERE / f"judge2_{name}.jsonl"
    done = {tuple(r["sig"]) for r in jl(out)}
    todo = [it for it in judge_items() if tuple(it["sig"]) not in done]
    print(f"  [{name}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    model, tok = load(JUDGES[name], args.gpu)
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), args.batch_size):
            ch = todo[s:s + args.batch_size]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": JUDGE.format(
                    q=c["q"], gold=c["gold"], cand=c["cand"])}],
                tokenize=False, add_generation_prompt=True) for c in ch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                up = t.upper()
                lab = ("INCORRECT" if "INCORRECT" in up
                       else ("CORRECT" if "CORRECT" in up else "UNPARSED"))
                fh.write(json.dumps({"sig": c["sig"], "judge": name,
                                     "label": lab}) + "\n")
            fh.flush()
            if (s // args.batch_size) % 5 == 0:
                print(f"    [{name}] "
                      f"{min(s + args.batch_size, len(todo))}/{len(todo)}",
                      flush=True)
    print(f"  [{name}] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--gpu", type=int, default=4)
    g.add_argument("--batch-size", type=int, default=16)
    g.set_defaults(func=gen)
    j = sub.add_parser("judge")
    j.add_argument("--judge", choices=list(JUDGES), required=True)
    j.add_argument("--gpu", type=int, default=4)
    j.add_argument("--batch-size", type=int, default=16)
    j.set_defaults(func=judge)
    a = ap.parse_args()
    a.func(a)
