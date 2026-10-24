#!/usr/bin/env python3
"""
scripts/evaluation/answer_audits/semanticttl_audit_generation_judging.py

Generation and judging stages of the SemanticTTL answer audit. The protocol is
byte-identical to the mixed-age audit (mixed_age_audit_generation_judging.py):
FORCED prompt, Llama-3.2-3B-Instruct, greedy, max_new_tokens=80, 1024-token
truncation, no system prompt, no chat template, answer cut at the first "." or
newline. Judges Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, greedy,
max_new_tokens=6, chat template, 2048-token truncation, both must agree.

Answers and verdicts already produced by the mixed-age audit
(outputs/mixed_age_audit/) are reused, never recomputed: identical greedy inputs
give identical outputs, so reuse cannot change a result.

Run:  python semanticttl_audit_generation_judging.py gen   --gpu 4
      python semanticttl_audit_generation_judging.py judge --judge llama8b --gpu 4
      python semanticttl_audit_generation_judging.py judge --judge qwen7b  --gpu 4
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json
import os
import pathlib
import sys

HERE = ((OUTPUT_ROOT / "semanticttl_audit"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
CORR = (OUTPUT_ROOT / "mixed_age_audit")
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))
import mixed_age_audit_generation_judging as G                                    # noqa: E402

GEN, JUDGES, FORCED, JUDGE = G.GEN, G.JUDGES, G.FORCED, G.JUDGE
h, jl, load = G.h, G.jl, G.load


def all_answers():
    """Mixed-age audit answers plus any answers produced by this script."""
    a = {tuple(r["key"]): r["answer"] for r in jl(CORR / "answers2.jsonl")}
    a.update({tuple(r["key"]): r["answer"]
              for r in jl(HERE / "sttl_answers.jsonl")})
    return a


def gen(args):
    import torch
    out = HERE / "sttl_answers.jsonl"
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / "sttl_gen_tasks.jsonl")
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
    """One item per (request, arm), deduplicated by (query, gold, answer)."""
    ans = all_answers()
    items, seen = [], set()
    for d in jl(HERE / "sttl_mixed_age_sample.jsonl"):
        fk = ("fresh", d["fresh_ctx_sha"], h(d["query"]))
        for key in (fk, tuple(d["fc_key"]), tuple(d["sttl_key"])):
            a = ans.get(tuple(key))
            if a is None:
                continue
            sig = (h(d["query"]), h(d["gold"]), h(a))
            if sig in seen:
                continue
            seen.add(sig)
            items.append({"sig": list(sig), "q": d["query"],
                          "gold": d["gold"], "cand": a})
    return items


def judge(args):
    import torch
    name = args.judge
    out = HERE / f"sttl_judge_{name}.jsonl"
    # reuse verdicts already produced by the mixed-age audit
    prior = {tuple(r["sig"]) for r in jl(CORR / f"judge2_{name}.jsonl")}
    done = {tuple(r["sig"]) for r in jl(out)} | prior
    todo = [it for it in judge_items() if tuple(it["sig"]) not in done]
    print(f"  [{name}] {len(prior)} reused from the reported audit, "
          f"{len(done)-len(prior)} already done here, {len(todo)} to judge",
          flush=True)
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
