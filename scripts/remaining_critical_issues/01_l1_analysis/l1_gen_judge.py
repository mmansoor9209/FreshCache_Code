#!/usr/bin/env python3
"""
Section 1 -- GPU stage: answers for every held-out L1 hit, and the
query-equivalence jury.

  gen    Llama-3.2-3B-Instruct, the ORIGINAL audit protocol (FORCED prompt, no
         chat template, greedy, 80 new tokens, 1024-token truncation), for both
         the STORED answer the cache returns and the FRESH answer a no-cache
         path produces at the same timestamp.
  judge  answer correctness against gold, four judges, blinded.
  equiv  query-equivalence jury over (incoming query, cached query), four
         models, identical prompt, greedy, blinded to policy identity.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "06_stronger_generator"))
import gen_judge as G   # noqa: E402

EQUIV = """You are comparing two search queries.

QUERY A: {a}
QUERY B: {b}

Do these two queries ask for the SAME fact, such that a single correct answer
would answer both? Reply with exactly one word: SAME or DIFFERENT."""


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def gen(a):
    import torch
    out = HERE / "l1_answers.jsonl"
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / "l1_gen_tasks.jsonl")
            if tuple(t["key"]) not in done]
    print(f"  {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    model, tok = G.load(G.GENS["llama3b"][0], a.gpu)
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch_size):
            ch = todo[s:s + a.batch_size]
            prompts = [G.FORCED.format(question=c["question"],
                                       context=c["context"]) for c in ch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                fh.write(json.dumps({"key": c["key"],
                                     "answer": G.clip(t) or "I don't know."}) + "\n")
            fh.flush()
            if (s // a.batch_size) % 10 == 0:
                print(f"    {min(s+a.batch_size, len(todo))}/{len(todo)}", flush=True)
    print("  generation done", flush=True)


def _items():
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "l1_answers.jsonl")}
    items, seen = [], set()
    for x in json.load(open(HERE / "l1_hits.json", encoding="utf-8"))["hits"]:
        if not x["gold"]:
            continue
        for key in (x["stored_key"], x["fresh_key"]):
            if not key:
                continue
            a = ans.get(tuple(key))
            if a is None:
                continue
            sig = (G.h(x["incoming_query"]), G.h(x["gold"]), G.h(a))
            if sig in seen:
                continue
            seen.add(sig)
            items.append({"sig": list(sig), "q": x["incoming_query"],
                          "gold": x["gold"], "cand": a})
    return items


def judge(a):
    import torch
    out = HERE / f"l1_judge_{a.judge}.jsonl"
    done = {tuple(r["sig"]) for r in jl(out)}
    todo = [i for i in _items() if tuple(i["sig"]) not in done]
    print(f"  [{a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    model, tok = G.load(G.JUDGES[a.judge], a.gpu)
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch_size):
            ch = todo[s:s + a.batch_size]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": G.JUDGE.format(
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
                fh.write(json.dumps({"sig": c["sig"], "judge": a.judge,
                                     "label": ("INCORRECT" if "INCORRECT" in up
                                               else "CORRECT" if "CORRECT" in up
                                               else "UNPARSED")}) + "\n")
            fh.flush()
    print(f"  [{a.judge}] done", flush=True)


def equiv(a):
    import torch
    out = HERE / f"l1_equiv_{a.judge}.jsonl"
    done = {r["qid"] for r in jl(out)}
    todo = [r for r in jl(HERE / "l1_equiv_tasks.jsonl") if r["qid"] not in done]
    print(f"  [equiv {a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    model, tok = G.load(G.JUDGES[a.judge], a.gpu)
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch_size):
            ch = todo[s:s + a.batch_size]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": EQUIV.format(a=c["incoming"],
                                                          b=c["cached"])}],
                tokenize=False, add_generation_prompt=True) for c in ch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                up = t.upper()
                fh.write(json.dumps({"qid": c["qid"], "judge": a.judge,
                                     "label": ("DIFFERENT" if "DIFFERENT" in up
                                               else "SAME" if "SAME" in up
                                               else "UNPARSED")}) + "\n")
            fh.flush()
    print(f"  [equiv {a.judge}] done", flush=True)


if __name__ == "__main__":
    (HERE / "equiv_prompt_verbatim.txt").write_text(EQUIV, encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("gen", gen), ("judge", judge), ("equiv", equiv)):
        s = sub.add_parser(nm)
        if nm != "gen":
            s.add_argument("--judge", choices=list(G.JUDGES), required=True)
        s.add_argument("--gpu", type=int, default=4)
        s.add_argument("--batch-size", type=int, default=16)
        s.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)
