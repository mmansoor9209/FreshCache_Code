#!/usr/bin/env python3
"""
Issue 2 -- GPU stage: the four-model equivalence jury over every realised L1
hit of BOTH arms, and answer generation + correctness judging.

Jury and prompts are the ones already used in the paper: identical prompt,
greedy decoding, blinded (the judge sees only the two query strings and never
which arm or policy produced the pair). Answers use the original audit protocol
(Llama-3.2-3B, FORCED prompt, no chat template, greedy, 80 new tokens, 1024
truncation); answers already produced by earlier runs are reused byte-for-byte.

  python gen_judge_strict.py gen             --gpu 3
  python gen_judge_strict.py judge --judge qwen7b  --gpu 3
  python gen_judge_strict.py equiv --judge qwen7b  --gpu 3
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parents[1]
RCI = V3 / "remaining_critical_issues"
sys.path.insert(0, str(RCI / "06_stronger_generator"))
sys.path.insert(0, str(RCI / "01_l1_analysis"))
import gen_judge as G            # noqa: E402  (prompts, loaders, JUDGES, clip)
from l1_gen_judge import EQUIV   # noqa: E402  (the same equivalence prompt)

try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-final3")
except Exception:
    pass


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def gen(a):
    import torch
    out = HERE / "answers_llama3b.jsonl"
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / "gen_tasks.jsonl") if tuple(t["key"]) not in done]
    print(f"  {len(done)} reused/done, {len(todo)} to generate", flush=True)
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


def _judge_items():
    """One blinded item per distinct (question, reference, candidate)."""
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "answers_llama3b.jsonl")}
    items, seen = [], set()
    import csv
    for x in csv.DictReader(open(HERE / "l1_hits.csv", encoding="utf-8")):
        if not x["reference_answer"]:
            continue
        for kf in ("stored_key", "fresh_key"):
            if not x[kf]:
                continue
            a = ans.get(tuple(x[kf].split("|")))
            if a is None:
                continue
            sig = (G.h(x["incoming_query"]), G.h(x["reference_answer"]), G.h(a))
            if sig in seen:
                continue
            seen.add(sig)
            items.append({"sig": list(sig), "q": x["incoming_query"],
                          "gold": x["reference_answer"], "cand": a})
    for d in json.load(open(HERE / "audit_sample.json", encoding="utf-8"))["requests"]:
        for kf in ("fresh_key", "published_key", "strict_key"):
            a = ans.get(tuple(d[kf]))
            if a is None:
                continue
            sig = (G.h(d["query"]), G.h(d["gold"]), G.h(a))
            if sig in seen:
                continue
            seen.add(sig)
            items.append({"sig": list(sig), "q": d["query"], "gold": d["gold"],
                          "cand": a})
    return items


def judge(a):
    import torch
    out = HERE / f"judge_{a.judge}.jsonl"
    done = {tuple(r["sig"]) for r in jl(out)}
    todo = [i for i in _judge_items() if tuple(i["sig"]) not in done]
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
    out = HERE / f"equiv_{a.judge}.jsonl"
    done = {r["pair_id"] for r in jl(out)}
    todo = [r for r in jl(HERE / "jury_tasks.jsonl") if r["pair_id"] not in done]
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
                fh.write(json.dumps({"pair_id": c["pair_id"], "judge": a.judge,
                                     "label": ("DIFFERENT" if "DIFFERENT" in up
                                               else "SAME" if "SAME" in up
                                               else "UNPARSED")}) + "\n")
            fh.flush()
    print(f"  [equiv {a.judge}] done", flush=True)


if __name__ == "__main__":
    (HERE / "prompts_verbatim.json").write_text(json.dumps(
        {"generator": G.GENS["llama3b"], "judges": G.JUDGES,
         "forced_prompt": G.FORCED, "correctness_prompt": G.JUDGE,
         "equivalence_prompt": EQUIV,
         "decoding": "greedy, do_sample=False",
         "max_new_tokens": {"generation": 80, "judging": 6},
         "blinding": "judges receive only the query strings / the question, "
                     "reference and candidate answer; never the arm or policy"},
        indent=2), encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("gen", gen), ("judge", judge), ("equiv", equiv)):
        s = sub.add_parser(nm)
        if nm != "gen":
            s.add_argument("--judge", choices=list(G.JUDGES), required=True)
        s.add_argument("--gpu", type=int, default=3)
        s.add_argument("--batch-size", type=int, default=16)
        s.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)
