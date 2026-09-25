#!/usr/bin/env python3
"""Generation and judging for the parameter-calibration answer audit.

Imports the paper's generator, judges, prompts and decoding settings from
remaining_critical_issues/06_stronger_generator/gen_judge.py verbatim -- the
same module the published held-out audit used. Nothing about the protocol is
restated or altered here; only the file paths differ.

  gen                 Llama-3.2-3B, FORCED prompt, greedy, 80 new tokens
  judge --judge <m>   answer correctness, greedy, 6 new tokens
"""
from __future__ import annotations
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
V3 = PC.parent
RCI = V3 / "remaining_critical_issues"
sys.path.insert(0, str(RCI / "06_stronger_generator"))
import gen_judge as G            # noqa: E402

try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-paramcal")
except Exception:
    pass

GEN_TASKS = HERE / "gen_tasks.jsonl"
ANSWERS = HERE / "answers_llama3b.jsonl"
SAMPLE = HERE / "audit_sample.json"
CACHED = HERE / "cached_answers.jsonl"
JUDGE_OUT = HERE / "judge_{j}.jsonl"
# every judgment store produced by earlier workstreams, reused by signature
PRIOR_JUDGES = [
    V3 / "l1_precision_gate" / "heldout_judge_{j}.jsonl",
    V3 / "l1_precision_gate" / "validation_judge_{j}.jsonl",
    V3 / "final_three_issues" / "02_l1_strict_gate" / "judge_{j}.jsonl",
    RCI / "06_stronger_generator" / "judge_{j}_llama8b_forced.jsonl",
]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def all_answers():
    a = {tuple(r["key"]): r["answer"] for r in jl(CACHED)}
    a.update({tuple(r["key"]): r["answer"] for r in jl(ANSWERS)})
    return a


def gen(a):
    import torch
    done = {tuple(r["key"]) for r in jl(ANSWERS)} | \
           {tuple(r["key"]) for r in jl(CACHED)}
    todo = [t for t in jl(GEN_TASKS) if tuple(t["key"]) not in done]
    print(f"  [gen] {len(done)} already available, {len(todo)} to generate",
          flush=True)
    if not todo:
        return
    model, tok = G.load(G.GENS["llama3b"][0], a.gpu)
    with open(ANSWERS, "a", encoding="utf-8") as fh:
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
            print(f"    {min(s+a.batch_size, len(todo))}/{len(todo)}", flush=True)
    print(f"  [gen] done", flush=True)


def _items():
    ans = all_answers()
    items, seen = [], set()
    S = json.load(open(SAMPLE, encoding="utf-8"))
    for d in S["requests"]:
        for k in d:
            if not k.endswith("_key"):
                continue
            v = ans.get(tuple(d[k]))
            if v is None:
                continue
            sig = (G.h(d["query"]), G.h(d["gold"]), G.h(v))
            if sig in seen:
                continue
            seen.add(sig)
            items.append({"sig": list(sig), "q": d["query"], "gold": d["gold"],
                          "cand": v})
    return items


def judge(a):
    import torch
    out = pathlib.Path(str(JUDGE_OUT).format(j=a.judge))
    done = {tuple(r["sig"]) for r in jl(out)}
    prior = {}
    for pat in PRIOR_JUDGES:
        for r in jl(str(pat).format(j=a.judge)):
            prior[tuple(r["sig"])] = r["label"]
    items = _items()
    reused = 0
    with open(out, "a", encoding="utf-8") as fh:
        for i in items:
            s = tuple(i["sig"])
            if s in done:
                continue
            if s in prior:
                fh.write(json.dumps({"sig": i["sig"], "judge": a.judge,
                                     "label": prior[s],
                                     "reused_from_prior_run": True}) + "\n")
                done.add(s); reused += 1
    todo = [i for i in items if tuple(i["sig"]) not in done]
    print(f"  [judge {a.judge}] {len(items)} items; reused {reused} prior "
          f"judgments byte-for-byte; {len(todo)} to judge", flush=True)
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
            print(f"    {min(s+a.batch_size, len(todo))}/{len(todo)}", flush=True)
    print(f"  [judge {a.judge}] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "judge"])
    ap.add_argument("--judge", default="llama8b")
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    (gen if a.stage == "gen" else judge)(a)
