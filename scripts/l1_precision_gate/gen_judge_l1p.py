#!/usr/bin/env python3
"""
GPU stage for the L1-Precision gate.

Reuses the paper's four-model equivalence jury and the paper's answer
generation/judging protocol verbatim, by importing them from the existing
experiment code rather than restating them.

  equiv  --set validation|heldout --judge <m>   four-model equivalence jury
  gen    --set validation|heldout               Llama-3.2-3B, FORCED prompt
  judge  --set validation|heldout --judge <m>   answer correctness

Judges are blinded: an equivalence item carries only the two query strings; a
correctness item carries only question / reference / candidate. Neither ever
sees the policy, the cell or the arm.
"""
from __future__ import annotations
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
RCI = V3 / "remaining_critical_issues"
sys.path.insert(0, str(RCI / "06_stronger_generator"))
sys.path.insert(0, str(RCI / "01_l1_analysis"))
import gen_judge as G            # noqa: E402
from l1_gen_judge import EQUIV   # noqa: E402

try:
    import setproctitle; setproctitle.setproctitle("anon-l1precision")
except Exception:
    pass

FILES = {
    "validation": {"equiv_tasks": "validation_jury_tasks.jsonl",
                   "equiv_out": "validation_equiv_{j}.jsonl",
                   "gen_tasks": "validation_gen_tasks.jsonl",
                   "answers": "validation_answers_llama3b.jsonl",
                   "judge_out": "validation_judge_{j}.jsonl",
                   "sample": "validation_audit_sample.json"},
    "heldout": {"equiv_tasks": "jury_tasks.jsonl",
                "equiv_out": "heldout_equiv_{j}.jsonl",
                "gen_tasks": "heldout_gen_tasks.jsonl",
                "answers": "heldout_answers_llama3b.jsonl",
                "judge_out": "heldout_judge_{j}.jsonl",
                "sample": "heldout_audit_sample.json"},
}


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def equiv(a):
    import torch
    F = FILES[a.set]
    out = HERE / F["equiv_out"].format(j=a.judge)
    done = {r["pair_id"] for r in jl(out)}
    todo = [r for r in jl(HERE / F["equiv_tasks"]) if r["pair_id"] not in done]
    print(f"  [{a.set} equiv {a.judge}] {len(done)} done, {len(todo)} to judge",
          flush=True)
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
            if (s // a.batch_size) % 20 == 0:
                print(f"    {min(s+a.batch_size, len(todo))}/{len(todo)}",
                      flush=True)
    print(f"  [{a.set} equiv {a.judge}] done", flush=True)


def gen(a):
    import torch
    F = FILES[a.set]
    out = HERE / F["answers"]
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / F["gen_tasks"]) if tuple(t["key"]) not in done]
    print(f"  [{a.set} gen] {len(done)} reused/done, {len(todo)} to generate",
          flush=True)
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
            if (s // a.batch_size) % 20 == 0:
                print(f"    {min(s+a.batch_size, len(todo))}/{len(todo)}",
                      flush=True)
    print(f"  [{a.set} gen] done", flush=True)


def _items(which):
    F = FILES[which]
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / F["answers"])}
    items, seen = [], set()
    S = json.load(open(HERE / F["sample"], encoding="utf-8"))
    for d in S["requests"]:
        for k in d:
            if not k.endswith("_key"):
                continue
            a = ans.get(tuple(d[k]))
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
    F = FILES[a.set]
    out = HERE / F["judge_out"].format(j=a.judge)
    done = {tuple(r["sig"]) for r in jl(out)}
    todo = [i for i in _items(a.set) if tuple(i["sig"]) not in done]
    print(f"  [{a.set} judge {a.judge}] {len(done)} done, {len(todo)} to judge",
          flush=True)
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
    print(f"  [{a.set} judge {a.judge}] done", flush=True)


if __name__ == "__main__":
    (HERE / "prompts_verbatim.json").write_text(json.dumps(
        {"generator": G.GENS["llama3b"], "judges": G.JUDGES,
         "forced_prompt": G.FORCED, "correctness_prompt": G.JUDGE,
         "equivalence_prompt": EQUIV,
         "decoding": "greedy, do_sample=False",
         "max_new_tokens": {"generation": 80, "judging": 6},
         "source": "imported verbatim from "
                   "remaining_critical_issues/06_stronger_generator/gen_judge.py "
                   "and 01_l1_analysis/l1_gen_judge.py"},
        indent=2), encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("equiv", equiv), ("gen", gen), ("judge", judge)):
        s = sub.add_parser(nm)
        s.add_argument("--set", choices=list(FILES), required=True)
        if nm != "gen":
            s.add_argument("--judge", choices=list(G.JUDGES), required=True)
        s.add_argument("--gpu", type=int, default=3)
        s.add_argument("--batch-size", type=int, default=16)
        s.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)
