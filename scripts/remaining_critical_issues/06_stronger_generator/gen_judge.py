#!/usr/bin/env python3
"""
Sections 6 and 7 -- generation and judging for the unified held-out audit.

GENERATORS
  llama3b          meta-llama/Llama-3.2-3B-Instruct, the ORIGINAL protocol:
                   FORCED prompt, no chat template, greedy, max_new_tokens=80,
                   1024-token truncation, answer cut at the first "." or
                   newline. Published answers are reused byte-for-byte; only
                   the tasks introduced by the two new arms are generated.
  llama8b_forced   meta-llama/Llama-3.1-8B-Instruct with its OFFICIAL chat
                   template, same forced-answer instruction.
  llama8b_abstain  the same model and template, abstention-capable: answer only
                   when the context supports it, otherwise emit exactly
                   INSUFFICIENT_CONTEXT.

JUDGES (identical prompt, greedy, chat template, blinded to policy identity --
a judge item carries only question / reference / candidate)
  llama3b   meta-llama/Llama-3.2-3B-Instruct
  llama8b   meta-llama/Llama-3.1-8B-Instruct
  qwen7b    Qwen/Qwen2.5-7B-Instruct
  mistral7b mistralai/Mistral-7B-Instruct-v0.3

Llama-3.1-8B is NEVER used as the primary judge of its own generations: the
section-6 analysis uses the qwen7b + mistral7b pair. All four are recorded so
section 7 can report per-model rates, majority, unanimity and kappas.

Usage:
  python gen_judge.py gen   --gen llama8b_forced --gpu 3
  python gen_judge.py judge --judge mistral7b    --gpu 4
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib

HERE = pathlib.Path(__file__).resolve().parent
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

GENS = {"llama3b":         ("meta-llama/Llama-3.2-3B-Instruct", "raw",  "forced"),
        "llama8b_forced":  ("meta-llama/Llama-3.1-8B-Instruct", "chat", "forced"),
        "llama8b_abstain": ("meta-llama/Llama-3.1-8B-Instruct", "chat", "abstain")}
JUDGES = {"llama3b":   "meta-llama/Llama-3.2-3B-Instruct",
          "llama8b":   "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b":    "Qwen/Qwen2.5-7B-Instruct",
          "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}

ABSTAIN = "INSUFFICIENT_CONTEXT"

FORCED = """You are a factual question answering assistant.
Answer the following question in one sentence using the provided context.
You must commit to a single best answer. Never reply that you do not know and
never state that the context is insufficient. If the context is incomplete,
give the most likely specific answer that the context points toward.

Question: {question}

Context:
{context}

Answer:"""

ABSTAIN_PROMPT = """You are a factual question answering assistant.
Answer the following question in one sentence using ONLY the provided context.
If, and only if, the context does not contain the information needed to answer
the question, reply with exactly this token and nothing else:
""" + ABSTAIN + """

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

PROMPTS = {"forced": FORCED, "abstain": ABSTAIN_PROMPT, "judge": JUDGE}


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


def clip(t):
    t = (t or "").strip()
    if t.startswith(ABSTAIN):
        return ABSTAIN
    for sep in (".", "\n"):
        if sep in t:
            return t[:t.index(sep) + 1].strip()
    return t


def gen(args):
    import torch
    model_id, fmt, mode = GENS[args.gen]
    out = HERE / f"answers_{args.gen}.jsonl"
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / "gen_tasks.jsonl") if tuple(t["key"]) not in done]
    print(f"  [{args.gen}] {model_id} fmt={fmt} mode={mode}: "
          f"{len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    model, tok = load(model_id, args.gpu)
    tmpl = PROMPTS[mode]
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), args.batch_size):
            ch = todo[s:s + args.batch_size]
            raw = [tmpl.format(question=c["question"], context=c["context"])
                   for c in ch]
            if fmt == "chat":
                raw = [tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True) for p in raw]
            enc = tok(raw, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024 if fmt == "raw" else 2048
                      ).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                a = clip(t)
                fh.write(json.dumps({"key": c["key"],
                                     "answer": a or "I don't know."}) + "\n")
            fh.flush()
            if (s // args.batch_size) % 10 == 0:
                print(f"    {min(s+args.batch_size, len(todo))}/{len(todo)}",
                      flush=True)
    print(f"  [{args.gen}] generation done", flush=True)


def judge_items(gen_name):
    """One blinded judge item per distinct (question, reference, candidate).
    Abstentions are not judged -- they are a third outcome, not a wrong answer."""
    ans = {tuple(r["key"]): r["answer"]
           for r in jl(HERE / f"answers_{gen_name}.jsonl")}
    items, seen = [], {}
    for d in json.load(open(HERE / "sample.json", encoding="utf-8"))["requests"]:
        for arm in ("fresh", "fc", "sttl", "l1only", "exactttl"):
            a = ans.get(tuple(d[f"{arm}_key"]))
            if a is None or a == ABSTAIN:
                continue
            sig = (h(d["query"]), h(d["gold"]), h(a))
            if sig not in seen:
                seen[sig] = {"sig": list(sig), "q": d["query"],
                             "gold": d["gold"], "cand": a}
                items.append(seen[sig])
    return items


def judge(args):
    import torch
    name, gname = args.judge, args.gen
    out = HERE / f"judge_{gname}__{name}.jsonl"
    done = {tuple(r["sig"]) for r in jl(out)}
    todo = [it for it in judge_items(gname) if tuple(it["sig"]) not in done]
    print(f"  [{gname} | {name}] {len(done)} done, {len(todo)} to judge",
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
                                     "generator": gname, "label": lab}) + "\n")
            fh.flush()
            if (s // args.batch_size) % 10 == 0:
                print(f"    [{name}] {min(s+args.batch_size, len(todo))}/"
                      f"{len(todo)}", flush=True)
    print(f"  [{gname} | {name}] done", flush=True)


if __name__ == "__main__":
    (HERE / "prompts_verbatim.json").write_text(
        json.dumps({"prompts": PROMPTS, "generators": GENS, "judges": JUDGES,
                    "abstain_token": ABSTAIN, "decoding": "greedy, do_sample=False",
                    "max_new_tokens": {"generation": 80, "judging": 6}},
                   indent=2), encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--gen", choices=list(GENS), required=True)
    g.add_argument("--gpu", type=int, default=3)
    g.add_argument("--batch-size", type=int, default=16)
    g.set_defaults(func=gen)
    j = sub.add_parser("judge")
    j.add_argument("--judge", choices=list(JUDGES), required=True)
    j.add_argument("--gen", choices=list(GENS), required=True)
    j.add_argument("--gpu", type=int, default=3)
    j.add_argument("--batch-size", type=int, default=16)
    j.set_defaults(func=judge)
    a = ap.parse_args()
    a.func(a)
