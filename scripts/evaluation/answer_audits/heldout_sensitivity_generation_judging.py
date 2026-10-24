#!/usr/bin/env python3
"""
heldout_sensitivity_generation_judging.py — seed from the existing audit, then generate/judge ONLY what is
missing for the new SemanticTTL arm. Protocol identical to the existing audits.

  seed   copy already-computed answers and judgments out of the existing
         held-out audit for keys/signatures it already contains
  gen    generate the remaining answers      (Llama-3.2-3B-Instruct)
  judge  judge the remaining signatures      (Llama-3.1-8B + Qwen2.5-7B)
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, hashlib, json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
HERE = ((OUTPUT_ROOT / "quality_selection/heldout_sensitivity"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
A2 = (OUTPUT_ROOT / "general_audit")
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
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
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]   # noqa


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def seed(_a):
    old = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
    want = {tuple(t["key"]) for t in jl(HERE / "h_gen_tasks.jsonl")}
    out = HERE / "h_answers.jsonl"
    have = {tuple(r["key"]) for r in jl(out)}
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for k in sorted(want & set(old) - have):
            f.write(json.dumps({"key": list(k), "answer": old[k],
                                "source": "reused_existing_audit"},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"  seeded {n} answers reused from the existing held-out audit")
    print(f"  remaining to generate: {len(want - set(old) - have)}")
    for j in JUDGES:
        o = HERE / f"h_judge_{j}.jsonl"
        oldj = {tuple(r["sig"]): r["label"]
                for r in jl(A2 / f"judge_{j}.jsonl")}
        havej = {tuple(r["sig"]) for r in jl(o)}
        m = 0
        with open(o, "a", encoding="utf-8") as f:
            for s, lab in oldj.items():
                if s not in havej:
                    f.write(json.dumps({"sig": list(s), "judge": j, "label": lab,
                                        "source": "reused_existing_audit"}) + "\n")
                    m += 1
        print(f"  seeded {m} {j} judgments reused from the existing held-out audit")


def gen(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = HERE / "h_answers.jsonl"
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / "h_gen_tasks.jsonl") if tuple(t["key"]) not in done]
    print(f"  {len(done)} present, {len(todo)} to generate", flush=True)
    if not todo:
        return
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else None
    tok = AutoTokenizer.from_pretrained(GEN, token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GEN, dtype=torch.bfloat16, device_map="cuda:0", token=hf)
    model.eval()
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch):
            ch = todo[s:s + a.batch]
            pr = [FORCED.format(question=c["question"], context=c["context"])
                  for c in ch]
            enc = tok(pr, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                t = t.strip()
                for sep in (".", "\n"):
                    if sep in t:
                        t = t[:t.index(sep) + 1].strip(); break
                fh.write(json.dumps({"key": c["key"],
                                     "answer": t or "I don't know.",
                                     "source": "new"}, ensure_ascii=False) + "\n")
            fh.flush()
    print("  generation done", flush=True)


def items():
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "h_answers.jsonl")}
    out, seen = [], set()
    for d in jl(HERE / "h_sample.jsonl"):
        a = ans.get(tuple(d["new_sttl_key"]))
        if a is None:
            continue
        sig = (h(d["query"]), h(d["gold"]), h(a))
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"sig": list(sig), "q": d["query"], "gold": d["gold"], "cand": a})
    return out


def judge(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = HERE / f"h_judge_{a.judge}.jsonl"
    done = {tuple(r["sig"]) for r in jl(out)}
    todo = [it for it in items() if tuple(it["sig"]) not in done]
    print(f"  [{a.judge}] {len(done)} present, {len(todo)} to judge", flush=True)
    if not todo:
        return
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else None
    tok = AutoTokenizer.from_pretrained(JUDGES[a.judge], token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        JUDGES[a.judge], dtype=torch.bfloat16, device_map="cuda:0", token=hf)
    model.eval()
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch):
            ch = todo[s:s + a.batch]
            tx = [tok.apply_chat_template(
                [{"role": "user", "content": JUDGE.format(
                    q=c["q"], gold=c["gold"], cand=c["cand"])}],
                tokenize=False, add_generation_prompt=True) for c in ch]
            enc = tok(tx, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                u = t.upper()
                fh.write(json.dumps({"sig": c["sig"], "judge": a.judge,
                                     "label": ("INCORRECT" if "INCORRECT" in u
                                               else "CORRECT" if "CORRECT" in u
                                               else "UNPARSED"),
                                     "source": "new"}) + "\n")
            fh.flush()
    print(f"  [{a.judge}] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["seed", "gen", "judge"])
    ap.add_argument("--judge", choices=list(JUDGES))
    ap.add_argument("--gpu", default="0"); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    {"seed": seed, "gen": gen, "judge": judge}[a.stage](a)
