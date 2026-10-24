"""GPU stage. Prompts and decoding are identical to scripts/experiments/ablations/l1_precision_gate/audit_generation_judging_protocol.py
(FORCED, JUDGE, load, clip) and scripts/experiments/ablations/l1_precision_gate/l1_hit_generation_judging_protocol.py (EQUIV), as used by scripts/experiments/ablations/l1_precision_gate/l1_precision_generation_judging.py.
Existing validation outputs in outputs/l1_precision/ are reused read-only where the key matches; new outputs are written only to outputs/l1_validation/outputs/gpu/.
Usage: --stage {equiv,gen,judge} [--judge NAME] --gpu N"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, os, json, pathlib, hashlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--judge"); ap.add_argument("--gpu", required=True); a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "l1_validation")); OLD = (OUTPUT_ROOT / "l1_precision"); O = W/"outputs/gpu"
JUDGES = {"llama3b": "meta-llama/Llama-3.2-3B-Instruct", "llama8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct", "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}
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
EQUIV = """You are comparing two search queries.

QUERY A: {a}
QUERY B: {b}

Do these two queries ask for the SAME fact, such that a single correct answer
would answer both? Reply with exactly one word: SAME or DIFFERENT."""
ABST = "INSUFFICIENT_CONTEXT"
def h(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def clip(t):
    t = (t or "").strip()
    if t.startswith(ABST): return ABST
    for sep in (".", "\n"):
        if sep in t: return t[:t.index(sep) + 1].strip()
    return t
def load(mid):
    tok = AutoTokenizer.from_pretrained(mid); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); return m, tok
def run(items, build, maxlen, mx, write, bs=16):
    m, tok = load(model_id)
    for s in range(0, len(items), bs):
        ch = items[s:s+bs]; enc = tok([build(tok, c) for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)): write(c, t)
        if (s // bs) % 20 == 0: print(f"  {min(s+bs, len(items))}/{len(items)}", flush=True)
if a.stage == "equiv":
    model_id = JUDGES[a.judge]; out = O/f"equiv_{a.judge}.jsonl"
    old = {r["pair_id"] for r in jl(OLD/f"validation_equiv_{a.judge}.jsonl")}; done = {r["pair_id"] for r in jl(out)}
    todo = [p for p in jl(W/"inputs/jury_pairs.jsonl") if p["pair_id"] not in old and p["pair_id"] not in done]
    print(f"equiv {a.judge}: reuse {len(old)}, new {len(todo)}", flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t):
        u = t.upper(); fh.write(json.dumps({"pair_id": c["pair_id"], "judge": a.judge, "raw": t, "label": "DIFFERENT" if "DIFFERENT" in u else "SAME" if "SAME" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: run(todo, lambda tok, c: tok.apply_chat_template([{"role": "user", "content": EQUIV.format(a=c["incoming"], b=c["cached"])}], tokenize=False, add_generation_prompt=True), 1024, 6, w)
elif a.stage == "gen":
    model_id = JUDGES["llama3b"]; out = O/"answers_llama3b.jsonl"
    old = {tuple(r["key"]) for r in jl(OLD/"validation_answers_llama3b.jsonl")}; done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(W/"inputs/gen_tasks.jsonl") if tuple(t["key"]) not in old and tuple(t["key"]) not in done]
    print(f"gen: reuse {len(old)}, new {len(todo)}", flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t): fh.write(json.dumps({"key": c["key"], "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n"); fh.flush()
    if todo: run(todo, lambda tok, c: FORCED.format(question=c["question"], context=c["context"]), 1024, 80, w)
else:
    model_id = JUDGES[a.judge]; out = O/f"judge_{a.judge}.jsonl"
    old = {tuple(r["sig"]) for r in jl(OLD/f"validation_judge_{a.judge}.jsonl")}; done = {tuple(r["sig"]) for r in jl(out)}
    todo = [i for i in jl(W/"inputs/judge_items.jsonl") if tuple(i["sig"]) not in old and tuple(i["sig"]) not in done]
    print(f"judge {a.judge}: reuse {len(old)}, new {len(todo)}", flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t):
        u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "judge": a.judge, "raw": t, "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: run(todo, lambda tok, c: tok.apply_chat_template([{"role": "user", "content": JUDGE.format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True), 2048, 6, w)
print("DONE", flush=True)
