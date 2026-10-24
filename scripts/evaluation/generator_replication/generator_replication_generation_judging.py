"""GPU stages of the generator replication. Prompts and decoding are copied verbatim from
scripts/experiments/ablations/l1_precision_gate/audit_generation_judging_protocol.py (Llama-3.1-8B-Instruct, chat template, greedy,
80 new tokens, 2048-token truncation, batch 16, clipped answers; JUDGE prompt, greedy, 6 new tokens) and, for the secondary
protocol, from the answer stage of scripts/evaluation/reference_validation/reference_validation_judging.py (fixed protocol file
outputs/reference_validation/config/protocol.json, sha256 prefix d601fac4; 3 judges, blinded, shuffled).
Existing Llama-3.1-8B answers and judgments are reused read-only where the exact key or signature matches; new outputs go to
outputs/generator_replication/outputs/ only. A seeded sample of 48 reused answers is regenerated as a determinism check.
Usage: gen --mode {forced,abstain} | judge --mode M --judge {qwen7b,mistral7b} | judge3 --mode M --judge {llama8b,qwen7b,mistral7b}  --gpu N"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, os, json, pathlib, hashlib, random, re, datetime; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--mode", required=True); ap.add_argument("--judge"); ap.add_argument("--gpu", required=True); a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "generator_replication")); O = W/"outputs"
SG = (OUTPUT_ROOT / "l1only_exactttl/stronger_generator"); CR = (OUTPUT_ROOT / "replay/gpu/out3")
MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct", "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}
ABST = "INSUFFICIENT_CONTEXT"
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
""" + ABST + """

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
def h(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def clip(t):
    t = (t or "").strip()
    if t.startswith(ABST): return ABST
    for sep in (".", "\n"):
        if sep in t: return t[:t.index(sep) + 1].strip()
    return t
def load(mid):
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"; m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); return m, tok
def batched(items, mid, build, maxlen, mx, write, bs=16):
    m, tok = load(mid)
    for s in range(0, len(items), bs):
        ch = items[s:s+bs]; enc = tok([build(tok, c) for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)): write(c, t)
        if (s // bs) % 10 == 0: print(f"  {a.stage}/{a.mode}/{a.judge} {min(s+bs, len(items))}/{len(items)}", flush=True)
def answers(mode):   # reused first (read-only), new last; exact key
    A = {}
    for p in (SG/f"answers_llama8b_{mode}.jsonl", CR/f"answers_llama8b_{mode}.jsonl", O/f"answers_llama8b_{mode}.jsonl"):
        for r in jl(p): A.setdefault(tuple(r["key"]), r["answer"])
    return A
if a.stage == "gen":
    tmpl = FORCED if a.mode == "forced" else ABSTAIN_PROMPT; A = answers(a.mode)
    todo = [t for t in jl(W/"inputs/gen_tasks.jsonl") if tuple(t["key"]) not in A]
    fid = random.Random(42).sample([t for t in jl(W/"inputs/gen_tasks.jsonl") if tuple(t["key"]) in A], 48)   # determinism check on reused answers
    print("gen", a.mode, "new", len(todo), "fidelity sample", len(fid), flush=True)
    fh = open(O/f"answers_llama8b_{a.mode}.jsonl", "a", encoding="utf-8"); ff = open(O/f"fidelity_llama8b_{a.mode}.jsonl", "w", encoding="utf-8")
    def w(c, t):
        rec = {"key": c["key"], "answer": clip(t) or "I don't know."}
        (ff if c.get("_fid") else fh).write(json.dumps({**rec, **({"reused": A[tuple(c["key"])]} if c.get("_fid") else {})}, ensure_ascii=False) + "\n")
    batched(todo + [{**t, "_fid": 1} for t in fid], MODELS["llama8b"], lambda tok, c: tok.apply_chat_template([{"role": "user", "content": tmpl.format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True), 2048, 80, w)
    fh.close(); ff.close()
elif a.stage == "judge":   # primary protocol of the Llama-3.1-8B replication
    have = {}
    for p in (SG/f"judge_llama8b_{a.mode}__{a.judge}.jsonl", CR/f"judge_llama8b_{a.mode}__{a.judge}.jsonl", O/f"judge_llama8b_{a.mode}__{a.judge}.jsonl"):
        for r in jl(p): have.setdefault(tuple(r["sig"]), r["label"])
    todo = [i for i in jl(W/f"inputs/judge_{a.mode}_items.jsonl") if tuple(i["sig"]) not in have]
    print("judge", a.mode, a.judge, "new", len(todo), flush=True); fh = open(O/f"judge_llama8b_{a.mode}__{a.judge}.jsonl", "a", encoding="utf-8")
    def w(c, t):
        u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "judge": a.judge, "generator": f"llama8b_{a.mode}", "raw": t, "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: batched(todo, MODELS[a.judge], lambda tok, c: tok.apply_chat_template([{"role": "user", "content": JUDGE.format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True), 2048, 6, w)
elif a.stage == "judge3":   # fixed blinded 3-judge protocol (secondary)
    P = json.load(open((OUTPUT_ROOT / "reference_validation/config/protocol.json"))); S = P["answer_stage"]; D = P["decoding"]
    assert hashlib.sha256(((OUTPUT_ROOT / "reference_validation/config/protocol.json")).read_bytes()).hexdigest().startswith("d601fac4")
    out = O/f"answer3_{a.mode}_{a.judge}.jsonl"; done = {r["task_id"] for r in jl(out)}
    todo = [t for t in jl(W/f"inputs/judge3_{a.mode}_tasks.jsonl") if t["task_id"] not in done]
    print("judge3", a.mode, a.judge, "todo", len(todo), flush=True)
    tok = AutoTokenizer.from_pretrained(P["judges"][a.judge]); m = AutoModelForCausalLM.from_pretrained(P["judges"][a.judge], dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    with open(out, "a", encoding="utf-8") as f:
        for i, t in enumerate(todo):
            u = S["user"].format(question=t["question"], target=t["target"], reference=t["reference"], candidate=t["candidate"])
            msgs = [{"role": "user", "content": S["system"] + "\n\n" + u}] if a.judge == "mistral7b" else [{"role": "system", "content": S["system"]}, {"role": "user", "content": u}]
            enc = tok([tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], return_tensors="pt", truncation=True, max_length=D["max_input_tokens"]).to(m.device)
            with torch.no_grad(): g = m.generate(**enc, max_new_tokens=D["max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            f.write(json.dumps({"task_id": t["task_id"], "query_id": t["query_id"], "judge": a.judge, "raw": tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(),
                                "utc": datetime.datetime.utcnow().isoformat() + "Z"}, ensure_ascii=False) + "\n"); f.flush()
            if i % 300 == 0: print(f"  {i}/{len(todo)}", flush=True)
print("DONE", a.stage, a.mode, a.judge, flush=True)
