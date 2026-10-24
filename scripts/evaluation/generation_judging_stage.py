"""Consolidated GPU stage for the results recompute (generation, judging, equivalence and sufficiency labels).
Prompts, decoding and parsers are copied verbatim from the frozen protocols:
  gen / judge2 / equiv : L1-validation generator, two-judge answer rule and equivalence check
  judge3               : scripts/evaluation/reference_validation/reference_validation_judging.py answer stage + reparse (frozen protocol d601fac4...)
  suff3                : tier-quality evidence sufficiency (study_config prompt, 3 judges)
  suffv                : L2-verifier answer-audit sufficiency (SUFFV, 2 judges)
Only items listed in gpu/in/*.jsonl are run; each builder already removed items present in the read-only pools of earlier runs.
Usage: generation_judging_stage.py STAGE [--judge J] --gpu N"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, os, json, pathlib, hashlib, re, datetime; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--judge"); ap.add_argument("--gpu", required=True); ap.add_argument("--indir", default="in"); ap.add_argument("--outdir", default="out"); ap.add_argument("--mode", default="forced"); a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); G = ((OUTPUT_ROOT / "replay/gpu")); I, O = G/a.indir, G/a.outdir; O.mkdir(exist_ok=True)
MODELS = {"llama3b": "meta-llama/Llama-3.2-3B-Instruct", "llama8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct", "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}
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
SUFFV = """You are evaluating search-result evidence for retrieval-augmented QA.

Incoming query:
{q}

Search results served from the cache for this request:
{ev}

Determine whether these search results contain sufficient information or
credible evidence that could support answering the incoming query.

Return exactly one label:

SUFFICIENT
INSUFFICIENT
UNCLEAR

SUFFICIENT means at least one or more results provide evidence directly
relevant enough to support answering the query.
INSUFFICIENT means the results are largely unrelated or do not provide
useful evidence for answering the query.
UNCLEAR means relevance cannot be determined reliably from the available
evidence."""
MB = ROOT / "configs/protocols/reference_validity_protocol.json"
assert hashlib.sha256(MB.read_bytes()).hexdigest() == "d601fac400ec42197b8d9e8f07bf7821f9b23a2eec40dd1aa0c5cc38c5abf591"
AP = json.load(open(MB)); TC = json.load(open((ROOT / "configs/protocols/tier_quality_study_config.json")))
assert hashlib.sha256(((ROOT / "configs/protocols/tier_quality_study_config.json")).read_bytes()).hexdigest() == open((ROOT / "configs/protocols/tier_quality_study_config.sha256")).read().strip()
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def clip(t):
    t = (t or "").strip()
    if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
    for sep in (".", "\n"):
        if sep in t: return t[:t.index(sep) + 1].strip()
    return t
def ans3(raw):   # reference_validation_reparse.ans
    u = raw.upper(); u = re.sub(r"\bINCORRECT\b", "WRONG", u); u = re.sub(r"^(WROONG|WONG)\b", "WRONG", u)
    hits = [(u.find(l), l) for l in ("CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE") if u.find(l) >= 0]; return min(hits)[1] if hits else "UNPARSED"
def load(mid, left=True):
    tok = AutoTokenizer.from_pretrained(mid); tok.pad_token = tok.pad_token or tok.eos_token
    if left: tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); return m, tok
def batched(items, mid, build, maxlen, mx, write, bs=16):
    m, tok = load(mid)
    for s in range(0, len(items), bs):
        ch = items[s:s+bs]; enc = tok([build(tok, c) for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)): write(c, t)
        if (s // bs) % 20 == 0: print(f"  {a.stage}/{a.judge} {min(s+bs, len(items))}/{len(items)}", flush=True)
def single(items, mid, msgs_of, maxlen, mx, write):
    tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    for i, t in enumerate(items):
        enc = tok([tok.apply_chat_template(msgs_of(t), tokenize=False, add_generation_prompt=True)], return_tensors="pt", truncation=True, max_length=maxlen).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        write(t, tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), int(enc["input_ids"].shape[1]))
        if i % 200 == 0: print(f"  {a.stage}/{a.judge} {i}/{len(items)}", flush=True)
now = lambda: datetime.datetime.utcnow().isoformat() + "Z"
if a.stage == "gen":
    out = O/"answers_llama3b.jsonl"; done = {tuple(r["key"]) for r in jl(out)}; todo = [t for t in jl(I/"gen_tasks.jsonl") if tuple(t["key"]) not in done]
    print("gen todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t): fh.write(json.dumps({"key": c["key"], "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n"); fh.flush()
    if todo: batched(todo, MODELS["llama3b"], lambda tok, c: FORCED.format(question=c["question"], context=c["context"]), 1024, 80, w)
elif a.stage == "judge2":
    out = O/f"judge_{a.judge}.jsonl"; done = {tuple(r["sig"]) for r in jl(out)}; todo = [i for i in jl(I/"judge2_items.jsonl") if tuple(i["sig"]) not in done]
    print("judge2", a.judge, "todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t):
        u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "judge": a.judge, "raw": t, "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: batched(todo, MODELS[a.judge], lambda tok, c: tok.apply_chat_template([{"role": "user", "content": JUDGE.format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True), 2048, 6, w)
elif a.stage == "equiv":
    out = O/f"equiv_{a.judge}.jsonl"; done = {r["pair_id"] for r in jl(out)}; todo = [p for p in jl(I/"jury_pairs.jsonl") if p["pair_id"] not in done]
    print("equiv", a.judge, "todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t):
        u = t.upper(); fh.write(json.dumps({"pair_id": c["pair_id"], "judge": a.judge, "raw": t, "label": "DIFFERENT" if "DIFFERENT" in u else "SAME" if "SAME" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: batched(todo, MODELS[a.judge], lambda tok, c: tok.apply_chat_template([{"role": "user", "content": EQUIV.format(a=c["incoming"], b=c["cached"])}], tokenize=False, add_generation_prompt=True), 1024, 6, w)
elif a.stage == "judge3":
    out = O/f"answer3_{a.judge}.jsonl"; done = {r["task_id"] for r in jl(out)}; todo = [t for t in jl(I/"judge3_tasks.jsonl") if t["task_id"] not in done]
    print("judge3", a.judge, "todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8"); S = AP["answer_stage"]
    def msgs(t):
        u = S["user"].format(question=t["question"], target=t["target"], reference=t["reference"], candidate=t["candidate"])
        return [{"role": "user", "content": S["system"] + "\n\n" + u}] if a.judge == "mistral7b" else [{"role": "system", "content": S["system"]}, {"role": "user", "content": u}]
    def w(t, raw, n): fh.write(json.dumps({"task_id": t["task_id"], "query_id": t["query_id"], "judge": a.judge, "raw": raw, "label": ans3(raw), "input_tokens": n, "utc": now()}, ensure_ascii=False) + "\n"); fh.flush()
    if todo: single(todo, AP["judges"][a.judge], msgs, AP["decoding"]["max_input_tokens"], AP["decoding"]["max_new_tokens"], w)
elif a.stage == "suff3":
    out = O/f"suff3_{a.judge}.jsonl"; done = {r["task_id"] for r in jl(out)}; todo = [t for t in jl(I/"suff3_tasks.jsonl") if t["task_id"] not in done]
    print("suff3", a.judge, "todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8")
    def w(t, raw, n):
        u = raw.upper(); lab = "INSUFFICIENT" if "INSUFFICIENT" in u else "UNCLEAR" if "UNCLEAR" in u else "SUFFICIENT" if "SUFFICIENT" in u else "UNPARSED"
        fh.write(json.dumps({"task_id": t["task_id"], "query_id": t["query_id"], "judge": a.judge, "raw": raw, "label": lab, "input_tokens": n, "utc": now()}, ensure_ascii=False) + "\n"); fh.flush()
    if todo: single(todo, AP["judges"][a.judge], lambda t: [{"role": "user", "content": TC["sufficiency_grading"]["prompt"].format(q=t["question"], ev=t["evidence"])}], 6144, 8, w)
elif a.stage == "suffv":
    out = O/f"suffv_{a.judge}.jsonl"; done = {(r["query_id"], r["arm"]) for r in jl(out)}; todo = [t for t in jl(I/"suffv_tasks.jsonl") if (t["query_id"], t["arm"]) not in done]
    print("suffv", a.judge, "todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8")
    def w(t, raw, n):
        u = raw.upper(); fh.write(json.dumps({"query_id": t["query_id"], "arm": t["arm"], "judge": a.judge, "raw": raw, "label": "INSUFFICIENT" if "INSUFFICIENT" in u else "UNCLEAR" if "UNCLEAR" in u else "SUFFICIENT" if "SUFFICIENT" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: single(todo, MODELS[a.judge], lambda t: [{"role": "user", "content": SUFFV.format(q=t["query"], ev=t["evidence"])}], 8192, 8, w)
elif a.stage == "gen8":   # audit_generation_judging_protocol.py gen(): llama8b_{forced,abstain}
    ABST8 = "INSUFFICIENT_CONTEXT"
    ABSTAIN_PROMPT = ("You are a factual question answering assistant.\nAnswer the following question in one sentence using ONLY the provided context.\n"
                      "If, and only if, the context does not contain the information needed to answer\nthe question, reply with exactly this token and nothing else:\n" + ABST8 +
                      "\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:")
    tmpl = FORCED if a.mode == "forced" else ABSTAIN_PROMPT
    out = O/f"answers_llama8b_{a.mode}.jsonl"; done = {tuple(r["key"]) for r in jl(out)}; todo = [t for t in jl(I/f"gen8_{a.mode}_tasks.jsonl") if tuple(t["key"]) not in done]
    print("gen8", a.mode, "todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t): fh.write(json.dumps({"key": c["key"], "answer": clip(t) or "I don't know."}) + "\n"); fh.flush()
    if todo: batched(todo, MODELS["llama8b"], lambda tok, c: tok.apply_chat_template([{"role": "user", "content": tmpl.format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True), 2048, 80, w)
elif a.stage == "judge8":   # judge(): same JUDGE prompt, chat template, 2048, 6 tokens
    out = O/f"judge_llama8b_{a.mode}__{a.judge}.jsonl"; done = {tuple(r["sig"]) for r in jl(out)}; todo = [i for i in jl(I/f"judge8_{a.mode}_items.jsonl") if tuple(i["sig"]) not in done]
    print("judge8", a.mode, a.judge, "todo", len(todo), flush=True); fh = open(out, "a", encoding="utf-8")
    def w(c, t):
        u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "judge": a.judge, "generator": f"llama8b_{a.mode}", "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: batched(todo, MODELS[a.judge], lambda tok, c: tok.apply_chat_template([{"role": "user", "content": JUDGE.format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True), 2048, 6, w)
print("DONE", a.stage, a.judge, flush=True)
