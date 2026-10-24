"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

GPU stage of the SemanticTTL frontier: answer generation (Llama-3.2-3B-Instruct, FORCED prompt) and two-judge grading
(Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct).
Prompts are read from scripts/evaluation/generation_judging_stage.py (AST literal extraction) and asserted byte-identical to the
FORCED prompt of scripts/evaluation/answer_audits/quality_selection_generation.py and the JUDGE prompt of
scripts/evaluation/answer_audits/quality_selection_judging.py. Decoding matches the `batched` helper of scripts/evaluation/generation_judging_stage.py:
greedy, batch 16, left padding; generation max_length 1024, 80 new tokens, answer clipped at the first '.' or newline;
judging with the chat template, max_length 2048, 6 new tokens, label INCORRECT if present, else CORRECT, else UNPARSED.
Only items listed in outputs/semanticttl_frontier/gpu/in/*.jsonl are run; outputs are appended to outputs/semanticttl_frontier/gpu/out/.
Usage: semanticttl_frontier_generation_judging.py gen|judge2 [--judge llama8b|qwen7b] --gpu N --tag val|ho"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, ast, datetime, json, os, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--judge"); ap.add_argument("--gpu", required=True); ap.add_argument("--tag", required=True)
a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); STTL_FRONTIER_DIR = (OUTPUT_ROOT / "semanticttl_frontier"); I, O = STTL_FRONTIER_DIR / "gpu/in", STTL_FRONTIER_DIR / "gpu/out"

def consts(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read()); out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out
RUN = consts(ROOT / "scripts/evaluation/generation_judging_stage.py", {"FORCED", "JUDGE", "MODELS"})
FORCED, JUDGE, MODELS = RUN["FORCED"], RUN["JUDGE"], RUN["MODELS"]
assert FORCED == consts(ROOT / "scripts/evaluation/answer_audits/quality_selection_generation.py", {"FORCED"})["FORCED"]
assert JUDGE == consts(ROOT / "scripts/evaluation/answer_audits/quality_selection_judging.py", {"JUDGE"})["JUDGE"]

import torch   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def clip(t):   # answer clipping, identical to scripts/evaluation/generation_judging_stage.py
    t = (t or "").strip()
    if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
    for sep in (".", "\n"):
        if sep in t: return t[:t.index(sep) + 1].strip()
    return t
def load(mid):
    tok = AutoTokenizer.from_pretrained(mid); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); return m, tok
def snapshot(mid):
    d = pathlib.Path.home() / ".cache/huggingface/hub" / ("models--" + mid.replace("/", "--")) / "refs/main"
    return d.read_text().strip() if d.exists() else "MISSING"
def batched(items, mid, build, maxlen, mx, write, bs=16):
    m, tok = load(mid)
    for s in range(0, len(items), bs):
        ch = items[s:s + bs]; enc = tok([build(tok, c) for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)): write(c, t)
        if (s // bs) % 20 == 0: print(f"  {a.stage}/{a.judge} {min(s + bs, len(items))}/{len(items)}", flush=True)
meta = {"stage": a.stage, "judge": a.judge, "tag": a.tag, "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__}
if a.stage == "gen":
    out = O / f"{a.tag}_answers_llama3b.jsonl"; done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(I / f"{a.tag}_gen_tasks.jsonl") if tuple(t["key"]) not in done]
    meta.update({"model": MODELS["llama3b"], "snapshot": snapshot(MODELS["llama3b"]), "todo": len(todo)}); print(meta, flush=True)
    fh = open(out, "a", encoding="utf-8")
    def w(c, t): fh.write(json.dumps({"key": c["key"], "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n"); fh.flush()
    if todo: batched(todo, MODELS["llama3b"], lambda tok, c: FORCED.format(question=c["question"], context=c["context"]), 1024, 80, w)
elif a.stage == "judge2":
    out = O / f"{a.tag}_judge_{a.judge}.jsonl"; done = {tuple(r["sig"]) for r in jl(out)}
    todo = [i for i in jl(I / f"{a.tag}_judge2_items.jsonl") if tuple(i["sig"]) not in done]
    meta.update({"model": MODELS[a.judge], "snapshot": snapshot(MODELS[a.judge]), "todo": len(todo)}); print(meta, flush=True)
    fh = open(out, "a", encoding="utf-8")
    def w(c, t):
        u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "judge": a.judge, "raw": t, "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n"); fh.flush()
    if todo: batched(todo, MODELS[a.judge], lambda tok, c: tok.apply_chat_template([{"role": "user", "content": JUDGE.format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True), 2048, 6, w)
meta["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(STTL_FRONTIER_DIR / "logs/gpu_runs.jsonl", "a") as f: f.write(json.dumps(meta) + "\n")
print("done", meta, flush=True)
