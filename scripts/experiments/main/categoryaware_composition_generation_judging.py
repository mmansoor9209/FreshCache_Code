"""GPU stage for the composition scripts/experiments: processes only items missing from every read-only pool. Prompts and decoding settings are read via AST from scripts/evaluation/generation_judging_stage.py:
  gen    Llama-3.2-3B-Instruct FORCED, batch 16, left pad, max_length 1024, 80 new tokens, clip at first '.'/newline
  judge2 Llama-3.1-8B / Qwen2.5-7B JUDGE, chat template, batch 16, max_length 2048, 6 new tokens
  suffv  Llama-3.1-8B / Qwen2.5-7B SUFFV, chat template (user turn), single, max_length 8192, 8 new tokens
Usage: categoryaware_composition_generation_judging.py gen|judge2|suffv [--judge J] --gpu N"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, ast, datetime, json, os, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--judge"); ap.add_argument("--gpu", required=True); a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "categoryaware_freshcache"); I, OUT = O / "gpu/in", O / "gpu/out"
K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body
     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("FORCED", "JUDGE", "SUFFV", "MODELS")}
import torch   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def clip(t):
    t = (t or "").strip()
    if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
    for sep in (".", "\n"):
        if sep in t: return t[:t.index(sep) + 1].strip()
    return t
def load(mid):
    tok = AutoTokenizer.from_pretrained(mid); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); return m, tok
snap = lambda mid: (pathlib.Path.home() / ".cache/huggingface/hub" / ("models--" + mid.replace("/", "--")) / "refs/main").read_text().strip()
meta = {"stage": a.stage, "judge": a.judge, "start": datetime.datetime.now(datetime.timezone.utc).isoformat()}
if a.stage == "gen":
    out = OUT / "answers_llama3b.jsonl"; done = {tuple(r["key"]) for r in jl(out)}; todo = [t for t in jl(I / "gen_tasks.jsonl") if tuple(t["key"]) not in done]
    mid = K["MODELS"]["llama3b"]; m, tok = load(mid); fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), 16):
        ch = todo[s:s + 16]; enc = tok([K["FORCED"].format(question=c["question"], context=c["context"]) for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)): fh.write(json.dumps({"key": c["key"], "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n")
        fh.flush()
elif a.stage == "judge2":
    out = OUT / f"judge_{a.judge}.jsonl"; done = {tuple(r["sig"]) for r in jl(out)}; todo = [i for i in jl(I / "judge2_items.jsonl") if tuple(i["sig"]) not in done]
    mid = K["MODELS"][a.judge]; m, tok = load(mid); fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), 16):
        ch = todo[s:s + 16]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": K["JUDGE"].format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                  return_tensors="pt", padding=True, truncation=True, max_length=2048).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "judge": a.judge, "raw": t, "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n")
        fh.flush()
elif a.stage == "suffv":
    out = OUT / f"suffv_{a.judge}.jsonl"; done = {(r["query_id"], r["arm"]) for r in jl(out)}; todo = [t for t in jl(I / "suffv_tasks.jsonl") if (t["query_id"], t["arm"]) not in done]
    mid = K["MODELS"][a.judge]; tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); fh = open(out, "a", encoding="utf-8")
    for t in todo:
        enc = tok([tok.apply_chat_template([{"role": "user", "content": K["SUFFV"].format(q=t["query"], ev=t["evidence"])}], tokenize=False, add_generation_prompt=True)], return_tensors="pt", truncation=True, max_length=8192).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(); u = raw.upper()
        fh.write(json.dumps({"query_id": t["query_id"], "arm": t["arm"], "judge": a.judge, "raw": raw, "label": "INSUFFICIENT" if "INSUFFICIENT" in u else "UNCLEAR" if "UNCLEAR" in u else "SUFFICIENT" if "SUFFICIENT" in u else "UNPARSED"}) + "\n"); fh.flush()
meta.update({"model": mid, "snapshot": snap(mid), "todo": len(todo), "end": datetime.datetime.now(datetime.timezone.utc).isoformat()})
with open(O / "work/gpu_runs.jsonl", "a") as f: f.write(json.dumps(meta) + "\n")
print(meta)
