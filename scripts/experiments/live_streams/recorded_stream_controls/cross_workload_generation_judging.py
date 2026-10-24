"""Cross-workload GPU stages. Frozen prompts/decoding are read by AST from scripts/evaluation/generation_judging_stage.py; the reference-validity protocol from
configs/protocols/reference_validity_protocol.json (sha-checked). Writes only outputs/cross_workload/gpu/out. Every stage skips items already present
in any of its output files (resumable, shardable).
  gen3    Llama-3.2-3B FORCED, batch 16, left pad, max_length 1024, 80 new tokens, first-sentence clip          (--tasks FILE --tag T)
  gen14   Qwen2.5-14B FORCED (chat template), max_length 9000, 80 new tokens, first-sentence clip                (--tasks FILE --tag T)
  judge2  frozen JUDGE (--judge llama8b|qwen7b|mistral7b), chat template, batch 16, max_length 2048, 6 new tokens (--tasks FILE)
  phi4    frozen JUDGE with microsoft/phi-4, chat template, 6 new tokens                                          (--tasks FILE)
  suffv   frozen SUFFV (--judge llama8b|qwen7b), single, max_length 8192, 8 new tokens                         (--tasks FILE --tag T)
  suff3   three-level sufficiency prompt, same decoding as suffv (not used in this release)                     (--tasks FILE --tag T)
  refval  frozen reference-validity protocol (--judge llama8b|qwen7b|mistral7b), 12 new tokens, 6,144 input      (--tasks FILE --tag T)"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, ast, datetime, hashlib, json, os, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--judge"); ap.add_argument("--gpu", required=True); ap.add_argument("--shard", default="0/1")
ap.add_argument("--tasks", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--bs", type=int, default=0); a = ap.parse_args()
SI, SN = map(int, a.shard.split("/")); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "cross_workload"); I, OUT = O / "gpu/in", O / "gpu/out"
K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body
     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("FORCED", "JUDGE", "SUFFV", "MODELS")}
RV = (OUTPUT_ROOT / "unseen_queries/refval")
PROT = json.load(open((ROOT / "configs/protocols/reference_validity_protocol.json"))); assert hashlib.sha256(((ROOT / "configs/protocols/reference_validity_protocol.json")).read_bytes()).hexdigest() == open((ROOT / "configs/protocols/reference_validity_protocol.sha256")).read().split()[0]
SUFF3 = None   # three-level stage not used here
import torch   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def clip(t):
    t = (t or "").strip()
    if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
    for sep in (".", "\n"):
        if sep in t: return t[:t.index(sep) + 1].strip()
    return t
def load(mid, left=True):
    tok = AutoTokenizer.from_pretrained(mid); tok.pad_token = tok.pad_token or tok.eos_token
    if left: tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); return m, tok
TASKS = jl(I / a.tasks)
def shard(items, keyf, done): return [x for i, x in enumerate(items) if i % SN == SI and keyf(x) not in done]
meta = {"stage": a.stage, "judge": a.judge, "tasks": a.tasks, "tag": a.tag, "shard": a.shard, "start": datetime.datetime.now(datetime.timezone.utc).isoformat()}
sfx = "" if SN == 1 else f"_shard{SI}of{SN}"
if a.stage in ("gen3", "gen14"):
    pre = "answers3b" if a.stage == "gen3" else "answers14"
    out = OUT / f"{pre}{a.tag}{sfx}.jsonl"; done = {tuple(r["key"]) for f in OUT.glob(f"{pre}*.jsonl") for r in jl(f)}
    todo = sorted(shard(TASKS, lambda t: tuple(t["key"]), done), key=lambda t: len(t["context"]))
    if a.stage == "gen3":
        mid = K["MODELS"]["llama3b"]; m, tok = load(mid); BS = a.bs or 16
        mk = lambda ch: tok([K["FORCED"].format(question=c["question"], context=c["context"]) for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=1024)
    else:
        mid = "Qwen/Qwen2.5-14B-Instruct"; m, tok = load(mid); BS = a.bs or 2
        mk = lambda ch: tok([tok.apply_chat_template([{"role": "user", "content": K["FORCED"].format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                            return_tensors="pt", padding=True, truncation=True, max_length=9000)
    fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), BS):
        ch = todo[s:s + BS]; enc = mk(ch).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            fh.write(json.dumps({"key": c["key"], "raw": t, "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n")
        fh.flush()
elif a.stage in ("judge2", "phi4"):
    jname = a.judge if a.stage == "judge2" else "phi4"; mid = K["MODELS"][a.judge] if a.stage == "judge2" else "microsoft/phi-4"
    out = OUT / f"judge_{jname}{sfx}.jsonl"; done = {tuple(r["sig"]) for f in OUT.glob(f"judge_{jname}*.jsonl") for r in jl(f)}
    todo = shard(TASKS, lambda x: tuple(x["sig"]), done); m, tok = load(mid); fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), 16):
        ch = todo[s:s + 16]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": K["JUDGE"].format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                  return_tensors="pt", padding=True, truncation=True, max_length=2048).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "judge": jname, "raw": t, "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n")
        fh.flush()
elif a.stage in ("suffv", "suff3"):
    pre = "cross_suffv" if a.stage == "suffv" else "cross_suff3"; out = OUT / f"{pre}_{a.judge}{a.tag}{sfx}.jsonl"
    done = {(r["query_id"], r["arm"]) for f in OUT.glob(f"{pre}_{a.judge}*.jsonl") for r in jl(f) if r["label"] != "UNPARSED" or r.get("retry")}
    todo = shard(TASKS, lambda t: (t["query_id"], t["arm"]), done); mid = K["MODELS"][a.judge]
    tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval(); fh = open(out, "a", encoding="utf-8")
    PROMPT = K["SUFFV"] if a.stage == "suffv" else SUFF3
    LABS = ("INSUFFICIENT", "UNCLEAR", "SUFFICIENT") if a.stage == "suffv" else ("PARTIAL", "NONE", "FULL")
    def parse(raw):
        u = raw.upper()
        for l in LABS:
            if l in u: return l
        return "UNPARSED"
    retried = {(r["query_id"], r["arm"]) for f in OUT.glob(f"{pre}_{a.judge}*.jsonl") for r in jl(f) if r["label"] == "UNPARSED"}
    for t in todo:
        enc = tok([tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=t["query"], ev=t["evidence"])}], tokenize=False, add_generation_prompt=True)], return_tensors="pt", truncation=True, max_length=8192).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        fh.write(json.dumps({"query_id": t["query_id"], "arm": t["arm"], "judge": a.judge, "raw": raw, "label": parse(raw), "retry": (t["query_id"], t["arm"]) in retried}) + "\n"); fh.flush()
elif a.stage == "refval":
    D = PROT["decoding"]; mid = PROT["judges"][a.judge]; LAB = PROT["reference_stage"]["labels"]
    tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    def parse(t):
        u = t.upper().replace(" ", "_"); hits = [(u.find(l), l) for l in LAB if u.find(l) >= 0]
        if not hits:
            for l, alt in (("VALID_AT_T", "VALID"), ("INVALID_AT_T", "INVALID"), ("TEMPORALLY_AMBIGUOUS", "AMBIGUOUS")):
                if re.search(rf"\b{alt}\b", t.upper()): hits.append((0, l))
            hits = [x for x in hits if not (x[1] == "VALID_AT_T" and re.search(r"\bINVALID\b", t.upper()))] or hits
        return min(hits)[1] if hits else "UNPARSED"
    out = OUT / f"cross_refval_{a.judge}{a.tag}{sfx}.jsonl"; done = {r["task_id"] for f in OUT.glob(f"cross_refval_{a.judge}*.jsonl") for r in jl(f)}
    todo = shard(TASKS, lambda t: t["task_id"], done); fh = open(out, "a", encoding="utf-8")
    for t in todo:
        msgs = [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"]}]
        if a.judge == "mistral7b": msgs = [{"role": "user", "content": t["system"] + "\n\n" + t["user"]}]
        enc = tok([tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], return_tensors="pt", truncation=True, max_length=D["max_input_tokens"]).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=D["max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        fh.write(json.dumps({"task_id": t["task_id"], "judge": a.judge, "raw": raw, "label": parse(raw), "input_tokens": int(enc["input_ids"].shape[1]),
                             "truncated": bool(enc["input_ids"].shape[1] >= D["max_input_tokens"])}, ensure_ascii=False) + "\n"); fh.flush()
else:
    sys.exit(f"unknown stage {a.stage}")
meta.update({"model": mid, "todo": len(todo), "end": datetime.datetime.now(datetime.timezone.utc).isoformat()})
with open(O / "work/gpu_runs.jsonl", "a") as f: f.write(json.dumps(meta) + "\n")
print(meta)
