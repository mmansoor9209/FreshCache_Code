"""Stage 2: run one judge on one stage. Usage: reference_validation_judging.py --stage {reference,answer} --judge {llama8b,qwen7b,mistral7b} --gpu N [--study {reference_validation,unseen_queries,model_adjudication}]"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, os, sys, json, hashlib, pathlib, random, datetime, re; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--judge", required=True); ap.add_argument("--gpu", required=True); ap.add_argument("--study", default="reference_validation", choices=["reference_validation", "unseen_queries", "model_adjudication"]); a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
if a.study == "reference_validation":   # general held-out audit: protocol frozen by reference_validation_protocol.py in the workspace
    W = ((OUTPUT_ROOT / "reference_validation")); PJ, PS = W / "config/protocol.json", W / "config/protocol.sha256"
elif a.study == "model_adjudication":   # three-family adjudication of the 400 audit requests: protocol copied by model_adjudication_inputs.py
    W = ((OUTPUT_ROOT / "model_adjudication")); PJ, PS = W / "config/protocol.json", W / "config/protocol.sha256"
else:                                    # live unseen-query cases: the same frozen protocol, shipped in configs/
    W = ((OUTPUT_ROOT / "unseen_queries/refval")); PJ, PS = REPO_ROOT / "configs/protocols/reference_validity_protocol.json", REPO_ROOT / "configs/protocols/reference_validity_protocol.sha256"
P = json.load(open(PJ)); h = hashlib.sha256(PJ.read_bytes()).hexdigest()
assert h == open(PS).read().split()[0], "protocol changed after freezing"
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
REQ = {r["query_id"]: r for r in jl(W/"inputs/requests.jsonl")}; EV = {r["query_id"]: r for r in jl(W/"inputs/evidence.jsonl")}
def date(ts): return ts[:10] if ts else "unknown"
def target(q): e = EV[q]; return date(e["evidence_time_max"])
tasks = []
if a.stage == "reference":
    S = P["reference_stage"]
    for q, r in REQ.items():
        e = EV[q]; used = [p for p in e["pages"] if p.get("used")]
        if not used: continue
        evid = "\n\n".join(f"[page {i+1}, captured {date(p['fetched_at'])}] {p['text']}" for i, p in enumerate(used))
        caps = ", ".join(sorted({date(p["fetched_at"]) for p in used}))
        tasks.append({"task_id": q, "query_id": q, "system": S["system"], "user": S["user"].format(question=r["question"], target=target(q), reference=r["reference_answer"], captured=caps, evidence=evid)})
else:
    S = P["answer_stage"]; seen = {}
    for o in jl(W/"inputs/policy_outputs.jsonl"):
        k = (o["query_id"], o["answer"])
        if k in seen: continue
        tid = hashlib.sha256(f"{o['query_id']}\x00{o['answer']}".encode()).hexdigest()[:16]; seen[k] = tid; r = REQ[o["query_id"]]
        tasks.append({"task_id": tid, "query_id": o["query_id"], "system": S["system"], "user": S["user"].format(question=r["question"], target=target(o["query_id"]), reference=r["reference_answer"], candidate=o["answer"])})
    random.Random(P["seed"]).shuffle(tasks)
out = W/"outputs/raw"/f"{a.stage}_{a.judge}.jsonl"
done = {r["task_id"] for r in jl(out)} if out.exists() else set()
todo = [t for t in tasks if t["task_id"] not in done]
print(f"{a.stage}/{a.judge}: tasks {len(tasks)}, already done {len(done)}, to run {len(todo)}", flush=True)
mid = P["judges"][a.judge]; D = P["decoding"]
tok = AutoTokenizer.from_pretrained(mid); model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); model.eval()
LAB = P["reference_stage"]["labels"] if a.stage == "reference" else P["answer_stage"]["labels"]
def parse(t):
    u = t.upper().replace(" ", "_") if a.stage == "reference" else t.upper()
    hits = [(u.find(l), l) for l in LAB if u.find(l) >= 0]
    if a.stage == "reference" and not hits:
        for l, alt in (("VALID_AT_T", "VALID"), ("INVALID_AT_T", "INVALID"), ("TEMPORALLY_AMBIGUOUS", "AMBIGUOUS")):
            if re.search(rf"\b{alt}\b", t.upper()): hits.append((0, l))
        hits = [x for x in hits if not (x[1] == "VALID_AT_T" and re.search(r"\bINVALID\b", t.upper()))] or hits
    return min(hits)[1] if hits else "UNPARSED"
with open(out, "a", encoding="utf-8") as f:
    for i, t in enumerate(todo):
        msgs = [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"]}]
        if a.judge == "mistral7b": msgs = [{"role": "user", "content": t["system"] + "\n\n" + t["user"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text], return_tensors="pt", truncation=True, max_length=D["max_input_tokens"]).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=D["max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        f.write(json.dumps({"task_id": t["task_id"], "query_id": t["query_id"], "judge": a.judge, "model": mid, "raw": raw, "label": parse(raw),
                            "input_tokens": int(enc["input_ids"].shape[1]), "truncated": bool(enc["input_ids"].shape[1] >= D["max_input_tokens"]),
                            "utc": datetime.datetime.utcnow().isoformat() + "Z"}, ensure_ascii=False) + "\n"); f.flush()
        if i % 200 == 0: print(f"  {i}/{len(todo)} last raw={raw!r}", flush=True)
print("done", flush=True)
