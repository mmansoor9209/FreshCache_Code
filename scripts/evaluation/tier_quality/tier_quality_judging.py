"""Run one judge on one task type. Usage: --kind {answer,sufficiency} --judge {llama8b,qwen7b,mistral7b} --gpu N"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, os, json, pathlib, datetime, hashlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True); ap.add_argument("--judge", required=True); ap.add_argument("--gpu", required=True); a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
W = ((OUTPUT_ROOT / "tier_quality"))
AP = json.load(open(REPO_ROOT / "configs/protocols/reference_validity_protocol.json")); C = json.load(open((REPO_ROOT / "configs/protocols/tier_quality_study_config.json")))
assert hashlib.sha256((REPO_ROOT / "configs/protocols/reference_validity_protocol.json").read_bytes()).hexdigest() == C["answer_grading"]["protocol_sha256_expected"]
assert hashlib.sha256(((REPO_ROOT / "configs/protocols/tier_quality_study_config.json")).read_bytes()).hexdigest() == open((REPO_ROOT / "configs/protocols/tier_quality_study_config.sha256")).read().strip()
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
tasks = jl(W/"inputs"/f"{a.kind}_tasks.jsonl"); out = W/"outputs/raw"/f"{a.kind}_{a.judge}.jsonl"
done = {r["task_id"] for r in jl(out)} if out.exists() else set(); todo = [t for t in tasks if t["task_id"] not in done]
mid = AP["judges"][a.judge]; print(f"{a.kind}/{a.judge}: {len(tasks)} tasks, {len(todo)} to run", flush=True)
tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
with open(out, "a", encoding="utf-8") as f:
    for i, t in enumerate(todo):
        if a.kind == "answer":
            S = AP["answer_stage"]; sysm = S["system"]; user = S["user"].format(question=t["question"], target=t["target"], reference=t["reference"], candidate=t["candidate"])
            msgs = [{"role": "user", "content": sysm + "\n\n" + user}] if a.judge == "mistral7b" else [{"role": "system", "content": sysm}, {"role": "user", "content": user}]
            mx = AP["decoding"]["max_new_tokens"]
        else:
            msgs = [{"role": "user", "content": C["sufficiency_grading"]["prompt"].format(q=t["question"], ev=t["evidence"])}]; mx = 8
        enc = tok([tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], return_tensors="pt", truncation=True, max_length=6144).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        f.write(json.dumps({"task_id": t["task_id"], "query_id": t["query_id"], "judge": a.judge, "model": mid, "raw": raw, "input_tokens": int(enc["input_ids"].shape[1]),
                            "utc": datetime.datetime.utcnow().isoformat() + "Z"}, ensure_ascii=False) + "\n"); f.flush()
        if i % 300 == 0: print(f"  {i}/{len(todo)} raw={raw!r}", flush=True)
print("done", flush=True)
