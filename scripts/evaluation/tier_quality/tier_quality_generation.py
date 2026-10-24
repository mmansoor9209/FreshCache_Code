"""Generate answers for tasks without a preserved answer, plus a fidelity sample of preserved ones. Usage: --gpu N [--fidelity 48]"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, os, json, pathlib, random; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", required=True); ap.add_argument("--fidelity", type=int, default=48); a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
W = ((OUTPUT_ROOT / "tier_quality"))
FORCED = """You are a factual question answering assistant.
Answer the following question in one sentence using the provided context.
You must commit to a single best answer. Never reply that you do not know and
never state that the context is insufficient. If the context is incomplete,
give the most likely specific answer that the context points toward.

Question: {question}

Context:
{context}

Answer:"""
T = [json.loads(l) for l in open(W/"inputs/generation_tasks.jsonl", encoding="utf-8")]
todo = [t for t in T if not t["preserved"]]; pres = [t for t in T if t["preserved"]]
fid = random.Random(42).sample(pres, min(a.fidelity, len(pres)))
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct"); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
def run(items, path):
    out = []
    with open(path, "w", encoding="utf-8") as fh:
        for s in range(0, len(items), 16):
            ch = items[s:s+16]
            enc = tok([FORCED.format(question=c["question"], context=c["context"]) for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(m.device)
            with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                t = t.strip()
                for sep in (".", "\n"):
                    if sep in t: t = t[:t.index(sep)+1].strip(); break
                r = {"key": c["key"], "answer": t or "I don't know."}; fh.write(json.dumps(r, ensure_ascii=False) + "\n"); out.append(r)
            print(f"  {path.name} {min(s+16, len(items))}/{len(items)}", flush=True)
    return out
f = run(fid, W/"outputs/raw/fidelity_regenerated.jsonl")
same = sum(1 for r, t in zip(f, fid) if r["answer"] == t["preserved"]["answer"])
json.dump({"n": len(fid), "byte_identical": same, "mismatches": [{"key": t["key"], "preserved": t["preserved"]["answer"], "regenerated": r["answer"]} for r, t in zip(f, fid) if r["answer"] != t["preserved"]["answer"]]},
          open(W/"results/generation_fidelity.json", "w"), indent=1)
print(f"FIDELITY {same}/{len(fid)} byte-identical", flush=True)
run(todo, W/"outputs/raw/generated_answers.jsonl"); print("GENERATION DONE", flush=True)
