#!/usr/bin/env python3
"""quality_selection_generation.py — answer generation, protocol identical to both existing audits."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
HERE = ((OUTPUT_ROOT / "quality_selection"))
GEN = "meta-llama/Llama-3.2-3B-Instruct"
FORCED = """You are a factual question answering assistant.
Answer the following question in one sentence using the provided context.
You must commit to a single best answer. Never reply that you do not know and
never state that the context is insufficient. If the context is incomplete,
give the most likely specific answer that the context points toward.

Question: {question}

Context:
{context}

Answer:"""


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch", type=int, default=16); a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = HERE / "q_answers.jsonl"
    done = {tuple(r["key"]) for r in jl(out)}
    todo = [t for t in jl(HERE / "q_gen_tasks.jsonl") if tuple(t["key"]) not in done]
    print(f"  {len(done)} done, {len(todo)} to generate", flush=True)
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
                                     "answer": t or "I don't know."},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            if (s // a.batch) % 10 == 0:
                print(f"    {min(s + a.batch, len(todo))}/{len(todo)}", flush=True)
    print("  generation done", flush=True)


if __name__ == "__main__":
    main()
