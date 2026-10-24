"""POST HOC, EXPLORATORY (not pre-registered; written after C results were seen): does each C paraphrase ask the same question as its base?
Two judges (Llama-3.1-8B, Qwen2.5-7B; chat template, greedy, 4 tokens) answer SAME/DIFFERENT with the prompt below. Used only to interpret
the C comparison by stratum; it changes no C exclusion or primary result. Modes: judge --judge J --gpu G | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, os, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); MODE = sys.argv[1]
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
PROMPT = ("Question A: {a}\nQuestion B: {b}\n\nDo Question A and Question B ask for the same piece of information, so that the correct answer to A "
          "is also the correct answer to B? Reply with exactly one word: SAME or DIFFERENT.")
F = json.load(open(O / "work/c_block.json")); REQ = {r["query_id"]: r for r in F["requests"]}
PAIRS = [(q, r["cluster_id"]) for q, r in REQ.items() if r["is_paraphrase"]]
MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
if MODE == "judge":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--judge"); ap.add_argument("--gpu"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[a.judge]); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODELS[a.judge], dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    out = O / f"gpu/out/c_paraeq_{a.judge}.jsonl"; done = {r["query_id"] for r in jl(out)}; todo = [p for p in PAIRS if p[0] not in done]; fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), 16):
        ch = todo[s:s + 16]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": PROMPT.format(a=REQ[b]["query"], b=REQ[q]["query"])}], tokenize=False, add_generation_prompt=True) for q, b in ch],
                  return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        for (q, b), t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            u = t.upper(); fh.write(json.dumps({"query_id": q, "raw": t, "label": "DIFFERENT" if "DIFF" in u else "SAME" if "SAME" in u else "UNPARSED"}) + "\n")
        fh.flush()
    sys.exit(0)
