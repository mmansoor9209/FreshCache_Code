#!/usr/bin/env python3
"""quality_selection_judging.py — answer judging, protocol identical to both existing audits."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, hashlib, json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
HERE = ((OUTPUT_ROOT / "quality_selection"))
JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
JUDGE = """You are grading a short answer against the reference answer.

QUESTION: {q}
REFERENCE ANSWER: {gold}
CANDIDATE ANSWER: {cand}

Does the candidate convey the same factual answer as the reference? Minor
wording, extra context, or added detail do not matter. Reply with exactly one
word: CORRECT or INCORRECT."""
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def items():
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "q_answers.jsonl")}
    arms = [a for a in json.load(open(HERE / "quality_sample_summary.json",
                                      encoding="utf-8"))["arms"]]
    out, seen = [], set()
    for d in jl(HERE / "q_sample.jsonl"):
        keys = [("fresh", d["fresh_ctx_sha"], h(d["query"]))]
        keys += [tuple(d[f"{a}_key"]) for a in arms]
        for k in keys:
            a_ = ans.get(tuple(k))
            if a_ is None:
                continue
            sig = (h(d["query"]), h(d["gold"]), h(a_))
            if sig in seen:
                continue
            seen.add(sig)
            out.append({"sig": list(sig), "q": d["query"], "gold": d["gold"],
                        "cand": a_})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(JUDGES))
    ap.add_argument("--gpu", default="0"); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = HERE / f"q_judge_{a.judge}.jsonl"
    done = {tuple(r["sig"]) for r in jl(out)}
    todo = [it for it in items() if tuple(it["sig"]) not in done]
    print(f"  [{a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else None
    tok = AutoTokenizer.from_pretrained(JUDGES[a.judge], token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        JUDGES[a.judge], dtype=torch.bfloat16, device_map="cuda:0", token=hf)
    model.eval()
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch):
            ch = todo[s:s + a.batch]
            tx = [tok.apply_chat_template(
                [{"role": "user", "content": JUDGE.format(
                    q=c["q"], gold=c["gold"], cand=c["cand"])}],
                tokenize=False, add_generation_prompt=True) for c in ch]
            enc = tok(tx, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(
                    g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                u = t.upper()
                lab = ("INCORRECT" if "INCORRECT" in u
                       else ("CORRECT" if "CORRECT" in u else "UNPARSED"))
                fh.write(json.dumps({"sig": c["sig"], "judge": a.judge,
                                     "label": lab}) + "\n")
            fh.flush()
            if (s // a.batch) % 10 == 0:
                print(f"    [{a.judge}] {min(s + a.batch, len(todo))}/{len(todo)}",
                      flush=True)
    print(f"  [{a.judge}] done", flush=True)


if __name__ == "__main__":
    main()
