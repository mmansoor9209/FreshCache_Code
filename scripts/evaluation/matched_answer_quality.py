#!/usr/bin/env python3
"""
scripts/evaluation/matched_answer_quality.py — matched answer-quality comparison.

For every policy, the cached-path answer is generated from the evidence THAT
POLICY actually served, and the fresh-path answer from the incoming query's own
URLs at rerun_24h. Identical generator, prompt, context length, gold source,
sampling design and judge protocol across policies; only the served evidence
differs. Quality numbers are never borrowed between policies.

  cached path   L1 event: the MATCHED query answered from the entry's evidence
                L2 event: the INCOMING query answered from the served URL list
  fresh path    the incoming query answered from its own urls at rerun_24h

Graded against independent benchmark gold with two judges, reported under the
conservative both-must-agree rule and under each judge alone.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/evaluation/matched_answer_quality.py
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import argparse, json, math, os, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
HERE = ((OUTPUT_ROOT / "matched_answers"))
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method"))
import answer_grading as e2e      # noqa: E402

GEN = "meta-llama/Llama-3.2-3B-Instruct"
JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
MAXCTX = 2000
norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
ABST = re.compile(r"^(i (don'?t|do not) know|i'?m not sure|unknown|not (enough|"
                  r"specified|available|provided|mentioned|stated)|the (context|"
                  r"text|passage) does not|there is (no|not enough)|cannot "
                  r"(be )?(determin|answer)|unable to)")
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


def jl(p):
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if pathlib.Path(p).exists() else [])


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k/n, 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, (c-m)/d), min(1.0, (c+m)/d))


def batched(model, tok, prompts, bs, label, maxnew=80):
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s+bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=maxnew, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for t in tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True):
            t = t.strip()
            for sep in (".", "\n"):
                if sep in t:
                    t = t[:t.index(sep)+1].strip(); break
            outs.append(t or "I don't know.")
        if (s//bs) % 10 == 0:
            print(f"      {label} {min(s+bs,len(prompts)):>5}/{len(prompts)}", flush=True)
    return outs


def main(a):
    gold = json.load(open((OUTPUT_ROOT / "gold_answers/reference_answers.json")))
    own = {}
    for m in (json.loads(l) for l in open((DATA_ROOT / "url_manifest.jsonl"))):
        if m.get("run_id") == "run_00" and m.get("snapshot_available"):
            own.setdefault(m["query_id"], []).append((m.get("rank", 99), m["url_hash"]))
    own = {k: [h for _, h in sorted(v)] for k, v in own.items()}
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else ""
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)

    samples = sorted(HERE.glob("matched_sample_*.jsonl"))
    gmodel = tok = None
    for sp in samples:
        tag = sp.stem.replace("matched_sample_", "")
        outf = HERE/f"matched_gen_{tag}.jsonl"
        rows = [r for r in jl(sp) if r.get("has_gold")]
        done = {r["row_id"] for r in jl(outf)}
        todo = [r for r in rows if r["row_id"] not in done]
        print(f"\n  {tag}: {len(rows)} gold-covered, {len(todo)} to generate", flush=True)
        if not todo:
            continue
        if gmodel is None:
            tok = AutoTokenizer.from_pretrained(GEN, token=hf or None)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            gmodel = AutoModelForCausalLM.from_pretrained(
                GEN, dtype=torch.bfloat16, device_map="cuda:0", token=hf or None)
            gmodel.eval()
        for s in range(0, len(todo), 300):
            ch = todo[s:s+300]
            d_p = [FORCED.format(
                       question=(r["matched_query"] if r["tier"] == "L1"
                                 and r.get("matched_query") else r["query"]),
                       context=" ".join(e2e.load_snapshot_text(h, "run_00")
                                        for h in r["evidence_urls"])[:MAXCTX])
                   for r in ch]
            c_p = [FORCED.format(
                       question=r["query"],
                       context=" ".join(e2e.load_snapshot_text(h, "rerun_24h")
                                        for h in own.get(r["query_id"], []))[:MAXCTX])
                   for r in ch]
            d = batched(gmodel, tok, d_p, a.batch_size, f"{tag}-D")
            c = batched(gmodel, tok, c_p, a.batch_size, f"{tag}-C")
            with open(outf, "a", encoding="utf-8") as f:
                for r, dd, cc in zip(ch, d, c):
                    f.write(json.dumps({"row_id": r["row_id"], "policy": tag,
                                        "tier": r["tier"], "query": r["query"],
                                        "query_id": r["query_id"],
                                        "band": r.get("band"),
                                        "fc_incoming": r.get("fc_incoming"),
                                        "answer_d": dd, "answer_c": cc},
                                       ensure_ascii=False)+"\n")
            print(f"    [{min(s+300,len(todo)):>5}/{len(todo)}] {tag}", flush=True)
    if gmodel is not None:
        del gmodel
        torch.cuda.empty_cache()

    # ---------------- grading, two judges ----------------
    allrows, need = [], []
    for sp in samples:
        tag = sp.stem.replace("matched_sample_", "")
        for r in jl(HERE/f"matched_gen_{tag}.jsonl"):
            g = gold[norm(r["query"])]
            r["gold"] = g["answer"]
            for side, key in (("cached", "answer_d"), ("fresh", "answer_c")):
                t = norm(r[key])
                if (not t) or ABST.match(t):
                    r[side+"_pre"] = "ABSTAIN"
                elif any(re.search(r"(?<!\w)"+re.escape(norm(al))+r"(?!\w)", t)
                         for al in [g["answer"]]+list(g.get("aliases") or [])
                         if norm(al) and len(norm(al)) > 1):
                    r[side+"_pre"] = "CORRECT"
                else:
                    r[side+"_pre"] = None
                    need.append((r, side, key, g))
            allrows.append(r)
    print(f"\n  {len(allrows):,} graded pairs; {len(need):,} answers need judging",
          flush=True)
    for jname, mid in JUDGES.items():
        outp = HERE/f"matched_judge_{jname}.jsonl"
        done = {(x["row_id"], x["side"]) for x in jl(outp)}
        todo = [x for x in need if (x[0]["row_id"], x[1]) not in done]
        print(f"  judge {jname}: {len(todo):,} to grade", flush=True)
        if not todo:
            continue
        tk = AutoTokenizer.from_pretrained(mid, token=hf or None)
        if tk.pad_token_id is None:
            tk.pad_token = tk.eos_token
        tk.padding_side = "left"
        jm = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.bfloat16, device_map="cuda:0", token=hf or None)
        jm.eval()
        BS = a.batch_size
        for s in range(0, len(todo), BS):
            ch = todo[s:s+BS]
            pr = [tk.apply_chat_template(
                      [{"role": "user", "content": JUDGE.format(
                          q=r["query"], gold=g["answer"], cand=r[k])}],
                      tokenize=False, add_generation_prompt=True)
                  for r, sd, k, g in ch]
            enc = tk(pr, return_tensors="pt", padding=True, truncation=True,
                     max_length=1024).to(jm.device)
            with torch.no_grad():
                o = jm.generate(**enc, max_new_tokens=6, do_sample=False,
                                pad_token_id=tk.pad_token_id)
            with open(outp, "a", encoding="utf-8") as f:
                for (r, sd, k, g), t in zip(ch, tk.batch_decode(
                        o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    low = t.strip().lower()[:14]
                    f.write(json.dumps({"row_id": r["row_id"], "side": sd,
                                        "label": ("CORRECT" if ("correct" in low
                                                  and "incorrect" not in low)
                                                  else "WRONG")})+"\n")
            if (s//BS) % 20 == 0:
                print(f"    [{min(s+BS,len(todo)):>5}/{len(todo)}]", flush=True)
        del jm
        torch.cuda.empty_cache()

    verdict = {j: {(x["row_id"], x["side"]): x["label"]
                   for x in jl(HERE/f"matched_judge_{j}.jsonl")} for j in JUDGES}

    def resolve(r, side, rule):
        pre = r[side+"_pre"]
        if pre is not None:
            return pre
        k = (r["row_id"], side)
        lv, qv = verdict["llama8b"].get(k), verdict["qwen7b"].get(k)
        if rule == "both":
            return "CORRECT" if (lv == "CORRECT" and qv == "CORRECT") else "WRONG"
        return (lv if rule == "llama" else qv) or "WRONG"

    res = {"n_pairs": len(allrows), "rules": {}}
    for rule, lbl in (("both", "both must agree"), ("llama", "Llama-3.1-8B only"),
                      ("qwen", "Qwen2.5-7B only")):
        print(f"\n{'='*128}\n  MATCHED ANSWER QUALITY — {lbl}\n{'='*128}")
        print(f"  {'policy':<22}{'tier':<6}{'n':>6}{'cached OK':>11}{'rate':>8}"
              f"{'fresh OK':>10}{'rate':>8}{'abst':>6}{'WAI':>5}{'WAI rate':>10}"
              f"{'95% CI':>18}{'FC->CNC':>9}{'rate':>8}")
        blk = {}
        for sp in samples:
            tag = sp.stem.replace("matched_sample_", "")
            pr = [r for r in allrows if r["policy"] == tag]
            for tier in ("L1", "L2", "ALL"):
                rs = [r for r in pr if tier == "ALL" or r["tier"] == tier]
                if not rs:
                    continue
                n = len(rs)
                cc = sum(1 for r in rs if resolve(r, "cached", rule) == "CORRECT")
                fc_ = sum(1 for r in rs if resolve(r, "fresh", rule) == "CORRECT")
                ab = sum(1 for r in rs if resolve(r, "cached", rule) == "ABSTAIN")
                wai = sum(1 for r in rs if resolve(r, "cached", rule) == "WRONG"
                          and resolve(r, "fresh", rule) == "CORRECT")
                fcnc = sum(1 for r in rs if resolve(r, "fresh", rule) == "CORRECT"
                           and resolve(r, "cached", rule) != "CORRECT")
                lo, hi = wilson(wai, n)
                blk.setdefault(tag, {})[tier] = {
                    "n": n, "cached_correct": cc,
                    "cached_correct_rate": round(cc/n, 5),
                    "cached_correct_ci95": [round(x, 5) for x in wilson(cc, n)],
                    "fresh_correct": fc_, "fresh_correct_rate": round(fc_/n, 5),
                    "cached_abstain": ab,
                    "wai": wai, "wai_rate": round(wai/n, 5),
                    "wai_ci95": [round(lo, 5), round(hi, 5)],
                    "fresh_correct_cache_not_correct": fcnc,
                    "fcnc_rate": round(fcnc/n, 5)}
                print(f"  {tag if tier=='L1' or len([t for t in ('L1','L2') if any(r['tier']==t for r in pr)])==1 else '':<22}"
                      f"{tier:<6}{n:>6}{cc:>11}{100*cc/n:>7.1f}%{fc_:>10}"
                      f"{100*fc_/n:>7.1f}%{ab:>6}{wai:>5}{100*wai/n:>9.2f}%"
                      f"{'['+format(100*lo,'.2f')+'%, '+format(100*hi,'.2f')+'%]':>18}"
                      f"{fcnc:>9}{100*fcnc/n:>7.2f}%")
        res["rules"][rule] = blk
    json.dump(res, open(HERE/"matched_answer_quality.json", "w"), indent=2)
    with open(HERE/"matched_answer_grades.jsonl", "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps({k: r.get(k) for k in
                                ("row_id", "policy", "tier", "query", "gold",
                                 "answer_d", "answer_c")}, ensure_ascii=False)+"\n")
    print(f"\n  wrote matched_answer_quality.json, matched_answer_grades.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    main(p.parse_args())
