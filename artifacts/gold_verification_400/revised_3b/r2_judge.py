#!/usr/bin/env python3
"""
revised_3b/r2_judge.py — question-type-aware timestamp validity judging.

Two prompts, selected by the deterministic question type from r1_build:

  CURRENT  (607 units) "was the benchmark answer still the current answer when
           this evidence was captured?"      VALID / SUPERSEDED / INDETERMINATE
  AS_STATED (108 units) "is the benchmark answer correct for the time or period
           the question itself specifies?"   VALID / CONTRADICTED / INDETERMINATE

AS_STATED prevents a historically correct, explicitly dated answer from being
called superseded merely because newer information exists.

Judges: llama8b, qwen7b (primary pair, all units) and mistral7b (third,
independent family; only the units where the pair disagrees).
Decoding: chat template, greedy, max_new_tokens=8, bfloat16, input truncation
8192 tokens to accommodate the expanded evidence.

Absence of evidence is never SUPERSEDED/CONTRADICTED: units with no evidence are
never sent to a judge (r1 reports zero such units).
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

HERE = pathlib.Path(__file__).resolve().parent
JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
          "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}

SYS_CURRENT = (
    "You are checking whether a benchmark answer was still up to date at a "
    "specific point in time, using web page text captured at that time.\n\n"
    "Reply with exactly one word:\n"
    "VALID         the evidence indicates the benchmark answer was still the "
    "current answer when the evidence was captured.\n"
    "SUPERSEDED    the evidence indicates a DIFFERENT current answer, so the "
    "benchmark answer was out of date.\n"
    "INDETERMINATE the evidence does not settle the question either way.\n\n"
    "Rules:\n"
    "- Absence of the benchmark answer from the evidence is NOT enough to say "
    "SUPERSEDED. Say SUPERSEDED only if the evidence states a different current "
    "answer to the same question.\n"
    "- Do not assume the benchmark answer is still correct merely because it was "
    "annotated in an earlier year. Judge only from the evidence.\n"
    "- Accept any listed alternative form of the answer as the answer.\n"
    "- If the evidence is off-topic or silent on the point, say INDETERMINATE.")

SYS_AS_STATED = (
    "You are checking whether a benchmark answer is correct FOR THE TIME OR "
    "PERIOD THE QUESTION ITSELF SPECIFIES, using web page text.\n\n"
    "The question refers to a specific date, period, or a one-time historical "
    "fact. The answer does not become wrong just because a later person, holder, "
    "or value exists now.\n\n"
    "Reply with exactly one word:\n"
    "VALID         the evidence supports the benchmark answer for the time or "
    "fact the question asks about.\n"
    "CONTRADICTED  the evidence states a DIFFERENT answer for that same time or "
    "fact.\n"
    "INDETERMINATE the evidence does not settle it.\n\n"
    "Rules:\n"
    "- A later successor, newer record, or more recent holder is NOT a "
    "contradiction. Say CONTRADICTED only if the evidence disagrees about the "
    "very time or fact the question specifies.\n"
    "- Absence of the benchmark answer from the evidence is NOT a contradiction.\n"
    "- Accept any listed alternative form of the answer as the answer.")

USER = ("QUESTION: {q}\n"
        "BENCHMARK ANSWER: {gold}\n"
        "BENCHMARK ANSWER ANNOTATED FOR YEAR: {year}\n"
        "EVIDENCE CAPTURED: {when}, in May-June 2026\n\n"
        "EVIDENCE:\n{ev}")

ROUND_LABEL = {"run_00": "at the baseline crawl",
               "rerun_1h": "1 hour after the baseline crawl",
               "rerun_12h": "12 hours after the baseline crawl",
               "rerun_24h": "24 hours after the baseline crawl",
               "rerun_7d": "7 days after the baseline crawl"}


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def parse(t, mode):
    u = (t or "").upper()
    if "INDETERMIN" in u:
        return "INDETERMINATE_AT_T"
    if mode == "AS_STATED":
        if "CONTRADICT" in u:
            return "CONTRADICTED_AS_STATED"
        if "VALID" in u:
            return "VALID_AT_T"
    else:
        if "SUPERSED" in u:
            return "SUPERSEDED_AT_T"
        if "VALID" in u:
            return "VALID_AT_T"
    return "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(JUDGES))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only-disagreements", action="store_true")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = jl(HERE / "r_validity_tasks.jsonl")
    if a.only_disagreements:
        ids = {r["unit_id"] for r in jl(HERE / "r_tiebreak_units.jsonl")}
        tasks = [t for t in tasks if t["unit_id"] in ids]
        print(f"  tie-break mode: {len(tasks)} units", flush=True)
    out = HERE / f"r_validity_{a.judge}.jsonl"
    done = {r["unit_id"] for r in jl(out)}
    todo = [t for t in tasks if t["has_evidence"] and t["unit_id"] not in done]
    print(f"  [{a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else None
    tok = AutoTokenizer.from_pretrained(JUDGES[a.judge], token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGES[a.judge], dtype=torch.bfloat16, device_map="cuda:0", token=hf)
    model.eval()
    with open(out, "a", encoding="utf-8") as fh:
        for i, t in enumerate(todo, 1):
            g = t["gold_display"]
            alt = [x for x in t["aliases"]
                   if x.strip().lower() != g.strip().lower()]
            if alt:
                g += "  (also accepted: " + "; ".join(alt) + ")"
            sysmsg = SYS_AS_STATED if t["mode"] == "AS_STATED" else SYS_CURRENT
            user = USER.format(q=t["base_query"], gold=g,
                               year=t["gold_year"] or "unknown",
                               when=ROUND_LABEL.get(t["snapshot_round"],
                                                    t["snapshot_round"]),
                               ev=t["evidence"])
            if a.judge == "mistral7b":      # no system role in this template
                msgs = [{"role": "user", "content": sysmsg + "\n\n" + user}]
            else:
                msgs = [{"role": "system", "content": sysmsg},
                        {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            enc = tok([text], return_tensors="pt", truncation=True,
                      max_length=8192).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            raw = tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].strip()
            fh.write(json.dumps({"unit_id": t["unit_id"], "judge": a.judge,
                                 "mode": t["mode"],
                                 "status": parse(raw, t["mode"]), "raw": raw},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"    [{a.judge}] {i}/{len(todo)}", flush=True)
    print(f"  [{a.judge}] done -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
