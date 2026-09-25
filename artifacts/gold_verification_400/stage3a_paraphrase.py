#!/usr/bin/env python3
"""
validation/gold_verification_400/stage3a_paraphrase.py

STAGE 3a — PARAPHRASE FIDELITY VERIFICATION (label A of the Revision 4 plan).

For each paraphrase request in the two 400-request audits, decide whether the
paraphrase preserves the original question's meaning AND the answer it requires,
so that the base question's gold answer would also be the correct answer to the
paraphrase.

  judges   meta-llama/Llama-3.1-8B-Instruct  (primary)
           Qwen/Qwen2.5-7B-Instruct          (independent second judge)
  decoding chat template, greedy, do_sample=False, max_new_tokens=6, bfloat16
  labels   SAME / DIFFERENT / UNCLEAR; a label is assigned only when BOTH
           judges agree, otherwise the request is JUDGE_DISAGREEMENT

The 12 requests that already carry a fidelity judgment in
data/paraphrase_fidelity_judgments.jsonl keep that stored verdict as their
final label, per the approved requirement. They are ALSO judged by the two new
judges so the old 3B binary verdict can be compared against the new pair; that
comparison is reported, never used to overwrite the stored verdict.

A SAME judgment means the two questions seek the same fact. It does NOT
establish that the gold answer is correct, and no such claim is made here.

READ-ONLY on every pre-existing artifact. No answer generation, no web calls.
All output lands in this directory.

Stages:  judge --judge {llama8b,qwen7b} --gpu N   |   analyze
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
PRIMARY = "llama8b"
LABELS = ("SAME", "DIFFERENT", "UNCLEAR")

# Semantics preserved from the original judge_paraphrase_fidelity.py SYSTEM_PROMPT,
# with UNCLEAR added as an explicit third option.
SYSTEM_PROMPT = (
    "You are evaluating a semantic cache for a question-answering system. "
    "You will be given two questions. Decide whether a single correct answer "
    "would satisfy BOTH questions, so that serving the stored answer for "
    "Question A in response to Question B would be correct.\n\n"
    "Rules:\n"
    "- SAME       if both questions ask for the same fact about the same "
    "entity, even if worded very differently or in different languages.\n"
    "- DIFFERENT  if they ask about different entities, different attributes "
    "of the same entity, different time references, or different events.\n"
    "- DIFFERENT  if Question B is garbled, incoherent, or so vague that it no "
    "longer asks for the specific fact Question A asks for.\n"
    "- UNCLEAR    only if Question B cannot be interpreted well enough to "
    "decide either way.\n\n"
    "Reply with exactly one word: SAME, DIFFERENT, or UNCLEAR."
)
USER_TEMPLATE = "Question A: {a}\nQuestion B: {b}"

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def parse_label(t):
    u = (t or "").upper()
    for lab in ("DIFFERENT", "UNCLEAR", "SAME"):
        if lab in u:
            return lab
    return "UNPARSED"


def pairs():
    ps = jl(HERE / "paraphrase_pairs.jsonl")
    assert ps, "paraphrase_pairs.jsonl missing — run Stage 1 first"
    return ps


# -------------------------------------------------------------------- judge --
def judge(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = args.judge
    out = HERE / f"paraphrase_labels_{name}.jsonl"
    done = {r["query_id"] for r in jl(out)}
    todo = [p for p in pairs() if p["query_id"] not in done]
    say(f"  [{name}] {len(done)} done, {len(todo)} to judge")
    if not todo:
        return
    tokp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tokp.read_text().strip() if tokp.exists() else None
    tok = AutoTokenizer.from_pretrained(JUDGES[name], token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGES[name], dtype=torch.bfloat16, device_map="cuda:0", token=hf)
    model.eval()
    with open(out, "a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        a=p["base_query"], b=p["paraphrase"])}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            enc = tok([text], return_tensors="pt", truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            raw = tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].strip()
            fh.write(json.dumps({"query_id": p["query_id"], "judge": name,
                                 "label": parse_label(raw), "raw": raw},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 100 == 0:
                say(f"    [{name}] {i}/{len(todo)}")
    say(f"  [{name}] done -> {out.name}")


# ------------------------------------------------------------------ analyze --
def kappa(a, b, cats):
    n = len(a)
    if not n:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def analyze(_args):
    ps = {p["query_id"]: p for p in pairs()}
    L = {r["query_id"]: r["label"] for r in jl(HERE / "paraphrase_labels_llama8b.jsonl")}
    Q = {r["query_id"]: r["label"] for r in jl(HERE / "paraphrase_labels_qwen7b.jsonl")}
    missing = [q for q in ps if q not in L or q not in Q]
    if missing:
        say(f"  STOP: {len(missing)} pairs not judged by both models")
        sys.exit(3)
    say(f"  both judges present for all {len(ps)} paraphrase requests")

    rmap = {r["query_id"]: r for r in csv.DictReader(
        open(HERE / "request_map.csv", encoding="utf-8"))}

    rows = []
    for q, p in ps.items():
        l, w = L[q], Q[q]
        agree = (l == w)
        reused = bool(p["existing_verdict"])
        final = (p["existing_verdict"] if reused
                 else (l if agree else "JUDGE_DISAGREEMENT"))
        r = rmap[q]
        rows.append({
            "query_id": q, "audits": r["audits"],
            "base_query_id": p["base_query_id"],
            "base_query": p["base_query"], "paraphrase": p["paraphrase"],
            "gold_source": r["gold_source"], "gold_year": r["gold_year"],
            "freshness_class": p["freshness_class"],
            "label_llama8b": l, "label_qwen7b": w,
            "judges_agree": int(agree),
            "reused_existing": int(reused),
            "existing_verdict": p["existing_verdict"] or "",
            "final_label": final,
        })

    new = [r for r in rows if not r["reused_existing"]]
    reu = [r for r in rows if r["reused_existing"]]
    say(f"\n  reused existing fidelity judgments: {len(reu)}")
    say(f"  newly judged by both models: {len(new)}")

    def block(sub, title):
        n = len(sub)
        if not n:
            return {}
        ag = sum(r["judges_agree"] for r in sub)
        fl = Counter(r["final_label"] for r in sub)
        b = {"n": n,
             "judges_agree": ag,
             "judges_agree_pct": round(100 * ag / n, 2),
             "judge_disagreement": n - ag,
             "SAME": fl.get("SAME", 0), "DIFFERENT": fl.get("DIFFERENT", 0),
             "UNCLEAR": fl.get("UNCLEAR", 0),
             "JUDGE_DISAGREEMENT": fl.get("JUDGE_DISAGREEMENT", 0),
             "UNPARSED": fl.get("UNPARSED", 0),
             "llama8b": dict(Counter(r["label_llama8b"] for r in sub)),
             "qwen7b": dict(Counter(r["label_qwen7b"] for r in sub)),
             "cohens_kappa": kappa([r["label_llama8b"] for r in sub],
                                   [r["label_qwen7b"] for r in sub], LABELS)}
        say(f"    {title:<34} n {n:>4}  SAME {b['SAME']:>4}  DIFFERENT "
            f"{b['DIFFERENT']:>4}  UNCLEAR {b['UNCLEAR']:>3}  "
            f"DISAGREE {b['JUDGE_DISAGREEMENT']:>3}  agree "
            f"{b['judges_agree_pct']:>5.1f}%  kappa {b['cohens_kappa']}")
        return b

    say("\n  OVERALL (final labels; the 12 reused keep their stored verdict)")
    overall = block(rows, "all paraphrase requests")
    newly = block(new, "newly judged only")

    say("\n  BY AUDIT")
    by_audit = {}
    for a in ("mixed_age_400", "heldout_400"):
        by_audit[a] = block([r for r in rows if a in r["audits"]], a)
    both = block([r for r in rows if "|" in r["audits"]], "in both audits")

    say("\n  BY SOURCE DATASET")
    by_src = {}
    for s in sorted({r["gold_source"] for r in rows if r["gold_source"]}):
        by_src[s] = block([r for r in rows if r["gold_source"] == s], s)

    say("\n  BY FRESHNESS CLASS")
    by_fc = {}
    for f in sorted({r["freshness_class"] for r in rows if r["freshness_class"]}):
        by_fc[f] = block([r for r in rows if r["freshness_class"] == f], f)

    # confusion matrix
    conf = Counter((r["label_llama8b"], r["label_qwen7b"]) for r in rows)
    say("\n  JUDGE CONFUSION MATRIX (rows Llama-3.1-8B, cols Qwen2.5-7B)")
    cats = sorted({c for p_ in conf for c in p_})
    say("    " + " " * 12 + "".join(f"{c:>12}" for c in cats))
    for a in cats:
        say(f"    {a:<12}" + "".join(f"{conf.get((a, b), 0):>12}" for b in cats))

    # old 3B verdict vs new pair, on the 12 reused
    if reu:
        say("\n  OVERLAP CHECK — stored 3B verdict vs the two new judges (12 reused)")
        for r in reu:
            say(f"    {r['query_id']:<18} stored {r['existing_verdict']:<10} "
                f"llama8b {r['label_llama8b']:<10} qwen7b {r['label_qwen7b']}")
        ol = {"stored_vs_llama8b_agree":
                  sum(1 for r in reu if r["existing_verdict"] == r["label_llama8b"]),
              "stored_vs_qwen7b_agree":
                  sum(1 for r in reu if r["existing_verdict"] == r["label_qwen7b"]),
              "n": len(reu),
              "note": "the stored verdicts come from Llama-3.2-3B-Instruct with a "
                      "binary SAME/DIFFERENT prompt; they are not directly "
                      "comparable to the new 8B/7B three-way labels"}
        say(f"    stored == llama8b on {ol['stored_vs_llama8b_agree']}/{len(reu)}, "
            f"stored == qwen7b on {ol['stored_vs_qwen7b_agree']}/{len(reu)}")
    else:
        ol = {}

    with open(HERE / "paraphrase_fidelity.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    json.dump({
        "stage": "3a — paraphrase fidelity",
        "judges": JUDGES, "primary_judge": PRIMARY,
        "decoding": "chat template, greedy, do_sample=False, max_new_tokens=6, bfloat16",
        "labels": list(LABELS),
        "agreement_rule": "a label is assigned only when both judges agree; "
                          "otherwise JUDGE_DISAGREEMENT",
        "prompt_system": SYSTEM_PROMPT,
        "paraphrase_requests": len(rows),
        "reused_existing_judgments": len(reu),
        "newly_judged": len(new),
        "overall": overall, "newly_judged_only": newly,
        "by_audit": by_audit, "in_both_audits": both,
        "by_source_dataset": by_src, "by_freshness_class": by_fc,
        "judge_confusion_matrix": {f"{a}|{b}": c for (a, b), c in conf.items()},
        "reused_overlap_check": ol,
        "interpretation_limit": (
            "A SAME judgment means the two questions seek the same fact. It does "
            "NOT establish that the gold answer is correct, nor that it was valid "
            "at the request timestamp."),
        "answer_generation": 0, "web_calls": 0,
    }, open(HERE / "stage3a_summary.json", "w"), indent=2, ensure_ascii=False)

    say("\n  LIMIT: SAME means the two questions seek the same fact. It does NOT")
    say("  establish gold-answer correctness, and none is claimed.")
    (HERE / "stage3a.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote paraphrase_fidelity.csv, stage3a_summary.json, stage3a.log")
    say("  STOP — Stages 3b, 3c and 4 not started, as instructed.")
    (HERE / "stage3a.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["judge", "analyze"])
    ap.add_argument("--judge", choices=list(JUDGES))
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    if a.stage == "judge":
        assert a.judge, "--judge required"
        judge(a)
    else:
        analyze(a)
