#!/usr/bin/env python3
"""
validation/gold_verification_400/stage3b_validity.py

STAGE 3b — TIMESTAMP-AWARE GOLD-ANSWER VALIDITY (label C of the Revision 4 plan).

For each of the 715 (base query, snapshot round) verification units, assemble the
FreshCache-Bench historical page snapshots captured at that unit's own round and
ask two judges, independently, whether the benchmark gold answer was still the
current answer AT THAT TIME.

  VALID_AT_T          the evidence indicates the gold was current at that time
  SUPERSEDED_AT_T     the evidence indicates a DIFFERENT current answer
  INDETERMINATE_AT_T  the evidence does not settle it

  judges   meta-llama/Llama-3.1-8B-Instruct  (primary)
           Qwen/Qwen2.5-7B-Instruct          (independent second judge)
  decoding chat template, greedy, do_sample=False, max_new_tokens=6, bfloat16

Rules enforced in code, not merely in the prompt:
  * MISSING EVIDENCE IS NEVER SUPERSEDED. A unit with no retrievable snapshot
    text is assigned INDETERMINATE_AT_T with reason NO_EVIDENCE and is not sent
    to a judge at all.
  * A 2023 annotation is never assumed to hold in 2026. The judge is told the
    annotation year and the evidence date and is asked about the evidence date.
  * A status is assigned only when BOTH judges agree; otherwise the unit is
    JUDGE_DISAGREEMENT and stays unresolved.

Evidence assembly reuses the audits' own loader and budget: per-page cap 2000
chars (e2e_answer_grading.MAX_CONTEXT_CHARS), pages under 200 chars dropped,
joined under a 2800-char budget — identical to prep2.ctx_for.

READ-ONLY on every pre-existing artifact. No web retrieval, no page fetching, no
answer generation. All output lands in this directory.

Stages:  build | judge --judge {llama8b,qwen7b} --gpu N | analyze
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import sys
from collections import Counter, defaultdict

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v9", "v16_exp12", "v14_baselines"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

MAXCHARS, MINCHARS = 2800, 200
CORPUS_DATE = "May-June 2026"
ROUND_LABEL = {"run_00": "the baseline crawl",
               "rerun_1h": "1 hour after the baseline crawl",
               "rerun_12h": "12 hours after the baseline crawl",
               "rerun_24h": "24 hours after the baseline crawl",
               "rerun_7d": "7 days after the baseline crawl"}

JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
PRIMARY = "llama8b"
STATUSES = ("VALID_AT_T", "SUPERSEDED_AT_T", "INDETERMINATE_AT_T")

SYSTEM_PROMPT = (
    "You are checking whether a benchmark answer was still up to date at a "
    "specific point in time, using web page text captured at that time.\n\n"
    "You are given a QUESTION, a BENCHMARK ANSWER with the year it was "
    "annotated, and EVIDENCE consisting of web page text captured later, in "
    f"{CORPUS_DATE}. Decide what the evidence says about the benchmark answer "
    "AT THE TIME THE EVIDENCE WAS CAPTURED.\n\n"
    "Reply with exactly one word:\n"
    "VALID        the evidence indicates the benchmark answer was still the "
    "current answer when the evidence was captured.\n"
    "SUPERSEDED   the evidence indicates a DIFFERENT current answer, so the "
    "benchmark answer was out of date.\n"
    "INDETERMINATE the evidence does not settle the question either way.\n\n"
    "Important:\n"
    "- Absence of the benchmark answer from the evidence is NOT enough to say "
    "SUPERSEDED. Say SUPERSEDED only if the evidence states a different "
    "current answer.\n"
    "- Do not assume the benchmark answer is still correct merely because it "
    "was annotated in an earlier year. Judge only from the evidence.\n"
    "- If the evidence is off-topic, incomplete, or silent on the point, say "
    "INDETERMINATE."
)

USER_TEMPLATE = (
    "QUESTION: {q}\n"
    "BENCHMARK ANSWER: {gold}\n"
    "BENCHMARK ANSWER ANNOTATED FOR YEAR: {year}\n"
    "EVIDENCE CAPTURED: {when}\n\n"
    "EVIDENCE:\n{ev}"
)

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


def parse_status(t):
    u = (t or "").upper()
    if "INDETERMIN" in u:
        return "INDETERMINATE_AT_T"
    if "SUPERSED" in u:
        return "SUPERSEDED_AT_T"
    if "VALID" in u:
        return "VALID_AT_T"
    return "UNPARSED"


# -------------------------------------------------------------------- build --
def build(_args):
    import experiment as exp
    import e2e_answer_grading as e2e

    units = list(csv.DictReader(open(HERE / "verification_units.csv",
                                     encoding="utf-8")))
    rmap = {r["query_id"]: r for r in csv.DictReader(
        open(HERE / "request_map.csv", encoding="utf-8"))}
    vol = {r["base_query_id"]: r for r in csv.DictReader(
        open(HERE / "gold_temporal_volatility.csv", encoding="utf-8"))}
    gold = json.load(open(ROOT / "v10_remaining_feedback" / "c2_gold_answers.json",
                          encoding="utf-8"))
    L = lambda p: exp.load_jsonl(pathlib.Path(p))              # noqa: E731
    recs = exp.build_query_records(
        L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
        L(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else [])
    urls_of = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in recs}
    norm_q = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())  # noqa: E731

    tasks, stats = [], Counter()
    for u in units:
        rd = u["snapshot_round"]
        members = u["request_ids"].split("|")
        uh = []
        for m in members:
            for x in urls_of.get(m, []):
                if x not in uh:
                    uh.append(x)
        parts = []
        for x in uh:
            t = e2e.load_snapshot_text(x, rd) or ""      # per-page 2000-char cap
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) >= MINCHARS:
                parts.append(t)
        keep, tot = [], 0
        for t in parts:
            room = MAXCHARS - tot
            if room <= 0:
                break
            keep.append(t[:room])
            tot += min(len(t), room)
        ev = "\n\n".join(keep)
        v = vol.get(u["base_query_id"], {})
        g = gold.get(norm_q(u["base_query"])) or {}
        aliases = [a for a in (g.get("aliases") or []) if a]
        fcs = sorted({rmap[m]["freshness_class"] for m in members if m in rmap})
        rec = {"unit_id": u["unit_id"], "base_query_id": u["base_query_id"],
               "base_query": u["base_query"], "snapshot_round": rd,
               "gold": u["gold"], "gold_display": (g.get("answer") or u["gold"]),
               "aliases": aliases[:6],
               "gold_source": u["gold_source"],
               "gold_year": u["gold_year"] or v.get("original_gold_year", ""),
               "volatility": v.get("volatility", ""),
               "freshness_classes": "|".join(fcs),
               "n_requests": int(u["n_requests"]),
               "request_ids": u["request_ids"],
               "n_urls": len(uh), "n_pages_kept": len(keep),
               "evidence_chars": len(ev), "evidence": ev,
               "has_evidence": int(bool(ev))}
        stats["units"] += 1
        stats["with_evidence" if ev else "no_evidence"] += 1
        tasks.append(rec)

    with open(HERE / "validity_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    say(f"  built {stats['units']} validity tasks")
    say(f"    with evidence: {stats['with_evidence']}   "
        f"NO evidence (forced INDETERMINATE, never judged): {stats['no_evidence']}")
    pg = Counter(t["n_pages_kept"] for t in tasks)
    say(f"    pages of evidence per unit: {dict(sorted(pg.items()))}")
    ch = [t["evidence_chars"] for t in tasks if t["evidence_chars"]]
    if ch:
        ch.sort()
        say(f"    evidence chars: median {ch[len(ch)//2]}, max {ch[-1]}")


# -------------------------------------------------------------------- judge --
def judge(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = args.judge
    out = HERE / f"validity_labels_{name}.jsonl"
    done = {r["unit_id"] for r in jl(out)}
    tasks = [t for t in jl(HERE / "validity_tasks.jsonl")
             if t["has_evidence"] and t["unit_id"] not in done]
    skipped = [t for t in jl(HERE / "validity_tasks.jsonl") if not t["has_evidence"]]
    say(f"  [{name}] {len(done)} done, {len(tasks)} to judge, "
        f"{len(skipped)} skipped as NO_EVIDENCE (never sent to a judge)")
    if not tasks:
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
        for i, t in enumerate(tasks, 1):
            gold = t["gold_display"]
            if t["aliases"] and any(a.strip().lower() != gold.strip().lower()
                                    for a in t["aliases"]):
                gold = gold + "  (also accepted: " + "; ".join(
                    a for a in t["aliases"] if a.strip().lower()
                    != t["gold_display"].strip().lower()) + ")"
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        q=t["base_query"], gold=gold,
                        year=t["gold_year"] or "unknown",
                        when=ROUND_LABEL.get(t["snapshot_round"],
                                             t["snapshot_round"]) +
                             f", in {CORPUS_DATE}",
                        ev=t["evidence"])}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            enc = tok([text], return_tensors="pt", truncation=True,
                      max_length=4096).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            raw = tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].strip()
            fh.write(json.dumps({"unit_id": t["unit_id"], "judge": name,
                                 "status": parse_status(raw), "raw": raw},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 100 == 0:
                say(f"    [{name}] {i}/{len(tasks)}")
    say(f"  [{name}] done -> {out.name}")


# ------------------------------------------------------------------ analyze --
ABSTAIN_PRIMARY = ("i don't know", "i dont know", "cannot determine",
                   "not enough information", "insufficient", "no information",
                   "unclear from the context")
ABSTAIN_HELDOUT = ("i don't know", "i do not know", "cannot be determined",
                   "not enough information", "insufficient",
                   "unable to determine", "no information")
h16 = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]  # noqa: E731


def wai_request_ids():
    """Distinct WAI request ids per audit, under each audit's OWN abstain rule."""
    A1 = ROOT / "validation" / "mixed_age_full_policy_audit"
    A1B = ROOT / "validation" / "baseline_operating_points"
    A2 = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"

    def outc(ans, j1, j2, q, gold, key, pat):
        a = ans.get(tuple(key))
        if a is None:
            return None
        if any(p in a.strip().lower() for p in pat):
            return "ABSTAIN"
        s = (h16(q), h16(gold), h16(a))
        l1, l2 = j1.get(s), j2.get(s)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    s1 = jl(A1 / "sample400.jsonl")
    a1 = {tuple(r["key"]): r["answer"] for r in jl(A1 / "answers2.jsonl")}
    for r in jl(A1B / "sttl_answers.jsonl"):
        a1[tuple(r["key"])] = r["answer"]
    j1 = {tuple(r["sig"]): r["label"] for r in jl(A1 / "judge2_llama8b.jsonl")}
    j2 = {tuple(r["sig"]): r["label"] for r in jl(A1 / "judge2_qwen7b.jsonl")}
    for r in jl(A1B / "sttl_judge_llama8b.jsonl"):
        j1[tuple(r["sig"])] = r["label"]
    for r in jl(A1B / "sttl_judge_qwen7b.jsonl"):
        j2[tuple(r["sig"])] = r["label"]
    ss = {d["query_id"]: d for d in jl(A1B / "sttl_sample400.jsonl")}
    w1 = set()
    for d in s1:
        fr = outc(a1, j1, j2, d["query"], d["gold"],
                  ("fresh", d["fresh_ctx_sha"], h16(d["query"])), ABSTAIN_PRIMARY)
        if fr != "CORRECT":
            continue
        keys = [d["on_key"], d["off_key"]]
        sd = ss.get(d["query_id"])
        if sd:
            keys.append(sd["sttl_key"])
        for k in keys:
            if outc(a1, j1, j2, d["query"], d["gold"], k, ABSTAIN_PRIMARY) == "WRONG":
                w1.add(d["query_id"])
                break

    s2 = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    a2 = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
    k1 = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_llama8b.jsonl")}
    k2 = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_qwen7b.jsonl")}
    w2 = set()
    for d in s2:
        if outc(a2, k1, k2, d["query"], d["gold"], d["fresh_key"],
                ABSTAIN_HELDOUT) != "CORRECT":
            continue
        for k in (d["fc_key"], d["sttl_key"]):
            if outc(a2, k1, k2, d["query"], d["gold"], k, ABSTAIN_HELDOUT) == "WRONG":
                w2.add(d["query_id"])
                break
    return w1, w2


def kappa(a, b, cats):
    n = len(a)
    if not n:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def analyze(_args):
    tasks = {t["unit_id"]: t for t in jl(HERE / "validity_tasks.jsonl")}
    L = {r["unit_id"]: r["status"] for r in jl(HERE / "validity_labels_llama8b.jsonl")}
    Q = {r["unit_id"]: r["status"] for r in jl(HERE / "validity_labels_qwen7b.jsonl")}
    rows = []
    for uid, t in tasks.items():
        if not t["has_evidence"]:
            rows.append({**{k: t[k] for k in
                            ("unit_id", "base_query_id", "base_query",
                             "snapshot_round", "gold", "gold_source",
                             "gold_year", "volatility", "freshness_classes",
                             "n_requests", "request_ids", "n_urls",
                             "n_pages_kept", "evidence_chars")},
                         "status_llama8b": "", "status_qwen7b": "",
                         "judges_agree": "", "final_status": "INDETERMINATE_AT_T",
                         "reason": "NO_EVIDENCE"})
            continue
        l, w = L.get(uid), Q.get(uid)
        if l is None or w is None:
            say(f"  STOP: unit {uid} missing a judge label")
            sys.exit(3)
        agree = (l == w)
        rows.append({**{k: t[k] for k in
                        ("unit_id", "base_query_id", "base_query",
                         "snapshot_round", "gold", "gold_source", "gold_year",
                         "volatility", "freshness_classes", "n_requests",
                         "request_ids", "n_urls", "n_pages_kept",
                         "evidence_chars")},
                     "status_llama8b": l, "status_qwen7b": w,
                     "judges_agree": int(agree),
                     "final_status": l if agree else "JUDGE_DISAGREEMENT",
                     "reason": "dual_judge_agreement" if agree
                               else "judges_disagree"})
    say(f"  units: {len(rows)}")

    judged = [r for r in rows if r["reason"] != "NO_EVIDENCE"]

    def block(sub, title):
        n = len(sub)
        if not n:
            return {}
        c = Counter(r["final_status"] for r in sub)
        jd = [r for r in sub if r["reason"] != "NO_EVIDENCE"]
        ag = sum(r["judges_agree"] for r in jd) if jd else 0
        b = {"n": n, "VALID_AT_T": c.get("VALID_AT_T", 0),
             "SUPERSEDED_AT_T": c.get("SUPERSEDED_AT_T", 0),
             "INDETERMINATE_AT_T": c.get("INDETERMINATE_AT_T", 0),
             "JUDGE_DISAGREEMENT": c.get("JUDGE_DISAGREEMENT", 0),
             "no_evidence": sum(1 for r in sub if r["reason"] == "NO_EVIDENCE"),
             "judged": len(jd),
             "judges_agree_pct": round(100 * ag / len(jd), 2) if jd else None,
             "cohens_kappa": kappa([r["status_llama8b"] for r in jd],
                                   [r["status_qwen7b"] for r in jd], STATUSES)}
        say(f"    {title:<26} n {n:>4}  VALID {b['VALID_AT_T']:>4}  SUPERSEDED "
            f"{b['SUPERSEDED_AT_T']:>4}  INDET {b['INDETERMINATE_AT_T']:>4}  "
            f"DISAGREE {b['JUDGE_DISAGREEMENT']:>4}  agree "
            f"{b['judges_agree_pct']}%  kappa {b['cohens_kappa']}")
        return b

    say("\n  OVERALL (715 verification units)")
    overall = block(rows, "all units")

    say("\n  BY FRESHNESS CLASS (unit may span classes; counted per class)")
    by_fc = {}
    for f in ("TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"):
        sub = [r for r in rows if f in (r["freshness_classes"] or "").split("|")]
        if sub:
            by_fc[f] = block(sub, f)

    say("\n  BY GOLD SOURCE")
    by_src = {}
    for s in sorted({r["gold_source"] for r in rows if r["gold_source"]}):
        by_src[s] = block([r for r in rows if r["gold_source"] == s], s)

    say("\n  BY ANNOTATION YEAR")
    by_year = {}
    for y in sorted({r["gold_year"] or "(none)" for r in rows}):
        by_year[y] = block([r for r in rows
                            if (r["gold_year"] or "(none)") == y], y)

    say("\n  BY TEMPORAL VOLATILITY (Stage 2)")
    by_vol = {}
    for v in sorted({r["volatility"] for r in rows if r["volatility"]}):
        by_vol[v] = block([r for r in rows if r["volatility"] == v], v)

    # ---- WAI requests, reported separately ----
    w1, w2 = wai_request_ids()
    wai_all = w1 | w2
    say(f"\n  WAI REQUESTS — primary {len(w1)}, held-out {len(w2)}, "
        f"distinct union {len(wai_all)}")
    unit_of = {}
    for r in rows:
        for q in r["request_ids"].split("|"):
            unit_of.setdefault(q, []).append(r)
    wai_units, missing_wai = [], []
    for q in sorted(wai_all):
        us = unit_of.get(q)
        if not us:
            missing_wai.append(q)
            continue
        for u in us:
            if u["unit_id"] not in {x["unit_id"] for x in wai_units}:
                wai_units.append(u)
    say(f"    they map to {len(wai_units)} distinct verification units"
        + (f"; {len(missing_wai)} request(s) unmapped" if missing_wai else ""))
    wai_block = block(wai_units, "WAI units")
    say("    WAI request-level status (a request inherits its unit's status)")
    per_req = Counter()
    for q in sorted(wai_all):
        us = unit_of.get(q)
        if us:
            per_req[us[0]["final_status"]] += 1
    for k, v in sorted(per_req.items()):
        say(f"      {k:<22} {v}")
    say(f"    primary-only WAI requests: {len(w1 - w2)}   "
        f"held-out-only: {len(w2 - w1)}   both: {len(w1 & w2)}")

    conf = Counter((r["status_llama8b"], r["status_qwen7b"]) for r in judged)
    say("\n  JUDGE CONFUSION MATRIX (rows Llama-3.1-8B, cols Qwen2.5-7B)")
    cats = sorted({c for p_ in conf for c in p_})
    say("    " + " " * 22 + "".join(f"{c:>22}" for c in cats))
    for a in cats:
        say(f"    {a:<22}" + "".join(f"{conf.get((a, b), 0):>22}" for b in cats))

    with open(HERE / "gold_validity_at_t.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(HERE / "wai_unit_validity.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(wai_units[0]))
        w.writeheader()
        w.writerows(wai_units)

    json.dump({
        "stage": "3b — timestamp-aware gold-answer validity",
        "judges": JUDGES, "primary_judge": PRIMARY,
        "decoding": "chat template, greedy, do_sample=False, max_new_tokens=6, bfloat16",
        "statuses": list(STATUSES),
        "agreement_rule": "a status is assigned only when both judges agree; "
                          "otherwise JUDGE_DISAGREEMENT",
        "missing_evidence_rule": "a unit with no retrievable snapshot text is "
                                 "INDETERMINATE_AT_T with reason NO_EVIDENCE and "
                                 "is never sent to a judge; absence of evidence is "
                                 "never SUPERSEDED",
        "evidence": {"source": "FreshCache-Bench historical snapshots at each "
                               "unit's own round",
                     "per_page_cap_chars": 2000, "min_page_chars": MINCHARS,
                     "budget_chars": MAXCHARS,
                     "assembly": "identical to prep2.ctx_for used by the audits"},
        "prompt_system": SYSTEM_PROMPT,
        "units": len(rows), "judged_units": len(judged),
        "overall": overall, "by_freshness_class": by_fc,
        "by_gold_source": by_src, "by_annotation_year": by_year,
        "by_volatility": by_vol,
        "wai": {"primary_requests": len(w1), "heldout_requests": len(w2),
                "distinct_requests": len(wai_all),
                "distinct_units": len(wai_units),
                "unmapped_requests": missing_wai,
                "unit_status": wai_block,
                "request_status": dict(per_req)},
        "judge_confusion_matrix": {f"{a}|{b}": c for (a, b), c in conf.items()},
        "interpretation_limits": [
            "VALID_AT_T means the snapshot evidence indicates the answer was "
            "current when captured. It is not a correctness verification.",
            "INDETERMINATE_AT_T includes units with no evidence; it never implies "
            "the gold is wrong.",
            "A gold annotated in 2023 is not assumed correct in 2026; the judge "
            "is asked about the evidence date only."],
        "web_calls": 0, "page_fetches": 0, "answer_generation": 0,
    }, open(HERE / "stage3b_summary.json", "w"), indent=2, ensure_ascii=False)

    say("\n  LIMITS: VALID_AT_T is evidence-based temporal validity, not a")
    say("  correctness verification. INDETERMINATE_AT_T never implies the gold")
    say("  is wrong. A 2023 annotation is not assumed to hold in 2026.")
    (HERE / "stage3b.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote gold_validity_at_t.csv, wai_unit_validity.csv, "
        "stage3b_summary.json, stage3b.log")
    say("  STOP — Stages 3c and 4 not started, as instructed.")
    (HERE / "stage3b.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "judge", "analyze"])
    ap.add_argument("--judge", choices=list(JUDGES))
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    if a.stage == "build":
        build(a)
    elif a.stage == "judge":
        assert a.judge, "--judge required"
        judge(a)
    else:
        analyze(a)
