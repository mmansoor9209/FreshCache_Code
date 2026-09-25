#!/usr/bin/env python3
"""
validation/gold_verification_400/stage4_analyze.py

STAGE 4 — combine Stage 3a (paraphrase fidelity) and Stage 3b (timestamp-aware
gold validity) and recompute both 400-request audits on the EVIDENCE-VALID
SUBSET.

A request enters the evidence-valid subset when BOTH hold:
  (A) paraphrase fidelity is SAME under both judges, OR the request is an
      original base question (no paraphrase step); and
  (C) its (base query, snapshot round) unit is VALID_AT_T under BOTH judges.

Everything else is reported separately: paraphrase DIFFERENT, paraphrase
JUDGE_DISAGREEMENT, validity SUPERSEDED_AT_T, INDETERMINATE_AT_T, and validity
JUDGE_DISAGREEMENT.

WORDING CONSTRAINT enforced throughout: the subset is described as
"LLM-supported temporal validity". It is NOT independently verified factual
correctness, and no output of this script says otherwise.

Stage 3c was skipped by instruction; no plausibility signal is used.

Each audit is scored with ITS OWN abstention vocabulary (analyze2.py /
analyze_sttl.py for the primary family, analyze.py for the held-out audit).
Original answers, judgments and gold labels are read only and never regenerated.

Reproduce:
    <HOME>/miniconda3/envs/graphrag/bin/python \
        validation/gold_verification_400/stage4_analyze.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import sys
from collections import Counter
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
A1 = ROOT / "validation" / "mixed_age_full_policy_audit"
A1B = ROOT / "validation" / "baseline_operating_points"
A2 = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"

ABSTAIN_PRIMARY = ("i don't know", "i dont know", "cannot determine",
                   "not enough information", "insufficient", "no information",
                   "unclear from the context")
ABSTAIN_HELDOUT = ("i don't know", "i do not know", "cannot be determined",
                   "not enough information", "insufficient",
                   "unable to determine", "no information")

h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]  # noqa: E731
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0.0, (c - m) / d), 4),
            round(100 * min(1.0, (c + m) / d), 4)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(Fraction(1), Fraction(
        sum(math.comb(n, i) for i in range(k + 1)) * 2, 2 ** n)))


def outcome(ans, j1, j2, q, gold, key, pat):
    a = ans.get(tuple(key))
    if a is None:
        return None
    if any(p in a.strip().lower() for p in pat):
        return "ABSTAIN"
    s = (h(q), h(gold), h(a))
    l1, l2 = j1.get(s), j2.get(s)
    if l1 is None or l2 is None:
        return None
    return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"


def arm_stats(rows, arm, fresh="fresh"):
    n = len(rows)
    c = Counter(r[arm] for r in rows)
    fc = sum(1 for r in rows if r[fresh] == "CORRECT")
    wai = sum(1 for r in rows if r[arm] == "WRONG" and r[fresh] == "CORRECT")
    rev = sum(1 for r in rows if r[arm] == "CORRECT" and r[fresh] == "WRONG")
    return {"N": n, "correct": c["CORRECT"], "wrong": c["WRONG"],
            "abstain": c["ABSTAIN"],
            "accuracy_pct": round(100 * c["CORRECT"] / n, 4) if n else None,
            "accuracy_ci95": wilson(c["CORRECT"], n),
            "fresh_correct_n": fc, "wai": wai,
            "wai_pct": round(100 * wai / n, 4) if n else None,
            "wai_ci95": wilson(wai, n),
            "conditional_wai_pct": round(100 * wai / fc, 4) if fc else None,
            "conditional_wai_ci95": wilson(wai, fc) if fc else [None, None],
            "reverse": rev}


def paired(rows, a, b, fresh="fresh"):
    wa = lambda r, x: r[x] == "WRONG" and r[fresh] == "CORRECT"   # noqa: E731
    both = sum(1 for r in rows if wa(r, a) and wa(r, b))
    ao = sum(1 for r in rows if wa(r, a) and not wa(r, b))
    bo = sum(1 for r in rows if wa(r, b) and not wa(r, a))
    return {"wai_both": both, f"wai_only_{a}": ao, f"wai_only_{b}": bo,
            "exact_mcnemar_p": mcnemar(ao, bo)}


def main():
    say("STAGE 4 — combine fidelity + timestamp-aware validity; recompute audits")
    say("  Stage 3c skipped by instruction; no plausibility signal is used")
    say("  original answers, judgments and gold labels are read-only")

    for p in (HERE / "paraphrase_fidelity.csv", HERE / "gold_validity_at_t.csv",
              HERE / "request_map.csv"):
        if not p.exists():
            say(f"  STOP: {p.name} missing")
            sys.exit(2)

    fid = {r["query_id"]: r for r in csv.DictReader(
        open(HERE / "paraphrase_fidelity.csv", encoding="utf-8"))}
    val = list(csv.DictReader(open(HERE / "gold_validity_at_t.csv",
                                   encoding="utf-8")))
    rmap = {r["query_id"]: r for r in csv.DictReader(
        open(HERE / "request_map.csv", encoding="utf-8"))}
    unit_of = {}
    for u in val:
        for q in u["request_ids"].split("|"):
            unit_of[q] = u

    # ---- per-request combined status ----
    comb = {}
    for q, r in rmap.items():
        f = fid.get(q)
        fl = f["final_label"] if f else "BASE_QUERY"
        u = unit_of.get(q)
        vs = u["final_status"] if u else "NO_UNIT"
        ok_a = (fl in ("SAME", "BASE_QUERY"))
        ok_c = (vs == "VALID_AT_T")
        if not ok_a:
            excl = ("paraphrase_" + ("judge_disagreement"
                                     if fl == "JUDGE_DISAGREEMENT"
                                     else fl.lower()))
        elif not ok_c:
            excl = ("validity_" + ("judge_disagreement"
                                   if vs == "JUDGE_DISAGREEMENT" else vs.lower()))
        else:
            excl = ""
        comb[q] = {"query_id": q, "fidelity": fl, "validity": vs,
                   "in_evidence_valid_subset": int(ok_a and ok_c),
                   "exclusion_reason": excl,
                   "gold_source": r["gold_source"], "gold_year": r["gold_year"] or "(none)",
                   "freshness_class": r["freshness_class"],
                   "volatility": (u or {}).get("volatility", ""),
                   "unit_id": (u or {}).get("unit_id", ""),
                   "snapshot_round": (u or {}).get("snapshot_round", "")}
    say(f"\n  unique requests combined: {len(comb)}")
    say(f"  fidelity: {dict(Counter(c['fidelity'] for c in comb.values()))}")
    say(f"  validity: {dict(Counter(c['validity'] for c in comb.values()))}")
    keep = {q for q, c in comb.items() if c["in_evidence_valid_subset"]}
    say(f"  EVIDENCE-VALID requests (SAME-or-base AND dual-judge VALID_AT_T): "
        f"{len(keep)}")
    say(f"  exclusion reasons: "
        f"{dict(Counter(c['exclusion_reason'] for c in comb.values() if c['exclusion_reason']))}")

    # ---- rebuild both audits ----
    out = {}

    def run_audit(tag, sample, arms, ans, j1, j2, pat, keyfn, pub):
        rows = []
        for d in sample:
            got = {a: outcome(ans, j1, j2, d["query"], d["gold"], keyfn(d, a), pat)
                   for a in arms}
            if any(v is None for v in got.values()):
                continue
            got["qid"] = d["query_id"]
            c = comb.get(d["query_id"], {})
            got["keep"] = int(d["query_id"] in keep)
            got["gold_source"] = c.get("gold_source", "")
            got["gold_year"] = c.get("gold_year", "")
            got["freshness_class"] = c.get("freshness_class", "")
            rows.append(got)
        full = rows
        sub = [r for r in rows if r["keep"]]
        res = {"full": {"N": len(full)}, "subset": {"N": len(sub)}}
        cached = [a for a in arms if a != "fresh"]
        for name, rs in (("full", full), ("subset", sub)):
            fr = Counter(r["fresh"] for r in rs)
            res[name]["fresh"] = {
                "correct": fr["CORRECT"], "wrong": fr["WRONG"],
                "abstain": fr["ABSTAIN"],
                "accuracy_pct": round(100 * fr["CORRECT"] / len(rs), 4) if rs else None,
                "accuracy_ci95": wilson(fr["CORRECT"], len(rs))}
            for a in cached:
                res[name][a] = arm_stats(rs, a)
            for i in range(len(cached)):
                for j2_ in range(i + 1, len(cached)):
                    a, b = cached[i], cached[j2_]
                    res[name][f"paired_{a}_vs_{b}"] = paired(rs, a, b)
        res["published_reference"] = pub
        res["rows"] = rows
        return res

    # primary
    s1 = jl(A1 / "sample400.jsonl")
    ss = {d["query_id"]: d for d in jl(A1B / "sttl_sample400.jsonl")}
    a1 = {tuple(r["key"]): r["answer"] for r in jl(A1 / "answers2.jsonl")}
    for r in jl(A1B / "sttl_answers.jsonl"):
        a1[tuple(r["key"])] = r["answer"]
    j1 = {tuple(r["sig"]): r["label"] for r in jl(A1 / "judge2_llama8b.jsonl")}
    j2 = {tuple(r["sig"]): r["label"] for r in jl(A1 / "judge2_qwen7b.jsonl")}
    for r in jl(A1B / "sttl_judge_llama8b.jsonl"):
        j1[tuple(r["sig"])] = r["label"]
    for r in jl(A1B / "sttl_judge_qwen7b.jsonl"):
        j2[tuple(r["sig"])] = r["label"]

    def k1(d, a):
        if a == "fresh":
            return ("fresh", d["fresh_ctx_sha"], h(d["query"]))
        if a == "sttl_k1_8":
            return tuple(ss[d["query_id"]]["sttl_key"])
        return tuple(d[f"{a}_key"])

    pub1 = json.load(open(A1 / "corrected_results.json", encoding="utf-8"))
    pub1b = json.load(open(A1B / "sttl_vs_freshcache_results.json", encoding="utf-8"))
    out["mixed_age_400"] = run_audit(
        "mixed_age_400", s1, ("fresh", "on", "off", "sttl_k1_8"), a1, j1, j2,
        ABSTAIN_PRIMARY, k1,
        {"fresh_correct": pub1["gate_on"]["fresh_correct"],
         "gate_on_correct": pub1["gate_on"]["correct"],
         "gate_on_wai": pub1["gate_on"]["wai"],
         "gate_off_correct": pub1["gate_off"]["correct"],
         "gate_off_wai": pub1["gate_off"]["wai"],
         "sttl_k1_8_correct": pub1b["semanticttl_k8"]["correct"],
         "sttl_k1_8_wai": pub1b["semanticttl_k8"]["wai"],
         "sttl_vs_fc_paired": pub1b["paired"]})

    # held-out
    s2 = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    a2 = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
    k1b = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_llama8b.jsonl")}
    k2b = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_qwen7b.jsonl")}
    pub2 = json.load(open(A2 / "answer_audit_results.json", encoding="utf-8"))
    out["heldout_400"] = run_audit(
        "heldout_400", s2, ("fresh", "fc", "sttl"), a2, k1b, k2b,
        ABSTAIN_HELDOUT, lambda d, a: tuple(d[f"{a}_key"]),
        {"fresh_correct": pub2["arms"]["fresh"]["correct"],
         "fc_correct": pub2["arms"]["fc"]["correct"],
         "fc_wai": pub2["arms"]["fc"]["wai"],
         "sttl_correct": pub2["arms"]["sttl"]["correct"],
         "sttl_wai": pub2["arms"]["sttl"]["wai"],
         "wai_mcnemar": pub2["wai_mcnemar"]})

    # ---- gates: full-sample must reproduce published ----
    say("\n  REPRODUCTION GATE on the FULL sample (must match published)")
    g = []
    m = out["mixed_age_400"]["full"]
    p = out["mixed_age_400"]["published_reference"]
    for nm, got, want in (("primary fresh correct", m["fresh"]["correct"], p["fresh_correct"]),
                          ("primary gate-on correct", m["on"]["correct"], p["gate_on_correct"]),
                          ("primary gate-on WAI", m["on"]["wai"], p["gate_on_wai"]),
                          ("primary gate-off WAI", m["off"]["wai"], p["gate_off_wai"]),
                          ("primary SemanticTTL correct", m["sttl_k1_8"]["correct"], p["sttl_k1_8_correct"]),
                          ("primary SemanticTTL WAI", m["sttl_k1_8"]["wai"], p["sttl_k1_8_wai"])):
        ok = got == want
        g.append(ok)
        say(f"    {'PASS' if ok else 'FAIL'}  {nm:<34} {got} vs {want}")
    hh = out["heldout_400"]["full"]
    ph = out["heldout_400"]["published_reference"]
    for nm, got, want in (("held-out fresh correct", hh["fresh"]["correct"], ph["fresh_correct"]),
                          ("held-out FreshCache correct", hh["fc"]["correct"], ph["fc_correct"]),
                          ("held-out FreshCache WAI", hh["fc"]["wai"], ph["fc_wai"]),
                          ("held-out SemanticTTL correct", hh["sttl"]["correct"], ph["sttl_correct"]),
                          ("held-out SemanticTTL WAI", hh["sttl"]["wai"], ph["sttl_wai"])):
        ok = got == want
        g.append(ok)
        say(f"    {'PASS' if ok else 'FAIL'}  {nm:<34} {got} vs {want}")
    if not all(g):
        say("  STOP: full-sample reproduction failed")
        sys.exit(3)
    say("  full-sample reproduction: PASSED")

    # ---- side-by-side ----
    def cell(num, den, pct):
        return "{}/{} ({}%)".format(num, den, pct)

    def side(tag, arms, labels):
        r = out[tag]
        say("\n  {} — PUBLISHED (n={}) vs EVIDENCE-VALID SUBSET (n={})".format(
            tag, r["full"]["N"], r["subset"]["N"]))
        say("    {:<24}{:>26}{:>30}".format("metric", "published", "evidence-valid"))
        f, s_ = r["full"]["fresh"], r["subset"]["fresh"]
        say("    {:<24}{:>26}{:>30}".format(
            "fresh accuracy",
            cell(f["correct"], r["full"]["N"], f["accuracy_pct"]),
            cell(s_["correct"], r["subset"]["N"], s_["accuracy_pct"])))
        for a, lab in zip(arms, labels):
            fa, sa = r["full"][a], r["subset"][a]
            say("    {:<24}{:>26}{:>30}".format(
                lab + " accuracy",
                cell(fa["correct"], fa["N"], fa["accuracy_pct"]),
                cell(sa["correct"], sa["N"], sa["accuracy_pct"])))
            say("    {:<24}{:>26}{:>30}".format(
                lab + " WAI",
                cell(fa["wai"], fa["N"], fa["wai_pct"]),
                cell(sa["wai"], sa["N"], sa["wai_pct"])))
            say("    {:<24}{:>26}{:>30}".format(
                lab + " conditional WAI",
                cell(fa["wai"], fa["fresh_correct_n"], fa["conditional_wai_pct"]),
                cell(sa["wai"], sa["fresh_correct_n"], sa["conditional_wai_pct"])))

    side("mixed_age_400", ("on", "off", "sttl_k1_8"),
         ("FreshCache", "gate-off", "SemanticTTL"))
    side("heldout_400", ("fc", "sttl"), ("FreshCache", "SemanticTTL"))

    say("\n  PAIRED WAI (FreshCache vs SemanticTTL)")
    for tag, a, b in (("mixed_age_400", "sttl_k1_8", "on"),
                      ("heldout_400", "sttl", "fc")):
        for nm in ("full", "subset"):
            pr = out[tag][nm][f"paired_{b}_vs_{a}" if f"paired_{b}_vs_{a}" in out[tag][nm]
                              else f"paired_{a}_vs_{b}"]
            say(f"    {tag:<16} {nm:<7} {json.dumps(pr)}")

    # ---- strata ----
    say("\n  EVIDENCE-VALID SUBSET COMPOSITION")
    sub_ids = [q for q in comb if q in keep]
    for field, name in (("gold_source", "gold source"),
                        ("freshness_class", "freshness class"),
                        ("gold_year", "annotation year"),
                        ("volatility", "volatility")):
        say(f"    by {name}: "
            f"{dict(sorted(Counter(comb[q][field] for q in sub_ids).items()))}")

    # WAI retention
    say("\n  ORIGINAL WAI EVENTS RETAINED IN THE EVIDENCE-VALID SUBSET")
    wai_ret = {}
    for tag, arms in (("mixed_age_400", ("on", "off", "sttl_k1_8")),
                      ("heldout_400", ("fc", "sttl"))):
        rows = out[tag]["rows"]
        for a in arms:
            f = [r["qid"] for r in rows if r[a] == "WRONG" and r["fresh"] == "CORRECT"]
            s = [q for q in f if q in keep]
            wai_ret[f"{tag}|{a}"] = {"published": len(f), "retained": len(s),
                                     "retained_pct": round(100 * len(s) / len(f), 2)
                                     if f else None}
            say(f"    {tag:<16} {a:<12} {len(s)} of {len(f)} retained")
        allw = {r["qid"] for r in rows
                for a in arms if r[a] == "WRONG" and r["fresh"] == "CORRECT"}
        wai_ret[f"{tag}|ANY"] = {"published": len(allw),
                                 "retained": len(allw & keep)}
        say(f"    {tag:<16} {'ANY arm':<12} {len(allw & keep)} of {len(allw)} distinct")

    # ---- write ----
    with open(HERE / "combined_request_status.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(comb.values()))))
        w.writeheader()
        w.writerows(comb.values())
    rows_out = []
    for tag in out:
        for r in out[tag]["rows"]:
            rows_out.append({"audit": tag, **r})
    with open(HERE / "subset_per_request.csv", "w", newline="",
              encoding="utf-8") as f:
        ks = sorted({k for r in rows_out for k in r})
        w = csv.DictWriter(f, fieldnames=ks, restval="")
        w.writeheader()
        w.writerows(rows_out)
    clean = {t: {k: v for k, v in out[t].items() if k != "rows"} for t in out}
    json.dump({
        "stage": "4 — evidence-valid subset recomputation",
        "stage3c_skipped": True,
        "subset_rule": "(paraphrase fidelity SAME under both judges OR original "
                       "base question) AND unit VALID_AT_T under both judges",
        "wording_constraint": "the subset reflects LLM-SUPPORTED TEMPORAL "
                              "VALIDITY against stored snapshots. It is NOT "
                              "independently verified factual correctness.",
        "requests_total": len(comb), "evidence_valid_requests": len(keep),
        "fidelity_distribution": dict(Counter(c["fidelity"] for c in comb.values())),
        "validity_distribution": dict(Counter(c["validity"] for c in comb.values())),
        "exclusion_reasons": dict(Counter(c["exclusion_reason"]
                                          for c in comb.values() if c["exclusion_reason"])),
        "subset_composition": {
            f: dict(Counter(comb[q][f] for q in sub_ids))
            for f in ("gold_source", "freshness_class", "gold_year", "volatility")},
        "audits": clean, "wai_retention": wai_ret,
        "full_sample_reproduction_passed": True,
        "answer_generation": 0, "judging": 0, "web_calls": 0,
    }, open(HERE / "stage4_summary.json", "w"), indent=2, ensure_ascii=False)
    (HERE / "stage4.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote combined_request_status.csv, subset_per_request.csv, "
        "stage4_summary.json, stage4.log")
    (HERE / "stage4.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
