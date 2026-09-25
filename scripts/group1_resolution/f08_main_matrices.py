#!/usr/bin/env python3
"""F08 -- fresh x cached 3x3 transition matrices for the four 400-request
answer audits, extracted from existing row-level artifacts.

Nothing is generated, judged or replayed. Arm 4's labels are re-derived from
the STORED answers and STORED judge verdicts using the audit's own labelling
rule (both primary judges must say CORRECT, abstention by pattern), exactly as
h4_analyze.py does; no model is invoked.
"""
from __future__ import annotations
import csv, json, math, pathlib, sys
from collections import Counter
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOP = ROOT / "validation" / "baseline_operating_points"
HOA = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"
QA = ROOT / "validation" / "quality_aware_baseline" / "heldout_sensitivity"
LAB = ["CORRECT", "WRONG", "ABSTAIN"]
ABST = ["i don't know", "i do not know", "cannot be determined",
        "not enough information", "insufficient", "unable to determine",
        "no information"]
log, checks = [], []


def say(s=""):
    print(s, flush=True); log.append(s)


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def h16(s):
    import hashlib
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def ck(name, got, want):
    ok = got == want
    checks.append({"check": name, "recomputed": got, "published": want,
                   "status": "PASS" if ok else "FAIL"})
    say(f"    {'PASS' if ok else 'FAIL'}  {name:<52} rows {got!s:>10}  "
        f"published {want!s:>10}")
    return ok


def mcnemar(b01, b10):
    n = b01 + b10
    if not n:
        return 1.0
    return min(1.0, float(sum(Fraction(math.comb(n, i), 2 ** n)
                              for i in range(0, min(b01, b10) + 1)) * 2))


def emit(title, rows, fkey, ckey):
    m = Counter((r[fkey], r[ckey]) for r in rows)
    off = {k: v for k, v in m.items() if k[0] not in LAB or k[1] not in LAB}
    say(f"\n  {title}   n = {len(rows)}   rows = fresh, cols = cached")
    say("    " + "fresh\\cached".rjust(14)
        + "".join(c.rjust(10) for c in LAB) + "total".rjust(8))
    for f in LAB:
        v = [m.get((f, c), 0) for c in LAB]
        say("    " + f.rjust(14) + "".join(str(x).rjust(10) for x in v)
            + str(sum(v)).rjust(8))
    tot = [sum(m.get((f, c), 0) for f in LAB) for c in LAB]
    say("    " + "total".rjust(14) + "".join(str(x).rjust(10) for x in tot)
        + str(sum(tot)).rjust(8))
    if off:
        say(f"    OFF-GRID: {off}")
    return m, off


def verify(tag, rows, fkey, ckey, pub):
    m, off = emit(tag, rows, fkey, ckey)
    say(f"    checks:")
    ck(f"{tag}: N", len(rows), 400)
    ck(f"{tag}: cached CORRECT", sum(m.get((f, "CORRECT"), 0) for f in LAB), pub["correct"])
    ck(f"{tag}: cached WRONG", sum(m.get((f, "WRONG"), 0) for f in LAB), pub["wrong"])
    ck(f"{tag}: cached ABSTAIN", sum(m.get((f, "ABSTAIN"), 0) for f in LAB), pub["abstain"])
    ck(f"{tag}: fresh CORRECT", sum(m.get(("CORRECT", c), 0) for c in LAB), pub["fresh_correct"])
    ck(f"{tag}: fresh WRONG", sum(m.get(("WRONG", c), 0) for c in LAB), pub["fresh_wrong"])
    ck(f"{tag}: fresh ABSTAIN", sum(m.get(("ABSTAIN", c), 0) for c in LAB), pub["fresh_abstain"])
    ck(f"{tag}: WAI cell (fresh CORRECT, cached WRONG)", m.get(("CORRECT", "WRONG"), 0), pub["wai"])
    if "reverse" in pub:
        ck(f"{tag}: reverse cell (fresh WRONG, cached CORRECT)",
           m.get(("WRONG", "CORRECT"), 0), pub["reverse"])
    return {"matrix": {f: {c: m.get((f, c), 0) for c in LAB} for f in LAB},
            "off_grid": {str(k): v for k, v in off.items()}, "n": len(rows)}


def main():
    say("F08 -- fresh x cached 3x3 transition matrices, four 400-request audits")
    out = {}

    # ---------------- 1 & 2: primary mixed-age ----------------
    say("\n" + "=" * 76)
    say("PRIMARY mixed-age 400-request audit")
    say(f"  source: validation/baseline_operating_points/"
        f"sttl_vs_freshcache_per_request.csv")
    pr = list(csv.DictReader(open(BOP / "sttl_vs_freshcache_per_request.csv",
                                  encoding="utf-8")))
    pj = json.load(open(BOP / "sttl_vs_freshcache_results.json", encoding="utf-8"))
    say(f"  same 400 ids as the corrected audit: "
        f"{pj['same_400_ids_as_corrected_audit']}; shared fresh reference "
        f"identical: {pj['shared_fresh_reference_identical']}")
    out["1_primary_freshcache"] = verify("1 primary FreshCache", pr, "fresh",
                                         "freshcache", pj["freshcache"])
    out["2_primary_semanticttl"] = verify("2 primary SemanticTTL (kappa=1/8)",
                                          pr, "fresh", "semanticttl_k8",
                                          pj["semanticttl_k8"])
    say(f"\n  paired WAI discordance, primary:")
    a = {r["query_id"] for r in pr if r["freshcache"] == "WRONG" and r["fresh"] == "CORRECT"}
    b = {r["query_id"] for r in pr if r["semanticttl_k8"] == "WRONG" and r["fresh"] == "CORRECT"}
    ck("primary: WAI only SemanticTTL", len(b - a), pj["paired"]["wai_only_semanticttl"])
    ck("primary: WAI only FreshCache", len(a - b), pj["paired"]["wai_only_freshcache"])
    p = mcnemar(len(b - a), len(a - b))
    say(f"    exact McNemar p = {p:.6g}  published "
        f"{pj['paired']['mcnemar_exact_two_sided_p']:.6g}  "
        f"{'PASS' if abs(p - pj['paired']['mcnemar_exact_two_sided_p']) < 1e-12 else 'FAIL'}")
    checks.append({"check": "primary: exact McNemar p", "recomputed": p,
                   "published": pj["paired"]["mcnemar_exact_two_sided_p"],
                   "status": "PASS" if abs(p - pj["paired"]["mcnemar_exact_two_sided_p"]) < 1e-12 else "FAIL"})
    out["primary_mcnemar"] = {"wai_only_sttl": len(b - a),
                              "wai_only_fc": len(a - b), "p_exact": p}

    # ---------------- 3: held-out FreshCache ----------------
    say("\n" + "=" * 76)
    say("HELD-OUT 400-request audit")
    say(f"  source: validation/heldout_baseline_tuning/answer_audit_k1_16/"
        f"paired_transitions.csv")
    ho = list(csv.DictReader(open(HOA / "paired_transitions.csv", encoding="utf-8")))
    hj = json.load(open(HOA / "answer_audit_results.json", encoding="utf-8"))
    fr = hj["arms"]["fresh"]
    pubfc = dict(hj["arms"]["fc"], fresh_correct=fr["correct"],
                 fresh_wrong=fr["wrong"], fresh_abstain=fr["abstain"])
    out["3_heldout_freshcache"] = verify("3 held-out FreshCache", ho, "fresh",
                                         "freshcache", pubfc)
    pubst = dict(hj["arms"]["sttl"], fresh_correct=fr["correct"],
                 fresh_wrong=fr["wrong"], fresh_abstain=fr["abstain"])
    out["3b_heldout_semanticttl_k1_16"] = verify(
        "3b held-out SemanticTTL (kappa=1/16, drift-selected)", ho, "fresh",
        "semanticttl", pubst)
    say(f"\n  paired discordance, held-out (FreshCache vs SemanticTTL 1/16):")
    A = {r["qid"] for r in ho if r["freshcache"] == "WRONG" and r["fresh"] == "CORRECT"}
    B = {r["qid"] for r in ho if r["semanticttl"] == "WRONG" and r["fresh"] == "CORRECT"}
    wm = hj["wai_mcnemar"]
    ck("held-out: WAI both", len(A & B), wm["wai_both"])
    ck("held-out: WAI only SemanticTTL", len(B - A), wm["wai_semanticttl_only"])
    ck("held-out: WAI only FreshCache", len(A - B), wm["wai_freshcache_only"])
    ck("held-out: WAI neither", 400 - len(A | B), wm["wai_neither"])
    cm = hj["correctness_mcnemar"]
    ck("held-out: both correct",
       sum(1 for r in ho if r["freshcache"] == "CORRECT" and r["semanticttl"] == "CORRECT"),
       cm["both_correct"])
    ck("held-out: FC correct / STTL wrong",
       sum(1 for r in ho if r["freshcache"] == "CORRECT" and r["semanticttl"] != "CORRECT"),
       cm["freshcache_correct_semanticttl_wrong"])
    ck("held-out: STTL correct / FC wrong",
       sum(1 for r in ho if r["semanticttl"] == "CORRECT" and r["freshcache"] != "CORRECT"),
       cm["semanticttl_correct_freshcache_wrong"])

    # ---------------- 4: held-out quality-aware SemanticTTL ----------------
    say("\n" + "=" * 76)
    say("HELD-OUT 400-request QUALITY-AWARE SemanticTTL (theta=0.60, kappa=1/2)")
    fz = json.load(open(QA / "freeze.json", encoding="utf-8"))
    h4 = json.load(open(QA / "h4_results.json", encoding="utf-8"))
    say(f"  frozen config {fz['config']}; frozen before held-out evaluation: "
        f"{fz['frozen_before_heldout_evaluation']}; selection used held-out "
        f"data: {fz['selection_used_heldout_data']}")
    ans = {tuple(r["key"]): r["answer"] for r in jl(HOA / "answers.jsonl")}
    for r in jl(QA / "h_answers.jsonl"):
        ans[tuple(r["key"])] = r["answer"]
    j1 = {tuple(r["sig"]): r["label"] for r in jl(HOA / "judge_llama8b.jsonl")}
    j2 = {tuple(r["sig"]): r["label"] for r in jl(HOA / "judge_qwen7b.jsonl")}
    for r in jl(QA / "h_judge_llama8b.jsonl"):
        j1[tuple(r["sig"])] = r["label"]
    for r in jl(QA / "h_judge_qwen7b.jsonl"):
        j2[tuple(r["sig"])] = r["label"]
    say(f"  stored answers {len(ans):,}; stored judgments "
        f"llama8b {len(j1):,}, qwen7b {len(j2):,} (no model invoked)")
    new = {r["query_id"]: r for r in jl(QA / "h_sample.jsonl")}
    samp = {r["query_id"]: r for r in
            json.load(open(HOA / "sample.json", encoding="utf-8"))["requests"]}

    def outc(q, key):
        a = ans.get(tuple(key))
        if a is None:
            return None
        if any(x in a.strip().lower() for x in ABST):
            return "ABSTAIN"
        sig = (h16(samp[q]["query"]), h16(samp[q]["gold"]), h16(a))
        l1, l2 = j1.get(sig), j2.get(sig)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    rows4 = []
    for r in ho:
        q = r["qid"]
        kk = new[q]["new_sttl_key"]
        k = kk if isinstance(kk, list) else json.loads(str(kk).replace("'", chr(34)))
        rows4.append({"qid": q, "fresh": r["fresh"],
                      "sttl_q": outc(q, k), "freshcache": r["freshcache"]})
    miss = [r for r in rows4 if r["sttl_q"] is None]
    say(f"  rows with a derivable label: {len(rows4)-len(miss)}/400")
    if miss:
        say(f"  NOT DERIVABLE for {len(miss)} rows -> matrix not emitted")
    pubq = dict(h4["answer_quality"]["full_400"]["sttl_t060_k1_2"],
                fresh_correct=h4["answer_quality"]["full_400"]["fresh_correct"],
                fresh_abstain=h4["answer_quality"]["full_400"]["fresh_abstain"])
    pubq["fresh_wrong"] = 400 - pubq["fresh_correct"] - pubq["fresh_abstain"]
    out["4_heldout_quality_aware_sttl"] = verify(
        "4 held-out quality-aware SemanticTTL", rows4, "fresh", "sttl_q", pubq)
    Aq = {r["qid"] for r in rows4 if r["freshcache"] == "WRONG" and r["fresh"] == "CORRECT"}
    Bq = {r["qid"] for r in rows4 if r["sttl_q"] == "WRONG" and r["fresh"] == "CORRECT"}
    pq = mcnemar(len(Bq - Aq), len(Aq - Bq))
    say(f"\n  paired WAI discordance vs FreshCache (DERIVED -- the full-400 "
        f"McNemar block is empty in h4_results.json):")
    say(f"    both {len(Aq & Bq)}, quality-aware only {len(Bq - Aq)}, "
        f"FreshCache only {len(Aq - Bq)}, exact McNemar p = {pq:.6g}")
    say(f"    for reference, the published evidence-valid (n=107) block is "
        f"{h4['answer_quality']['evidence_valid']['paired_sttl_t060_k1_2_vs_fc']}")
    out["4_mcnemar_derived"] = {"both": len(Aq & Bq), "sttl_only": len(Bq - Aq),
                                "fc_only": len(Aq - Bq), "p_exact": pq,
                                "status": "DERIVED, not previously published"}

    nf = sum(1 for c in checks if c["status"] == "FAIL")
    say("\n" + "=" * 76)
    say(f"  TOTAL checks {len(checks)}   PASS {len(checks)-nf}   FAIL {nf}")
    for c in checks:
        if c["status"] == "FAIL":
            say(f"    FAIL {c['check']}: {c['recomputed']} vs {c['published']}")
    json.dump({"matrices": out, "checks": checks, "n_fail": nf},
              open(HERE / "out" / "f08_main_matrices.json", "w"), indent=2)
    (HERE / "out" / "f08_main.log").write_text("\n".join(log) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
