#!/usr/bin/env python3
"""
SECTIONS 5-8 -- stronger generation, paired policy evaluation on populations
A/B/C, and reference-quality sensitivity.

All generator/judge outputs are REUSED from existing verified artifacts; no
new model inference is performed.

  G0 = Llama-3.2-3B (original)        judges llama8b + qwen7b   [original protocol]
  G1 = Llama-3.1-8B-Instruct forced   judges qwen7b + mistral7b [8B is not its own judge]
  G2 = Llama-3.1-8B-Instruct abstain  judges qwen7b + mistral7b

Because G0 and G1 use DIFFERENT judge pairs, their accuracies are NOT
compared as if only the generator changed; both are reported and the
protocol difference is stated at every comparison.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, pathlib, sys
from collections import Counter
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
SRE = HERE.parent
V3 = SRE.parent
ROOT = V3.parent
HO = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"
RCI = V3 / "remaining_critical_issues" / "06_stronger_generator"
QA = ROOT / "validation" / "quality_aware_baseline" / "heldout_sensitivity"
DIAG = SRE / "02_reference_verification" / "reference_and_diagnosis.csv"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]
ABSTAIN_TOK = "INSUFFICIENT_CONTEXT"


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    t = (t or "").strip()
    if t == ABSTAIN_TOK:
        return True
    tl = t.lower()
    return any(x in tl for x in ABSTAIN_PAT)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 2),
            round(100 * min(1.0, (c + m) / d), 2))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tot = sum(Fraction(math.comb(n, i)) for i in range(k + 1))
    return float(min(Fraction(1), 2 * tot / Fraction(2) ** n))


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("SECTIONS 5-8 -- stronger generation, paired policies, sensitivity")
    say("  ALL generator and judge outputs reused; no new inference")

    diag = {r["query_id"]: r for r in csv.DictReader(open(DIAG, encoding="utf-8"))}
    S = json.load(open(RCI / "sample.json", encoding="utf-8"))
    reqs = S["requests"]
    HOS = json.load(open(HO / "sample.json", encoding="utf-8"))["requests"]
    assert [d["query_id"] for d in reqs] == [d["query_id"] for d in HOS]
    say(f"  400 held-out requests, identical IDs across all arms")

    # the quality-aware SemanticTTL arm lives in a separate artifact
    qa_rows = {}
    if (QA / "h_sample.jsonl").exists():
        for r in jl(QA / "h_sample.jsonl"):
            qa_rows[r.get("query_id")] = r
    say(f"  quality-aware SemanticTTL (theta=0.60 kappa=1/2) rows: "
        f"{len(qa_rows)}")

    GEN = {
        "G0_llama3b": {"ans": HO / "answers.jsonl",
                       "judges": [HO / "judge_llama8b.jsonl",
                                  HO / "judge_qwen7b.jsonl"],
                       "names": ("llama8b", "qwen7b"),
                       "protocol": "ORIGINAL (judges Llama-3.1-8B + Qwen2.5-7B)"},
        "G1_llama8b_forced": {"ans": RCI / "answers_llama8b_forced.jsonl",
                              "judges": [RCI / "judge_llama8b_forced__qwen7b.jsonl",
                                         RCI / "judge_llama8b_forced__mistral7b.jsonl"],
                              "names": ("qwen7b", "mistral7b"),
                              "protocol": "STRONGER (judges Qwen2.5-7B + Mistral-7B; "
                                          "8B is never its own judge)"},
        "G2_llama8b_abstain": {"ans": RCI / "answers_llama8b_abstain.jsonl",
                               "judges": [RCI / "judge_llama8b_abstain__qwen7b.jsonl",
                                          RCI / "judge_llama8b_abstain__mistral7b.jsonl"],
                               "names": ("qwen7b", "mistral7b"),
                               "protocol": "STRONGER + abstention permitted"},
    }
    ARMS = [("fresh", "NoCache fresh reference"), ("fc", "FreshCache"),
            ("sttl", "SemanticTTL theta=0.40 kappa=1/16 (drift-selected)")]

    POPS = {}
    POPS["A_all_400"] = [d["query_id"] for d in reqs]
    POPS["B_supported_reference"] = [q for q in POPS["A_all_400"]
                                     if diag[q]["reference_status_final"] == "SUPPORTED"]
    POPS["C_sufficient_evidence"] = [q for q in POPS["A_all_400"]
                                     if int(diag[q]["observable_pages"]) > 0
                                     and int(diag[q]["gold_lexically_in_context"])]
    ov = set(POPS["B_supported_reference"]) & set(POPS["C_sufficient_evidence"])
    say(f"\n  == populations ==")
    say(f"    A all requests                      {len(POPS['A_all_400'])}")
    say(f"    B independently supported reference {len(POPS['B_supported_reference'])} "
        f"(LLM-assisted assessment, NOT human)")
    say(f"    C fresh path has observable evidence with the gold string "
        f"lexically present {len(POPS['C_sufficient_evidence'])}")
    say(f"    B and C OVERLAP on {len(ov)} requests -- they are NOT disjoint, "
        f"and both are conditional/sensitivity populations, not replacements "
        f"for A")

    RES = {}
    for gname, G in GEN.items():
        ans = {tuple(r["key"]): r["answer"] for r in jl(G["ans"])}
        J = [{tuple(r["sig"]): r["label"] for r in jl(p)} for p in G["judges"]]
        say(f"\n  ==== {gname} -- {G['protocol']} ====")
        say(f"    answers {len(ans):,}; judges "
            f"{ {G['names'][i]: len(J[i]) for i in range(2)} }")

        def lab(d, arm):
            a = ans.get(tuple(d[f"{arm}_key"]))
            if a is None:
                return None
            if is_abstain(a):
                return "ABSTAIN"
            sig = (h16(d["query"]), h16(d["gold"]), h16(a))
            l1, l2 = J[0].get(sig), J[1].get(sig)
            if l1 is None or l2 is None:
                return None
            return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

        L = {d["query_id"]: {a: lab(d, a) for a, _ in ARMS} for d in reqs}
        RES[gname] = {"protocol": G["protocol"], "populations": {}}
        for pname, ids in POPS.items():
            ids = [q for q in ids]
            n = len(ids)
            fr = sum(1 for q in ids if L[q]["fresh"] == "CORRECT")
            e = {"n": n, "fresh_correct": fr,
                 "fresh_accuracy_pct": round(100*fr/n, 4) if n else None,
                 "fresh_ci": wilson(fr, n), "arms": {}}
            for arm, nm in ARMS:
                if arm == "fresh":
                    continue
                cor = sum(1 for q in ids if L[q][arm] == "CORRECT")
                ab = sum(1 for q in ids if L[q][arm] == "ABSTAIN")
                wai = sum(1 for q in ids
                          if L[q][arm] == "WRONG" and L[q]["fresh"] == "CORRECT")
                rev = sum(1 for q in ids
                          if L[q][arm] == "CORRECT" and L[q]["fresh"] == "WRONG")
                e["arms"][arm] = {
                    "name": nm, "correct": cor,
                    "accuracy_pct": round(100*cor/n, 4) if n else None,
                    "accuracy_ci": wilson(cor, n),
                    "abstain": ab,
                    "abstention_rate_pct": round(100*ab/n, 4) if n else None,
                    "wai": wai, "wai_pct": round(100*wai/n, 4) if n else None,
                    "wai_ci": wilson(wai, n),
                    "conditional_wai_pct": (round(100*wai/fr, 4) if fr else None),
                    "conditional_wai_ci": wilson(wai, fr) if fr else None,
                    "c2w_vs_fresh": wai, "w2c_vs_fresh": rev,
                    "mcnemar_p_vs_fresh": mcnemar(wai, rev)}
            RES[gname]["populations"][pname] = e
            say(f"\n    -- population {pname} (n = {n}) --")
            say(f"      fresh reference   {fr}/{n} = "
                f"{(100*fr/n if n else 0):.2f}%  CI{e['fresh_ci']}")
            say(f"      {'arm':<46}{'acc':>16}{'WAI':>14}{'cWAI':>10}"
                f"{'C>W':>5}{'W>C':>5}{'McNemar p':>11}")
            for arm, nm in ARMS:
                if arm == "fresh":
                    continue
                a = e["arms"][arm]
                cw = (f"{a['conditional_wai_pct']:.2f}%"
                      if a["conditional_wai_pct"] is not None else "-")
                say(f"      {nm:<46}{a['correct']:>6}/{n:<4}"
                    f"{a['accuracy_pct']:>5.1f}%"
                    f"{a['wai']:>7}/{n:<3}{a['wai_pct']:>5.1f}%{cw:>10}"
                    f"{a['c2w_vs_fresh']:>5}{a['w2c_vs_fresh']:>5}"
                    f"{a['mcnemar_p_vs_fresh']:>11.4g}")
            # FreshCache vs SemanticTTL, paired
            b = sum(1 for q in ids if L[q]["fc"] == "CORRECT" and L[q]["sttl"] == "WRONG")
            c = sum(1 for q in ids if L[q]["fc"] == "WRONG" and L[q]["sttl"] == "CORRECT")
            p = mcnemar(b, c)
            e["fc_vs_sttl"] = {"fc_correct_sttl_wrong": b,
                               "fc_wrong_sttl_correct": c, "p": p}
            say(f"      FreshCache vs SemanticTTL (paired): "
                f"FC-right/STTL-wrong {b}, FC-wrong/STTL-right {c}, "
                f"exact McNemar p = {p:.4g}"
                + ("   (not significant; NOT equivalence -- no "
                   "non-inferiority margin was pre-specified)" if p > 0.05 else ""))

    # ---------------- section 8: does the contrast persist? ----------------
    say(f"\n  == SECTION 8: reference-quality sensitivity ==")
    say(f"    N_fresh_correct by generator and population")
    say(f"      {'generator':<22}{'A (400)':>10}{'B (supported)':>16}"
        f"{'C (evidence)':>15}")
    for g in GEN:
        r = RES[g]["populations"]
        say(f"      {g:<22}{r['A_all_400']['fresh_correct']:>10}"
            f"{r['B_supported_reference']['fresh_correct']:>16}"
            f"{r['C_sufficient_evidence']['fresh_correct']:>15}")
    say(f"\n    FreshCache WAI and conditional WAI")
    say(f"      {'generator':<22}{'WAI/400':>10}{'WAI%':>8}{'N_fresh_ok':>12}"
        f"{'cWAI%':>9}{'cWAI CI':>18}")
    for g in GEN:
        a = RES[g]["populations"]["A_all_400"]
        f_ = a["arms"]["fc"]
        say(f"      {g:<22}{f_['wai']:>10}{f_['wai_pct']:>7.2f}%"
            f"{a['fresh_correct']:>12}"
            + (f"{f_['conditional_wai_pct']:>8.2f}%" if f_['conditional_wai_pct'] is not None else f"{'-':>9}")
            + f"{str(f_['conditional_wai_ci']):>18}")
    say(f"\n    SemanticTTL (kappa=1/16) WAI")
    for g in GEN:
        a = RES[g]["populations"]["A_all_400"]
        s_ = a["arms"]["sttl"]
        say(f"      {g:<22}{s_['wai']:>10}{s_['wai_pct']:>7.2f}%"
            + (f"   cWAI {s_['conditional_wai_pct']:.2f}%"
               if s_['conditional_wai_pct'] is not None else ""))
    fcA = RES["G0_llama3b"]["populations"]["A_all_400"]["arms"]
    fcB = RES["G1_llama8b_forced"]["populations"]["A_all_400"]["arms"]
    say(f"\n    contrast FreshCache WAI vs SemanticTTL WAI on all 400:")
    say(f"      G0: {fcA['fc']['wai']} vs {fcA['sttl']['wai']}  "
        f"(ratio {fcA['sttl']['wai']/max(1,fcA['fc']['wai']):.1f}x)")
    say(f"      G1: {fcB['fc']['wai']} vs {fcB['sttl']['wai']}  "
        f"(ratio {fcB['sttl']['wai']/max(1,fcB['fc']['wai']):.1f}x)")
    persists = (fcB['sttl']['wai'] > fcB['fc']['wai'])
    say(f"      the answer-quality contrast "
        f"{'PERSISTS' if persists else 'DISAPPEARS/REVERSES'} under the "
        f"stronger generator")

    json.dump({"populations": {k: len(v) for k, v in POPS.items()},
               "B_C_overlap": len(ov), "results": RES},
              open(HERE / "paired_policy_results.json", "w"), indent=2)
    (SRE / "logs").mkdir(exist_ok=True)
    (SRE / "logs" / "s5s8.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote paired_policy_results.json")


if __name__ == "__main__":
    main()
