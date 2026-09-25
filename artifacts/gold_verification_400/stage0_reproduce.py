#!/usr/bin/env python3
"""
validation/gold_verification_400/stage0_reproduce.py

STAGE 0 of the Feedback 1 gold-verification experiment — REPRODUCTION GATE.

Rebuilds all arms of both 400-request answer audits from their saved
request-level artifacts and asserts every value listed in Revision 3 §3.2 of
FINAL_EXPERIMENT_PLAN_FOR_APPROVAL.md.

  Audit 1  primary mixed-age        arms: fresh, gate-on, gate-off
  Audit 1b primary SemanticTTL k=1/8 arms: fresh, FreshCache, SemanticTTL
           (same 400 request ids, shared fresh reference)
  Audit 2  held-out k=1/16          arms: fresh, FreshCache, SemanticTTL

CRITICAL: each audit's ORIGINAL abstention vocabulary is used. The primary
family (analyze2.py:21-23, analyze_sttl.py:25-27) and the held-out audit
(analyze.py:23-25) use DIFFERENT lists; applying one to the other flips
p_q_1ed44863_1 and changes primary gate-on WAI from 6 to 5. The two lists are
asserted to differ, and a cross-application diagnostic is recorded.

READ-ONLY on every pre-existing artifact. No answer generation, no judging, no
GPU, no web/API call. Everything written lands in this directory.

Reproduce:
    <HOME>/miniconda3/envs/graphrag/bin/python \
        validation/gold_verification_400/stage0_reproduce.py
"""
from __future__ import annotations

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

# ---- ORIGINAL abstention vocabularies, copied verbatim from each analyzer ----
ABSTAIN_PRIMARY = ("i don't know", "i dont know", "cannot determine",
                   "not enough information", "insufficient", "no information",
                   "unclear from the context")                    # analyze2.py:21-23
ABSTAIN_HELDOUT = ("i don't know", "i do not know", "cannot be determined",
                   "not enough information", "insufficient",
                   "unable to determine", "no information")       # analyze.py:23-25

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def mcnemar(b, c):
    """Exact two-sided binomial McNemar, as in analyze2.py:50-55."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(Fraction(1),
                     Fraction(sum(math.comb(n, i) for i in range(k + 1)) * 2,
                              2 ** n)))


# --------------------------------------------------------------- rebuilding --
def rebuild(sample, ans, j1, j2, arm_keys, abstain_pat):
    """Reproduce one audit's per-request outcomes.

    arm_keys: dict arm -> callable(request_dict) returning the answer key tuple.
    Returns (rows, dropped). Outcome rule copied from analyze2.py:65-74 /
    analyze.py:84-93: abstain pattern first, then CORRECT only if BOTH judges
    return CORRECT; None if the answer or either judge label is missing.
    """
    def is_abstain(t):
        t = (t or "").strip().lower()
        return any(p in t for p in abstain_pat)

    def outcome(q, gold, a):
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h(q), h(gold), h(a))
        l1, l2 = j1.get(sig), j2.get(sig)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    rows, dropped = [], 0
    for d in sample:
        got = {arm: outcome(d["query"], d["gold"], ans.get(tuple(fn(d))))
               for arm, fn in arm_keys.items()}
        if any(v is None for v in got.values()):
            dropped += 1
            continue
        got["qid"] = d["query_id"]
        got["freshness_class"] = d.get("fc")   # NOT "fc": that is an arm name
        rows.append(got)
    return rows, dropped


def arm_block(rows, arm):
    n = len(rows)
    c = Counter(r[arm] for r in rows)
    wai = sum(1 for r in rows if r[arm] == "WRONG" and r["fresh"] == "CORRECT")
    rev = sum(1 for r in rows if r[arm] == "CORRECT" and r["fresh"] == "WRONG")
    fc = sum(1 for r in rows if r["fresh"] == "CORRECT")
    lo, hi = wilson(wai, n)
    return {"N": n, "correct": c["CORRECT"], "wrong": c["WRONG"],
            "abstain": c["ABSTAIN"],
            "correct_pct": round(100 * c["CORRECT"] / n, 4) if n else None,
            "wai": wai, "wai_pct": round(100 * wai / n, 4) if n else None,
            "wai_ci95_pct": [round(100 * lo, 4), round(100 * hi, 4)],
            "reverse": rev, "fresh_correct": fc,
            "conditional_wai_pct": (100 * wai / fc) if fc else None}


def paired_wai(rows, a, b):
    """Discordant WAI counts between two cached arms, both vs the shared fresh."""
    wa = lambda r, x: r[x] == "WRONG" and r["fresh"] == "CORRECT"   # noqa: E731
    both = sum(1 for r in rows if wa(r, a) and wa(r, b))
    a_only = sum(1 for r in rows if wa(r, a) and not wa(r, b))
    b_only = sum(1 for r in rows if wa(r, b) and not wa(r, a))
    return {"wai_both": both, f"wai_only_{a}": a_only, f"wai_only_{b}": b_only,
            "exact_mcnemar_p": mcnemar(a_only, b_only)}


# -------------------------------------------------------------------- main --
CHECKS = []


def check(name, got, want, tol=None):
    if tol is None:
        ok = (got == want)
    else:
        ok = (got is not None and abs(got - want) <= tol)
    CHECKS.append({"check": name, "reproduced": got, "required": want, "ok": ok})
    say(f"    {'PASS' if ok else 'FAIL'}  {name:<52} got {got}   required {want}")
    return ok


def main():
    say("STAGE 0 — Feedback 1 gold verification: REPRODUCTION GATE")
    say("  read-only on every existing artifact; no generation, no judging, no GPU")

    reads = [
        A1 / "sample400.jsonl", A1 / "answers2.jsonl",
        A1 / "judge2_llama8b.jsonl", A1 / "judge2_qwen7b.jsonl",
        A1 / "corrected_results.json",
        A1B / "sttl_sample400.jsonl", A1B / "sttl_answers.jsonl",
        A1B / "sttl_judge_llama8b.jsonl", A1B / "sttl_judge_qwen7b.jsonl",
        A1B / "sttl_vs_freshcache_results.json",
        A2 / "sample.json", A2 / "answers.jsonl",
        A2 / "judge_llama8b.jsonl", A2 / "judge_qwen7b.jsonl",
        A2 / "answer_audit_results.json",
    ]
    for p in reads:
        if not p.exists():
            say(f"  MISSING INPUT: {p}")
            sys.exit(2)
    pre = {str(p.relative_to(ROOT)): sha(p) for p in reads}
    say(f"  hashed {len(pre)} input artifacts")

    # the two vocabularies must genuinely differ; this gate depends on it
    assert set(ABSTAIN_PRIMARY) != set(ABSTAIN_HELDOUT), \
        "abstention vocabularies are identical — plan premise is wrong"
    say(f"  abstention vocabularies differ: primary-only "
        f"{sorted(set(ABSTAIN_PRIMARY) - set(ABSTAIN_HELDOUT))}, held-out-only "
        f"{sorted(set(ABSTAIN_HELDOUT) - set(ABSTAIN_PRIMARY))}")

    results = {}

    # ================= AUDIT 1 — primary mixed-age =================
    say("\n  AUDIT 1 — primary mixed-age (fresh / gate-on / gate-off)")
    s1 = jl(A1 / "sample400.jsonl")
    a1 = {tuple(r["key"]): r["answer"] for r in jl(A1 / "answers2.jsonl")}
    j1a = {tuple(r["sig"]): r["label"] for r in jl(A1 / "judge2_llama8b.jsonl")}
    j2a = {tuple(r["sig"]): r["label"] for r in jl(A1 / "judge2_qwen7b.jsonl")}
    keys1 = {"fresh": lambda d: ("fresh", d["fresh_ctx_sha"], h(d["query"])),
             "on": lambda d: tuple(d["on_key"]),
             "off": lambda d: tuple(d["off_key"])}
    r1, drop1 = rebuild(s1, a1, j1a, j2a, keys1, ABSTAIN_PRIMARY)
    pub1 = json.load(open(A1 / "corrected_results.json", encoding="utf-8"))
    on, off = arm_block(r1, "on"), arm_block(r1, "off")
    fresh1 = Counter(r["fresh"] for r in r1)
    say(f"    rebuilt {len(r1)} rows, dropped {drop1}")
    check("A1 N", len(r1), pub1["N"])
    check("A1 dropped_incomplete", drop1, pub1["dropped_incomplete"])
    check("A1 fresh correct", fresh1["CORRECT"], pub1["gate_on"]["fresh_correct"])
    check("A1 fresh abstain", fresh1["ABSTAIN"], pub1["gate_on"]["fresh_abstain"])
    check("A1 gate-on correct", on["correct"], pub1["gate_on"]["correct"])
    check("A1 gate-on abstain", on["abstain"], pub1["gate_on"]["abstain"])
    check("A1 gate-on WAI", on["wai"], pub1["gate_on"]["wai"])
    check("A1 gate-on reverse", on["reverse"], pub1["gate_on"]["reverse"])
    check("A1 gate-off correct", off["correct"], pub1["gate_off"]["correct"])
    check("A1 gate-off abstain", off["abstain"], pub1["gate_off"]["abstain"])
    check("A1 gate-off WAI", off["wai"], pub1["gate_off"]["wai"])
    pw1 = paired_wai(r1, "off", "on")
    check("A1 paired WAI off-only", pw1["wai_only_off"],
          pub1["mcnemar"]["discordant_off_only"])
    check("A1 paired WAI on-only", pw1["wai_only_on"],
          pub1["mcnemar"]["discordant_on_only"])
    check("A1 paired WAI McNemar p", round(pw1["exact_mcnemar_p"], 10),
          round(pub1["mcnemar"]["exact_two_sided_p"], 10), tol=1e-9)
    results["audit1_primary"] = {"fresh": dict(fresh1), "gate_on": on,
                                 "gate_off": off, "paired_wai": pw1,
                                 "dropped": drop1}

    # ================= AUDIT 1b — primary SemanticTTL k=1/8 =================
    say("\n  AUDIT 1b — primary SemanticTTL k=1/8 (same 400 ids)")
    pub1b = json.load(open(A1B / "sttl_vs_freshcache_results.json", encoding="utf-8"))
    check("A1b same_400_ids_as_corrected_audit",
          pub1b["same_400_ids_as_corrected_audit"], True)
    check("A1b shared_fresh_reference_identical",
          pub1b["shared_fresh_reference_identical"], True)
    s1b = jl(A1B / "sttl_sample400.jsonl")
    check("A1b sample ids identical to audit 1, in order",
          [d["query_id"] for d in s1b] == [d["query_id"] for d in s1], True)
    a1b = dict(a1)
    for r in jl(A1B / "sttl_answers.jsonl"):
        a1b[tuple(r["key"])] = r["answer"]
    j1b = dict(j1a)
    for r in jl(A1B / "sttl_judge_llama8b.jsonl"):
        j1b[tuple(r["sig"])] = r["label"]
    j2b = dict(j2a)
    for r in jl(A1B / "sttl_judge_qwen7b.jsonl"):
        j2b[tuple(r["sig"])] = r["label"]
    keys1b = {"fresh": lambda d: ("fresh", d["fresh_ctx_sha"], h(d["query"])),
              "fc": lambda d: tuple(d["fc_key"]),
              "sttl": lambda d: tuple(d["sttl_key"])}
    r1b, drop1b = rebuild(s1b, a1b, j1b, j2b, keys1b, ABSTAIN_PRIMARY)
    fcb, stb = arm_block(r1b, "fc"), arm_block(r1b, "sttl")
    say(f"    rebuilt {len(r1b)} rows, dropped {drop1b}")
    check("A1b N", len(r1b), pub1b["N"])
    check("A1b dropped_incomplete", drop1b, pub1b["dropped_incomplete"])
    check("A1b FreshCache correct", fcb["correct"], pub1b["freshcache"]["correct"])
    check("A1b FreshCache WAI", fcb["wai"], pub1b["freshcache"]["wai"])
    check("A1b SemanticTTL correct", stb["correct"],
          pub1b["semanticttl_k8"]["correct"])
    check("A1b SemanticTTL abstain", stb["abstain"],
          pub1b["semanticttl_k8"]["abstain"])
    check("A1b SemanticTTL WAI", stb["wai"], pub1b["semanticttl_k8"]["wai"])
    check("A1b SemanticTTL reverse", stb["reverse"],
          pub1b["semanticttl_k8"]["reverse"])
    pw1b = paired_wai(r1b, "sttl", "fc")
    check("A1b paired WAI SemanticTTL-only", pw1b["wai_only_sttl"],
          pub1b["paired"]["wai_only_semanticttl"])
    check("A1b paired WAI FreshCache-only", pw1b["wai_only_fc"],
          pub1b["paired"]["wai_only_freshcache"])
    check("A1b paired WAI McNemar p", pw1b["exact_mcnemar_p"],
          pub1b["paired"]["mcnemar_exact_two_sided_p"], tol=1e-15)
    check("A1b WAI difference pp",
          round(stb["wai_pct"] - fcb["wai_pct"], 4),
          round(pub1b["wai_difference_pp"], 4), tol=1e-4)
    results["audit1b_semanticttl_k1_8"] = {"freshcache": fcb, "semanticttl": stb,
                                           "paired_wai": pw1b, "dropped": drop1b}

    # ================= AUDIT 2 — held-out k=1/16 =================
    say("\n  AUDIT 2 — held-out k=1/16 (fresh / FreshCache / SemanticTTL)")
    s2 = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    a2 = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
    j1c = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_llama8b.jsonl")}
    j2c = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_qwen7b.jsonl")}
    keys2 = {"fresh": lambda d: tuple(d["fresh_key"]),
             "fc": lambda d: tuple(d["fc_key"]),
             "sttl": lambda d: tuple(d["sttl_key"])}
    r2, drop2 = rebuild(s2, a2, j1c, j2c, keys2, ABSTAIN_HELDOUT)
    pub2 = json.load(open(A2 / "answer_audit_results.json", encoding="utf-8"))
    fc2, st2 = arm_block(r2, "fc"), arm_block(r2, "sttl")
    fresh2 = Counter(r["fresh"] for r in r2)
    say(f"    rebuilt {len(r2)} rows, dropped {drop2}")
    check("A2 N_scored", len(r2), pub2["N_scored"])
    check("A2 dropped_unjudgeable", drop2, pub2["dropped_unjudgeable"])
    check("A2 fresh correct", fresh2["CORRECT"], pub2["arms"]["fresh"]["correct"])
    check("A2 fresh abstain", fresh2["ABSTAIN"], pub2["arms"]["fresh"]["abstain"])
    check("A2 FreshCache correct", fc2["correct"], pub2["arms"]["fc"]["correct"])
    check("A2 FreshCache abstain", fc2["abstain"], pub2["arms"]["fc"]["abstain"])
    check("A2 FreshCache WAI", fc2["wai"], pub2["arms"]["fc"]["wai"])
    check("A2 FreshCache conditional WAI %", round(fc2["conditional_wai_pct"], 6),
          round(pub2["arms"]["fc"]["conditional_wai_pct"], 6), tol=1e-6)
    check("A2 SemanticTTL correct", st2["correct"], pub2["arms"]["sttl"]["correct"])
    check("A2 SemanticTTL abstain", st2["abstain"], pub2["arms"]["sttl"]["abstain"])
    check("A2 SemanticTTL WAI", st2["wai"], pub2["arms"]["sttl"]["wai"])
    check("A2 SemanticTTL conditional WAI %", round(st2["conditional_wai_pct"], 6),
          round(pub2["arms"]["sttl"]["conditional_wai_pct"], 6), tol=1e-6)
    pw2 = paired_wai(r2, "sttl", "fc")
    check("A2 WAI both", pw2["wai_both"], pub2["wai_mcnemar"]["wai_both"])
    check("A2 WAI SemanticTTL-only", pw2["wai_only_sttl"],
          pub2["wai_mcnemar"]["wai_semanticttl_only"])
    check("A2 WAI FreshCache-only", pw2["wai_only_fc"],
          pub2["wai_mcnemar"]["wai_freshcache_only"])
    check("A2 WAI McNemar p", pw2["exact_mcnemar_p"],
          pub2["wai_mcnemar"]["exact_mcnemar_p"], tol=1e-16)
    bc = sum(1 for r in r2 if r["fc"] == "CORRECT" and r["sttl"] == "CORRECT")
    fo = sum(1 for r in r2 if r["fc"] == "CORRECT" and r["sttl"] != "CORRECT")
    so = sum(1 for r in r2 if r["sttl"] == "CORRECT" and r["fc"] != "CORRECT")
    check("A2 correctness both_correct", bc,
          pub2["correctness_mcnemar"]["both_correct"])
    check("A2 correctness FreshCache-only", fo,
          pub2["correctness_mcnemar"]["freshcache_correct_semanticttl_wrong"])
    check("A2 correctness SemanticTTL-only", so,
          pub2["correctness_mcnemar"]["semanticttl_correct_freshcache_wrong"])
    results["audit2_heldout_k1_16"] = {"fresh": dict(fresh2), "freshcache": fc2,
                                       "semanticttl": st2, "paired_wai": pw2,
                                       "dropped": drop2}

    # ---- diagnostic: cross-applied vocabulary (recorded, never used) ----
    say("\n  DIAGNOSTIC — cross-applied abstention vocabulary (not used by any gate)")
    rX, _ = rebuild(s1, a1, j1a, j2a, keys1, ABSTAIN_HELDOUT)
    onX = arm_block(rX, "on")
    say(f"    primary gate-on under the HELD-OUT vocabulary: WAI {onX['wai']}, "
        f"abstain {onX['abstain']}  (correct rule gives WAI {on['wai']}, "
        f"abstain {on['abstain']})")
    flipped = [r1[i]["qid"] for i in range(len(r1))
               if r1[i]["on"] != rX[i]["on"]]
    say(f"    requests whose gate-on outcome flips: {flipped}")
    diagnostic = {"primary_gate_on_wai_under_heldout_vocabulary": onX["wai"],
                  "primary_gate_on_wai_correct": on["wai"],
                  "flipped_query_ids": flipped,
                  "note": "records why an earlier reconstruction reported WAI 5; "
                          "the published value 6 is correct. No gate uses this."}

    # ---- verdict ----
    post = {str(p.relative_to(ROOT)): sha(p) for p in reads}
    unchanged = (pre == post)
    npass = sum(1 for c in CHECKS if c["ok"])
    allok = (npass == len(CHECKS)) and unchanged
    say(f"\n  CHECKS PASSED: {npass}/{len(CHECKS)}")
    say(f"  input artifacts unchanged: {unchanged}")
    say(f"  STAGE 0 GATE: {'PASSED' if allok else 'FAILED'}")
    if not allok:
        for c in CHECKS:
            if not c["ok"]:
                say(f"    FAILED: {c['check']}  got {c['reproduced']} "
                    f"required {c['required']}")

    json.dump({"stage": "0 — reproduction gate",
               "gate_passed": allok,
               "checks_passed": npass, "checks_total": len(CHECKS),
               "checks": CHECKS,
               "abstention_vocabularies": {
                   "primary_analyze2_py_21_23": list(ABSTAIN_PRIMARY),
                   "heldout_analyze_py_23_25": list(ABSTAIN_HELDOUT),
                   "applied_per_audit": True},
               "results": results,
               "cross_vocabulary_diagnostic": diagnostic,
               "answer_generation": 0, "judging": 0, "gpu_used": False,
               "web_calls": 0,
               "input_artifacts_unchanged": unchanged,
               "hashes_pre": pre, "hashes_post": post},
              open(HERE / "reproduction_gate.json", "w"), indent=2)
    json.dump({"pre": pre, "post": post, "unchanged": unchanged},
              open(HERE / "input_hashes.json", "w"), indent=2)
    (HERE / "stage0.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote reproduction_gate.json, input_hashes.json, stage0.log")
    say("  STOP — Stage 1 not started, as instructed.")
    (HERE / "stage0.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    sys.exit(0 if allok else 3)


if __name__ == "__main__":
    main()
