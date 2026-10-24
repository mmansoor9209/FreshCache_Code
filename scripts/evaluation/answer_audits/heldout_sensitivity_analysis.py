#!/usr/bin/env python3
"""
heldout_sensitivity_analysis.py -- held-out answer quality for the frozen SemanticTTL(theta=0.60, kappa=1/2)
sensitivity configuration, against the reused FreshCache and fresh-reference arms of the existing general audit.

Inputs: outputs/quality_selection/heldout_sensitivity/ (freeze.json, h_sample.jsonl, h_answers.jsonl,
h_judge_llama8b.jsonl, h_judge_qwen7b.jsonl, heldout_sample_summary.json), outputs/general_audit/ (sample.json, answers.jsonl,
judge_llama8b.jsonl, judge_qwen7b.jsonl), outputs/gold_verification/paraphrase_fidelity.csv and the gold-validity
table r_gold_validity.csv.

Results are reported on both the full audit sample and the evidence-valid subset (paraphrase fidelity SAME or a base
query, and majority gold-validity status VALID_AT_T), with the reported SemanticTTL k=1/16 arm alongside for a
three-way comparison: accuracy, WAI and conditional WAI with Wilson 95% intervals, and exact McNemar tests against
FreshCache. Held-out operational metrics are printed from heldout_sample_summary.json.

The general audit's own abstention vocabulary is used throughout; an answer is CORRECT only if both judges
(llama8b and qwen7b) say CORRECT.

Writes heldout_sensitivity_results.json and heldout_sensitivity.log to outputs/quality_selection/heldout_sensitivity/.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, hashlib, json, math, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
from collections import Counter
from fractions import Fraction

HERE = ((OUTPUT_ROOT / "quality_selection/heldout_sensitivity"))
QAB = ((OUTPUT_ROOT / "quality_selection"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
A2 = (OUTPUT_ROOT / "general_audit")
GV = (OUTPUT_ROOT / "gold_verification")
RV = ((OUTPUT_ROOT / "gold_verification/timestamp_validity"))
ABST = ("i don't know", "i do not know", "cannot be determined",
        "not enough information", "insufficient", "unable to determine",
        "no information")
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]   # noqa
LOG = []


def say(s=""):
    print(s, flush=True); LOG.append(s)


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
    return [round(100 * max(0, (c - m) / d), 3), round(100 * min(1, (c + m) / d), 3)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(Fraction(1), Fraction(
        sum(math.comb(n, i) for i in range(k + 1)) * 2, 2 ** n)))


def main():
    fz = json.load(open(HERE / "freeze.json", encoding="utf-8"))
    say("H4 — held-out answer quality for the FROZEN sensitivity configuration")
    say(f"  config: SemanticTTL theta={fz['config']['theta']:.2f} "
        f"kappa={fz['config']['kappa_frac']}   freeze sha "
        f"{fz['freeze_sha256'][:20]}...")
    say(f"  admitted by: {fz['admitted_by']}")
    say(f"  PRIMARY RESULT PRESERVED: {fz['primary_constraint_result']}")

    sample = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    new = {d["query_id"]: d for d in jl(HERE / "h_sample.jsonl")}
    ans = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
    for r in jl(HERE / "h_answers.jsonl"):
        ans[tuple(r["key"])] = r["answer"]
    j1 = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_llama8b.jsonl")}
    j2 = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_qwen7b.jsonl")}
    for r in jl(HERE / "h_judge_llama8b.jsonl"):
        j1[tuple(r["sig"])] = r["label"]
    for r in jl(HERE / "h_judge_qwen7b.jsonl"):
        j2[tuple(r["sig"])] = r["label"]

    # evidence-valid subset: fidelity SAME (or base query) and majority status VALID_AT_T
    fid = {r["query_id"]: r["final_label"] for r in csv.DictReader(
        open(GV / "paraphrase_fidelity.csv", encoding="utf-8"))}
    unit_of = {}
    for r in csv.DictReader(open(RV / "r_gold_validity.csv", encoding="utf-8")):
        for q in r["request_ids"].split("|"):
            unit_of[q] = r
    EVids = {q for q in unit_of
             if fid.get(q, "BASE_QUERY") in ("SAME", "BASE_QUERY")
             and unit_of[q]["status_majority"] == "VALID_AT_T"}

    def outc(d, key):
        a = ans.get(tuple(key))
        if a is None:
            return None
        if any(p in a.strip().lower() for p in ABST):
            return "ABSTAIN"
        s = (h(d["query"]), h(d["gold"]), h(a))
        l1, l2 = j1.get(s), j2.get(s)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    rows, dropped = [], 0
    for d in sample:
        r = {"qid": d["query_id"],
             "fresh": outc(d, d["fresh_key"]), "fc": outc(d, d["fc_key"]),
             "sttl_k1_16": outc(d, d["sttl_key"]),
             "sttl_t060_k1_2": outc(d, new[d["query_id"]]["new_sttl_key"])}
        if any(v is None for v in r.values() if not isinstance(v, str) or True) \
                and any(r[k] is None for k in ("fresh", "fc", "sttl_k1_16",
                                               "sttl_t060_k1_2")):
            dropped += 1
            continue
        r["ev"] = int(d["query_id"] in EVids)
        rows.append(r)
    say(f"\n  scored {len(rows)} of 400 (dropped {dropped})")
    EV = [r for r in rows if r["ev"]]
    say(f"  expanded-evidence evidence-valid subset: {len(EV)} requests")

    ARMS = [("fc", "FreshCache"), ("sttl_k1_16", "SemanticTTL k=1/16 (recorded)"),
            ("sttl_t060_k1_2", "SemanticTTL t=0.60 k=1/2 (frozen sensitivity)")]

    def stats(sub, a):
        n = len(sub)
        c = Counter(r[a] for r in sub)
        fcn = sum(1 for r in sub if r["fresh"] == "CORRECT")
        w = sum(1 for r in sub if r[a] == "WRONG" and r["fresh"] == "CORRECT")
        return {"N": n, "correct": c["CORRECT"], "wrong": c["WRONG"],
                "abstain": c["ABSTAIN"],
                "accuracy_pct": round(100 * c["CORRECT"] / n, 3) if n else None,
                "accuracy_ci95": wilson(c["CORRECT"], n),
                "wai": w, "wai_pct": round(100 * w / n, 3) if n else None,
                "wai_ci95": wilson(w, n), "fresh_correct_n": fcn,
                "conditional_wai_pct": round(100 * w / fcn, 3) if fcn else None,
                "conditional_wai_ci95": wilson(w, fcn)}

    out = {}
    for tag, sub in (("full_sample", rows), ("evidence_valid", EV)):
        fr = Counter(r["fresh"] for r in sub)
        say(f"\n  === {tag}  (n = {len(sub)}) ===")
        say(f"    fresh reference: {fr['CORRECT']}/{len(sub)} "
            f"({100*fr['CORRECT']/len(sub):.2f}%) correct, {fr['ABSTAIN']} abstain")
        blk = {"N": len(sub), "fresh_correct": fr["CORRECT"],
               "fresh_abstain": fr["ABSTAIN"]}
        say(f"    {'arm':<44}{'acc':>14}{'WAI':>10}{'cond.WAI':>12}{'CI95':>18}")
        for a, lab in ARMS:
            s = stats(sub, a)
            blk[a] = s
            say("    {:<44}{:>14}{:>10}{:>11.2f}%{:>18}".format(
                lab, "{}/{}".format(s["correct"], s["N"]), s["wai"],
                s["conditional_wai_pct"] or 0.0,
                str(s["conditional_wai_ci95"])))
        say(f"    paired McNemar vs FreshCache:")
        for a, lab in ARMS[1:]:
            wa = lambda r, x: r[x] == "WRONG" and r["fresh"] == "CORRECT"   # noqa
            both = sum(1 for r in sub if wa(r, a) and wa(r, "fc"))
            ao = sum(1 for r in sub if wa(r, a) and not wa(r, "fc"))
            bo = sum(1 for r in sub if wa(r, "fc") and not wa(r, a))
            p = mcnemar(ao, bo)
            blk[f"paired_{a}_vs_fc"] = {"both": both, "sttl_only": ao,
                                        "freshcache_only": bo,
                                        "exact_mcnemar_p": p}
            say(f"      {lab:<42} both {both:>3}  sttl-only {ao:>3}  "
                f"fc-only {bo:>3}  p {p:.4g}")
        out[tag] = blk

    ops = json.load(open(HERE / "heldout_sample_summary.json", encoding="utf-8"))["heldout_operational"]
    say(f"\n  HELD-OUT OPERATIONAL")
    say(f"    {'policy':<40}{'saved%':>10}{'drift%':>10}{'coverage%':>12}{'fetch/1k':>10}")
    for k, v in ops.items():
        say(f"    {k:<40}{v['search_saved_pct']:>10.4f}{v['drift_pct']:>10.4f}"
            f"{v['coverage_pct']:>12.4f}{v['fetches_per_1k']:>10.2f}")

    json.dump({"freeze_sha256": fz["freeze_sha256"],
               "primary_constraint_result": fz["primary_constraint_result"],
               "n_candidates_satisfying_primary": 0,
               "selection_used_heldout_data": False,
               "answer_quality": out, "heldout_operational": ops,
               "evidence_valid_rule": "timestamp-validity MAJORITY: "
                                      "(fidelity SAME or base query) AND VALID_AT_T",
               "reused": {"fresh_and_freshcache_answers": True,
                          "existing_judgments": True,
                          "expanded_gold_verification": True},
               "wording": "VALID_AT_T is LLM-supported temporal validity, not "
                          "independently verified factual correctness."},
              open(HERE / "heldout_sensitivity_results.json", "w"), indent=2, ensure_ascii=False)
    full = out["full_sample"]; t060 = full["sttl_t060_k1_2"]; pr = full["paired_sttl_t060_k1_2_vs_fc"]
    REF.record("heldout_audit/SemanticTTL_0.60_k0.5_fixed/CORRECT", t060["correct"]); REF.record("heldout_audit/SemanticTTL_0.60_k0.5_fixed/WAI", t060["wai"])
    REF.record("heldout_audit/SemanticTTL_0.60_k0.5_fixed/paired_vs_FreshCache_WAI", [pr["sttl_only"], pr["freshcache_only"], pr["both"]])
    (HERE / "heldout_sensitivity.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote heldout_sensitivity_results.json, heldout_sensitivity.log")


if __name__ == "__main__":
    main()
