#!/usr/bin/env python3
"""TASK 3/4/6 -- answer quality of configurations A-D on the same 400
held-out requests.

Labelling is the manuscript rule, transcribed verbatim from
l1_precision_gate/analyze_heldout.py: ABSTAIN on the abstention pattern,
otherwise CORRECT only if BOTH primary judges say CORRECT, else WRONG. The
three-way rule (judge disagreement kept separate) is reported alongside it so
the convention is never hidden.

WAI = served answer WRONG and fresh answer CORRECT.
"""
from __future__ import annotations
import json, math, pathlib, sys
from collections import Counter
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
V3 = PC.parent
RCI = V3 / "remaining_critical_issues"
sys.path.insert(0, str(RCI / "06_stronger_generator"))
import gen_judge as G            # noqa: E402

CFG = ["A_original", "B_mult_only", "C_budget_only", "D_full"]
PRIM = ["llama8b", "qwen7b"]
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    tl = (t or "").strip().lower()
    return any(p in tl for p in ABSTAIN_PAT)


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c, m = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0.0, (c - m) / d), 4), round(100 * min(1.0, (c + m) / d), 4)]


def mcnemar(b01, b10):
    n = b01 + b10
    if not n:
        return 1.0
    return min(1.0, float(sum(Fraction(math.comb(n, i), 2 ** n)
                              for i in range(0, min(b01, b10) + 1)) * 2))


def main():
    say("TASK 3/4/6 -- answer quality of configurations A-D (same 400 requests)")
    S = json.load(open(HERE / "audit_sample.json", encoding="utf-8"))["requests"]
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "cached_answers.jsonl")}
    ans.update({tuple(r["key"]): r["answer"]
                for r in jl(HERE / "answers_llama3b.jsonl")})
    jud = {j: {tuple(r["sig"]): r["label"]
               for r in jl(HERE / f"judge_{j}.jsonl")} for j in PRIM}
    say(f"  answers {len(ans):,}; judgments " +
        ", ".join(f"{j} {len(jud[j]):,}" for j in PRIM))

    def label(q, gold, key, three=False):
        a = ans.get(tuple(key))
        if a is None or not gold:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (G.h(q), G.h(gold), G.h(a))
        ls = [jud[j].get(sig) for j in PRIM]
        if any(v is None or v == "UNPARSED" for v in ls):
            return "UNJUDGED"
        if three:
            return ("JUDGE_DISAGREEMENT" if ls[0] != ls[1] else
                    "CORRECT" if ls[0] == "CORRECT" else "WRONG")
        return "CORRECT" if all(v == "CORRECT" for v in ls) else "WRONG"

    rows = []
    for d in S:
        r = {"query_id": d["query_id"], "cluster_id": d["cluster_id"],
             "fc": d["fc"],
             "fresh": label(d["query"], d["gold"], d["fresh_key"]),
             "fresh3": label(d["query"], d["gold"], d["fresh_key"], True)}
        for c in CFG:
            r[c] = label(d["query"], d["gold"], d[f"{c}_key"])
            r[c + "3"] = label(d["query"], d["gold"], d[f"{c}_key"], True)
            r[c + "_tier"] = d[f"{c}_tier"]
        rows.append(r)
    say(f"  audited requests: {len(rows)}")

    say(f"\n  == per-configuration answer quality (manuscript rule) ==")
    say(f"  {'config':<16}{'CORRECT':>9}{'WRONG':>8}{'ABSTAIN':>9}{'UNJ':>5}"
        f"{'acc/400':>10}{'95% CI':>20}{'WAI':>6}{'WAI/400':>10}"
        f"{'95% CI':>20}{'cWAI':>9}")
    out = {}
    for c in CFG:
        cnt = Counter(r[c] for r in rows)
        n = len(rows)
        corr = cnt["CORRECT"]
        wai = sum(1 for r in rows if r[c] == "WRONG" and r["fresh"] == "CORRECT")
        freshc = sum(1 for r in rows if r["fresh"] == "CORRECT")
        cwai = 100 * wai / freshc if freshc else None
        out[c] = {"counts": dict(cnt), "n": n,
                  "accuracy_pct": round(100 * corr / n, 4),
                  "accuracy_ci95": wilson(corr, n),
                  "abstain_pct": round(100 * cnt["ABSTAIN"] / n, 4),
                  "wai": wai, "wai_pct": round(100 * wai / n, 4),
                  "wai_ci95": wilson(wai, n),
                  "fresh_correct": freshc,
                  "conditional_wai_pct": (round(cwai, 4) if cwai is not None else None),
                  "conditional_wai_ci95": wilson(wai, freshc),
                  "tiers": dict(Counter(r[c + "_tier"] for r in rows))}
        say(f"  {c:<16}{corr:>9}{cnt['WRONG']:>8}{cnt['ABSTAIN']:>9}"
            f"{cnt['UNJUDGED']:>5}{100*corr/n:>9.2f}%{str(wilson(corr,n)):>20}"
            f"{wai:>6}{100*wai/n:>9.2f}%{str(wilson(wai,n)):>20}"
            f"{(f'{cwai:.2f}%' if cwai is not None else '-'):>9}")
    say(f"  fresh reference: CORRECT {out['A_original']['fresh_correct']}/400 "
        f"({100*out['A_original']['fresh_correct']/len(rows):.2f}%); "
        f"cWAI denominator = fresh CORRECT")

    # reproduction gate
    say(f"\n  == reproduction gate ==")
    pub = json.load(open(V3 / "l1_precision_gate" / "summary.json",
                         encoding="utf-8")) if (V3 / "l1_precision_gate" /
                                                "summary.json").exists() else {}
    say(f"    configuration A WAI = {out['A_original']['wai']}/400 "
        f"({out['A_original']['wai_pct']:.2f}%); manuscript headline = 7/400 "
        f"(1.75%)")
    gate = out["A_original"]["wai"] == 7
    say(f"    GATE {'PASSED' if gate else 'FAILED'} -- configuration A "
        f"{'reproduces' if gate else 'does NOT reproduce'} the published WAI")

    # three-way convention
    say(f"\n  == same table under the three-way rule (disagreement separate) ==")
    say(f"  {'config':<16}{'CORRECT':>9}{'WRONG':>8}{'DISAGREE':>10}"
        f"{'ABSTAIN':>9}{'WAI3':>6}")
    for c in CFG:
        cnt = Counter(r[c + "3"] for r in rows)
        w3 = sum(1 for r in rows if r[c + "3"] == "WRONG"
                 and r["fresh3"] == "CORRECT")
        out[c]["threeway"] = {"counts": dict(cnt), "wai": w3}
        say(f"  {c:<16}{cnt['CORRECT']:>9}{cnt['WRONG']:>8}"
            f"{cnt['JUDGE_DISAGREEMENT']:>10}{cnt['ABSTAIN']:>9}{w3:>6}")

    # paired discordances vs A
    say(f"\n  == paired correctness discordances vs configuration A ==")
    say(f"  {'contrast':<22}{'n paired':>10}{'A ok/X bad':>12}"
        f"{'A bad/X ok':>12}{'delta acc pp':>14}{'McNemar p':>12}")
    pair = {}
    for c in CFG[1:]:
        pr = [r for r in rows if r["A_original"] in ("CORRECT", "WRONG")
              and r[c] in ("CORRECT", "WRONG")]
        b01 = sum(1 for r in pr if r["A_original"] == "CORRECT" and r[c] == "WRONG")
        b10 = sum(1 for r in pr if r["A_original"] == "WRONG" and r[c] == "CORRECT")
        p = mcnemar(b01, b10)
        da = 100 * (out[c]["counts"].get("CORRECT", 0)
                    - out["A_original"]["counts"].get("CORRECT", 0)) / len(rows)
        pair[f"A_vs_{c}"] = {"n_paired": len(pr), "A_correct_X_wrong": b01,
                             "A_wrong_X_correct": b10, "mcnemar_p": p,
                             "delta_accuracy_pp": round(da, 4)}
        say(f"  {'A vs '+c:<22}{len(pr):>10}{b01:>12}{b10:>12}{da:>+14.4f}"
            f"{p:>12.4g}")

    # WAI discordance
    say(f"\n  == paired WAI discordances vs configuration A ==")
    say(f"  {'contrast':<22}{'A WAI only':>12}{'X WAI only':>12}{'both':>7}"
        f"{'McNemar p':>12}")
    for c in CFG[1:]:
        aw = [r for r in rows if r["A_original"] == "WRONG" and r["fresh"] == "CORRECT"]
        xw = [r for r in rows if r[c] == "WRONG" and r["fresh"] == "CORRECT"]
        A, X = {r["query_id"] for r in aw}, {r["query_id"] for r in xw}
        p = mcnemar(len(A - X), len(X - A))
        pair[f"A_vs_{c}"]["wai_only_A"] = len(A - X)
        pair[f"A_vs_{c}"]["wai_only_X"] = len(X - A)
        pair[f"A_vs_{c}"]["wai_both"] = len(A & X)
        pair[f"A_vs_{c}"]["wai_mcnemar_p"] = p
        say(f"  {'A vs '+c:<22}{len(A-X):>12}{len(X-A):>12}{len(A&X):>7}"
            f"{p:>12.4g}")

    json.dump({"n": len(rows), "reproduction_gate_wai_7of400": gate,
               "per_config": out, "paired_vs_A": pair},
              open(HERE / "answer_quality.json", "w"), indent=2)
    with open(HERE / "answer_quality_rows.csv", "w", encoding="utf-8") as fh:
        fh.write("query_id,cluster_id,fc,fresh," +
                 ",".join(f"{c},{c}_tier" for c in CFG) + "\n")
        for r in rows:
            fh.write(f"{r['query_id']},{r['cluster_id']},{r['fc']},{r['fresh']},"
                     + ",".join(f"{r[c]},{r[c+'_tier']}" for c in CFG) + "\n")
    (PC / "logs" / "aq_analyze.log").write_text("\n".join(log) + "\n",
                                                encoding="utf-8")
    say(f"\n  wrote answer_quality.json, answer_quality_rows.csv")


if __name__ == "__main__":
    main()
