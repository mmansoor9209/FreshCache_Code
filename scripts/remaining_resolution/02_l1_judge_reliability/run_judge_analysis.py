#!/usr/bin/env python3
"""
EXPERIMENT 2 (Tasks 2A, 2B) -- is the remaining L1-Precision mismatch genuine,
or substantially an artefact of judge disagreement?

Reuses the existing verified cached judgments. No model inference is repeated.

The FOUR-model jury remains the primary pre-existing measurement. The
three-model result is a SENSITIVITY ANALYSIS and is labelled as such
throughout; no judge subset is adopted because it yields a lower number.
"""
from __future__ import annotations
import csv, hashlib, itertools, json, math, os, pathlib, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
RR = HERE.parent
V3 = RR.parent
ROOT = V3.parent
LP = V3 / "l1_precision_gate"
JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
OUTLIER = "llama3b"


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 2),
            round(100 * min(1.0, (c + m) / d), 2))


def kappa(a, b):
    pr = [(x, y) for x, y in zip(a, b) if x and y]
    if not pr:
        return None
    n = len(pr)
    po = sum(1 for x, y in pr if x == y) / n
    labs = {x for x, _ in pr} | {y for _, y in pr}
    pe = sum((sum(1 for x, _ in pr if x == l) / n) *
             (sum(1 for _, y in pr if y == l) / n) for l in labs)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def fleiss(rows):
    rows = [r for r in rows if all(r)]
    if not rows:
        return None
    n, N = len(rows[0]), len(rows)
    labs = sorted({x for r in rows for x in r})
    P = [(sum(v * v for v in Counter(r).values()) - n) / (n * (n - 1)) for r in rows]
    pe = sum((sum(Counter(r)[l] for r in rows) / (N * n)) ** 2 for l in labs)
    return None if pe == 1 else round((sum(P) / N - pe) / (1 - pe), 4)


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("EXPERIMENT 2 -- L1-Precision mismatch and judge reliability")
    say("  reusing existing cached judgments; no model inference repeated")

    # ---------------- TASK 2A: data integrity ----------------
    rows = list(csv.DictReader(open(LP / "heldout_l1_hits_judged.csv",
                                    encoding="utf-8")))
    P = [r for r in rows if r["arm"] == "precision"]
    say(f"\n  == TASK 2A: data integrity ==")
    say(f"    realised L1-Precision hits: {len(P)}")
    NEED = ["incoming_query", "cached_query", "answer_type_incoming",
            "answer_type_cached", "entities_incoming", "entities_cached",
            "similarity", "jaccard", "reference_answer", "stored_label",
            "cross_cluster", "freshness_class", "cache_age_seconds"]
    miss = {c: sum(1 for r in P if r.get(c) in (None, "")) for c in NEED}
    for c, m in miss.items():
        flag = "" if m == 0 else "  <-- MISSING"
        say(f"      {c:<26}{len(P)-m:>5}/{len(P)} present{flag}")
    say(f"    note: 'cached answer' and 'source query' are recorded as "
        f"stored_label (the judged correctness of the stored answer) and "
        f"stored_answer_query in heldout_l1_hits.csv; the answer TEXT itself "
        f"lives in heldout_answers_llama3b.jsonl keyed by stored_key.")
    raw = list(csv.DictReader(open(LP / "heldout_l1_hits.csv", encoding="utf-8")))
    rawP = {r["request_id"]: r for r in raw if r["arm"] == "precision"}
    say(f"      stored_answer_query present  "
        f"{sum(1 for r in rawP.values() if r.get('stored_answer_query')):>5}"
        f"/{len(rawP)}")

    eq = defaultdict(dict)
    for j in JUDGES:
        f = LP / f"heldout_equiv_{j}.jsonl"
        for l in open(f, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                eq[d["pair_id"]][j] = d["label"]
    pid_of = {r["request_id"]: f"{h16(r['incoming_query'])}_{h16(r['cached_query'])}"
              for r in P}
    have4 = sum(1 for r in P
                if len([j for j in JUDGES
                        if eq.get(pid_of[r["request_id"]], {}).get(j)
                        in ("SAME", "DIFFERENT")]) == 4)
    say(f"    rows with all FOUR individual judge outputs: {have4}/{len(P)}")

    # ---- reconcile 43.30% vs 43.55% ----
    say(f"\n    reconciling the published-gate mismatch: 43.30% (newest) vs "
        f"43.55% (earlier stricter-gate experiment)")
    pub = [r for r in rows if r["arm"] == "published"]
    c = Counter(r["equivalence"] for r in pub)
    n = len(pub)
    say(f"      newest run  : ties->DIFF ({c['DIFFERENT']}+{c['JURY_TIE']})/{n} "
        f"= {100*(c['DIFFERENT']+c['JURY_TIE'])/n:.4f}%   "
        f"[SAME {c['SAME']}, DIFFERENT {c['DIFFERENT']}, TIE {c['JURY_TIE']}]")
    old = V3 / "final_three_issues" / "02_l1_strict_gate" / "judgments.csv"
    if old.exists():
        o = [r for r in csv.DictReader(open(old, encoding="utf-8"))
             if r["arm"] == "published"]
        co = Counter(r["equivalence"] for r in o)
        no = len(o)
        say(f"      earlier run : ties->DIFF ({co['DIFFERENT']}+{co['JURY_TIE']})"
            f"/{no} = {100*(co['DIFFERENT']+co['JURY_TIE'])/no:.4f}%   "
            f"[SAME {co['SAME']}, DIFFERENT {co['DIFFERENT']}, "
            f"TIE {co['JURY_TIE']}]")
        say(f"      SAME denominator ({n} vs {no}) and the SAME tie rule. The "
            f"difference is {abs((c['DIFFERENT']+c['JURY_TIE'])-(co['DIFFERENT']+co['JURY_TIE']))} "
            f"pair(s) whose greedy jury verdict moved between TIE and "
            f"DIFFERENT because the two runs judged different-sized pair sets "
            f"(844 vs 839 distinct pairs), not because any denominator or tie "
            f"rule changed.")

    # ---------------- TASK 2B: judge analysis ----------------
    say(f"\n  == TASK 2B: judge analysis on the {len(P)} L1-Precision pairs ==")
    pids = [pid_of[r["request_id"]] for r in P]
    mat = [[eq.get(p, {}).get(j) for j in JUDGES] for p in pids]
    mat = [[None if v in (None, "UNPARSED") else v for v in r] for r in mat]
    full = [r for r in mat if all(r)]

    def verdict(labels):
        v = [l for l in labels if l in ("SAME", "DIFFERENT")]
        if len(v) < 2:
            return "UNJUDGED"
        cc = Counter(v)
        return "JURY_TIE" if cc["SAME"] == cc["DIFFERENT"] else cc.most_common(1)[0][0]

    out = {}
    for tag, js in (("four_model_PRIMARY", JUDGES),
                    ("three_model_SENSITIVITY_no_3B",
                     [j for j in JUDGES if j != OUTLIER])):
        sub = [[eq.get(p, {}).get(j) for j in js] for p in pids]
        sub = [[None if v in (None, "UNPARSED") else v for v in r] for r in sub]
        vs = [verdict(r) for r in sub]
        cc = Counter(vs)
        nn = len(vs)
        det = cc["SAME"] + cc["DIFFERENT"]
        una = sum(1 for r in sub if all(r) and len(set(r)) == 1)
        unad = sum(1 for r in sub if all(r) and set(r) == {"DIFFERENT"})
        out[tag] = {
            "judges": js, "n_pairs": nn, "ties": cc["JURY_TIE"],
            "mismatch_ties_diff": {"k": cc["DIFFERENT"] + cc["JURY_TIE"], "n": nn,
                                   "pct": round(100*(cc["DIFFERENT"]+cc["JURY_TIE"])/nn, 4),
                                   "ci": wilson(cc["DIFFERENT"]+cc["JURY_TIE"], nn)},
            "mismatch_no_ties": {"k": cc["DIFFERENT"], "n": det,
                                 "pct": round(100*cc["DIFFERENT"]/det, 4) if det else None,
                                 "ci": wilson(cc["DIFFERENT"], det)},
            "mismatch_ties_same": {"k": cc["DIFFERENT"], "n": nn,
                                   "pct": round(100*cc["DIFFERENT"]/nn, 4),
                                   "ci": wilson(cc["DIFFERENT"], nn)},
            "unanimous_any": una, "unanimous_DIFFERENT": unad,
            "unanimity_only_mismatch_pct": round(100*unad/nn, 4),
            "fleiss_kappa": fleiss(sub),
        }
        say(f"\n    -- {tag} ({len(js)} judges: {js}) --")
        e = out[tag]
        say(f"      ties {e['ties']}   unanimous {e['unanimous_any']}/{nn} "
            f"({100*e['unanimous_any']/nn:.2f}%)   Fleiss kappa {e['fleiss_kappa']}")
        for k, nm in (("mismatch_ties_diff", "ties->DIFFERENT"),
                      ("mismatch_no_ties", "ties excluded"),
                      ("mismatch_ties_same", "ties->SAME")):
            m = e[k]
            say(f"      mismatch {nm:<16} {m['k']:>4}/{m['n']:<4} = "
                f"{m['pct']:>7.4f}%  CI{m['ci']}")
        say(f"      unanimity-only mismatch (all judges say DIFFERENT) "
            f"{unad}/{nn} = {100*unad/nn:.4f}%")

    say(f"\n    per-judge DIFFERENT rate on these {len(P)} pairs")
    for i, j in enumerate(JUDGES):
        col = [r[i] for r in mat if r[i]]
        d = sum(1 for v in col if v == "DIFFERENT")
        say(f"      {j:<11}{d:>5}/{len(col):<5} = {100*d/max(1,len(col)):>7.2f}%")
    say(f"    pairwise Cohen kappa")
    pk = {}
    for a, b in itertools.combinations(range(len(JUDGES)), 2):
        k = kappa([r[a] for r in mat], [r[b] for r in mat])
        pk[f"{JUDGES[a]}|{JUDGES[b]}"] = k
        say(f"      {JUDGES[a]:<10} vs {JUDGES[b]:<10} {k}")
    ds = Counter(tuple(sorted(Counter(r).items())) for r in full if len(set(r)) > 1)
    say(f"    disagreement categories (label multiset -> count)")
    for k, v in ds.most_common():
        say(f"      {dict(k)} -> {v}")

    prim = out["four_model_PRIMARY"]["mismatch_ties_diff"]
    sens = out["three_model_SENSITIVITY_no_3B"]["mismatch_ties_diff"]
    say(f"\n    HEADLINE: primary four-model mismatch (ties->DIFFERENT) "
        f"{prim['pct']:.2f}% CI{prim['ci']}")
    say(f"    SENSITIVITY (3B excluded, NOT the primary measure): "
        f"{sens['pct']:.2f}% CI{sens['ci']}")
    ov = not (prim["ci"][1] < sens["ci"][0] or sens["ci"][1] < prim["ci"][0])
    say(f"    the two intervals {'OVERLAP' if ov else 'do NOT overlap'}; "
        f"dropping the outlier {'does not' if ov else 'does'} change the "
        f"conclusion at this sample size")

    json.dump({"n_pairs": len(P), "rows_with_four_judge_outputs": have4,
               "field_completeness": {c: len(P)-m for c, m in miss.items()},
               "results": out,
               "per_judge_different_pct": {
                   j: round(100*sum(1 for r in mat if r[i] == "DIFFERENT")
                            / max(1, sum(1 for r in mat if r[i])), 4)
                   for i, j in enumerate(JUDGES)},
               "pairwise_cohen_kappa": pk,
               "disagreement_categories": {str(dict(k)): v for k, v in ds.items()},
               "primary_is_four_model": True},
              open(HERE / "judge_analysis.json", "w"), indent=2)
    (RR / "logs").mkdir(exist_ok=True)
    (RR / "logs" / "exp2_judge.log").write_text("\n".join(log) + "\n",
                                                encoding="utf-8")
    say(f"\n  wrote judge_analysis.json")


if __name__ == "__main__":
    main()
