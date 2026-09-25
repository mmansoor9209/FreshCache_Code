#!/usr/bin/env python3
"""
Issue 2 -- analysis.

Compares the two arms on their COMPLETE REALISED L1 POPULATIONS. It does not
report "mismatch on the shared hits": arm B's population is a different set of
requests, produced by a different cache trajectory, and is scored as such.
"""
from __future__ import annotations
import csv, hashlib, itertools, json, math, pathlib, sys
from collections import Counter, defaultdict
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parents[1]
RCI = V3 / "remaining_critical_issues"
JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
PRIMARY = ["llama8b", "qwen7b"]        # the generator (3B) is never primary
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    tl = (t or "").strip().lower()
    return any(p in tl for p in ABSTAIN_PAT)


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
    n = len(rows[0]); N = len(rows)
    labs = sorted({x for r in rows for x in r})
    P = [(sum(v * v for v in Counter(r).values()) - n) / (n * (n - 1)) for r in rows]
    pbar = sum(P) / N
    pe = sum((sum(Counter(r)[l] for r in rows) / (N * n)) ** 2 for l in labs)
    return None if pe == 1 else round((pbar - pe) / (1 - pe), 4)


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    prep = json.load(open(HERE / "prep_summary.json", encoding="utf-8"))
    hits = list(csv.DictReader(open(HERE / "l1_hits.csv", encoding="utf-8")))
    say("ISSUE 2 -- stricter L1 gate, full held-out evaluation")
    say(f"  frozen gate {prep['frozen_gate']}")

    # ---------------- equivalence jury ----------------
    eq = defaultdict(dict)
    for j in JUDGES:
        for r in jl(HERE / f"equiv_{j}.jsonl"):
            eq[r["pair_id"]][j] = r["label"]
    present = sorted({j for v in eq.values() for j in v})
    say(f"\n  == equivalence jury == judges: {present}")

    def verdicts(pid):
        v = {j: l for j, l in eq.get(pid, {}).items()
             if l in ("SAME", "DIFFERENT")}
        if len(v) < 2:
            return "UNJUDGED", v
        c = Counter(v.values())
        if c["SAME"] == c["DIFFERENT"]:
            return "JURY_TIE", v
        return c.most_common(1)[0][0], v

    for x in hits:
        pid = f"{h(x['incoming_query'])}_{h(x['serving_cached_query'])}"
        x["pair_id"] = pid
        x["equivalence"], _ = verdicts(pid)

    # ---------------- answers ----------------
    ans = {tuple(r["key"]): r["answer"]
           for r in jl(HERE / "answers_llama3b.jsonl")}
    jud = {j: {tuple(r["sig"]): r["label"] for r in jl(HERE / f"judge_{j}.jsonl")}
           for j in JUDGES}
    jud = {j: v for j, v in jud.items() if v}
    prim = [j for j in PRIMARY if j in jud] or sorted(jud)[:2]
    say(f"  primary correctness judges: {prim} "
        f"(Llama-3.2-3B generated, so it is never its own primary judge)")

    def lab(q, gold, key):
        if not key or not gold:
            return None
        a = ans.get(tuple(key.split("|")))
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h(q), h(gold), h(a))
        ls = [jud[j].get(sig) for j in prim]
        if any(v is None or v == "UNPARSED" for v in ls):
            return "UNJUDGED"
        if ls[0] != ls[1]:
            return "JUDGE_DISAGREEMENT"
        return "CORRECT" if ls[0] == "CORRECT" else "WRONG"

    for x in hits:
        x["stored_label"] = lab(x["incoming_query"], x["reference_answer"],
                                x["stored_key"])
        x["fresh_label"] = lab(x["incoming_query"], x["reference_answer"],
                               x["fresh_key"])
        x["wai"] = int(x["stored_label"] == "WRONG"
                       and x["fresh_label"] == "CORRECT")

    # ---------------- the two realised populations ----------------
    say(f"\n  == realised L1 populations (NOT a shared-hit comparison) ==")
    pops, out = {}, {}
    for arm in ("published", "strict"):
        P = [x for x in hits if x["arm"] == arm]
        pops[arm] = P
        j = [x for x in P if x["equivalence"] in ("SAME", "DIFFERENT")]
        mm = sum(1 for x in j if x["equivalence"] == "DIFFERENT")
        ties = sum(1 for x in P if x["equivalence"] == "JURY_TIE")
        n = len(P)
        conv = {
            "ties_to_different": (mm + ties, n),
            "ties_excluded": (mm, len(j)),
            "ties_to_same": (mm, n)}
        xc = [x for x in P if x["cross_cluster"] == "1"]
        xcj = [x for x in xc if x["equivalence"] in ("SAME", "DIFFERENT")]
        xcm = sum(1 for x in xcj if x["equivalence"] == "DIFFERENT")
        det = [x for x in P if x["stored_label"] in ("CORRECT", "WRONG")
               and x["fresh_label"] in ("CORRECT", "WRONG")]
        sc = sum(1 for x in det if x["stored_label"] == "CORRECT")
        fcorr = sum(1 for x in det if x["fresh_label"] == "CORRECT")
        wai = sum(x["wai"] for x in det)
        out[arm] = {
            "l1_hits": n, "jury_determinate": len(j), "jury_ties": ties,
            "mismatch": {k: {"k": a, "n": b,
                             "pct": (round(100 * a / b, 4) if b else None),
                             "ci": wilson(a, b) if b else None}
                         for k, (a, b) in conv.items()},
            "cross_cluster_hits": len(xc),
            "cross_cluster_mismatch": {"k": xcm, "n": len(xcj),
                                       "pct": (round(100 * xcm / len(xcj), 4)
                                               if xcj else None),
                                       "ci": wilson(xcm, len(xcj)) if xcj else None},
            "answer_determinate": len(det),
            "stored_accuracy": {"k": sc, "n": len(det),
                                "pct": (round(100 * sc / len(det), 4) if det else None),
                                "ci": wilson(sc, len(det)) if det else None},
            "fresh_correct": fcorr,
            "wai": {"k": wai, "n": len(det),
                    "pct": (round(100 * wai / len(det), 4) if det else None),
                    "ci": wilson(wai, len(det)) if det else None},
            "conditional_wai": {"k": wai, "n": fcorr,
                                "pct": (round(100 * wai / fcorr, 4) if fcorr else None),
                                "ci": wilson(wai, fcorr) if fcorr else None}}
    say(f"    {'metric':<34}{'published gate':>22}{'stricter gate':>22}")
    def row(nm, f):
        say(f"    {nm:<34}{f('published'):>22}{f('strict'):>22}")
    row("realised L1 hits", lambda a: f"{out[a]['l1_hits']:,}")
    for k, nm in (("ties_to_different", "mismatch (ties->DIFFERENT)"),
                  ("ties_excluded", "mismatch (ties excluded)"),
                  ("ties_to_same", "mismatch (ties->SAME)")):
        row(nm, lambda a, k=k: (f"{out[a]['mismatch'][k]['k']}/"
                                f"{out[a]['mismatch'][k]['n']} "
                                f"{out[a]['mismatch'][k]['pct']:.2f}%"
                                if out[a]['mismatch'][k]['pct'] is not None else "-"))
    row("jury ties", lambda a: f"{out[a]['jury_ties']}")
    row("cross-cluster hits", lambda a: f"{out[a]['cross_cluster_hits']:,}")
    row("cross-cluster mismatch", lambda a: (
        f"{out[a]['cross_cluster_mismatch']['k']}/"
        f"{out[a]['cross_cluster_mismatch']['n']} "
        f"{out[a]['cross_cluster_mismatch']['pct']:.2f}%"
        if out[a]['cross_cluster_mismatch']['pct'] is not None else "-"))
    row("answer-determinate hits", lambda a: f"{out[a]['answer_determinate']:,}")
    row("stored-answer accuracy", lambda a: (
        f"{out[a]['stored_accuracy']['k']}/{out[a]['stored_accuracy']['n']} "
        f"{out[a]['stored_accuracy']['pct']:.2f}%"
        if out[a]['stored_accuracy']['pct'] is not None else "-"))
    row("WAI", lambda a: (f"{out[a]['wai']['k']}/{out[a]['wai']['n']} "
                          f"{out[a]['wai']['pct']:.2f}%"
                          if out[a]['wai']['pct'] is not None else "-"))
    row("conditional WAI", lambda a: (
        f"{out[a]['conditional_wai']['k']}/{out[a]['conditional_wai']['n']} "
        f"{out[a]['conditional_wai']['pct']:.2f}%"
        if out[a]['conditional_wai']['pct'] is not None else "-"))

    for k, nm in (("ties_to_different", "ties->DIFFERENT"),
                  ("ties_excluded", "ties excluded"),
                  ("ties_to_same", "ties->SAME")):
        a, b = out["published"]["mismatch"][k], out["strict"]["mismatch"][k]
        if a["pct"] is not None and b["pct"] is not None:
            ov = not (a["ci"][1] < b["ci"][0] or b["ci"][1] < a["ci"][0])
            say(f"    mismatch {nm:<18} published {a['pct']:.2f}% CI{a['ci']}  vs "
                f"strict {b['pct']:.2f}% CI{b['ci']}  -> CIs "
                f"{'OVERLAP' if ov else 'do NOT overlap'}")

    # ---------------- cost ----------------
    d = prep["displacement"]
    say(f"\n  == cost of the stricter gate ==")
    say(f"    L1 hits            {prep['l1_hits_published']:,} -> "
        f"{prep['l1_hits_strict']:,}  ({prep['l1_hits_strict']-prep['l1_hits_published']:+,})")
    say(f"    rejected originals {sum(d['lost_l1'].values()):,}  -> "
        f"{d['lost_l1']}")
    say(f"    newly-L1 (cache divergence) {d['gained_l1']:,}")
    say(f"    search calls       {prep['arm_published']['search_calls']:,} -> "
        f"{prep['arm_strict']['search_calls']:,}  ({d['delta_searches']:+,})")
    say(f"    fetches            {prep['arm_published']['fetches']:,} -> "
        f"{prep['arm_strict']['fetches']:,}  ({d['delta_fetches']:+,})")
    say(f"    generations        {prep['arm_published']['generations']:,} -> "
        f"{prep['arm_strict']['generations']:,}  ({d['delta_generations']:+,})"
        f"   <- the entire cost")
    say(f"    search savings     {prep['arm_published']['search_saved_pct']:.4f}% -> "
        f"{prep['arm_strict']['search_saved_pct']:.4f}%  (unchanged)")

    # ---------------- 400-request audit ----------------
    say(f"\n  == 400-request held-out answer audit, both arms ==")
    A = json.load(open(HERE / "audit_sample.json", encoding="utf-8"))
    arows, alab = [], {}
    for d_ in A["requests"]:
        r = {"query_id": d_["query_id"], "fc": d_["fc"],
             "fresh": lab(d_["query"], d_["gold"], "|".join(d_["fresh_key"]))}
        for arm in ("published", "strict"):
            r[arm] = lab(d_["query"], d_["gold"], "|".join(d_[f"{arm}_key"]))
            r[f"{arm}_tier"] = d_[f"{arm}_tier"]
        arows.append(r)
    say(f"    execution paths differing between arms: {A['paths_differ']}")
    say(f"    {'arm':<14}{'CORRECT':>9}{'WRONG':>7}{'ABSTAIN':>9}{'DISAGREE':>10}"
        f"{'acc% det':>10}")
    audit = {}
    for arm in ("fresh", "published", "strict"):
        c = Counter(r[arm] for r in arows)
        det = c["CORRECT"] + c["WRONG"]
        audit[arm] = {"correct": c["CORRECT"], "wrong": c["WRONG"],
                      "abstain": c["ABSTAIN"],
                      "judge_disagreement": c["JUDGE_DISAGREEMENT"],
                      "determinate": det,
                      "accuracy_pct": (round(100 * c["CORRECT"] / det, 4) if det else None),
                      "accuracy_ci": wilson(c["CORRECT"], det) if det else None}
        say(f"    {arm:<14}{c['CORRECT']:>9}{c['WRONG']:>7}{c['ABSTAIN']:>9}"
            f"{c['JUDGE_DISAGREEMENT']:>10}"
            + (f"{100*c['CORRECT']/det:>9.2f}%" if det else f"{'-':>10}"))
    fc_ok = [r for r in arows if r["fresh"] == "CORRECT"]
    for arm in ("published", "strict"):
        w = sum(1 for r in fc_ok if r[arm] == "WRONG")
        b = sum(1 for r in arows if r["fresh"] == "CORRECT" and r[arm] == "WRONG")
        c_ = sum(1 for r in arows if r["fresh"] == "WRONG" and r[arm] == "CORRECT")
        audit[arm]["wai_all400"] = {"k": w, "n": len(arows),
                                    "pct": round(100 * w / len(arows), 4),
                                    "ci": wilson(w, len(arows))}
        audit[arm]["wai_fresh_correct"] = {"k": w, "n": len(fc_ok),
                                           "pct": (round(100 * w / len(fc_ok), 4)
                                                   if fc_ok else None),
                                           "ci": wilson(w, len(fc_ok)) if fc_ok else None}
        audit[arm]["mcnemar_vs_fresh"] = {"b": b, "c": c_, "p": mcnemar(b, c_)}
        say(f"    {arm:<14} WAI {w}/{len(arows)} = {100*w/len(arows):.2f}% "
            f"CI{wilson(w, len(arows))};  on fresh-CORRECT {w}/{len(fc_ok)} = "
            f"{(100*w/len(fc_ok) if fc_ok else 0):.2f}%;  exact McNemar vs "
            f"fresh C->W {b} W->C {c_} p = {mcnemar(b, c_):.4g}")
    bb = sum(1 for r in arows if r["published"] == "CORRECT" and r["strict"] == "WRONG")
    cc = sum(1 for r in arows if r["published"] == "WRONG" and r["strict"] == "CORRECT")
    p_pair = mcnemar(bb, cc)
    audit["published_vs_strict_mcnemar"] = {"b": bb, "c": cc, "p": p_pair}
    say(f"    published vs strict, exact McNemar: C->W {bb}  W->C {cc}  "
        f"p = {p_pair:.4g}"
        + ("   (not significant; NOT an equivalence claim -- no "
           "non-inferiority margin was pre-specified)" if p_pair > 0.05 else ""))

    # ---------------- jury reliability ----------------
    say(f"\n  == equivalence-jury reliability ==")
    pids = sorted({x["pair_id"] for x in hits})
    mat = [[eq.get(p, {}).get(j) for j in present] for p in pids]
    mat = [[None if v in (None, "UNPARSED") else v for v in r] for r in mat]
    full = [r for r in mat if all(r)]
    say(f"    pairs {len(pids):,}; complete rows {len(full):,}")
    for i, j in enumerate(present):
        col = [r[i] for r in mat if r[i]]
        k = sum(1 for v in col if v == "DIFFERENT")
        say(f"    {j:<11} DIFFERENT {k:>5}/{len(col):<5} = "
            f"{100*k/max(1,len(col)):>6.2f}%")
    una = sum(1 for r in full if len(set(r)) == 1)
    say(f"    unanimous {una}/{len(full)} = {100*una/max(1,len(full)):.2f}%;  "
        f"NON-unanimous {len(full)-una} (own category, never mapped)")
    say(f"    Fleiss kappa = {fleiss(full)}")
    pk = {}
    for a_, b_ in itertools.combinations(range(len(present)), 2):
        kk = kappa([r[a_] for r in mat], [r[b_] for r in mat])
        pk[f"{present[a_]}|{present[b_]}"] = kk
        say(f"      {present[a_]:<10} vs {present[b_]:<10} kappa = {kk}")

    cols = ["arm", "query_id", "cluster_id", "incoming_query",
            "serving_cached_query", "stored_answer_query", "similarity",
            "jaccard", "answer_type_incoming", "answer_type_cached",
            "answer_type_match", "entities_incoming", "entities_cached",
            "entity_match", "freshness_class", "cache_age_seconds",
            "cross_cluster", "is_paraphrase", "reference_answer",
            "equivalence", "stored_label", "fresh_label", "wai"]
    with open(HERE / "judgments.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(hits)
    with open(HERE / "answer_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arows[0]))
        w.writeheader(); w.writerows(arows)
    json.dump({"frozen_gate": prep["frozen_gate"], "populations": out,
               "cost": prep["displacement"],
               "arm_published": prep["arm_published"],
               "arm_strict": prep["arm_strict"],
               "audit": audit,
               "jury": {"judges": present, "pairs": len(pids),
                        "complete": len(full), "unanimous": una,
                        "fleiss_kappa": fleiss(full),
                        "pairwise_cohen_kappa": pk},
               "primary_answer_judges": prim},
              open(HERE / "summary.json", "w"), indent=2)
    (HERE / "analyze.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("\n  wrote judgments.csv, answer_audit.csv, summary.json")


if __name__ == "__main__":
    main()
