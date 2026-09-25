#!/usr/bin/env python3
"""
Section 1 (B, C) -- does FreshCache's L1 query mismatch actually cause harm?

Every held-out FreshCache L1 hit is scored on:
  * query equivalence, by a four-model jury (majority vote; non-unanimity is
    reported as its own category and never silently resolved),
  * answer correctness of the STORED answer the cache returns,
  * answer correctness of the FRESH answer the same request would have got,
  * WAI  =  stored WRONG and fresh CORRECT,
and cross-tabulated by equivalence, provenance, similarity band, Jaccard band,
answer-type agreement and freshness class.

The question the reviewer asked -- what fraction of mismatches are harmful
versus harmless evidence-sharing paraphrases -- is answered directly from that
cross-tabulation.
"""
from __future__ import annotations
import csv, hashlib, json, math, pathlib, sys
from collections import Counter, defaultdict
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
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


def mcnemar_exact(b, c):
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
    pe = sum((sum(1 for x, _ in pr if x == l)/n) *
             (sum(1 for _, y in pr if y == l)/n) for l in labs)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def band(v, edges):
    if v is None:
        return "n/a"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return f">={edges[-1]:.2f}"


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    D = json.load(open(HERE / "l1_hits.json", encoding="utf-8"))
    hits = D["hits"]
    say("V3 SECTION 1 (B, C) -- held-out L1 mismatch and harm")
    say(f"  held-out FreshCache L1 hits: {len(hits):,} (all of them)")
    say(f"  FreshCache replay: saved "
        f"{D['freshcache_metrics']['search_saved_pct']:.4f}%, "
        f"L1 {D['freshcache_metrics']['l1_hits']:,}, "
        f"L2 {D['freshcache_metrics']['l2_hits']:,}")

    # ---- equivalence jury ----
    eq = {}
    for j in JUDGES:
        for r in jl(HERE / f"l1_equiv_{j}.jsonl"):
            eq.setdefault(r["qid"], {})[j] = r["label"]
    present = sorted({j for v in eq.values() for j in v})
    say(f"\n  == equivalence jury == judges present: {present}")
    verdict = {}
    for x in hits:
        v = {j: l for j, l in eq.get(x["query_id"], {}).items()
             if l in ("SAME", "DIFFERENT")}
        if len(v) < 2:
            verdict[x["query_id"]] = "UNJUDGED"
            continue
        c = Counter(v.values())
        if c["SAME"] == c["DIFFERENT"]:
            verdict[x["query_id"]] = "JURY_TIE"
        else:
            verdict[x["query_id"]] = c.most_common(1)[0][0]
    vc = Counter(verdict.values())
    judged = vc["SAME"] + vc["DIFFERENT"]
    say(f"    majority SAME      {vc['SAME']:>5}")
    say(f"    majority DIFFERENT {vc['DIFFERENT']:>5}")
    say(f"    jury tie           {vc['JURY_TIE']:>5}   (own category; not mapped)")
    say(f"    unjudged           {vc['UNJUDGED']:>5}")
    if judged:
        mm = vc["DIFFERENT"]
        say(f"    MISMATCH RATE = {mm}/{judged} = {100*mm/judged:.2f}%  "
            f"CI{wilson(mm, judged)}   (majority of a {len(present)}-model jury)")
    for j in present:
        col = [eq[q].get(j) for q in verdict if q in eq]
        col = [c for c in col if c in ("SAME", "DIFFERENT")]
        if col:
            say(f"    per-juror {j:<10} DIFFERENT "
                f"{sum(1 for c in col if c=='DIFFERENT'):>5}/{len(col):<5} = "
                f"{100*sum(1 for c in col if c=='DIFFERENT')/len(col):>6.2f}%")
    una = sum(1 for q in verdict
              if len({l for l in eq.get(q, {}).values()
                      if l in ('SAME', 'DIFFERENT')}) == 1
              and len(eq.get(q, {})) >= len(present))
    say(f"    unanimous across all {len(present)} jurors: {una}/{len(hits)} = "
        f"{100*una/max(1,len(hits)):.2f}%")
    say(f"    pairwise Cohen kappa")
    for i in range(len(present)):
        for k in range(i + 1, len(present)):
            a = [eq.get(q, {}).get(present[i]) for q in verdict]
            b = [eq.get(q, {}).get(present[k]) for q in verdict]
            say(f"      {present[i]:<10} vs {present[k]:<10} = {kappa(a, b)}")

    # ---- correctness ----
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "l1_answers.jsonl")}
    jud = {j: {tuple(r["sig"]): r["label"]
               for r in jl(HERE / f"l1_judge_{j}.jsonl")} for j in JUDGES}
    jud = {j: v for j, v in jud.items() if v}
    prim = [j for j in ("llama8b", "qwen7b") if j in jud] or sorted(jud)[:2]
    say(f"\n  == answer correctness == primary judge pair {prim} "
        f"(Llama-3.2-3B generated, so it is not its own primary judge)")

    def lab(x, which):
        k = x.get(f"{which}_key")
        if not k:
            return None
        a = ans.get(tuple(k))
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        if not x["gold"]:
            return None
        sig = (h(x["incoming_query"]), h(x["gold"]), h(a))
        ls = [jud[j].get(sig) for j in prim]
        if any(v is None or v == "UNPARSED" for v in ls):
            return "UNJUDGED"
        if ls[0] != ls[1]:
            return "JUDGE_DISAGREEMENT"
        return "CORRECT" if ls[0] == "CORRECT" else "WRONG"

    rows = []
    for x in hits:
        s, f = lab(x, "stored"), lab(x, "fresh")
        rows.append({**x, "equivalence": verdict[x["query_id"]],
                     "stored_label": s, "fresh_label": f,
                     "wai": int(s == "WRONG" and f == "CORRECT"),
                     "sim_band": band(x["similarity"], [0.40, 0.80, 0.90, 0.95, 1.0]),
                     "jac_band": band(x["jaccard"], [0.0, 0.3, 0.5, 0.7, 1.0])})

    det = [r for r in rows if r["stored_label"] in ("CORRECT", "WRONG")
           and r["fresh_label"] in ("CORRECT", "WRONG")]
    say(f"    hits with BOTH arms determinate: {len(det):,}/{len(rows):,}")
    sc = sum(1 for r in det if r["stored_label"] == "CORRECT")
    fc_ = sum(1 for r in det if r["fresh_label"] == "CORRECT")
    wai = sum(r["wai"] for r in det)
    c2w = sum(1 for r in det if r["fresh_label"] == "CORRECT"
              and r["stored_label"] == "WRONG")
    w2c = sum(1 for r in det if r["fresh_label"] == "WRONG"
              and r["stored_label"] == "CORRECT")
    say(f"    stored-answer accuracy {sc}/{len(det)} = "
        f"{100*sc/max(1,len(det)):.2f}%  CI{wilson(sc, len(det))}")
    say(f"    fresh-answer accuracy  {fc_}/{len(det)} = "
        f"{100*fc_/max(1,len(det)):.2f}%  CI{wilson(fc_, len(det))}")
    say(f"    WAI (stored WRONG and fresh CORRECT) = {wai}/{len(det)} = "
        f"{100*wai/max(1,len(det)):.2f}%  CI{wilson(wai, len(det))}")
    say(f"      conditional on the fresh answer being CORRECT: "
        f"{wai}/{fc_} = {100*wai/max(1,fc_):.2f}%  CI{wilson(wai, fc_)}")
    say(f"    exact McNemar stored vs fresh: C->W {c2w}  W->C {w2c}  "
        f"p = {mcnemar_exact(c2w, w2c):.4g}")

    # ---- the reviewer's actual question ----
    say(f"\n  == are the mismatches harmful? ==")
    grp = defaultdict(list)
    for r in det:
        grp[r["equivalence"]].append(r)
    say(f"    {'equivalence':<20}{'n':>6}{'stored acc%':>13}{'WAI':>6}"
        f"{'WAI%':>8}{'WAI 95% CI':>18}")
    harm = {}
    for k in ("SAME", "DIFFERENT", "JURY_TIE", "UNJUDGED"):
        g = grp.get(k, [])
        if not g:
            continue
        a = sum(1 for r in g if r["stored_label"] == "CORRECT")
        w = sum(r["wai"] for r in g)
        harm[k] = {"n": len(g), "stored_correct": a, "wai": w,
                   "stored_accuracy_pct": round(100*a/len(g), 4),
                   "wai_pct": round(100*w/len(g), 4),
                   "wai_ci": wilson(w, len(g))}
        say(f"    {k:<20}{len(g):>6}{100*a/len(g):>12.2f}%{w:>6}"
            f"{100*w/len(g):>7.2f}%{str(wilson(w, len(g))):>18}")
    d = grp.get("DIFFERENT", [])
    if d:
        bad = sum(1 for r in d if r["stored_label"] == "WRONG")
        ok = sum(1 for r in d if r["stored_label"] == "CORRECT")
        say(f"\n    of the {len(d)} determinate MISMATCHED hits:")
        say(f"      {bad} ({100*bad/len(d):.2f}%) returned a WRONG answer "
            f"-- genuinely harmful")
        say(f"      {ok} ({100*ok/len(d):.2f}%) returned a CORRECT answer "
            f"anyway -- harmless evidence-sharing / near-paraphrase")
        s_ = grp.get("SAME", [])
        if s_:
            b1 = sum(r["wai"] for r in d); n1 = len(d)
            b2 = sum(r["wai"] for r in s_); n2 = len(s_)
            say(f"      WAI among MISMATCHED {b1}/{n1} = {100*b1/n1:.2f}%  "
                f"vs MATCHED {b2}/{n2} = {100*b2/n2:.2f}%")

    for key, name in (("provenance", "provenance"), ("sim_band", "similarity"),
                      ("jac_band", "Jaccard"),
                      ("answer_type_match", "answer-type agreement"),
                      ("fc_incoming", "freshness class")):
        say(f"\n    by {name}")
        g2 = defaultdict(list)
        for r in det:
            g2[r[key]].append(r)
        for k in sorted(g2, key=str):
            g = g2[k]
            a = sum(1 for r in g if r["stored_label"] == "CORRECT")
            mmm = sum(1 for r in g if r["equivalence"] == "DIFFERENT")
            say(f"      {str(k):<28}n {len(g):>4}  stored acc "
                f"{100*a/len(g):>6.2f}%  mismatch {100*mmm/len(g):>6.2f}%  "
                f"WAI {100*sum(r['wai'] for r in g)/len(g):>6.2f}%")

    cols = ["query_id", "incoming_query", "cached_query", "similarity",
            "jaccard", "entity_match", "answer_type_match", "provenance",
            "fc_incoming", "age_seconds", "equivalence", "stored_label",
            "fresh_label", "wai", "sim_band", "jac_band"]
    with open(HERE / "l1_per_hit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    json.dump({"n_hits": len(hits), "jury": dict(vc),
               "judges": present, "primary_answer_judges": prim,
               "mismatch_rate_pct": (round(100*vc["DIFFERENT"]/judged, 4)
                                     if judged else None),
               "mismatch_ci": wilson(vc["DIFFERENT"], judged) if judged else None,
               "determinate": len(det), "stored_correct": sc,
               "fresh_correct": fc_, "wai": wai,
               "mcnemar": {"c2w": c2w, "w2c": w2c,
                           "p": mcnemar_exact(c2w, w2c)},
               "by_equivalence": harm},
              open(HERE / "l1_analysis.json", "w"), indent=2)
    (HERE / "l1_analyze.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("\n  wrote l1_per_hit.csv, l1_analysis.json")


if __name__ == "__main__":
    main()
