#!/usr/bin/env python3
"""
revised_3b/r4_analyze.py — revised statuses, evidence-valid subset, WAI, and
side-by-side comparison against original Stage 3b, Stage 4 and published V30.

Two status resolutions are reported separately, never merged:
  PAIR      llama8b + qwen7b must agree, else JUDGE_DISAGREEMENT (same rule as
            the original Stage 3b, so the comparison is like-for-like)
  MAJORITY  2-of-3 with mistral7b breaking ties; reported as its own column so
            the third judge's effect is isolated

VALID_AT_T is LLM-supported temporal validity against stored snapshots. It is
NOT independently verified factual correctness, and nothing here says otherwise.
"""
from __future__ import annotations
import csv, hashlib, json, math, pathlib, sys
from collections import Counter
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
GV = HERE.parent
ROOT = GV.parent.parent
A1 = ROOT / "validation" / "mixed_age_full_policy_audit"
A1B = ROOT / "validation" / "baseline_operating_points"
A2 = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"

ABST_P = ("i don't know", "i dont know", "cannot determine",
          "not enough information", "insufficient", "no information",
          "unclear from the context")
ABST_H = ("i don't know", "i do not know", "cannot be determined",
          "not enough information", "insufficient", "unable to determine",
          "no information")
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]  # noqa
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


def arm(rows, a):
    n = len(rows)
    c = Counter(r[a] for r in rows)
    fc = sum(1 for r in rows if r["fresh"] == "CORRECT")
    w = sum(1 for r in rows if r[a] == "WRONG" and r["fresh"] == "CORRECT")
    return {"N": n, "correct": c["CORRECT"], "abstain": c["ABSTAIN"],
            "accuracy_pct": round(100 * c["CORRECT"] / n, 3) if n else None,
            "wai": w, "wai_pct": round(100 * w / n, 3) if n else None,
            "wai_ci95": wilson(w, n), "fresh_correct_n": fc,
            "conditional_wai_pct": round(100 * w / fc, 3) if fc else None}


def paired(rows, a, b):
    wa = lambda r, x: r[x] == "WRONG" and r["fresh"] == "CORRECT"   # noqa
    return {"wai_both": sum(1 for r in rows if wa(r, a) and wa(r, b)),
            f"only_{a}": sum(1 for r in rows if wa(r, a) and not wa(r, b)),
            f"only_{b}": sum(1 for r in rows if wa(r, b) and not wa(r, a)),
            "exact_mcnemar_p": mcnemar(
                sum(1 for r in rows if wa(r, a) and not wa(r, b)),
                sum(1 for r in rows if wa(r, b) and not wa(r, a)))}


def main():
    tasks = {t["unit_id"]: t for t in jl(HERE / "r_validity_tasks.jsonl")}
    L = {r["unit_id"]: r["status"] for r in jl(HERE / "r_validity_llama8b.jsonl")}
    Q = {r["unit_id"]: r["status"] for r in jl(HERE / "r_validity_qwen7b.jsonl")}
    M = {r["unit_id"]: r["status"] for r in jl(HERE / "r_validity_mistral7b.jsonl")}
    miss = [u for u in tasks if u not in L or u not in Q]
    if miss:
        say(f"  STOP: {len(miss)} units missing a primary judge label"); sys.exit(3)
    say(f"  units judged by the primary pair: {len(tasks)}")
    say(f"  units judged by the third model : {len(M)}")

    VALIDish = {"VALID_AT_T"}
    rows = []
    for u, t in tasks.items():
        l, q = L[u], Q[u]
        pair = l if l == q else "JUDGE_DISAGREEMENT"
        if l == q:
            maj, src = l, "pair_agreement"
        elif u in M:
            m = M[u]
            maj = (m if (m == l or m == q) else "NO_MAJORITY")
            src = "third_judge_majority" if maj != "NO_MAJORITY" else "three_way_split"
        else:
            maj, src = "JUDGE_DISAGREEMENT", "no_third_judge"
        rows.append({k: t[k] for k in ("unit_id", "base_query_id", "base_query",
                                       "snapshot_round", "gold", "gold_source",
                                       "gold_year", "volatility",
                                       "freshness_classes", "question_type",
                                       "mode", "n_pages_kept", "evidence_chars",
                                       "old_evidence_chars", "old_pages_kept",
                                       "old_status", "request_ids")}
                    | {"status_llama8b": l, "status_qwen7b": q,
                       "status_mistral7b": M.get(u, ""),
                       "status_pair": pair, "status_majority": maj,
                       "majority_source": src})

    def dist(key, sub=None):
        return dict(Counter(r[key] for r in (sub or rows)))
    say(f"\n  PAIR resolution    : {dist('status_pair')}")
    say(f"  MAJORITY resolution: {dist('status_majority')}")
    say(f"  majority source    : {dist('majority_source')}")
    say(f"\n  vs ORIGINAL Stage 3b: {dict(Counter(r['old_status'] for r in rows))}")
    trans = Counter((r["old_status"], r["status_majority"]) for r in rows)
    say("  transition old -> revised(majority), top 12:")
    for (a, b), c in trans.most_common(12):
        say(f"    {a:<22} -> {b:<26} {c}")
    say(f"\n  by question mode (majority):")
    for m in ("AS_STATED", "CURRENT"):
        sub = [r for r in rows if r["mode"] == m]
        say(f"    {m:<10} n={len(sub):>4}  {dist('status_majority', sub)}")
    say(f"  by freshness class (majority VALID rate):")
    for f in ("TIMELESS", "SLOW", "MEDIUM", "FAST"):
        sub = [r for r in rows if f in r["freshness_classes"].split("|")]
        if sub:
            v = sum(1 for r in sub if r["status_majority"] in VALIDish)
            say(f"    {f:<9} VALID {v:>3}/{len(sub):<4} = {100*v/len(sub):>5.1f}%")
    say(f"  by gold source (majority VALID rate):")
    for s in sorted({r["gold_source"] for r in rows if r["gold_source"]}):
        sub = [r for r in rows if r["gold_source"] == s]
        v = sum(1 for r in sub if r["status_majority"] in VALIDish)
        say(f"    {s:<20} VALID {v:>3}/{len(sub):<4} = {100*v/len(sub):>5.1f}%")

    # ---- combine with fidelity ----
    fid = {r["query_id"]: r["final_label"] for r in csv.DictReader(
        open(GV / "paraphrase_fidelity.csv", encoding="utf-8"))}
    rmap = {r["query_id"]: r for r in csv.DictReader(
        open(GV / "request_map.csv", encoding="utf-8"))}
    unit_of = {}
    for r in rows:
        for qid in r["request_ids"].split("|"):
            unit_of[qid] = r
    keep = {}
    for res in ("pair", "majority"):
        ks = set()
        for qid in rmap:
            f = fid.get(qid, "BASE_QUERY")
            u = unit_of.get(qid)
            if f in ("SAME", "BASE_QUERY") and u and u[f"status_{res}"] in VALIDish:
                ks.add(qid)
        keep[res] = ks
        say(f"\n  EVIDENCE-VALID requests ({res}): {len(ks)}")

    # ---- audits ----
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
    s2 = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    a2 = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
    k1 = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_llama8b.jsonl")}
    k2 = {tuple(r["sig"]): r["label"] for r in jl(A2 / "judge_qwen7b.jsonl")}

    def build(sample, arms, ans, ja, jb, pat, keyfn):
        out = []
        for d in sample:
            g = {a: outcome(ans, ja, jb, d["query"], d["gold"], keyfn(d, a), pat)
                 for a in arms}
            if any(v is None for v in g.values()):
                continue
            g["qid"] = d["query_id"]
            out.append(g)
        return out

    P = build(s1, ("fresh", "on", "off", "sttl_k1_8"), a1, j1, j2, ABST_P,
              lambda d, a: (("fresh", d["fresh_ctx_sha"], h(d["query"])) if a == "fresh"
                            else tuple(ss[d["query_id"]]["sttl_key"]) if a == "sttl_k1_8"
                            else tuple(d[f"{a}_key"])))
    H = build(s2, ("fresh", "fc", "sttl"), a2, k1, k2, ABST_H,
              lambda d, a: tuple(d[f"{a}_key"]))

    pub = {"primary": {"fresh": 66, "on": 61, "off": 59, "sttl_k1_8": 9,
                       "wai_on": 6, "wai_off": 8, "wai_sttl": 58},
           "heldout": {"fresh": 81, "fc": 75, "sttl": 16, "wai_fc": 7,
                       "wai_sttl": 64}}
    gates = [(len(P), 400), (sum(1 for r in P if r["fresh"] == "CORRECT"), 66),
             (sum(1 for r in P if r["on"] == "WRONG" and r["fresh"] == "CORRECT"), 6),
             (sum(1 for r in P if r["sttl_k1_8"] == "WRONG" and r["fresh"] == "CORRECT"), 58),
             (len(H), 400), (sum(1 for r in H if r["fresh"] == "CORRECT"), 81),
             (sum(1 for r in H if r["fc"] == "WRONG" and r["fresh"] == "CORRECT"), 7),
             (sum(1 for r in H if r["sttl"] == "WRONG" and r["fresh"] == "CORRECT"), 64)]
    ok = all(a == b for a, b in gates)
    say(f"\n  full-sample reproduction gate: {'PASSED' if ok else 'FAILED'} "
        f"{[f'{a}/{b}' for a, b in gates]}")
    if not ok:
        sys.exit(3)

    res = {}
    for tag, rws, arms, labs in (("mixed_age_400", P, ("on", "off", "sttl_k1_8"),
                                  ("FreshCache", "gate-off", "SemanticTTL")),
                                 ("heldout_400", H, ("fc", "sttl"),
                                  ("FreshCache", "SemanticTTL"))):
        res[tag] = {"full": {"N": len(rws),
                             "fresh_correct": sum(1 for r in rws if r["fresh"] == "CORRECT")}}
        for a in arms:
            res[tag]["full"][a] = arm(rws, a)
        res[tag]["full"]["paired"] = paired(rws, arms[-1], arms[0])
        for resn in ("pair", "majority"):
            sub = [r for r in rws if r["qid"] in keep[resn]]
            blk = {"N": len(sub),
                   "fresh_correct": sum(1 for r in sub if r["fresh"] == "CORRECT")}
            for a in arms:
                blk[a] = arm(sub, a)
            blk["paired"] = paired(sub, arms[-1], arms[0])
            res[tag][f"subset_{resn}"] = blk
        say(f"\n  === {tag} ===")
        say(f"    {'metric':<26}{'published':>18}{'orig Stage4':>18}"
            f"{'revised PAIR':>18}{'revised MAJORITY':>20}")
        o4 = json.load(open(GV / "stage4_summary.json", encoding="utf-8"))["audits"][tag]["subset"]
        def row(name, f_, o_, p_, m_):
            say(f"    {name:<26}{str(f_):>18}{str(o_):>18}{str(p_):>18}{str(m_):>20}")
        row("N", res[tag]["full"]["N"], o4["N"],
            res[tag]["subset_pair"]["N"], res[tag]["subset_majority"]["N"])
        row("fresh correct", res[tag]["full"]["fresh_correct"],
            o4["fresh"]["correct"], res[tag]["subset_pair"]["fresh_correct"],
            res[tag]["subset_majority"]["fresh_correct"])
        for a, lab in zip(arms, labs):
            row(f"{lab} correct", res[tag]["full"][a]["correct"], o4[a]["correct"],
                res[tag]["subset_pair"][a]["correct"],
                res[tag]["subset_majority"][a]["correct"])
            row(f"{lab} WAI", res[tag]["full"][a]["wai"], o4[a]["wai"],
                res[tag]["subset_pair"][a]["wai"],
                res[tag]["subset_majority"][a]["wai"])
            row(f"{lab} cond.WAI %", res[tag]["full"][a]["conditional_wai_pct"],
                o4[a]["conditional_wai_pct"],
                res[tag]["subset_pair"][a]["conditional_wai_pct"],
                res[tag]["subset_majority"][a]["conditional_wai_pct"])
        say(f"    paired McNemar full     : {res[tag]['full']['paired']}")
        say(f"    paired McNemar PAIR     : {res[tag]['subset_pair']['paired']}")
        say(f"    paired McNemar MAJORITY : {res[tag]['subset_majority']['paired']}")

    with open(HERE / "r_gold_validity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"stage": "revised 3b + 4",
               "expanded_evidence": True, "question_type_aware": True,
               "third_judge": "mistralai/Mistral-7B-Instruct-v0.3",
               "resolutions": {"pair": "llama8b+qwen7b must agree",
                               "majority": "2-of-3 with mistral7b breaking ties"},
               "status_pair": dist("status_pair"),
               "status_majority": dist("status_majority"),
               "majority_source": dist("majority_source"),
               "original_stage3b_status": dict(Counter(r["old_status"] for r in rows)),
               "transition_old_to_revised": {f"{a}->{b}": c for (a, b), c in trans.items()},
               "evidence_valid_requests": {k: len(v) for k, v in keep.items()},
               "audits": res, "published_v30": pub,
               "full_sample_reproduction_passed": ok,
               "wording_constraint": "VALID_AT_T is LLM-supported temporal validity "
                                     "against stored snapshots, NOT independently "
                                     "verified factual correctness.",
               "web_retrieval": 0, "answer_generation": 0},
              open(HERE / "r4_summary.json", "w"), indent=2, ensure_ascii=False)
    (HERE / "r4.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote r_gold_validity.csv, r4_summary.json, r4.log")


if __name__ == "__main__":
    main()
