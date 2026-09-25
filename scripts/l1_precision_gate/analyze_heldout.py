#!/usr/bin/env python3
"""
Held-out analysis for the L1-Precision gate.

Mismatch is computed on each arm's COMPLETE REALISED L1 POPULATION, under all
three tie conventions. Answer quality uses the manuscript rule and the
manuscript's collection-class arm, and is gated on reproducing WAI 7/400.
"""
from __future__ import annotations
import csv, hashlib, itertools, json, math, pathlib, sys
from collections import Counter, defaultdict
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
PRIM = ["llama8b", "qwen7b"]
ARMS = ["published", "strict090", "precision"]
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]


def h16(s):
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
    n, N = len(rows[0]), len(rows)
    labs = sorted({x for r in rows for x in r})
    P = [(sum(v * v for v in Counter(r).values()) - n) / (n * (n - 1)) for r in rows]
    pe = sum((sum(Counter(r)[l] for r in rows) / (N * n)) ** 2 for l in labs)
    return None if pe == 1 else round((sum(P) / N - pe) / (1 - pe), 4)


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    R = json.load(open(HERE / "heldout_replay.json", encoding="utf-8"))
    FR = R["frozen"]
    hits = list(csv.DictReader(open(HERE / "heldout_l1_hits.csv",
                                    encoding="utf-8")))
    say("L1-PRECISION GATE -- held-out analysis")
    say(f"  frozen gate: sim>={FR['sim_floor']} jaccard>={FR['jaccard_floor']} "
        f"answer_type={FR['answer_type']} guards={FR['guards']}")

    # ---------------- equivalence jury ----------------
    eq = defaultdict(dict)
    for j in JUDGES:
        for r in jl(HERE / f"heldout_equiv_{j}.jsonl"):
            eq[r["pair_id"]][j] = r["label"]
    present = sorted({j for v in eq.values() for j in v})
    say(f"\n  equivalence jury: {present}; pairs judged {len(eq):,}")

    def verdict(pid):
        v = {j: l for j, l in eq.get(pid, {}).items()
             if l in ("SAME", "DIFFERENT")}
        if len(v) < 2:
            return "UNJUDGED"
        c = Counter(v.values())
        return "JURY_TIE" if c["SAME"] == c["DIFFERENT"] else c.most_common(1)[0][0]

    V = {pid: verdict(pid) for pid in eq}
    for x in hits:
        x["equivalence"] = V.get(x["pair_id"], "UNJUDGED")

    # ---------------- answers ----------------
    ans = {tuple(r["key"]): r["answer"]
           for r in jl(HERE / "heldout_answers_llama3b.jsonl")}
    jud = {j: {tuple(r["sig"]): r["label"]
               for r in jl(HERE / f"heldout_judge_{j}.jsonl")} for j in JUDGES}
    jud = {j: v for j, v in jud.items() if v}

    def lab_manuscript(q, gold, key):
        """The manuscript rule, verbatim: ABSTAIN on the abstention pattern,
        else CORRECT only if BOTH primary judges say CORRECT, else WRONG."""
        if not key or not gold:
            return None
        a = ans.get(tuple(key.split("|")) if isinstance(key, str) else tuple(key))
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h16(q), h16(gold), h16(a))
        l1 = jud.get(PRIM[0], {}).get(sig)
        l2 = jud.get(PRIM[1], {}).get(sig)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    def lab_threeway(q, gold, key):
        if not key or not gold:
            return None
        a = ans.get(tuple(key.split("|")) if isinstance(key, str) else tuple(key))
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h16(q), h16(gold), h16(a))
        ls = [jud.get(j, {}).get(sig) for j in PRIM]
        if any(v is None or v == "UNPARSED" for v in ls):
            return "UNJUDGED"
        if ls[0] != ls[1]:
            return "JUDGE_DISAGREEMENT"
        return "CORRECT" if ls[0] == "CORRECT" else "WRONG"

    for x in hits:
        x["stored_label"] = lab_manuscript(x["incoming_query"],
                                           x["reference_answer"], x["stored_key"])
        x["fresh_label"] = lab_manuscript(x["incoming_query"],
                                          x["reference_answer"], x["fresh_key"])
        x["wai"] = int(x["stored_label"] == "WRONG"
                       and x["fresh_label"] == "CORRECT")

    # ---------------- realised populations ----------------
    say(f"\n  == mismatch on the COMPLETE realised L1 population of each arm ==")
    pops = {}
    for arm in ARMS:
        P = [x for x in hits if x["arm"] == arm]
        n = len(P)
        det = [x for x in P if x["equivalence"] in ("SAME", "DIFFERENT")]
        diff = sum(1 for x in det if x["equivalence"] == "DIFFERENT")
        ties = sum(1 for x in P if x["equivalence"] == "JURY_TIE")
        xc = [x for x in P if x["cross_cluster"] == "1"]
        xdet = [x for x in xc if x["equivalence"] in ("SAME", "DIFFERENT")]
        xdiff = sum(1 for x in xdet if x["equivalence"] == "DIFFERENT")
        xties = sum(1 for x in xc if x["equivalence"] == "JURY_TIE")
        ad = [x for x in P if x["stored_label"] in ("CORRECT", "WRONG")
              and x["fresh_label"] in ("CORRECT", "WRONG")]
        sc = sum(1 for x in ad if x["stored_label"] == "CORRECT")
        fc = sum(1 for x in ad if x["fresh_label"] == "CORRECT")
        w = sum(x["wai"] for x in ad)
        pops[arm] = {
            "l1_hits": n, "jury_ties": ties, "jury_determinate": len(det),
            "mismatch_ties_diff": {"k": diff + ties, "n": n,
                                   "pct": (round(100*(diff+ties)/n, 4) if n else None),
                                   "ci": wilson(diff + ties, n)},
            "mismatch_no_ties": {"k": diff, "n": len(det),
                                 "pct": (round(100*diff/len(det), 4) if det else None),
                                 "ci": wilson(diff, len(det))},
            "mismatch_ties_same": {"k": diff, "n": n,
                                   "pct": (round(100*diff/n, 4) if n else None),
                                   "ci": wilson(diff, n)},
            "cross_cluster_hits": len(xc),
            "cross_cluster_mismatch_ties_diff": {
                "k": xdiff + xties, "n": len(xc),
                "pct": (round(100*(xdiff+xties)/len(xc), 4) if xc else None)},
            "cross_cluster_mismatch_no_ties": {
                "k": xdiff, "n": len(xdet),
                "pct": (round(100*xdiff/len(xdet), 4) if xdet else None)},
            "answer_determinate": len(ad),
            "stored_accuracy": {"k": sc, "n": len(ad),
                                "pct": (round(100*sc/len(ad), 4) if ad else None)},
            "hit_level_wai": {"k": w, "n": len(ad),
                              "pct": (round(100*w/len(ad), 4) if ad else None)},
            "hit_level_conditional_wai": {"k": w, "n": fc,
                                          "pct": (round(100*w/fc, 4) if fc else None)},
        }
    say(f"    {'metric':<30}" + "".join(f"{a:>16}" for a in ARMS))
    def row(nm, f):
        say(f"    {nm:<30}" + "".join(f"{f(a):>16}" for a in ARMS))
    row("realised L1 hits", lambda a: f"{pops[a]['l1_hits']:,}")
    for k, nm in (("mismatch_ties_diff", "mismatch ties->DIFF"),
                  ("mismatch_no_ties", "mismatch no ties"),
                  ("mismatch_ties_same", "mismatch ties->SAME")):
        row(nm, lambda a, k=k: (f"{pops[a][k]['k']}/{pops[a][k]['n']} "
                                f"{pops[a][k]['pct']:.2f}%"
                                if pops[a][k]['pct'] is not None else "-"))
    row("jury ties", lambda a: f"{pops[a]['jury_ties']}")
    row("cross-cluster hits", lambda a: f"{pops[a]['cross_cluster_hits']:,}")
    row("x-cluster mm ties->DIFF", lambda a: (
        f"{pops[a]['cross_cluster_mismatch_ties_diff']['pct']:.2f}%"
        if pops[a]['cross_cluster_mismatch_ties_diff']['pct'] is not None else "-"))
    row("stored-answer accuracy", lambda a: (
        f"{pops[a]['stored_accuracy']['k']}/{pops[a]['stored_accuracy']['n']}"
        if pops[a]['stored_accuracy']['n'] else "-"))
    for k, nm in (("mismatch_ties_diff", "ties->DIFF"),
                  ("mismatch_no_ties", "no ties")):
        p_, s_ = pops["published"][k], pops["precision"][k]
        if p_["pct"] is not None and s_["pct"] is not None:
            ov = not (p_["ci"][1] < s_["ci"][0] or s_["ci"][1] < p_["ci"][0])
            say(f"    mismatch {nm:<12} published {p_['pct']:.2f}% CI{p_['ci']} "
                f"vs precision {s_['pct']:.2f}% CI{s_['ci']} -> CIs "
                f"{'OVERLAP' if ov else 'do NOT overlap'}")

    # ---------------- 400-request audit ----------------
    A = json.load(open(HERE / "heldout_audit_sample.json", encoding="utf-8"))
    say(f"\n  == held-out 400-request audit ==")
    say(f"    freshness-class source: {A['freshness_class_source']} "
        f"(manuscript headline arm)")
    say(f"    labelling: the manuscript rule (CORRECT only if BOTH "
        f"{PRIM} say CORRECT; disagreement -> WRONG)")
    arows = []
    for d in A["requests"]:
        r = {"query_id": d["query_id"], "fc": d["fc"],
             "fresh": lab_manuscript(d["query"], d["gold"], d["fresh_key"])}
        for a in ARMS:
            r[a] = lab_manuscript(d["query"], d["gold"], d[f"{a}_key"])
            r[f"{a}_tier"] = d[f"{a}_tier"]
            r[f"{a}_3way"] = lab_threeway(d["query"], d["gold"], d[f"{a}_key"])
        r["fresh_3way"] = lab_threeway(d["query"], d["gold"], d["fresh_key"])
        arows.append(r)
    fc_ok = [r for r in arows if r["fresh"] == "CORRECT"]
    say(f"    fresh CORRECT {len(fc_ok)}/{len(arows)}")
    audit = {}
    say(f"    {'arm':<12}{'CORRECT':>9}{'WRONG':>7}{'ABSTAIN':>9}{'acc%':>8}"
        f"{'WAI':>6}{'WAI%':>8}{'cWAI%':>8}{'C->W':>6}{'W->C':>6}{'McNemar p':>12}")
    for a in ["fresh"] + ARMS:
        c = Counter(r[a] for r in arows)
        det = c["CORRECT"] + c["WRONG"]
        e = {"correct": c["CORRECT"], "wrong": c["WRONG"],
             "abstain": c["ABSTAIN"], "determinate": det,
             "accuracy_pct": round(100 * c["CORRECT"] / len(arows), 4),
             "accuracy_ci": wilson(c["CORRECT"], len(arows))}
        if a != "fresh":
            w = sum(1 for r in arows if r[a] == "WRONG" and r["fresh"] == "CORRECT")
            b = w
            cc = sum(1 for r in arows if r[a] == "CORRECT" and r["fresh"] == "WRONG")
            e.update({"wai": w, "wai_pct": round(100 * w / len(arows), 4),
                      "wai_ci": wilson(w, len(arows)),
                      "conditional_wai_pct": (round(100 * w / len(fc_ok), 4)
                                              if fc_ok else None),
                      "reverse": cc,
                      "mcnemar_vs_fresh_p": mcnemar(b, cc),
                      "wai_request_ids": [r["query_id"] for r in arows
                                          if r[a] == "WRONG"
                                          and r["fresh"] == "CORRECT"]})
            say(f"    {a:<12}{c['CORRECT']:>9}{c['WRONG']:>7}{c['ABSTAIN']:>9}"
                f"{100*c['CORRECT']/len(arows):>7.2f}%{w:>6}"
                f"{100*w/len(arows):>7.2f}%"
                f"{(100*w/len(fc_ok) if fc_ok else 0):>7.2f}%{b:>6}{cc:>6}"
                f"{mcnemar(b, cc):>12.4g}")
        else:
            say(f"    {a:<12}{c['CORRECT']:>9}{c['WRONG']:>7}{c['ABSTAIN']:>9}"
                f"{100*c['CORRECT']/len(arows):>7.2f}%")
        audit[a] = e
    gate_ok = audit["published"]["wai"] == 7
    say(f"\n    REPRODUCTION GATE: published arm WAI = "
        f"{audit['published']['wai']}/400 "
        f"({audit['published']['wai_pct']:.2f}%) vs the manuscript headline "
        f"7/400 (1.75%) -> {'PASS' if gate_ok else 'MISMATCH'}")
    for a in ARMS:
        if a == "published":
            continue
        b = sum(1 for r in arows if r["published"] == "CORRECT" and r[a] == "WRONG")
        c_ = sum(1 for r in arows if r["published"] == "WRONG" and r[a] == "CORRECT")
        p = mcnemar(b, c_)
        audit[a]["mcnemar_vs_published"] = {"b": b, "c": c_, "p": p}
        say(f"    published vs {a}: exact McNemar C->W {b}  W->C {c_}  "
            f"p = {p:.4g}"
            + ("   (not significant; NOT an equivalence claim -- no "
               "non-inferiority margin was pre-specified)" if p > 0.05 else ""))
    say(f"    WAI request IDs:")
    for a in ARMS:
        say(f"      {a:<12} {audit[a].get('wai_request_ids')}")
    say(f"\n    SECONDARY (three-way rule, judge disagreement separated):")
    for a in ["fresh"] + ARMS:
        c = Counter(r[f"{a}_3way"] for r in arows)
        say(f"      {a:<12} CORRECT {c['CORRECT']:>4}  WRONG {c['WRONG']:>4}  "
            f"ABSTAIN {c['ABSTAIN']:>3}  DISAGREE {c['JUDGE_DISAGREEMENT']:>4}")

    # ---------------- cost ----------------
    arms = R["arms"]
    n = R["n_requests"]
    pubm = arms["published"]
    cost = {}
    say(f"\n  == cost analysis ==")
    say(f"    {'arm':<12}{'L1':>7}{'saved%':>10}{'searches':>10}{'fetches':>9}"
        f"{'fetch/1k':>10}{'gens':>9}{'gen/1k':>9}{'drift%':>9}{'cov%':>9}")
    for a in ARMS:
        m = arms[a]
        cost[a] = {"l1_hits": m["l1_hits"], "search_saved_pct": m["search_saved_pct"],
                   "searches": m["search_calls"], "fetches": m["fetches"],
                   "fetch_per_1k": m["fetch_per_1k"],
                   "generations": m["generations"], "gen_per_1k": m["gen_per_1k"],
                   "drift_pct": m["drift_pct"], "coverage_pct": m["coverage_pct"],
                   "delta_vs_published": {
                       "l1_hits": m["l1_hits"] - pubm["l1_hits"],
                       "search_saved_pp": round(m["search_saved_pct"] - pubm["search_saved_pct"], 4),
                       "searches": m["search_calls"] - pubm["search_calls"],
                       "fetches": m["fetches"] - pubm["fetches"],
                       "fetch_per_1k": round(m["fetch_per_1k"] - pubm["fetch_per_1k"], 3),
                       "generations": m["generations"] - pubm["generations"],
                       "gen_per_1k": round(m["gen_per_1k"] - pubm["gen_per_1k"], 3),
                       "drift_pp": (round(m["drift_pct"] - pubm["drift_pct"], 4)
                                    if m["drift_pct"] is not None else None),
                       "coverage_pp": (round(m["coverage_pct"] - pubm["coverage_pct"], 4)
                                       if m["coverage_pct"] is not None else None)}}
        say(f"    {a:<12}{m['l1_hits']:>7,}{m['search_saved_pct']:>9.4f}%"
            f"{m['search_calls']:>10,}{m['fetches']:>9,}{m['fetch_per_1k']:>10.2f}"
            f"{m['generations']:>9,}{m['gen_per_1k']:>9.2f}"
            + (f"{m['drift_pct']:>8.4f}%" if m['drift_pct'] is not None else f"{'-':>9}")
            + (f"{m['coverage_pct']:>8.4f}%" if m['coverage_pct'] is not None else f"{'-':>9}"))
    say(f"\n    displacement of published L1 hits:")
    for a, d in R["displacement"].items():
        say(f"      {a:<12} rejected {d['rejected_total']:,} -> "
            f"absorbed by L2 {d['absorbed_by_L2']:,}, "
            f"fresh search {d['require_fresh_search']:,}, "
            f"other {d['other']:,}; newly-L1 {d['newly_L1_from_cache_divergence']:,}")
    cost["displacement"] = R["displacement"]
    json.dump(cost, open(HERE / "cost_analysis.json", "w"), indent=2)

    # ---------------- jury reliability ----------------
    pids = sorted({x["pair_id"] for x in hits})
    mat = [[eq.get(p, {}).get(j) for j in present] for p in pids]
    mat = [[None if v in (None, "UNPARSED") else v for v in r] for r in mat]
    full = [r for r in mat if all(r)]
    say(f"\n  == equivalence-jury reliability ==")
    say(f"    pairs {len(pids):,}; complete {len(full):,}; "
        f"Fleiss kappa {fleiss(full)}")
    for i, j in enumerate(present):
        col = [r[i] for r in mat if r[i]]
        say(f"    {j:<11} DIFFERENT {sum(1 for v in col if v=='DIFFERENT'):>5}/"
            f"{len(col):<5} = {100*sum(1 for v in col if v=='DIFFERENT')/max(1,len(col)):>6.2f}%")
    pk = {}
    for a_, b_ in itertools.combinations(range(len(present)), 2):
        pk[f"{present[a_]}|{present[b_]}"] = kappa([r[a_] for r in mat],
                                                   [r[b_] for r in mat])
    say(f"    pairwise Cohen kappa: {pk}")

    cols = ["arm", "request_id", "cluster_id", "incoming_query", "cached_query",
            "similarity", "jaccard", "entities_incoming", "entities_cached",
            "entity_match", "answer_type_incoming", "answer_type_cached",
            "numeric_tokens_incoming", "numeric_tokens_cached",
            "numeric_guard_pass", "negation_incoming", "negation_cached",
            "negation_guard_pass", "relation_incoming", "relation_cached",
            "comparative_guard_pass", "freshness_class", "cache_age_seconds",
            "reference_answer", "cross_cluster", "equivalence",
            "stored_label", "fresh_label", "wai"]
    with open(HERE / "heldout_l1_hits_judged.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(hits)
    with open(HERE / "answer_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arows[0]))
        w.writeheader(); w.writerows(arows)
    with open(HERE / "jury_results.jsonl", "w", encoding="utf-8") as fh:
        for pid in pids:
            fh.write(json.dumps({"pair_id": pid, "verdict": V.get(pid),
                                 "labels": eq.get(pid, {})}) + "\n")
    json.dump({"frozen": FR, "populations": pops, "audit": audit,
               "cost": cost, "reproduction_gate_wai_7of400": gate_ok,
               "jury": {"models": present, "pairs": len(pids),
                        "fleiss_kappa": fleiss(full),
                        "pairwise_cohen_kappa": pk}},
              open(HERE / "summary.json", "w"), indent=2)
    (HERE / "logs" / "heldout_analysis.log").write_text("\n".join(log) + "\n",
                                                        encoding="utf-8")
    say("\n  wrote heldout_l1_hits_judged.csv, answer_audit.csv, "
        "jury_results.jsonl, cost_analysis.json, summary.json")


if __name__ == "__main__":
    main()
