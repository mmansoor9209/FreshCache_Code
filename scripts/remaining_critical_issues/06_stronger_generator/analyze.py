#!/usr/bin/env python3
"""
Sections 6, 7 and 13 -- analysis of the unified held-out 400-request audit.

Section 6  per generator (Llama-3.2-3B original protocol; Llama-3.1-8B-Instruct
           with its official chat template, forced and abstention-capable),
           per arm (fresh reference / FreshCache / SemanticTTL kappa=1/16 /
           FreshCache-L1Only / ExactTTL): accuracy, abstention, wrong answers,
           WAI, conditional WAI, the full correct/wrong/abstain transition
           matrix against the fresh reference, and exact McNemar.
           Llama-3.1-8B is NEVER its own judge: its primary label is the
           Qwen2.5-7B + Mistral-7B agreement.

Section 7  judge reliability: each of the four judges separately, majority
           vote, unanimous vote, Fleiss kappa, all pairwise Cohen kappas,
           percentage unanimous and the disagreement structure. Disagreement is
           reported as its own category and is never silently mapped to
           CORRECT or INCORRECT.

Section 13 fresh-reference quality diagnosis: every fresh-path request is
           placed in exactly one failure category, and WAI is reported on all
           400, on the fresh-correct subset, on the evidence-valid subset and
           on their intersection. WAI is never printed without its denominator.
"""
from __future__ import annotations
import csv, hashlib, itertools, json, math, pathlib, re, sys
from collections import Counter, defaultdict
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]
ARMS = [("fresh", "Fresh reference"), ("fc", "FreshCache"),
        ("sttl", "SemanticTTL k=1/16"), ("l1only", "FreshCache-L1Only"),
        ("exactttl", "ExactTTL")]
GENS = ["llama3b", "llama8b_forced", "llama8b_abstain"]
JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
# a generator is never its own judge
PRIMARY_JUDGES = {"llama3b": ["llama8b", "qwen7b"],
                  "llama8b_forced": ["qwen7b", "mistral7b"],
                  "llama8b_abstain": ["qwen7b", "mistral7b"]}


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    t = (t or "").strip()
    if t == ABSTAIN_TOKEN:
        return True
    tl = t.lower()
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


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    labs = set(x for x, _ in pairs) | set(y for _, y in pairs)
    pe = sum((sum(1 for x, _ in pairs if x == l) / n) *
             (sum(1 for _, y in pairs if y == l) / n) for l in labs)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def fleiss_kappa(rows):
    """rows: list of label lists, one per item, all the same length."""
    rows = [r for r in rows if all(x is not None for x in r)]
    if not rows:
        return None
    n = len(rows[0])
    labs = sorted({x for r in rows for x in r})
    N = len(rows)
    P = []
    for r in rows:
        c = Counter(r)
        P.append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
    pbar = sum(P) / N
    pj = [sum(Counter(r)[l] for r in rows) / (N * n) for l in labs]
    pe = sum(p * p for p in pj)
    return None if pe == 1 else round((pbar - pe) / (1 - pe), 4)


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    sample = json.load(open(HERE / "sample.json", encoding="utf-8"))
    reqs = sample["requests"]
    say("V3 SECTIONS 6, 7, 13 -- unified held-out 400-request audit")
    say(f"  requests {len(reqs)}   arms {[a for a, _ in ARMS]}")
    say(f"  composition {sample['composition']}")

    # ---------- evidence validity, from the gold-verification study ----------
    evalid = {}
    p = ROOT / "validation" / "gold_verification_400" / "combined_request_status.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            evalid[r["query_id"]] = (r.get("in_evidence_valid_subset") == "1")
    cov = sum(1 for d in reqs if d["query_id"] in evalid)
    say(f"  evidence-validity labels available for {cov}/{len(reqs)} requests "
        f"(source: validation/gold_verification_400/combined_request_status.csv)")

    # ---------- per generator ----------
    out = {"generators": {}, "judges": {}}
    for gen in GENS:
        af = HERE / f"answers_{gen}.jsonl"
        if not af.exists():
            say(f"\n  ==== generator {gen}: NOT RUN ====")
            continue
        ans = {tuple(r["key"]): r["answer"] for r in jl(af)}
        jud = {}
        for j in JUDGES:
            f = HERE / f"judge_{gen}__{j}.jsonl"
            if f.exists():
                jud[j] = {tuple(r["sig"]): r["label"] for r in jl(f)}
        say(f"\n  ==== generator {gen} ====")
        say(f"    answers {len(ans):,}   judges present {sorted(jud)}")
        prim = [j for j in PRIMARY_JUDGES[gen] if j in jud]
        say(f"    PRIMARY judge pair (never the generator itself): {prim}")
        if len(prim) < 2:
            say("    insufficient judges for a primary label; skipping")
            continue

        def label(d, arm):
            a = ans.get(tuple(d[f"{arm}_key"]))
            if a is None:
                return None, None
            if is_abstain(a):
                return "ABSTAIN", a
            sig = (h(d["query"]), h(d["gold"]), h(a))
            ls = [jud[j].get(sig) for j in prim]
            if any(x is None or x == "UNPARSED" for x in ls):
                return "UNJUDGED", a
            if ls[0] != ls[1]:
                return "JUDGE_DISAGREEMENT", a
            return ("CORRECT" if ls[0] == "CORRECT" else "WRONG"), a

        lab = {arm: {} for arm, _ in ARMS}
        for d in reqs:
            for arm, _ in ARMS:
                lab[arm][d["query_id"]] = label(d, arm)[0]

        say(f"\n    {'arm':<22}{'n':>5}{'CORRECT':>9}{'WRONG':>8}"
            f"{'ABSTAIN':>9}{'DISAGREE':>10}{'acc% (determinate)':>21}")
        garm = {}
        for arm, nm in ARMS:
            c = Counter(lab[arm].values())
            n = sum(v for k, v in c.items() if k is not None)
            det = c["CORRECT"] + c["WRONG"]
            acc = 100 * c["CORRECT"] / det if det else None
            garm[arm] = {"n": n, "correct": c["CORRECT"], "wrong": c["WRONG"],
                         "abstain": c["ABSTAIN"],
                         "judge_disagreement": c["JUDGE_DISAGREEMENT"],
                         "unjudged": c["UNJUDGED"], "determinate": det,
                         "accuracy_pct_determinate": (round(acc, 4) if acc is not None else None),
                         "accuracy_ci": wilson(c["CORRECT"], det) if det else None,
                         "accuracy_pct_all400": round(100*c["CORRECT"]/len(reqs), 4)}
            say(f"    {nm:<22}{n:>5}{c['CORRECT']:>9}{c['WRONG']:>8}"
                f"{c['ABSTAIN']:>9}{c['JUDGE_DISAGREEMENT']:>10}"
                + (f"{acc:>19.2f}%" if acc is not None else f"{'-':>20}"))

        # transition matrices + WAI, against the fresh reference
        say(f"\n    full transition matrix vs the fresh reference "
            f"(rows = fresh, cols = cached arm)")
        cats = ["CORRECT", "WRONG", "ABSTAIN", "JUDGE_DISAGREEMENT", "UNJUDGED"]
        for arm, nm in ARMS[1:]:
            M = Counter((lab["fresh"][q], lab[arm][q]) for q in lab["fresh"])
            say(f"      -- {nm} --")
            say("         " + "".join(f"{c[:9]:>11}" for c in cats))
            for r in cats:
                say(f"      {r[:8]:<8}" + "".join(f"{M[(r, c)]:>11,}" for c in cats))
            # WAI, project convention: cached WRONG and fresh CORRECT
            fresh_correct = [q for q in lab["fresh"] if lab["fresh"][q] == "CORRECT"]
            wai = [q for q in fresh_correct if lab[arm][q] == "WRONG"]
            ev = [q for q in lab["fresh"] if evalid.get(q)]
            evc = [q for q in ev if lab["fresh"][q] == "CORRECT"]
            def blk(name, pool):
                k = sum(1 for q in pool if lab["fresh"][q] == "CORRECT"
                        and lab[arm][q] == "WRONG")
                n = len(pool)
                ci = wilson(k, n)
                say(f"         WAI on {name:<28} {k:>4}/{n:<4} = "
                    f"{(100*k/n if n else 0):>6.2f}%  CI{ci}")
                return {"k": k, "n": n, "pct": round(100*k/n, 4) if n else None,
                        "ci": ci}
            wb = {"all_400": blk("all 400 requests", list(lab["fresh"])),
                  "fresh_correct": blk("the fresh-CORRECT subset", fresh_correct),
                  "evidence_valid": blk("the evidence-valid subset", ev),
                  "evidence_valid_and_fresh_correct":
                      blk("evidence-valid AND fresh-CORRECT", evc)}
            b = sum(1 for q in lab["fresh"] if lab["fresh"][q] == "CORRECT"
                    and lab[arm][q] == "WRONG")
            c_ = sum(1 for q in lab["fresh"] if lab["fresh"][q] == "WRONG"
                     and lab[arm][q] == "CORRECT")
            pm = mcnemar_exact(b, c_)
            say(f"         exact McNemar vs fresh: C->W {b}  W->C {c_}  "
                f"p = {pm:.4g}" + ("   (not significant; this is NOT an "
                                   "equivalence claim -- no non-inferiority "
                                   "margin was pre-specified)" if pm > 0.05 else ""))
            garm[arm]["transition_matrix"] = {f"{r}|{c}": M[(r, c)]
                                              for r in cats for c in cats}
            garm[arm]["wai"] = wb
            garm[arm]["mcnemar_vs_fresh"] = {"b": b, "c": c_, "p": pm}
        out["generators"][gen] = garm

        # ---------- section 13: fresh-path failure categories ----------
        if gen in ("llama3b", "llama8b_forced"):
            say(f"\n    section 13 -- fresh-path failure categories ({gen})")
            cat = Counter()
            rowsout = []
            for d in reqs:
                q = d["query_id"]
                L = lab["fresh"][q]
                gold_toks = set(re.findall(r"\w+", (d["gold"] or "").lower()))
                grounded = None
                trunc = (d.get("fresh_pages", 0) or 0) * 700 > sample["max_context_chars"]
                if L == "CORRECT":
                    k = "correct"
                elif L == "JUDGE_DISAGREEMENT":
                    k = "judge_disagreement"
                elif L == "ABSTAIN":
                    k = "abstention"
                elif not evalid.get(q, True):
                    k = "temporally_invalid_or_unverifiable_reference"
                else:
                    k = ("context_truncation" if trunc
                         else "retrieval_lacks_support_or_generator_failure")
                cat[k] += 1
                rowsout.append({"query_id": q, "category": k,
                                "fresh_label": L, "fc": d["fc"],
                                "evidence_valid": evalid.get(q),
                                "fresh_pages": d.get("fresh_pages")})
            for k, v in cat.most_common():
                say(f"      {k:<52}{v:>5}  {100*v/len(reqs):>6.2f}%")
            out.setdefault("fresh_failure_categories", {})[gen] = dict(cat)
            with open(HERE / f"fresh_failure_categories_{gen}.csv", "w",
                      newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rowsout[0]))
                w.writeheader(); w.writerows(rowsout)

        # ---------- section 7: judge reliability ----------
        if len(jud) >= 2:
            say(f"\n    section 7 -- judge reliability ({gen}), "
                f"{len(jud)} judges: {sorted(jud)}")
            sigs = []
            for d in reqs:
                for arm, _ in ARMS:
                    a = ans.get(tuple(d[f"{arm}_key"]))
                    if a is None or is_abstain(a):
                        continue
                    sigs.append((h(d["query"]), h(d["gold"]), h(a)))
            sigs = sorted(set(sigs))
            js = sorted(jud)
            mat = [[jud[j].get(s) for j in js] for s in sigs]
            mat = [[None if x in (None, "UNPARSED") else x for x in r] for r in mat]
            full = [r for r in mat if all(x is not None for x in r)]
            say(f"      judged items {len(sigs):,}; complete rows {len(full):,}")
            for i, j in enumerate(js):
                col = [r[i] for r in mat if r[i] is not None]
                k = sum(1 for x in col if x == "CORRECT")
                say(f"      {j:<11} CORRECT {k:>5}/{len(col):<5} = "
                    f"{100*k/len(col):>6.2f}%")
            if full:
                maj = sum(1 for r in full
                          if sum(1 for x in r if x == "CORRECT") * 2 > len(r))
                una = sum(1 for r in full if len(set(r)) == 1)
                unac = sum(1 for r in full if set(r) == {"CORRECT"})
                say(f"      majority CORRECT      {maj:>5}/{len(full)} = "
                    f"{100*maj/len(full):.2f}%")
                say(f"      unanimous (any label) {una:>5}/{len(full)} = "
                    f"{100*una/len(full):.2f}%")
                say(f"      unanimous CORRECT     {unac:>5}/{len(full)} = "
                    f"{100*unac/len(full):.2f}%")
                say(f"      NON-unanimous         {len(full)-una:>5}/{len(full)}"
                    f" = {100*(len(full)-una)/len(full):.2f}%  "
                    f"-- reported as its own category, not mapped to either label")
                fk = fleiss_kappa(full)
                say(f"      Fleiss kappa ({len(js)} judges) = {fk}")
                say(f"      pairwise Cohen kappa")
                pk = {}
                for a, b in itertools.combinations(range(len(js)), 2):
                    k = cohen_kappa([r[a] for r in mat], [r[b] for r in mat])
                    pk[f"{js[a]}|{js[b]}"] = k
                    say(f"        {js[a]:<10} vs {js[b]:<10} kappa = {k}")
                ds = Counter(tuple(sorted(Counter(r).items())) for r in full
                             if len(set(r)) > 1)
                say(f"      disagreement structure (label multiset -> count)")
                for k, v in ds.most_common():
                    say(f"        {dict(k)} -> {v}")
                out["judges"][gen] = {
                    "judges": js, "items": len(sigs), "complete_rows": len(full),
                    "per_judge_correct_pct": {
                        j: round(100*sum(1 for r in mat if r[i] == "CORRECT")
                                 / max(1, sum(1 for r in mat if r[i] is not None)), 4)
                        for i, j in enumerate(js)},
                    "majority_correct": maj, "unanimous_any": una,
                    "unanimous_correct": unac,
                    "non_unanimous": len(full) - una,
                    "fleiss_kappa": fk, "pairwise_cohen_kappa": pk,
                    "disagreement_structure": {str(dict(k)): v
                                               for k, v in ds.items()}}

    json.dump(out, open(HERE / "audit_analysis.json", "w"), indent=2, default=str)
    (HERE / "analyze.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("\n  wrote audit_analysis.json, fresh_failure_categories_*.csv, analyze.log")


if __name__ == "__main__":
    main()
