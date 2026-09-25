#!/usr/bin/env python3
"""
SECTION 1 -- verify the manuscript's audit numbers and the fresh-path
diagnosis against the actual saved artifacts. Nothing is assumed correct
because it appears in the manuscript.

Read-only on every existing artifact.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
SRE = HERE.parent
V3 = SRE.parent
ROOT = V3.parent
HO = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"
PR = ROOT / "validation" / "mixed_age_full_policy_audit"
RCI = V3 / "remaining_critical_issues" / "06_stronger_generator"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]
CLAIM = {
    "primary": {"fresh": 16.50, "fc": 15.25, "fc_wai": 1.50,
                "sttl": 2.25, "sttl_wai": 14.50},
    "heldout": {"fresh": 20.25, "fc": 18.75, "fc_wai": 1.75,
                "sttl": 16.00, "sttl_wai": 6.25},
}
DIAG = {"primary": {"missing_support": 171, "unverifiable": 99,
                    "generator_failure": 63, "success": 66},
        "heldout": {"missing_support": 166, "unverifiable": 86,
                    "generator_failure": 64, "success": 81}}


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    tl = (t or "").strip().lower()
    return any(x in tl for x in ABSTAIN_PAT)


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("SECTION 1 -- reproduce the original weak-reference results")
    OUT = {}

    # ---------------- held-out audit ----------------
    say(f"\n  == HELD-OUT 400-request audit ==")
    say(f"    dir {HO.relative_to(ROOT)}")
    S = json.load(open(HO / "sample.json", encoding="utf-8"))
    reqs = S["requests"]
    say(f"    requests {len(reqs)}   sample.json sha {sha(HO/'sample.json')[:16]}")
    ans = {tuple(r["key"]): r["answer"] for r in jl(HO / "answers.jsonl")}
    jud = {j: {tuple(r["sig"]): r["label"]
               for r in jl(HO / f"judge_{j}.jsonl")}
           for j in ("llama8b", "qwen7b")}
    say(f"    answers {len(ans):,}; judges "
        f"{ {j: len(v) for j, v in jud.items()} }")

    def lab(d, arm):
        a = ans.get(tuple(d[f"{arm}_key"]))
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h16(d["query"]), h16(d["gold"]), h16(a))
        l1, l2 = jud["llama8b"].get(sig), jud["qwen7b"].get(sig)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    rows = []
    for d in reqs:
        rows.append({"query_id": d["query_id"], "fc": d["fc"],
                     "cluster_id": d["cluster_id"],
                     "fresh": lab(d, "fresh"), "fc_arm": lab(d, "fc"),
                     "sttl": lab(d, "sttl"),
                     "fc_tier": d["fc_tier"], "sttl_tier": d["sttl_tier"],
                     "fresh_pages": d.get("fresh_pages"),
                     "gold": d["gold"], "query": d["query"]})
    n = len(rows)
    fr = sum(1 for r in rows if r["fresh"] == "CORRECT")
    fcc = sum(1 for r in rows if r["fc_arm"] == "CORRECT")
    stc = sum(1 for r in rows if r["sttl"] == "CORRECT")
    fw = sum(1 for r in rows if r["fc_arm"] == "WRONG" and r["fresh"] == "CORRECT")
    sw = sum(1 for r in rows if r["sttl"] == "WRONG" and r["fresh"] == "CORRECT")
    say(f"\n    {'quantity':<34}{'manuscript':>12}{'recomputed':>12}{'match':>8}")
    def chk(nm, claim, got):
        ok = abs(claim - got) < 0.02
        say(f"    {nm:<34}{claim:>11.2f}%{got:>11.2f}%{('YES' if ok else 'NO'):>8}")
        return ok
    a1 = chk("fresh reference accuracy", CLAIM["heldout"]["fresh"], 100*fr/n)
    a2 = chk("FreshCache accuracy", CLAIM["heldout"]["fc"], 100*fcc/n)
    a3 = chk("FreshCache WAI", CLAIM["heldout"]["fc_wai"], 100*fw/n)
    a4 = chk("SemanticTTL accuracy", CLAIM["heldout"]["sttl"], 100*stc/n)
    a5 = chk("SemanticTTL WAI", CLAIM["heldout"]["sttl_wai"], 100*sw/n)
    say(f"    counts: fresh {fr}/{n}, FreshCache {fcc}/{n} (WAI {fw}), "
        f"SemanticTTL {stc}/{n} (WAI {sw})")
    say(f"    NOTE the manuscript's held-out SemanticTTL row is the "
        f"theta=0.60 kappa=1/2 quality-aware point; this audit stores the "
        f"theta=0.40 kappa=1/16 drift-selected point. If the numbers differ, "
        f"they are DIFFERENT OPERATING POINTS, not a failed reproduction.")
    OUT["heldout"] = {"n": n, "fresh_correct": fr, "fc_correct": fcc,
                      "sttl_correct": stc, "fc_wai": fw, "sttl_wai": sw,
                      "matches": {"fresh": a1, "fc": a2, "fc_wai": a3,
                                  "sttl": a4, "sttl_wai": a5}}
    say(f"\n    composition: FreshCache {Counter(r['fc_tier'] for r in rows)}")
    say(f"    composition: SemanticTTL {Counter(r['sttl_tier'] for r in rows)}")
    say(f"    freshness classes {dict(Counter(r['fc'] for r in rows))}")
    with open(HERE / "heldout_per_request.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---------------- primary audit ----------------
    say(f"\n  == PRIMARY (mixed-age) 400-request audit ==")
    pj = json.load(open(PR / "corrected_results.json", encoding="utf-8"))
    g = pj["gate_on"]
    say(f"    fresh {g['fresh_correct']}/400 = {g['fresh_correct_pct']}%  "
        f"(manuscript {CLAIM['primary']['fresh']}%)  "
        f"match {abs(g['fresh_correct_pct']-CLAIM['primary']['fresh'])<0.02}")
    say(f"    FreshCache {g['correct']}/400 = {g['correct_pct']}%  "
        f"(manuscript {CLAIM['primary']['fc']}%)  "
        f"match {abs(g['correct_pct']-CLAIM['primary']['fc'])<0.02}")
    say(f"    FreshCache WAI {g['wai']}/400 = {g['wai_pct']}%  "
        f"(manuscript {CLAIM['primary']['fc_wai']}%)  "
        f"match {abs(g['wai_pct']-CLAIM['primary']['fc_wai'])<0.02}")
    say(f"    this file records gate_on / gate_off, NOT SemanticTTL; the "
        f"manuscript's primary SemanticTTL row (2.25% / 14.50%) comes from a "
        f"different artifact and is located separately below")
    OUT["primary"] = {"fresh_correct": g["fresh_correct"],
                      "fc_correct": g["correct"], "fc_wai": g["wai"],
                      "gate_off_wai": pj["gate_off"]["wai"]}
    for cand in ("all_reuse_answer_results.json",):
        p = ROOT / "validation" / "mixed_age_answer_quality" / cand
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            say(f"    {cand}: top-level keys {list(d)[:8]}")

    # ---------------- fresh-path diagnosis ----------------
    say(f"\n  == manuscript fresh-path diagnosis, checked for internal "
        f"consistency ==")
    for k, v in DIAG.items():
        tot = sum(v.values())
        say(f"    {k:<9} {v}  sum = {tot}  "
            + ("consistent with 400" if tot == 400
               else f"**DOES NOT SUM TO 400 (short by {400-tot})**"))
    say(f"    the held-out success count {DIAG['heldout']['success']} matches "
        f"the recomputed fresh-correct {fr}: {DIAG['heldout']['success']==fr}")
    say(f"    the primary success count {DIAG['primary']['success']} matches "
        f"the recomputed fresh-correct {g['fresh_correct']}: "
        f"{DIAG['primary']['success']==g['fresh_correct']}")
    fcat = HERE.parent.parent / "remaining_critical_issues" / "06_stronger_generator"
    for f in sorted(fcat.glob("fresh_failure_categories_*.csv")):
        c = Counter(r["category"] for r in csv.DictReader(open(f, encoding="utf-8")))
        say(f"    existing V3 categorisation {f.name}: {dict(c)} "
            f"(sum {sum(c.values())})")
    say(f"    -> the manuscript's four-way split and the V3 categorisation use "
        f"DIFFERENT category definitions; section 3 rebuilds the diagnosis "
        f"from evidence rather than adopting either.")

    # ---------------- stronger-generator artifacts ----------------
    say(f"\n  == existing stronger-generator artifacts (to be REUSED) ==")
    for f in ("answers_llama3b.jsonl", "answers_llama8b_forced.jsonl",
              "answers_llama8b_abstain.jsonl", "sample.json"):
        p = RCI / f
        say(f"    {'PRESENT' if p.exists() else 'MISSING':<8}{f:<34}"
            + (f"{len(jl(p)):>6,} rows  sha {sha(p)[:16]}" if p.exists()
               and f.endswith('jsonl') else
               (f"  sha {sha(p)[:16]}" if p.exists() else "")))
    for j in ("llama3b", "llama8b", "qwen7b", "mistral7b"):
        for gname in ("llama3b", "llama8b_forced", "llama8b_abstain"):
            p = RCI / f"judge_{gname}__{j}.jsonl"
            if p.exists():
                say(f"    judge {gname:<16} x {j:<10} {len(jl(p)):>5,} rows")
    A = json.load(open(RCI / "sample.json", encoding="utf-8"))
    same = [d["query_id"] for d in A["requests"]] == [d["query_id"] for d in reqs]
    say(f"    the 8B experiment uses the SAME 400 held-out request IDs in the "
        f"same order: {same}")
    OUT["reuse"] = {"same_request_ids": same,
                    "generators": ["llama3b", "llama8b_forced",
                                   "llama8b_abstain"],
                    "judges_present": sorted({p.name.split('__')[1][:-6]
                                              for p in RCI.glob('judge_*__*.jsonl')})}

    json.dump(OUT, open(HERE / "reproduction.json", "w"), indent=2)
    (SRE / "logs").mkdir(exist_ok=True)
    (SRE / "logs" / "s1_reproduce.log").write_text("\n".join(log) + "\n",
                                                   encoding="utf-8")
    say(f"\n  wrote reproduction.json, heldout_per_request.csv")


if __name__ == "__main__":
    main()
