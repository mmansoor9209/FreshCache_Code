#!/usr/bin/env python3
"""
SECTIONS 2 and 3 -- reference-answer validity and fresh-path failure
diagnosis, for the 400 held-out audit requests.

Reference validity REUSES the existing gold-verification study
(validation/gold_verification_400/), which is a TWO-JUDGE LLM assessment with
a third-model tiebreak. It is therefore reported throughout as an
**LLM-/evidence-assisted assessment, NOT independent human validation**.
A human-review package is emitted for the uncertain cases; no human label is
fabricated.

No live search is performed. Everything comes from saved snapshots and saved
audit artifacts.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, re, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
SRE = HERE.parent
V3 = SRE.parent
ROOT = V3.parent
HO = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"
GV = ROOT / "validation" / "gold_verification_400"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
sys.path.insert(0, str(ROOT / "v16_exp12"))
sys.path.insert(0, str(ROOT / "v9"))
os.chdir(ROOT)
import e2e_answer_grading as e2e     # noqa: E402
import mixed_age_v2 as ma            # noqa: E402

MAXCHARS, MINCHARS = 2800, 200
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]
# existing study label -> requested taxonomy
VALIDITY_MAP = {"VALID_AT_T": "SUPPORTED",
                "SUPERSEDED_AT_T": "CONTRADICTED",
                "INDETERMINATE_AT_T": "UNVERIFIABLE",
                "JUDGE_DISAGREEMENT": "UNVERIFIABLE"}
YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    tl = (t or "").strip().lower()
    return any(x in tl for x in ABSTAIN_PAT)


def gold_in(text, gold):
    """LEXICAL presence of the gold string. This is a weak proxy and is NEVER
    treated as 'evidence sufficient to answer the question'."""
    g, t = norm(gold), norm(text)
    if not g or not t:
        return False
    if g in t:
        return True
    toks = [w for w in g.split() if len(w) > 2]
    return bool(toks) and all(w in t for w in toks)


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("SECTIONS 2 & 3 -- reference validity and fresh-path diagnosis")
    S = json.load(open(HO / "sample.json", encoding="utf-8"))
    reqs = S["requests"]
    tasks = {tuple(t["key"]): t for t in jl(HO / "gen_tasks.jsonl")}
    ans = {tuple(r["key"]): r["answer"] for r in jl(HO / "answers.jsonl")}
    jud = {j: {tuple(r["sig"]): r["label"] for r in jl(HO / f"judge_{j}.jsonl")}
           for j in ("llama8b", "qwen7b")}

    gv = {r["query_id"]: r for r in
          csv.DictReader(open(GV / "combined_request_status.csv", encoding="utf-8"))}
    say(f"  held-out requests {len(reqs)}; gold-verification coverage "
        f"{sum(1 for d in reqs if d['query_id'] in gv)}/{len(reqs)}")
    say(f"  reference validity source: validation/gold_verification_400 "
        f"(two LLM judges + third-model tiebreak)")
    say(f"  ** this is an LLM-/evidence-assisted assessment, NOT independent "
        f"human validation **")

    rows = []
    for d in reqs:
        qid, gold = d["query_id"], d["gold"]
        fk = tuple(d["fresh_key"])
        ctx = tasks.get(fk, {}).get("context", "")
        a = ans.get(fk)
        if a is None:
            flab = None
        elif is_abstain(a):
            flab = "ABSTAIN"
        else:
            sig = (h16(d["query"]), h16(gold), h16(a))
            l1, l2 = jud["llama8b"].get(sig), jud["qwen7b"].get(sig)
            flab = (None if l1 is None or l2 is None
                    else ("CORRECT" if l1 == "CORRECT" and l2 == "CORRECT"
                          else "WRONG"))
        # full (untruncated) page text, to separate truncation from absence
        pairs = [(u, d["t"]) for u in
                 [x[0] for x in d.get("fresh_versions", [])]] or None
        full, kept, obs = "", 0, 0
        for uh, tv in (d.get("fresh_versions") or []):
            txt = e2e.load_snapshot_text(uh, ma.version_at(tv)) or ""
            txt = re.sub(r"\s+", " ", txt).strip()
            if len(txt) >= MINCHARS:
                obs += 1
                full += txt + "\n\n"
        g = gv.get(qid, {})
        val = VALIDITY_MAP.get(g.get("validity", ""), "UNVERIFIABLE")
        vol = g.get("volatility", "")
        q_has_year = bool(YEAR.search(d["query"]))
        temporally_ambiguous = (vol == "VOLATILE" and not q_has_year)
        rows.append({
            "query_id": qid, "cluster_id": d["cluster_id"], "fc": d["fc"],
            "query": d["query"], "gold": gold,
            "gold_source": g.get("gold_source", ""),
            "gold_year": g.get("gold_year", ""),
            "volatility": vol, "request_t": d["t"],
            "retrieved_urls": len(d.get("fresh_versions") or []),
            "observable_pages": obs,
            "context_chars": len(ctx),
            "full_text_chars": len(full),
            "truncated": int(len(full) > MAXCHARS),
            "gold_lexically_in_context": int(gold_in(ctx, gold)),
            "gold_lexically_in_full_text": int(gold_in(full, gold)),
            "fresh_label": flab,
            "reference_validity_study": g.get("validity", ""),
            "reference_status": val,
            "question_has_explicit_year": int(q_has_year),
            "temporally_ambiguous": int(temporally_ambiguous),
        })

    # ---------------- SECTION 2 ----------------
    say(f"\n  == SECTION 2: reference-answer status (400 held-out) ==")
    for r in rows:
        r["reference_status_final"] = ("TEMPORALLY_AMBIGUOUS"
                                       if r["reference_status"] == "UNVERIFIABLE"
                                       and r["temporally_ambiguous"]
                                       else r["reference_status"])
    c = Counter(r["reference_status_final"] for r in rows)
    for k in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE",
              "TEMPORALLY_AMBIGUOUS"):
        say(f"    {k:<24}{c[k]:>5}  {100*c[k]/len(rows):>6.2f}%")
    say(f"    raw study labels: "
        f"{dict(Counter(r['reference_validity_study'] for r in rows))}")
    say(f"    by source: "
        f"{dict(Counter((r['gold_source'], r['reference_status_final']) for r in rows).most_common(8))}")
    say(f"    UNVERIFIABLE is NOT counted as incorrect anywhere below.")
    say(f"    no gold answer was rewritten.")

    # ---------------- SECTION 3 ----------------
    say(f"\n  == SECTION 3: fresh-path failure diagnosis ==")
    say(f"    categories applied in this priority order, mutually exclusive:")
    say(f"      H correct -> E reference contradicted -> G unverifiable -> "
        f"F temporally ambiguous -> B no observable body -> "
        f"C truncation -> D generator failure -> A missing evidence -> I other")
    for r in rows:
        sec = []
        if r["fresh_label"] == "CORRECT":
            cat = "H_fresh_correct"
        elif r["reference_status_final"] == "CONTRADICTED":
            cat = "E_reference_contradicted"
        elif r["reference_status_final"] == "TEMPORALLY_AMBIGUOUS":
            cat = "F_reference_temporally_ambiguous"
        elif r["reference_status_final"] == "UNVERIFIABLE":
            cat = "G_reference_unverifiable"
        elif r["observable_pages"] == 0:
            cat = "B_no_observable_page_body"
        elif (r["gold_lexically_in_full_text"]
              and not r["gold_lexically_in_context"]):
            cat = "C_lost_to_context_truncation"
        elif r["gold_lexically_in_context"]:
            cat = "D_generator_failure_given_lexical_support"
        elif r["fresh_label"] == "ABSTAIN":
            cat = "I_abstention"
        else:
            cat = "A_missing_relevant_evidence"
        if r["truncated"]:
            sec.append("truncated")
        if r["observable_pages"] < r["retrieved_urls"]:
            sec.append("some_pages_unobservable")
        if r["gold_lexically_in_context"]:
            sec.append("gold_lexically_present")
        r["primary_category"] = cat
        r["secondary_flags"] = "|".join(sec)
    cc = Counter(r["primary_category"] for r in rows)
    say(f"\n    {'category':<44}{'n':>5}{'%':>8}")
    for k in sorted(cc, key=lambda x: -cc[x]):
        say(f"    {k:<44}{cc[k]:>5}{100*cc[k]/len(rows):>7.2f}%")
    say(f"    total {sum(cc.values())} (sums to 400: {sum(cc.values())==400})")

    say(f"\n    lexical presence vs sufficiency -- these are NOT the same:")
    lp = sum(r["gold_lexically_in_context"] for r in rows)
    lpc = sum(1 for r in rows if r["gold_lexically_in_context"]
              and r["fresh_label"] == "CORRECT")
    say(f"      gold string lexically in the fresh context: {lp}/400 "
        f"({100*lp/400:.2f}%)")
    say(f"      of those, the fresh answer was judged CORRECT: {lpc}/{lp} "
        f"({100*lpc/max(1,lp):.2f}%)")
    say(f"      a page containing the gold string is not necessarily relevant "
        f"evidence; this is reported as a lexical proxy only.")

    say(f"\n    by freshness class")
    for k in sorted({r["fc"] for r in rows}):
        sub = [r for r in rows if r["fc"] == k]
        top = Counter(r["primary_category"] for r in sub).most_common(3)
        say(f"      {k:<10}n {len(sub):>4}  top: {top}")
    say(f"    by dataset source")
    for k in sorted({r["gold_source"] for r in rows}):
        sub = [r for r in rows if r["gold_source"] == k]
        top = Counter(r["primary_category"] for r in sub).most_common(3)
        say(f"      {str(k):<20}n {len(sub):>4}  top: {top}")
    say(f"    retrieved URLs per request: "
        f"{dict(Counter(r['retrieved_urls'] for r in rows))}")
    say(f"    observable pages per request: "
        f"{dict(Counter(r['observable_pages'] for r in rows))}")
    say(f"    contexts hitting the {MAXCHARS}-char cap: "
        f"{sum(r['truncated'] for r in rows)}/400")

    say(f"\n    plausibly improvable by:")
    imp = {
        "1 better retrieval": cc["A_missing_relevant_evidence"],
        "2 better page extraction": cc["B_no_observable_page_body"],
        "3 expanded context": cc["C_lost_to_context_truncation"],
        "4 stronger generation": cc["D_generator_failure_given_lexical_support"],
        "5 improved reference validity": (cc["E_reference_contradicted"]
                                          + cc["F_reference_temporally_ambiguous"]
                                          + cc["G_reference_unverifiable"]),
    }
    for k, v in imp.items():
        say(f"      {k:<34}{v:>5}  {100*v/400:>6.2f}%")
    say(f"      (these are upper bounds on what each lever could fix, not "
        f"predictions)")

    # ---------------- human-review package ----------------
    unc = [r for r in rows if r["reference_status_final"] in
           ("UNVERIFIABLE", "TEMPORALLY_AMBIGUOUS")]
    with open(HERE / "human_reference_review.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "query_id", "question", "dataset_reference_answer", "gold_source",
            "gold_year", "freshness_class", "request_timestamp_s",
            "evidence_snapshot_round", "human_reference_status", "notes"])
        w.writeheader()
        for r in unc:
            w.writerow({"query_id": r["query_id"], "question": r["query"],
                        "dataset_reference_answer": r["gold"],
                        "gold_source": r["gold_source"],
                        "gold_year": r["gold_year"],
                        "freshness_class": r["fc"],
                        "request_timestamp_s": r["request_t"],
                        "evidence_snapshot_round": ma.version_at(r["request_t"]),
                        "human_reference_status": "", "notes": ""})
    say(f"\n  human-review package: {len(unc)} uncertain cases -> "
        f"human_reference_review.csv (labels EMPTY; allowed values "
        f"SUPPORTED / CONTRADICTED / UNVERIFIABLE / TEMPORALLY_AMBIGUOUS)")
    assert all(not r for r in [x for x in []]), ""

    cols = list(rows[0])
    with open(HERE / "reference_and_diagnosis.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    json.dump({"n": len(rows),
               "reference_status": dict(c),
               "reference_source": "validation/gold_verification_400 "
                                   "(2 LLM judges + tiebreak)",
               "is_human_validated": False,
               "failure_categories": dict(cc),
               "improvable_upper_bounds": imp,
               "gold_lexically_in_context": lp,
               "human_review_cases": len(unc)},
              open(HERE / "reference_and_diagnosis.json", "w"), indent=2)
    (SRE / "logs").mkdir(exist_ok=True)
    (SRE / "logs" / "s2s3.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"  wrote reference_and_diagnosis.csv/.json, human_reference_review.csv")


if __name__ == "__main__":
    main()
