#!/usr/bin/env python3
"""
validation/gold_verification_400/stage2_volatility.py

STAGE 2 — GOLD TEMPORAL VOLATILITY from the ORIGINAL cached source datasets.
Offline, no LLM, no GPU, no web/API call.

For each of the 703 distinct base queries behind the 715 Stage 1 verification
units, recover the source dataset's own temporal annotation and decide whether
the benchmark answer CHANGES across annotated years:

  STABLE          same answer set in every annotated year (>= 2 years)
  VOLATILE        answer set differs between annotated years
  SINGLE_YEAR     only one annotated year -- volatility UNDETERMINED
  NOT_TIME_KEYED  no year-keyed annotation (TriviaQA)
  NO_ANNOTATION   the question could not be matched in any source dataset

Sources, read-only from the local HuggingFace cache:
  ROIM/temporal-alignment-qa [test]  year -> answer-list dictionary
  natyou/freshqa_10_06 [test]        effective_year, next_review, answer_0..9
  mandarjoshi/trivia_qa              not time-keyed; classified as such by
                                     gold_source, never loaded

IMPORTANT INTERPRETATION LIMIT, enforced in the output schema:
  STABLE means "unchanged across the years the benchmark annotated". The latest
  annotated year is 2023 for almost every row while the corpus was collected in
  2026, so STABLE is NOT evidence that the answer was still correct in 2026. The
  field is named `stable_across_annotated_years` for that reason, and every row
  carries `years_annotated`, `latest_annotated_year` and `years_gap_to_corpus`.

READ-ONLY on every pre-existing artifact. All output lands in this directory.

Reproduce:
    <HOME>/miniconda3/envs/graphrag/bin/python \
        validation/gold_verification_400/stage2_volatility.py
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import pathlib
import re
import sys
from collections import Counter

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
os.chdir(ROOT)

CORPUS_YEAR = 2026          # snapshots were collected 2026-05 / 2026-06
norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())    # noqa: E731

LOG = []
ISSUES = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def issue(kind, ident, detail):
    ISSUES.append({"kind": kind, "base_query_id": ident, "detail": detail})


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def answer_set(v):
    """Normalised, order-insensitive answer set for one year."""
    if not isinstance(v, list):
        v = [v]
    return frozenset(norm(str(x)) for x in v if str(x).strip())


def main():
    say("STAGE 2 — gold temporal volatility from original cached annotations")
    say("  offline; no LLM, no GPU, no web calls; source datasets never modified")

    s1 = HERE / "stage1_summary.json"
    if not s1.exists():
        say("  STOP: stage1_summary.json not found — Stage 1 has not run.")
        sys.exit(2)
    st1 = json.load(open(s1, encoding="utf-8"))
    say(f"  Stage 1: {st1['verification_units']} units, "
        f"{st1['base_queries']} base queries, "
        f"{st1['unique_requests']} requests")

    units = list(csv.DictReader(open(HERE / "verification_units.csv",
                                     encoding="utf-8")))
    rmap = list(csv.DictReader(open(HERE / "request_map.csv", encoding="utf-8")))
    assert len(units) == st1["verification_units"]

    # base query -> {text, gold, source, year, request ids, unit ids}
    base = {}
    for r in rmap:
        b = r["base_query_id"]
        e = base.setdefault(b, {"base_query_id": b, "base_query": r["base_query"],
                                "golds": set(), "sources": set(), "years": set(),
                                "requests": [], "aliases": 0})
        e["golds"].add(r["gold"])
        if r["gold_source"]:
            e["sources"].add(r["gold_source"])
        if r["gold_year"]:
            e["years"].add(r["gold_year"])
        e["requests"].append(r["query_id"])
        e["aliases"] = max(e["aliases"], int(r["n_aliases"] or 0))
    say(f"  distinct base queries to classify: {len(base)}")
    assert len(base) == st1["base_queries"]

    # ---- load the original cached annotations, read-only ----
    from datasets import load_dataset
    say("\n  loading ORIGINAL cached source datasets (read-only, offline)")

    taq = {}
    d = load_dataset("ROIM/temporal-alignment-qa", split="test")
    bad_parse = 0
    for row in d:
        a = row["answer"]
        if isinstance(a, str):
            try:
                a = ast.literal_eval(a)
            except Exception:
                bad_parse += 1
                continue
        if not isinstance(a, dict) or not a:
            bad_parse += 1
            continue
        taq[norm(row["question"])] = {str(k): v for k, v in a.items()}
    say(f"    ROIM/temporal-alignment-qa[test]: {len(d):,} rows, "
        f"{len(taq):,} with a parsable year dictionary ({bad_parse} unparsable)")

    fq = {}
    f = load_dataset("natyou/freshqa_10_06", split="test")
    for row in f:
        ans = [row.get(f"answer_{i}") for i in range(10)]
        ans = [a for a in ans if a and str(a).strip() and str(a) != "None"]
        fq[norm(row["question"])] = {
            "effective_year": str(row.get("effective_year", "") or ""),
            "next_review": str(row.get("next_review", "") or ""),
            "false_premise": str(row.get("false_premise", "") or ""),
            "fact_type": str(row.get("fact_type", "") or ""),
            "answers": ans}
    say(f"    natyou/freshqa_10_06[test]: {len(f):,} rows, {len(fq):,} indexed")
    say(f"    mandarjoshi/trivia_qa: not year-keyed by construction; classified "
        f"as NOT_TIME_KEYED from gold_source without loading")

    # ---- classify ----
    rows = []
    for b, e in sorted(base.items()):
        qn = norm(e["base_query"])
        src = "|".join(sorted(e["sources"])) or ""
        gold_year = "|".join(sorted(e["years"]))
        gold = sorted(e["golds"])[0]
        rec = {
            "base_query_id": b, "base_query": e["base_query"],
            "gold_answer": gold, "n_aliases": e["aliases"],
            "gold_source": src, "original_gold_year": gold_year,
            "n_requests": len(e["requests"]),
            "volatility": None, "years_annotated": "", "n_years_annotated": 0,
            "latest_annotated_year": "", "years_gap_to_corpus": "",
            "n_distinct_answer_sets": "", "stable_across_annotated_years": "",
            "gold_equals_latest_annotated": "",
            "gold_differs_from_some_annotated_year": "",
            "answer_at_latest_year": "", "freshqa_next_review": "",
            "freshqa_fact_type": "", "freshqa_false_premise": "",
            "note": "",
        }

        if src == "triviaqa":
            rec["volatility"] = "NOT_TIME_KEYED"
            rec["note"] = "TriviaQA carries no year-keyed annotation"
            rows.append(rec)
            continue

        yd = taq.get(qn)
        if yd:
            years = sorted(yd, key=lambda y: str(y))
            sets = {y: answer_set(yd[y]) for y in years}
            distinct = {frozenset(s) for s in sets.values()}
            latest = years[-1]
            rec.update({
                "years_annotated": ",".join(years),
                "n_years_annotated": len(years),
                "latest_annotated_year": latest,
                "n_distinct_answer_sets": len(distinct),
                "answer_at_latest_year": " | ".join(sorted(sets[latest])),
            })
            try:
                rec["years_gap_to_corpus"] = CORPUS_YEAR - int(latest)
            except ValueError:
                rec["years_gap_to_corpus"] = ""
            gn = norm(gold)
            rec["gold_equals_latest_annotated"] = int(gn in sets[latest])
            rec["gold_differs_from_some_annotated_year"] = int(
                any(gn not in s for s in sets.values()))
            if len(years) < 2:
                rec["volatility"] = "SINGLE_YEAR"
                rec["stable_across_annotated_years"] = ""
                rec["note"] = "one annotated year — volatility UNDETERMINED"
            elif len(distinct) == 1:
                rec["volatility"] = "STABLE"
                rec["stable_across_annotated_years"] = 1
                rec["note"] = ("unchanged across annotated years only; NOT "
                               "evidence of correctness in 2026")
            else:
                rec["volatility"] = "VOLATILE"
                rec["stable_across_annotated_years"] = 0
                rec["note"] = f"{len(distinct)} distinct answer sets across {len(years)} years"
            rows.append(rec)
            continue

        fe = fq.get(qn)
        if fe:
            rec["freshqa_next_review"] = fe["next_review"]
            rec["freshqa_fact_type"] = fe["fact_type"]
            rec["freshqa_false_premise"] = fe["false_premise"]
            rec["latest_annotated_year"] = fe["effective_year"]
            rec["years_annotated"] = fe["effective_year"]
            rec["n_years_annotated"] = 1 if fe["effective_year"] else 0
            try:
                rec["years_gap_to_corpus"] = CORPUS_YEAR - int(fe["effective_year"])
            except ValueError:
                rec["years_gap_to_corpus"] = ""
            rec["volatility"] = "SINGLE_YEAR"
            rec["note"] = ("FreshQA carries a single effective_year, not a year "
                           f"series — volatility UNDETERMINED; next_review="
                           f"{fe['next_review']!r}, fact_type={fe['fact_type']!r}")
            gn = norm(gold)
            rec["gold_equals_latest_annotated"] = int(
                any(gn == norm(a) for a in fe["answers"]))
            if fe["false_premise"].lower() == "true":
                issue("false_premise", b, "FreshQA marks this question as a "
                                          "false premise")
            rows.append(rec)
            continue

        rec["volatility"] = "NO_ANNOTATION"
        rec["note"] = f"not matched in any source dataset (gold_source={src!r})"
        issue("no_source_annotation", b,
              f"gold_source={src!r} but question not found in TemporalAlignQA "
              f"or FreshQA by normalised text")
        rows.append(rec)

    # ---- attach to units ----
    byb = {r["base_query_id"]: r for r in rows}
    urows = []
    for u in units:
        r = byb[u["base_query_id"]]
        urows.append({
            "unit_id": u["unit_id"], "base_query_id": u["base_query_id"],
            "snapshot_round": u["snapshot_round"], "n_requests": u["n_requests"],
            "gold_source": r["gold_source"],
            "original_gold_year": r["original_gold_year"],
            "volatility": r["volatility"],
            "years_annotated": r["years_annotated"],
            "latest_annotated_year": r["latest_annotated_year"],
            "years_gap_to_corpus": r["years_gap_to_corpus"],
            "stable_across_annotated_years": r["stable_across_annotated_years"],
            "gold_equals_latest_annotated": r["gold_equals_latest_annotated"],
            "gold_differs_from_some_annotated_year":
                r["gold_differs_from_some_annotated_year"],
        })

    # ---- report ----
    vc = Counter(r["volatility"] for r in rows)
    vcu = Counter(u["volatility"] for u in urows)
    say("\n  VOLATILITY — base queries (n = %d)" % len(rows))
    for k in ("STABLE", "VOLATILE", "SINGLE_YEAR", "NOT_TIME_KEYED", "NO_ANNOTATION"):
        say(f"    {k:<16} {vc.get(k, 0)}")
    say("  VOLATILITY — verification units (n = %d)" % len(urows))
    for k in ("STABLE", "VOLATILE", "SINGLE_YEAR", "NOT_TIME_KEYED", "NO_ANNOTATION"):
        say(f"    {k:<16} {vcu.get(k, 0)}")

    tk = [r for r in rows if r["volatility"] in ("STABLE", "VOLATILE")]
    diff_latest = [r for r in tk if r["gold_equals_latest_annotated"] == 0]
    diff_any = [r for r in tk if r["gold_differs_from_some_annotated_year"] == 1]
    say(f"\n  gold vs source annotation (year-keyed base queries, n = {len(tk)})")
    say(f"    gold NOT equal to the latest annotated answer : {len(diff_latest)}")
    say(f"    gold differs from at least one annotated year : {len(diff_any)}")
    vol = [r for r in rows if r["volatility"] == "VOLATILE"]
    say(f"    among VOLATILE ({len(vol)}), gold differs from some year: "
        f"{sum(1 for r in vol if r['gold_differs_from_some_annotated_year'] == 1)}")

    gaps = Counter(str(r["years_gap_to_corpus"]) for r in rows
                   if r["years_gap_to_corpus"] != "")
    say(f"\n  years between latest annotation and corpus collection ({CORPUS_YEAR}): "
        f"{dict(sorted(gaps.items(), key=lambda kv: kv[0]))}")
    say(f"  latest annotated year spread: "
        f"{dict(sorted(Counter(r['latest_annotated_year'] or '(none)' for r in rows).items()))}")
    ny = Counter(r["n_years_annotated"] for r in rows if r["n_years_annotated"])
    say(f"  number of annotated years per question: {dict(sorted(ny.items()))}")

    say(f"\n  missing or ambiguous source annotations: {len(ISSUES)}")
    for k, c in sorted(Counter(i["kind"] for i in ISSUES).items()):
        say(f"    {k}: {c}")

    say("\n  INTERPRETATION LIMIT: STABLE means unchanged across the years the")
    say("  benchmark annotated. It is NOT evidence that the answer was still")
    say("  correct in 2026; SINGLE_YEAR and NOT_TIME_KEYED are UNDETERMINED,")
    say("  not stable.")

    # ---- write ----
    def wcsv(path, data, fields=None):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields or list(data[0]))
            w.writeheader()
            w.writerows(data)

    wcsv(HERE / "gold_temporal_volatility.csv", rows)
    wcsv(HERE / "unit_temporal_volatility.csv", urows)
    if ISSUES:
        wcsv(HERE / "stage2_issues.csv", ISSUES)
    else:
        open(HERE / "stage2_issues.csv", "w", encoding="utf-8").write(
            "kind,base_query_id,detail\n")

    json.dump({
        "stage": "2 — gold temporal volatility",
        "corpus_year": CORPUS_YEAR,
        "sources": {
            "temporal_align_qa": {"rows": len(d), "indexed": len(taq),
                                  "unparsable": bad_parse},
            "freshqa_10_06": {"rows": len(f), "indexed": len(fq)},
            "trivia_qa": "not loaded; NOT_TIME_KEYED by gold_source"},
        "base_queries": len(rows), "verification_units": len(urows),
        "volatility_base_queries": dict(vc),
        "volatility_units": dict(vcu),
        "gold_vs_annotation": {
            "year_keyed_base_queries": len(tk),
            "gold_not_equal_latest_annotated": len(diff_latest),
            "gold_differs_from_some_annotated_year": len(diff_any),
            "volatile_with_gold_differing": sum(
                1 for r in vol if r["gold_differs_from_some_annotated_year"] == 1)},
        "latest_annotated_year": dict(Counter(r["latest_annotated_year"] or "(none)"
                                              for r in rows)),
        "years_gap_to_corpus": dict(gaps),
        "n_years_annotated": dict(ny),
        "issues_total": len(ISSUES),
        "issues_by_kind": dict(Counter(i["kind"] for i in ISSUES)),
        "interpretation_limit": (
            "STABLE = unchanged across annotated years only. The latest annotated "
            "year precedes corpus collection, so STABLE is NOT evidence of "
            "correctness in 2026. SINGLE_YEAR and NOT_TIME_KEYED are UNDETERMINED."),
        "llm_calls": 0, "gpu_used": False, "web_calls": 0,
        "source_datasets_modified": False,
    }, open(HERE / "stage2_summary.json", "w"), indent=2, ensure_ascii=False)

    # ---- integrity of Stage 1 outputs and repo artifacts ----
    watch = [HERE / "verification_units.csv", HERE / "request_map.csv",
             ROOT / "v10_remaining_feedback" / "c2_gold_answers.json",
             ROOT / "data" / "queries.jsonl"]
    say(f"\n  Stage 1 outputs and source metadata re-hashed: "
        f"{ {p.name: sha(p)[:12] for p in watch} }")
    (HERE / "stage2.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote gold_temporal_volatility.csv, unit_temporal_volatility.csv, "
        "stage2_issues.csv, stage2_summary.json, stage2.log")
    say("  STOP — Stage 3 and all LLM judging not started, as instructed.")
    (HERE / "stage2.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
