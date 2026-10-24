#!/usr/bin/env python3
"""
Recover INDEPENDENT gold answers across all source datasets,
with provenance, and report class coverage before any grading.

Sources inspected:
  ROIM/temporal-alignment-qa [test]  year-keyed answer dictionary
  natyou/freshqa_10_06 [test]        answer_0..answer_9 plus effective_year
  mandarjoshi/trivia_qa rc [train]   value + aliases
  vtllms/sealqa                      only 1 query in the benchmark, skipped

TEMPORAL ALIGNMENT, AND THE LEAK QUESTION
  TemporalAlignQA answers are keyed by year. Our snapshots were collected in
  2026. The dictionary's latest annotated year is therefore in the PAST relative
  to the scripts/evaluation window, so taking it cannot leak future information. It can
  however be STALE relative to the scripts/evaluation date, which is recorded per row as
  gold_year and reported, not hidden.

  FreshQA carries effective_year and a next_review cadence; rows whose premise is
  false are excluded, following the benchmark's own construction rule already
  used in query_construction.py.

PARAPHRASE PROPAGATION
  A gold answer is propagated to a paraphrase ONLY if the paraphrase was judged
  faithful to its base question in data/paraphrase_fidelity_judgments.jsonl.
  Unverified paraphrases are left without gold rather than assumed.

Writes only outputs/gold_answers/.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os; import ast, csv, json, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter, defaultdict

ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
OUT = ((OUTPUT_ROOT / "gold_answers"))
norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())


def main():
    from datasets import load_dataset
    q = [json.loads(l) for l in open((DATA_ROOT / "queries.jsonl"))]
    para = [json.loads(l) for l in open((DATA_ROOT / "paraphrase_clusters.jsonl"))]
    qbytext = {norm(r["query"]): r for r in q}
    gold = {}

    def add(k, ans, aliases, source, year, note=""):
        if not ans or k in gold:
            return False
        gold[k] = {"answer": str(ans), "aliases": [str(a) for a in aliases][:12],
                   "gold_source": source, "gold_year": year, "note": note}
        return True

    print("="*96); print("  STEP 4A — INDEPENDENT GOLD RECOVERY"); print("="*96)

    n = 0
    d = load_dataset("ROIM/temporal-alignment-qa", split="test")
    years = Counter()
    for row in d:
        k = norm(row["question"])
        if k not in qbytext:
            continue
        a = row["answer"]
        if isinstance(a, str):
            try:
                a = ast.literal_eval(a)
            except Exception:
                continue
        if not isinstance(a, dict) or not a:
            continue
        yr = max(a, key=lambda y: str(y))
        v = a[yr] if isinstance(a[yr], list) else [a[yr]]
        v = [str(x) for x in v if str(x).strip()]
        if not v:
            continue
        years[str(yr)] += 1
        n += add(k, v[0], v, "temporal_align_qa", str(yr),
                 "latest annotated year in the answer dictionary")
    print(f"\n  temporal-alignment-qa[test]  matched {n:,}")
    print(f"      gold year distribution: "
          + ", ".join(f"{y}:{c}" for y, c in sorted(years.items())[-6:]))

    n = 0
    d = load_dataset("natyou/freshqa_10_06", split="test")
    for row in d:
        k = norm(row["question"])
        if k not in qbytext or row.get("false_premise") in (True, "True"):
            continue
        al = [row.get(f"answer_{i}") for i in range(10)]
        al = [str(x).strip() for x in al if x and str(x).strip()]
        if not al:
            continue
        n += add(k, al[0], al, "freshqa", str(row.get("effective_year", "")))
    print(f"  freshqa_10_06[test]          matched {n:,}")

    n = 0
    d = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")
    for row in d:
        k = norm(row["question"])
        if k not in qbytext:
            continue
        a = row.get("answer") or {}
        val = a.get("value") or ""
        if not val:
            continue
        n += add(k, val, [val]+list(a.get("aliases") or []), "triviaqa", None)
    print(f"  trivia_qa rc[train]          matched {n:,}")

    # paraphrase propagation, gated on verified fidelity
    faith = set()
    fp = (DATA_ROOT / "paraphrase_fidelity_judgments.jsonl")
    if fp.exists():
        for r in (json.loads(l) for l in open(fp)):
            v = str(r.get("verdict", r.get("faithful", ""))).upper()
            if v in ("YES", "FAITHFUL", "TRUE"):
                faith.add(norm(r.get("paraphrase") or r.get("query") or ""))
    base_text = {r["query_id"]: norm(r["query"]) for r in q}
    npar = nskip = 0
    for p in para:
        b = base_text.get(p.get("base_query_id", ""))
        k = norm(p["query"])
        if not b or b not in gold or k in gold:
            continue
        if faith and k not in faith:
            nskip += 1
            continue
        gold[k] = {**gold[b], "note": "propagated from verified-faithful paraphrase",
                   "paraphrase_of": b}
        npar += 1
    print(f"\n  propagated to {npar:,} paraphrases "
          f"({'fidelity-verified only; ' if faith else ''}{nskip:,} skipped as unverified)")

    # ---- coverage by class over realized hits ---------------------------
    def cover(path, label):
        hits = [json.loads(l) for l in open(DATA_ROOT / path[5:])]
        rows, tot = [], Counter()
        cov = Counter(); src = defaultdict(Counter)
        for h in hits:
            fc = h.get("fc_incoming", "?")
            tot[fc] += 1
            g = gold.get(norm(h["query"]))
            if g:
                cov[fc] += 1
                src[fc][g["gold_source"]] += 1
        print(f"\n  {label}: {len(hits):,} hits")
        print(f"  {'class':<12}{'hits':>8}{'with gold':>11}{'coverage':>10}   gold sources")
        for fc in ("TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"):
            if not tot[fc]:
                continue
            s = ", ".join(f"{k} {v}" for k, v in src[fc].most_common())
            print(f"  {fc:<12}{tot[fc]:>8,}{cov[fc]:>11,}"
                  f"{100*cov[fc]/tot[fc]:>9.1f}%   {s or '-'}")
            rows.append({"population": label, "freshness_class": fc,
                         "hits": tot[fc], "with_gold": cov[fc],
                         "coverage_pct": round(100*cov[fc]/tot[fc], 2),
                         "gold_sources": s})
        print(f"  {'ALL':<12}{sum(tot.values()):>8,}{sum(cov.values()):>11,}"
              f"{100*sum(cov.values())/max(sum(tot.values()),1):>9.1f}%")
        rows.append({"population": label, "freshness_class": "ALL",
                     "hits": sum(tot.values()), "with_gold": sum(cov.values()),
                     "coverage_pct": round(100*sum(cov.values())/max(sum(tot.values()), 1), 2),
                     "gold_sources": ""})
        return rows

    print(f"\n{'='*96}\n  STEP 4B — COVERAGE BY FRESHNESS CLASS\n{'='*96}")
    rows = []   # coverage of earlier rule-based L1/L2 hit lists (optional diagnostic inputs, not distributed)
    for _p, _lab in (("data/l1_hits_rulebased_t24h.jsonl", "L1"), ("data/l2_hits_t24h_floor075.jsonl", "L2")):
        if (DATA_ROOT / _p[5:]).exists(): rows += cover(_p, _lab)
    if rows:
      with open(OUT/"reference_coverage.csv", "w", newline="", encoding="utf-8") as f:
        wc = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wc.writeheader()
        wc.writerows(rows)

    with open(OUT/"reference_provenance.csv", "w", newline="", encoding="utf-8") as f:
        wc = csv.writer(f)
        wc.writerow(["query_norm", "gold_answer", "n_aliases", "gold_source",
                     "gold_year", "note"])
        for k, g in gold.items():
            wc.writerow([k[:160], g["answer"][:120], len(g["aliases"]),
                         g["gold_source"], g["gold_year"] or "", g["note"]])
    json.dump(gold, open(OUT/"reference_answers.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  total gold {len(gold):,}  by source: "
          + ", ".join(f"{k} {v:,}" for k, v in
                      Counter(g['gold_source'] for g in gold.values()).items()))
    print(f"  wrote reference_answers.json, reference_coverage.csv, reference_provenance.csv")


if __name__ == "__main__":
    main()
