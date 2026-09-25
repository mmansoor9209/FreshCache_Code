#!/usr/bin/env python3
"""
TASK 3A -- reconcile the observability denominators.

The earlier report stated, side by side:
    "distinct URLs that would need recollection: 1,976"
    "top domains by contribution: instagram.com 1,186, reddit.com 973, ..."
1,186 + 973 = 2,159 > 1,976, so those numbers cannot all be counts of disjoint
unique URLs from one 1,976-URL population.

Cause, read directly from run_bounds.py: `urls_needed` is a SET, so 1,976 is a
count of UNIQUE URLs; `missurl[domain]` is incremented once per
(request, URL) OCCURRENCE, so the per-domain numbers count how many paired
requests touch a URL on that domain -- a URL reused by 50 requests contributes
50. Two different denominators were printed next to each other without saying
so.

This script recomputes both quantities from row-level data and publishes the
corrected table with each denominator labelled.
"""
from __future__ import annotations
import csv, json, os, pathlib, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
RR = HERE.parent
V3 = RR.parent
ROOT = V3.parent
OB = V3 / "final_three_issues" / "03_observability_bounds"
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402
import prep2 as p2                       # noqa: E402

UNOBS = me.UNOBS
SCHEDULE, SEED = "zipf_uniform", 42


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("TASK 3A -- reconciling the observability denominators")
    prev = json.load(open(OB / "bounds_overall.json", encoding="utf-8"))
    rec = prev["recollection_requirement"]
    say(f"\n  as previously reported:")
    say(f"    paired requests with >=1 unobservable arm : "
        f"{rec['paired_requests_with_unobservable_arm']:,}")
    say(f"    'distinct URLs that would need recollection': "
        f"{rec['distinct_urls_to_recollect']:,}")
    top = rec["top_domains"][:6]
    for d, c in top:
        say(f"      {d:<32}{c:>7,}")
    s2 = sum(c for _, c in rec["top_domains"][:2])
    say(f"    first two domains sum to {s2:,} > "
        f"{rec['distinct_urls_to_recollect']:,}  <-- the inconsistency")

    # ---- recompute from row level ----
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    dom = {}
    for m in manifest:
        if m.get("url_hash") and m.get("domain"):
            dom.setdefault(m["url_hash"], m["domain"])
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c = set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    say(f"\n  held-out stream {len(stream):,} requests")

    _, rowsA = p2.replay(stream, {}, "FreshCache")
    _, rowsB = p2.replay(stream, {}, "FreshCache_AlwaysPassTemporal")
    pr = list(csv.DictReader(open(OB / "per_request_pair.csv", encoding="utf-8")))
    idx = {r["query_id"]: r for r in pr}

    def reused_urls(row):
        if not row:
            return []
        if row.get("tier") == "L1":
            return list(row.get("served") or [])
        return [u for u, _ in (row.get("ev") or [])]

    uniq = set()
    occ = Counter()          # (request, URL) occurrences, the old quantity
    uniq_by_dom = defaultdict(set)
    req_by_dom = defaultdict(set)
    nreq = 0
    for t, r in stream:
        q = r["query_id"]
        rr = idx.get(q)
        if not rr or rr["paired"] != "1":
            continue
        if rr["outcome_on"] != UNOBS and rr["outcome_off"] != UNOBS:
            continue
        nreq += 1
        for u in set(reused_urls(rowsA.get(q)) + reused_urls(rowsB.get(q))):
            per = rounds.get(u) or {}
            if not (per.get("run_00", {}).get("substantive")):
                uniq.add(u)
                occ[dom.get(u, "?")] += 1
                uniq_by_dom[dom.get(u, "?")].add(u)
                req_by_dom[dom.get(u, "?")].add(q)
    say(f"  recomputed: paired requests with >=1 unobservable arm {nreq:,} "
        f"(previously {rec['paired_requests_with_unobservable_arm']:,})")
    say(f"  recomputed: UNIQUE non-substantive URLs {len(uniq):,} "
        f"(previously {rec['distinct_urls_to_recollect']:,})")
    say(f"  sum of per-domain OCCURRENCE counts = {sum(occ.values()):,} "
        f"-- this is what the old per-domain column actually was")

    say(f"\n  == CORRECTED TABLE, with every denominator labelled ==")
    say(f"    {'domain':<34}{'unique URLs':>13}{'% of 'f'{len(uniq):,}':>14}"
        f"{'(req,URL) occurrences':>23}{'affected requests':>19}")
    rows = []
    for d in sorted(uniq_by_dom, key=lambda k: -len(uniq_by_dom[k]))[:20]:
        u, o, rq = len(uniq_by_dom[d]), occ[d], len(req_by_dom[d])
        rows.append({"domain": d, "unique_urls": u,
                     "pct_of_unique_urls": round(100 * u / len(uniq), 3),
                     "request_url_occurrences": o, "affected_requests": rq})
        say(f"    {d:<34}{u:>13,}{100*u/len(uniq):>13.2f}%{o:>23,}{rq:>19,}")
    tot_u = sum(len(v) for v in uniq_by_dom.values())
    say(f"    {'(all domains)':<34}{tot_u:>13,}{100.0:>13.2f}%"
        f"{sum(occ.values()):>23,}{nreq:>19,}")
    say(f"\n  VERDICT: not a computational error in the bound -- a REPORTING "
        f"error. 1,976 counted unique URLs; the per-domain column counted "
        f"(request, URL) occurrences. They were printed together as if they "
        f"shared a denominator. The paired bound itself never used the "
        f"per-domain numbers and is unaffected.")

    with open(HERE / "task3a_domain_reconciliation.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump({"previously_reported": rec,
               "recomputed_paired_requests_with_unobservable_arm": nreq,
               "recomputed_unique_non_substantive_urls": len(uniq),
               "sum_of_per_domain_occurrence_counts": sum(occ.values()),
               "explanation": "1,976 = unique URLs (a set); the per-domain "
                              "numbers were (request, URL) occurrences, "
                              "incremented once per paired request touching "
                              "that URL. Different denominators, printed "
                              "together without saying so.",
               "bound_affected": False,
               "corrected_table": rows},
              open(HERE / "task3a_reconciliation.json", "w"), indent=2)
    (HERE.parent / "logs").mkdir(exist_ok=True)
    (HERE.parent / "logs" / "task3a.log").write_text("\n".join(log) + "\n",
                                                     encoding="utf-8")
    say(f"\n  wrote task3a_reconciliation.json, task3a_domain_reconciliation.csv")


if __name__ == "__main__":
    main()
