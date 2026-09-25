#!/usr/bin/env python3
"""TASK 3 -- ExactTTL on the primary 31,201-request seven-day mixed-age workload.

Same replay schedule and request order (zipf_uniform, seed 42), the same
normalised exact-query matching rule, the documented class-conditioned TTL
schedule (experiment.FIXED_TTL), the same snapshot observations and freshness
protocol, and the same metric and common-support definitions as tab:main.

Unlike the earlier 02_l1only run, this one WRITES REQUEST-LEVEL OUTPUT and
recomputes every aggregate from those rows, so the table row can be checked
against the per-request file rather than trusted.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
BFR = HERE.parent
V3 = BFR.parent
ROOT = V3.parent
RCI = V3 / "remaining_critical_issues"
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(RCI / "lib"))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402
import v3_engine as V3E                  # noqa: E402
from e1_robustness import support_of      # noqa: E402

CHANGED, UNCHANGED = me.CHANGED, me.UNCHANGED
SEED, SCHEDULE = 42, "zipf_uniform"
ANCHOR = {"search_saved_pct": 62.7320, "drift_pct": 3.4154,
          "coverage_pct": 73.3736, "l1_hits": 1175, "l2_hits": 18398}
SAVED = {"search_saved_pct": 0.0128, "drift_pct": 0.0,
         "coverage_pct": 0.011187559433909493, "l1_hits": 4,
         "searches": 31197, "fetch_per_1k": 1496.68, "gen_per_1k": 999.87}
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def score(m, per, S):
    """tab:main's metric definitions, transcribed from
    remaining_critical_issues/02_l1only/run_baselines.py:score()."""
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == CHANGED)
    un = sum(1 for o in ins if o == UNCHANGED)
    det = ch + un
    n = m["n_requests"]
    return {"search_saved_pct": m["search_saved_pct"],
            "drift_pct": (100 * ch / det) if det else None,
            "coverage_pct": (100 * det / len(S)) if S else None,
            "determinable": det, "changed": ch,
            "searches": m["search_calls"], "fetches": m["fetches"],
            "generations": m["generations"],
            "fetch_per_1k": round(1000 * m["fetches"] / n, 2),
            "gen_per_1k": round(1000 * m["generations"] / n, 2),
            "search_per_1k": round(1000 * m["search_calls"] / n, 2),
            "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
            "l3_hits": m["l3_hits"], "n_requests": n}


def main():
    say("TASK 3 -- ExactTTL on the PRIMARY mixed-age workload (31,201 requests)")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    for line in open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
                     encoding="utf-8"):
        d = json.loads(line)
        rounds[d["url_hash"]] = d["rounds"]

    full = sc.build_stream(records, SCHEDULE, SEED)
    say(f"  stream: schedule={SCHEDULE} seed={SEED} -> {len(full):,} requests")
    assert len(full) == 31_201, f"expected 31,201 requests, got {len(full)}"
    S = support_of(full, rounds)
    say(f"  common support |S| = {len(S):,}")

    # ---------------- reproduction gate ----------------
    fm, fper = me.replay(full, rounds, "FreshCache", rich=rich)
    fs = score(fm, fper, S)
    bad = [k for k, v in ANCHOR.items()
           if abs(fs[k] - v) > (1e-4 if isinstance(v, float) else 0)]
    if bad:
        say(f"  REPRODUCTION GATE FAILED on {bad}: { {k: fs[k] for k in ANCHOR} }")
        sys.exit(2)
    say(f"  REPRODUCTION GATE PASSED: FreshCache reproduces tab:main -- "
        f"saved {fs['search_saved_pct']:.4f}%, drift {fs['drift_pct']:.4f}%, "
        f"coverage {fs['coverage_pct']:.4f}%, L1 {fs['l1_hits']:,}, "
        f"L2 {fs['l2_hits']:,}")

    # ---------------- ExactTTL ----------------
    say(f"\n  == ExactTTL ==")
    say(f"    matching: {V3E.normalize.__doc__.splitlines()[0] if V3E.normalize.__doc__ else 'normalised exact query match'}")
    say(f"    class-conditioned TTL (experiment.FIXED_TTL): {exp.FIXED_TTL}")
    em, eper, elog = V3E.replay_l1cache(full, rounds, match="exact", ttl="fixed")
    es = score(em, eper, S)

    # ---------------- request-level output ----------------
    outcome_of = dict(eper)
    # v3_engine returns `rows` as a dict query_id -> {"tier": ...}; the
    # exact-match cache labels a served request "HIT".
    tier_of = {q: v.get("tier") for q, v in (elog or {}).items()}
    rows = []
    for t, r in full:
        q = r["query_id"]
        o = outcome_of.get(q)
        rows.append({"query_id": q, "t": t,
                     "cluster_id": r.get("cluster_id") or q,
                     "freshness_class": r["freshness_class"],
                     "in_support": int(q in S),
                     "tier": ("L1" if tier_of.get(q) == "HIT" else tier_of.get(q, "miss")),
                     "outcome": ("" if o is None else o)})
    with open(HERE / "exactttl_mixed_per_request.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    say(f"    wrote {len(rows):,} request-level rows")

    # ---------------- verify aggregates FROM the per-request file ----------
    say(f"\n  == verification against the request-level file ==")
    rr = list(csv.DictReader(open(HERE / "exactttl_mixed_per_request.csv",
                                  encoding="utf-8")))
    ins = [x for x in rr if x["in_support"] == "1" and x["outcome"]]
    ch = sum(1 for x in ins if x["outcome"] == CHANGED)
    det = sum(1 for x in ins if x["outcome"] in (CHANGED, UNCHANGED))
    l1 = sum(1 for x in rr if x["tier"] == "L1")
    cov = 100 * det / len(S)
    drift = (100 * ch / det) if det else None
    checks = [
        ("n_requests", len(rr), es["n_requests"]),
        ("L1 hits", l1, es["l1_hits"]),
        ("determinable", det, es["determinable"]),
        ("changed", ch, es["changed"]),
        ("coverage_pct", round(cov, 6), round(es["coverage_pct"], 6)),
        ("drift_pct", (round(drift, 6) if drift is not None else None),
         (round(es["drift_pct"], 6) if es["drift_pct"] is not None else None)),
    ]
    for name, a, b in checks:
        say(f"    {name:<16} from rows {str(a):>10}   engine {str(b):>10}   "
            f"{'OK' if a == b else 'MISMATCH'}")
    ok_rows = all(a == b for _, a, b in checks)

    say(f"\n  == cross-check against the earlier 02_l1only run ==")
    for k, v in SAVED.items():
        cur = es[k]
        agree = (abs(cur - v) < 1e-4 if isinstance(v, float) and v is not None
                 else cur == v)
        say(f"    {k:<18} now {str(round(cur,4) if isinstance(cur,float) else cur):>12}"
            f"   saved {str(v):>12}   {'OK' if agree else 'MISMATCH'}")
    ok_saved = all((abs(es[k] - v) < 1e-4 if isinstance(v, float) else es[k] == v)
                   for k, v in SAVED.items())

    # ---------------- the tab:main row ----------------
    say(f"\n  == ExactTTL row for tab:main (PRIMARY mixed-age, n=31,201) ==")
    say(f"    search savings        {es['search_saved_pct']:.4f}%")
    say(f"    hash drift            {es['drift_pct']:.4f}%  "
        f"({es['changed']} changed of {es['determinable']} determinable)")
    say(f"    observable coverage   {es['coverage_pct']:.4f}%  "
        f"({es['determinable']} of |S| = {len(S):,})")
    say(f"    fetches / 1k          {es['fetch_per_1k']:.2f}")
    say(f"    generations / 1k      {es['gen_per_1k']:.2f}")
    say(f"    L1 hits               {es['l1_hits']:,}")
    say(f"    total search calls    {es['searches']:,}")
    say(f"\n    NOTE the held-out ExactTTL figure is 0.0092% savings on 21,848")
    say(f"    cluster-disjoint requests. That is a DIFFERENT population and is")
    say(f"    NOT interchangeable with the {es['search_saved_pct']:.4f}% above.")

    json.dump({"schedule": SCHEDULE, "seed": SEED, "n_requests": len(full),
               "support": len(S),
               "anchor_reproduced": True, "freshcache_row": fs,
               "exactttl_row": es,
               "verified_against_request_level_rows": ok_rows,
               "agrees_with_02_l1only": ok_saved,
               "heldout_value_kept_separate": {"population": "21,848 "
                                               "cluster-disjoint held-out "
                                               "requests",
                                               "search_saved_pct": 0.0092},
               "fixed_ttl": exp.FIXED_TTL},
              open(HERE / "exactttl_mixed_results.json", "w"), indent=2)
    (BFR / "logs" / "t3.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote exactttl_mixed_results.json, exactttl_mixed_per_request.csv")


if __name__ == "__main__":
    main()
