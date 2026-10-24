#!/usr/bin/env python3
"""vCache-style baseline: our reimplementation of vCache's per-entry learned-threshold policy (Schroeder et al., ICLR 2026), not
the complete published system; it uses proxy correctness feedback and does not test the published error guarantee.


scripts/baselines/vcache/vcache_reference_replay.py

Fidelity check of the vCache baseline. Runs the reference-consistent policy at
fixed 24h and on the heterogeneous-age replay, under the same scoring protocol
used for every other method, and documents how it differs from the
static-threshold variant.

Nothing existing is modified or overwritten. vCache outputs of earlier runs
(optional read-only pools) are read, not touched.

The released reference math is reused verbatim from vcache_policy.py
(_select_action_explore, _estimate_parameters, _get_tau). Only the
nearest-neighbour scan is vectorised, preserving the reference's semantics:
argmax over every cache entry with no similarity floor, ties going to the
earliest entry.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import inspect
import json
import math
import os
import pathlib
import pickle
import random
import sys
import time
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = ((OUTPUT_ROOT / "replay/baseline_measurement"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402
_AGG = getattr(me, 'aggregate', None) or (lambda labs: (CHANGED if CHANGED in labs else (UNCHANGED if UNCHANGED in labs else UNOBS)) if labs else None)
import vcache_policy as V     # noqa: E402
from schedule_robustness import support_of  # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS
SCHEDULE, SEED = "zipf_uniform", 42
DELTA = V.PAPER_DELTA           # 0.10, the value the paper reports vCache at
RNG_SEED = V.RANDOM_SEED        # 42

FC_TRACE = (OUTPUT_ROOT / "archived_runs/mixed_age_precision_consistency/traces.pkl")
STATIC_VARIANT_MIXED = (OUTPUT_ROOT / "archived_runs/vcache_mixed_age/vcache_mixed_age_metrics.json")

# How to handle a verification whose verdict the benchmark cannot determine.
#   "no_update"      record nothing; the verification simply did not resolve
#   "assume_correct" record c=1, which is what the static-threshold variant does
UNOBS_POLICY = "no_update"


def reference_replay(items, rounds=None, labels=None, delta=DELTA,
                     unobs_policy=UNOBS_POLICY):
    """vCache VerifiedDecisionPolicy on either protocol.

    items: fixed-24h -> list of (None, record); mixed-age -> list of (t, record)
    labels: fixed-horizon label map (fixed-24h mode)
    rounds: per-URL round table (mixed-age mode)

    Decision math is vcache_policy's, unchanged. The cache, the
    explore/exploit branches, the entry-insertion rule and the observation
    timing follow sim_vcache_reference line for line.
    """
    mixed = rounds is not None
    rng = random.Random(RNG_SEED)
    IDX, SIM = exp._QUERY_TO_IDX, exp._SIM_MATRIX
    n_max = len(items)

    c_qi = np.empty(n_max, dtype=np.int64)
    c_t = np.empty(n_max, dtype=np.float64)
    c_urls, c_set, c_obs = [], [], []
    n = 0

    search = fetches = hits = misses = feedback = 0
    skipped_unverifiable = 0
    rt_requests = 0
    ev = Counter()
    outcomes, events = [], []

    def label_of(served, t_cached, t_req):
        """Three-valued outcome of the served evidence, protocol-consistent."""
        if mixed:
            labs = [me.outcome(u, t_cached, t_req, rounds) for u in served]
        else:
            labs = [labels.get(u, UNOBS) for u in served]
        return (_AGG(labs) if _AGG(labs) is not None else UNOBS)   # engine aggregation

    for i, (t, r) in enumerate(items):
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        qi = IDX[q]
        own = [u["url_hash"] for u in urls]
        own_set = frozenset(own)
        row = SIM[qi]

        ev_row = {"request_index": i, "query_id": qid, "timestamp": t,
                  "freshness_class": fc, "action": "", "similarity": "",
                  "tau": "", "fitted": "", "matched_entry": "",
                  "feedback_label": "", "feedback_recorded": 0,
                  "determinable": 0, "content_drift_status": "",
                  "search": 0, "fetches": 0, "generation": 0}

        def full_pipeline(add_entry, label_note=""):
            nonlocal search, fetches, misses, n
            search += 1
            fetches += len(own)
            misses += 1
            ev_row.update({"action": "miss" + label_note, "search": 1,
                           "fetches": len(own), "generation": 1})
            if add_entry:
                c_qi[n] = qi
                c_t[n] = t if t is not None else 0.0
                c_urls.append(list(own)); c_set.append(own_set)
                c_obs.append([])
                nonlocal_inc()

        def nonlocal_inc():
            nonlocal n
            n += 1

        if fc == "REAL_TIME":
            rt_requests += 1
            full_pipeline(add_entry=True, label_note="_realtime")
            outcomes.append((qid, None))
            events.append(ev_row)
            continue

        if n == 0:
            full_pipeline(add_entry=True, label_note="_cold")
            outcomes.append((qid, None))
            events.append(ev_row)
            continue

        sims = row[c_qi[:n]]
        j = int(np.argmax(sims))          # first max, as the reference loop does
        best_sim = float(sims[j])
        explore, fitted, tau = V._select_action_explore(
            best_sim, c_obs[j], delta, rng)
        ev_row.update({"similarity": round(best_sim, 6),
                       "tau": round(float(tau), 6), "fitted": int(fitted),
                       "matched_entry": j})

        if not explore:
            # exploit: serve the cached answer
            hits += 1
            out = label_of(c_urls[j], float(c_t[j]), t)
            ev[out] += 1
            outcomes.append((qid, out))
            ev_row.update({"action": "hit",
                           "determinable": 1 if out in (CHANGED, UNCHANGED) else 0,
                           "content_drift_status": out})
            events.append(ev_row)
            continue

        # explore: full pipeline, then record the verified label
        out = label_of(c_urls[j], float(c_t[j]), t)
        same_question = (c_set[j] == own_set)
        if not same_question:
            c = 0                      # wrong question: verifiably incorrect
            record = True
        elif out == CHANGED:
            c = 0
            record = True
        elif out == UNCHANGED:
            c = 1
            record = True
        else:                          # verdict unavailable
            if unobs_policy == "assume_correct":
                c, record = 1, True
            else:
                c, record = None, False
                skipped_unverifiable += 1
        if record:
            c_obs[j].append((round(best_sim, 3), c))
            feedback += 1
            ev_row.update({"feedback_label": c, "feedback_recorded": 1})
        outcomes.append((qid, None))
        full_pipeline(add_entry=(c == 0), label_note="_explore")
        events.append(ev_row)

    return {
        "n_requests": len(items), "search_calls": search, "fetches": fetches,
        "search_saved_pct": round(100 * (1 - search / len(items)), 4),
        "hits": hits, "misses": misses, "generations": search,
        "conditional_gets": 0,
        "feedback_updates": feedback,
        "unverifiable_skipped": skipped_unverifiable,
        "cache_entries": n, "realtime_requests": rt_requests,
        "changed_events": ev[CHANGED], "unchanged_events": ev[UNCHANGED],
        "unobservable_events": ev[UNOBS],
        "delta": delta, "unobservable_policy": unobs_policy,
    }, outcomes, events


def score(outcomes, S):
    ins = [o for qid, o in outcomes if o is not None and qid in S]
    ch = sum(1 for o in ins if o == CHANGED)
    un = sum(1 for o in ins if o == UNCHANGED)
    det = ch + un
    return {"changed": ch, "unchanged": un, "determinable": det,
            "drift_pct": (100 * ch / det) if det else None,
            "coverage_pct": (100 * det / len(S)) if S else None,
            "support": len(S)}


def main():
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
    with open((DATA_ROOT / "derived/round_table.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]

    # ---------------- PART C: causality ----------------
    src = inspect.getsource(reference_replay)
    checks = {
        "decision_uses_only_prior_feedback":
            src.index("_select_action_explore(") < src.index("c_obs[j].append("),
        "label_appended_only_after_serving":
            "c_obs[j].append(" in src.split("# explore: full pipeline")[1],
        "no_future_web_state_in_threshold":
            "label_of" not in src.split("explore, fitted, tau")[0].split(
                "sims = row[c_qi[:n]]")[-1],
        "heterogeneous_feedback_uses_cache_and_request_time_only":
            "me.outcome(u, t_cached, t_req, rounds)" in src,
    }
    print("=" * 96)
    print("  PART C — CAUSALITY (reference-consistent implementation)")
    print("=" * 96)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise SystemExit("  ABORT: causality condition failed.")

    # ---------------- PART D: fixed 24h ----------------
    print("\n" + "=" * 96)
    print("  PART D — FIXED 24H")
    print("=" * 96, flush=True)
    labels24 = ce.load_labels("rerun_24h")
    obs24 = {u for u, v in labels24.items() if v in (CHANGED, UNCHANGED)}
    nonrt = [r for r in records if r["freshness_class"] != "REAL_TIME"]
    S24 = {r["query_id"] for r in nonrt
           if any(u["url_hash"] in obs24 for u in r["urls"])}

    t0 = time.time()
    m24, o24, _ = reference_replay([(None, r) for r in records],
                                   labels=labels24)
    s24 = score(o24, S24)
    m24.update(s24)
    print(f"  reference-consistent: saved {m24['search_saved_pct']:.4f}%  "
          f"drift {s24['drift_pct'] if s24['drift_pct'] is None else round(s24['drift_pct'],4)}%  "
          f"coverage {s24['coverage_pct']:.4f}%  hits {m24['hits']:,}  "
          f"feedback {m24['feedback_updates']:,}  ({time.time()-t0:.0f}s)",
          flush=True)

    mm24, mo24 = ea.replay(records, labels24, 86_400, "vCache", rich=rich)
    ms24 = score(mo24, S24)
    print(f"  static-threshold variant: saved {mm24['search_saved_pct']:.4f}%  "
          f"drift {ms24['drift_pct']:.4f}%  coverage {ms24['coverage_pct']:.4f}%  "
          f"hits {mm24['l1_hits']:,}")

    json.dump({"delta": DELTA, "rng_seed": RNG_SEED,
               "unobservable_policy": UNOBS_POLICY,
               "support_24h": len(S24),
               "reference_consistent": m24,
               "static_threshold_variant": {
                   "search_saved_pct": mm24["search_saved_pct"],
                   "search_calls": mm24["search_calls"],
                   "fetches": mm24["fetches"], "l1_hits": mm24["l1_hits"],
                   **ms24}},
              open(HERE / "fixed24_reference_run.json", "w"), indent=2)

    # ---------------- PART E: mixed age ----------------
    print("\n" + "=" * 96)
    print("  PART E — MIXED AGE")
    print("=" * 96, flush=True)
    stream = sc.build_stream(records, SCHEDULE, SEED)
    S = support_of(stream, rounds)
    t0 = time.time()
    mM, oM, evM = reference_replay(stream, rounds=rounds)
    sM = score(oM, S)
    mM.update(sM)
    print(f"  saved {mM['search_saved_pct']:.4f}%  "
          f"drift {sM['drift_pct'] if sM['drift_pct'] is None else round(sM['drift_pct'],4)}%  "
          f"coverage {sM['coverage_pct']:.4f}%  hits {mM['hits']:,}  "
          f"feedback {mM['feedback_updates']:,}  "
          f"skipped {mM['unverifiable_skipped']:,}  ({time.time()-t0:.0f}s)",
          flush=True)

    # sensitivity: the UNOBSERVABLE handling of the static-threshold variant
    mM2, oM2, _ = reference_replay(stream, rounds=rounds,
                                   unobs_policy="assume_correct")
    sM2 = score(oM2, S)
    mM2.update(sM2)
    print(f"  sensitivity (UNOBSERVABLE treated as correct): "
          f"saved {mM2['search_saved_pct']:.4f}%  "
          f"drift {sM2['drift_pct'] if sM2['drift_pct'] is None else round(sM2['drift_pct'],4)}%  "
          f"hits {mM2['hits']:,}")

    json.dump({"schedule": SCHEDULE, "seed": SEED, "delta": DELTA,
               "rng_seed": RNG_SEED, "support": len(S),
               "primary_no_update_on_unverifiable": mM,
               "sensitivity_assume_correct": mM2},
              open(HERE / "mixed_age_reference_run.json", "w"), indent=2)

    with open(HERE / "mixed_age_reference_events.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(evM[0]))
        w.writeheader()
        w.writerows(evM)

    # ---------------- PART F: common determinate ----------------
    pair = None
    if FC_TRACE.exists():
        with open(FC_TRACE, "rb") as fh:
            T = pickle.load(fh)
        fc_sig = T["methods"]["FreshCache"]["_outcome_sig"]
        assert T["qids"] == [r["query_id"] for _, r in stream]
        v_sig = [o for _, o in oM]
        idxs = [i for i in range(len(stream))
                if stream[i][1]["query_id"] in S
                and fc_sig[i] in (CHANGED, UNCHANGED)
                and v_sig[i] in (CHANGED, UNCHANGED)]
        N = len(idxs)
        fch = sum(1 for i in idxs if fc_sig[i] == CHANGED)
        vch = sum(1 for i in idxs if v_sig[i] == CHANGED)
        pair = {"shared_determinable_N": N,
                "shared_workload_pct": round(100 * N / len(stream), 4),
                "freshcache_changed": fch,
                "freshcache_drift_pct": round(100 * fch / N, 4) if N else None,
                "vcache_changed": vch,
                "vcache_drift_pct": round(100 * vch / N, 4) if N else None,
                "difference_pp": round(100 * (fch - vch) / N, 4) if N else None,
                "significance_testing": (
                    "not reported: the two arms' determinable sets are produced "
                    "by different reuse mechanisms and the shared set is small, "
                    "so a paired test here would not answer a well-posed "
                    "question")}
        with open(HERE / "common_determinate.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair))
            w.writeheader()
            w.writerow(pair)
        print(f"\n  common determinate: N {N:,}  FreshCache "
              f"{pair['freshcache_drift_pct']}%  vCache "
              f"{pair['vcache_drift_pct']}%")

    write_docs(m24, ms24, mm24, mM, mM2, pair, checks, len(S24), len(S))
    final_print(m24, ms24, mm24, mM, pair, checks)


def write_docs(m24, ms24, mm24, mM, mM2, pair, checks, n_s24, n_s):
    man = (json.load(open(STATIC_VARIANT_MIXED, encoding="utf-8"))
           if STATIC_VARIANT_MIXED.exists() else None)

    (HERE / "implementation_comparison.md").write_text(f"""# vCache implementations in this repository

## Every implementation found

| | `tiered_cache_model.py` / `method_replay.py` | `vcache_policy.py` |
|---|---|---|
| file path | `tiered_cache_model.py` (`_vcache_threshold`, `sim_vcache`); used by `freshcache/selection_replay_engine/method_replay.py` `replay()` L1 mode `"vcache"` | `vcache_policy.py` |
| functions | `_vcache_threshold`, `sim_vcache`; engine branch `elif mode == "vcache"` | `sim_vcache_reference`, `_select_action_explore`, `_estimate_parameters`, `_get_tau`, `_fit_logistic` |
| threshold initialization | no observations returns `L1_SIM_THRESHOLD` = 0.40 | fewer than 6 observations returns EXPLORE; no threshold is formed |
| threshold update equation | scan observed similarities ascending, keep the **largest** t whose subset error rate <= delta | logistic fit of label on [1, s]; `t_hat = -intercept/gamma`; `tau = min over eps grid of 1 - delta/(1 - alpha)`, Eq 11 |
| feedback definition | `0 if any served URL CHANGED else 1` — a property of the entry alone | `c = 1 iff entry.url_set == x.url_set AND no served URL stale` — a property of the (request, entry) pair |
| feedback timing | appended on the **exploit/hit** branch | appended on the **explore** branch, matching Algorithm 1 lines 7-9 |
| similarity metric | BGE-M3 cosine from the shared matrix | same matrix, via `exp.cosine_sim` |
| threshold scope | per entry | per entry |
| clipping / bounds | threshold floored at 0.40; candidate must clear 0.40 | `t_hat` clipped to [0,1]; `gamma >= 1e-6`; similarities rounded to 3 dp; tau not clamped below 0 |
| minimum observations | none | `MIN_OBSERVATIONS = 6` |
| smoothing / confidence | none | `var_t` by delta method, or a ported empirical `variance_map` when observations are perfectly separable; epsilon grid of 50 points |
| cache write semantics | an entry is written for **every** request | on explore, an entry is inserted **only when the cached response was wrong** (`add_entry = (c == 0)`) |
| hit semantics | serve iff similarity >= learned threshold | exploit iff `u > tau` with `u ~ U(0,1)`; explore otherwise |
| basis | none cited in code | ported against `vcache/vcache_policy/strategies/verified.py` and Schroeder et al., arXiv:2502.03771v4, Algorithms 1-2 and Eqs 9-11 |

## Which produced the static-threshold variant

**`method_replay.py` L1 mode `"vcache"`**, with `delta = EPS_ANSWER = 0.10`.
It reproduces the static-threshold variant's recorded savings, drift and coverage.

## Which is faithful to the reference

**`vcache_policy.py`.** Its own header documents two differences from the reference in the
`tiered_cache_model.py` rule, and both are visible in the code above:

1. the threshold scan keeps the **largest** passing threshold, so at the top
   candidate the subset is a single observation whose error rate is 0 or 1, the
   constraint almost always passes, and the threshold collapses to the maximum
   observed similarity regardless of delta;
2. observations are appended on **exploit**, so an entry can only learn once it
   is already being served, whereas Algorithm 1 records them on **explore**.

The consequence is checkable rather than rhetorical: a delta sweep from 0.01 to
0.90 under the `tiered_cache_model.py` rule produced byte-identical results, meaning the
error budget was inert. Under the reference the same sweep moves savings
substantially (the per-delta values are in this folder).

There is a third difference the header does not list, and it matters most for
this benchmark: the `method_replay` label is a property of the **entry** alone.
Every observation recorded against a given entry therefore carries the same
label whenever its evidence is stable, which is exactly the degenerate,
single-class case the reference file warns about.

## Neither is declared "correct" here

The `method_replay` rule is a defensible similarity-threshold cache. It is not the
vCache VerifiedDecisionPolicy, and the difference is large enough that the two
should not share a name in a paper.
""", encoding="utf-8")

    (HERE / "feedback_semantics.md").write_text(f"""# What vCache's feedback is, and what ours is

## The reference definition

vCache's verification label is `c(x) = 1` iff serving the cached response to
**this** prompt would have been correct, i.e. `r(nn(x)) == r(x)`. It is a
property of the pair (request, cached entry), produced by comparing the cached
answer against the correct answer for the new prompt. It is answer-level.

## Our benchmark's adaptation

`vcache_policy.py` states its adaptation openly: `c(x) = 1` iff the
entry answers the same question, i.e. its URL set is the one `x` would itself
have retrieved, **and** that evidence has not gone stale.

The static-threshold variant (`method_replay`) drops the first condition and keeps only
staleness.

## Answers

1. **Is URL-content change a faithful realization of the reference feedback?**
   No. The reference compares answers; we compare evidence. Evidence change is
   neither necessary nor sufficient for an answer to be wrong: a page can change
   without the answer changing, and an answer can be wrong from unchanged
   evidence that never supported it.

2. **Is it a proxy rather than answer correctness?**
   Yes, a proxy. The two-part form (same question AND evidence fresh) is the
   closer proxy, because correctness of a cached answer for a new prompt
   requires both. The staleness-only form used in the static-threshold variant is a weaker
   proxy and is entry-level rather than pair-level.

3. **Is treating UNOBSERVABLE as correct defensible?**
   No, not as a default. It fabricates a verification verdict the benchmark
   cannot produce, and it does so in one direction only: every unresolvable case
   becomes evidence that reuse was safe, which pushes the learned threshold down
   and admits more hits. The bias favours the baseline, so this is not a
   FreshCache-favourable choice to remove.

4. **Does the reference algorithm require information unavailable here?**
   Yes. It requires the correct answer for the new prompt at verification time,
   which this benchmark does not have for every request. That is precisely why a
   proxy is used, and why the proxy should be described as such.

5. **Closest reproducible adaptation without privileged future information.**
   Keep the pair-level label: `c = 0` when the entry answers a different
   question, `c = 0` when the served evidence changed between caching and reuse,
   `c = 1` when the same question's evidence is verifiably unchanged, and
   **record nothing when the verdict cannot be determined**. All three
   determinable cases use only information available at request time about the
   interval from cache time to request time. No future state is consulted.

## The UNOBSERVABLE decision

Chosen: **no update**. The reference's verifier always returns a verdict; when
ours cannot, the faithful analogue is that the verification did not happen, not
that it returned "correct". Excluding the request entirely would also distort
the run, since the request still occurred and still cost a search.

Both are reported. Under this replay the choice changes the result materially,
and the direction is the one the reasoning predicts:

| UNOBSERVABLE handling | Saved % | Drift % | Hits | Feedback updates |
|---|---|---|---|---|
| no update (chosen) | {mM['search_saved_pct']:.4f} | {mM['drift_pct'] if mM['drift_pct'] is None else round(mM['drift_pct'], 4)} | {mM['hits']:,} | {mM['feedback_updates']:,} |
| assume correct (static-threshold variant) | {mM2['search_saved_pct']:.4f} | {mM2['drift_pct'] if mM2['drift_pct'] is None else round(mM2['drift_pct'], 4)} | {mM2['hits']:,} | {mM2['feedback_updates']:,} |
""", encoding="utf-8")

    man_gen = man["mixed_age"]["generations"] if man else None
    (HERE / "operation_accounting.md").write_text(f"""# Generation accounting

## The logically correct rule for an answer cache

A hit returns a stored answer and generates nothing. A miss runs the pipeline
and generates once. So generations = misses = searches, for any answer-only
cache.

| run | searches | hits | logically correct generations |
|---|---|---|---|
| reference-consistent, mixed age | {mM['search_calls']:,} | {mM['hits']:,} | {mM['search_calls']:,} |
| static-threshold variant, mixed age | {man['mixed_age']['search_calls']:,} | {man['mixed_age']['l1_hits']:,} | {man_gen:,} |

Both of those already follow the rule.

## Where a different rule was used

`freshcache/selection_replay_engine/method_replay.py` charges `LAT["llm_generate"]` on the L1 hit
branch:

```
lat += LAT["l1_lookup"] + LAT["llm_generate"]
```

so every answer-cache hit pays a generation in the **latency** model even though
it returns a stored answer. `vcache_policy.py` does the same in
`sim_vcache_reference`, charging `exp.LATENCY["llm_generate"]` on exploit.

This does not affect any count reported here, which are derived from searches
and hits directly. It does affect any latency or total-cost figure taken from
those engines, including the recorded vCache latency numbers, which are
therefore pessimistic for vCache by one generation per hit.

Historical outputs are left exactly as they are; this is a documentation note,
not a change.
""", encoding="utf-8")

    rec_gen = mM["search_calls"]
    (HERE / "final_recommendation.md").write_text(f"""# vCache-style baseline

## The baseline

The vCache-style baseline is `vcache_policy.py` (`sim_vcache_reference`) at the paper's `delta = {DELTA}`.

## The heterogeneous-age adaptation, in one sentence

Because a single fixed-horizon staleness label is undefined when cached entries
have different ages, vCache's verification label is evaluated per (cache time,
request time) pair against the same Web step function used for every other
method, leaving the policy, its delta and its update rule unchanged.
""", encoding="utf-8")


def final_print(m24, ms24, mm24, mM, pair, checks):
    print()
    print("VCACHE BASELINE FIDELITY")
    print()
    print("STATIC-THRESHOLD VARIANT:")
    print("  experiment._vcache_threshold via freshcache/selection_replay_engine/method_replay.py replay(), L1 mode 'vcache', delta = EPS_ANSWER = 0.10.")
    print("  Largest-passing-threshold scan, observations appended on exploit, "
          "entry-level staleness label.")
    print()
    print("REFERENCE-CONSISTENT IMPLEMENTATION:")
    print("  vcache_policy.py sim_vcache_reference, ported against vcache/vcache_policy/strategies/verified.py")
    print("  and Schroeder et al. arXiv:2502.03771v4 Algorithms 1-2, Eqs 9-11. "
          f"delta = {DELTA}, seed {RNG_SEED}.")
    print()
    print("KEY POLICY DIFFERENCES:")
    print("  1. threshold: largest passing similarity vs logistic fit with the "
          "Eq-11 tau rule")
    print("  2. learning: observations on exploit vs on explore")
    print("  3. label scope: entry-level staleness vs pair-level "
          "(same question AND evidence fresh)")
    print("  4. entry insertion: every request vs only when the cached response "
          "was wrong")
    print("  5. minimum observations: none vs 6 before any fit")
    print("  Consequence: under the static-threshold rule a delta sweep 0.01-0.90 is "
          "byte-identical, i.e. the error budget is inert.")
    print()
    print("FEEDBACK SEMANTICS:")
    print("  Reference grades ANSWERS. Ours grades EVIDENCE: a proxy. The "
          "faithful proxy is pair-level")
    print("  (entry answers the same question AND its evidence did not change "
          "between caching and reuse).")
    print()
    print("UNOBSERVABLE HANDLING:")
    print("  Chosen: record NO update. Treating it as correct fabricates a "
          "verdict and biases the")
    print("  threshold downward, which favours vCache; removing it is therefore "
          "not a FreshCache-favourable choice.")
    print()
    print("CAUSALITY:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print()
    print("FIXED 24H")
    print(f"Static-threshold variant: saved {mm24['search_saved_pct']:.4f}%  "
          f"drift {ms24['drift_pct']:.4f}%  coverage {ms24['coverage_pct']:.4f}%  "
          f"hits {mm24['l1_hits']:,}")
    d = m24["drift_pct"]
    print(f"Reference-consistent: saved {m24['search_saved_pct']:.4f}%  "
          f"drift {'n/a (no determinable reuse)' if d is None else f'{d:.4f}%'}  "
          f"coverage {m24['coverage_pct']:.4f}%  hits {m24['hits']:,}  "
          f"feedback {m24['feedback_updates']:,}")
    print()
    print("MIXED AGE")
    dm = mM["drift_pct"]
    print(f"Saved: {mM['search_saved_pct']:.4f}%")
    print(f"Drift: {'n/a (no determinable reuse)' if dm is None else f'{dm:.4f}%'}")
    print(f"Coverage: {mM['coverage_pct']:.4f}%")
    print(f"Searches: {mM['search_calls']:,}")
    print(f"Fetches: {mM['fetches']:,}")
    print(f"Generations: {mM['generations']:,}")
    print(f"Hits: {mM['hits']:,}")
    print(f"Feedback updates: {mM['feedback_updates']:,}")
    print(f"Changed / determinable: {mM['changed']:,} / {mM['determinable']:,}")
    print()
    print("COMMON DETERMINATE VS FRESHCACHE")
    if pair and pair["shared_determinable_N"]:
        print(f"N: {pair['shared_determinable_N']:,} "
              f"({pair['shared_workload_pct']:.4f}% of workload)")
        print(f"FreshCache drift: {pair['freshcache_drift_pct']}% "
              f"({pair['freshcache_changed']:,}/{pair['shared_determinable_N']:,})")
        print(f"vCache drift: {pair['vcache_drift_pct']}% "
              f"({pair['vcache_changed']:,}/{pair['shared_determinable_N']:,})")
        print("  No significance test is reported: the two determinable sets "
              "arise from different reuse mechanisms.")
    else:
        print("N: 0 — the reference policy produces too little determinable "
              "reuse for a comparison on this replay.")
    print()
    print("GENERATION ACCOUNTING:")
    print(f"  hit -> no generation, miss -> generation. Reference-consistent "
          f"mixed age: {mM['generations']:,} generations for "
          f"{mM['misses']:,} misses.")
    print("  method_replay and sim_vcache_reference both charge an llm_generate in "
          "their LATENCY model on a hit;")
    print("  counts here are unaffected; latency figures taken from those engines are "
          "pessimistic by one generation per hit.")
    print()
    print("VCACHE-STYLE BASELINE:")
    print(f"  vcache_policy.py sim_vcache_reference at delta = {DELTA}, "
          "pair-level evidence-change label, no update when undeterminable.")
    print("  The static-threshold variant is reported separately "
          "(fixed24_reference_run.json).")
    print()
    print("FEEDBACK DESCRIPTION:")
    print("  \"evidence-change verification feedback\" — per-entry feedback on "
          "whether the cached entry answers the")
    print("  same question and whether its evidence changed between caching and "
          "reuse. Not realized answer correctness.")
    print()
    print("MIXED-AGE ADAPTATION:")
    print("  Because a single fixed-horizon staleness label is undefined when "
          "cached entries have different ages,")
    print("  vCache's verification label is evaluated per (cache time, request "
          "time) pair against the same Web")
    print("  step function used for every other method, leaving the policy, its "
          "delta and its update rule unchanged.")


if __name__ == "__main__":
    main()
