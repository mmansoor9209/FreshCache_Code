#!/usr/bin/env python3
"""
Mixed-age precision consistency analysis.

Stage 2: Parts A, B, C, D and F, computed from the reported traces built by
mixed_age_trace_construction.py. Part E runs separately because it needs a second request
stream.

Methodology is unchanged from the original analyses:

  Part B bootstrap   paired cluster bootstrap, B = 1000, seed 42, one index
                     draw per replicate used for BOTH arms, resampling unit is
                     the query cluster, eligible pool is the common observable
                     support S. Plus the strictly-paired-denominator
                     sensitivity over requests determinate in both arms.

  Part C age bins    request-level binning on the age of the OLDEST cache entry
                     behind the answer, half-open [lo, hi), restricted to
                     determinate requests in S. Bins must partition exactly.

  Part D pairwise    elementwise intersection of two methods' determinate masks
                     over the same request index, no support restriction,
                     workload denominator = all traced requests. Strict 4- and 5-method
                     intersections.

Baselines that the precision fix cannot touch (SCALM, SemanticTTL,
TieredFixedTTL, gate-off) are NOT re-derived from scratch for Part D: their
stored code vectors in outputs/class_predictor/class_code_tasks.jsonl are reused. They
are nevertheless cross-checked against this run's own replay of the same
scripts/baselines, and any mismatch is reported rather than hidden.

Writes primary_metrics.csv, temporal_gate_bootstrap.json, age_bins.csv,
common_determinate.csv, mixed_age_operation_counts.csv and parts_abcdf.json.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import csv; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json
import math
import os
import pathlib
import pickle
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")

ENG = os.environ["RECOMPUTE_ENGINE"]   # reported (the comparison copy is not released)
HERE = (OUTPUT_ROOT / "replay/table2") / ENG; HERE.mkdir(parents=True, exist_ok=True)   # outputs only here
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / {"reported": "freshcache/replay_engine"}[ENG]))   # engine copy first
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import execution_semantics as ce         # noqa: E402
from schedule_robustness import wilson      # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS
WORKLOAD = None   # number of requests in the traced primary stream; set from the trace in main()
H = 3600.0
BINS = [("0-12h", 0.0, 12 * H), ("12-24h", 12 * H, 24 * H),
        ("1-3d", 24 * H, 72 * H), ("3-7d", 72 * H, 168 * H)]
L1_LINE, L3_LINE = {"keyset": (227, 288), "reported": (286, 349)}[ENG]   # outcome() call sites in the engine copies
B_BOOT, BOOT_SEED = 1000, 42
ON, OFF = "FreshCache", "FreshCache_AlwaysPassTemporal"
CODE = {CHANGED: "C", UNCHANGED: "U", UNOBS: "O", None: "-"}
X3 = (OUTPUT_ROOT / "class_predictor/class_code_tasks.jsonl")
PAIR_BASELINES = ["SCALM", "SemanticTTL", "TieredFixedTTL"]
STRICT4 = ["FreshCache", "SCALM", "SemanticTTL", "TieredFixedTTL"]

OPS_ORDER = ["SemanticTTL", "SCALM", "TieredFixedTTL", "L3Only", "L2Only",
             "FreshCache_AlwaysPassTemporal", "EquivalentTTL", "FreshCache"]
OPS_LABEL = {"FreshCache_AlwaysPassTemporal": "FreshCache gate off",
             "SCALM": "SCALM style"}


def load():
    with open(HERE / "traces.pkl", "rb") as fh:
        return pickle.load(fh)


def rebuild_arrival_times(T):
    """Fallback if the trace file predates arrival-time capture.

    build_stream is deterministic in (records, schedule, seed), so this returns
    the same times the traced run used. The reconstruction is verified against
    the stored query-id order before it is trusted.
    """
    import tiered_cache_model as exp
    import request_schedules as sc
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    stream = sc.build_stream(records, T["schedule"], T["seed"])
    assert [r["query_id"] for _, r in stream] == T["qids"], \
        "rebuilt stream does not match the traced request order"
    return [t for t, _ in stream]


# ---------------------------------------------------------------- Part A
def part_a(T):
    rows = []
    for name in ("FreshCache", "EquivalentTTL"):
        m = T["methods"][name]
        pub = T["recorded"].get(name, {})
        rows.append({
            "policy": name, "run": "after_precision_fix",
            "search_saved_pct_full": m["search_saved_pct"],
            "search_saved_pct_manuscript": round(m["search_saved_pct"], 2),
            "drift_pct_full": 100 * m["_drift"],
            "drift_pct_manuscript": round(100 * m["_drift"], 2),
            "coverage_pct_full": 100 * m["_coverage"],
            "coverage_pct_manuscript": round(100 * m["_coverage"], 2),
            "searches": m["search_calls"], "fetches": m["fetches"],
            "conditional_get_validations": m["conditional_gets"],
            "generations": m["generations"],
            "l1_serves": m["l1_hits"], "l2_serves": m["l2_hits"],
            "l3_serves": m["l3_hits"],
            "determinable": m["_determinate"], "changed": m["_changed"],
            "unchanged": m["_unchanged"],
            "support_size": len(T["support"]),
        })
        if pub:
            rows.append({
                "policy": name, "run": "before_precision_fix",
                "search_saved_pct_full": pub.get("search_saved_pct"),
                "search_saved_pct_manuscript": round(pub["search_saved_pct"], 2),
                "drift_pct_full": 100 * pub["drift_common_support"],
                "drift_pct_manuscript": round(100 * pub["drift_common_support"], 2),
                "coverage_pct_full": 100 * pub["common_support_coverage"],
                "coverage_pct_manuscript": round(100 * pub["common_support_coverage"], 2),
                "searches": pub.get("search_calls"), "fetches": pub.get("fetches"),
                "conditional_get_validations": pub.get("conditional_gets", 0),
                "generations": pub.get("generations"),
                "l1_serves": pub.get("l1_hits"), "l2_serves": pub.get("l2_hits"),
                "l3_serves": pub.get("l3_hits"),
                "determinable": pub.get("observable_hits_in_support"),
                "changed": pub.get("changed_in_support"),
                "unchanged": (pub.get("observable_hits_in_support", 0)
                              - pub.get("changed_in_support", 0)),
                "support_size": pub.get("support_size"),
            })
    with open(HERE / "primary_metrics.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    a, b = T["methods"]["FreshCache"], T["methods"]["EquivalentTTL"]
    fields = ("search_saved_pct", "search_calls", "fetches", "generations",
              "l1_hits", "l2_hits", "l3_hits", "conditional_gets",
              "_changed", "_determinate", "_drift", "_coverage")
    identical = {f: (a[f] == b[f]) for f in fields}
    sig_div = sum(1 for x, y in zip(a["_outcome_sig"], b["_outcome_sig"])
                  if x != y)
    return {"rows": rows,
            "freshcache_vs_equivttl_identical": identical,
            "all_identical": all(identical.values()),
            "per_request_outcome_divergences": sig_div}


# ---------------------------------------------------------------- Part B
def part_b(T):
    S, qids, clusters = T["support"], T["qids"], T["clusters"]
    on_sig = T["methods"][ON]["_outcome_sig"]
    off_sig = T["methods"][OFF]["_outcome_sig"]

    per = defaultdict(lambda: [0, 0, 0, 0, 0])
    for qid, cl, a, b in zip(qids, clusters, on_sig, off_sig):
        if qid not in S:
            continue
        v = per[cl]
        v[4] += 1
        if a in (CHANGED, UNCHANGED):
            v[1] += 1
            v[0] += (a == CHANGED)
        if b in (CHANGED, UNCHANGED):
            v[3] += 1
            v[2] += (b == CHANGED)
    cl_list = sorted(per)
    C = len(cl_list)
    M = np.array([per[c] for c in cl_list], dtype=np.int64)
    on_ch, on_det, off_ch, off_det, nreq = (M[:, i] for i in range(5))

    d_on = on_ch.sum() / on_det.sum()
    d_off = off_ch.sum() / off_det.sum()

    rng = np.random.RandomState(BOOT_SEED)
    idx = rng.randint(0, C, size=(B_BOOT, C))
    b_on = on_ch[idx].sum(axis=1) / on_det[idx].sum(axis=1)
    b_off = off_ch[idx].sum(axis=1) / off_det[idx].sum(axis=1)
    diff = b_off - b_on
    lo, hi = np.percentile(diff, [2.5, 97.5])
    n_pos = int((diff > 0).sum())

    # strictly paired denominator
    per2 = defaultdict(lambda: [0, 0, 0])
    for qid, cl, a, b in zip(qids, clusters, on_sig, off_sig):
        if qid not in S:
            continue
        if a in (CHANGED, UNCHANGED) and b in (CHANGED, UNCHANGED):
            v = per2[cl]
            v[0] += (a == CHANGED)
            v[1] += (b == CHANGED)
            v[2] += 1
    cl2 = sorted(per2)
    M2 = np.array([per2[c] for c in cl2], dtype=np.int64)
    C2, N2 = len(cl2), int(M2[:, 2].sum())
    rng2 = np.random.RandomState(BOOT_SEED)
    idx2 = rng2.randint(0, C2, size=(B_BOOT, C2))
    n2 = M2[:, 2][idx2].sum(axis=1).astype(np.float64)
    d2 = M2[:, 1][idx2].sum(axis=1) / n2 - M2[:, 0][idx2].sum(axis=1) / n2
    lo2, hi2 = np.percentile(d2, [2.5, 97.5])

    mon, moff = T["methods"][ON], T["methods"][OFF]
    out = {
        "schedule": T["schedule"], "seed": T["seed"], "replicates": B_BOOT,
        "bootstrap_seed": BOOT_SEED, "paired": True,
        "resampling_unit": "query cluster (cluster_id, fallback query_id)",
        "support_size": len(S), "workload": T["n_requests"],
        "clusters": C, "requests_in_support": int(nreq.sum()),
        "gate_on": {"variant": ON, "saved_pct": mon["search_saved_pct"],
                    "drift_pct": 100 * d_on,
                    "coverage_pct": 100 * mon["_coverage"],
                    "fetches": mon["fetches"],
                    "changed": int(on_ch.sum()),
                    "determinate": int(on_det.sum())},
        "gate_off": {"variant": OFF, "saved_pct": moff["search_saved_pct"],
                     "drift_pct": 100 * d_off,
                     "coverage_pct": 100 * moff["_coverage"],
                     "fetches": moff["fetches"],
                     "changed": int(off_ch.sum()),
                     "determinate": int(off_det.sum())},
        "point": {"gate_on_drift": float(d_on), "gate_off_drift": float(d_off),
                  "difference_pp": 100 * float(d_off - d_on)},
        "bootstrap": {
            "mean_difference_pp": 100 * float(diff.mean()),
            "std_difference_pp": 100 * float(diff.std(ddof=1)),
            "ci95_difference_pp": [100 * float(lo), 100 * float(hi)],
            "replicates_off_gt_on": n_pos,
            "pct_replicates_off_gt_on": 100 * n_pos / B_BOOT},
        "sensitivity_strict_paired_denominator": {
            "clusters": C2, "requests": N2,
            "gate_on_drift_pct": 100 * float(M2[:, 0].sum() / N2),
            "gate_off_drift_pct": 100 * float(M2[:, 1].sum() / N2),
            "difference_pp": 100 * float((M2[:, 1].sum() - M2[:, 0].sum()) / N2),
            "ci95_difference_pp": [100 * float(lo2), 100 * float(hi2)],
            "replicates_off_gt_on": int((d2 > 0).sum())},
        "gate_off_unchanged_by_fix": (
            T["methods"][OFF]["_vs_recorded"] is None or
            all(v["same"] for v in T["methods"][OFF]["_vs_recorded"].values()
                if v["recorded"] is not None)),
    }
    json.dump(out, open(HERE / "temporal_gate_bootstrap.json", "w"), indent=2)
    return out


# ---------------------------------------------------------------- Part C
def part_c(T):
    S, qids = T["support"], T["qids"]
    # arrival time -> stream index
    times = T.get("arrival_times")
    if times is None:
        times = rebuild_arrival_times(T)
    t2i = {t: i for i, t in enumerate(times)}
    assert len(t2i) == len(times), "arrival times not unique"
    inS = [q in S for q in qids]

    rows, agg = [], {}
    for arm, variant in (("gate_on", ON), ("gate_off", OFF)):
        evs = T["age_events"][variant]
        lines = {e[0] for e in evs}
        assert lines <= {L1_LINE, L3_LINE}, \
            f"unexpected outcome() call sites {lines}; mixed_age_replay.py moved"
        maxage = defaultdict(float)
        for ln, uh, tc, tr, lab in evs:
            i = t2i[tr]
            maxage[i] = max(maxage[i], tr - tc)
        sig = T["methods"][variant]["_outcome_sig"]
        rows_arm = [(i, maxage[i], o) for i, o in enumerate(sig)
                    if inS[i] and o in (CHANGED, UNCHANGED)]
        tot_det = len(rows_arm)
        tot_ch = sum(1 for _, _, o in rows_arm if o == CHANGED)
        acc_d = acc_c = 0
        for name, lo, hi in BINS:
            sub = [x for x in rows_arm if lo <= x[1] < hi]
            d = len(sub)
            c = sum(1 for _, _, o in sub if o == CHANGED)
            acc_d += d
            acc_c += c
            wl, wh = wilson(c, d)
            rows.append({"arm": arm, "age_bin": name, "determinable_n": d,
                         "changed_n": c,
                         "drift_pct": (100 * c / d) if d else None,
                         "ci95_lo_pct": (100 * wl) if d else None,
                         "ci95_hi_pct": (100 * wh) if d else None,
                         "pct_of_determinable": (100 * d / tot_det)
                         if tot_det else None})
        outside = [x for x in rows_arm if not (0 <= x[1] < 168 * H)]
        rows.append({"arm": arm, "age_bin": "SUM_OF_BINS",
                     "determinable_n": acc_d, "changed_n": acc_c,
                     "drift_pct": (100 * acc_c / acc_d) if acc_d else None,
                     "ci95_lo_pct": None, "ci95_hi_pct": None,
                     "pct_of_determinable": 100.0})
        rows.append({"arm": arm, "age_bin": "AGGREGATE",
                     "determinable_n": tot_det, "changed_n": tot_ch,
                     "drift_pct": (100 * tot_ch / tot_det) if tot_det else None,
                     "ci95_lo_pct": None, "ci95_hi_pct": None,
                     "pct_of_determinable": 100.0})
        agg[arm] = {"changed": tot_ch, "determinate": tot_det,
                    "drift_pct": 100 * tot_ch / tot_det,
                    "bins_sum_changed": acc_c, "bins_sum_determinate": acc_d,
                    "reconciles_exactly": (acc_d == tot_det and acc_c == tot_ch),
                    "events_outside_0_168h": len(outside)}
        # cross-check against the arm's own aggregate from the trace
        m = T["methods"][variant]
        agg[arm]["matches_trace_aggregate"] = (
            tot_det == m["_determinate"] and tot_ch == m["_changed"])

    with open(HERE / "age_bins.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "age_bin", "determinable_n",
                                           "changed_n", "drift_pct",
                                           "ci95_lo_pct", "ci95_hi_pct",
                                           "pct_of_determinable"])
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows, "aggregate": agg,
            "paired_difference_pp": agg["gate_off"]["drift_pct"]
            - agg["gate_on"]["drift_pct"]}


# ---------------------------------------------------------------- Part D
def pair_stats(a_codes, b_codes):
    """Elementwise intersection of two determinate masks (pairwise-determinate statistics)."""
    A = np.array(a_codes, dtype=object)
    Bv = np.array(b_codes, dtype=object)
    det = np.array([(x in (CHANGED, UNCHANGED)) and (y in (CHANGED, UNCHANGED))
                    for x, y in zip(A, Bv)])
    idx = np.nonzero(det)[0]
    return (idx, np.array([A[i] == CHANGED for i in idx]),
            np.array([Bv[i] == CHANGED for i in idx]))


def part_d(T):
    # stored baseline code vectors, reused rather than re-derived
    x3 = {}
    if X3.exists():
        with open(X3, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("schedule") == T["schedule"] and r.get("seed") == T["seed"]:
                    x3[r["method"]] = r["codes"]

    fc = T["methods"]["FreshCache"]["_outcome_sig"]
    l3 = T["methods"]["L3Only"]["_outcome_sig"]
    fc_str = "".join(CODE[o] for o in fc)

    # consistency: do the stored baseline codes match this run's own replay?
    baseline_check = {}
    for m in PAIR_BASELINES:
        if m in x3 and m in T["methods"]:
            mine = "".join(CODE[o] for o in T["methods"][m]["_outcome_sig"])
            baseline_check[m] = {
                "stored_matches_rerun": mine == x3[m],
                "n_differing_positions": sum(1 for a, b in zip(mine, x3[m])
                                             if a != b)}
    fc_changed_vs_stored = (None if "FreshCache" not in x3 else
                            sum(1 for a, b in zip(fc_str, x3["FreshCache"])
                                if a != b))

    def decode(s):
        inv = {"C": CHANGED, "U": UNCHANGED, "O": UNOBS, "-": None}
        return [inv[c] for c in s]

    rows, pairs = [], {}
    for b in PAIR_BASELINES + ["L3Only"]:
        other = l3 if b == "L3Only" else decode(x3[b])
        idx, ach, bch = pair_stats(fc, other)
        N = len(idx)
        dfc, dbl = float(ach.mean()), float(bch.mean())
        flo, fhi = wilson(int(ach.sum()), N)
        blo, bhi = wilson(int(bch.sum()), N)
        pairs[b] = {
            "shared_determinate_N": N,
            "pct_of_workload": 100 * N / WORKLOAD,
            "freshcache_changed": int(ach.sum()),
            "freshcache_drift_pct": 100 * dfc,
            "freshcache_wilson95_pct": [100 * flo, 100 * fhi],
            "baseline_changed": int(bch.sum()),
            "baseline_drift_pct": 100 * dbl,
            "baseline_wilson95_pct": [100 * blo, 100 * bhi],
            "diff_pp": 100 * (dfc - dbl),
            "source_of_baseline_codes": ("rerun (affected by the fix)"
                                         if b == "L3Only"
                                         else "stored class_code_tasks.jsonl (unaffected)")}
        rows.append({"analysis": "pairwise", "pair": f"FreshCache vs {b}",
                     "subset": "no support restriction",
                     "shared_N": N,
                     "pct_of_workload": round(100 * N / WORKLOAD, 4),
                     "freshcache_changed": int(ach.sum()),
                     "freshcache_drift_pct": round(100 * dfc, 4),
                     "baseline_changed": int(bch.sum()),
                     "baseline_drift_pct": round(100 * dbl, 4),
                     "diff_pp": round(100 * (dfc - dbl), 4)})

    # strict intersections
    strict = {}
    if all(m in x3 or m == "FreshCache" for m in STRICT4):
        codes4 = {m: (fc_str if m == "FreshCache" else x3[m]) for m in STRICT4}
        det4 = [all(codes4[m][i] in "CU" for m in STRICT4)
                for i in range(WORKLOAD)]
        det5 = [det4[i] and (l3[i] in (CHANGED, UNCHANGED))
                for i in range(WORKLOAD)]
        for tag, det in (("four methods", det4),
                         ("five methods (incl. L3Only)", det5)):
            ii = [i for i, x in enumerate(det) if x]
            rec = {"subset": tag, "N": len(ii),
                   "pct_of_workload": 100 * len(ii) / WORKLOAD}
            row = {"analysis": "strict_intersection", "pair": tag,
                   "subset": tag, "shared_N": len(ii),
                   "pct_of_workload": round(100 * len(ii) / WORKLOAD, 4)}
            for m in STRICT4:
                ch = sum(1 for i in ii if codes4[m][i] == "C")
                rec[f"{m}_changed"] = ch
                rec[f"{m}_drift_pct"] = 100 * ch / len(ii) if ii else None
                row[f"{m}_changed"] = ch
                row[f"{m}_drift_pct"] = round(100 * ch / len(ii), 4) if ii else None
            if tag.startswith("five"):
                ch = sum(1 for i in ii if l3[i] == CHANGED)
                rec["L3Only_changed"] = ch
                rec["L3Only_drift_pct"] = 100 * ch / len(ii) if ii else None
                row["L3Only_changed"] = ch
                row["L3Only_drift_pct"] = round(100 * ch / len(ii), 4) if ii else None
            strict[tag] = rec
            rows.append(row)

    cols = (["analysis", "pair", "subset", "shared_N", "pct_of_workload",
             "freshcache_changed", "freshcache_drift_pct", "baseline_changed",
             "baseline_drift_pct", "diff_pp"]
            + [f"{m}_changed" for m in STRICT4]
            + [f"{m}_drift_pct" for m in STRICT4]
            + ["L3Only_changed", "L3Only_drift_pct"])
    with open(HERE / "common_determinate.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return {"pairs": pairs, "strict": strict,
            "baseline_code_consistency": baseline_check,
            "freshcache_codes_differing_from_stored": fc_changed_vs_stored}


# ---------------------------------------------------------------- Part F
def part_f(T):
    n = T["n_requests"]
    rows = []
    for name in OPS_ORDER:
        m = T["methods"][name]
        label = OPS_LABEL.get(name, name)
        # L2Only reuses URL lists only and performs no L3 content reuse, so a
        # content-drift figure does not exist for it.
        no_content_reuse = (m["l3_hits"] == 0 and m["_determinate"] == 0)
        rows.append({
            "method": label,
            "search_saved_pct": round(m["search_saved_pct"], 4),
            "content_drift_pct": ("N/A" if no_content_reuse
                                  else round(100 * m["_drift"], 4)),
            "observable_coverage_pct": ("N/A" if no_content_reuse
                                        else round(100 * m["_coverage"], 4)),
            "total_searches": m["search_calls"],
            "total_page_fetches": m["fetches"],
            "total_llm_generations": m["generations"],
            "conditional_get_validations": m["conditional_gets"],
            "fetches_per_1000_requests": round(1000 * m["fetches"] / n, 2),
            "generations_per_1000_requests": round(1000 * m["generations"] / n, 2),
            "validations_per_1000_requests": round(1000 * m["conditional_gets"] / n, 2),
            "changed": ("N/A" if no_content_reuse else m["_changed"]),
            "determinable": ("N/A" if no_content_reuse else m["_determinate"]),
            "affected_by_precision_fix": name not in (
                "EquivalentTTL", "FreshCache_AlwaysPassTemporal", "SCALM",
                "SemanticTTL", "TieredFixedTTL"),
        })
    with open(HERE / "mixed_age_operation_counts.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------- driver
def main():
    global WORKLOAD
    T = load(); WORKLOAD = len(T["qids"])
    A = part_a(T)
    Bb = part_b(T)
    C = part_c(T)
    D = part_d(T)
    F = part_f(T)

    E = None
    pe = HERE / "part_e.json"
    if pe.exists():
        E = json.load(open(pe, encoding="utf-8"))

    out = {"part_a": A, "part_b": Bb, "part_c": C, "part_d": D, "part_f": F,
           "part_e_present": E is not None}
    json.dump(out, open(HERE / "parts_abcdf.json", "w"), indent=2, default=str)

    fc = T["methods"]["FreshCache"]
    eq = T["methods"]["EquivalentTTL"]
    pub = T["recorded"].get("FreshCache", {})

    print()
    print("REPORTED FRESHCACHE")
    print(f"Saved: {fc['search_saved_pct']:.4f}%  (manuscript {fc['search_saved_pct']:.2f}%)")
    print(f"Drift: {100*fc['_drift']:.4f}%  (manuscript {100*fc['_drift']:.2f}%)")
    print(f"Coverage: {100*fc['_coverage']:.4f}%  (manuscript {100*fc['_coverage']:.2f}%)")
    print(f"Searches: {fc['search_calls']:,}")
    print(f"Fetches: {fc['fetches']:,}")
    print(f"Generations: {fc['generations']:,}")
    print(f"Validations: {fc['conditional_gets']:,}")
    print(f"L1 serves: {fc['l1_hits']:,}")
    print(f"L2 serves: {fc['l2_hits']:,}")
    print(f"Changed / determinable: {fc['_changed']:,} / {fc['_determinate']:,}")
    print()
    print("EQUIVALENT TTL")
    print(f"Saved: {eq['search_saved_pct']:.4f}%")
    print(f"Drift: {100*eq['_drift']:.4f}%")
    print(f"Coverage: {100*eq['_coverage']:.4f}%")
    print(f"Searches: {eq['search_calls']:,}")
    print(f"Fetches: {eq['fetches']:,}")
    print(f"Decision divergences: {A['per_request_outcome_divergences']} "
          f"(all metrics identical: {A['all_identical']})")
    print()
    print("GATE ABLATION")
    g_on, g_off = Bb["gate_on"], Bb["gate_off"]
    print(f"Gate on: saved {g_on['saved_pct']:.4f}%  drift {g_on['drift_pct']:.4f}%  "
          f"coverage {g_on['coverage_pct']:.4f}%  fetches {g_on['fetches']:,}")
    print(f"Gate off: saved {g_off['saved_pct']:.4f}%  drift {g_off['drift_pct']:.4f}%  "
          f"coverage {g_off['coverage_pct']:.4f}%  fetches {g_off['fetches']:,}")
    bt = Bb["bootstrap"]
    print(f"Bootstrap difference: {Bb['point']['difference_pp']:+.4f} pp "
          f"(bootstrap mean {bt['mean_difference_pp']:+.4f} pp)")
    print(f"95% CI: [{bt['ci95_difference_pp'][0]:+.4f}, "
          f"{bt['ci95_difference_pp'][1]:+.4f}] pp   positive replicates "
          f"{bt['replicates_off_gt_on']:,}/{B_BOOT:,}")
    st = Bb["sensitivity_strict_paired_denominator"]
    print(f"Same-determinable difference: {st['difference_pp']:+.4f} pp "
          f"(N = {st['requests']:,})")
    print(f"95% CI: [{st['ci95_difference_pp'][0]:+.4f}, "
          f"{st['ci95_difference_pp'][1]:+.4f}] pp   positive replicates "
          f"{st['replicates_off_gt_on']:,}/{B_BOOT:,}")
    print()
    print("AGE BINS")
    byb = {(r["arm"], r["age_bin"]): r for r in C["rows"]}
    for nm in ("0-12h", "12-24h", "1-3d", "3-7d"):
        on = byb[("gate_on", nm)]
        off = byb[("gate_off", nm)]
        label = "<12h" if nm == "0-12h" else nm
        print(f"{label}: gate-on {on['changed_n']:,}/{on['determinable_n']:,} = "
              f"{on['drift_pct']:.4f}%   gate-off {off['changed_n']:,}/"
              f"{off['determinable_n']:,} = {off['drift_pct']:.4f}%")
    ag = C["aggregate"]
    ok = all(ag[a]["reconciles_exactly"] and ag[a]["matches_trace_aggregate"]
             for a in ("gate_on", "gate_off"))
    print(f"Aggregate check: bins sum exactly to the reported aggregate in "
          f"both arms: {'PASS' if ok else 'FAIL'}  "
          f"(gate-on {ag['gate_on']['bins_sum_changed']:,}/"
          f"{ag['gate_on']['bins_sum_determinate']:,} vs "
          f"{ag['gate_on']['changed']:,}/{ag['gate_on']['determinate']:,}; "
          f"gate-off {ag['gate_off']['bins_sum_changed']:,}/"
          f"{ag['gate_off']['bins_sum_determinate']:,} vs "
          f"{ag['gate_off']['changed']:,}/{ag['gate_off']['determinate']:,})")
    print()
    print("COMMON DETERMINATE")
    for b, label in (("SCALM", "SCALM"), ("SemanticTTL", "SemanticTTL"),
                     ("TieredFixedTTL", "TieredFixedTTL"), ("L3Only", "L3Only")):
        p = D["pairs"][b]
        print(f"{label}: N {p['shared_determinate_N']:,} "
              f"({p['pct_of_workload']:.2f}% of workload)   "
              f"FreshCache {p['freshcache_drift_pct']:.4f}%   "
              f"{label} {p['baseline_drift_pct']:.4f}%   "
              f"diff {p['diff_pp']:+.4f} pp")
    for tag, label in (("four methods", "4-method intersection"),
                       ("five methods (incl. L3Only)", "5-method intersection")):
        s = D["strict"].get(tag)
        if not s:
            continue
        parts = "  ".join(f"{m} {s[f'{m}_drift_pct']:.4f}%" for m in STRICT4)
        extra = (f"  L3Only {s['L3Only_drift_pct']:.4f}%"
                 if "L3Only_drift_pct" in s else "")
        print(f"{label}: N {s['N']:,} ({s['pct_of_workload']:.2f}%)   "
              f"{parts}{extra}")
    print()
    print("PREDICTED CLASS")
    if E:
        for arm, label in (("oracle_dataset_classes", "Collection labels"),
                           ("predicted_classes", "Predicted labels")):
            a = E["arms"][arm]
            d = a["drift_fixed_support"]
            print(f"{label}: saved {a['search_saved_pct']:.4f}%  "
                  f"drift {100*d['drift']:.4f}%  "
                  f"coverage {100*d['coverage']:.4f}%  "
                  f"fetches {a['fetches']:,}")
    else:
        print("Collection labels: not run")
        print("Predicted labels: not run")
    print()
    print("MIXED-AGE OPERATION COUNTS")
    for r in F:
        drift = r["content_drift_pct"]
        drift_s = drift if drift == "N/A" else f"{drift:.4f}%"
        cov = r["observable_coverage_pct"]
        cov_s = cov if cov == "N/A" else f"{cov:.4f}%"
        print(f"{r['method']}: saved {r['search_saved_pct']:.4f}%  "
              f"drift {drift_s}  coverage {cov_s}  "
              f"searches {r['total_searches']:,}  "
              f"fetches {r['total_page_fetches']:,}  "
              f"generations {r['total_llm_generations']:,}  "
              f"validations {r['conditional_get_validations']:,}  "
              f"[per 1k: fetch {r['fetches_per_1000_requests']:.1f}, "
              f"gen {r['generations_per_1000_requests']:.1f}, "
              f"val {r['validations_per_1000_requests']:.1f}]")
    print()
    print("CONSISTENCY CHECK:")
    changes = []
    if pub:
        def add(name, old, new, fmt="{:.4f}"):
            if old is None or new is None:
                return
            if isinstance(old, float) or isinstance(new, float):
                same = abs(float(old) - float(new)) < 1e-9
            else:
                same = old == new
            if not same:
                changes.append(f"  {name}: {fmt.format(old)} -> {fmt.format(new)}")
        add("FreshCache Saved %", pub.get("search_saved_pct"), fc["search_saved_pct"])
        add("FreshCache Drift %", 100*pub.get("drift_common_support", 0),
            100*fc["_drift"])
        add("FreshCache Coverage %", 100*pub.get("common_support_coverage", 0),
            100*fc["_coverage"])
        add("FreshCache searches", pub.get("search_calls"), fc["search_calls"], "{:,}")
        add("FreshCache fetches", pub.get("fetches"), fc["fetches"], "{:,}")
        add("FreshCache generations", pub.get("generations"), fc["generations"], "{:,}")
        add("FreshCache L1 serves", pub.get("l1_hits"), fc["l1_hits"], "{:,}")
        add("FreshCache L2 serves", pub.get("l2_hits"), fc["l2_hits"], "{:,}")
        add("FreshCache L3 serves", pub.get("l3_hits"), fc["l3_hits"], "{:,}")
        add("FreshCache changed", pub.get("changed_in_support"), fc["_changed"], "{:,}")
        add("FreshCache determinable", pub.get("observable_hits_in_support"),
            fc["_determinate"], "{:,}")
    for name in OPS_ORDER:
        c = T["methods"][name]["_vs_recorded"]
        if not c:
            continue
        moved = [k for k, v in c.items()
                 if v["recorded"] is not None and not v["same"]]
        if moved and name != "FreshCache":
            changes.append(f"  {OPS_LABEL.get(name, name)}: {', '.join(moved)}")
    if changes:
        for c in sorted(set(changes)):
            print(c)
    else:
        print("  no manuscript value changes")


if __name__ == "__main__":
    main()
