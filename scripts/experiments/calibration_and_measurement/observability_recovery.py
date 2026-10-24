#!/usr/bin/env python3
"""
Observability recovery: recover what can legitimately be recovered, then
recompute coverage, paired drift, cluster-bootstrap CIs and the worst-case
bound on the expanded observable population, and test whether the
full-population effect is identified.

The ONLY recovery available is a threshold decision on text that WAS captured
at the correct historical time: the minimum-body-length threshold that marks an
observation as substantive is lowered from 400 to 300, 200, 100 and 50. No page
is re-fetched, no current page is substituted for a historical one, no content
is fabricated, and no access restriction is touched. The block-page filter is
retained at every threshold.

Routing is unaffected by the substantive threshold -- the policy never sees it
-- so the two replays (gate on, gate off) are run once and every threshold is
scored from the same per-request reused-URL lists. Threshold 400 must reproduce
the bound in outputs/observability_bounds/bounds_overall.json (written by
scripts/experiments/calibration_and_measurement/observability_bounds.py).

Inputs: DATA_DIR/derived/round_table.jsonl, DATA_DIR/snapshots/,
outputs/validation_split/. Outputs: recovery_results.json and
per_request_recovered.csv in outputs/observability_resolution/recovery/.
Stream zipf_uniform, seed 42, 10,000 bootstrap resamples.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, json, math, os, pathlib, random, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = ((OUTPUT_ROOT / "observability_resolution/recovery"))
OR = ((OUTPUT_ROOT / "observability_resolution"))
PRIOR = ((OUTPUT_ROOT / "archived_runs"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
OB = ((OUTPUT_ROOT / "observability_bounds"))
SNAP = (DATA_ROOT / "snapshots")
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import tiered_cache_model as exp                 # noqa: E402
import method_replay as ea                  # noqa: E402
import mixed_age_replay as me                # noqa: E402
import request_schedules as sc                   # noqa: E402
import page_version_replay as ma                # noqa: E402
import audit_replay as p2                       # noqa: E402

CHANGED, UNCHANGED, UNOBS = me.CHANGED, me.UNCHANGED, me.UNOBS
SEED, B = 42, 10_000
SCHEDULE = "zipf_uniform"
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|"
                   r"access denied|are you a robot|captcha|cloudflare|"
                   r"403 forbidden|404 not found|page not found", re.I)
THRESHOLDS = [400, 300, 200, 100, 50]
ANCHOR = ("search_saved_pct", "l1_hits", "l2_hits"); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # run-time reference values (outputs/reference_values.json)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 4),
            round(100 * min(1.0, (c + m) / d), 4))


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("TASKS 3-5 -- legitimate recovery and recomputation")

    # ---- load the round table and the snapshot texts of SHORT observations ----
    base = {}
    with open((DATA_ROOT / "derived/round_table.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            base[d["url_hash"]] = d["rounds"]
    short = [(u, r) for u, rr in base.items() for r, e in rr.items()
             if not e.get("substantive") and e.get("status_code") == 200
             and 0 < e.get("n_chars", 0) < 400]
    say(f"  short-text observations to re-examine: {len(short):,}")
    blocked = 0
    texts = {}
    for u, r in short:
        p = SNAP / r / f"{u}.json"
        if not p.exists():
            continue
        try:
            t = (json.load(open(p, encoding="utf-8")).get("extracted_text") or "")
        except Exception:
            continue
        if BLOCK.search(t[:400]):
            blocked += 1
            continue
        texts[(u, r)] = t
    say(f"  re-read from local snapshots: {len(texts):,}; "
        f"rejected as block pages: {blocked:,}")
    say(f"  NOTE these bodies were captured at the correct historical time; "
        f"nothing is re-fetched and nothing is substituted.")

    def rounds_at(thr):
        """A round table whose `substantive` flag uses threshold `thr`.
        The original table is never modified."""
        out = {}
        for u, rr in base.items():
            out[u] = {}
            for r, e in rr.items():
                s = bool(e.get("substantive"))
                if not s and (u, r) in texts and e.get("n_chars", 0) >= thr:
                    s = True
                out[u][r] = {**e, "substantive": s}
        return out

    # ---- stream (the ORIGINAL held-out scripts/evaluation population) ----
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    dom = {}
    for m in manifest:
        if m.get("url_hash") and m.get("domain"):
            dom.setdefault(m["url_hash"], m["domain"])
    split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"), encoding="utf-8"))
    test_c = set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(stream) == split["test"]["requests"]
    say(f"  held-out stream {len(stream):,} requests "
        f"(the ORIGINAL scripts/evaluation population, unchanged)")
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    fco = {r["query_id"]: r["freshness_class"] for r in records}

    def arms(rnds):
        """The scripts/evaluation path used for the reported results: mixed_age_replay.replay
        decides routing AND emits the per-request drift outcome from the round table."""
        m_on, per_on = me.replay(stream, rnds, "FreshCache", rich=rich)
        bad = [k for k, v in {k: REF.value("selection_engine/heldout/FreshCache/" + k) for k in ANCHOR}.items()
               if abs(m_on[k] - v) > (1e-4 if isinstance(v, float) else 0)]
        assert not bad, f"anchor gate failed on {bad}: {m_on}"
        m_off, per_off = me.replay(stream, rnds,
                                   "FreshCache_AlwaysPassTemporal", rich=rich)
        return dict(per_on), dict(per_off)

    # ---- verify threshold 400 reproduces the recorded bound exactly ----
    ON0, OFF0 = arms(rounds_at(400))
    say("  ANCHOR GATE PASSED at the recorded threshold")
    prev = json.load(open(OB / "bounds_overall.json", encoding="utf-8"))["overall"]

    def scoreset(ON, OFF):
        pr = [q for q in ON if ON[q] is not None and OFF.get(q) is not None]
        N = len(pr)
        c_on = sum(1 for q in pr if ON[q] == CHANGED)
        c_off = sum(1 for q in pr if OFF[q] == CHANGED)
        u_on = sum(1 for q in pr if ON[q] == UNOBS)
        u_off = sum(1 for q in pr if OFF[q] == UNOBS)
        jo = [q for q in pr if ON[q] != UNOBS and OFF[q] != UNOBS]
        con = sum(1 for q in jo if ON[q] == CHANGED)
        coff = sum(1 for q in jo if OFF[q] == CHANGED)
        d = (100*coff/len(jo) - 100*con/len(jo)) if jo else None
        byc = defaultdict(list)
        for q in jo:
            byc[cid[q]].append(q)
        ks = list(byc); rng = random.Random(SEED); ds = []
        for _ in range(B):
            s = [byc[ks[rng.randrange(len(ks))]] for _ in range(len(ks))]
            fl = [x for g in s for x in g]
            if fl:
                ds.append(100*sum(1 for x in fl if OFF[x] == CHANGED)/len(fl)
                          - 100*sum(1 for x in fl if ON[x] == CHANGED)/len(fl))
        ds.sort()
        ci = ([round(ds[int(.025*len(ds))], 4), round(ds[int(.975*len(ds))], 4)]
              if ds else None)
        return {"N_paired": N, "jointly_observable": len(jo),
                "coverage_pct": round(100*len(jo)/N, 4) if N else None,
                "changed_on": c_on, "changed_off": c_off,
                "unobs_on": u_on, "unobs_off": u_off,
                "drift_on_pct": round(100*con/len(jo), 4) if jo else None,
                "drift_off_pct": round(100*coff/len(jo), 4) if jo else None,
                "observed_delta_pp": round(d, 4) if d is not None else None,
                "bootstrap_ci95": ci, "clusters": len(ks),
                "delta_min_pp": round(100*(c_off-(c_on+u_on))/N, 4) if N else None,
                "delta_max_pp": round(100*((c_off+u_off)-c_on)/N, 4) if N else None}
    s400 = scoreset(ON0, OFF0)
    ok = (abs(s400["delta_min_pp"] - prev["delta_min_pp"]) < 0.01
          and abs(s400["delta_max_pp"] - prev["delta_max_pp"]) < 0.01
          and abs(s400["observed_delta_pp"] - prev["observed_delta_pp"]) < 0.01)
    say(f"\n  REPRODUCTION GATE at the original threshold (400): {ok}")
    say(f"    observed {s400['observed_delta_pp']:+.4f} pp (recorded "
        f"{prev['observed_delta_pp']:+.4f}); bound "
        f"[{s400['delta_min_pp']:+.4f}, {s400['delta_max_pp']:+.4f}] "
        f"(recorded [{prev['delta_min_pp']:+.4f}, {prev['delta_max_pp']:+.4f}])")
    if not ok:
        say("    ABORT: the recomputation path does not reproduce the "
            "recorded bound; no recovery result would be trustworthy.")
        sys.exit(2)

    # ---- sweep the threshold ----
    say(f"\n  == TASK 4: coverage, paired drift and bounds vs the "
        f"substantive threshold ==")
    say(f"    (400 = the ORIGINAL recorded definition; lower thresholds admit "
        f"shorter historical bodies)")
    say(f"    {'MIN_BODY':>9}{'jointly obs':>13}{'coverage':>10}"
        f"{'drift ON':>10}{'drift OFF':>11}{'observed':>10}"
        f"{'bootstrap 95% CI':>24}{'Dmin':>10}{'Dmax':>10}")
    res, cache = {}, {}
    for thr in THRESHOLDS:
        ON_t, OFF_t = ((ON0, OFF0) if thr == 400 else arms(rounds_at(thr)))
        cache[thr] = (ON_t, OFF_t)
        s = scoreset(ON_t, OFF_t)
        res[thr] = s
        say(f"    {thr:>9}{s['jointly_observable']:>13,}"
            f"{s['coverage_pct']:>9.2f}%{s['drift_on_pct']:>9.4f}%"
            f"{s['drift_off_pct']:>10.4f}%{s['observed_delta_pp']:>+9.4f}"
            f"{str(s['bootstrap_ci95']):>24}{s['delta_min_pp']:>+10.3f}"
            f"{s['delta_max_pp']:>+10.3f}")

    # ---- is the full-population effect identifiable? ----
    say(f"\n  == TASK 5: is the full-population effect identified? ==")
    best = min(THRESHOLDS)
    sb = res[best]
    say(f"    at the most permissive threshold examined (MIN_BODY={best}):")
    say(f"      coverage {sb['coverage_pct']:.2f}% "
        f"(was {res[400]['coverage_pct']:.2f}%)")
    say(f"      unobservable still: ON {sb['unobs_on']:,}, "
        f"OFF {sb['unobs_off']:,} of {sb['N_paired']:,} paired requests")
    say(f"      worst-case bound [{sb['delta_min_pp']:+.4f}, "
        f"{sb['delta_max_pp']:+.4f}] pp")
    ident = sb["delta_min_pp"] > 0
    say(f"      Delta_min > 0 -> full-population effect IDENTIFIED: {ident}")
    if not ident:
        need = sb["unobs_on"]
        c_off, c_on, N = sb["changed_off"], sb["changed_on"], sb["N_paired"]
        max_u = c_off - c_on
        say(f"      WHAT REMAINS UNIDENTIFIED, precisely: with "
            f"{c_off - c_on:,} more observed changes under gate-off than "
            f"gate-on, Delta_min stays <= 0 while the gate-on arm has more "
            f"than {max_u:,} unobservable requests. It currently has "
            f"{sb['unobs_on']:,}.")
        say(f"      Identification would require reducing gate-ON "
            f"unobservables from {sb['unobs_on']:,} to below {max_u:,} "
            f"-- a further {sb['unobs_on']-max_u:,} observations "
            f"({100*(sb['unobs_on']-max_u)/max(1,sb['unobs_on']):.1f}% of "
            f"those still missing).")

    json.dump({"reproduction_gate_passed": bool(ok),
               "recovery_method": "threshold on locally stored historical "
                                  "extracted_text; block-page filter retained; "
                                  "no refetch, no substitution, no fabrication",
               "short_observations_reexamined": len(short),
               "readable_locally": len(texts), "rejected_as_block": blocked,
               "by_threshold": {str(k): v for k, v in res.items()},
               "identified_at_most_permissive": bool(ident)},
              open(HERE / "recovery_results.json", "w"), indent=2)
    with open(HERE / "per_request_recovered.csv", "w", newline="",
              encoding="utf-8") as fh:
        ONb, OFFb = cache[min(THRESHOLDS)]
        w = csv.writer(fh)
        w.writerow(["query_id", "cluster_id", "outcome_on_thr400",
                    "outcome_off_thr400", f"outcome_on_thr{min(THRESHOLDS)}",
                    f"outcome_off_thr{min(THRESHOLDS)}"])
        for t, r in stream:
            q = r["query_id"]
            w.writerow([q, cid[q], ON0.get(q), OFF0.get(q),
                        ONb.get(q), OFFb.get(q)])
    (OR / "logs").mkdir(exist_ok=True)
    (OR / "logs" / "observability_recovery.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote recovery_results.json, per_request_recovered.csv")


if __name__ == "__main__":
    main()
