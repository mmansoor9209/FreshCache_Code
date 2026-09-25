#!/usr/bin/env python3
"""
Sections 2 and 3: FreshCache-L1Only, ExactTTL and ExactNoTTL.

Runs the full mixed-age workload and the cluster-disjoint held-out test split
for every policy under ONE execution path and ONE scoring rule, after
reproducing the published FreshCache anchors.

Writes 02_l1only/ (L1Only) and 03_exactttl/ (Exact*) plus a joint table.
Read-only on every existing file; CPU only.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(BASE / "lib"))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402
from e1_robustness import support_of      # noqa: E402
import v3_engine as v3                   # noqa: E402

CHANGED, UNCHANGED = v3.CHANGED, v3.UNCHANGED
SCHEDULE, SEED = "zipf_uniform", 42
WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
         "v14_baselines/engine_all.py", "v16_exp12/schedules.py"]
ANCHOR = {"search_saved_pct": 62.7320, "drift_pct": 3.4154,
          "coverage_pct": 73.3736, "l1_hits": 1175, "l2_hits": 18398}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def score(m, per, S):
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
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("V3 SECTIONS 2 & 3 -- FreshCache-L1Only, ExactTTL, ExactNoTTL")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    cfg = v3.register()
    say(f"  registered FreshCache_L1Only = {cfg}")
    say(f"  FreshCache_Full              = {ea.CONFIG['FreshCache_Full']}")
    say("  -> L1 half is byte-identical; only l2/l3 are disabled")

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
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]

    full = sc.build_stream(records, SCHEDULE, SEED)
    assert len(full) == 31201, len(full)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c, val_c = set(split["test_clusters"]), set(split["validation_clusters"])
    assert not (test_c & val_c)
    test = [(t, r) for t, r in full
            if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(test) == split["test"]["requests"]
    say(f"\n  full workload {len(full):,} requests; held-out test "
        f"{len(test):,} requests over {len(test_c):,} clusters "
        f"(validation leakage: 0)")

    # ---- the frozen SemanticTTL operating point, read from Stage 1 ----
    sel = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                         / "validation_selection.json", encoding="utf-8"))
    c = sel["selection"]["SemanticTTL"]["savings_matched"]
    KAPPA, THETA = c["kappa"], c["theta"]
    say(f"  frozen SemanticTTL point (Stage 1, not chosen here): "
        f"theta={THETA} kappa={KAPPA}")

    ORIG_TTL, ORIG_T = dict(exp.FIXED_TTL), exp.L1_SIM_THRESHOLD
    results = {}

    for split_name, stream in (("full_mixed_age", full), ("heldout_test", test)):
        S = support_of(stream, rounds)
        say(f"\n  ==== {split_name}: {len(stream):,} requests, |S| = {len(S):,} ====")

        # ---- reproduction gate ----
        fm, fper = me.replay(stream, rounds, "FreshCache", rich=rich)
        fs = score(fm, fper, S)
        if split_name == "full_mixed_age":
            bad = [k for k, v in ANCHOR.items()
                   if abs(fs[k] - v) > (1e-4 if isinstance(v, float) else 0)]
            if bad:
                say(f"  REPRODUCTION GATE FAILED on {bad}: "
                    f"{ {k: fs[k] for k in ANCHOR} }")
                sys.exit(2)
            say(f"  reproduction gate PASSED: saved {fs['search_saved_pct']:.4f}%  "
                f"drift {fs['drift_pct']:.4f}%  cov {fs['coverage_pct']:.4f}%  "
                f"L1 {fs['l1_hits']:,}  L2 {fs['l2_hits']:,}")

        # ---- fidelity gate for the transcribed L1 cache ----
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG_TTL)
        try:
            v3.assert_fidelity(stream, rounds)
            say("  fidelity gate PASSED: transcribed L1 cache == mixed_engine "
                "SemanticTTL (metrics and per-request outcomes)")
        finally:
            exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG_TTL)

        rows = {"FreshCache": fs}
        for name in ("FreshCache_L1Only", "L2Only", "SemanticTTL",
                     "TieredFixedTTL", "SCALM"):
            m, per = me.replay(stream, rounds, name, rich=rich)
            rows[name] = score(m, per, S)

        # FreshCache without L1 (Without-L1 == L2Only+L3); and NoCache
        m, per = me.replay(stream, rounds, "FreshCache", rich=rich)  # sanity
        nc_m = {"n_requests": len(stream), "search_calls": len(stream),
                "search_saved_pct": 0.0, "search_avoided": 0, "l1_hits": 0,
                "l2_hits": 0, "l3_hits": 0,
                "fetches": sum(len(r["urls"]) for _, r in stream),
                "generations": len(stream)}
        rows["NoCache"] = score(nc_m, [(r["query_id"], None) for _, r in stream], S)

        # tuned SemanticTTL (theta, kappa) -- the validation-selected point
        exp.L1_SIM_THRESHOLD = THETA
        exp.FIXED_TTL.update({k: v * KAPPA for k, v in ORIG_TTL.items()})
        try:
            m, per = me.replay(stream, rounds, "SemanticTTL", rich=rich)
            rows["SemanticTTL_tuned"] = score(m, per, S)
            em, eper, _ = v3.replay_l1cache(stream, rounds, "exact", "fixed")
            rows["ExactTTL_tuned"] = score(em, eper, S)
        finally:
            exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG_TTL)
            exp.L1_SIM_THRESHOLD = ORIG_T

        em, eper, _ = v3.replay_l1cache(stream, rounds, "exact", "fixed")
        rows["ExactTTL"] = score(em, eper, S)
        em, eper, _ = v3.replay_l1cache(stream, rounds, "exact", "none")
        rows["ExactNoTTL"] = score(em, eper, S)

        order = ["NoCache", "ExactNoTTL", "ExactTTL", "ExactTTL_tuned",
                 "SemanticTTL", "SemanticTTL_tuned", "TieredFixedTTL", "SCALM",
                 "L2Only", "FreshCache_L1Only", "FreshCache"]
        say(f"\n  {'policy':<22}{'saved%':>9}{'drift%':>9}{'cov%':>8}"
            f"{'det':>7}{'fetch/1k':>10}{'gen/1k':>9}{'L1':>8}{'L2':>8}{'L3':>8}")
        for k in order:
            r = rows[k]
            d = f"{r['drift_pct']:>8.4f}" if r["drift_pct"] is not None else "       -"
            cv = f"{r['coverage_pct']:>7.4f}" if r["coverage_pct"] is not None else "      -"
            say(f"  {k:<22}{r['search_saved_pct']:>8.4f}%{d}%{cv}%"
                f"{r['determinable']:>7,}{r['fetch_per_1k']:>10.1f}"
                f"{r['gen_per_1k']:>9.1f}{r['l1_hits']:>8,}{r['l2_hits']:>8,}"
                f"{r['l3_hits']:>8,}")
        results[split_name] = {"support": len(S), "requests": len(stream),
                               "rows": rows, "order": order}

    assert exp.FIXED_TTL == ORIG_TTL and exp.L1_SIM_THRESHOLD == ORIG_T
    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post, "implementation files changed"
    say("\n  implementation files unchanged: True")

    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "schedule": SCHEDULE, "seed": SEED,
           "anchor": ANCHOR, "anchor_reproduced": True,
           "l1only_config": cfg, "freshcache_config": ea.CONFIG["FreshCache_Full"],
           "frozen_semanticttl": {"theta": THETA, "kappa": KAPPA},
           "exact_normalisation": "casefold; collapse whitespace; strip trailing .?!,;:",
           "hashes": pre, "results": results}
    for d in (HERE, BASE / "03_exactttl"):
        json.dump(out, open(d / "baseline_results.json", "w"), indent=2)
        cols = ["split", "policy"] + list(next(iter(
            results["full_mixed_age"]["rows"].values())).keys())
        with open(d / "baseline_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for sn, blk in results.items():
                for k in blk["order"]:
                    w.writerow({"split": sn, "policy": k, **blk["rows"][k]})
        (d / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote baseline_results.json/.csv and run.log into 02_l1only/ and 03_exactttl/")


if __name__ == "__main__":
    main()
