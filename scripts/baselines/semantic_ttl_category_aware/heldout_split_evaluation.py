#!/usr/bin/env python3
"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.


scripts/baselines/semantic_ttl_category_aware/heldout_split_evaluation.py

Held-out stage only: single scripts/evaluation of the held-out TEST split.

Nothing is tuned here. The configurations replayed are read verbatim from
validation_selection.json, which was frozen in the validation stage before any
test cluster was touched. No additional parameter is tried, and the selection
rules are not re-applied.

Stream construction matches the validation stage: the complete
stream is built once with schedule=zipf_uniform, seed=42, then filtered to TEST
clusters. Original timestamps, ages and snapshot rounds are preserved. Every
method starts from an empty, independent cache; no validation state is carried
over, because each mixed_age_replay.replay call allocates its own L1/L2/L3.

The paired cluster bootstrap needs per-request search and L2-hit indicators,
which mixed_age_replay.replay does not return. Rather than edit the implementation,
an instrumented in-memory copy of mixed_age_replay is compiled from the on-disk
source with three mechanical insertions that only record counters. Every
method is replayed twice (once through the untouched module, once through the
instrumented copy) and the script asserts that all aggregate metrics and the
entire per-request outcome sequence are identical before any bootstrap uses
the instrumented trace. The file on disk is never written.

CPU only. No GPU. No web or API calls. No answer generation, no judging.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import copy, csv, hashlib, json, os, pathlib, sys, types, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = ((OUTPUT_ROOT / "validation_split"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402
from schedule_robustness import support_of  # noqa: E402

CHANGED, UNCHANGED = ce.CHANGED, ce.UNCHANGED
SCHEDULE, SEED, B = "zipf_uniform", 42, 1000
WATCH = ["freshcache/method/tiered_cache_model.py", "freshcache/selection_replay_engine/mixed_age_replay.py",
         "freshcache/selection_replay_engine/method_replay.py", "freshcache/simulation/request_schedules.py"]
OUTS = ["test_results.json", "test_results.csv", "test_common_determinate.csv",
        "test_bootstrap.json", "stage2_test_report.md"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_instrumented():
    """Compile a counter-recording copy of mixed_age_replay from its source; the file on disk is not modified."""
    import re
    src = (ROOT / "freshcache/selection_replay_engine/mixed_age_replay.py").read_text()
    out, counts = [], {"srlog": 0, "reset": 0, "search": 0, "l2h": 0, "ret": 0}
    for ln in src.split("\n"):
        ind = ln[:len(ln) - len(ln.lstrip())]
        body = ln.strip()
        if body == "per_req = []":
            out.append(ln); out.append(ind + "_SRLOG = []")
            counts["srlog"] += 1; continue
        if body.startswith('q, fc, urls = r["query"]'):
            out.append(ind + "_sr = [0, 0]"); out.append(ln)
            counts["reset"] += 1; continue
        if body == "search += 1":
            out.append(ln + "; _sr[0] += 1"); counts["search"] += 1; continue
        if body == "l2h += 1":
            out.append(ln + "; _sr[1] += 1"); counts["l2h"] += 1; continue
        if body.startswith("per_req.append("):
            out.append(ind + "_SRLOG.append(tuple(_sr))"); out.append(ln); continue
        if body == "}, per_req":
            out.append(ln + ", _SRLOG"); counts["ret"] += 1; continue
        out.append(ln)
    assert counts == {"srlog": 1, "reset": 1, "search": 3, "l2h": 1, "ret": 1}, counts
    mod = types.ModuleType("mixed_engine_instrumented")
    mod.__file__ = str(ROOT / "freshcache/selection_replay_engine/mixed_age_replay.py")
    exec(compile("\n".join(out), "<mixed_age_replay instrumented copy>", "exec"),
         mod.__dict__)
    return mod


def assert_frozen_freshcache():
    checks = {
        "L1_SIM_THRESHOLD": (exp.L1_SIM_THRESHOLD, 0.40),
        "_L2_EQ_SIM_FLOOR": (exp._L2_EQ_SIM_FLOOR, 0.75),
        "EPS_ANSWER": (exp.EPS_ANSWER, 0.10),
        "EPS_URL_LIST": (exp.EPS_URL_LIST, 0.20),
        "EPS_CONTENT": (exp.EPS_CONTENT, 0.35),
        "TIER_MULT.answer": (exp.TIER_MULT["answer"], 1.5),
        "TIER_MULT.url_list": (exp.TIER_MULT["url_list"], 1.2),
        "TIER_MULT.content": (exp.TIER_MULT["content"], 1.0),
        "HALF_LIFE.TIMELESS": (exp.HALF_LIFE["TIMELESS"], 30 * 86400.0),
        "HALF_LIFE.SLOW": (exp.HALF_LIFE["SLOW"], 993600.0),
        "HALF_LIFE.MEDIUM": (exp.HALF_LIFE["MEDIUM"], 457920.0),
        "HALF_LIFE.FAST": (exp.HALF_LIFE["FAST"], 449280.0),
        "HALF_LIFE.REAL_TIME": (exp.HALF_LIFE["REAL_TIME"], 30.0),
        "_EQ_SIM_FLOOR": (exp._EQ_SIM_FLOOR, 0.80),
        "_EQ_JACCARD_MIN": (exp._EQ_JACCARD_MIN, 0.30),
    }
    bad = {k: v for k, (v, w) in checks.items() if abs(v - w) > 1e-9}
    assert not bad, f"FreshCache configuration is not frozen: {bad}"
    return {k: v for k, (v, _) in checks.items()}


def score(m, per, S):
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == CHANGED)
    un = sum(1 for o in ins if o == UNCHANGED)
    det = ch + un
    return {"n_requests": m["n_requests"], "search_saved_pct": m["search_saved_pct"],
            "drift_pct": (100 * ch / det) if det else None,
            "coverage_pct": (100 * det / len(S)) if S else None,
            "determinable": det, "changed": ch, "unchanged": un,
            "searches": m["search_calls"], "fetches": m["fetches"],
            "generations": m["generations"], "l1_hits": m["l1_hits"],
            "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
            "realtime_requests": m["realtime_requests"]}


def main():
    log = []

    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("STAGE 2 - held-out TEST scripts/evaluation, executed exactly once")
    say("  pre-run hashes:")
    for f, h in pre.items():
        say(f"    {f:<32}{h[:16]}")

    for o in OUTS:
        assert not (HERE / o).exists(), f"test artifact already exists: {o}"
    say("\n  no pre-existing test artifact found -> the test split has never "
        "been scored")

    # ---- 1. verify Stage 1 artifacts ----
    split = json.load(open(HERE / "split.json", encoding="utf-8"))
    sel = json.load(open(HERE / "validation_selection.json", encoding="utf-8"))
    assert sel["test_split_evaluated"] is False, "Stage 1 already saw the test split"
    val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])
    assert not (val_c & test_c), "splits intersect"
    assert len(val_c) + len(test_c) == split["totals"]["clusters"]
    say(f"  split verified: {len(val_c):,} validation / {len(test_c):,} test "
        f"clusters, intersection 0, union = all {split['totals']['clusters']:,}")

    # re-derive the Stage 1 selections from the frozen frontier and require a match
    frontier = list(csv.DictReader(open(HERE / "validation_frontier.csv",
                                        encoding="utf-8")))
    fcv = sel["frozen_freshcache_validation"]

    def num(x):
        return None if x in ("", None) else float(x)

    def recheck(meth):
        rows = [r for r in frontier if r["method"] == meth
                and num(r["drift_pct"]) is not None]
        def keys(r):
            return (-num(r["coverage_pct"]), int(r["fetches"]),
                    -(num(r["theta"]) or 0), (num(r["kappa"]) or 0))
        a = [r for r in rows if num(r["drift_pct"]) <= fcv["drift_pct"]]
        b = [r for r in rows
             if num(r["search_saved_pct"]) >= fcv["search_saved_pct"]]
        f = sorted(a, key=lambda r: (-num(r["search_saved_pct"],),) + keys(r))[0] if a else None
        s = sorted(b, key=lambda r: (num(r["drift_pct"]),) + keys(r))[0] if b else None
        return f, s

    say("\n  SELECTED CONFIGURATIONS IMPORTED FROM STAGE 1 (not re-derived, "
        "only re-checked):")
    plan, labels = [], {}
    for meth in ("SemanticTTL", "SCALM", "TieredFixedTTL"):
        rf, rs = recheck(meth)
        for lab, chosen, rc in (("freshness_matched",
                                 sel["selection"][meth]["freshness_matched"], rf),
                                ("savings_matched",
                                 sel["selection"][meth]["savings_matched"], rs)):
            if isinstance(chosen, str):
                assert rc is None, f"{meth} {lab}: rule now finds a point"
                say(f"    {meth:<16}{lab:<20}{chosen}")
                continue
            assert rc is not None, f"{meth} {lab}: rule now finds no point"
            for k in ("theta", "kappa", "search_saved_pct", "drift_pct"):
                a, b = chosen[k], num(rc[k])
                assert (a is None and b is None) or abs(a - b) < 1e-6, \
                    f"{meth} {lab} {k}: stored {a} != re-checked {b}"
            key = (meth, chosen["theta"], chosen["kappa"])
            labels.setdefault(key, []).append(lab)
            if key not in [p[0] for p in plan]:
                plan.append((key, chosen))
            th = "frozen" if chosen["theta"] is None else f"{chosen['theta']:.2f}"
            ka = "n/a" if chosen["kappa"] is None else f"{chosen['kappa']:.4f}"
            say(f"    {meth:<16}{lab:<20}theta={th:<8}kappa={ka:<9}"
                f"val saved {chosen['search_saved_pct']:.4f}%  "
                f"val drift {chosen['drift_pct']:.4f}%")
    say(f"  every stored selection reproduces the pre-registered rule exactly")
    dup = [k for k, v in labels.items() if len(v) == 2]
    for k in dup:
        say(f"    NOTE {k[0]}: both selection labels name the SAME point "
            f"(theta={k[1]}, kappa={k[2]}) -- it is replayed once")

    # ---- 2. test stream ----
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

    full = sc.build_stream(records, SCHEDULE, SEED)
    assert DS.check("primary_requests", len(full))
    tstream = [(t, r) for t, r in full
               if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(tstream) == split["test"]["requests"]
    S = support_of(tstream, rounds)
    assert len(S) == split["test"]["support"]
    say(f"\n  test stream: {len(tstream):,} requests, "
        f"policy-independent common support |S| = {len(S):,}")
    say("  built from the full zipf_uniform seed-42 stream, then "
        "filtered; timestamps preserved; schedule NOT rebuilt from test records")

    cl_names = sorted({(r.get("cluster_id") or r["query_id"]) for _, r in tstream})
    cl_idx = {c: i for i, c in enumerate(cl_names)}
    req_cl = np.array([cl_idx[(r.get("cluster_id") or r["query_id"])]
                       for _, r in tstream], dtype=np.int64)
    in_S = np.array([r["query_id"] in S for _, r in tstream], dtype=bool)
    say(f"  bootstrap resampling unit: cluster ({len(cl_names):,} test clusters)")

    frozen = assert_frozen_freshcache()
    say(f"  frozen FreshCache assertions passed ({len(frozen)} settings)")

    inst = build_instrumented()
    say("  instrumented in-memory copy of mixed_age_replay compiled "
        "(disk file untouched)")

    ORIG = {"FIXED_TTL": copy.deepcopy(exp.FIXED_TTL),
            "TIER_MULT": copy.deepcopy(exp.TIER_MULT),
            "L1T": exp.L1_SIM_THRESHOLD}

    def restore():
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG["FIXED_TTL"])
        exp.TIER_MULT.clear(); exp.TIER_MULT.update(ORIG["TIER_MULT"])
        exp.L1_SIM_THRESHOLD = ORIG["L1T"]

    def run(method, theta, kappa):
        def patch():
            if method == "SemanticTTL":
                exp.L1_SIM_THRESHOLD = theta
                exp.FIXED_TTL.update({c: v * kappa
                                      for c, v in ORIG["FIXED_TTL"].items()})
            elif method == "SCALM":
                exp.L1_SIM_THRESHOLD = theta
            elif method == "TieredFixedTTL":
                exp.TIER_MULT.update({t_: v * kappa
                                      for t_, v in ORIG["TIER_MULT"].items()})
        patch()
        try:
            m, per = me.replay(tstream, rounds, method, rich=rich)
        finally:
            restore()
        patch()
        try:
            mi, peri, srlog = inst.replay(tstream, rounds, method, rich=rich)
        finally:
            restore()
        # fidelity of the instrumented copy
        assert per == peri, f"{method}: instrumented per-request outcomes differ"
        for k in ("search_calls", "fetches", "l1_hits", "l2_hits", "l3_hits",
                  "generations", "n_requests", "search_saved_pct",
                  "realtime_requests"):
            assert m[k] == mi[k], f"{method}: instrumented {k} differs"
        assert len(srlog) == len(tstream)
        srch = np.array([a for a, _ in srlog], dtype=np.int64)
        l2ha = np.array([b for _, b in srlog], dtype=np.int64)
        assert int(srch.sum()) == m["search_calls"]
        assert int(l2ha.sum()) == m["l2_hits"]
        out = np.array([0 if o is None else (1 if o == CHANGED else
                                             (2 if o == UNCHANGED else 0))
                        for _, o in per], dtype=np.int8)
        return score(m, per, S), srch, out

    results, traces = {}, {}
    t0 = time.time()
    say("\n  replaying (each method twice: untouched module + instrumented copy)")

    r, srch, out = run("FreshCache", None, None)
    results["FreshCache"] = {**r, "method": "FreshCache", "label": "frozen",
                             "theta": None, "kappa": None}
    traces["FreshCache"] = (srch, out)
    say(f"    FreshCache            saved {r['search_saved_pct']:.4f}%  "
        f"drift {r['drift_pct']:.4f}%  cov {r['coverage_pct']:.4f}%  "
        f"({time.time()-t0:.0f}s)")

    for (meth, th, ka), chosen in plan:
        key = f"{meth}|theta={th}|kappa={ka}"
        r, srch, out = run(meth, th, ka)
        results[key] = {**r, "method": meth, "theta": th, "kappa": ka,
                        "label": "+".join(labels[(meth, th, ka)]),
                        "validation": {k: chosen[k] for k in
                                       ("search_saved_pct", "drift_pct",
                                        "coverage_pct")}}
        traces[key] = (srch, out)
        say(f"    {meth:<16}{str(th):>6}{str(ka):>8}  "
            f"saved {r['search_saved_pct']:.4f}%  drift {r['drift_pct']:.4f}%  "
            f"cov {r['coverage_pct']:.4f}%")

    assert (exp.FIXED_TTL == ORIG["FIXED_TTL"]
            and exp.TIER_MULT == ORIG["TIER_MULT"]
            and exp.L1_SIM_THRESHOLD == ORIG["L1T"]), "globals not restored"
    assert_frozen_freshcache()

    # ---- 5. pairwise common-determinate ----
    fc_s, fc_o = traces["FreshCache"]
    n_req = len(tstream)
    matched, boot = {}, {}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(cl_names), size=(B, len(cl_names)))
    cnt = np.zeros((B, len(cl_names)), dtype=np.int64)
    for i in range(B):
        cnt[i] = np.bincount(draws[i], minlength=len(cl_names))

    def per_cluster(v):
        return np.bincount(req_cl, weights=v, minlength=len(cl_names))

    ones = per_cluster(np.ones(n_req))
    fc_sc = per_cluster(fc_s.astype(float))
    for key in [k for k in results if k != "FreshCache"]:
        b_s, b_o = traces[key]
        both = in_S & (fc_o > 0) & (b_o > 0)
        fch = int(((fc_o == 1) & both).sum()); fcu = int(((fc_o == 2) & both).sum())
        bch = int(((b_o == 1) & both).sum()); bcu = int(((b_o == 2) & both).sum())
        n = int(both.sum())
        fd = 100 * fch / (fch + fcu) if fch + fcu else None
        bd = 100 * bch / (bch + bcu) if bch + bcu else None
        matched[key] = {"shared_n": n, "shared_pct_of_test_requests": 100 * n / n_req,
                        "freshcache_changed": fch, "freshcache_unchanged": fcu,
                        "freshcache_drift_pct": fd, "baseline_changed": bch,
                        "baseline_unchanged": bcu, "baseline_drift_pct": bd,
                        "drift_diff_baseline_minus_freshcache":
                            (bd - fd) if (bd is not None and fd is not None) else None}
        # ---- 6. paired cluster bootstrap (identical clusters for both arms) ----
        b_sc = per_cluster(b_s.astype(float))
        fcc = per_cluster(((fc_o == 1) & both).astype(float))
        fcd = per_cluster((((fc_o == 1) | (fc_o == 2)) & both).astype(float))
        bcc = per_cluster(((b_o == 1) & both).astype(float))
        bcd = per_cluster((((b_o == 1) | (b_o == 2)) & both).astype(float))
        N = cnt @ ones
        sav = (100 * (1 - (cnt @ b_sc) / N)) - (100 * (1 - (cnt @ fc_sc) / N))
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = (100 * (cnt @ bcc) / (cnt @ bcd)) - (100 * (cnt @ fcc) / (cnt @ fcd))
        dd = dd[np.isfinite(dd)]

        def summ(v, point):
            return {"point_estimate": point,
                    "ci95_lo": float(np.percentile(v, 2.5)),
                    "ci95_hi": float(np.percentile(v, 97.5)),
                    "frac_gt_0": float((v > 0).mean()),
                    "frac_lt_0": float((v < 0).mean()),
                    "replicates": int(v.size)}
        boot[key] = {
            "savings_diff": summ(sav, results[key]["search_saved_pct"]
                                 - results["FreshCache"]["search_saved_pct"]),
            "matched_drift_diff": summ(
                dd, matched[key]["drift_diff_baseline_minus_freshcache"])}
        say(f"    matched {key:<40} shared N {n:,}  "
            f"drift diff {matched[key]['drift_diff_baseline_minus_freshcache']:+.4f} pp")

    post = {f: sha(ROOT / f) for f in WATCH}
    unchanged = all(pre[f] == post[f] for f in WATCH)
    say(f"\n  implementation files unchanged: {unchanged}")

    json.dump({"stage": 2, "test_evaluated_once": True,
               "selection_source": "validation_selection.json (Stage 1, frozen)",
               "retuning_performed": False, "web_calls": 0, "llm_calls": 0,
               "gpu_used": False, "bootstrap_replicates": B,
               "bootstrap_unit": "cluster", "bootstrap_seed": SEED,
               "test": {"requests": n_req, "clusters": len(cl_names),
                        "support": len(S)},
               "results": results, "hashes_pre": pre, "hashes_post": post,
               "implementation_unchanged": unchanged},
              open(HERE / "test_results.json", "w"), indent=2, default=str)
    for mk in ("search_saved_pct", "l1_hits", "l2_hits"): REF.record(f"selection_engine/heldout/FreshCache/{mk}", results["FreshCache"][mk])
    json.dump(boot, open(HERE / "test_bootstrap.json", "w"), indent=2)

    cols = ["method", "label", "theta", "kappa", "n_requests", "search_saved_pct",
            "drift_pct", "coverage_pct", "determinable", "changed", "unchanged",
            "searches", "fetches", "generations", "l1_hits", "l2_hits", "l3_hits",
            "realtime_requests"]
    with open(HERE / "test_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results.values():
            w.writerow(r)
    with open(HERE / "test_common_determinate.csv", "w", newline="",
              encoding="utf-8") as fh:
        mc = ["comparison", "shared_n", "shared_pct_of_test_requests",
              "freshcache_changed", "freshcache_unchanged", "freshcache_drift_pct",
              "baseline_changed", "baseline_unchanged", "baseline_drift_pct",
              "drift_diff_baseline_minus_freshcache"]
        w = csv.DictWriter(fh, fieldnames=mc)
        w.writeheader()
        for k, v in matched.items():
            w.writerow({"comparison": f"{k} vs FreshCache", **v})

    write_report(results, matched, boot, sel, split, S, n_req, len(cl_names),
                 pre, post, unchanged, labels)
    (HERE / "stage2_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote " + ", ".join(OUTS) + ", stage2_run.log")
    (HERE / "stage2_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")


def write_report(results, matched, boot, sel, split, S, n_req, n_cl,
                 pre, post, unchanged, labels):
    fc = results["FreshCache"]

    def nm(k):
        r = results[k]
        if r["method"] == "FreshCache":
            return "FreshCache (frozen)"
        th = "frozen" if r["theta"] is None else f"{r['theta']:.2f}"
        ka = "n/a" if r["kappa"] is None else f"{r['kappa']:.4f}"
        return f"{r['method']} (theta={th}, kappa={ka})"

    def row(k):
        r = results[k]
        return ("| " + nm(k) + " | " + r["label"] + " | "
                + " | ".join([f"{r['n_requests']:,}",
                              f"{r['search_saved_pct']:.4f}",
                              f"{r['drift_pct']:.4f}", f"{r['coverage_pct']:.4f}",
                              f"{r['determinable']:,}", f"{r['changed']:,}",
                              f"{r['searches']:,}", f"{r['fetches']:,}",
                              f"{r['generations']:,}", f"{r['l1_hits']:,}",
                              f"{r['l2_hits']:,}",
                              f"{r['l3_hits']:,}" if r["l3_hits"] else "n/a"])
                + " |")

    HDR = ("| Method | Selection label | Test requests | Saved % | Drift % | "
           "Coverage % | Determinable | Changed | Searches | Fetches | "
           "Generations | L1 | L2 | L3 |\n|" + "---|" * 14)

    def block(meth):
        ks = [k for k in results if results[k]["method"] == meth]
        return HDR + "\n" + "\n".join(row(k) for k in ks) if ks else "_none_"

    mrows = "\n".join(
        f"| {nm(k)} | {v['shared_n']:,} | {v['shared_pct_of_test_requests']:.4f} | "
        f"{v['freshcache_changed']:,} | {v['freshcache_unchanged']:,} | "
        f"{v['freshcache_drift_pct']:.4f} | {v['baseline_changed']:,} | "
        f"{v['baseline_unchanged']:,} | {v['baseline_drift_pct']:.4f} | "
        f"{v['drift_diff_baseline_minus_freshcache']:+.4f} |"
        for k, v in matched.items())

    def brow(k, which, unit):
        b = boot[k][which]
        return (f"| {nm(k)} | {unit} | {b['point_estimate']:+.4f} | "
                f"[{b['ci95_lo']:+.4f}, {b['ci95_hi']:+.4f}] | "
                f"{b['frac_gt_0']:.3f} | {b['frac_lt_0']:.3f} | {b['replicates']:,} |")

    brows = "\n".join(
        brow(k, "savings_diff", "search savings (pp)") + "\n"
        + brow(k, "matched_drift_diff", "matched-denominator drift (pp)")
        for k in boot)

    grows = "\n".join(
        f"| {nm(k)} | {results[k]['validation']['search_saved_pct']:.4f} | "
        f"{results[k]['search_saved_pct']:.4f} | "
        f"{results[k]['search_saved_pct']-results[k]['validation']['search_saved_pct']:+.4f} | "
        f"{results[k]['validation']['drift_pct']:.4f} | "
        f"{results[k]['drift_pct']:.4f} | "
        f"{results[k]['drift_pct']-results[k]['validation']['drift_pct']:+.4f} | "
        f"{results[k]['validation']['coverage_pct']:.4f} | "
        f"{results[k]['coverage_pct']:.4f} | "
        f"{results[k]['coverage_pct']-results[k]['validation']['coverage_pct']:+.4f} |"
        for k in results if k != "FreshCache")
    fcv = sel["frozen_freshcache_validation"]
    grows = (f"| FreshCache (frozen) | {fcv['search_saved_pct']:.4f} | "
             f"{fc['search_saved_pct']:.4f} | "
             f"{fc['search_saved_pct']-fcv['search_saved_pct']:+.4f} | "
             f"{fcv['drift_pct']:.4f} | {fc['drift_pct']:.4f} | "
             f"{fc['drift_pct']-fcv['drift_pct']:+.4f} | "
             f"{fcv['coverage_pct']:.4f} | {fc['coverage_pct']:.4f} | "
             f"{fc['coverage_pct']-fcv['coverage_pct']:+.4f} |\n") + grows

    srows = []
    for meth in ("SemanticTTL", "SCALM", "TieredFixedTTL"):
        for lab in ("freshness_matched", "savings_matched"):
            c = sel["selection"][meth][lab]
            if isinstance(c, str):
                srows.append(f"| {meth} | {lab} | **{c}** | — | — | — |")
            else:
                th = "frozen" if c["theta"] is None else f"{c['theta']:.2f}"
                ka = "n/a" if c["kappa"] is None else f"{c['kappa']:.4f}"
                srows.append(f"| {meth} | {lab} | theta = {th}, kappa = {ka} | "
                             f"{c['search_saved_pct']:.4f} | {c['drift_pct']:.4f} "
                             f"| {c['coverage_pct']:.4f} |")
    srows = "\n".join(srows)
    dup = "\n".join(
        f"- **{k[0]}**: the freshness-matched and savings-matched rules selected "
        f"the same point (theta={k[1]}, kappa={k[2]}); it was replayed once and "
        f"carries both labels."
        for k, v in labels.items() if len(v) == 2)

    hrows = "\n".join(
        f"| `{f}` | `{pre[f][:16]}` | `{post[f][:16]}` | "
        f"{'unchanged' if pre[f]==post[f] else 'CHANGED'} |" for f in pre)

    md = f"""# Stage 2 — held-out test scripts/evaluation

**The test split was evaluated exactly once, with configurations frozen in
Stage 1.** Nothing was retuned, no additional baseline parameter was swept, and
the selection rules were not changed. CPU only. **No web or API calls, no LLM
generation, no LLM judging.** No manuscript file and no Stage 1 artifact was
modified.

## 1. Frozen selections imported from Stage 1

Read verbatim from `validation_selection.json`. Each stored selection was
re-derived from the frozen `validation_frontier.csv` with the pre-registered
rules and asserted to match before any test replay ran.

| Baseline | Selection label | Configuration | Val saved % | Val drift % | Val coverage % |
|---|---|---|---|---|---|
{srows}

{dup if dup else ""}

SCALM has **no freshness-matched point**: its lowest validation drift over all
nine thresholds was above frozen FreshCache's validation drift. No substitute
was invented, so SCALM is evaluated at its savings-matched point only.

## 2. Confirmation the test split was evaluated exactly once

- `validation_selection.json` carried `test_split_evaluated: false`.
- No `test_results.*`, `test_common_determinate.csv`, `test_bootstrap.json` or
  `stage2_test_report.md` existed before this run; the script aborts if any does.
- Validation and test cluster sets were re-verified disjoint
  ({split['validation']['clusters']:,} / {split['test']['clusters']:,}), and
  their union is all {split['totals']['clusters']:,} clusters.
- The test replay stream was built by filtering the complete
  `zipf_uniform` seed-42 stream, **not** by rebuilding a schedule from test
  records, so timestamps and cache ages are the original ones.
- Every method was replayed with its own freshly allocated, empty L1/L2/L3. No
  validation state crosses into the test replay.
- Test split: **{n_req:,} requests**, {n_cl:,} clusters, policy-independent
  common observable support **|S| = {len(S):,}**. That single support set is the
  denominator for every method's coverage in sections 3–6.

## 3. Frozen FreshCache — test result

{HDR}
{row('FreshCache')}

## 4. SemanticTTL — selected configuration(s)

{block('SemanticTTL')}

## 5. SCALM style — selected configuration

{block('SCALM')}

## 6. TieredFixedTTL — selected configuration

{block('TieredFixedTTL')}

## 7. Pairwise common-determinate comparison (mandatory matched denominator)

Restricted to test requests in the common support where **both** FreshCache and
the baseline produced a determinable outcome. Method-specific denominators are
not used anywhere in this table.

| Baseline vs FreshCache | Shared N | Shared % of test requests | FC changed | FC unchanged | FC drift % | Base changed | Base unchanged | Base drift % | Drift diff (base − FC) pp |
|---|---|---|---|---|---|---|---|---|---|
{mrows}

## 8. Paired cluster bootstrap

Test split only. Resampling unit is the **cluster**; 1,000 replicates; the
**identical** resampled cluster multiset is applied to FreshCache and to the
baseline in every replicate. Savings differences use each replicate's own
request total; drift differences use the pairwise common-determinate set of
section 7.

This is a **stability diagnostic**, not a population-level hypothesis test, and
it was not used to select or retune anything.

| Comparison | Quantity | Point estimate | 95% percentile CI | Frac > 0 | Frac < 0 | Replicates |
|---|---|---|---|---|---|---|
{brows}

## 9. Validation → test generalization

| Configuration | Val saved % | Test saved % | Δ saved | Val drift % | Test drift % | Δ drift | Val coverage % | Test coverage % | Δ coverage |
|---|---|---|---|---|---|---|---|---|---|
{grows}

Drift and coverage here use each split's own policy-independent support, so the
Δ columns describe whether an operating point moves, not a matched contrast.

## 10. No answer-quality conclusion is possible from this experiment

This stage generated no answers and ran no judge. The drift metric is
**content drift of the served evidence**, not answer correctness. Nothing here
supports or refutes a claim that any method produces better or worse answers,
and no WAI comparison is made or implied. Because observable coverage and
determinable denominators differ substantially across methods, no method is
described as "better" on raw drift alone; section 7 is the only drift contrast
in this report with a common denominator.

## 11. No web, API or LLM calls

Every replay read local artifacts only: the cached BGE-M3 similarity matrix, the
reported round table and the benchmark records. Zero network requests, zero GPU
use, zero generation, zero judging.

## 12. Implementation integrity

| File | SHA256 before | SHA256 after | Status |
|---|---|---|---|
{hrows}

All implementation files are byte-identical before and after
(`implementation_unchanged: {unchanged}`). Swept globals (`FIXED_TTL`,
`TIER_MULT`, `L1_SIM_THRESHOLD`) were restored in a `finally` block after every
replay, and the 15 frozen-FreshCache assertions were re-checked after the last
replay and passed.

Per-request search and L2-hit indicators, needed for the cluster bootstrap, are
not returned by `mixed_age_replay.replay`. Instead of editing it, an instrumented
copy was compiled **in memory** from the on-disk source with counter-recording
insertions only. Every method was replayed through both the untouched module and
the instrumented copy, and the run asserts that the full per-request outcome
sequence and all aggregate metrics are identical before any bootstrap uses the
instrumented trace. The file on disk was never written, as the hashes confirm.
"""
    (HERE / "stage2_test_report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
