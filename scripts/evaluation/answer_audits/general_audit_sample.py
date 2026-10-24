#!/usr/bin/env python3
"""CPU stage of the held-out answer-quality audit: frozen FreshCache vs. SemanticTTL with kappa = 1/16.

SemanticTTL uses the validation-selected configuration (kappa = 1/16), read verbatim from the
validation-selection artifact and asserted against the held-out test result. Test clusters only,
over one common policy-neutral sample. Nothing is tuned. The sampling frame, the
28-stratum design, the seed, the evidence builder and the context limits are those of the
mixed-age full-policy audit (audit_replay.py), whose helpers are imported directly so the evidence
construction is byte-identical.

Read-only on every existing artifact. Writes only into this folder. No web calls.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import hashlib, json, os, pathlib, random, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = ((OUTPUT_ROOT / "general_audit"))
STAGE = ((OUTPUT_ROOT / "validation_split"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402
from schedule_robustness import support_of  # noqa: E402
import audit_replay as p2                    # noqa: E402  (band, ctx_for, h, GOLD...)

SCHEDULE, SEED, TARGET = "zipf_uniform", 42, 400
WATCH = ["freshcache/method/tiered_cache_model.py", "freshcache/selection_replay_engine/mixed_age_replay.py",
         "freshcache/selection_replay_engine/method_replay.py", "freshcache/simulation/request_schedules.py"]
band, ctx_for, h = p2.band, p2.ctx_for, p2.h


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def assert_frozen_freshcache():
    checks = {
        "L1_SIM_THRESHOLD": (exp.L1_SIM_THRESHOLD, 0.40),
        "_L2_EQ_SIM_FLOOR": (exp._L2_EQ_SIM_FLOOR, 0.75),
        "EPS_ANSWER": (exp.EPS_ANSWER, 0.10), "EPS_URL_LIST": (exp.EPS_URL_LIST, 0.20),
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
    assert not bad, f"FreshCache not frozen: {bad}"
    return {k: v for k, (v, _) in checks.items()}


def replay_freshcache(stream):
    """FreshCache replay path of audit_replay.replay, imported verbatim."""
    return p2.replay(stream, {}, "FreshCache")


def replay_semanticttl(stream, theta, ttl):
    """SemanticTTL exactly as method_replay.CONFIG['SemanticTTL'] defines it:
    l1=sim_ttl (similarity only, TTL gate on the INCOMING class), no L2, no L3,
    REAL_TIME bypass. No entity / lexical / answer-type equivalence gate is
    added -- that gate belongs to FreshCache and adding it would change the
    baseline. Records, per L1 entry, the creating query and evidence so a hit
    can return the STORED answer."""
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    n_max = len(stream)
    l1_qi = np.empty(n_max, np.int64); l1_t = np.empty(n_max, np.float64)
    l1_q, l1_fc, l1_urls, l1_src = [], [], [], []
    n1 = 0
    l1h = search = fetches = 0
    rows = {}
    for t, r in stream:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        row = SIM[IDX[q]]
        own = [u["url_hash"] for u in urls]
        if fc == "REAL_TIME":                       # rt_bypass=True
            search += 1; fetches += len(own)
            rows[qid] = {"tier": "REAL_TIME", "t": t,
                         "ev": [(u, t) for u in own], "own": own}
            continue
        hit1 = -1; sim1 = None
        if n1:
            sims = row[l1_qi[:n1]]
            cand = np.nonzero(sims >= theta)[0]
            if cand.size:
                okm = np.fromiter((bool(ttl.get(l1_fc[int(j)], 0))
                                   and (t - l1_t[j]) <= ttl.get(l1_fc[int(j)], 0)
                                   for j in cand), bool, cand.size)
                keep = cand[okm]
                if keep.size:
                    hit1 = int(keep[np.argsort(-sims[keep], kind="stable")[0]])
                    sim1 = float(sims[hit1])
        if hit1 >= 0:
            l1h += 1
            src = l1_src[hit1]
            rows[qid] = {"tier": "HIT", "t": t, "similarity": sim1,
                         "t_cached": float(l1_t[hit1]),
                         "age": t - float(l1_t[hit1]),
                         "matched_query": l1_q[hit1],
                         "stored_answer_query": src["query"],
                         "stored_answer_ev": src["ev"],
                         "stored_answer_t": src["t"], "own": own}
            continue
        search += 1; fetches += len(own)
        ev = [(u, t) for u in own]
        rows[qid] = {"tier": "miss", "t": t, "similarity": None,
                     "age": 0.0, "ev": ev, "own": own}
        l1_qi[n1] = qi_ = IDX[q]; l1_t[n1] = t
        l1_q.append(q); l1_fc.append(fc); l1_urls.append(list(own))
        l1_src.append({"query": q, "ev": ev, "t": t}); n1 += 1
    m = {"search_saved_pct": round(100 * (1 - search / len(stream)), 4),
         "l1_hits": l1h, "search_calls": search, "fetches": fetches}
    return m, rows


def main():
    log = []

    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("HELD-OUT ANSWER AUDIT -- general_audit_sample (CPU)")
    for f, hh in pre.items():
        say(f"    {f:<32}{hh[:16]}")

    split = json.load(open(STAGE / "split.json", encoding="utf-8"))
    sel = json.load(open(STAGE / "validation_selection.json", encoding="utf-8"))
    test_res = json.load(open(STAGE / "test_results.json", encoding="utf-8"))

    # ---- 1. frozen baseline ----
    c = sel["selection"]["SemanticTTL"]["savings_matched"]
    assert not isinstance(c, str), "no savings-matched SemanticTTL point"
    assert abs(c["kappa"] - 1 / 16) < 1e-12, f"kappa is {c['kappa']}, not 1/16"
    theta = c["theta"]
    stage2 = [v for v in test_res["results"].values()
              if v["method"] == "SemanticTTL"
              and abs(v["kappa"] - c["kappa"]) < 1e-12
              and abs(v["theta"] - theta) < 1e-12]
    assert len(stage2) == 1, "Stage 2 did not evaluate exactly this point"
    stage2 = stage2[0]
    assert "savings_matched" in stage2["label"]
    ORIG_TTL = dict(exp.FIXED_TTL)
    ttl = {k: v * c["kappa"] for k, v in ORIG_TTL.items()}
    say("\n  FROZEN SemanticTTL configuration (read from Stage 1, not chosen here)")
    say(f"    selection rule            savings-matched (pre-registered)")
    say(f"    semantic threshold theta  {theta}")
    say(f"    TTL multiplier kappa      {c['kappa']}  (= 1/16)")
    for k in ("TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"):
        say(f"    TTL[{k:<10}]          {ORIG_TTL[k]:>9,} s * kappa = "
            f"{ttl[k]:>10,.1f} s")
    say(f"    engine config             {ea.CONFIG['SemanticTTL']}")
    say(f"    validation  saved {c['search_saved_pct']:.4f}%  "
        f"drift {c['drift_pct']:.4f}%  coverage {c['coverage_pct']:.4f}%")
    say(f"    Stage 2 test saved {stage2['search_saved_pct']:.4f}%  "
        f"drift {stage2['drift_pct']:.4f}%  l1_hits {stage2['l1_hits']:,}")
    frozen = assert_frozen_freshcache()
    say(f"  frozen FreshCache assertions passed ({len(frozen)} settings)")

    # ---- 2. test clusters only ----
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    rounds = {}
    with open((DATA_ROOT / "derived/round_table.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]

    test_c = set(split["test_clusters"])
    val_c = set(split["validation_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    assert DS.check("primary_requests", len(full))
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(stream) == split["test"]["requests"] == test_res["test"]["requests"]
    leak = [r["query_id"] for _, r in stream
            if (r.get("cluster_id") or r["query_id"]) in val_c]
    assert not leak, f"{len(leak)} validation requests leaked into the audit"
    S = support_of(stream, rounds)
    say(f"\n  test stream {len(stream):,} requests, {len(test_c):,} clusters, "
        f"|S| = {len(S):,}; validation requests present: 0")

    # ---- policy traces, each from an empty independent cache ----
    mf, fcrows = replay_freshcache(stream)
    e = test_res["results"]["FreshCache"]
    assert abs(mf["search_saved_pct"] - e["search_saved_pct"]) < 1e-6 \
        and mf["l1_hits"] == e["l1_hits"] and mf["l2_hits"] == e["l2_hits"], \
        f"FreshCache path does not reproduce Stage 2: {mf} vs {e}"
    say(f"  FreshCache      saved {mf['search_saved_pct']:.4f}%  "
        f"L1 {mf['l1_hits']:,}  L2 {mf['l2_hits']:,}  -> reproduces Stage 2")

    exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ttl)
    try:
        ms, strows = replay_semanticttl(stream, theta, ttl)
    finally:
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG_TTL)
    assert abs(ms["search_saved_pct"] - stage2["search_saved_pct"]) < 1e-6 \
        and ms["l1_hits"] == stage2["l1_hits"], \
        f"SemanticTTL path does not reproduce Stage 2: {ms} vs {stage2}"
    say(f"  SemanticTTL     saved {ms['search_saved_pct']:.4f}%  "
        f"hits {ms['l1_hits']:,}  -> reproduces Stage 2")
    assert exp.FIXED_TTL == ORIG_TTL

    # ---- 3. policy-neutral frame, 28 strata, seed 42 ----
    gold = json.load(open(p2.GOLD, encoding="utf-8"))

    def goldof(qid, qt):
        g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower())
        return (g.get("answer") if isinstance(g, dict) else g)

    base = ea._CLUSTER_BASE
    IDX, SIM = exp._QUERY_TO_IDX, exp._SIM_MATRIX
    simbase = {}
    for _, r in stream:
        bq = base.get(r["query_id"], "")
        simbase[r["query_id"]] = (float(SIM[IDX[r["query"]], IDX[bq]])
                                  if r.get("is_paraphrase") and bq and bq in IDX
                                  else None)
    frame = []
    for t, r in stream:
        qid = r["query_id"]
        if r["freshness_class"] == "REAL_TIME":
            continue
        g = goldof(qid, r["query"])
        if not g:
            continue
        own = [u["url_hash"] for u in r["urls"]]
        if not ctx_for(tuple((u, t) for u in own)):
            continue
        frame.append({"qid": qid, "t": t, "query": r["query"], "gold": g,
                      "fc": r["freshness_class"], "simbase": simbase.get(qid),
                      "cluster": r.get("cluster_id") or qid, "in_S": qid in S})
    say(f"\n  policy-neutral frame {len(frame):,} test requests "
        f"(non-REAL_TIME, gold present, non-empty fresh context at t)")
    say("  eligibility uses no FreshCache and no SemanticTTL hit/miss or "
        "correctness information")

    strata = defaultdict(list)
    for d in frame:
        strata[(d["fc"], band(d["simbase"]))].append(d)
    rng = random.Random(SEED)
    per = max(1, TARGET // max(len(strata), 1))
    sample = []
    for k in sorted(strata):
        g = strata[k][:]
        rng.shuffle(g)
        sample.extend(g[:per])
    if len(sample) < TARGET:
        chosen = {d["qid"] for d in sample}
        rest = [d for d in frame if d["qid"] not in chosen]
        rng.shuffle(rest)
        sample.extend(rest[:TARGET - len(sample)])
    sample = sample[:TARGET]
    assert len(sample) == len({d["qid"] for d in sample}) == TARGET
    say(f"  strata {len(strata)}  cap {per}  sampled {len(sample)}  "
        f"unique {len({d['qid'] for d in sample})}  seed {SEED}")
    say("  sampled request IDs are frozen BEFORE either policy is consulted")

    # ---- arms ----
    out, tasks = [], {}
    comp = {"fc": Counter(), "sttl": Counter()}
    for d in sample:
        qid, t = d["qid"], d["t"]
        fctx = ctx_for(tuple((u, t) for u in fcrows[qid]["own"]))
        rec = {"query_id": qid, "cluster_id": d["cluster"], "query": d["query"],
               "gold": d["gold"], "fc": d["fc"], "t": t, "in_S": d["in_S"],
               "simbase": d["simbase"], "simbase_band": band(d["simbase"]),
               "fresh_ctx_sha": h(fctx), "fresh_pages": len(fcrows[qid]["own"]),
               "fresh_versions": [[u, t] for u in fcrows[qid]["own"]]}
        tasks[("fresh", h(fctx), h(d["query"]))] = (d["query"], fctx)
        rec["fresh_key"] = ["fresh", h(fctx), h(d["query"])]
        for tag, rowset in (("fc", fcrows), ("sttl", strows)):
            p = rowset[qid]
            tier = p["tier"]
            comp[tag][tier] += 1
            rec[f"{tag}_tier"] = tier
            rec[f"{tag}_age"] = p.get("age", 0.0)
            rec[f"{tag}_similarity"] = p.get("similarity")
            if tier in ("L1", "HIT"):
                sctx = ctx_for(tuple(p["stored_answer_ev"]))
                sq = p["stored_answer_query"]
                key = ("stored", h(sctx), h(sq))
                tasks[key] = (sq, sctx)
                rec[f"{tag}_mode"] = "stored_answer"
                rec[f"{tag}_gen_query"] = sq
                rec[f"{tag}_matched_query"] = p["matched_query"]
                rec[f"{tag}_ctx_sha"] = h(sctx)
                rec[f"{tag}_key"] = list(key)
                rec[f"{tag}_stored_answer_t"] = p["stored_answer_t"]
            else:
                pctx = ctx_for(tuple(p["ev"]))
                key = ("gen", h(pctx), h(d["query"]))
                tasks[key] = (d["query"], pctx)
                rec[f"{tag}_mode"] = ("regenerate_l2" if tier == "L2"
                                      else "regenerate_miss")
                rec[f"{tag}_gen_query"] = d["query"]
                rec[f"{tag}_ctx_sha"] = h(pctx)
                rec[f"{tag}_key"] = list(key)
                rec[f"{tag}_l3_reused_pages"] = p.get("l3_reused_pages", 0)
                rec[f"{tag}_ctx_equals_fresh"] = (h(pctx) == h(fctx))
        out.append(rec)

    with open(HERE / "sample.json", "w", encoding="utf-8") as fh:
        json.dump({"target": TARGET, "seed": SEED, "n_strata": len(strata),
                   "per_stratum_cap": per, "frame_size": len(frame),
                   "test_clusters_only": True,
                   "validation_requests_present": 0,
                   "sampling_is_policy_neutral": True,
                   "frozen_semanticttl": {"theta": theta, "kappa": c["kappa"],
                                          "ttl_seconds": ttl},
                   "composition": {k: dict(v) for k, v in comp.items()},
                   "unique_generation_tasks": len(tasks),
                   "in_support_S": sum(1 for r in out if r["in_S"]),
                   "max_context_chars": p2.MAXCHARS,
                   "min_page_chars": p2.MINCHARS,
                   "requests": out}, fh, indent=2)
    with open(HERE / "gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    say(f"\n  composition FreshCache  {dict(comp['fc'])}")
    say(f"  composition SemanticTTL {dict(comp['sttl'])}")
    say(f"  unique generation tasks {len(tasks):,} (vs {3*len(sample):,} naive)")
    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post, "implementation files changed"
    say(f"  implementation files unchanged: True")
    say(f"  wrote sample.json, gen_tasks.jsonl")
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
