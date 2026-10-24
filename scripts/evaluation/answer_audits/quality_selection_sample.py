#!/usr/bin/env python3
"""
quality_selection_sample.py — build the frozen validation sample and every policy's served
evidence, then emit generation tasks.

Validation clusters only. FreshCache uses audit_replay.replay (its real path with full
per-request provenance); each SemanticTTL candidate uses a parameterised
transcription of the SemanticTTL replay at that (theta, kappa).
Contexts use audit_replay.ctx_for, identical to the other answer audits.

Read-only on every pre-existing artifact. No web retrieval.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import copy, csv, hashlib, json, os, pathlib, random, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = ((OUTPUT_ROOT / "quality_selection"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
CORR = (OUTPUT_ROOT / "mixed_age_audit")
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))
os.chdir(ROOT)

import numpy as np                      # noqa: E402
import tiered_cache_model as exp                # noqa: E402
import method_replay as ea                 # noqa: E402
import request_schedules as sc                  # noqa: E402
import audit_replay as P2                      # noqa: E402

SCHEDULE, SEED = "zipf_uniform", 42
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]  # noqa


def sttl_replay(stream, theta, kappa, orig_ttl):
    """SemanticTTL (l1=sim_ttl, no L2/L3) at (theta, kappa), with provenance."""
    exp.L1_SIM_THRESHOLD = theta
    exp.FIXED_TTL.clear()
    exp.FIXED_TTL.update({c: v * kappa for c, v in orig_ttl.items()})
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    n_max = len(stream)
    ok = lambda fc, age: bool(exp.FIXED_TTL.get(fc, 0)                      # noqa
                              and age <= exp.FIXED_TTL.get(fc, 0))
    l1_qi = np.empty(n_max, np.int64); l1_t = np.empty(n_max, np.float64)
    l1_q, l1_fc, l1_src = [], [], []
    n1 = l1h = search = 0
    rows = {}
    for t, r in stream:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]; row = SIM[IDX[q]]
        own = [u["url_hash"] for u in urls]
        if fc == "REAL_TIME":
            search += 1
            rows[qid] = {"tier": "REAL_TIME", "t": t, "ev": [(u, t) for u in own],
                         "gen_query": q}
            continue
        hit = -1; sim = None
        if n1:
            sims = row[l1_qi[:n1]]
            cand = np.nonzero(sims >= theta)[0]
            if cand.size:
                keep = cand[np.fromiter((ok(l1_fc[int(j)], t - l1_t[j])
                                         for j in cand), bool, cand.size)]
                if keep.size:
                    hit = int(keep[np.argsort(-sims[keep], kind="stable")][0])
                    sim = float(sims[hit])
        if hit >= 0:
            l1h += 1; src = l1_src[hit]
            rows[qid] = {"tier": "L1", "t": t, "similarity": sim,
                         "age": t - float(l1_t[hit]), "matched_query": l1_q[hit],
                         "ev": src["ev"], "gen_query": src["query"],
                         "stored_answer_t": src["t"]}
            continue
        search += 1
        ev = [(u, t) for u in own]
        rows[qid] = {"tier": "miss", "t": t, "age": 0.0, "ev": ev, "gen_query": q}
        l1_qi[n1] = IDX[q]; l1_t[n1] = t
        l1_q.append(q); l1_fc.append(fc); l1_src.append({"query": q, "ev": ev, "t": t})
        n1 += 1
    return ({"search_saved_pct": round(100 * (1 - search / len(stream)), 4),
             "l1_hits": l1h, "search_calls": search}, rows)


def main():
    pre = json.load(open((OUTPUT_ROOT / "quality_selection/quality_selection_prereg.json"), encoding="utf-8"))
    print(f"  prereg sha256 {pre['prereg_sha256'][:24]}...  "
          f"{pre['n_candidates']} candidates, target n={pre['sample']['target_n']}")

    split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"), encoding="utf-8"))
    VAL = set(split["validation_clusters"]); TEST = set(split["test_clusters"])
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

    full = sc.build_stream(records, SCHEDULE, SEED)
    assert DS.check("primary_requests", len(full))
    vstream = [(t, r) for t, r in full
               if (r.get("cluster_id") or r["query_id"]) in VAL]
    assert len(vstream) == split["validation"]["requests"], len(vstream)
    assert not any((r.get("cluster_id") or r["query_id"]) in TEST
                   for _, r in vstream), "test cluster leaked into the frame"
    print(f"  validation stream: {len(vstream):,} requests, "
          f"{len({(r.get('cluster_id') or r['query_id']) for _, r in vstream}):,} clusters"
          f"  (test clusters used: 0)")

    ORIG = copy.deepcopy(exp.FIXED_TTL); L1T0 = exp.L1_SIM_THRESHOLD
    try:
        fc_m, fc_rows = P2.replay(vstream, rounds, "FreshCache")
    finally:
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
        exp.L1_SIM_THRESHOLD = L1T0
    print(f"  FreshCache on validation: saved {fc_m['search_saved_pct']:.4f}%  "
          f"L1 {fc_m['l1_hits']:,}  L2 {fc_m['l2_hits']:,}")

    gold = json.load(open((OUTPUT_ROOT / "gold_answers/reference_answers.json"),
                          encoding="utf-8"))
    import re
    nq = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())            # noqa
    byq = {r["query_id"]: r for _, r in vstream}
    elig = [qid for qid in fc_rows if gold.get(nq(byq[qid]["query"]))]
    print(f"  requests with a gold answer: {len(elig):,} of {len(fc_rows):,}")

    rng = random.Random(pre["sample"]["seed"])
    strata = defaultdict(list)
    for qid in sorted(elig):
        strata[(byq[qid]["freshness_class"], fc_rows[qid]["tier"])].append(qid)
    target = pre["sample"]["target_n"]
    keys = sorted(strata)
    per = max(1, target // len(keys))
    sample = []
    for k in keys:
        v = sorted(strata[k]); rng.shuffle(v)
        sample += v[:per]
    pool = [q for k in keys for q in sorted(strata[k]) if q not in set(sample)]
    rng.shuffle(pool)
    sample += pool[:max(0, target - len(sample))]
    sample = sorted(set(sample))[:target]
    print(f"  sampled {len(sample)} requests over {len(keys)} strata "
          f"(freshness class x FreshCache tier)")
    print(f"    tier mix: {dict(Counter(fc_rows[q]['tier'] for q in sample))}")
    print(f"    class mix: {dict(Counter(byq[q]['freshness_class'] for q in sample))}")

    arms = {"fc": fc_rows}
    ops = {"fc": fc_m}
    for c in pre["candidates"]:
        try:
            m, rws = sttl_replay(vstream, c["theta"], c["kappa"], ORIG)
        finally:
            exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
            exp.L1_SIM_THRESHOLD = L1T0
        gap = abs(m["search_saved_pct"] - c["saved"])
        print(f"    {c['arm']:<24} replay saved {m['search_saved_pct']:8.4f}%  "
              f"grid {c['saved']:8.4f}%  delta {gap:.4f}")
        arms[c["arm"]] = rws; ops[c["arm"]] = m
    assert exp.FIXED_TTL == ORIG and exp.L1_SIM_THRESHOLD == L1T0, "globals not restored"

    tasks, rows = {}, []
    for qid in sample:
        rec = byq[qid]; q = rec["query"]
        g = gold.get(nq(q)); gtext = g["answer"]
        t = fc_rows[qid]["t"]
        fresh_ev = tuple((u["url_hash"], t) for u in rec["urls"])
        fctx = P2.ctx_for(fresh_ev)
        row = {"query_id": qid, "query": q, "gold": gtext,
               "aliases": (g.get("aliases") or [])[:8],
               "gold_source": g.get("gold_source", ""),
               "gold_year": g.get("gold_year", "") or "",
               "fc": rec["freshness_class"], "t": t,
               "cluster_id": rec.get("cluster_id") or qid,
               "fresh_ctx_sha": h(fctx)}
        tasks[("fresh", h(fctx), h(q))] = {"key": ["fresh", h(fctx), h(q)],
                                           "question": q, "context": fctx}
        for arm, rws in arms.items():
            p = rws[qid]
            ctx = P2.ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""
            gq = p.get("gen_query", q)
            k = ("gen", h(ctx), h(gq))
            tasks[k] = {"key": list(k), "question": gq, "context": ctx}
            row[f"{arm}_key"] = list(k)
            row[f"{arm}_tier"] = p["tier"]
            row[f"{arm}_similarity"] = p.get("similarity")
            row[f"{arm}_age"] = p.get("age")
            row[f"{arm}_ctx_equals_fresh"] = int(h(ctx) == h(fctx))
        rows.append(row)

    with open(HERE / "q_sample.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(HERE / "q_gen_tasks.jsonl", "w", encoding="utf-8") as f:
        for t_ in tasks.values():
            f.write(json.dumps(t_, ensure_ascii=False) + "\n")
    json.dump({"validation_requests": len(vstream),
               "validation_clusters": len(VAL), "test_clusters_used": 0,
               "eligible_with_gold": len(elig), "sampled": len(rows),
               "unique_generation_tasks": len(tasks),
               "arms": sorted(arms), "operational": ops,
               "tier_mix": {a: dict(Counter(arms[a][q]["tier"] for q in sample))
                            for a in arms},
               "prereg_sha256": pre["prereg_sha256"], "web_calls": 0},
              open(HERE / "quality_sample_summary.json", "w"), indent=2)
    print(f"  unique generation tasks: {len(tasks):,} "
          f"(for {len(rows)} requests x {len(arms)+1} arms)")
    print("  wrote q_sample.jsonl, q_gen_tasks.jsonl, quality_sample_summary.json")


if __name__ == "__main__":
    main()
