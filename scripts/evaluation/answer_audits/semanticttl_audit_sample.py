#!/usr/bin/env python3
"""
scripts/evaluation/answer_audits/semanticttl_audit_sample.py

First stage of the paired SemanticTTL(kappa=1/8) vs FreshCache answer audit; generation and judging follow in
scripts/evaluation/answer_audits/semanticttl_audit_generation_judging.py.

Uses the exact 400 request IDs, timestamps, gold answers and fresh contexts from
outputs/mixed_age_audit/mixed_age_sample.jsonl. Nothing is resampled and no new sampling frame is built.

The SemanticTTL mechanism is unchanged: l1 = sim_ttl, no L2, no L3, similarity threshold 0.40, per-class fixed TTL.
The only change is that every FIXED_TTL entry is multiplied by 1/8, patched on the tiered_cache_model module at
runtime and restored afterwards. No FreshCache guard is added: no L1 equivalence gate, no entity guard, no 0.75 L2
floor, no risk model. The replay's savings are recorded as a run-time reference.

Execution path, same rules as the mixed-age audit:
  L1 hit -> the STORED answer is returned. SemanticTTL writes an entry on every non-hit request, and a non-hit
            request for this policy is a miss whose evidence is its own URLs at its own time. So a stored answer is
            always "the creating request's query answered from that request's own fresh evidence".
  miss   -> l3 = none, so every served URL is fetched at request time. The context is exactly the fresh context of
            that request.

Timestamp discipline: every page is read at version_at of its own content time, always at or before the request.
No future leakage.

Read-only on all existing outputs. Writes sttl_mixed_age_sample.jsonl, sttl_gen_tasks.jsonl (only generations not already
in outputs/mixed_age_audit/answers2.jsonl) and sttl_manifest.json to outputs/semanticttl_audit/.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import copy; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
import json
import os
import pathlib
import sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = ((OUTPUT_ROOT / "semanticttl_audit"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
CORR = (OUTPUT_ROOT / "mixed_age_audit")
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import audit_replay as P2                    # noqa: E402

KAPPA = 0.125
SCHEDULE, SEED = "zipf_uniform", 42
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # run-time reference values (outputs/reference_values.json)


def replay_semanticttl(stream):
    """SemanticTTL's own path, with stored-answer provenance for L1 hits.

    Transcribes mixed_age_replay.replay for cfg l1=sim_ttl / l2=none / l3=none with
    ttl='fixed'. No semantic guard, matching the real baseline.
    """
    L1T = exp.L1_SIM_THRESHOLD
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    n_max = len(stream)

    def ok(fc, age):
        v = exp.FIXED_TTL.get(fc, 0)
        return bool(v and age <= v)

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

        if fc == "REAL_TIME":
            search += 1
            fetches += len(own)
            rows[qid] = {"tier": "REAL_TIME", "t": t, "own": own,
                         "ev": [(u, t) for u in own]}
            continue

        hit1 = -1
        sim1 = None
        if n1:
            sims = row[l1_qi[:n1]]
            cand = np.nonzero(sims >= L1T)[0]
            if cand.size:
                okm = np.fromiter((ok(l1_fc[int(j)], t - l1_t[j])
                                   for j in cand), bool, cand.size)
                keep = cand[okm]
                if keep.size:
                    hit1 = int(keep[np.argsort(-sims[keep], kind="stable")][0])
                    sim1 = float(sims[hit1])
        if hit1 >= 0:
            l1h += 1
            src = l1_src[hit1]
            rows[qid] = {"tier": "L1", "t": t, "similarity": sim1,
                         "age": t - float(l1_t[hit1]),
                         "matched_query": l1_q[hit1],
                         "stored_answer_query": src["query"],
                         "stored_answer_ev": src["ev"],
                         "stored_answer_t": src["t"], "own": own}
            continue

        # miss: no L2 and no L3, so every own URL is fetched at t
        search += 1
        fetches += len(own)
        ev = [(u, t) for u in own]
        rows[qid] = {"tier": "miss", "t": t, "age": 0.0, "ev": ev, "own": own}

        l1_qi[n1] = IDX[q]; l1_t[n1] = t
        l1_q.append(q); l1_fc.append(fc); l1_urls.append(list(own))
        l1_src.append({"query": q, "ev": ev, "t": t})
        n1 += 1

    m = {"search_saved_pct": round(100 * (1 - search / len(stream)), 4),
         "l1_hits": l1h, "search_calls": search, "fetches": fetches}
    return m, rows


def main():
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    stream = sc.build_stream(records, SCHEDULE, SEED)

    sample = [json.loads(l) for l in
              open(CORR / "mixed_age_sample.jsonl", encoding="utf-8") if l.strip()]
    ids = [d["query_id"] for d in sample]
    assert DS.check("primary_audit_sample_requests", len(sample)) and len(set(ids)) == len(sample), "sample is not the mixed-age audit sample"
    print(f"  reusing the exact sample: {len(sample)} rows, "
          f"{len(set(ids))} unique IDs", flush=True)

    ORIG = copy.deepcopy(exp.FIXED_TTL)
    exp.FIXED_TTL.update({c: v * KAPPA for c, v in ORIG.items()})
    try:
        m, prov = replay_semanticttl(stream)
    finally:
        exp.FIXED_TTL.clear()
        exp.FIXED_TTL.update(ORIG)
    REF.record("semanticttl_audit/primary/SemanticTTL_k0.125/search_saved_pct", m["search_saved_pct"])
    print(f"  SemanticTTL kappa={KAPPA}: saved {m['search_saved_pct']:.4f}%  L1 hits {m['l1_hits']:,}",
          flush=True)
    print("  FIXED_TTL restored:", exp.FIXED_TTL == ORIG, flush=True)

    tasks, out = {}, []
    comp = Counter()
    for d in sample:
        qid = d["query_id"]
        p = prov[qid]
        comp[p["tier"]] += 1
        rec = {"query_id": qid, "query": d["query"], "gold": d["gold"],
               "fc": d["fc"], "t": d["t"],
               "fresh_ctx_sha": d["fresh_ctx_sha"],
               "fc_key": d["on_key"], "fc_tier": d["on_tier"],
               "fc_mode": d["on_mode"],
               "sttl_tier": p["tier"], "sttl_age": p.get("age", 0.0),
               "sttl_similarity": p.get("similarity")}
        if p["tier"] == "L1":
            sctx = P2.ctx_for(tuple(p["stored_answer_ev"]))
            sq = p["stored_answer_query"]
            key = ("stored", P2.h(sctx), P2.h(sq))
            tasks[key] = (sq, sctx)
            rec.update({"sttl_mode": "stored_answer", "sttl_key": list(key),
                        "sttl_gen_query": sq, "sttl_ctx_sha": P2.h(sctx),
                        "sttl_stored_answer_t": p["stored_answer_t"]})
        else:
            pctx = P2.ctx_for(tuple(p["ev"]))
            key = ("gen", P2.h(pctx), P2.h(d["query"]))
            tasks[key] = (d["query"], pctx)
            rec.update({"sttl_mode": "regenerate_miss", "sttl_key": list(key),
                        "sttl_gen_query": d["query"], "sttl_ctx_sha": P2.h(pctx),
                        "sttl_ctx_equals_fresh": (P2.h(pctx) == d["fresh_ctx_sha"])})
        out.append(rec)

    have = {tuple(r["key"]) for r in
            (json.loads(l) for l in open(CORR / "answers2.jsonl",
                                         encoding="utf-8") if l.strip())}
    new = {k: v for k, v in tasks.items() if k not in have}
    print(f"\n  SemanticTTL path composition on the 400: {dict(comp)}")
    print(f"  distinct SemanticTTL generation keys: {len(tasks)}")
    print(f"  already generated by the reported audit: {len(tasks)-len(new)}")
    print(f"  NEW generations required: {len(new)}")

    with open(HERE / "sttl_mixed_age_sample.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    with open(HERE / "sttl_gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in new.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    json.dump({"kappa": KAPPA, "n": len(out),
               "reused_sample_from": str((CORR / "mixed_age_sample.jsonl").relative_to(ROOT)),
               "same_ids": True,
               "semanticttl_replay": m,
               "reproduces_sweep_point": True,
               "path_composition": dict(comp),
               "distinct_generation_keys": len(tasks),
               "new_generations_required": len(new),
               "fixed_ttl_restored": exp.FIXED_TTL == ORIG,
               "ctx_equals_fresh_on_miss": sum(
                   1 for r in out if r.get("sttl_ctx_equals_fresh"))},
              open(HERE / "sttl_manifest.json", "w"), indent=2)
    print(f"  wrote sttl_mixed_age_sample.jsonl, sttl_gen_tasks.jsonl, "
          f"sttl_manifest.json")


if __name__ == "__main__":
    main()
