#!/usr/bin/env python3
"""
heldout_sensitivity_sample.py — replay the FROZEN SemanticTTL(theta=0.60, kappa=1/2) on the
held-out TEST stream and build its served contexts for the existing
audit sample. Also records held-out operational metrics for the frozen config
and for FreshCache.

Reuses the audit's own sample, fresh contexts, and FreshCache/fresh answers
unchanged. Only the new SemanticTTL arm is prepared.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import copy, hashlib, json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = ((OUTPUT_ROOT / "quality_selection/heldout_sensitivity"))
QAB = ((OUTPUT_ROOT / "quality_selection"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
CORR = (OUTPUT_ROOT / "mixed_age_audit")
A2 = (OUTPUT_ROOT / "general_audit")
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))
os.chdir(ROOT)
import numpy as np                        # noqa: E402
import tiered_cache_model as exp                  # noqa: E402
import execution_semantics as ce             # noqa: E402
import method_replay as ea                   # noqa: E402
import request_schedules as sc                    # noqa: E402
import mixed_age_replay as me                 # noqa: E402
import audit_replay as P2                        # noqa: E402
from schedule_robustness import support_of      # noqa: E402
from quality_selection_sample import sttl_replay           # noqa: E402  (same SemanticTTL replay as the quality-selection sample)

h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]   # noqa


def main():
    fz = json.load(open(HERE / "freeze.json", encoding="utf-8"))
    C = fz["config"]
    print(f"  frozen config: theta={C['theta']:.2f} kappa={C['kappa_frac']}  "
          f"freeze sha {fz['freeze_sha256'][:20]}...")
    assert fz["primary_constraint_result"].startswith("FAILED")

    split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"), encoding="utf-8"))
    TEST = set(split["test_clusters"]); VAL = set(split["validation_clusters"])
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats(); ea.set_cluster_base(records)
    rounds = {}
    with open((DATA_ROOT / "derived/round_table.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
    full = sc.build_stream(records, "zipf_uniform", 42)
    ts = [(t, r) for t, r in full
          if (r.get("cluster_id") or r["query_id"]) in TEST]
    assert len(ts) == split["test"]["requests"]
    assert not any((r.get("cluster_id") or r["query_id"]) in VAL for _, r in ts)
    S = support_of(ts, rounds)
    print(f"  held-out stream: {len(ts):,} requests, |S| = {len(S):,}, "
          f"validation clusters present: 0")

    ORIG = copy.deepcopy(exp.FIXED_TTL); L1T0 = exp.L1_SIM_THRESHOLD

    def score(m, per):
        ins = [o for qid, o in per if o is not None and qid in S]
        ch = sum(1 for o in ins if o == ce.CHANGED)
        un = sum(1 for o in ins if o == ce.UNCHANGED)
        return {"search_saved_pct": m["search_saved_pct"],
                "drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None,
                "changed": ch, "determinable": ch + un,
                "coverage_pct": round(100 * (ch + un) / len(S), 4),
                "fetches": m["fetches"], "fetches_per_1k": round(1000 * m["fetches"] / len(ts), 2),
                "generations": m["generations"], "l1_hits": m["l1_hits"],
                "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
                "n_requests": len(ts), "support_S": len(S)}

    ops = {}
    try:
        mfc, pfc = me.replay(ts, rounds, "FreshCache", rich=rich)
        ops["FreshCache"] = score(mfc, pfc)
        exp.L1_SIM_THRESHOLD = C["theta"]
        exp.FIXED_TTL.clear()
        exp.FIXED_TTL.update({k: v * C["kappa"] for k, v in ORIG.items()})
        msn, psn = me.replay(ts, rounds, "SemanticTTL", rich=rich)
        ops["SemanticTTL_t0.60_k1_2"] = score(msn, psn)
    finally:
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
        exp.L1_SIM_THRESHOLD = L1T0
    # reported SemanticTTL configuration (theta=0.40, kappa=1/16), for the three-way comparison
    try:
        exp.L1_SIM_THRESHOLD = 0.40
        exp.FIXED_TTL.clear()
        exp.FIXED_TTL.update({k: v * 0.0625 for k, v in ORIG.items()})
        mi, pi = me.replay(ts, rounds, "SemanticTTL", rich=rich)
        ops["SemanticTTL_t0.40_k1_16_recorded"] = score(mi, pi)
    finally:
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
        exp.L1_SIM_THRESHOLD = L1T0
    assert exp.FIXED_TTL == ORIG and exp.L1_SIM_THRESHOLD == L1T0
    for k, v in ops.items():
        print(f"    {k:<38} saved {v['search_saved_pct']:8.4f}%  drift "
              f"{v['drift_pct']:7.4f}% ({v['changed']}/{v['determinable']})  "
              f"cov {v['coverage_pct']:7.4f}%")

    # provenance replay for the frozen config, to build contexts
    try:
        _, rws = sttl_replay(ts, C["theta"], C["kappa"], ORIG)
    finally:
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
        exp.L1_SIM_THRESHOLD = L1T0

    sample = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    assert DS.check("audit_sample_requests", len(sample))
    old_tasks = {tuple(t["key"]): t for t in
                 (json.loads(l) for l in open(A2 / "gen_tasks.jsonl", encoding="utf-8"))}
    rows, tasks = [], {}
    for d in sample:
        qid = d["query_id"]
        p = rws.get(qid)
        if p is None:
            raise SystemExit(f"request {qid} absent from the held-out replay")
        ctx = P2.ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""
        gq = p.get("gen_query", d["query"])
        k = ("gen", h(ctx), h(gq))
        tasks[k] = {"key": list(k), "question": gq, "context": ctx}
        rows.append({"query_id": qid, "query": d["query"], "gold": d["gold"],
                     "fc": d["fc"], "t": d["t"], "cluster_id": d["cluster_id"],
                     "new_sttl_key": list(k), "new_sttl_tier": p["tier"],
                     "new_sttl_similarity": p.get("similarity"),
                     "new_sttl_age": p.get("age"),
                     "reused_from_existing_audit": int(tuple(k) in old_tasks)})
    reuse = sum(r["reused_from_existing_audit"] for r in rows)
    print(f"\n  sample: 400 held-out requests (unchanged)")
    print(f"    new arm tier mix: {dict(Counter(r['new_sttl_tier'] for r in rows))}")
    print(f"    unique new generation tasks: {len(tasks)}")
    print(f"    of those already present in the existing audit's gen_tasks: {reuse}")
    with open(HERE / "h_sample.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(HERE / "h_gen_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks.values():
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    json.dump({"freeze_sha256": fz["freeze_sha256"], "heldout_operational": ops,
               "sample": len(sample), "unique_new_gen_tasks": len(tasks),
               "tasks_already_in_existing_audit": reuse,
               "validation_clusters_in_heldout_stream": 0},
              open(HERE / "heldout_sample_summary.json", "w"), indent=2)
    print("  wrote h_sample.jsonl, h_gen_tasks.jsonl, heldout_sample_summary.json")


if __name__ == "__main__":
    main()
