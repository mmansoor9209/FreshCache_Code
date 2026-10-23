"""Shared loader. ENGINE = reported: the engine folder goes first on sys.path so mixed_age_replay/audit_replay/execution_semantics/method_replay
resolve to it; everything else (experiment, schedules, data) is read-only project code."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, os, json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
W = ((OUTPUT_ROOT / "replay")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); ENGINE_DIRS = {"reported": ROOT / "freshcache/replay_engine", "keyset": ROOT / "freshcache/keyset_replay_engine"}   # keyset: the key-set engine, used only to rebuild the recorded comparison keys
def load(engine):
    sys.path[:0] = [str(ENGINE_DIRS[engine])] + [str(ROOT/p) if p else str(ROOT) for p in ("freshcache/method", "scripts/evaluation", "scripts/data", "", "scripts/baselines/vcache", "freshcache/selection_replay_engine", "freshcache/simulation", "scripts/evaluation/answer_audits")]
    os.chdir(ROOT)
    import numpy as np, tiered_cache_model as exp, method_replay as ea, request_schedules as sc, execution_semantics as ce, mixed_age_replay as me, audit_replay as p2
    for m in (me, p2, ea, ce): assert pathlib.Path(m.__file__).resolve().parent == ENGINE_DIRS[engine].resolve(), m.__file__
    from schedule_robustness import support_of
    recs = exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}; exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r"); ea.set_cluster_base(recs)
    rounds = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open((DATA_ROOT / "derived/round_table.jsonl"))}
    split = json.load(open((OUTPUT_ROOT / "validation_split/split.json")))
    # FreshCache without L1 (timing and ablation arm): FreshCache_Full with the answer tier removed, registered in memory only
    if "FreshCache_NoL1" not in ea.CONFIG:
        ea.CONFIG["FreshCache_NoL1"] = dict(ea.CONFIG["FreshCache_Full"], l1="none")
    me.VARIANTS.setdefault("FreshCache_NoL1", ("FreshCache_NoL1", "risk"))
    return dict(exp=exp, ea=ea, sc=sc, ce=ce, me=me, p2=p2, recs=recs, rounds=rounds, split=split, support_of=support_of)
def streams(E, schedule="zipf_uniform", seed=42):
    full = E["sc"].build_stream(E["recs"], schedule, seed); T = set(E["split"]["test_clusters"]); V = set(E["split"]["validation_clusters"])
    return {"primary": full, "heldout": [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in T],
            "validation": [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in V]}
def metrics(E, stream, variant, S=None, rich=None):
    me = E["me"]; tr = {}
    m, per = me.replay(stream, E["rounds"], variant, rich=rich, trace=tr); per = dict(per); n = len(stream)
    S = S if S is not None else E["support_of"](stream, E["rounds"])
    ch = sum(1 for q in S if per.get(q) == me.CHANGED); un = sum(1 for q in S if per.get(q) == me.UNCHANGED)
    return {"variant": variant, "n": n, "support": len(S), "search_saved_pct": m["search_saved_pct"], "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
            "fetches": m["fetches"], "generations": m["generations"], "fetch_per_1k": round(1000*m["fetches"]/n, 1), "gen_per_1k": round(1000*m["generations"]/n, 1),
            "changed": ch, "unchanged": un, "drift_pct": round(100*ch/(ch+un), 4) if ch+un else None, "coverage_pct": round(100*(ch+un)/len(S), 4)}, per, tr
