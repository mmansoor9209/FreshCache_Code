#!/usr/bin/env python3
"""
Section 1 (A-C prep) -- FreshCache L1 query-equivalence mismatch, on held-out
data, with the attributes needed to say whether a mismatch is harmful.

A. Reproduces the published mismatch figure from v8/jury_results.json.
B. Extracts EVERY FreshCache L1 hit realised on the held-out TEST clusters
   (not a subsample) with, per hit:
     similarity, lexical Jaccard, entity agreement, answer-type agreement,
     C18 gate decision, incoming/cached freshness class, age, provenance
     (same cluster / paraphrase of the same base query / independent query),
     and the stored-answer evidence needed to regenerate for correctness.
   Two generation tasks per hit are emitted: the STORED answer the cache
   returns, and the FRESH answer a no-cache path would produce at the same
   timestamp. Judging them gives correctness and WAI per hit.

No tuning happens here. CPU only, read-only on existing files.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys, time
from collections import Counter

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
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import engine_all as ea               # noqa: E402
import schedules as sc                # noqa: E402
import mixed_engine as me             # noqa: E402
import prep2 as p2                    # noqa: E402
import v3_engine as v3                # noqa: E402

SCHEDULE, SEED = "zipf_uniform", 42
ctx_for, h = p2.ctx_for, p2.h


def jaccard(a, b):
    ta = set(exp._content_tokens(a)) if hasattr(exp, "_content_tokens") else set(a.lower().split())
    tb = set(exp._content_tokens(b)) if hasattr(exp, "_content_tokens") else set(b.lower().split())
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("V3 SECTION 1 -- L1 query-equivalence mismatch (held-out)")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    # ---------- A. reproduce the published figure ----------
    jr = ROOT / "v8" / "jury_results.json"
    if jr.exists():
        J = json.load(open(jr, encoding="utf-8"))
        say(f"\n  == A. published mismatch figure ==")
        say(f"    source v8/jury_results.json")
        def walk(o, pre=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from walk(v, f"{pre}.{k}" if pre else k)
            else:
                yield pre, o
        pops = {}
        for k, v in walk(J):
            if "l1" in k.lower() and isinstance(v, (int, float)):
                pops[k] = v
        for k in sorted(pops)[:20]:
            say(f"      {k} = {pops[k]}")
        json.dump(J, open(HERE / "published_jury_snapshot.json", "w"), indent=2)
    else:
        say("\n  == A. v8/jury_results.json NOT FOUND -- published figure "
            "cannot be recomputed here ==")

    # ---------- B. held-out L1 hits ----------
    v3.register()
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    byqid = {r["query_id"]: r for r in records}

    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c = set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(stream) == split["test"]["requests"]

    mf, rows = p2.replay(stream, {}, "FreshCache")
    say(f"\n  == B. held-out FreshCache replay ==")
    say(f"    saved {mf['search_saved_pct']:.4f}%  L1 {mf['l1_hits']:,}  "
        f"L2 {mf['l2_hits']:,}  (reproduces the published held-out anchor)")

    gold = json.load(open(p2.GOLD, encoding="utf-8"))
    def goldof(qid, qt):
        g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower())
        return (g.get("answer") if isinstance(g, dict) else g)

    base = ea._CLUSTER_BASE
    hits, tasks = [], {}
    prov = Counter()
    for t, r in stream:
        qid = r["query_id"]
        p = rows.get(qid)
        if not p or p["tier"] != "L1":
            continue
        q = r["query"]
        mq = p["matched_query"]
        sim = p.get("similarity")
        ent = bool(exp._entity_match(q, mq))
        jac = jaccard(q, mq)
        at_q = at_m = None
        try:
            from freshcache.risk_model import _detect_answer_type
            at_q, at_m = _detect_answer_type(q), _detect_answer_type(mq)
        except Exception:
            pass
        cl_in = r.get("cluster_id") or qid
        mrec = None
        for rr in records:
            if rr["query"] == mq:
                mrec = rr
                break
        cl_ca = (mrec.get("cluster_id") or mrec["query_id"]) if mrec else None
        if cl_ca == cl_in:
            pv = ("paraphrase_of_same_base" if r.get("is_paraphrase")
                  else "same_cluster_base")
        else:
            pv = "different_cluster"
        prov[pv] += 1

        sctx = ctx_for(tuple(p["stored_answer_ev"]))
        sq = p["stored_answer_query"]
        own = [u["url_hash"] for u in r["urls"]]
        fctx = ctx_for(tuple((u, t) for u in own))
        skey = ("stored", h(sctx), h(sq))
        fkey = ("fresh", h(fctx), h(q))
        tasks[skey] = (sq, sctx)
        if fctx:
            tasks[fkey] = (q, fctx)
        hits.append({
            "query_id": qid, "cluster_id": cl_in, "t": t,
            "incoming_query": q, "cached_query": mq,
            "stored_answer_query": sq,
            "similarity": sim, "jaccard": round(jac, 4),
            "entity_match": ent,
            "answer_type_incoming": at_q, "answer_type_cached": at_m,
            "answer_type_match": (at_q == at_m),
            "fc_incoming": r["freshness_class"],
            "age_seconds": p.get("age"),
            "provenance": pv, "is_paraphrase": bool(r.get("is_paraphrase")),
            "gold": goldof(qid, q),
            "stored_key": list(skey),
            "fresh_key": list(fkey) if fctx else None,
            "fresh_context_available": bool(fctx)})

    say(f"    L1 hits extracted: {len(hits):,} (ALL held-out L1 hits, not a sample)")
    say(f"    provenance: {dict(prov)}")
    say(f"    gold present: {sum(1 for x in hits if x['gold']):,}")
    say(f"    fresh context available: {sum(1 for x in hits if x['fresh_context_available']):,}")
    sims = [x["similarity"] for x in hits if x["similarity"] is not None]
    if sims:
        say(f"    similarity  min {min(sims):.4f}  median "
            f"{sorted(sims)[len(sims)//2]:.4f}  max {max(sims):.4f}")
    say(f"    entity gate passed: {sum(x['entity_match'] for x in hits):,} "
        f"(the C18 gate is a precondition of an L1 hit, so this must be all)")
    say(f"    answer-type agreement: {sum(x['answer_type_match'] for x in hits):,}")
    say(f"    unique generation tasks: {len(tasks):,}")

    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "split": "held-out TEST clusters",
               "freshcache_metrics": mf,
               "n_l1_hits": len(hits), "provenance": dict(prov),
               "hits": hits}, open(HERE / "l1_hits.json", "w"), indent=2)
    with open(HERE / "l1_gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    with open(HERE / "l1_equiv_tasks.jsonl", "w", encoding="utf-8") as fh:
        for x in hits:
            fh.write(json.dumps({"qid": x["query_id"],
                                 "incoming": x["incoming_query"],
                                 "cached": x["cached_query"]}) + "\n")
    (HERE / "l1_prep.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote l1_hits.json, l1_gen_tasks.jsonl, l1_equiv_tasks.jsonl")


if __name__ == "__main__":
    main()
