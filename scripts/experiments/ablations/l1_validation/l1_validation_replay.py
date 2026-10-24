"""Sequential validation replay for every pre-registered cell (cosine floor x structural-guard subset). Validation clusters only; results are written to outputs/l1_validation/outputs/replay/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, hashlib, pathlib, sys, os, time, collections; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "l1_validation")); OUT = W/"outputs/replay"
sys.path.insert(0, str(ROOT / "freshcache/method"))
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "scripts/evaluation/answer_audits", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT/p) if p else str(ROOT))
os.chdir(ROOT)
import numpy as np, tiered_cache_model as exp, method_replay as ea, mixed_age_replay as me, request_schedules as sc, audit_replay as p2
from schedule_robustness import support_of
import l1_structural_guards as G
P = json.load(open(W/"config/prereg.json")); assert hashlib.sha256((W/"config/prereg.json").read_bytes()).hexdigest() == open(W/"config/prereg.sha256").read().strip()
assert hashlib.sha256(((ROOT / "freshcache/method/l1_structural_guards.py")).read_bytes()).hexdigest() == "952e8c9cb59a796328481bc8cc23d7dc9042db09cd96980d6be9fecc972c3873"
h, ctx_for = p2.h, p2.ctx_for
GF = {"numeric": G.numeric_guard, "negation": G.negation_guard, "comparative": G.comparative_guard}
def make_gate(sim_floor, guards):
    fs = [GF[g] for g in guards]
    def semantic_equivalent(q1, q2, sim):                      # the recorded gate (semantic_equivalent in freshcache/method/tiered_cache_model.py) with a cosine floor and optional guards
        if sim < sim_floor: return False
        t1, t2 = exp._eq_content_tokens(q1), exp._eq_content_tokens(q2); u = t1 | t2
        if not u or len(t1 & t2)/len(u) < exp._EQ_JACCARD_MIN: return False
        a1, a2 = exp._eq_answer_type(q1), exp._eq_answer_type(q2)
        if a1 is not None and a2 is not None and a1 != a2: return False
        if t1 != t2 and (t1 <= t2 or t2 <= t1): return False
        return all(f(q1, q2) for f in fs)
    return semantic_equivalent
recs = exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))
exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}; exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r"); ea.set_cluster_base(recs)
rounds = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open((DATA_ROOT / "derived/round_table.jsonl"))}
split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"))); VAL, TEST = set(split["validation_clusters"]), set(split["test_clusters"])
stream = [(t, r) for t, r in sc.build_stream(recs, "zipf_uniform", 42) if (r.get("cluster_id") or r["query_id"]) in VAL]
assert DS.check("validation_requests", len(stream)) and not any((r.get("cluster_id") or r["query_id"]) in TEST for _, r in stream)
S = support_of(stream, rounds); N = len(stream); cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in recs}; qcl = {r["query"]: cid[r["query_id"]] for r in recs}
gold = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid, qt):
    g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
AUD = json.load(open((OUTPUT_ROOT / "l1_precision/validation_audit_sample.json")))["requests"]; AUDQ = {d["query_id"] for d in AUD}
def drift(rows):
    ch = un = 0
    for t, r in stream:
        q = r["query_id"]; p = rows.get(q)
        if not p or q not in S: continue
        if p["tier"] == "L1": labs = [me.outcome(x, p["t_cached"], t, rounds) for x in p.get("served", [])]
        elif p["tier"] in ("L2", "miss"): labs = [me.outcome(u, tc, t, rounds) for u, tc in p.get("ev", []) if tc < t]
        else: labs = []
        o = me.CHANGED if me.CHANGED in labs else (me.UNCHANGED if me.UNCHANGED in labs else None)
        ch += o == me.CHANGED; un += o == me.UNCHANGED
    return ch, un
ORIG = exp.semantic_equivalent
m0, rows0 = p2.replay(stream, {}, "FreshCache")
cells = [(s, g) for s in P["grid"]["cosine_floor"] for g in P["grid"]["guard_subsets"]]
res, gen_tasks, pairs, log = [], {}, {}, []
for i, (sf, gd) in enumerate(cells):
    tag = f"s{sf:.2f}_" + ("+".join(gd) if gd else "noguard"); t0 = time.time()
    exp.semantic_equivalent = make_gate(sf, gd)
    try: m, rows = p2.replay(stream, {}, "FreshCache")
    finally: exp.semantic_equivalent = ORIG
    if (sf, gd) == (0.80, []):
        assert m == m0 and all(rows[q]["tier"] == rows0[q]["tier"] for q in rows0), "recorded-rule reproduction gate failed"
    hits, audit = [], {}
    for t, r in stream:
        q = r["query_id"]; p = rows[q]
        if p["tier"] == "L1":
            sq = p["stored_answer_query"]; sctx = ctx_for(tuple(p["stored_answer_ev"])); fctx = ctx_for(tuple((u["url_hash"], t) for u in r["urls"]))
            sk, fk = ("stored", h(sctx), h(sq)), ("fresh", h(fctx), h(r["query"]))
            gen_tasks.setdefault(sk, {"key": list(sk), "question": sq, "context": sctx}); gen_tasks.setdefault(fk, {"key": list(fk), "question": r["query"], "context": fctx})
            pid = f"{h(r['query'])}_{h(p['matched_query'])}"; pairs.setdefault(pid, {"pair_id": pid, "incoming": r["query"], "cached": p["matched_query"]})
            hits.append({"query_id": q, "pair_id": pid, "stored_key": list(sk), "fresh_key": list(fk), "question": r["query"], "gold": goldof(q, r["query"]),
                         "cross_cluster": int(qcl.get(p["matched_query"]) != cid[q]), "similarity": p.get("similarity")})
        if q in AUDQ:
            if p["tier"] == "L1": k = ("stored", h(ctx_for(tuple(p["stored_answer_ev"]))), h(p["stored_answer_query"])); qq, cc = p["stored_answer_query"], ctx_for(tuple(p["stored_answer_ev"]))
            else: cc = ctx_for(tuple(p["ev"])); k = ("gen", h(cc), h(r["query"])); qq = r["query"]
            gen_tasks.setdefault(k, {"key": list(k), "question": qq, "context": cc}); audit[q] = {"tier": p["tier"], "key": list(k)}
    ch, un = drift(rows); gens = m["search_calls"] + m["l2_hits"]
    row = {"cell": tag, "cosine_floor": sf, "guards": gd, "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "search_calls": m["search_calls"],
           "search_saved_pct": round(m["search_saved_pct"], 4), "generations": gens, "generation_saved_pct": round(100*(1 - gens/N), 4),
           "drift_pct": round(100*ch/(ch+un), 4) if ch+un else None, "coverage_pct": round(100*(ch+un)/len(S), 4), "cross_cluster_hits": sum(x["cross_cluster"] for x in hits)}
    res.append(row); json.dump({"row": row, "hits": hits, "audit": audit}, open(OUT/f"{tag}.json", "w"))
    msg = f"[{i+1}/{len(cells)}] {tag:<40} L1 {row['l1_hits']:>4} saved {row['search_saved_pct']:.4f} gen_saved {row['generation_saved_pct']:.4f} ({time.time()-t0:.0f}s)"; print(msg, flush=True)
chk = next(r for r in res if r["cell"] == "s0.90_numeric+negation+comparative")
import csv as _csv; ref = next(r for r in _csv.DictReader(open(OUTPUT_ROOT / "l1_precision/validation_grid.csv")) if r["cell"] == "s0.9_j0.3_as_implemented_G"); assert (chk["l1_hits"], chk["search_saved_pct"]) == (int(ref["l1_hits"]), round(float(ref["search_saved_pct"]), 4)), ("earlier-study reproduction gate failed", chk, ref)
json.dump(res, open(OUT/"cells.json", "w"), indent=1)
for name, d in (("gen_tasks.jsonl", gen_tasks), ("jury_pairs.jsonl", pairs)):
    with open(W/"inputs"/name, "w", encoding="utf-8") as f:
        for v in d.values(): f.write(json.dumps(v, ensure_ascii=False) + "\n")
print(f"GATES PASSED; cells {len(res)}; generation keys {len(gen_tasks)}; jury pairs {len(pairs)}", flush=True)
