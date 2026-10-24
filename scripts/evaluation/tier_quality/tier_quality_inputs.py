"""Stage 0: deterministic held-out FreshCache replay (read-only project code) and construction of each path's (question, context).
Covers every realized held-out L1 hit and the FreshCache L2 hits of the held-out audit. Writes only into outputs/tier_quality/inputs/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, csv, hashlib, pathlib, sys, os, re, collections, glob; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "tier_quality")); IN = W/"inputs"
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "scripts/evaluation/answer_audits", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT/p) if p else str(ROOT))
os.chdir(ROOT)
import numpy as np, tiered_cache_model as exp, method_replay as ea, request_schedules as sc, audit_replay as p2, answer_grading as e2e, page_version_replay as ma
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
h, ctx_for = p2.h, p2.ctx_for
A2 = (OUTPUT_ROOT / "general_audit"); L1A = (OUTPUT_ROOT / "l1only_exactttl/l1_hit_analysis")
HITS = (OUTPUT_ROOT / "l1_precision/heldout_l1_hits_judged.csv")
SRC = [(OUTPUT_ROOT / "l1_precision/heldout_l1_hits.csv"), A2/"sample.json", A2/"answers.jsonl", L1A/"l1_answers.jsonl", HITS, (OUTPUT_ROOT / "validation_split/split.json"), (OUTPUT_ROOT / "reference_validation/results/reference_validation_results.csv")]
recs = exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))
exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}; exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r"); ea.set_cluster_base(recs)
split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"))); TEST = set(split["test_clusters"])
stream = [(t, r) for t, r in sc.build_stream(recs, "zipf_uniform", 42) if (r.get("cluster_id") or r["query_id"]) in TEST]
m, rows = p2.replay(stream, {}, "FreshCache")
gate = {"n": len(stream), "search_saved_pct": round(m["search_saved_pct"], 4), "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"]}
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); assert all(abs(m[k] - REF.value("selection_engine/heldout/FreshCache/" + k)) <= 1e-4 for k in ("search_saved_pct", "l1_hits", "l2_hits")) and DS.check("heldout_requests", gate["n"]), gate
print("ANCHOR GATE PASSED", gate, flush=True)
preserved = {}
for f in (A2/"answers.jsonl", L1A/"l1_answers.jsonl"):
    for r in jl(f): preserved.setdefault(tuple(r["key"]), {"answer": r["answer"], "source": str(f.relative_to(ROOT))})
T = {t_r[1]["query_id"]: t_r for t_r in stream}
# ---------- L1 reuse study: every realized held-out L1 hit ----------
hitrow = {r["request_id"]: r for r in csv.DictReader(open(HITS, encoding="utf-8")) if r["arm"] == "recorded"}
for r in csv.DictReader(open((OUTPUT_ROOT / "l1_precision/heldout_l1_hits.csv"), encoding="utf-8")):
    if r["arm"] == "recorded": hitrow[r["request_id"]]["stored_key"] = r["stored_key"]
t1, gates1 = [], collections.Counter()
for qid, p in rows.items():
    if p["tier"] != "L1": continue
    t, r = T[qid]; q = r["query"]; sq = p["stored_answer_query"]; own = [u["url_hash"] for u in r["urls"]]
    sctx = ctx_for(tuple(p["stored_answer_ev"])); fctx = ctx_for(tuple((u, t) for u in own)); hr = hitrow[qid]
    A = ("stored", h(sctx), h(sq)); B = ("gen", h(sctx), h(q)); C = ("fresh", h(fctx), h(q))
    gates1["stored_key_matches_recorded_record"] += int("|".join(A) == hr["stored_key"])
    t1.append({"query_id": qid, "cluster_id": r.get("cluster_id") or qid, "freshness_class": r["freshness_class"], "t": t, "incoming_query": q,
               "stored_answer_query": sq, "matched_query": p["matched_query"], "similarity": p.get("similarity"), "age_s": p.get("age"),
               "reference_answer": hr["reference_answer"], "equivalence_4juror": hr["equivalence"], "cross_cluster": hr["cross_cluster"],
               "stored_context_empty": not sctx, "fresh_context_empty": not fctx,
               "paths": {"A_stored_answer": {"key": A, "question": sq, "context": sctx}, "B_regen_stored_evidence": {"key": B, "question": q, "context": sctx},
                         "C_fresh": {"key": C, "question": q, "context": fctx}}})
assert gates1["stored_key_matches_recorded_record"] == m["l1_hits"], gates1   # every realized held-out L1 hit (l1_hits gated above)
# ---------- L2 reuse study: FreshCache L2 hits of the held-out audit ----------
S = json.load(open(A2/"sample.json"))["requests"]; man = {}
for mm in exp.load_jsonl(exp.MANIFEST_FILE):
    if mm.get("run_id") == "run_00" and mm["url_hash"] not in man: man[mm["url_hash"]] = {"title": mm.get("title", "") or "", "snippet": mm.get("snippet", "") or ""}
def evblock(ev):   # verbatim evidence formatting of scripts/experiments/admission_controls/verifier_answer_audit.py:101-113
    lines = []
    for uh, tv in ev:
        mt = man.get(uh, {}); b = re.sub(r"\s+", " ", (e2e.load_snapshot_text(uh, ma.version_at(tv)) or "")).strip()[:2000]
        blk = f"{len(lines)+1}. {mt.get('title','').strip()}"
        if mt.get("snippet"): blk += f"\n   {mt['snippet'].strip()}"
        if len(b) >= 200: blk += f"\n   PAGE TEXT: {b}"
        lines.append(blk)
    return "\n".join(lines) or "(no stored evidence)"
t2, gates2 = [], collections.Counter()
for d in S:
    if d["fc_tier"] != "L2": continue
    qid = d["query_id"]; p = rows[qid]; t, r = T[qid]; q = r["query"]; own = [u["url_hash"] for u in r["urls"]]
    evA = [(u, t) for u in own]; evB = [(u, t) for u in p["served"]]; evC = [tuple(x) for x in p["ev"]]
    cA, cB, cC = ctx_for(tuple(evA)), ctx_for(tuple(evB)), ctx_for(tuple(evC))
    gates2["A_ctx_matches_audit_fresh_key"] += int(h(cA) == d["fresh_key"][1]); gates2["C_ctx_matches_audit_fc_key"] += int(h(cC) == d["fc_key"][1])
    gates2["B_equals_C_context"] += int(cB == cC); gates2["C_has_l3_reused_page"] += int(any(tc != t for _, tc in evC))
    t2.append({"query_id": qid, "cluster_id": d["cluster_id"], "freshness_class": d["fc"], "t": t, "question": q, "reference_answer": d["gold"],
               "served_urls": p["served"], "list_discovery_time": p["t_cached"], "list_age_s": p["age"],
               "paths": {"A_fresh_retrieval": {"key": ("fresh", h(cA), h(q)), "question": q, "context": cA, "ev": evA, "evidence_block": evblock(evA)},
                         "B_cached_list_current_pages": {"key": ("gen", h(cB), h(q)), "question": q, "context": cB, "ev": evB, "evidence_block": evblock(evB)},
                         "C_actual_freshcache": {"key": ("gen", h(cC), h(q)), "question": q, "context": cC, "ev": evC, "evidence_block": evblock(evC)}}})
assert gates2["A_ctx_matches_audit_fresh_key"] == len(t2) == gates2["C_ctx_matches_audit_fc_key"], gates2
# ---------- generation tasks (reuse preserved answers only on exact key match) ----------
gen = {}
for x in t1 + t2:
    for name, v in x["paths"].items():
        k = tuple(v["key"]); gen.setdefault(k, {"key": list(k), "question": v["question"], "context": v["context"], "preserved": preserved.get(k)})
json.dump({"anchor_gate": gate, "l1_reuse_gates": dict(gates1), "l2_reuse_gates": dict(gates2), "l1_reuse_n": len(t1), "l2_reuse_n": len(t2),
           "generation_keys": len(gen), "preserved_reused": sum(1 for g in gen.values() if g["preserved"]), "to_generate": sum(1 for g in gen.values() if not g["preserved"]),
           "l1_reuse_stored_context_empty": sum(x["stored_context_empty"] for x in t1), "l1_reuse_fresh_context_empty": sum(x["fresh_context_empty"] for x in t1),
           "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in SRC}}, open(IN/"build_manifest.json", "w"), indent=1)
for name, rs in (("l1_reuse_hits.jsonl", t1), ("l2_reuse_hits.jsonl", t2), ("generation_tasks.jsonl", list(gen.values()))):
    with open(IN/name, "w", encoding="utf-8") as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(open(IN/"build_manifest.json").read()[:1200])
