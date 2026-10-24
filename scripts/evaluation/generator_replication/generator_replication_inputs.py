"""Build generation tasks for the 400 held-out general-audit requests (outputs/general_audit/sample.json) with the
replay engine in freshcache/replay_engine, imported read-only through engine_loader. Arms: fresh, FreshCache, and the frozen
SemanticTTL(theta=0.60, kappa=1/2).

Gates: the FreshCache held-out replay and SemanticTTL(0.60, 1/2) reproduce the recorded run-time references
(scripts/analysis/reference_values.py). Keys follow the audit convention (role, sha16(context), sha16(generation question)) and are checked
against the held-out keys in outputs/replay/results/audit_keys_reported.json.

Writes outputs/generator_replication/inputs/requests.json and inputs/gen_tasks.jsonl."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import sys, json, copy, importlib.util, pathlib, hashlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "generator_replication"))
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C
E = C.load("reported"); exp, p2 = E["exp"], E["p2"]; h, ctx_for = p2.h, p2.ctx_for
assert "/freshcache/replay_engine/" in p2.__file__ and "/freshcache/replay_engine/" in E["me"].__file__
spec = importlib.util.spec_from_file_location("quality_selection_sample", ROOT / "scripts/evaluation/answer_audits/quality_selection_sample.py"); Q2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q2)
assert Q2.P2 is p2
H = C.streams(E)["heldout"]; S = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); m, fc = p2.replay(H, {}, "FreshCache"); assert abs(m["search_saved_pct"] - REF.value("reported_engine/heldout/FreshCache/search_saved_pct")) < 1e-4 and all(m[k] == REF.value("reported_engine/heldout/FreshCache/" + k) for k in ("l1_hits", "l2_hits")), m
ORIG = copy.deepcopy(exp.FIXED_TTL); L1T = exp.L1_SIM_THRESHOLD
try: ms, st = Q2.sttl_replay(H, 0.60, 0.5, ORIG)
finally: exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG); exp.L1_SIM_THRESHOLD = L1T
assert round(ms["search_saved_pct"], 3) == round(REF.value("reported_engine/heldout/SemanticTTL_0.60_k0.5/search_saved_pct"), 3), ms
tasks, recs = {}, []
for d in S:
    q, t = d["query_id"], d["t"]; r = {"query_id": q, "cluster_id": d["cluster_id"], "query": d["query"], "gold": d["gold"]}
    fctx = ctx_for(tuple((u, t) for u in fc[q]["own"])); k = ["fresh", h(fctx), h(d["query"])]; tasks[tuple(k)] = {"key": k, "question": d["query"], "context": fctx}
    r["fresh_key"], r["fresh_tier"] = k, "fresh"
    p = fc[q]
    if p["tier"] == "L1": c = ctx_for(tuple(p["stored_answer_ev"])); k = ["stored", h(c), h(p["stored_answer_query"])]; gq = p["stored_answer_query"]
    else: c = ctx_for(tuple(p["ev"])) if p.get("ev") else ""; k = ["gen", h(c), h(d["query"])]; gq = d["query"]
    tasks[tuple(k)] = {"key": k, "question": gq, "context": c}; r["freshcache_key"], r["freshcache_tier"] = k, p["tier"]
    p = st[q]; c = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", d["query"]); k = ["gen", h(c), h(gq)]
    tasks[tuple(k)] = {"key": k, "question": gq, "context": c}; r["sttl_q060_k1_2_key"], r["sttl_q060_k1_2_tier"] = k, p["tier"]
    recs.append(r)
ref = json.load(open((OUTPUT_ROOT / "replay/results/audit_keys_reported.json")))["keys"]["heldout"]
mism = [(r["query_id"], a) for r in recs for a, b in (("fresh", "fresh"), ("freshcache", "freshcache"), ("sttl_q060_k1_2", "sttl_q060_k1_2")) if r[f"{a}_key"][1:] != ref[r["query_id"]][b][0][1:]]
assert not mism, mism[:5]
json.dump(recs, open(W/"inputs/requests.json", "w"), indent=0)
with open(W/"inputs/gen_tasks.jsonl", "w", encoding="utf-8") as f:
    for tk in tasks.values(): f.write(json.dumps(tk, ensure_ascii=False) + "\n")
print("requests", len(recs), "distinct generation tasks", len(tasks), "keys equal reported_results_recompute: True",
      "sttl tiers", {t: sum(1 for r in recs if r["sttl_q060_k1_2_tier"] == t) for t in {r["sttl_q060_k1_2_tier"] for r in recs}},
      "fc tiers", {t: sum(1 for r in recs if r["freshcache_tier"] == t) for t in {r["freshcache_tier"] for r in recs}})
