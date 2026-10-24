"""Rebuild every answer key of the two audits with one engine (argv[1]), using the arm replays
(audit_replay.replay for FreshCache arms; the audits' own SemanticTTL transcriptions) and that engine's audit_replay.ctx_for. Keys are
(role, sha16(context), sha16(generation question)). Gate: the run with the keyset engine must reproduce every recorded key."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, copy, importlib.util, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); exp, p2, sc = E["exp"], E["p2"], E["sc"]; h, ctx_for = p2.h, p2.ctx_for
V = (OUTPUT_ROOT / "archived_runs")
def mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
AP = mod((REPO_ROOT / "scripts/evaluation/answer_audits/general_audit_sample.py"), "ho_prep"); Q2 = mod((REPO_ROOT / "scripts/evaluation/answer_audits/quality_selection_sample.py"), "quality_selection_sample"); BOP = mod((REPO_ROOT / "scripts/evaluation/answer_audits/semanticttl_audit_sample.py"), "bop_prep")
assert AP.p2 is p2 and Q2.P2 is p2 if hasattr(Q2, "P2") else True
def key_of(p, q):
    if p["tier"] in ("L1", "HIT"):   # SemanticTTL replays label L1 hits "HIT"
        sctx = ctx_for(tuple(p["stored_answer_ev"])); return ["stored", h(sctx), h(p["stored_answer_query"])], sctx, p["stored_answer_query"]
    c = ctx_for(tuple(p["ev"])) if p.get("ev") else ""; return ["gen", h(c), h(q)], c, q
ST = C.streams(E); OUT = {"heldout": {}, "primary": {}}; TASKS = {}
def add(k, q, c): TASKS[tuple(k[1:])] = {"key": k, "question": q, "context": c}
ORIG = copy.deepcopy(exp.FIXED_TTL); L1T = exp.L1_SIM_THRESHOLD
# ---------- held-out audit ----------
H = ST["heldout"]; S = json.load(open(((OUTPUT_ROOT / "general_audit/sample.json"))))["requests"]
_, fc = p2.replay(H, {}, "FreshCache")
exp.FIXED_TTL.clear(); exp.FIXED_TTL.update({c: v*0.0625 for c, v in ORIG.items()})
try: _, st = AP.replay_semanticttl(H, 0.40, dict(exp.FIXED_TTL))
finally: exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG); exp.L1_SIM_THRESHOLD = L1T
try: _, qs = Q2.sttl_replay(H, 0.60, 0.5, ORIG)
finally: exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG); exp.L1_SIM_THRESHOLD = L1T
preds = {json.loads(l)["query_id"]: json.loads(l)["pred_class"] for l in open((OUTPUT_ROOT / "class_predictor/predicted_classes.jsonl"))}
PH = [(t, dict(r, freshness_class=preds[r["query_id"]])) for t, r in sc.build_stream([dict(r) for r in E["recs"]], "zipf_uniform", 42) if (r.get("cluster_id") or r["query_id"]) in set(E["split"]["test_clusters"])]
_, pc = p2.replay(PH, {}, "FreshCache")
for d in S:
    q, t = d["query_id"], d["t"]; own = fc[q]["own"]; fctx = ctx_for(tuple((u, t) for u in own)); rec = {"fresh": (["fresh", h(fctx), h(d["query"])], fc[q]["tier"])}; add(rec["fresh"][0], d["query"], fctx)
    for arm, rows in (("freshcache", fc), ("sttl_k1_16", st), ("predicted_class", pc)):
        k, c, gq = key_of(rows[q], d["query"]); rec[arm] = (k, rows[q]["tier"]); add(k, gq, c)
    p = qs[q]; c = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", d["query"]); k = ["gen", h(c), h(gq)]; rec["sttl_q060_k1_2"] = (k, p["tier"]); add(k, gq, c)
    OUT["heldout"][q] = rec
# ---------- primary mixed-age audit ----------
P = ST["primary"]; S1 = [json.loads(l) for l in open(((OUTPUT_ROOT / "mixed_age_audit/mixed_age_sample.jsonl")))]
_, on = p2.replay(P, {}, "FreshCache"); _, off = p2.replay(P, {}, "FreshCache_AlwaysPassTemporal")
exp.FIXED_TTL.update({c: v*BOP.KAPPA for c, v in ORIG.items()})
try: _, sb = BOP.replay_semanticttl(P)
finally: exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
for d in S1:
    q, t = d["query_id"], d["t"]; fctx = ctx_for(tuple((u, t) for u in on[q]["own"])); rec = {"fresh": (["fresh", h(fctx), h(d["query"])], on[q]["tier"])}; add(rec["fresh"][0], d["query"], fctx)
    for arm, rows in (("freshcache", on), ("gate_off", off), ("sttl_k1_8", sb)):
        k, c, gq = key_of(rows[q], d["query"]); rec[arm] = (k, rows[q]["tier"]); add(k, gq, c)
    OUT["primary"][q] = rec
# ---------- recorded keys ----------
HS = {json.loads(l)["query_id"]: json.loads(l) for l in open(((OUTPUT_ROOT / "quality_selection/heldout_sensitivity/h_sample.jsonl")))}
PS = {r["query_id"]: r for r in json.load(open(((OUTPUT_ROOT / "predicted_class_audit/sample_spec.json"))))["requests"]}
SB = {json.loads(l)["query_id"]: json.loads(l) for l in open(((OUTPUT_ROOT / "semanticttl_audit/sttl_mixed_age_sample.jsonl")))}
pub = {"heldout": {d["query_id"]: {"fresh": d["fresh_key"], "freshcache": d["fc_key"], "sttl_k1_16": d["sttl_key"], "sttl_q060_k1_2": HS[d["query_id"]]["new_sttl_key"],
                                   "predicted_class": [None, PS[d["query_id"]]["B_predicted_ctx_sha"], h(PS[d["query_id"]]["B_predicted_gen_query"])]} for d in S},
       "primary": {d["query_id"]: {"fresh": ["fresh", d["fresh_ctx_sha"], h(d["query"])], "freshcache": d["on_key"], "gate_off": d["off_key"], "sttl_k1_8": SB[d["query_id"]]["sttl_key"]} for d in S1}}
cmp = {}
for aud in ("heldout", "primary"):
    for arm in pub[aud][next(iter(pub[aud]))]:
        same = [q for q in pub[aud] if (OUT[aud][q][arm][0][1:] == pub[aud][q][arm][1:])]
        cmp[f"{aud}/{arm}"] = {"n": len(pub[aud]), "key_equal_to_recorded": len(same), "changed_ids": sorted(set(pub[aud]) - set(same))}
json.dump({"engine": ENG, "keys": OUT, "compare_to_recorded": cmp}, open(C.W/"results"/f"audit_keys_{ENG}.json", "w"), indent=1)
json.dump(list(TASKS.values()), open(C.W/"results"/f"audit_gen_tasks_{ENG}.json", "w"))
for k, v in cmp.items(): print(ENG, k, v["key_equal_to_recorded"], "/", v["n"], v["changed_ids"][:12], flush=True)
