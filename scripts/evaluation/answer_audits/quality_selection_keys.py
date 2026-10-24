"""Selection check for the quality-selected SemanticTTL (Table 16), validation only: rebuild the answer keys of FreshCache and the nine
preregistered SemanticTTL candidates on the RECORDED validation sample (q_sample.jsonl; membership and the screened set are
kept fixed) with one engine, using quality_selection_sample's own replays and ctx_for. Keyset must reproduce every recorded key."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, copy, importlib.util, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); exp, p2 = E["exp"], E["p2"]; h, ctx_for = p2.h, p2.ctx_for
QB = (OUTPUT_ROOT / "quality_selection")
spec = importlib.util.spec_from_file_location("quality_selection_sample", (REPO_ROOT / "scripts/evaluation/answer_audits/quality_selection_sample.py")); Q2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q2)
assert Q2.P2 is p2, (Q2.P2.__file__, p2.__file__)
pre = json.load(open((OUTPUT_ROOT / "quality_selection/quality_selection_prereg.json"), encoding="utf-8")); S = [json.loads(l) for l in open(QB/"q_sample.jsonl")]
V = C.streams(E)["validation"]; byq = {r["query_id"]: r for _, r in V}
ORIG = copy.deepcopy(exp.FIXED_TTL); L1T = exp.L1_SIM_THRESHOLD
_, fc = p2.replay(V, E["rounds"], "FreshCache"); arms = {"fc": fc}
for c in pre["candidates"]:
    try: _, rw = Q2.sttl_replay(V, c["theta"], c["kappa"], ORIG)
    finally: exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG); exp.L1_SIM_THRESHOLD = L1T
    arms[c["arm"]] = rw
out, tasks, changed = [], {}, {a: [] for a in ["fresh"] + list(arms)}
for d in S:
    q = d["query_id"]; rec = byq[q]; t = fc[q]["t"]
    fctx = ctx_for(tuple((u["url_hash"], t) for u in rec["urls"])); fk = ["fresh", h(fctx), h(d["query"])]; tasks[tuple(fk)] = {"key": fk, "question": d["query"], "context": fctx}
    r = {"query_id": q, "fresh_key": fk}
    if fk[1] != d["fresh_ctx_sha"]: changed["fresh"].append(q)
    for a, rw in arms.items():
        p = rw[q]; ctx = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", d["query"]); k = ["gen", h(ctx), h(gq)]
        tasks[tuple(k)] = {"key": k, "question": gq, "context": ctx}; r[f"{a}_key"] = k; r[f"{a}_tier"] = p["tier"]
        if k != d[f"{a}_key"]: changed[a].append(q)
    out.append(r)
json.dump({"engine": ENG, "records": out, "changed_vs_recorded": changed}, open(C.W/"results"/f"quality_selection_keys_{ENG}.json", "w"), indent=0)
json.dump(list(tasks.values()), open(C.W/"results"/f"quality_selection_gen_tasks_{ENG}.json", "w"))
print(ENG, {a: len(v) for a, v in changed.items()}, flush=True)
