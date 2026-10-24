"""Stricter-L1 diagnostic under one engine (CPU stage), following the frozen L1-precision protocol (outputs/l1_precision/l1_precision_prereg.json, written by l1_precision_protocol.py):
validation grid (96 pre-registered cells + recorded), held-out census arms (recorded, 0.90 only, frozen 0.90+guards).
Writes realised L1 hit pairs per cell/arm, and answer keys for the validation and held-out audits, for the GPU stage."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, itertools, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); exp, p2 = E["exp"], E["p2"]; h, ctx_for = p2.h, p2.ctx_for
sys.path.insert(0, str(REPO_ROOT / "freshcache/method")); import l1_admission_gate
PR = json.load(open(C.OUTPUT_ROOT / "l1_precision/l1_precision_prereg.json")); g = PR["grid"]
FZ = json.load(open((OUTPUT_ROOT / "l1_precision/frozen_l1_precision_gate.json")))
ST = C.streams(E); ORIG = exp.semantic_equivalent; cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in E["recs"]}; qcl = {r["query"]: cid[r["query_id"]] for r in E["recs"]}
VA = json.load(open((OUTPUT_ROOT / "l1_precision/validation_audit_sample.json")))["requests"]
HA = json.load(open((OUTPUT_ROOT / "l1_precision/heldout_audit_sample.json")))["requests"]
def arm(stream, sf, jf, at, gd, audit):
    exp.semantic_equivalent = l1_admission_gate.make_gate(exp, sf, jf, at, gd) if (sf, jf, at, gd) != (None,)*4 else ORIG
    try: m, rows = p2.replay(stream, {}, "FreshCache")
    finally: exp.semantic_equivalent = ORIG
    hits = [{"query_id": r["query_id"], "pair_id": f"{h(r['query'])}_{h(rows[r['query_id']]['matched_query'])}", "incoming": r["query"], "cached": rows[r["query_id"]]["matched_query"],
             "cross_cluster": int(qcl.get(rows[r["query_id"]]["matched_query"]) != cid[r["query_id"]])} for _, r in stream if rows[r["query_id"]]["tier"] == "L1"]
    keys = {}
    for d in audit:
        p = rows[d["query_id"]]
        if p["tier"] == "L1": c = ctx_for(tuple(p["stored_answer_ev"])); keys[d["query_id"]] = {"tier": "L1", "key": ["stored", h(c), h(p["stored_answer_query"])], "question": p["stored_answer_query"], "context": c}
        else: c = ctx_for(tuple(p["ev"])); keys[d["query_id"]] = {"tier": p["tier"], "key": ["gen", h(c), h(d["query"])], "question": d["query"], "context": c}
    gens = m["search_calls"] + m["l2_hits"]
    return {"search_saved_pct": round(m["search_saved_pct"], 4), "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "generations": gens, "hits": hits, "audit": keys}
out = {"engine": ENG, "validation": {}, "heldout": {}}
out["validation"]["recorded"] = arm(ST["validation"], None, None, None, None, VA)
for sf, jf, at, gd in itertools.product(g["sim_floor"], g["jaccard_floor"], ["as_implemented", "required"], [False, True]):
    tag = f"s{sf}_j{jf}_{at}_{'G' if gd else 'g'}"; out["validation"][tag] = arm(ST["validation"], sf, jf, at, gd, VA)
print("validation recorded", {k: v for k, v in out["validation"]["recorded"].items() if k not in ("hits", "audit")}, flush=True)
for tag, spec in (("recorded", (None,)*4), ("strict090", (0.90, 0.30, "as_implemented", False)), ("precision_frozen", (FZ["sim_floor"], FZ["jaccard_floor"], FZ["answer_type"], bool(FZ["guards"])))):
    out["heldout"][tag] = arm(ST["heldout"], *spec, HA); print("heldout", tag, {k: v for k, v in out["heldout"][tag].items() if k not in ("hits", "audit")}, flush=True)
json.dump(out, open(C.W/"results"/f"stricter_cpu_{ENG}.json", "w"))
