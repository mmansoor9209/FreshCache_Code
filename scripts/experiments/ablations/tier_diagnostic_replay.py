"""CPU stage of the tier-specific diagnostics under the replay engine, same construction as scripts/evaluation/tier_quality/tier_quality_inputs.py.
L1 diagnostic: every realised held-out L1 hit -> A stored answer (stored query on stored evidence), B incoming query on stored evidence, C fresh.
L2 diagnostic: FreshCache L2 hits among the 400 held-out audit requests -> A fresh retrieval, B cached list with current pages, C actual FreshCache."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, re, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
E = C.load("reported"); exp, p2 = E["exp"], E["p2"]; h, ctx_for = p2.h, p2.ctx_for
import answer_grading as e2e, page_version_replay as ma
H = C.streams(E)["heldout"]; T = {r["query_id"]: (t, r) for t, r in H}
m, rows = p2.replay(H, {}, "FreshCache"); print("reported held-out", round(m["search_saved_pct"], 4), m["l1_hits"], m["l2_hits"], flush=True)
gold = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid, qt):
    g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
t1 = []
for qid, p in rows.items():
    if p["tier"] != "L1": continue
    t, r = T[qid]; q = r["query"]; sq = p["stored_answer_query"]; own = [u["url_hash"] for u in r["urls"]]
    sctx = ctx_for(tuple(p["stored_answer_ev"])); fctx = ctx_for(tuple((u, t) for u in own))
    t1.append({"query_id": qid, "t": t, "incoming_query": q, "stored_answer_query": sq, "matched_query": p["matched_query"], "pair_id": f"{h(q)}_{h(p['matched_query'])}",
               "reference_answer": goldof(qid, q), "stored_context_empty": not sctx, "fresh_context_empty": not fctx, "age_s": p.get("age"),
               "paths": {"A_stored_answer": {"key": ["stored", h(sctx), h(sq)], "question": sq, "context": sctx}, "B_regen_stored_evidence": {"key": ["gen", h(sctx), h(q)], "question": q, "context": sctx},
                         "C_fresh": {"key": ["fresh", h(fctx), h(q)], "question": q, "context": fctx}}})
S = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
man = {}
for mm in exp.load_jsonl(exp.MANIFEST_FILE):
    if mm.get("run_id") == "run_00" and mm["url_hash"] not in man: man[mm["url_hash"]] = {"title": mm.get("title", "") or "", "snippet": mm.get("snippet", "") or ""}
def evblock(ev):   # verifier study evidence formatting, page version chosen by the snapshot rule
    lines = []
    for uh, tv in ev:
        mt = man.get(uh, {}); b = re.sub(r"\s+", " ", (e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv)) or "")).strip()[:2000]
        blk = f"{len(lines)+1}. {mt.get('title','').strip()}"
        if mt.get("snippet"): blk += f"\n   {mt['snippet'].strip()}"
        if len(b) >= 200: blk += f"\n   PAGE TEXT: {b}"
        lines.append(blk)
    return "\n".join(lines) or "(no stored evidence)"
t2 = []
for d in S:
    qid = d["query_id"]; p = rows[qid]
    if p["tier"] != "L2": continue
    t, r = T[qid]; q = r["query"]; own = [u["url_hash"] for u in r["urls"]]
    evA = [(u, t) for u in own]; evB = [(u, t) for u in p["served"]]; evC = [tuple(x) for x in p["ev"]]
    cA, cB, cC = ctx_for(tuple(evA)), ctx_for(tuple(evB)), ctx_for(tuple(evC))
    t2.append({"query_id": qid, "t": t, "question": q, "reference_answer": d["gold"], "paths": {
        "A_fresh_retrieval": {"key": ["fresh", h(cA), h(q)], "question": q, "context": cA, "evidence_block": evblock(evA)},
        "B_cached_list_current_pages": {"key": ["gen", h(cB), h(q)], "question": q, "context": cB, "evidence_block": evblock(evB)},
        "C_actual_freshcache": {"key": ["gen", h(cC), h(q)], "question": q, "context": cC, "evidence_block": evblock(evC)}}})
json.dump({"l1_reuse": t1, "l2_reuse": t2, "heldout_agg": {"search_saved_pct": m["search_saved_pct"], "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"]}}, open(C.W/"results/tier_cpu_reported.json", "w"), ensure_ascii=False)
print("l1_reuse hits", len(t1), "l2_reuse L2 audit hits", len(t2), "B==C", sum(x["paths"]["B_cached_list_current_pages"]["context"] == x["paths"]["C_actual_freshcache"]["context"] for x in t2), flush=True)
