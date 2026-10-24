"""Held-out replay of the four-way composition ablation A-D. Modes: replay | items | score.
All four policies share the frozen CategoryAware-style front end (ca_d+0.000_m1.00) and replay the held-out stream from empty caches; each
maintains its own answer / URL-list / page caches (composition_engine = hybrid composition engine with l2_on / l3_on switches). A = CA only;
B = CA + L3; C = CA + L2 (theta 0.80); D = CA + L2/L3 (theta 0.80, the frozen hybrid H80 policy). Answer-cache entries store the evidence
each policy actually used; hits serve answers generated from that evidence (no answer is copied across policies). Gates: A reproduces the
CategoryAware row and D reproduces H80 (metrics and every audit answer key). The analysis was configured before running."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, pickle, sys, time, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import composition_helpers as X, composition_engine as RE
MODE = sys.argv[1]; O = X.O; W = O / "work"
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
POL = {"A": dict(l2l3=False), "B": dict(l2_on=False), "C": dict(l3_on=False), "D": dict()}
DESC = {"A": "CategoryAware only", "B": "CategoryAware + L3 (exact-URL page cache)", "C": "CategoryAware + L2 (URL-list reuse, pages fetched)", "D": "CategoryAware + L2/L3 (composition H80)"}
def suff_pool():   # categoryaware_composition.suff_pool logic, with earlier outputs and this workspace as pools (read-only)
    jl = X.jl; SL = {}
    def addsuff(tasks, l8, q7):
        for t in tasks:
            a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
            if a is not None and b is not None: SL.setdefault((t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()), (a, b))
    lab = lambda f: {(r["query_id"], r["arm"]): r["label"] for r in jl(f)}
    addsuff(jl(X.OV / "audit_suff_tasks.jsonl"), lab(X.OV / "audit_suff_llama8b.jsonl"), lab(X.OV / "audit_suff_qwen7b.jsonl"))
    RU = json.load(open(X.CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
    cl8.update(lab(X.CRR / "gpu/out/suffv_llama8b.jsonl")); cq7.update(lab(X.CRR / "gpu/out/suffv_qwen7b.jsonl")); addsuff(jl(X.CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7)
    for D in (X.L2_POINTS_DIR, X.CALIBRATION_AUDIT_DIR, X.CA_FRONTIER_DIR, X.CA_COMPOSITION_DIR, O):
        addsuff(jl(D / "gpu/in/suffv_tasks.jsonl"), lab(D / "gpu/out/suffv_llama8b.jsonl"), lab(D / "gpu/out/suffv_qwen7b.jsonl"))
    return SL
if MODE == "replay":
    E = X.engine(); H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0); OUT = {}
    for p, kw in POL.items():
        t0 = time.perf_counter(); m, rows = RE.replay(E, H, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **kw); el = time.perf_counter() - t0
        sig = {q: (r["tier"], tuple(tuple(x) for x in (r.get("ev") or [])), r.get("gen_query")) for q, r in rows.items()}
        ops = {"served_urls_looked_up_L3": sum(len(r.get("served") or []) for r in rows.values() if r["tier"] in ("L2", "miss", "miss_rejected")) if kw.get("l3_on", True) and kw.get("l2l3", True) else 0,
               "L2_lookups": sum(1 for r in rows.values() if r["tier"] in ("L2", "miss", "miss_rejected")) if kw.get("l2_on", True) and kw.get("l2l3", True) else 0,
               "answer_cache_lookups": sum(1 for r in rows.values() if r["tier"] != "REAL_TIME"), "answer_cache_writes": sum(1 for r in rows.values() if r["tier"] not in ("L1", "REAL_TIME")),
               "L2_writes": sum(1 for r in rows.values() if r["tier"] in ("L2", "miss", "miss_rejected")) if kw.get("l2_on", True) and kw.get("l2l3", True) else 0,
               "L3_page_writes": m["fetches"] - sum(len(r.get("own") or []) for r in rows.values() if r["tier"] == "REAL_TIME") if kw.get("l3_on", True) and kw.get("l2l3", True) else 0,
               "tier_counts": dict(collections.Counter(r["tier"] for r in rows.values())), "decision_replay_seconds_in_process": round(el, 2)}
        aud = {}
        for q in IDS:
            r = rows[q]; k, ctx, gq = X.key_for(r, SQ[q]["query"]); ev = [tuple(x) for x in (r.get("ev") or [])]; blk = X.evblock(ev) if ev else None
            aud[q] = {"tier": r["tier"], "key": k, "gen_question": gq, "context": ctx, "evidence_block": blk, "evidence_sha": hashlib.sha256(blk.encode()).hexdigest() if blk else None}
        OUT[p] = {"metrics": m, "ops": ops, "sig": sig, "audit": aud}; X.log("replay", p, DESC[p], json.dumps(m), json.dumps({k: v for k, v in ops.items() if k != "tier_counts"}))
    COMPOSITION_H80 = pickle.load(open(X.CA_COMPOSITION_DIR / "work/heldout.pkl", "rb"))["H80"]; import csv; CA_ROW = next(r for r in csv.DictReader(open(X.BASELINE_SEL_DIR / "baseline_selection_points.csv")) if r["arm"] == "CategoryAware_style")
    G = {"D reproduces composition H80 metrics": {k: OUT["D"]["metrics"][k] for k in ("search_calls", "fetches", "generations", "l1_hits", "l2_hits")} == {k: COMPOSITION_H80["metrics"][k] for k in ("search_calls", "fetches", "generations", "l1_hits", "l2_hits")},
         "D reproduces composition H80 audit answer keys (400)": all(OUT["D"]["audit"][q]["key"] == COMPOSITION_H80["audit"][q]["key"] for q in IDS),
         "A reproduces the baseline-selection CategoryAware held-out row (saved, fetch/1k)": round(OUT["A"]["metrics"]["search_saved_pct"], 4) == round(float(CA_ROW["search_saved_pct"]), 4) and round(1000 * OUT["A"]["metrics"]["fetches"] / len(H), 1) == float(CA_ROW["fetch_per_1k"]),
         "answer-cache hit counts identical across policies (front end independent of L2/L3)": len({OUT[p]["metrics"]["l1_hits"] for p in POL}) == 1}
    for k, v in G.items(): X.log("GATE", k, v)
    if not all(G.values()): X.log("STOP"); sys.exit(1)
    pickle.dump({"policies": OUT, "gates": G, "n": len(H)}, open(W / "composition_ablation_replay.pkl", "wb")); sys.exit(0)
ST = pickle.load(open(W / "composition_ablation_replay.pkl", "rb")); P = ST["policies"]; hl = X.Held()
if MODE == "items":
    gen = {}
    for p in P:
        for q, a in P[p]["audit"].items():
            if hl.answer(a["key"]) is None: gen.setdefault(tuple(a["key"]), {"key": a["key"], "question": a["gen_question"], "context": a["context"]})
    X.wjl(O / "gpu/in/gen_tasks.jsonl", list(gen.values())); X.log("generation tasks:", len(gen))
    if not gen:
        items, miss = X.judge_items(hl, [(SQ[q]["query"], SQ[q]["gold"], a["key"]) for p in P for q, a in P[p]["audit"].items()]); assert not miss
        X.wjl(O / "gpu/in/judge2_items.jsonl", items); SL = suff_pool()
        need = {}
        for p in P:
            for q, a in P[p]["audit"].items():
                if a["evidence_block"] and (q, a["evidence_sha"]) not in SL: need.setdefault((q, a["evidence_sha"]), {"query_id": q, "arm": a["evidence_sha"][:16], "query": SQ[q]["query"], "evidence": a["evidence_block"]})
        X.wjl(O / "gpu/in/suffv_tasks.jsonl", list(need.values())); X.log("judge2 items:", len(items), "suffv tasks:", len(need))
    sys.exit(0)
