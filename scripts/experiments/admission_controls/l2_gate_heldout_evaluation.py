"""Learned L2 compatibility gate, held-out stage (frozen gate). Modes: replay | items | score | live.
replay: held-out replay with the frozen gate (top admitted L2 candidate scored; p < tau -> fresh search); operations + common cost model;
        400-audit answer keys and evidence blocks. Gate check: the gate features reproduce the stored validation features.
items:  2-judge items and SUFFV tasks for anything not in the frozen pools.
score:  accuracy / WAI / C-WAI / sufficiency on the 400 audit, paired vs FreshCache 0.75, 0.80 and L2Verify 0.55 (frozen labels).
live:   unseen-query live cases: gate score for every case's cached list (FreshCache-held page versions = the cached-path texts; BGE-M3
        embeddings computed here); report fresh-vs-cached sufficiency among gate-accepted vs gate-rejected cases."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, math, os, pathlib, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
sys.dont_write_bytecode = True
O = ((OUTPUT_ROOT / "learned_l2_gate")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/admission_controls")); MODE = sys.argv[1]
import l2_gate_features as Lb   # noqa: E402
import numpy as np   # noqa: E402
GZ = pickle.load(open(O / "gate_frozen.pkl", "rb")); GJ = json.load(open(O / "gate_frozen.json")); FEATS, TAU, MODEL = GZ["features"], GZ["tau"], GZ["model"]
assert GJ["heldout_consulted"] is False
TAG = os.environ.get("GATE_POINT", "")   # "" = Youden point (prereg.json); "_tau2" = savings-matched point (prereg2.json + tau2_frozen.json)
if TAG == "_tau2":
    T2J = json.load(open(O / "tau2_frozen.json")); assert T2J["heldout_consulted"] is False and T2J["prereg2_sha256"] == json.load(open(O / "prereg2.json"))["sha256"]; TAU = T2J["tau2"]
ROOT, CRR = Lb.ROOT, Lb.CRR; W = O / "work"
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
def pscore(f): return float(MODEL.predict_proba(np.array([[f[k] for k in FEATS]], float))[:, 1][0])
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
if MODE == "replay":
    VC = jl(W / "val_candidates.jsonl")[:400]
    G = {"l2_gate_features features reproduce the stored validation features (first 400 candidates)": all(abs(Lb.features(r["query"], [tuple(x) for x in r["ev"]], r["entry_query"], r["qq_sim"], r["fc"], 0, r["entry_fc"])[k] - r[k]) < 1e-9
             for r in VC for k in FEATS if k != "entry_age_frac_of_window")}
    class Gate:
        def __init__(self): self.n = 0; self.acc = 0; self.scores = []
        def score(self, q, tent_ev, eq, sim, fc, age, efc, qid, urls):
            p = pscore(Lb.features(q, [tuple(x) for x in tent_ev], eq, sim, fc, age, efc)); self.n += 1; self.acc += p >= TAU; self.scores.append((qid, p)); return p
    H = Lb.C.streams(Lb.E)["heldout"]; n = len(H); gate = Gate()
    m, rows, _ = Lb.S2.replay(H, gamma=TAU, support=gate)
    VR = pickle.load(open(CRR / "verifier/out/test_rows.pkl", "rb"))["FreshCache"]
    m0, rows0, _ = Lb.S2.replay(H, gamma=None, support=None); G["engine copy with gate disabled reproduces frozen FreshCache rows"] = all(rows0[q]["tier"] == VR[q]["tier"] and rows0[q].get("ev") == VR[q].get("ev") for q in VR)
    CO = json.load(open((OUTPUT_ROOT / "concurrency_cost/component_costs.json")))["coefficients"]; nonrt = sum(1 for _, r in H if r["freshness_class"] != "REAL_TIME")
    LK = CO["C_local_lookup"]["per_method_ms_per_request_cold"]["FreshCache"]
    cost = {f"{sc}_ms_per_request": round((m["search_calls"] * CO["c_search_ms"][sc] + m["fetches"] * CO["c_fetch_ms"][sc] + m["generations"] * CO["c_gen_ms"]["value"] + nonrt * CO["c_embed_ms"]["value"]) / n + LK, 3) for sc in ("local", "live_p50")}
    OPS = {"search_saved_pct": m["search_saved_pct"], "search_calls": m["search_calls"], "fetches": m["fetches"], "generations": m["generations"], "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1),
           "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l2_scored": gate.n, "l2_accepted": int(gate.acc), "l2_rejected": m["l2_rejected"], **cost, "gate_lookup_note": "features use precomputed embeddings + page texts; FreshCache cold lookup (component_costs.json) charged; scorer time not added"}
    AUD = {}; h = Lb.p2.h; ctx_for = Lb.p2.ctx_for
    for q in IDS:
        b = rows[q]
        if b["tier"] == "L1": ev = [tuple(x) for x in b["stored_answer_ev"]]; ctx = ctx_for(tuple(ev)); key = ["stored", h(ctx), h(b["stored_answer_query"])]; gq = b["stored_answer_query"]
        else: ev = [tuple(x) for x in (b.get("ev") or [])]; ctx = ctx_for(tuple(ev)) if ev else ""; key = ["gen", h(ctx), h(SQ[q]["query"])]; gq = SQ[q]["query"]
        eb = Lb.evblock(ev); AUD[q] = {"tier": b["tier"], "key": key, "gen_question": gq, "context": ctx, "evidence_sha": hashlib.sha256(eb.encode()).hexdigest(), "evidence": eb}
    for k, v in G.items(): print("GATE", k, v, flush=True)
    assert all(G.values())
    pickle.dump({"ops": OPS, "audit": AUD, "gates": G, "scores": gate.scores}, open(W / f"stage3{TAG}.pkl", "wb")); print(json.dumps(OPS), flush=True); sys.exit(0)
ST = pickle.load(open(W / f"stage3{TAG}.pkl", "rb")) if MODE in ("items", "score") else None
if MODE in ("items", "score"):
    T1 = (OUTPUT_ROOT / "standalone_operating_points"); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/operating_points")); import operating_point_helpers as X   # noqa: E402  (Held labeler + pools)
    hl = X.Held(); SL = {}
    def addsuff(tasks, l8, q7):
        for t in tasks:
            a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
            if a is not None and b is not None: SL.setdefault((t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()), (a, b))
    L8 = lambda f: {(r["query_id"], r["arm"]): r["label"] for r in jl(f)}
    addsuff(jl(X.OV / "audit_suff_tasks.jsonl"), L8(X.OV / "audit_suff_llama8b.jsonl"), L8(X.OV / "audit_suff_qwen7b.jsonl"))
    RU = json.load(open(CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
    cl8.update(L8(CRR / "gpu/out/suffv_llama8b.jsonl")); cq7.update(L8(CRR / "gpu/out/suffv_qwen7b.jsonl")); addsuff(jl(CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7)
    for D, ins in ((X.L2_POINTS_DIR, "gpu/in/suffv_tasks.jsonl"), (T1, "gpu/in/suffv_tasks.jsonl"), (O, "gpu/in/ho_suffv_tasks.jsonl")):
        sh = lambda j: {k: v for f in [D / "gpu/out" / f"suffv_{j}.jsonl"] + sorted((D / "gpu/out").glob(f"suffv_{j}_shard*.jsonl")) for k, v in L8(f).items()}
        addsuff(jl(D / ins), sh("llama8b"), sh("qwen7b"))
if MODE == "items":
    items, miss = X.judge_items(hl, [(SQ[q]["query"], SQ[q]["gold"], a["key"]) for q, a in ST["audit"].items()])
    gen = {tuple(a["key"]): {"key": a["key"], "question": a["gen_question"], "context": a["context"]} for a in ST["audit"].values() if hl.answer(a["key"]) is None}
    X.wjl(O / f"gpu/in/gen_tasks{TAG}.jsonl", list(gen.values())); X.wjl(O / f"gpu/in/judge2_items{TAG}.jsonl", items)
    need = [{"query_id": q, "arm": a["evidence_sha"][:16], "query": SQ[q]["query"], "evidence": a["evidence"]} for q, a in ST["audit"].items() if (q, a["evidence_sha"]) not in SL]
    X.wjl(O / f"gpu/in/ho_suffv_tasks{TAG}.jsonl", need); print(json.dumps({"tau": TAU, "gen": len(gen), "judge2_items": len(items), "answers_missing_for_items": len(miss), "suffv": len(need)})); sys.exit(0)
if MODE == "score":
    import csv
    L2_REQUESTS = collections.defaultdict(dict)
    for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")):
        if r["point"] in ("theta_0.75", "theta_0.80", "gamma_0.55"): L2_REQUESTS[r["point"]][r["request_id"]] = r
    FRESH = {q: L2_REQUESTS["theta_0.75"][q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values())
    def suff(q, sha):
        x = SL.get((q, sha)); return "UNRESOLVED" if x is None else (x[0] if x[0] == x[1] else "JUDGE_DISAGREEMENT")
    LAB = {pt: {q: (L2_REQUESTS[pt][q]["answer_label"], L2_REQUESTS[pt][q]["sufficiency"]) for q in IDS} for pt in L2_REQUESTS}
    LAB["gate"] = {q: (hl.labels(SQ[q]["query"], SQ[q]["gold"], a["key"])[0] or "UNRESOLVED", suff(q, a["evidence_sha"])) for q, a in ST["audit"].items()}
    def mcn(b, c):
        n = b + c; return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
    OUT = {"tau": TAU, "point": TAG or "youden", "gate_ops": ST["ops"], "audit": {}}
    for pt, L in LAB.items():
        c = collections.Counter(a for a, _ in L.values()); s = collections.Counter(x for _, x in L.values()); w = sum(1 for q in IDS if FRESH[q] == "CORRECT" and L[q][0] == "WRONG")
        OUT["audit"][pt] = {"correct": c["CORRECT"], "accuracy_pct": round(100 * c["CORRECT"] / len(IDS), 2), "abstain": c["ABSTAIN"], "unresolved": c["UNRESOLVED"], "WAI": w, "C_WAI_pct": round(100 * w / FCN, 2), "sufficient": s["SUFFICIENT"], "suff_unresolved": s["UNRESOLVED"]}
    G_ = LAB["gate"]; OUT["paired_gate_vs"] = {}
    for pt in ("theta_0.75", "theta_0.80", "gamma_0.55"):
        L = LAB[pt]; a = sum(1 for q in IDS if G_[q][1] == "SUFFICIENT" and L[q][1] != "SUFFICIENT"); b = sum(1 for q in IDS if L[q][1] == "SUFFICIENT" and G_[q][1] != "SUFFICIENT")
        wa = sum(1 for q in IDS if FRESH[q] == "CORRECT" and G_[q][0] == "WRONG" and L[q][0] != "WRONG"); wb = sum(1 for q in IDS if FRESH[q] == "CORRECT" and L[q][0] == "WRONG" and G_[q][0] != "WRONG")
        OUT["paired_gate_vs"][pt] = {"suff_gate_only": a, "suff_other_only": b, "suff_p": mcn(a, b), "WAI_gate_only": wa, "WAI_other_only": wb, "WAI_p": mcn(wa, wb)}
    json.dump(OUT, open(O / f"gate_heldout_results{TAG}.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); sys.exit(0)
if MODE == "live":   # live unseen-query cases
    T2 = (OUTPUT_ROOT / "unseen_queries"); import re   # noqa: E402
    import importlib.util, io, contextlib   # noqa: E402
    sys.argv = ["unseen_query_scoring.py", "score"]; spec = importlib.util.spec_from_file_location("t2s", (REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py")); s = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(s)
    FZ = {c["case_id"]: c for c in json.load(open(T2 / "sample_frozen.json"))["cases"]}; MV = {json.loads(l)["query_id"]: json.loads(l) for l in open(T2 / "cache_state_l2_matching.jsonl")}
    UZ = np.load(T2 / "emb_untouched.npz", allow_pickle=True); UQ = {k: i for i, k in enumerate(UZ["keys"])}
    from sentence_transformers import SentenceTransformer   # noqa: E402
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
    uid = lambda u: hashlib.sha1(u.encode()).hexdigest()
    def live_text(u):
        for d in (T2 / "raw/pages_pass2", T2 / "raw/pages"):
            p = d / f"{uid(u)}.json"
            if p.exists():
                x = json.load(open(p, encoding="utf-8"))
                if x.get("ok"): return re.sub(r"\s+", " ", x.get("extracted_text") or "").strip()[:2000]
        return ""
    OUTR = []
    for c in s.FINAL:
        z = FZ[c["case_id"]]; l3t = MV[z["query_id"]].get("l3_cache_time", {}); texts = []
        for uh, u in zip(z["cached_url_hashes"], z["cached_urls"]):
            texts.append(Lb.page_text(uh, l3t[uh]) if uh in l3t else live_text(u))   # exactly the texts FreshCache would hold (path C)
        qv = UZ["emb"][UQ[z["incoming_query"]]]; pe = enc.encode([t for t in texts if t], normalize_embeddings=True) if any(texts) else []
        sims = [float(qv @ e) for e in pe]
        f = Lb.features_core(z["incoming_query"], sims, texts, len(texts), MV[z["query_id"]]["entry_creator_query"], z["similarity"], z["freshness_class"], MV[z["query_id"]]["entry_age_s"], z["entry_class"])
        p = pscore(f); i = c["case_id"]
        OUTR.append({"case_id": i, "gate_p": p, "accepted": p >= TAU, "A_suff": s.L["A_matched"][i][1], "B_suff": s.L["B_cached_live"][i][1], "C_suff": s.L["C_freshcache"][i][1], "A_correct": s.L["A_matched"][i][0], "C_correct": s.L["C_freshcache"][i][0]})
    def grp(rows):
        n = len(rows); a = sum(r["A_suff"] == "SUFFICIENT" for r in rows); cc = sum(r["C_suff"] == "SUFFICIENT" for r in rows)
        return {"N": n, "fresh_sufficient": a, "cached_C_sufficient": cc, "fresh_correct": sum(r["A_correct"] == "CORRECT" for r in rows), "cached_C_correct": sum(r["C_correct"] == "CORRECT" for r in rows)}
    RES = {"tau": TAU, "accepted": grp([r for r in OUTR if r["accepted"]]), "rejected": grp([r for r in OUTR if not r["accepted"]]), "all": grp(OUTR)}
    json.dump({"summary": RES, "cases": OUTR}, open(O / f"gate_live_cases{TAG}.json", "w"), indent=1); print(json.dumps(RES, indent=1))
