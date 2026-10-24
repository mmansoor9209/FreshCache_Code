"""Full-state live policies (no new API calls). The live candidates are served as one stream in arrival order (eval_t). Each policy keeps
its own L2 state: besides each candidate's held-out L2 entry (recorded similarity, eligible by construction), every served candidate
REGISTERS the list it used (rejected -> its own live search list at matched depth, discovery = its arrival time; accepted -> the reused list
with that entry's discovery time), and later candidates may reuse those lists under the frozen L2 rule (cosine >= floor between incoming
queries with frozen BGE-M3 embeddings; URL-list staleness p_stale(entry class, age) <= eps). The policy takes the highest-similarity eligible
entry and applies its frozen admission (primary 0.75; stricter 0.80; learned gate the frozen Youden tau on that entry's serving-time features).
Accepted reuse of another candidate's list serves that list's recorded pages (page reuse: identical texts); these new (query, evidence) pairs
are generated and judged with the frozen generator, two-judge rule and SUFFV (scripts/experiments/main/composition_generation_judging.py). Rejection -> own recorded live search (A).
Recorded searches are reused (2026-10-04); no new Serper calls. Modes: build | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import collections, hashlib, json, math, pickle, sys, pathlib
import numpy as np
O = ((OUTPUT_ROOT / "composition")); T2 = ((OUTPUT_ROOT / "unseen_queries"))
GATE = ((OUTPUT_ROOT / "learned_l2_gate")); MODE = sys.argv[1]; sys.dont_write_bytecode = True
src = open((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_evidence_paths.py")).read(); cut = src.index("CASES, FLOW, GEN, SUF = ")
g = {"__name__": "t2b", "__file__": str((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_evidence_paths.py"))}; exec(compile(src[:cut], "<unseen_query_evidence_paths definitions, read-only>", "exec"), g)
FZ, MV, page, ctx, evblock, h16, man, E = g["FZ"], g["MV"], g["page"], g["ctx"], g["evblock"], g["h16"], g["man"], g["E"]
exp = E["exp"]; ok = lambda fc, age: exp.p_stale(fc, age, "url_list") <= exp.EPS_URL_LIST
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/admission_controls")); import l2_gate_features as Lb   # noqa: E402
GZ = pickle.load(open(GATE / "gate_frozen.pkl", "rb")); FEATS, TAU, MODEL = GZ["features"], GZ["tau"], GZ["model"]
Z = np.load(T2 / "emb_untouched.npz", allow_pickle=True); QI = {k: i for i, k in enumerate(Z["keys"])}; QE = Z["emb"] / np.linalg.norm(Z["emb"], axis=1, keepdims=True)
GS = {r["case_id"]: r for r in json.load(open(O / "outputs/gate_scores.json"))["cases"]}
cases = sorted(FZ["cases"], key=lambda z: MV[z["query_id"]]["eval_t"])
def items_fresh(z):
    s = json.load(open(T2 / "raw/search" / f"{z['case_id']}.json", encoding="utf-8")); org = [o for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")]
    return [(o.get("title"), o.get("snippet"), page(o["link"])["text"]) for o in org[:len(z["cached_url_hashes"])]]
def items_cachedC(z):
    import re, answer_grading as e2e
    l3t = MV[z["query_id"]].get("l3_cache_time", {}); out = []
    for uh, u in zip(z["cached_url_hashes"], z["cached_urls"]):
        t = re.sub(r"\s+", " ", e2e.load_snapshot_text(uh, E["p2"].snapshot_round_on_disk(uh, l3t[uh])) or "").strip() if uh in l3t else page(u)["text"]
        out.append((man.get(uh, {}).get("title"), man.get(uh, {}).get("snippet"), t))
    return out
FR = {z["case_id"]: items_fresh(z) for z in cases}; CC = {z["case_id"]: items_cachedC(z) for z in cases}
enc = None
def gate_score(z, entry, sim):
    global enc
    if enc is None:
        from sentence_transformers import SentenceTransformer; enc = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
    texts = [x[2][:2000] for x in entry["items"]]; qv = Z["emb"][QI[z["incoming_query"]]]
    pe = enc.encode([t for t in texts if t], normalize_embeddings=True) if any(texts) else []; sims = [float(qv @ e) for e in pe]
    f = Lb.features_core(z["incoming_query"], sims, texts, len(texts), entry["creator_query"], sim, z["freshness_class"], MV[z["query_id"]]["eval_t"] - entry["t_reg"], entry["entry_class"])
    return float(MODEL.predict_proba(np.array([[f[k] for k in FEATS]], float))[:, 1][0])
def simulate(policy):
    stream, out = [], []
    for z in cases:
        j = z["case_id"]; tj = MV[z["query_id"]]["eval_t"]; qv = QE[QI[z["incoming_query"]]]; floor = 0.80 if policy == "stricter" else 0.75
        cands = []
        if policy != "always" and z["similarity"] >= floor: cands.append((z["similarity"], {"kind": "heldout", "": j}))
        if policy != "always":
            for e in stream:
                s = float(qv @ QE[QI[e["query"]]])
                if s >= floor and ok(e["entry_class"], tj - e["t_reg"]): cands.append((s, e))
        acc, served = False, None
        if cands:
            s, best = max(cands, key=lambda x: x[0])
            if policy == "gate":
                p = GS[j]["gate_p"] if best["kind"] == "heldout" else gate_score(z, best, s); acc = p >= TAU
            else: acc = True
        if acc and best["kind"] == "heldout":
            served = {"path": "C_freshcache", "": j}; reg = {"query": z["incoming_query"], "t_reg": MV[z["query_id"]]["eval_t"] - z.get("entry_age_s", MV[z["query_id"]]["entry_age_s"]), "entry_class": z["entry_class"], "items": CC[j], "kind": "stream_cached", "": j, "creator_query": z["incoming_query"]}
        elif acc:
            served = {"path": "stream", "": best[""], "src_kind": best["kind"], "items": best["items"], "sim": s}; reg = {**best, "query": z["incoming_query"], "creator_query": z["incoming_query"]}
        else:
            served = {"path": "A_matched", "": j}; reg = {"query": z["incoming_query"], "t_reg": tj, "entry_class": z["freshness_class"], "items": FR[j], "kind": "stream_fresh", "": j, "creator_query": z["incoming_query"]}
        stream.append(reg); out.append({"case_id": j, "accepted": acc, **{k: v for k, v in served.items() if k != "items"}, "items": served.get("items")})
    return out
POLS = ("always", "primary", "stricter", "gate")
if MODE == "build":
    SIM = {p: simulate(p) for p in POLS}; GEN, SUF = {}, {}
    for p, rows in SIM.items():
        for r in rows:
            if r["path"] != "stream": continue
            z = next(c for c in cases if c["case_id"] == r["case_id"]); cx = ctx([x[2] for x in r["items"]]); eb = evblock(r["items"]); key = ["t2fs", h16(cx), h16(z["incoming_query"])]
            r["key"] = key; r["evidence_sha"] = hashlib.sha256(eb.encode()).hexdigest()[:16]; r["usable"] = any(len(x[2]) >= g["MINC"] for x in r["items"])
            GEN.setdefault(tuple(key), {"key": key, "question": z["incoming_query"], "context": cx}); SUF.setdefault((z["query_id"], r["evidence_sha"]), {"query_id": z["query_id"], "arm": r["evidence_sha"], "query": z["incoming_query"], "evidence": eb})
    for rows in SIM.values():
        for r in rows: r.pop("items", None)
    (O / "gpu/in").mkdir(parents=True, exist_ok=True)
    with open(O / "gpu/in/gen_tasks.jsonl", "w") as f: f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in GEN.values()))
    with open(O / "gpu/in/suffv_tasks.jsonl", "w") as f: f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in SUF.values()))
    pickle.dump(SIM, open(O / "work/fullstate_sim.pkl", "wb"))
    print({p: collections.Counter(r["path"] for r in rows) for p, rows in SIM.items()}, "new gen", len(GEN), "new suffv", len(SUF))
if MODE == "score":
    SIM = pickle.load(open(O / "work/fullstate_sim.pkl", "rb"))
    s2 = open((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py")).read(); c2 = s2.index("PAIR = ["); g2 = {"__name__": "t2s", "__file__": str((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py"))}; sys.argv = ["unseen_query_scoring.py", "score"]
    exec(compile(s2[:c2], "<unseen_query_scoring functions>", "exec"), g2); label, suff, C2 = g2["label"], g2["suff"], {c["case_id"]: c for c in g2["C"]}
    jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    ANS = {tuple(r["key"]): r["answer"] for r in jl(O / "gpu/out/answers_llama3b.jsonl")}
    J = {j: {tuple(r["sig"]): r["label"] for r in jl(O / f"gpu/out/judge_{j}.jsonl")} for j in ("llama8b", "qwen7b")}
    SV = {j: {(r["query_id"], r["arm"]): r["label"] for r in jl(O / f"gpu/out/suffv_{j}.jsonl")} for j in ("llama8b", "qwen7b")}
    HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
    CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # component_costs.json (serving_cost_summary.py)
    def lab_stream(z, r):
        a = ANS[tuple(r["key"])]
        if any(x in a.strip().lower() for x in HELD): return "ABSTAIN"
        sig = (h16(z["incoming_query"]), h16(z["reference_answer"]), h16(a)); l = [J[j].get(sig) for j in J]
        return "UNRESOLVED" if None in l else ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG")
    def suf_stream(z, r):
        l = [SV[j].get((z["query_id"], r["evidence_sha"])) for j in SV]; return "UNRESOLVED" if None in l else (l[0] if l[0] == l[1] else "JUDGE_DISAGREEMENT")
    def mcn(b, c): n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    ZZ = {z["case_id"]: z for z in cases}; R = {}; OUTC = {}
    for p, rows in SIM.items():
        res = []
        for r in rows:
            z = ZZ[r["case_id"]]; c = C2[r["case_id"]]; insp = 0.0 if p == "always" else (GS[r["case_id"]]["inspect_ms"] if p == "gate" else 0.1)
            if r["path"] == "stream": lab, sf, usable, cost = lab_stream(z, r), suf_stream(z, r), r["usable"], CG + CE + insp
            elif r["path"] == "C_freshcache": lab, sf, usable, cost = label(c, "C_freshcache"), suff(c, "C_freshcache"), c["cached_avail"], (len(c["cached_urls"]) - c.get("l3_served_pages_C", 0)) * CF + CG + CE + insp
            else: lab, sf, usable, cost = label(c, "A_matched"), suff(c, "A_matched"), c["fresh_avail"], CS + c["k"] * CF + CG + CE + insp
            if not z["reference_answer"]: lab = "NO_REFERENCE"
            res.append({"case_id": r["case_id"], "path": r["path"], "accepted": r["accepted"], "label": lab, "suff": sf, "usable": usable, "cost": cost})
        OUTC[p] = {x["case_id"]: x for x in res}; ref = [x for x in res if x["label"] != "NO_REFERENCE"]; n = len(res)
        R[p] = {"accepted": sum(x["accepted"] for x in res), "served_from_stream_entries": sum(x["path"] == "stream" for x in res), "searches_avoided_pct": round(100 * sum(x["path"] != "A_matched" for x in res) / n, 2),
                "mean_cost_ms": round(sum(x["cost"] for x in res) / n, 1), "sufficient": sum(x["suff"] == "SUFFICIENT" for x in res), "sufficient_pct": round(100 * sum(x["suff"] == "SUFFICIENT" for x in res) / n, 2),
                "accuracy_pct": round(100 * sum(x["label"] == "CORRECT" for x in ref) / len(ref), 2), "wrong": sum(x["label"] == "WRONG" for x in ref), "abstain": sum(x["label"] == "ABSTAIN" for x in ref),
                "unresolved": sum(x["label"] == "UNRESOLVED" for x in ref), "unusable": sum(not x["usable"] for x in res)}
    base = OUTC["always"]
    for p in ("primary", "stricter", "gate"):
        S = OUTC[p]; sl = sum(1 for i in S if base[i]["suff"] == "SUFFICIENT" and S[i]["suff"] != "SUFFICIENT"); sg = sum(1 for i in S if base[i]["suff"] != "SUFFICIENT" and S[i]["suff"] == "SUFFICIENT")
        ids = [i for i in S if S[i]["label"] != "NO_REFERENCE"]; cl = sum(1 for i in ids if base[i]["label"] == "CORRECT" and S[i]["label"] != "CORRECT"); cg = sum(1 for i in ids if base[i]["label"] != "CORRECT" and S[i]["label"] == "CORRECT")
        R[p]["vs_always_search"] = {"suff_lost": sl, "suff_gained": sg, "suff_p": mcn(sl, sg), "suff_diff_pp": round(100 * (sg - sl) / len(S), 2), "correct_lost": cl, "correct_gained": cg, "correct_p": mcn(cl, cg),
                                    "cost_saving_pct": round(100 * (1 - R[p]["mean_cost_ms"] / R["always"]["mean_cost_ms"]), 2)}
    FX = json.load(open(O / "outputs/live_policies.json"))["policies"]; KEYMAP = {"always": "Always search", "primary": "Primary L2 (theta 0.75)", "stricter": "Stricter L2 (theta 0.80)", "gate": "Learned gate (Youden tau)"}
    for p in R: R[p]["fixed_state_reference"] = {k: FX[KEYMAP[p]][k] for k in ("sufficient", "accuracy_pct", "mean_cost_ms", "accepted")}
    json.dump({"semantics": "full-state stream over the live candidates (arrival order; per-policy L2 registration and reuse of earlier candidates' lists; recorded live searches reused; no new API calls)", "policies": R},
              open(O / "outputs/live_policies_fullstate.json", "w"), indent=1)
    for p, v in R.items(): print(p, json.dumps(v))
