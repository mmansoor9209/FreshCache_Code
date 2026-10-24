"""Optional control on the recorded natural workload: the ungated composition at L2 floor 0.75 (D075), configuration P only.
B, D, Dgamma, Dgate are read unchanged from work/qrecc_ctrl_P.pkl; only D075 is replayed.
Same candidate-selection semantics as the gated 0.75 arms (st.replay with floor=0.75 and no check). Census against B, D, and the gated arms.
Modes: replay | score   (score writes pending generation / judge / sufficiency tasks until none remain; outputs/samefloor_P.json)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, math, pickle, sys, time, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True; sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import live_stream_library as S
O = S.O; MODE = sys.argv[1]; CFGN = "P"
st = S.Stream("nat"); k = S.CFG[CFGN]["k"]; n = len(st.REQ)
if MODE == "replay":
    QC_RUN = pickle.load(open(O / "work/qrecc_ctrl_P.pkl", "rb")); rows, mets = dict(QC_RUN["rows"]), dict(QC_RUN["metrics"])
    t0 = time.time(); mets["D075"], rows["D075"] = st.replay(k, floor=0.75, l2_check=None); print("D075", mets["D075"], round(time.time() - t0, 1), "s", flush=True)
    V = st.common_sample_rows(); gate = {p: all(S.sig(rows[p][q]) == S.sig(V[(k, p, q)]) for q in st.BYQ) for p in ("B", "D")}; assert all(gate.values()), gate
    unusable = sorted({q for R in rows.values() for q, r in R.items() if any(st.page_file(u, tv) is None for u, tv in r["ev"])})
    pickle.dump({"rows": rows, "metrics": mets, "unusable": unusable}, open(O / "work/shared_floor_P.pkl", "wb"))
    print("gate B,D equal CS_DIR online rows:", gate, "unusable", len(unusable))
if MODE == "score":
    W = pickle.load(open(O / "work/shared_floor_P.pkl", "rb")); rows, mets, skip = W["rows"], W["metrics"], set(W["unusable"])
    P = S.answer_pool(); J = S.Judges(); SUF = {"llama8b": {}, "qwen7b": {}}
    for d in (S.CS_DIR / "gpu/out", S.NW_DIR / "gpu/out", O / "gpu/out"):
        for j in SUF:
            for f in sorted(d.glob(f"suffv_{j}*.jsonl")) + sorted(d.glob(f"cross_suffv_{j}*.jsonl")):
                for r in S.jl(f): SUF[j].setdefault((r["query_id"], r["arm"]), r["label"])
    def suff(q, arm):
        v = [SUF[j].get((q, arm)) for j in SUF]; return None if None in v else v[0] == v[1] == "SUFFICIENT"
    G, JT, SV = {}, {}, {}; R = {"freeze": "frozen configuration", "n_operational": n, "unusable": len(skip), "config": CFGN, "policies": {}, "comparisons": {}}
    for p, m in mets.items(): R["policies"][p] = dict(m, cost_ms=S.cost_ms(m, n))
    def lab(q, row):
        kk, cx, ok = S.Stream.key(st, row, CFGN); a = P.get(kk)
        if a is None: G.setdefault(kk, {"key": list(kk), "question": row["gen_query"], "context": cx}); return None
        l, it = J.two(st.BYQ[q]["query"], st.BYQ[q]["reference"], a)
        if it: JT[tuple(it["sig"])] = it
        return l
    for base, arm in (("B", "D075"), ("D", "D075"), ("D075", "Dgamma"), ("D075", "Dgate")):
        ids = [q for q in st.BYQ if q not in skip]; d = {q: 0 for q in ids}; c = collections.Counter(); sf = sc = 0; sun = 0
        for q in ids:
            ra, rb = rows[base][q], rows[arm][q]
            if S.sig(ra) == S.sig(rb): continue
            c["changed"] += 1; la, lb = lab(q, ra), lab(q, rb)
            for row in (ra, rb):
                eb = st.evblock([tuple(x) for x in row["ev"]]); cx = st.key(row, CFGN)[1]
                for arm_id, ev in (("cfull_" + S.h16(eb), eb), ("cctx_" + S.h16(cx), cx or "(no stored evidence)")):
                    if suff(q, arm_id) is None: SV[(q, arm_id)] = {"query_id": q, "arm": arm_id, "query": st.BYQ[q]["query"], "evidence": ev}
            fa, fb = suff(q, "cfull_" + S.h16(st.evblock([tuple(x) for x in ra["ev"]]))), suff(q, "cfull_" + S.h16(st.evblock([tuple(x) for x in rb["ev"]])))
            xa, xb = suff(q, "cctx_" + S.h16(st.key(ra, CFGN)[1])), suff(q, "cctx_" + S.h16(st.key(rb, CFGN)[1]))
            if None in (fa, fb, xa, xb): sun += 1
            else: sf += int(fb) - int(fa); sc += int(xb) - int(xa)
            if None in (la, lb): c["unresolved"] += 1; continue
            if la == "CORRECT" and lb != "CORRECT": d[q] = -1; c["losses"] += 1; c["losses_wrong" if lb == "WRONG" else "losses_abstain"] += 1
            elif la != "CORRECT" and lb == "CORRECT": d[q] = 1; c["gains"] += 1
        m_ = len(ids); mu = sum(d.values()) / m_; gg = collections.defaultdict(float)
        for q in ids: gg[st.BYQ[q]["cluster_id"]] += d[q] - mu
        se = math.sqrt(len(gg) / (len(gg) - 1) * sum(v * v for v in gg.values())) / m_
        ca, cb = R["policies"][base]["cost_ms"], R["policies"][arm]["cost_ms"]
        R["comparisons"][f"{base}->{arm}"] = dict(c, n_eligible=m_, stream_pp=round(100 * mu, 3), ci_pp=[round(100 * (mu - 1.96 * se), 3), round(100 * (mu + 1.96 * se), 3)],
            losses_per_1k=round(1000 * c["losses"] / m_, 2), gains_per_1k=round(1000 * c["gains"] / m_, 2), cost_base_ms=ca, cost_arm_ms=cb, cost_change_pct=round(100 * (cb / ca - 1), 2),
            suff_full_change_pp=round(100 * sf / m_, 3), suff_ctx_change_pp=round(100 * sc / m_, 3), suff_unresolved=sun)
    S.wjl(O / "gpu/in/shared_floor_gen_P.jsonl", list(G.values())); S.wjl(O / "gpu/in/shared_floor_judge_P.jsonl", list(JT.values())); S.wjl(O / "gpu/in/shared_floor_suffv_P.jsonl", list(SV.values()))
    R["pending"] = {"gen": len(G), "judge": len(JT), "suffv": len(SV)}
    json.dump(R, open(O / "outputs/samefloor_P.json", "w"), indent=1)
    print("pending gen", len(G), "judge", len(JT), "suffv", len(SV))
    for kk, v in R["comparisons"].items(): print(kk, {x: v.get(x) for x in ("changed", "losses", "losses_wrong", "losses_abstain", "gains", "unresolved", "stream_pp", "ci_pp", "cost_change_pct", "suff_ctx_change_pp", "suff_unresolved")})
if MODE == "fixedstate":   # §3: fixed-state decision semantics. Replay D(0.75) with gamma and the gate evaluated at every candidate but never
    # applied (always accept), so every decision is taken at the identical candidate in the identical (D(0.75)) cache state.
    GATE = (OUTPUT_ROOT / "learned_l2_gate"); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/admission_controls"))
    import numpy as np, l2_gate_features as Lb
    GZ = pickle.load(open(GATE / "gate_frozen.pkl", "rb")); GF, GM = GZ["features"], GZ["model"]
    _EC = O / "work/emb_cache_cctl.pkl"; _emb = pickle.load(open(_EC, "rb")); _model = [None]
    def emb(texts):
        need = [t for t in set(texts) if t not in _emb]
        if need:
            if _model[0] is None:
                from sentence_transformers import SentenceTransformer; _model[0] = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
            for t, v in zip(need, _model[0].encode(need, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)): _emb[t] = v
        return [_emb[t] for t in texts]
    def sims(q, ev):
        ts = [(st.text(u, tv) or "")[:2000] for u, tv in ev]; ok = [t for t in ts if len(t) >= 200]
        if not ok: return [], ts
        qv, *pv = emb([q] + ok); return [float(qv @ p) for p in pv], ts
    dec = []
    def record(q, ev, meta):
        s_, ts = sims(q, ev); g_ok = (not s_) or max(s_) >= 0.55
        f = Lb.features_core(q, s_, ts, len(ev), meta["eq"], meta["sim"], meta["fc"], meta["age"], meta["efc"])
        gate_ok = float(GM.predict_proba(np.array([[f[x] for x in GF]], float))[:, 1][0]) >= GZ["tau"]
        dec.append({"query": q, "gamma_accept": bool(g_ok), "gate_accept": bool(gate_ok)}); return True
    m_, rows_ = st.replay(k, floor=0.75, l2_check=record)
    W = pickle.load(open(O / "work/shared_floor_P.pkl", "rb"))
    same = all(S.sig(rows_[q]) == S.sig(W["rows"]["D075"][q]) for q in st.BYQ)
    R = {"trajectory_identical_to_D075": same, "candidates": len(dec), "gamma_reject": sum(not d["gamma_accept"] for d in dec), "gate_reject": sum(not d["gate_accept"] for d in dec),
         "both_reject": sum(not d["gamma_accept"] and not d["gate_accept"] for d in dec), "own_state_rejects": {"gamma": W["metrics"]["Dgamma"]["l2_rejected"], "gate": W["metrics"]["Dgate"]["l2_rejected"]}}
    json.dump(R, open(O / "outputs/fixedstate_P.json", "w"), indent=1); pickle.dump(_emb, open(_EC, "wb")); print(R)
