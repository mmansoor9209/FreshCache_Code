"""Answer verifier (AV) for composition D, trained and thresholded on the VALIDATION stream only.
Modes:
  valset   replay D on validation; sample 1,500 answer-cache serves and 1,000 writes with a reference (seed 20261015); generation tasks
  valitems judge items for the sampled events (frozen two-judge rule) once answers exist
  train    features (BGE-M3 cosines, length, abstention, class, age, token overlap), logistic regression, cluster-grouped 5-fold OOF, Youden
           threshold; frozen model -> work/av_frozen.pkl (sha recorded in outputs/av_train.json)
  heldout  iterative replay of D+AV on the held-out stream (answers missing from every pool are generated, then the replay is rerun until no
           answer is missing); writes work/av_traces.pkl and outputs/av_replay.json
Inspection cost: 30 ms charged per check (pre-registered, matched to the learned L2 compatibility gate)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, math, pickle, random, re, sys, time, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "cross_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
MODE = sys.argv[1]
import output_workspace; output_workspace.use("cross_workload"); import composition_study_helpers as X, composition_engine_admission_hooks as RV   # noqa: E402
import numpy as np   # noqa: E402
E = X.engine(); p2 = E["p2"]; REC = {r["query_id"]: r for r in E["recs"]}; GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid):
    qt = REC[qid]["query"]; g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
TH, KA = X.ca_params(0.0, 1.0); CHECK_MS = 30.0
POOLS = [O.parent / d / "gpu/out" for d in ("composition_checks", "write_controls", "phi4_judge", "common_sample", "natural_workload")] + [O / "gpu/out"]
class Pool:
    def __init__(self):
        self.hl = X.Held(); self.A = {}
        for d in POOLS:
            for f in sorted(list(d.glob("answers_llama3b*.jsonl")) + list(d.glob("answers3b*.jsonl"))):
                for r in X.jl(f): self.A.setdefault(tuple(r["key"]), r["answer"])
    def answer(self, k):
        a = self.hl.answer(list(k)); return a if a is not None else self.A.get(tuple(k))
def key_of(ev, gq): return tuple(X.key_for({"ev": [tuple(x) for x in ev], "gen_query": gq}, gq)[0])
def ctx_of(ev, gq): return X.key_for({"ev": [tuple(x) for x in ev], "gen_query": gq}, gq)[1]
TOK = re.compile(r"[a-z0-9]+"); STOP = set("the a an of in on for to and or is was were are be by with at from as what which who whom whose when where why how did does do has have had that this these those it its".split())
def toks(s): return {w for w in TOK.findall((s or "").lower()) if w not in STOP and len(w) > 1}
FEATS = ["cos_in_stored_q", "cos_in_answer", "cos_stored_q_answer", "answer_len", "abstain", "age_h", "overlap_in_answer", "is_write"] + [f"class_{c}" for c in ("TIMELESS", "SLOW", "MEDIUM", "FAST")]
_emb = {}; _model = None
import atexit as _ax, pickle as _pk
_EC = O / "work/emb_cache_av.pkl"
try: _emb.update(_pk.load(open(_EC, "rb")))   # embedding cache across passes (pure function of the text)
except Exception: pass
_ax.register(lambda: _pk.dump(_emb, open(_EC, "wb")))
def emb(texts):
    global _model
    need = [t for t in set(texts) if t not in _emb]
    if need:
        if _model is None:
            from sentence_transformers import SentenceTransformer; _model = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
        for t, v in zip(need, _model.encode(need, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)): _emb[t] = v
    return [_emb[t] for t in texts]
def feats(q_in, q_stored, ans, fc, age_s, is_write):
    a, b, c = emb([q_in, q_stored, ans]); ab = any(h in ans.strip().lower() for h in X.HELD); ti = toks(q_in)
    f = {"cos_in_stored_q": float(a @ b), "cos_in_answer": float(a @ c), "cos_stored_q_answer": float(b @ c), "answer_len": len(ans.split()), "abstain": float(ab),
         "age_h": age_s / 3600.0, "overlap_in_answer": len(ti & toks(ans)) / len(ti) if ti else 0.0, "is_write": float(is_write)}
    for k in ("TIMELESS", "SLOW", "MEDIUM", "FAST"): f[f"class_{k}"] = float(fc == k)
    return [f[k] for k in FEATS]
if MODE == "valset":
    V = X.C.streams(E)["validation"]; s = RV.State(E, len(V), floor=0.80, theta_by_class=TH, kappa_by_class=KA); rows = {}
    for t, r in V: rows[r["query_id"]] = s.step(t, r)
    serves = sorted(q for q, r in rows.items() if r["tier"] == "L1" and goldof(q)); writes = sorted(q for q, r in rows.items() if r["tier"] in ("miss", "L2") and goldof(q))
    rng = random.Random(20261015); S1 = rng.sample(serves, 1500); S2 = rng.sample(writes, 1000); pool = Pool(); EV = []; GEN = {}
    for q in S1:
        r = rows[q]; EV.append({"type": "serve", "q": q, "q_stored": r["gen_query"], "ev": [list(x) for x in r["ev"]], "fc": REC[q]["freshness_class"], "age": r["t"] - r["stored_answer_t"]})
    for q in S2:
        r = rows[q]; EV.append({"type": "write", "q": q, "q_stored": REC[q]["query"], "ev": [list(x) for x in r["ev"]], "fc": REC[q]["freshness_class"], "age": 0.0})
    for e in EV:
        k = key_of(e["ev"], e["q_stored"]); e["key"] = list(k)
        if pool.answer(k) is None: GEN.setdefault(k, {"key": list(k), "question": e["q_stored"], "context": ctx_of(e["ev"], e["q_stored"])})
    json.dump(EV, open(O / "work/av_valset.json", "w")); X.wjl(O / "gpu/in/gen_tasks.jsonl", list(GEN.values()))
    print("serves", len(serves), "writes", len(writes), "sampled", len(EV), "gen", len(GEN)); sys.exit(0)
if MODE == "valitems":
    EV = json.load(open(O / "work/av_valset.json")); pool = Pool(); IT = {}; miss = 0
    lab = __import__("live_stream_labeller").Lab()
    for e in EV:
        a = pool.answer(tuple(e["key"]))
        if a is None: miss += 1; continue
        l, ans, sg = lab.label_answer(REC[e["q"]]["query"], goldof(e["q"]), a)
        if l is None: IT.setdefault(sg, {"sig": list(sg), "q": REC[e["q"]]["query"], "gold": goldof(e["q"]), "cand": a})
    X.wjl(O / "gpu/in/judge2_items.jsonl", list(IT.values())); print("judge2", len(IT), "missing answers", miss); sys.exit(0)
if MODE == "train":
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    EV = json.load(open(O / "work/av_valset.json")); pool = Pool(); lab = __import__("live_stream_labeller").Lab(); Xf, y, g = [], [], []
    for e in EV:
        a = pool.answer(tuple(e["key"])); l = lab.label_answer(REC[e["q"]]["query"], goldof(e["q"]), a)[0]
        if l is None: continue
        Xf.append(feats(REC[e["q"]]["query"], e["q_stored"], a, e["fc"], e["age"], e["type"] == "write")); y.append(int(l == "CORRECT")); g.append(REC[e["q"]].get("cluster_id") or e["q"])
    Xf, y = np.array(Xf, float), np.array(y)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xf, y, g):
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xf[tr], y[tr]); oof[te] = m.predict_proba(Xf[te])[:, 1]
    best = max(((float(t), (oof[y == 1] >= t).mean() + (oof[y == 0] < t).mean() - 1) for t in np.unique(oof)), key=lambda x: x[1])
    M = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xf, y); blob = pickle.dumps({"model": M, "features": FEATS, "tau": best[0]})
    open(O / "work/av_frozen.pkl", "wb").write(blob)
    R = {"n_events": int(len(y)), "positives": int(y.sum()), "oof_auc": round(float(roc_auc_score(y, oof)), 3), "tau_youden": round(best[0], 4), "youden_J": round(float(best[1]), 3),
         "model_sha256": hashlib.sha256(blob).hexdigest(), "features": FEATS, "check_ms_charged": CHECK_MS}
    json.dump(R, open(O / "outputs/av_train.json", "w"), indent=1); print(json.dumps(R, indent=1)); sys.exit(0)
if MODE == "heldout":
    AV = pickle.load(open((OUTPUT_ROOT / "natural_workload/work/av_frozen.pkl"), "rb")); M, TAU = AV["model"], AV["tau"]; pool = Pool(); H = X.C.streams(E)["heldout"]; n = len(H)
    missing = {}
    def fit(q_in, q_stored, ev, fc, age, is_write):
        k = key_of(ev, q_stored); a = pool.answer(k)
        if a is None: missing.setdefault(k, {"key": list(k), "question": q_stored, "context": ctx_of(ev, q_stored)}); return True
        return float(M.predict_proba(np.array([feats(q_in, q_stored, a, fc, age, is_write)], float))[:, 1][0]) >= TAU
    RECQ = {r["query"]: r for _, r in H}
    def serve_check(q, src, t): return fit(q, src["query"], src["ev"], RECQ[q]["freshness_class"], t - src["t"], False)
    def write_check(q, ev, t): return fit(q, q, ev, RECQ[q]["freshness_class"], 0.0, True)
    s = RV.State(E, n, floor=0.80, theta_by_class=TH, kappa_by_class=KA, serve_check=serve_check, write_check=write_check); rows = {}
    for t, r in H: rows[r["query_id"]] = s.step(t, r)
    X.wjl(O / "gpu/in/gen_tasks.jsonl", list(missing.values())); print("missing answers this pass", len(missing))
    if True:   # written every pass; residual missing answers are fail-open (accepted) and recorded
        m = s.metrics(); CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]; nonrt = sum(1 for _, r in H if r["freshness_class"] != "REAL_TIME") / n   # cost coefficients from component_costs.json (serving_cost_summary.py)
        insp = CHECK_MS * (m["serve_checks"] + m["write_checks"]) / n
        R = dict(m, missing_answers_fail_open=len(missing), search_per_1k=round(1000 * m["search_calls"] / n, 1), fetch_per_1k=round(1000 * m["fetches"] / n, 1), gen_per_1k=round(1000 * m["generations"] / n, 1),
                 inspection_ms_per_request=round(insp, 2), cost_ms=round((m["search_calls"] * CS + m["fetches"] * CF + m["generations"] * CG) / n + nonrt * CE + insp, 1))
        pickle.dump({"DAV": rows}, open(O / "work/av_traces.pkl", "wb")); json.dump(R, open(O / "outputs/av_replay.json", "w"), indent=1); print(json.dumps(R, indent=1))
