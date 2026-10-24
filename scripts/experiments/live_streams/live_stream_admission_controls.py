"""Admission controls on the live confirmation stream: D+gamma, D+gate and D+AV in configurations P (depth 2, 3B, 2,800 characters) and
S (depth 5, 14B, 8,000 tokens), replayed offline from the stored searches (the D replay is gated against the online rows). Gate features: the
learned L2 compatibility gate's feature code (features_core) with BGE-M3 query/page embeddings computed here; gamma: max query-page cosine >= 0.55
(unscored accepted); AV: the frozen validation-trained answer verifier on generated answers (iterative: missing answers are generated, then the
replay is rerun). Pages: the evaluator store's version at that request, else the latest stored version before it, else fetched now (counted). Census
labelling of every request an arm changes relative to B, with cluster-robust intervals. Modes: replay CFG | gen14 --gpu G --shard i/n | items CFG | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, collections, contextlib, hashlib, io, json, math, os, pickle, re, sys, time, types, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "natural_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); RC = CS_DIR / "raw/c"; MODE = sys.argv[1]
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
CFG = {"P": {"k": 2, "cell": "a3b", "pfx": "cP"}, "S": {"k": 5, "cell": "q14L", "pfx": "cS"}}
if MODE == "gen14":   # 14B generator for S (same settings as the paraphrase-stream scoring)
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gpu", required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu; si, sn = map(int, a.shard.split("/"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FORCED"}
    mid = "Qwen/Qwen2.5-14B-Instruct"; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    out = O / f"gpu/out/cctl_answers14_shard{si}_of{sn}.jsonl"; done = {tuple(r["key"]) for f in (O / "gpu/out").glob("cctl_answers14_*.jsonl") for r in jl(f)}
    todo = sorted([t for i, t in enumerate(jl(O / "gpu/in/cctl_tasks14.jsonl")) if i % sn == si and tuple(t["key"]) not in done], key=lambda t: len(t["context"]))
    def clip(t):
        t = (t or "").strip()
        if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
        for sep in (".", "\n"):
            if sep in t: return t[:t.index(sep) + 1].strip()
        return t
    BS = int(os.environ.get("GEN_BS", "2")); fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), BS):
        ch = todo[s:s + BS]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": K["FORCED"].format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                  return_tensors="pt", padding=True, truncation=True, max_length=9000).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            fh.write(json.dumps({"key": c["key"], "raw": t, "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n")
        fh.flush()
    sys.exit(0)
sys.path.insert(0, str(ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(ROOT / "freshcache/composition"))
import numpy as np   # noqa: E402
import output_workspace; output_workspace.use("natural_workload"); import composition_study_helpers as X, composition_engine_admission_hooks as RV, live_stream_labeller as CL   # noqa: E402
E0 = X.engine(); exp0 = E0["exp"]
F = json.load(open(CS_DIR / "work/c_block.json")); REQ = F["requests"]; BYQ = {r["query_id"]: r for r in REQ}; TORD = {r["t_virtual"]: i for i, r in enumerate(REQ)}
SIM = np.load(CS_DIR / "work/c_sim.npy"); IDX = {}
for i, r in enumerate(REQ): IDX.setdefault(r["query"], i)
class Exp(types.SimpleNamespace):
    def __getattr__(self, k): return getattr(exp0, k)
E = {"exp": Exp(_SIM_MATRIX=SIM, _QUERY_TO_IDX=IDX)}; TH, KA = X.ca_params(0.0, 1.0)
LINKS = {}
for f in (RC / "search").glob("*.json"):
    s = json.load(open(f, encoding="utf-8")); LINKS[s["query_id"]] = [o["link"] for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")][:10]
def stream(k): return [(r["t_virtual"], {"query": r["query"], "query_id": r["query_id"], "freshness_class": r["freshness_class"], "urls": [{"url_hash": u} for u in LINKS[r["query_id"]][:k]]}) for r in REQ]
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
PG = RC / "pages"; NEW = O / "raw/cctl_pages"; NEW.mkdir(parents=True, exist_ok=True); VERS = collections.defaultdict(list)
for f in PG.glob("*.json"):
    h, o = f.stem.rsplit("_", 1); VERS[h].append(int(o))
uid = lambda u: hashlib.sha1(u.encode()).hexdigest(); FETCHED_NOW = set()
def page_file(u, tv):
    h, i = uid(u), TORD[tv]
    if (PG / f"{h}_{i:05d}.json").exists(): return PG / f"{h}_{i:05d}.json"
    earlier = [o for o in VERS.get(h, []) if o <= i]
    if earlier: return PG / f"{h}_{max(earlier):05d}.json"
    return NEW / f"{h}.json"
_txt = {}
def text(u, tv):
    f = page_file(u, tv)
    if f.parent == NEW and not f.exists():
        sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); import web_collection
        _o = web_collection._is_dynamic_tag; web_collection._is_dynamic_tag = lambda tag: False if getattr(tag, "name", None) in ("html", "body") else _o(tag)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try: d = web_collection.fetch_url(u)
            except Exception: d = None
        json.dump({"url": u, "ok": d is not None, "extracted_text": (d or {}).get("extracted_text", "")}, open(f, "w"), ensure_ascii=False); FETCHED_NOW.add(u)
    if f not in _txt:
        d = json.load(open(f, encoding="utf-8")) if f.exists() else {"ok": False}; _txt[f] = re.sub(r"\s+", " ", d.get("extracted_text") or "").strip() if d.get("ok") else ""
    return _txt[f]
_QT = None
def ctx(ev, cell):
    global _QT
    parts, total = [], 0
    for u, tv in ev:
        t = text(u, tv)
        if len(t) < 200: continue
        if cell == "a3b":
            room = 2800 - total
            if room <= 0: break
            parts.append(t[:room]); total += min(len(t), room)
        else:
            if _QT is None:
                from transformers import AutoTokenizer; _QT = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
            room = 8000 - total
            if room <= 0: break
            ids = _QT(t, add_special_tokens=False)["input_ids"][:room]; parts.append(_QT.decode(ids)); total += len(ids)
    return "\n\n".join(parts)
def key(row, cfg): cx = ctx([tuple(x) for x in row["ev"]], CFG[cfg]["cell"]); return (CFG[cfg]["pfx"], h16(cx), h16(row["gen_query"])), cx
POOLS = [CS_DIR / "gpu/out", O / "gpu/out"]
def pool():
    P = {}
    for d in POOLS:
        for f in sorted(list(d.glob("answers_llama3b*.jsonl")) + list(d.glob("c_answers14_*.jsonl")) + list(d.glob("cctl_answers14_*.jsonl"))):
            for r in X.jl(f): P.setdefault(tuple(r["key"]), r["answer"])
    return P
_emb, _model = {}, None
import atexit as _ax, pickle as _pk
_EC = O / "work/emb_cache_cctl.pkl"
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
CHECK_MS = 30.0
if MODE == "replay":
    cfg = sys.argv[2]; k = CFG[cfg]["k"]; st0 = stream(k); n = len(st0); CR = pickle.load(open(CS_DIR / "work/c_rows.pkl", "rb"))["rows"]
    GATE = (OUTPUT_ROOT / "learned_l2_gate"); sys.path.insert(0, str(ROOT / "scripts/experiments/admission_controls")); import l2_gate_features as Lb
    GZ = pickle.load(open(GATE / "gate_frozen.pkl", "rb")); FEATS, TAU, MODEL = GZ["features"], GZ["tau"], GZ["model"]
    def sims(q, ev):
        ts = [text(u, tv)[:2000] for u, tv in ev]; ok = [t for t in ts if len(t) >= 200]
        if not ok: return [], ts
        qv, *pv = emb([q] + ok); return [float(qv @ p) for p in pv], ts
    def gamma_check(q, ev, meta):
        s, _ = sims(q, ev); return (not s) or max(s) >= 0.55
    def gate_check(q, ev, meta):
        s, ts = sims(q, ev); f = Lb.features_core(q, s, ts, len(ev), meta["eq"], meta["sim"], meta["fc"], meta["age"], meta["efc"])
        return float(MODEL.predict_proba(np.array([[f[x] for x in FEATS]], float))[:, 1][0]) >= TAU
    AV = pickle.load(open(O / "work/av_frozen.pkl", "rb")); P_ = pool(); missing = {}
    import answer_verifier as AVM
    def fit(q_in, q_stored, ev, fc, age, is_write):
        kk, cx = key({"ev": ev, "gen_query": q_stored}, cfg); a = P_.get(kk)
        if a is None: missing.setdefault(kk, {"key": list(kk), "question": q_stored, "context": cx}); return True
        AVM._emb = _emb; AVM._model = _model
        return float(AV["model"].predict_proba(np.array([AVM.feats(q_in, q_stored, a, fc, age, is_write)], float))[:, 1][0]) >= AV["tau"]
    FC = {r["query"]: r["freshness_class"] for r in REQ}
    def serve_check(q, src, t): return fit(q, src["query"], src["ev"], FC[q], t - src["t"], False)
    def write_check(q, ev, t): return fit(q, q, ev, FC[q], 0.0, True)
    def run(**kw):
        s = RV.State(E, n, theta_by_class=TH, kappa_by_class=KA, **kw); rows = {}
        for t, r in st0: rows[r["query_id"]] = s.step(t, r)
        return s.metrics(), rows
    mD, rD = run(floor=0.80); assert all(sig(rD[q]) == sig(CR[(k, "D", q)]) for q in rD), "offline D must equal the online rows"
    mG, rG = run(floor=0.75, l2_check=gamma_check); mT, rT = run(floor=0.75, l2_check=gate_check); mA, rA = run(floor=0.80, serve_check=serve_check, write_check=write_check)
    tk = "gen_tasks_cctl.jsonl" if cfg == "P" else "cctl_tasks14.jsonl"; X.wjl(O / "gpu/in" / tk, list(missing.values()))
    pickle.dump({"metrics": {"D": mD, "Dgamma": mG, "Dgate": mT, "DAV": mA}, "rows": {"D": rD, "Dgamma": rG, "Dgate": rT, "DAV": rA}, "fetched_now": sorted(FETCHED_NOW)}, open(O / f"work/cctl_{cfg}.pkl", "wb"))
    print(cfg, "AV missing answers", len(missing), "fetched now", len(FETCHED_NOW), {a: (m["l2_hits"], m["l1_hits"], m["search_calls"]) for a, m in (("D", mD), ("Dgamma", mG), ("Dgate", mT), ("DAV", mA))}); sys.exit(0)
lab = CL.Lab()
CR = pickle.load(open(CS_DIR / "work/c_rows.pkl", "rb"))["rows"]
def brow(cfg, q): return CR[(CFG[cfg]["k"], "B", q)]
if MODE == "items":
    cfg = sys.argv[2]; W = pickle.load(open(O / f"work/cctl_{cfg}.pkl", "rb")); P_ = pool(); G, J = {}, {}
    for a in ("D", "Dgamma", "Dgate", "DAV"):
        for q in BYQ:
            ra = W["rows"][a][q]
            if sig(ra) == sig(brow(cfg, q)): continue
            for row in (ra, brow(cfg, q)):
                kk, cx = key(row, cfg); ans = P_.get(kk)
                if ans is None: G.setdefault(kk, {"key": list(kk), "question": row["gen_query"], "context": cx}); continue
                l, _, s = lab.label_answer(BYQ[q]["query"], BYQ[q]["reference"], ans)
                if l is None: J.setdefault(s, {"sig": list(s), "q": BYQ[q]["query"], "gold": BYQ[q]["reference"], "cand": ans})
    X.wjl(O / "gpu/in" / ("gen_tasks_cctl.jsonl" if cfg == "P" else "cctl_tasks14.jsonl"), list(G.values())); X.wjl(O / "gpu/in/judge2_items.jsonl", list(J.values()))
    print(cfg, "gen", len(G), "judge2", len(J)); sys.exit(0)
if MODE == "score":
    P_ = pool(); R = {}; COST = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # component_costs.json (serving_cost_summary.py)
    for cfg in CFG:
        W = pickle.load(open(O / f"work/cctl_{cfg}.pkl", "rb")); ids = list(BYQ); n = len(ids); out = {}
        for a in ("D", "Dgamma", "Dgate", "DAV"):
            m = W["metrics"][a]; checks = m["l2_checks"] + m["serve_checks"] + m["write_checks"]
            cost = round((m["search_calls"] * COST[0] + m["fetches"] * COST[1] + m["generations"] * COST[2]) / n + COST[3] + CHECK_MS * checks / n, 1)
            d, lo, ga, un = {}, 0, 0, 0
            for q in ids:
                ra = W["rows"][a][q]; d[q] = 0
                if sig(ra) == sig(brow(cfg, q)): continue
                la = lab.label_answer(BYQ[q]["query"], BYQ[q]["reference"], P_.get(key(ra, cfg)[0]))[0]; lb = lab.label_answer(BYQ[q]["query"], BYQ[q]["reference"], P_.get(key(brow(cfg, q), cfg)[0]))[0]
                if None in (la, lb): un += 1; continue
                if lb == "CORRECT" and la != "CORRECT": d[q] = -1; lo += 1
                elif lb != "CORRECT" and la == "CORRECT": d[q] = 1; ga += 1
            mu = sum(d.values()) / n; g = collections.defaultdict(float)
            for q in ids: g[BYQ[q]["cluster_id"]] += d[q] - mu
            se = math.sqrt(len(g) / (len(g) - 1) * sum(v * v for v in g.values())) / n
            out[a] = {"losses": lo, "gains": ga, "net": ga - lo, "unresolved": un, "stream_pp": round(100 * mu, 3), "ci_pp_cluster": [round(100 * (mu - 1.96 * se), 3), round(100 * (mu + 1.96 * se), 3)],
                      "cost_ms": cost, "checks": checks, "l2_hits": m["l2_hits"], "l1_hits": m["l1_hits"], "searches_per_1k": round(1000 * m["search_calls"] / n, 1)}
        for a in ("Dgamma", "Dgate", "DAV"):
            dc = out[a]["cost_ms"] - out["D"]["cost_ms"]; dl = out["D"]["losses"] - out[a]["losses"]
            out[a]["losses_removed_vs_D"] = dl; out[a]["extra_cost_ms_vs_D"] = round(dc, 1); out[a]["losses_removed_per_ms"] = round(dl / dc, 3) if dc > 0 else None
        R[cfg] = out; R[cfg + "_fetched_now"] = len(W["fetched_now"])
    json.dump(R, open(O / "outputs/live_stream_admission_controls.json", "w"), indent=1); print(json.dumps(R, indent=1))
if MODE == "unsuitable":   # (pre-registered) the 'correct answer reused for an unsuitable question' losses of D vs. B and which arm removes them
    P_ = pool(); R = {}
    def lb(q, row, cfg): return lab.label_answer(BYQ[q]["query"], BYQ[q]["reference"], P_.get(key(row, cfg)[0]))[0]
    for cfg in CFG:
        W = pickle.load(open(O / f"work/cctl_{cfg}.pkl", "rb")); D = W["rows"]["D"]; cases = []
        for q in BYQ:
            x = D[q]
            if sig(x) == sig(brow(cfg, q)) or x["tier"] != "L1": continue
            if lb(q, brow(cfg, q), cfg) != "CORRECT" or lb(q, x, cfg) == "CORRECT": continue
            if lb(x["root"], D[x["root"]], cfg) != "CORRECT": continue
            arms = {}
            for a in ("Dgamma", "Dgate", "DAV"):
                y = W["rows"][a][q]
                arms[a] = "removed" if sig(y) == sig(brow(cfg, q)) or lb(q, y, cfg) == "CORRECT" else ("unresolved" if lb(q, y, cfg) is None else "still_lost")
            cases.append({"query_id": q, "root": x["root"], "root_route": x.get("root_route"), "arms": arms})
        R[cfg] = cases
    json.dump(R, open(O / "outputs/unsuitable_answers.json", "w"), indent=1); print(json.dumps(R, indent=1))
