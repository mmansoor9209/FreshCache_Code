"""Root diagnostic (2x2) on the confirmation stream (configuration P) and leave-one-root-out on both streams.
The confirmation states are rebuilt offline from the stored searches (gate: identical to the online rows of work/c_rows.pkl for D).
Roots: 40 D list-reuse events (seed 20261017; 30 whose entry D serves later, 10 unexposed). Arms: (fresh|reuse) x (write|no write) from a clone
of D's state at the root; the rest of the stream continues under D's rules. Pages for counterfactual fetches: the store's version at that
request if any policy fetched it, else the latest earlier stored version, else fetched now (counted, disclosed).
Modes: replay | items | fetchmissing | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import collections, contextlib, hashlib, io, json, pickle, random, re, sys, types, pathlib
O = ((OUTPUT_ROOT / "natural_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); RC = CS_DIR / "raw/c"; MODE = sys.argv[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(ROOT / "freshcache/composition"))
import numpy as np   # noqa: E402
import output_workspace; output_workspace.use("natural_workload"); import composition_study_helpers as X, composition_engine_admission_hooks as RV   # noqa: E402
E0 = X.engine(); exp0 = E0["exp"]
F = json.load(open(CS_DIR / "work/c_block.json")); REQ = F["requests"]; BYQ = {r["query_id"]: r for r in REQ}; TORD = {r["t_virtual"]: i for i, r in enumerate(REQ)}
SIM = np.load(CS_DIR / "work/c_sim.npy"); IDX = {}
for i, r in enumerate(REQ): IDX.setdefault(r["query"], i)
class Exp(types.SimpleNamespace):
    def __getattr__(self, k): return getattr(exp0, k)
E = {"exp": Exp(_SIM_MATRIX=SIM, _QUERY_TO_IDX=IDX)}; TH, KA = X.ca_params(0.0, 1.0); K = 2
LINKS = {}
for f in (RC / "search").glob("*.json"):
    s = json.load(open(f, encoding="utf-8")); LINKS[s["query_id"]] = [o["link"] for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")][:10]
stream = [(r["t_virtual"], {"query": r["query"], "query_id": r["query_id"], "freshness_class": r["freshness_class"], "urls": [{"url_hash": u} for u in LINKS[r["query_id"]][:K]]}) for r in REQ]
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
ARMS = [("fresh", True), ("fresh", False), ("reuse", True), ("reuse", False)]; an = lambda a: f"{a[0]}_{'write' if a[1] else 'nowrite'}"
if MODE == "replay":
    CR = pickle.load(open(CS_DIR / "work/c_rows.pkl", "rb"))["rows"]
    st = RV.State(E, len(stream), floor=0.80, theta_by_class=TH, kappa_by_class=KA); D = {}
    for t, r in stream: D[r["query_id"]] = st.step(t, r)
    assert all(sig(D[q]) == sig(CR[(K, "D", q)]) for q in D), "offline replay must equal the online D rows"
    expo = collections.Counter(r["root"] for r in D.values() if r["tier"] == "L1")
    cand = sorted(q for q, r in D.items() if r["tier"] == "L2"); ex = [q for q in cand if expo[q] >= 1]; ux = [q for q in cand if expo[q] == 0]
    rng = random.Random(20261017); roots = sorted(rng.sample(ex, min(30, len(ex)))) + sorted(rng.sample(ux, min(10, len(ux)))); RS = set(roots)
    st = RV.State(E, len(stream), floor=0.80, theta_by_class=TH, kappa_by_class=KA); out = {}
    for i, (t, r) in enumerate(stream):
        q = r["query_id"]
        if q in RS:
            res = {"stratum": "exposed" if expo[q] else "unexposed", "exposure_D": expo[q], "arms": {}}
            for a in ARMS:
                s = st.clone(); root_row = s.step(t, r, force_evidence=a[0], force_write=a[1]); diff = {}
                for t2, r2 in stream[i + 1:]:
                    x = s.step(t2, r2)
                    if sig(x) != sig(D[r2["query_id"]]): diff[r2["query_id"]] = {k: x[k] for k in ("tier", "ev", "gen_query", "root", "root_route") if k in x}
                if a == ("reuse", True): assert not diff and sig(root_row) == sig(D[q])
                res["arms"][an(a)] = {"root_row": {k: root_row[k] for k in ("tier", "ev", "gen_query")}, "diff": diff}
            out[q] = res
        st.step(t, r)
    pickle.dump({"roots": roots, "res": out, "D": D, "candidates": {"exposed": len(ex), "unexposed": len(ux)}}, open(O / "work/root_candidates.pkl", "wb"))
    print("roots", len(roots), "candidates", len(ex), len(ux), "diff sizes", {an(a): sum(len(out[q]["arms"][an(a)]["diff"]) for q in roots) for a in ARMS}); sys.exit(0)
R = pickle.load(open(O / "work/root_candidates.pkl", "rb")); D = R["D"]
PG = RC / "pages"; NEW = O / "raw/root_diagnostic_pages"; NEW.mkdir(parents=True, exist_ok=True)
VERS = collections.defaultdict(list)
for f in PG.glob("*.json"):
    h, o = f.stem.rsplit("_", 1); VERS[h].append(int(o))
uid = lambda u: hashlib.sha1(u.encode()).hexdigest()
def page_file(u, tv):
    h, i = uid(u), TORD[tv]
    if (PG / f"{h}_{i:05d}.json").exists(): return PG / f"{h}_{i:05d}.json", "same_request"
    earlier = [o for o in VERS.get(h, []) if o <= i]
    if earlier: return PG / f"{h}_{max(earlier):05d}.json", "earlier_version"
    return NEW / f"{h}.json", "fetched_now"
def text(u, tv):
    f, _ = page_file(u, tv)
    if not f.exists(): return ""
    d = json.load(open(f, encoding="utf-8")); return re.sub(r"\s+", " ", d.get("extracted_text") or "").strip() if d.get("ok") else ""
def ctx(ev):
    parts, total = [], 0
    for u, tv in ev:
        t = text(u, tv)
        if len(t) < 200: continue
        room = 2800 - total
        if room <= 0: break
        parts.append(t[:room]); total += min(len(t), room)
    return "\n\n".join(parts)
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
def items():
    for q in R["roots"]:
        for a, v in R["res"][q]["arms"].items():
            yield q, a, "root", v["root_row"]
            for q2, x in v["diff"].items(): yield q2, a, q, x; yield q2, "reuse_write", q, D[q2]
if MODE == "fetchmissing":
    sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); import web_collection
    _o = web_collection._is_dynamic_tag; web_collection._is_dynamic_tag = lambda tag: False if getattr(tag, "name", None) in ("html", "body") else _o(tag)
    need = {u for _, _, _, row in items() for u, tv in row["ev"] if page_file(u, tv)[1] == "fetched_now" and not (NEW / f"{uid(u)}.json").exists()}
    import datetime
    for u in sorted(need):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try: d = web_collection.fetch_url(u)
            except Exception: d = None
        json.dump({"url": u, "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "ok": d is not None, "extracted_text": (d or {}).get("extracted_text", "")}, open(NEW / f"{uid(u)}.json", "w"), ensure_ascii=False)
    print("fetched now", len(need)); sys.exit(0)
import live_stream_labeller as CL   # noqa: E402
pool = {}
for d in [CS_DIR / "gpu/out", O / "gpu/out"]:
    for f in sorted(d.glob("answers_llama3b*.jsonl")):
        for r in X.jl(f):
            if r["key"][0] == "cP": pool.setdefault(tuple(r["key"]), r["answer"])
def key(row): cx = ctx([tuple(x) for x in row["ev"]]); return ("cP", h16(cx), h16(row["gen_query"])), cx
lab = CL.Lab()
if MODE == "items":
    G, J, origin = {}, {}, collections.Counter()
    for q, a, _, row in items():
        for u, tv in row["ev"]: origin[page_file(u, tv)[1]] += 1
        k, cx = key(row); ans = pool.get(k)
        if ans is None: G.setdefault(k, {"key": list(k), "question": row["gen_query"], "context": cx}); continue
        l, _, s = lab.label_answer(BYQ[q]["query"], BYQ[q]["reference"], ans)
        if l is None: J.setdefault(s, {"sig": list(s), "q": BYQ[q]["query"], "gold": BYQ[q]["reference"], "cand": ans})
    X.wjl(O / "gpu/in/gen_tasks.jsonl", list(G.values())); X.wjl(O / "gpu/in/judge2_items.jsonl", list(J.values()))
    json.dump(dict(origin), open(O / "work/root_page_origin.json", "w")); print("gen", len(G), "judge2", len(J), "page origin", dict(origin)); sys.exit(0)
if MODE == "score":
    def lb(q, row): return lab.label_answer(BYQ[q]["query"], BYQ[q]["reference"], pool.get(key(row)[0]))[0] or "UNRESOLVED"
    A = [an(a) for a in ARMS]; per = []
    for q in R["roots"]:
        res = R["res"][q]; d = {"root": q, "stratum": res["stratum"]}
        for a in A: d[f"root_{a}"] = lb(q, res["arms"][a]["root_row"])
        down = sorted({q2 for a in A for q2 in res["arms"][a]["diff"]})
        for a in A: d[f"down_correct_{a}"] = sum(lb(q2, res["arms"][a]["diff"].get(q2, D[q2])) == "CORRECT" for q2 in down)
        d["downstream_requests_differing"] = len(down); d["unresolved"] = sum(lb(q2, res["arms"][a]["diff"].get(q2, D[q2])) == "UNRESOLVED" for q2 in down for a in A); per.append(d)
    json.dump({"per_root": per}, open(O / "outputs/root_diagnostic_per_root.json", "w"), indent=1); print("scored", len(per), "unresolved", sum(d["unresolved"] for d in per))
