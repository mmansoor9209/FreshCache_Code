"""Shared library for the two live-searched streams (paraphrase live stream and natural QReCC workload).
Store-only pages (no fetching): version at the request, else the latest stored version before it; else 'missing' (the request is unusable where needed).
Context assembly, generator inputs and keys are byte-identical to the common-sample scripts/evaluation, so every earlier answer and label is reused."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, os, pickle, re, sys, types, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
O = ((OUTPUT_ROOT / "cross_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); NW_DIR = ((OUTPUT_ROOT / "natural_workload")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(ROOT / "scripts/experiments/live_streams/recorded_stream_controls")); sys.path.insert(0, str(ROOT / "freshcache/composition"))
import numpy as np   # noqa: E402
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
CFG = {"P": {"k": 2, "cell": "a3b"}, "S": {"k": 5, "cell": "q14L"}}
SPEC = {"para": {"block": CS_DIR / "work/c_block.json", "sim": CS_DIR / "work/c_sim.npy", "rc": CS_DIR / "raw/c", "rows": CS_DIR / "work/c_rows.pkl", "pfx": {"P": "cP", "S": "cS"}},
        "nat": {"block": NW_DIR / "work/qr_block.json", "sim": NW_DIR / "work/qr_sim.npy", "rc": NW_DIR / "raw/qr", "rows": NW_DIR / "work/qr_rows.pkl", "pfx": {"P": "qP", "S": "qS"}}}
POLS = {"B": dict(l2_on=False, write_rule="all"), "D": dict(l2_on=True, write_rule="all"), "Dp": dict(l2_on=True, write_rule="no_l2"),
        "Be": dict(l2_on=False, write_rule="elig")}   # Bm = mask(D' L2 set), built in replay_policies

class Stream:
    def __init__(self, name, order=None, classes=None, any_version=False):
        sp = SPEC[name]; self.name, self.sp = name, sp; self.any_version = any_version
        F = json.load(open(sp["block"])); self.REQ = [dict(r) for r in F["requests"]]
        for r in self.REQ: r["t_virtual"] = float(r["t_virtual"])
        self.BYQ = {r["query_id"]: r for r in self.REQ}
        self.TORD = {r["query_id"]: i for i, r in enumerate(self.REQ)}   # original collection order (page versions)
        self.TQ = {r["t_virtual"]: r["query_id"] for r in self.REQ}
        if classes: [r.update(freshness_class=classes[r["query_id"]]) for r in self.REQ]
        self.order = order   # optional {query_id: new virtual time}
        SIM = np.load(sp["sim"]); IDX = {}
        for i, r in enumerate(self.REQ): IDX.setdefault(r["query"], i)
        import output_workspace; output_workspace.use("cross_workload"); import composition_study_helpers as X
        self.X = X; E0 = X.engine(); exp0 = E0["exp"]
        class Exp(types.SimpleNamespace):
            def __getattr__(s, k): return getattr(exp0, k)
        self.E = {"exp": Exp(_SIM_MATRIX=SIM, _QUERY_TO_IDX=IDX)}; self.TH, self.KA = X.ca_params(0.0, 1.0)
        self.LINKS = {}
        for f in (sp["rc"] / "search").glob("*.json"):
            s = json.load(open(f, encoding="utf-8")); self.LINKS[s["query_id"]] = [o["link"] for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")][:10]
        self.META = {}
        for f in sorted((sp["rc"] / "search").glob("*.json")):
            s = json.load(open(f, encoding="utf-8"))
            for o in ((s.get("response") or {}).get("organic") or []):
                if o.get("link"): self.META.setdefault(o["link"], (o.get("title") or "", o.get("snippet") or ""))
        self.PG = sp["rc"] / "pages"; self.VERS = collections.defaultdict(list)
        for f in self.PG.glob("*.json"):
            h, o = f.stem.rsplit("_", 1); self.VERS[h].append(int(o))
        self._txt = {}; self._QT = None; self.missing_pages = set()
    # ---- stream in arrival order (virtual time); the evidence timestamp tv maps back to the collection request for page versions
    def stream(self, k):
        T = self.order or {r["query_id"]: r["t_virtual"] for r in self.REQ}
        out = [(T[r["query_id"]], {"query": r["query"], "query_id": r["query_id"], "freshness_class": r["freshness_class"],
                                   "urls": [{"url_hash": u} for u in self.LINKS.get(r["query_id"], [])[:k]]}) for r in self.REQ]
        return sorted(out, key=lambda x: x[0])
    def req_of_t(self, tv):
        if self.order is None: return self.TQ[tv]
        if not hasattr(self, "_rt"): self._rt = {t: q for q, t in self.order.items()}
        return self._rt[tv]
    def page_file(self, u, tv, any_version=None):
        any_version = self.any_version if any_version is None else any_version
        h = hashlib.sha1(u.encode()).hexdigest(); i = self.TORD[self.req_of_t(tv)]; v = self.VERS.get(h, [])
        if i in v: return self.PG / f"{h}_{i:05d}.json"
        earlier = [o for o in v if o <= i]
        if earlier: return self.PG / f"{h}_{max(earlier):05d}.json"
        if any_version and v: return self.PG / f"{h}_{min(v):05d}.json"
        return None
    def text(self, u, tv, any_version=None):
        f = self.page_file(u, tv, any_version)
        if f is None: self.missing_pages.add(u); return None
        if f not in self._txt:
            d = json.load(open(f, encoding="utf-8")); self._txt[f] = re.sub(r"\s+", " ", d.get("extracted_text") or "").strip() if d.get("ok") else ""
        return self._txt[f]
    def ctx(self, ev, cell, any_version=None):
        """frozen assembly; returns (context, complete) — complete is False when a page is absent from the store"""
        parts, total, ok = [], 0, True
        for u, tv in ev:
            t = self.text(u, tv, any_version)
            if t is None: ok = False; continue
            if len(t) < 200: continue
            if cell == "a3b":
                room = 2800 - total
                if room <= 0: break
                parts.append(t[:room]); total += min(len(t), room)
            else:
                if self._QT is None:
                    from transformers import AutoTokenizer; self._QT = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
                room = 8000 - total
                if room <= 0: break
                ids = self._QT(t, add_special_tokens=False)["input_ids"][:room]; parts.append(self._QT.decode(ids)); total += len(ids)
        return "\n\n".join(parts), ok
    def evblock(self, ev, any_version=None):
        lines = []
        for u, tv in ev:
            title, snip = self.META.get(u, ("", "")); t = self.text(u, tv, any_version) or ""; blk = f"{len(lines) + 1}. {title.strip()}"
            if snip: blk += f"\n   {snip.strip()}"
            if len(t) >= 200: blk += f"\n   PAGE TEXT: {t[:2000]}"
            lines.append(blk)
        return "\n".join(lines) or "(no stored evidence)"
    def key(self, row, cfg, any_version=None):
        cx, ok = self.ctx([tuple(x) for x in row["ev"]], CFG[cfg]["cell"], any_version)
        return (self.sp["pfx"][cfg], h16(cx), h16(row["gen_query"])), cx, ok
    # ---- replays
    def state(self, k, **kw):
        import composition_engine_admission_hooks as RV
        return RV.State(self.E, len(self.REQ), theta_by_class=self.TH, kappa_by_class=self.KA, **dict(dict(floor=0.80), **kw))
    def replay(self, k, **kw):
        s = self.state(k, **kw); rows = {}
        for t, r in self.stream(k): rows[r["query_id"]] = s.step(t, r)
        return s.metrics(), rows
    def replay_policies(self, k):
        out = {}
        for p, kw in POLS.items(): out[p] = self.replay(k, **kw)
        mask = {q for q, r in out["Dp"][1].items() if r["tier"] == "L2"}
        out["Bm"] = self.replay(k, l2_on=False, write_rule="mask", mask=mask); return out
    def common_sample_rows(self):
        return pickle.load(open(self.sp["rows"], "rb"))["rows"]

# ---- answers and labels (all answer pools)
ANSWER_GLOBS = ["answers_llama3b*.jsonl", "answers3b*.jsonl", "c_answers14_*.jsonl", "cctl_answers14_*.jsonl", "qr_answers14_*.jsonl", "answers14*.jsonl"]
def answer_pool():
    P = {}
    for d in [CS_DIR / "gpu/out", NW_DIR / "gpu/out", O / "gpu/out"]:
        for g in ANSWER_GLOBS:
            for f in sorted(d.glob(g)):
                for r in jl(f): P.setdefault(tuple(r["key"]), r["answer"])
    return P
class Judges:
    def __init__(self, extra=("mistral7b", "phi4")):
        import live_stream_labeller as CL; self.lab = CL.Lab()
        self.J = {j: {} for j in ("llama8b", "qwen7b") + tuple(extra)}
        for d in [CS_DIR / "gpu/out", NW_DIR / "gpu/out", O / "gpu/out"]:
            for j in self.J:
                for f in sorted(d.glob(f"judge_{j}*.jsonl")):
                    for r in jl(f): self.J[j].setdefault(tuple(r["sig"]), r["label"])
    def raw(self, j, s):
        """raw vote of judge j for signature s from every pool (frozen project pools first)"""
        if j in ("llama8b", "qwen7b"):
            for d in (self.lab.hl.j[j], self.lab.hl.GJ[j], self.lab.J[j]):
                v = d.get(s)
                if v is not None: return v
        return self.J[j].get(s)
    def two(self, q, gold, a):
        """frozen two-judge rule; returns (label, item_or_None)"""
        if a is None: return None, None
        l, _, s = self.lab.label_answer(q, gold, a)
        if l is None:
            s = (h16(q), h16(gold), h16(a)); v = [self.J[j].get(s) for j in ("llama8b", "qwen7b")]
            if None in v: return None, {"sig": list(s), "q": q, "gold": gold, "cand": a}
            l = "CORRECT" if v == ["CORRECT", "CORRECT"] else "WRONG"
        return l, None

def census(st, rows_a, rows_b, cfg, P, J, skip=frozenset()):
    """paired census of B-type rows_a vs policy rows_b: losses, gains, clustered CI; returns stats and missing gen/judge tasks"""
    import math
    ids = [r["query_id"] for r in st.REQ]; n = len(ids); d = {q: 0 for q in ids}; G, Jt = {}, {}; lo = ga = un = 0; unusable = 0
    for q in ids:
        if q in skip: continue
        a, b = rows_a[q], rows_b[q]
        if sig(a) == sig(b): continue
        la = lb = None
        for row in (a, b):
            kk, cx, ok = st.key(row, cfg)
            ans = P.get(kk)
            if ans is None: G.setdefault(kk, {"key": list(kk), "question": row["gen_query"], "context": cx}); continue
            l, it = J.two(st.BYQ[q]["query"], st.BYQ[q]["reference"], ans)
            if it: Jt[tuple(it["sig"])] = it
            if row is a: la = l
            else: lb = l
        if None in (la, lb): un += 1; continue
        if la == "CORRECT" and lb != "CORRECT": d[q] = -1; lo += 1
        elif la != "CORRECT" and lb == "CORRECT": d[q] = 1; ga += 1
    mu = sum(d.values()) / n; g = collections.defaultdict(float)
    for q in ids: g[st.BYQ[q]["cluster_id"]] += d[q] - mu
    se = math.sqrt(len(g) / (len(g) - 1) * sum(v * v for v in g.values())) / n
    return {"losses": lo, "gains": ga, "unresolved": un, "stream_pp": round(100 * mu, 3), "ci_pp": [round(100 * (mu - 1.96 * se), 3), round(100 * (mu + 1.96 * se), 3)], "d": d}, G, Jt

COST = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]; CHECK_MS = 30.0   # cost coefficients from component_costs.json (serving_cost_summary.py)
def cost_ms(m, n):
    checks = m.get("l2_checks", 0) + m.get("serve_checks", 0) + m.get("write_checks", 0)
    return round((m["search_calls"] * COST[0] + m["fetches"] * COST[1] + m["generations"] * COST[2]) / n + COST[3] + CHECK_MS * checks / n, 1)
