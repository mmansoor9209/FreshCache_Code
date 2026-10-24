"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

Shared helpers for the SemanticTTL / CategoryAware baseline selection. All pre-existing files are read-only; outputs only inside outputs/baseline_selection/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import copy, csv, hashlib, importlib.util, json, math, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "baseline_selection"); CRR = (OUTPUT_ROOT / "replay")
STTL_FRONTIER_DIR = (OUTPUT_ROOT / "semanticttl_frontier")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine"))
import engine_loader as C, answer_pools as P   # noqa: E402
import numpy as np   # noqa: E402
jl, h16 = P.jl, P.h16
LOGF = open(O / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
_E = None
def engine():
    global _E
    if _E is None: _E = C.load("reported")
    return _E
def q2():
    E = engine(); spec = importlib.util.spec_from_file_location("quality_selection_sample", ROOT / "scripts/evaluation/answer_audits/quality_selection_sample.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); assert m.P2 is E["p2"]; return m
class patched:   # SemanticTTL (theta, kappa) in memory exactly as in supplementary_replays; always restored
    def __init__(self, theta, kappa): self.t, self.k = theta, kappa; self.exp = engine()["exp"]
    def __enter__(self):
        self.ottl = copy.deepcopy(self.exp.FIXED_TTL); self.ol1 = self.exp.L1_SIM_THRESHOLD
        self.exp.L1_SIM_THRESHOLD = self.t; self.exp.FIXED_TTL.clear(); self.exp.FIXED_TTL.update({c: v * self.k for c, v in self.ottl.items()}); return self
    def __exit__(self, *a):
        self.exp.FIXED_TTL.clear(); self.exp.FIXED_TTL.update(self.ottl); self.exp.L1_SIM_THRESHOLD = self.ol1
        assert self.exp.FIXED_TTL == {"TIMELESS": 2592000, "SLOW": 604800, "MEDIUM": 86400, "FAST": 3600, "REAL_TIME": 0} and self.exp.L1_SIM_THRESHOLD == 0.4

def ca_replay(stream, theta_by_class, kappa_by_class):
    """CategoryAware-style answer-only cache: quality_selection_sample.sttl_replay (SemanticTTL: l1 sim_ttl, no L2/L3) with the similarity threshold
    taken from the REQUEST's class and the TTL (base FIXED_TTL x kappa) from the cached ENTRY's class. One shared cache (no partition).
    With identical parameters for every class it must equal sttl_replay (gated by the caller)."""
    E = engine(); exp = E["exp"]; SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    BASE = {"TIMELESS": 2592000, "SLOW": 604800, "MEDIUM": 86400, "FAST": 3600, "REAL_TIME": 0}
    ttl = {c: BASE[c] * kappa_by_class.get(c, 0.0) for c in BASE}
    ok = lambda fc, age: bool(ttl.get(fc, 0) and age <= ttl.get(fc, 0))
    n = len(stream); qi_ = np.empty(n, np.int64); t_ = np.empty(n, np.float64); fcs, qs, src = [], [], []; n1 = l1h = search = fetches = 0; rows = {}
    for t, r in stream:
        q, fc, qid = r["query"], r["freshness_class"], r["query_id"]; own = [u["url_hash"] for u in r["urls"]]
        if fc == "REAL_TIME":
            search += 1; fetches += len(own); rows[qid] = {"tier": "REAL_TIME", "t": t, "ev": [(u, t) for u in own], "gen_query": q}; continue
        row = SIM[IDX[q]]; hit = -1
        if n1:
            sims = row[qi_[:n1]]; cand = np.nonzero(sims >= theta_by_class[fc])[0]
            if cand.size:
                keep = cand[np.fromiter((ok(fcs[int(j)], t - t_[j]) for j in cand), bool, cand.size)]
                if keep.size: hit = int(keep[np.argsort(-sims[keep], kind="stable")][0])
        if hit >= 0:
            l1h += 1; s = src[hit]; rows[qid] = {"tier": "L1", "t": t, "similarity": float(row[qi_[hit]]), "age": t - float(t_[hit]), "matched_query": qs[hit],
                                                  "ev": s["ev"], "gen_query": s["query"], "stored_answer_t": s["t"], "entry_class": fcs[hit]}; continue
        search += 1; fetches += len(own); ev = [(u, t) for u in own]
        rows[qid] = {"tier": "miss", "t": t, "age": 0.0, "ev": ev, "gen_query": q}
        qi_[n1] = IDX[q]; t_[n1] = t; fcs.append(fc); qs.append(q); src.append({"query": q, "ev": ev, "t": t}); n1 += 1
    return {"n": n, "search_calls": search, "search_saved_pct": round(100 * (1 - search / n), 4), "l1_hits": l1h, "fetches": fetches, "generations": search,
            "fetch_per_1k": round(1000 * fetches / n, 1), "gen_per_1k": round(1000 * search / n, 1)}, rows

def drift_from_rows(E, stream, rows, S):
    """Engine L1-path drift semantics for an answer-only cache: an L1 hit is scored over the stored entry's evidence (pages at write time t)
    with mixed_age_replay.outcome/aggregate (empty -> UNOBSERVABLE); misses and REAL_TIME have no reuse. Also returns page events for the cross-round audit."""
    me = E["me"]; rounds = E["rounds"]; per, events = {}, {}
    for t, r in stream:
        q = r["query_id"]; p = rows[q]
        if p["tier"] != "L1": per[q] = None; continue
        labs = [me.outcome(u, tc, t, rounds) for u, tc in p["ev"]]; o = me.aggregate(labs); per[q] = me.UNOBS if o is None else o
        events[q] = [(u, tc, t, l) for (u, tc), l in zip(p["ev"], labs)]
    ch = sum(1 for q in S if per.get(q) == me.CHANGED); un = sum(1 for q in S if per.get(q) == me.UNCHANGED)
    return per, events, {"changed": ch, "unchanged": un, "drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None, "coverage_pct": round(100 * (ch + un) / len(S), 4)}

def cross_round_audit(E, S, per, events):
    me = E["me"]; RT = E["rounds"]; ch = un = 0
    for q in S:
        o = per.get(q)
        if o is None: continue
        pages = []
        for u, tc, tr, lab in events.get(q, []):
            p_ = RT.get(u) or {}; rc, rs = me.round_at(p_, tc), me.round_at(p_, tr); pages.append((rc is not None and rc == rs, lab))
        if any((not s) and l == me.CHANGED for s, l in pages): ch += 1
        elif pages and all((not s) and l == me.UNCHANGED for s, l in pages): un += 1
    return {"cross_round_determinate": ch + un, "cross_round_changed": ch, "cross_round_drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None}

PRIM = ("i don't know", "i dont know", "cannot determine", "not enough information", "insufficient", "no information", "unclear from the context")
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
class Val:   # validation labelling rule (as in quality_selection_and_8b_replication.py), with earlier answer pools as fallback
    def __init__(self):
        QB = (OUTPUT_ROOT / "quality_selection")
        self.qa = {tuple(r["key"]): r["answer"] for r in jl(QB / "q_answers.jsonl")}
        self.qj = {x: {tuple(r["sig"]): r["label"] for r in jl(QB / f"q_judge_{x}.jsonl")} for x in ("llama8b", "qwen7b")}
        self.GA = P.AnswerLookup(extra=[CRR / "gpu/out/answers_llama3b.jsonl", CRR / "gpu/out3/answers_llama3b.jsonl", STTL_FRONTIER_DIR / "gpu/out/val_answers_llama3b.jsonl", O / "gpu/out/answers_llama3b.jsonl"])
        self.GJ = P.judge2(extra=[CRR / "gpu/out", CRR / "gpu/out3", STTL_FRONTIER_DIR / "gpu/out", O / "gpu/out"])
    def answer(self, key): a = self.qa.get(tuple(key)); return a if a is not None else self.GA.get(list(key))
    def labels(self, q, gold, key):
        a = self.answer(key)
        if a is None: return None, None, None
        if any(p in a.strip().lower() for p in PRIM): return "ABSTAIN", a, None
        s = (h16(q), h16(gold), h16(a)); l = [self.qj[x].get(s, self.GJ[x].get(s)) for x in ("llama8b", "qwen7b")]
        if None in l: return None, a, s
        return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), a, tuple(l)
class Held:  # held-out labelling rule (as in answer_quality_evaluation.py), with earlier answer pools as fallback
    def __init__(self):
        V = (OUTPUT_ROOT / "archived_runs"); A2, HS, PC = ((OUTPUT_ROOT / "general_audit")), ((OUTPUT_ROOT / "quality_selection/heldout_sensitivity")), ((OUTPUT_ROOT / "predicted_class_audit"))
        self.exact, self.content = {}, {}
        for f in (A2 / "answers.jsonl", HS / "h_answers.jsonl"):
            for r in jl(f): self.exact[tuple(r["key"])] = r["answer"]
        for r in jl(PC / "new_answers.jsonl"): self.content.setdefault((r["ctx_sha"], r["q_sha"]), r["answer"])
        for k, a in self.exact.items(): self.content.setdefault((k[1], k[2]), a)
        self.j = {"llama8b": {}, "qwen7b": {}}
        for j in self.j:
            for f in (A2 / f"judge_{j}.jsonl", HS / f"h_judge_{j}.jsonl", PC / f"new_judge_{j}.jsonl"):
                for r in jl(f):
                    if "sig" in r: self.j[j][tuple(r["sig"])] = r["label"]
        self.GA = P.AnswerLookup(extra=[CRR / "gpu/out/answers_llama3b.jsonl", STTL_FRONTIER_DIR / "gpu/out/ho_answers_llama3b.jsonl", O / "gpu/out/answers_llama3b.jsonl"])
        self.GJ = P.judge2(extra=[CRR / "gpu/out", STTL_FRONTIER_DIR / "gpu/out", O / "gpu/out"])
    def answer(self, key):
        if tuple(key) in self.exact: return self.exact[tuple(key)]
        k = (key[1], key[2]); return self.content[k] if k in self.content else self.GA.get(list(key))
    def labels(self, q, gold, key):
        a = self.answer(key)
        if a is None: return None, None, None
        if any(p in a.strip().lower() for p in HELD): return "ABSTAIN", a, None
        s = (h16(q), h16(gold), h16(a)); l = []
        for j in ("llama8b", "qwen7b"):
            v = self.j[j].get(s); l.append(v if v is not None else self.GJ[j].get(s))
        if None in l: return None, a, s
        return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), a, tuple(l)
def judge_items(lab, items):
    out, miss = {}, []
    for q, gold, key in items:
        l, a, s = lab.labels(q, gold, key)
        if a is None: miss.append(key); continue
        if l is None and s is not None: out.setdefault(s, {"sig": list(s), "q": q, "gold": gold, "cand": a})
    return list(out.values()), miss
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
