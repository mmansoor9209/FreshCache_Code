"""Common 1,500-request sample of B, B-minus(mask), B-minus(elig), D, D' with answer-cache lineage (design fixed in advance).
Frame U = held-out requests with a reference (not REAL_TIME) whose signature differs from B under at least one other policy; SRS n=1,500,
seed 20261008. Modes: build | items | score.
build : sample, per-(request, policy) generation key, context, full evidence block; root requests of every answer-cache hit -> work/common.pkl
items : generation tasks for answers missing from every pool; when complete, judge items (requests and roots) and SUFFV tasks (full block, context)
score : outputs/common_sample_construction.json (accounting table, decomposition, lineage); see common_sample_scoring.py"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, pickle, random, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("common_sample"); import composition_study_helpers as X   # noqa: E402
MODE = sys.argv[1]; IMP = (OUTPUT_ROOT)
POL = ("B", "Bm", "Be", "D", "Dp")
POOLDIRS = [((OUTPUT_ROOT / "composition_checks/gpu/out")), ((OUTPUT_ROOT / "write_controls/gpu/out")), ((OUTPUT_ROOT / "phi4_judge/gpu/out")), O / "gpu/out"]   # earlier outputs (read-only) + this one
E = X.engine(); p2 = E["p2"]; REC = {r["query_id"]: r for r in E["recs"]}; GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid):
    qt = REC[qid]["query"]; g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))


class Lab:
    """Frozen two-judge rule (Held) over every earlier answer/judge pool plus the outputs of this script."""
    def __init__(self):
        self.hl = X.Held(); self.A = {}; self.J = {"llama8b": {}, "qwen7b": {}}
        for d in POOLDIRS:
            for f in sorted(d.glob("answers_llama3b*.jsonl")):
                for r in X.jl(f): self.A.setdefault(tuple(r["key"]), r["answer"])
            for j in self.J:
                for f in sorted(d.glob(f"judge_{j}*.jsonl")):
                    for r in X.jl(f): self.J[j].setdefault(tuple(r["sig"]), r["label"])
    def answer(self, k):
        a = self.hl.answer(k); return a if a is not None else self.A.get(tuple(k))
    def label(self, q, gold, k):
        a = self.answer(k)
        if a is None: return None, None, None
        if any(x in a.strip().lower() for x in X.HELD): return "ABSTAIN", a, None
        s = (X.h16(q), X.h16(gold), X.h16(a)); l = []
        for j in ("llama8b", "qwen7b"):
            v = self.hl.j[j].get(s); v = v if v is not None else self.hl.GJ[j].get(s); l.append(v if v is not None else self.J[j].get(s))
        if None in l: return None, a, s
        return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), a, s


if MODE == "build":
    T = pickle.load(open(O / "work/traces.pkl", "rb"))
    ref = [q for q in T["B"] if goldof(q) and T["B"][q]["tier"] != "REAL_TIME"]
    U = sorted(q for q in ref if any(sig(T[p][q]) != sig(T["B"][q]) for p in POL[1:]))
    S = sorted(random.Random(20261008).sample(U, 1500))
    rows, roots = {}, set()
    for q in S:
        for p in POL:
            r = T[p][q]; k, ctx, gq = X.key_for({"ev": r["ev"], "gen_query": r["gen_query"]}, REC[q]["query"]); ev = [tuple(x) for x in r["ev"]]; blk = X.evblock(ev) if ev else None
            rows[(q, p)] = {"tier": r["tier"], "root": r.get("root"), "root_route": r.get("root_route"), "list_root": r.get("list_root"), "key": k, "ctx": ctx, "gq": gq, "blk": blk,
                            "blk_sha": hashlib.sha256(blk.encode()).hexdigest()[:16] if blk else None, "ctx_sha": hashlib.sha256(ctx.encode()).hexdigest()[:16] if ctx else None}
            if r["tier"] == "L1": roots.add(r["root"])
    pickle.dump({"U": U, "N_ref": len(ref), "S": S, "rows": rows, "roots": sorted(roots)}, open(O / "work/common.pkl", "wb"))
    print("U", len(U), "N_ref", len(ref), "n", len(S), "roots", len(roots)); sys.exit(0)

C = pickle.load(open(O / "work/common.pkl", "rb")); rows = C["rows"]
if MODE == "items":
    lab = Lab(); gen = {}
    for a in rows.values():
        if lab.answer(a["key"]) is None: gen.setdefault(tuple(a["key"]), {"key": a["key"], "question": a["gq"], "context": a["ctx"]})
    X.wjl(O / "gpu/in/gen_tasks.jsonl", list(gen.values())); print("gen", len(gen))
    if gen: sys.exit(0)
    J, miss = {}, 0
    def add(q, gold, k):
        global miss
        l, a, s = lab.label(q, gold, k)
        if a is None: miss += 1; return
        if l is None: J.setdefault(s, {"sig": list(s), "q": q, "gold": gold, "cand": a})
    for (q, p), a in rows.items():
        add(REC[q]["query"], goldof(q), a["key"])
        if a["tier"] == "L1" and goldof(a["root"]): add(REC[a["root"]]["query"], goldof(a["root"]), a["key"])   # root answer judged on the root's own question
    X.wjl(O / "gpu/in/judge2_items.jsonl", list(J.values()))
    SU = {}
    for (q, p), a in rows.items():
        if a["blk"]: SU.setdefault((q, a["blk_sha"]), {"query_id": q, "arm": a["blk_sha"], "query": REC[q]["query"], "evidence": a["blk"]})
        if a["ctx"]: SU.setdefault((q, "ctx_" + a["ctx_sha"]), {"query_id": q, "arm": "ctx_" + a["ctx_sha"], "query": REC[q]["query"], "evidence": a["ctx"]})
    done = set()
    for d in [X.RCD / "gpu/out"] + POOLDIRS:
        for j in ("llama8b", "qwen7b"):
            for f in sorted(d.glob(f"suffv_{j}*.jsonl")):
                for r in X.jl(f): done.add((j, r["query_id"], r["arm"]))
    todo = [t for k, t in SU.items() if not all((j, k[0], k[1]) in done for j in ("llama8b", "qwen7b"))]
    X.wjl(O / "gpu/in/suffv_tasks.jsonl", todo); print("judge2", len(J), "answer-missing", miss, "suffv", len(todo), "of", len(SU))
