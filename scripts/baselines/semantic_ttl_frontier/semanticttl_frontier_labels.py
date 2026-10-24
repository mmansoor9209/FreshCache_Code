"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

Answer labelers for the SemanticTTL frontier baseline: transcriptions of the frozen scoring rules.

Answers and judgments produced by this baseline (outputs/semanticttl_frontier/gpu/out) are used only for keys and
judgment signatures that no earlier output file contains.

  Val   validation split, same rule as scripts/evaluation/answer_audits/quality_selection.py. Answers come from
        outputs/quality_selection/q_answers.jsonl by exact key, then from the shared answer pools (plus
        outputs/replay/gpu/out, outputs/replay/gpu/out3 and this baseline's outputs); judgments come from
        q_judge_<judge>.jsonl, then the pools. ABSTAIN on the PRIM abstention patterns; CORRECT iff the llama8b and
        qwen7b judges both say CORRECT; otherwise WRONG; None if anything is missing.
  Held  held-out split, same rule as the Labeler in scripts/evaluation/answer_audits/answer_quality_evaluation.py. Answers
        come from outputs/general_audit/answers.jsonl and outputs/quality_selection/heldout_sensitivity/h_answers.jsonl
        by exact key, then outputs/predicted_class_audit/new_answers.jsonl by content hash, then the pools; judgments
        come from the general audit, the held-out sensitivity audit and the predicted-class audit, then the pools.
        ABSTAIN on the HELD abstention patterns; same two-judge rule."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_frontier"))
import semanticttl_frontier_helpers as X
P, h = X.P, X.h16
PRIM = ("i don't know", "i dont know", "cannot determine", "not enough information", "insufficient", "no information", "unclear from the context")
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
STTL_FRONTIER_GPU_OUT = X.STTL_FRONTIER_DIR / "gpu/out"

class Val:
    def __init__(self):
        QB = (OUTPUT_ROOT / "quality_selection")
        self.qa = {tuple(r["key"]): r["answer"] for r in X.jl(QB / "q_answers.jsonl")}
        self.qj = {x: {tuple(r["sig"]): r["label"] for r in X.jl(QB / f"q_judge_{x}.jsonl")} for x in ("llama8b", "qwen7b")}
        self.GA = P.AnswerLookup(extra=[X.CRR / "gpu/out/answers_llama3b.jsonl", X.CRR / "gpu/out3/answers_llama3b.jsonl", STTL_FRONTIER_GPU_OUT / "val_answers_llama3b.jsonl"])
        self.GJ = P.judge2(extra=[X.CRR / "gpu/out", X.CRR / "gpu/out3", STTL_FRONTIER_GPU_OUT])
    def answer(self, key):
        a = self.qa.get(tuple(key)); return a if a is not None else self.GA.get(list(key))
    def labels(self, q, gold, key):
        a = self.answer(key)
        if a is None: return None, None, None
        if any(p in a.strip().lower() for p in PRIM): return "ABSTAIN", a, None
        s = (h(q), h(gold), h(a)); l = [self.qj[x].get(s, self.GJ[x].get(s)) for x in ("llama8b", "qwen7b")]
        if None in l: return None, a, s
        return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), a, tuple(l)

class Held:
    def __init__(self):
        V = (OUTPUT_ROOT / "archived_runs"); A2, HS, PC = ((OUTPUT_ROOT / "general_audit")), ((OUTPUT_ROOT / "quality_selection/heldout_sensitivity")), ((OUTPUT_ROOT / "predicted_class_audit"))
        self.exact, self.content = {}, {}
        for f in (A2 / "answers.jsonl", HS / "h_answers.jsonl"):
            for r in X.jl(f): self.exact[tuple(r["key"])] = r["answer"]
        for r in X.jl(PC / "new_answers.jsonl"): self.content.setdefault((r["ctx_sha"], r["q_sha"]), r["answer"])
        for k, a in self.exact.items(): self.content.setdefault((k[1], k[2]), a)
        self.j = {"llama8b": {}, "qwen7b": {}}
        for j in self.j:
            for f in (A2 / f"judge_{j}.jsonl", HS / f"h_judge_{j}.jsonl", PC / f"new_judge_{j}.jsonl"):
                for r in X.jl(f):
                    if "sig" in r: self.j[j][tuple(r["sig"])] = r["label"]
        self.GA = P.AnswerLookup(extra=[X.CRR / "gpu/out/answers_llama3b.jsonl", STTL_FRONTIER_GPU_OUT / "ho_answers_llama3b.jsonl"])
        self.GJ = P.judge2(extra=[X.CRR / "gpu/out", STTL_FRONTIER_GPU_OUT])
    def answer(self, key):
        if tuple(key) in self.exact: return self.exact[tuple(key)]
        k = (key[1], key[2])
        if k in self.content: return self.content[k]
        return self.GA.get(list(key))
    def labels(self, q, gold, key):
        a = self.answer(key)
        if a is None: return None, None, None
        if any(p in a.strip().lower() for p in HELD): return "ABSTAIN", a, None
        s = (h(q), h(gold), h(a)); l = []
        for j in ("llama8b", "qwen7b"):
            v = self.j[j].get(s)
            if v is None: v = self.GJ[j].get(s)
            l.append(v)
        if None in l: return None, a, s
        return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), a, tuple(l)

def judge_items(labeler, items):
    """Collect the judging items that are still missing.

items is an iterable of (question, gold, key). Returns (judge_items, missing_answers): one two-judge item per judgment
signature that no judgment source covers (only for keys whose answer exists), and the keys that have no answer yet."""
    out, missing_answers = {}, []
    for q, gold, key in items:
        lab, a, s = labeler.labels(q, gold, key)
        if a is None: missing_answers.append(key); continue
        if lab is None and s is not None: out.setdefault(s, {"sig": list(s), "q": q, "gold": gold, "cand": a})
    return list(out.values()), missing_answers
