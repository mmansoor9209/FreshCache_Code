"""Shared helpers for the standalone FreshCache operating points (Pareto; derived from scripts/baselines/matched_admission/matched_admission_helpers.py); all writes under outputs/standalone_operating_points.
Answer and judge outputs of the matched-admission scripts/baselines, GPTCache, L2 operating points, earlier runs (optional read-only pools) and the CategoryAware frontier
are reused as read-only pools; generation and judging are requested only for items missing from every pool.
Read-only on every pre-existing file. ca_replay is extracted read-only from baseline_selection_helpers via ast (importing baseline_selection_helpers would append to its run.log)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import ast, copy, hashlib, json, math, os, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "standalone_operating_points"); BLD = (OUTPUT_ROOT / "baselines/matched_admission"); PRIOR_TTTL_FCRULE_DIR = (OUTPUT_ROOT / "archived_runs/tieredttl_fcrule_l2sim080"); PRIOR_SCALM_TTTL_DIR = (OUTPUT_ROOT / "archived_runs/scalm_tieredttl_fcl1"); PRIOR_STTL080_DIR = (OUTPUT_ROOT / "archived_runs/semanticttl_080_k1"); PRIOR_STTL_FCL1_DIR = (OUTPUT_ROOT / "archived_runs/semanticttl_fcl1"); PRIOR_FC_L1COS040_DIR = (OUTPUT_ROOT / "archived_runs/freshcache_l1cos040_fixedttl"); CA_COMPOSITION_DIR = (OUTPUT_ROOT / "categoryaware_freshcache"); PRIOR_FC_STTL_DIR = (OUTPUT_ROOT / "archived_runs/freshcache_semanticttl"); CA_FRONTIER_DIR = (OUTPUT_ROOT / "categoryaware_frontier"); CRR = (OUTPUT_ROOT / "replay")
BASELINE_SEL_DIR, L2_POINTS_DIR, CALIBRATION_AUDIT_DIR, COST_DIR = (OUTPUT_ROOT / "baseline_selection"), (OUTPUT_ROOT / "l2_operating_points"), (OUTPUT_ROOT / "calibration_audit"), (OUTPUT_ROOT / "concurrency_cost")
STTL_FRONTIER_DIR = (OUTPUT_ROOT / "semanticttl_frontier"); OV = (OUTPUT_ROOT / "l2_verifier/out")
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); sys.path.insert(0, str(ROOT / "freshcache/replay_engine"))
import engine_loader as C, answer_pools as P   # noqa: E402
import numpy as np   # noqa: E402
jl, h16 = P.jl, P.h16
for d in ("work", "gpu/in", "gpu/out"): (O / d).mkdir(parents=True, exist_ok=True)
LOGF = open(O / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
sha_obj = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [None] * len(ps); m = len(ps); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
_E = None
def engine():
    global _E
    if _E is None: _E = C.load("reported")
    return _E
NS = {"np": np, "copy": copy, "engine": lambda: engine()}
for node in ast.parse(open((ROOT / "scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py")).read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "ca_replay": exec(compile(ast.Module([node], []), "baseline_selection_helpers.py", "exec"), NS)
ca_replay = NS["ca_replay"]
BASE_CA = json.load(open(BASELINE_SEL_DIR / "frozen_selection.json"))["CategoryAware_style"]["per_category"]
CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST"]
def ca_params(dtheta, m):
    th = {c: min(round(BASE_CA[c]["theta"] + dtheta, 6), 0.99) for c in CLASSES}; ka = {c: BASE_CA[c]["kappa"] * m for c in CLASSES}
    th["REAL_TIME"] = 1.1; ka["REAL_TIME"] = 0.0; return th, ka
def key_for(p, query):   # identical to baseline_heldout_replay.key_for
    E = engine(); ctx = E["p2"].ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", query); return ["gen", E["p2"].h(ctx), E["p2"].h(gq)], ctx, gq
_MAN = None
def evblock(ev):   # evidence-block construction of verifier_audit_keys.evblock, verbatim
    global _MAN
    E = engine(); p2 = E["p2"]
    import answer_grading as e2e
    if _MAN is None:
        _MAN = {}
        for m in E["exp"].load_jsonl(E["exp"].MANIFEST_FILE):
            if m.get("run_id") == "run_00" and m["url_hash"] not in _MAN: _MAN[m["url_hash"]] = {"title": m.get("title", "") or "", "snippet": m.get("snippet", "") or ""}
    lines = []
    for uh, tv in ev:
        mt = _MAN.get(uh, {}); b = re.sub(r"\s+", " ", (e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv)) or "")).strip()[:2000]
        blk = f"{len(lines)+1}. {mt.get('title','').strip()}"
        if mt.get("snippet"): blk += f"\n   {mt['snippet'].strip()}"
        if len(b) >= 200: blk += f"\n   PAGE TEXT: {b}"
        lines.append(blk)
    return "\n".join(lines) or "(no stored evidence)"
COST = json.load(open(COST_DIR / "component_costs.json"))["coefficients"]
LOOKUP = {"FreshCache": COST["C_local_lookup"]["per_method_ms_per_request_cold"]["FreshCache"], "CategoryAware": COST["C_local_lookup"]["per_method_ms_per_request_cold"]["CategoryAware-style"]}
def cost(search, fetch, gen, n, embeds, lookup_ms):
    k = 1000 / n; out = {}
    for sc in ("local", "live_p50"):
        tot = search * k * COST["c_search_ms"][sc] + fetch * k * COST["c_fetch_ms"][sc] + gen * k * COST["c_gen_ms"]["value"] + lookup_ms * 1000 + embeds * k * COST["c_embed_ms"]["value"]
        out[f"{sc}_ms_per_request"] = round(tot / 1000, 3)
    return out
PRIM = ("i don't know", "i dont know", "cannot determine", "not enough information", "insufficient", "no information", "unclear from the context")
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
class Val:   # validation labelling rule (as in quality_selection_and_8b_replication.py), with answer pools
    def __init__(self):
        QB = (OUTPUT_ROOT / "quality_selection")
        self.qa = {tuple(r["key"]): r["answer"] for r in jl(QB / "q_answers.jsonl")}
        self.qj = {x: {tuple(r["sig"]): r["label"] for r in jl(QB / f"q_judge_{x}.jsonl")} for x in ("llama8b", "qwen7b")}
        self.GA = P.AnswerLookup(extra=[CRR / "gpu/out/answers_llama3b.jsonl", CRR / "gpu/out3/answers_llama3b.jsonl", STTL_FRONTIER_DIR / "gpu/out/val_answers_llama3b.jsonl", BASELINE_SEL_DIR / "gpu/out/answers_llama3b.jsonl", CA_FRONTIER_DIR / "gpu/out/answers_llama3b.jsonl", CA_COMPOSITION_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_FC_L1COS040_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_STTL080_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_SCALM_TTTL_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_TTTL_FCRULE_DIR / "gpu/out/answers_llama3b.jsonl", BLD / "gpu/out/answers_llama3b.jsonl", ((OUTPUT_ROOT / "baselines/gptcache/gpu/out/answers_llama3b.jsonl")), L2_POINTS_DIR / "gpu/out/answers_llama3b.jsonl", O / "gpu/out/answers_llama3b.jsonl"])
        self.GJ = P.judge2(extra=[CRR / "gpu/out", CRR / "gpu/out3", STTL_FRONTIER_DIR / "gpu/out", BASELINE_SEL_DIR / "gpu/out", CA_FRONTIER_DIR / "gpu/out", CA_COMPOSITION_DIR / "gpu/out", PRIOR_FC_L1COS040_DIR / "gpu/out", PRIOR_STTL080_DIR / "gpu/out", PRIOR_SCALM_TTTL_DIR / "gpu/out", PRIOR_TTTL_FCRULE_DIR / "gpu/out", BLD / "gpu/out", ((OUTPUT_ROOT / "baselines/gptcache/gpu/out")), L2_POINTS_DIR / "gpu/out", O / "gpu/out"])
    def answer(self, key): a = self.qa.get(tuple(key)); return a if a is not None else self.GA.get(list(key))
    def labels(self, q, gold, key):
        a = self.answer(key)
        if a is None: return None, None, None
        if any(p in a.strip().lower() for p in PRIM): return "ABSTAIN", a, None
        s = (h16(q), h16(gold), h16(a)); l = [self.qj[x].get(s, self.GJ[x].get(s)) for x in ("llama8b", "qwen7b")]
        if None in l: return None, a, s
        return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), a, tuple(l)
class Held:  # held-out labelling rule (as in answer_quality_evaluation.py), with answer pools
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
        self.GA = P.AnswerLookup(extra=[CRR / "gpu/out/answers_llama3b.jsonl", STTL_FRONTIER_DIR / "gpu/out/ho_answers_llama3b.jsonl", BASELINE_SEL_DIR / "gpu/out/answers_llama3b.jsonl", CA_FRONTIER_DIR / "gpu/out/answers_llama3b.jsonl", CA_COMPOSITION_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_FC_L1COS040_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_STTL080_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_SCALM_TTTL_DIR / "gpu/out/answers_llama3b.jsonl", PRIOR_TTTL_FCRULE_DIR / "gpu/out/answers_llama3b.jsonl", BLD / "gpu/out/answers_llama3b.jsonl", ((OUTPUT_ROOT / "baselines/gptcache/gpu/out/answers_llama3b.jsonl")), L2_POINTS_DIR / "gpu/out/answers_llama3b.jsonl", O / "gpu/out/answers_llama3b.jsonl"])
        self.GJ = P.judge2(extra=[CRR / "gpu/out", STTL_FRONTIER_DIR / "gpu/out", BASELINE_SEL_DIR / "gpu/out", CA_FRONTIER_DIR / "gpu/out", CA_COMPOSITION_DIR / "gpu/out", PRIOR_FC_L1COS040_DIR / "gpu/out", PRIOR_STTL080_DIR / "gpu/out", PRIOR_SCALM_TTTL_DIR / "gpu/out", PRIOR_TTTL_FCRULE_DIR / "gpu/out", BLD / "gpu/out", ((OUTPUT_ROOT / "baselines/gptcache/gpu/out")), L2_POINTS_DIR / "gpu/out", O / "gpu/out"])
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
