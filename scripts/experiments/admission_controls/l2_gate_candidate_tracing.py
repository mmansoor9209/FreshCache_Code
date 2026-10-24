"""L2 admission controls, stage 1 (validation only): trace every L2 candidate FreshCache would admit and build sufficiency label tasks.
Runs on the validation stream, computes SERVING-TIME features of (incoming query, cached URL list + the page versions L3 would
serve), and builds two-judge sufficiency label tasks. Engine: in-memory copy of freshcache/replay_engine/verifier_replay.py (FreshCache-L2Verify; file untouched)
whose verification call is extended to pass the candidate's metadata; the scorer used here accepts every candidate, so the replay must reproduce
FreshCache primary exactly (gate). Features use only information available when the request arrives: the incoming query, the cached entry's query,
similarity, age, class, and the cached page texts / precomputed BGE-M3 page embeddings at the versions L3 holds. No held-out data is read."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, datetime, hashlib, json, os, pathlib, re, sys, types; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "learned_l2_gate")); CRR = (OUTPUT_ROOT / "replay"); OV = (OUTPUT_ROOT / "l2_verifier/out")
for d in ("work", "gpu/in", "gpu/out"): (O / d).mkdir(parents=True, exist_ok=True)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
PRE = {"written_utc": now(), "status": "L2 compatibility/sufficiency gate; validation-only training and threshold selection; held-out evaluated once",
       "question": "does this cached source set contain sufficient evidence for the incoming query (instead of query-query cosine alone)?",
       "features": ["qq_sim", "qq_jaccard", "entity_match", "answer_type_match", "strict_subset", "support_max", "support_mean", "support_min", "n_pages", "n_pages_text", "evidence_chars",
                    "qtok_coverage_all", "qtok_coverage_max_page", "num_coverage", "cap_coverage", "entry_age_frac_of_window", "class_onehot"],
       "labels": "frozen SUFFV two-judge rule (Llama-3.1-8B + Qwen2.5-7B agree) on the evidence block FreshCache would serve (verifier_audit_keys.evblock format); disagreements excluded from training",
       "models": "L2-regularised logistic regression and sklearn HistGradientBoosting; choose by cluster-grouped 5-fold out-of-fold ROC-AUC on validation",
       "threshold_rule": "Youden J on validation out-of-fold scores (same rule that selected the frozen gamma = 0.55 verifier), so learned gate vs single-feature verifier are compared under one rule",
       "deployment": "identical mechanics to L2Verify: the top admitted L2 candidate is scored before reuse; score < tau -> fresh search (the rejected entry stays cached)",
       "heldout": "after freezing model + tau: held-out replay (operations, cost-model cost), audit (accuracy, WAI, C-WAI, evidence sufficiency), unseen-query live cases (accepted vs rejected sufficiency)",
       "comparators": ["FreshCache primary theta_L2 0.75", "FreshCache theta_L2 0.80", "L2Verify gamma 0.55 (single-feature support)"], "seed": 42}
PRE["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in PRE.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (O / "prereg.json").exists(): json.dump(PRE, open(O / "prereg.json", "w"), indent=1)
PRE = json.load(open(O / "prereg.json")); print(f"[{now()}] prereg {PRE['sha256'][:16]}", flush=True)
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
E = C.load("reported"); exp, p2 = E["exp"], E["p2"]
import numpy as np, answer_grading as e2e, page_version_replay as ma   # noqa: E402
V = C.streams(E)["validation"]; assert DS.check("validation_requests", len(V))
src = open((ROOT / "freshcache/replay_engine/verifier_replay.py"), encoding="utf-8").read()
old = "            vscore = support.score(q, tent_ev)"; assert src.count(old) == 1
src = src.replace(old, "            vscore = support.score(q, tent_ev, l2_q[hit2], sim2, fc, t - l2_t[hit2], l2_fc[hit2], qid, list(l2_urls[hit2]))")
S2 = types.ModuleType("verifier_replay_copy"); S2.__file__ = str((ROOT / "freshcache/replay_engine/verifier_replay.py")); p0 = list(sys.path)
exec(compile(src, "<copy of verifier_replay.py: candidate metadata passed to the scorer>", "exec"), S2.__dict__); sys.path[:] = p0
assert S2.exp is exp
QZ = np.load(OV / "emb_queries.npz", allow_pickle=True); QI = {k: i for i, k in enumerate(QZ["keys"])}; QE = QZ["emb"]
PZ = np.load(OV / "emb_pages.npz", allow_pickle=True); PI = {k: i for i, k in enumerate(PZ["keys"])}; PE = PZ["emb"]
man = {}
for l in open((DATA_ROOT / "url_manifest.jsonl"), encoding="utf-8"):
    x = json.loads(l)
    if x["run_id"] == "run_00": man.setdefault(x["url_hash"], {"title": x.get("title") or "", "snippet": x.get("snippet") or ""})
TOK = re.compile(r"[a-z0-9]+"); STOP = set("the a an of in on for to and or is was were are be by with at from as what which who whom whose when where why how did does do has have had that this these those it its most recent latest current".split())
def content(s): return {w for w in TOK.findall((s or "").lower()) if w not in STOP and len(w) > 1}
def page_text(uh, tv): return re.sub(r"\s+", " ", e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv)) or "").strip()[:2000]
def features(q, ev, eq, sim, fc, age, efc):
    qv = QE[QI[q]]; sims = [float(qv @ PE[PI[f"{uh}|{ma.version_at(tv)}"]]) for uh, tv in ev if f"{uh}|{ma.version_at(tv)}" in PI]
    texts = [page_text(uh, tv) for uh, tv in ev]; full = " ".join(texts).lower(); ct = content(q); toks = set(TOK.findall(full))
    nums = set(re.findall(r"\d+", q)); caps = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z]+", q)[1:]}
    win = next((a for a in (3600 * 2 ** i for i in range(0, 14)) if exp.p_stale(efc, a, "url_list") > exp.EPS_URL_LIST), 3600 * 2 ** 14)
    t1, t2 = exp._eq_content_tokens(q), exp._eq_content_tokens(eq); a1, a2 = exp._eq_answer_type(q), exp._eq_answer_type(eq)
    f = {"qq_sim": sim, "qq_jaccard": len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0, "entity_match": float(exp._entity_match(q, eq)), "answer_type_match": float(a1 is None or a2 is None or a1 == a2),
         "strict_subset": float(t1 != t2 and (t1 <= t2 or t2 <= t1)), "support_max": max(sims) if sims else 0.0, "support_mean": float(np.mean(sims)) if sims else 0.0, "support_min": min(sims) if sims else 0.0,
         "n_pages": len(ev), "n_pages_text": sum(1 for t in texts if len(t) >= 200), "evidence_chars": sum(len(t) for t in texts),
         "qtok_coverage_all": len(ct & toks) / len(ct) if ct else 0.0, "qtok_coverage_max_page": max((len(ct & set(TOK.findall(t.lower()))) / len(ct) for t in texts), default=0.0) if ct else 0.0,
         "num_coverage": len(nums & toks) / len(nums) if nums else 1.0, "cap_coverage": len(caps & toks) / len(caps) if caps else 1.0, "entry_age_frac_of_window": min(age / win, 1.0)}
    for c in ("TIMELESS", "SLOW", "MEDIUM", "FAST"): f[f"class_{c}"] = float(fc == c)
    return f
def evblock(ev):   # evidence-block construction of verifier_audit_keys.evblock (as in operating_point_helpers)
    lines = []
    for uh, tv in ev:
        mt = man.get(uh, {}); b = page_text(uh, tv)
        blk = f"{len(lines) + 1}. {mt.get('title', '').strip()}"
        if mt.get("snippet"): blk += f"\n   {mt['snippet'].strip()}"
        if len(b) >= 200: blk += f"\n   PAGE TEXT: {b}"
        lines.append(blk)
    return "\n".join(lines) or "(no stored evidence)"
class Recorder:
    def __init__(self): self.rows = []
    def score(self, q, tent_ev, eq, sim, fc, age, efc, qid, urls):
        ev = [tuple(x) for x in tent_ev]; self.rows.append({"query_id": qid, "query": q, "entry_query": eq, "fc": fc, "entry_fc": efc, "ev": ev, "urls": urls, **features(q, ev, eq, sim, fc, age, efc)}); return 1.0
rec = Recorder(); m, rows, _ = S2.replay(V, gamma=0.0, support=rec)
fm, _, _ = C.metrics(E, V, "FreshCache", E["support_of"](V, E["rounds"]))
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); REF.record("reported_engine/validation/FreshCache/search_saved_pct", fm["search_saved_pct"]); G = {"accept-all replay reproduces FreshCache primary validation savings and L2 hits": m["search_saved_pct"] == fm["search_saved_pct"] and m["l2_hits"] == fm["l2_hits"],
     "one scored candidate per L2 hit": len(rec.rows) == m["l2_hits"]}
print(json.dumps(G), flush=True); assert all(G.values())
CL = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in E["recs"]}; SUF = {}
for r in rec.rows:
    r["cluster_id"] = CL[r["query_id"]]; eb = evblock(r["ev"]); r["evidence_sha"] = hashlib.sha256(eb.encode()).hexdigest()
    SUF.setdefault((r["query_id"], r["evidence_sha"]), {"query_id": r["query_id"], "arm": r["evidence_sha"][:16], "query": r["query"], "evidence": eb})
with open(O / "work/val_candidates.jsonl", "w", encoding="utf-8") as f:
    for r in rec.rows: f.write(json.dumps({k: v for k, v in r.items() if k != "ev"} | {"ev": [list(x) for x in r["ev"]]}, ensure_ascii=False) + "\n")
with open(O / "gpu/in/suffv_tasks.jsonl", "w", encoding="utf-8") as f:
    for t in SUF.values(): f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"[{now()}] validation L2 candidates {len(rec.rows)}; distinct label tasks {len(SUF)}; classes {dict(collections.Counter(r['fc'] for r in rec.rows))}", flush=True)
