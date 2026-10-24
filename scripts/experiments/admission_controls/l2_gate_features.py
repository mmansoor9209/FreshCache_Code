"""Shared library for the learned L2 compatibility gate: engine copy that passes candidate metadata to the scorer, serving-time features and
evidence blocks, factored out of l2_gate_candidate_tracing.py unchanged so that training (stages 1/2) and serving (stage 3) compute identical features."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, hashlib, json, os, pathlib, re, sys, types; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "learned_l2_gate")); CRR = (OUTPUT_ROOT / "replay"); OV = (OUTPUT_ROOT / "l2_verifier/out")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
E = C.load("reported"); exp, p2 = E["exp"], E["p2"]
import numpy as np, answer_grading as e2e, page_version_replay as ma   # noqa: E402
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
    return features_core(q, sims, [page_text(uh, tv) for uh, tv in ev], len(ev), eq, sim, fc, age, efc)
def features_core(q, sims, texts, n_ev, eq, sim, fc, age, efc):
    full = " ".join(texts).lower(); ct = content(q); toks = set(TOK.findall(full))
    nums = set(re.findall(r"\d+", q)); caps = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z]+", q)[1:]}
    win = next((a for a in (3600 * 2 ** i for i in range(0, 14)) if exp.p_stale(efc, a, "url_list") > exp.EPS_URL_LIST), 3600 * 2 ** 14)
    t1, t2 = exp._eq_content_tokens(q), exp._eq_content_tokens(eq); a1, a2 = exp._eq_answer_type(q), exp._eq_answer_type(eq)
    f = {"qq_sim": sim, "qq_jaccard": len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0, "entity_match": float(exp._entity_match(q, eq)), "answer_type_match": float(a1 is None or a2 is None or a1 == a2),
         "strict_subset": float(t1 != t2 and (t1 <= t2 or t2 <= t1)), "support_max": max(sims) if sims else 0.0, "support_mean": float(np.mean(sims)) if sims else 0.0, "support_min": min(sims) if sims else 0.0,
         "n_pages": n_ev, "n_pages_text": sum(1 for t in texts if len(t) >= 200), "evidence_chars": sum(len(t) for t in texts),
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
