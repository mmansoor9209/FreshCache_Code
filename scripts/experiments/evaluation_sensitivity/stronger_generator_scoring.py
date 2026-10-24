"""Scoring for the Qwen2.5-14B generator rerun (answers from stronger_generator_answers.py).
Modes: items -> merge shards into gpu/out/answers14b.jsonl and write gpu/in/judge2_items.jsonl (frozen two-judge rule; frozen held-out
abstention patterns first, so abstentions are not judged); score -> outputs/stronger_generator_answers.json.
Audit set: fresh pipeline, A, B, D on the held-out audit (gold = sample.json gold), WAI = fresh-correct (same generator) turned wrong.
Live set: own search at matched depth (S, path A_matched) and FreshCache service (F, path C_freshcache) for the candidates (reference =
frozen gold); the four composed policies are built exactly as in live_admission_policies.py (accept -> F, reject -> S).
Valid-reference subsets: held-out majority-valid / unanimous-valid (model adjudication, frozen labels); live majority-valid / unanimous-valid (reference validation).
The 3B labels on the same subsets are recomputed with the frozen pools for comparison. Note: Qwen2.5-14B shares a family with the Qwen2.5-7B judge."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import ast, collections, csv, hashlib, json, math, pickle, sys, pathlib
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "composition_checks"); RCD = (OUTPUT_ROOT / "composition")
T2 = (OUTPUT_ROOT / "unseen_queries"); TABLE_ROWS = (OUTPUT_ROOT / "model_adjudication")
MODE = sys.argv[1]
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
SAMPLE = {d["query_id"]: d for d in json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]}
CASES = {c["case_id"]: c for c in jl(T2 / "work/cases_built.jsonl")}
def ctx_of(r): return (SAMPLE[r["query_id"]]["query"], SAMPLE[r["query_id"]]["gold"]) if r["set"] == "audit" else (CASES[r["query_id"]]["query"], CASES[r["query_id"]]["reference"])
if MODE == "items":
    rows = {}
    for f in sorted((O / "gpu/out").glob("answers14b_shard*.jsonl")):
        for r in jl(f): rows[r["tid"]] = r
    assert len(rows) == len(jl(O / "gpu/in/gen14b_tasks.jsonl")), len(rows)
    with open(O / "gpu/out/answers14b.jsonl", "w", encoding="utf-8") as fh:
        for t in sorted(rows): fh.write(json.dumps(rows[t], ensure_ascii=False) + "\n")
    items = {}
    for r in rows.values():
        q, g = ctx_of(r)
        if not g or any(x in r["answer"].strip().lower() for x in HELD): continue
        s = [h16(q), h16(g), h16(r["answer"])]; items.setdefault(tuple(s), {"sig": s, "q": q, "gold": g, "cand": r["answer"]})
    with open(O / "gpu/in/judge2_items.jsonl", "w", encoding="utf-8") as fh:
        for x in items.values(): fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("answers", len(rows), "judge2 items", len(items)); sys.exit(0)

# ---------------- score
J = {}
for j in ("llama8b", "qwen7b"):
    J[j] = {}
    for f in [O / f"gpu/out/judge_{j}.jsonl"] + sorted((O / "gpu/out").glob(f"judge_{j}_shard*.jsonl")):
        for r in jl(f): J[j][tuple(r["sig"])] = r["label"]
ANS = {(r["set"], r["query_id"], r["arm"]): r["answer"] for r in jl(O / "gpu/out/answers14b.jsonl")}
def lab14(st, qid, arm):
    a = ANS[(st, qid, arm)]; q, g = ctx_of({"set": st, "query_id": qid})
    if not g: return "NO_REFERENCE"
    if any(x in a.strip().lower() for x in HELD): return "ABSTAIN"
    l = [J[j].get((h16(q), h16(g), h16(a))) for j in J]
    if None in l: return "UNRESOLVED"
    return "CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"
def mcn(b, c): n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [round(100 * (c - m) / d, 1), round(100 * (c + m) / d, 1)]
# held-out subsets (frozen adjudication reference labels, read-only exec of the analysis prefix)
src = open((ROOT / "scripts/evaluation/reference_validation/model_adjudication_analysis.py")).read(); g = {"__name__": "model_adjudication_analysis_copy", "__file__": str((ROOT / "scripts/evaluation/reference_validation/model_adjudication_analysis.py"))}; exec(compile(src[:src.index("# ---------------- answer stage")], "model_adjudication_analysis_copy", "exec"), g)
HSUB = {"all": sorted(SAMPLE), "majority_valid": g["SUB"]["majority_valid"], "unanimous_valid": g["SUB"]["unanimous_valid"]}
# 3B labels on the audit (frozen pools)
sys.path.insert(0, str(ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(ROOT / "freshcache/composition")); import output_workspace; output_workspace.use("composition_checks"); import composition_study_helpers as X   # noqa: E402
hl = X.Held(); ST = pickle.load(open(RCD / "work/composition_ablation_replay.pkl", "rb"))["policies"]
FRESH3 = {r["request_id"]: r["fresh_label"] for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv")) if r["point"] == "theta_0.75"}
def lab3(qid, p):
    k = ST[p]["audit"][qid]["key"]; k = ast.literal_eval(k) if isinstance(k, str) else k; d = SAMPLE[qid]
    l = hl.labels(d["query"], d["gold"], k)[0]; return l or "UNRESOLVED"
L3 = {(q, p): lab3(q, p) for q in SAMPLE for p in ("A", "B", "D")}
for q in SAMPLE: L3[(q, "fresh")] = FRESH3[q]
L14 = {(q, p): lab14("audit", q, p) for q in SAMPLE for p in ("fresh", "A", "B", "D")}
def audit_table(L, ids):
    out = {"n": len(ids)}
    for p in ("fresh", "A", "B", "D"):
        c = collections.Counter(L[(q, p)] for q in ids)
        out[p] = {"correct": c["CORRECT"], "accuracy_pct": round(100 * c["CORRECT"] / len(ids), 2), "accuracy_ci": wilson(c["CORRECT"], len(ids)), "abstain": c["ABSTAIN"], "unresolved": c["UNRESOLVED"]}
        if p != "fresh": out[p]["WAI"] = sum(1 for q in ids if L[(q, "fresh")] == "CORRECT" and L[(q, p)] == "WRONG")
    out["fresh_correct"] = out["fresh"]["correct"]
    for a, b in (("B", "D"), ("A", "D"), ("A", "B")):
        lost = sum(1 for q in ids if L[(q, a)] == "CORRECT" and L[(q, b)] != "CORRECT"); gained = sum(1 for q in ids if L[(q, a)] != "CORRECT" and L[(q, b)] == "CORRECT")
        wl = sum(1 for q in ids if L[(q, "fresh")] == "CORRECT" and L[(q, a)] != "WRONG" and L[(q, b)] == "WRONG"); wg = sum(1 for q in ids if L[(q, "fresh")] == "CORRECT" and L[(q, a)] == "WRONG" and L[(q, b)] != "WRONG")
        out[f"{a}->{b}"] = {"correct_lost": lost, "correct_gained": gained, "p": round(mcn(lost, gained), 4), "WAI_discordant_new_old": [wl, wg], "WAI_p": round(mcn(wl, wg), 4)}
    return out
R = {"generator": "Qwen/Qwen2.5-14B-Instruct", "note": "same family as the Qwen2.5-7B judge", "audit": {}, "live": {}}
for nm, ids in HSUB.items(): R["audit"][nm] = {"qwen14b": audit_table(L14, ids), "llama3b": audit_table(L3, ids)}
# live policies
src = open((ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py")).read(); g2 = {"__name__": "t2s", "__file__": str((ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py"))}; argv = sys.argv; sys.argv = ["unseen_query_scoring.py", "score"]
exec(compile(src[:src.index("PAIR = [")], "t2s", "exec"), g2); sys.argv = argv; label3 = g2["label"]
GS = {r["case_id"]: r for r in json.load(open(RCD / "outputs/gate_scores.json"))["cases"]}
RJ = {j: {r["query_id"]: r["label"] for r in jl(T2 / f"refval/outputs/raw/reference_{j}.jsonl")} for j in ("llama8b", "qwen7b", "mistral7b")}
def maj(q):
    v = collections.Counter(RJ[j].get(q) for j in RJ).most_common(1)[0]; return v[0] if v[1] >= 2 else "NO_MAJORITY"
PAIR = [c for c in CASES.values() if c["both_evaluable"] and c["reference_available"]]
MV = sorted(c["case_id"] for c in PAIR if maj(c["case_id"]) == "VALID_AT_T"); UN = sorted(q for q in MV if len({RJ[j].get(q) for j in RJ}) == 1)
POL = {"Always search": lambda c: False, "Primary L2 (0.75)": lambda c: True, "Stricter L2 (0.80)": lambda c: GS[c["case_id"]]["similarity"] >= 0.80, "Learned gate (tau)": lambda c: GS[c["case_id"]]["gate_accept"]}
REF = sorted(q for q, c in CASES.items() if c["reference"])
def live_table(lab, ids):
    out = {"n": len(ids)}; base = {q: lab(q, "A_matched") for q in ids}
    for nm, acc in POL.items():
        L = {q: lab(q, "C_freshcache" if acc(CASES[q]) else "A_matched") for q in ids}; c = collections.Counter(L.values())
        lost = sum(1 for q in ids if base[q] == "CORRECT" and L[q] != "CORRECT"); gained = sum(1 for q in ids if base[q] != "CORRECT" and L[q] == "CORRECT")
        out[nm] = {"correct": c["CORRECT"], "accuracy_pct": round(100 * c["CORRECT"] / len(ids), 2), "abstain": c["ABSTAIN"], "unresolved": c["UNRESOLVED"], "lost_gained_vs_always": [lost, gained], "p": round(mcn(lost, gained), 4)}
    return out
l14 = lambda q, p: lab14("live", q, p); l3 = lambda q, p: label3(CASES[q], p)
for nm, ids in (("all_with_reference", REF), ("majority_valid", MV), ("unanimous_valid", UN)):
    R["live"][nm] = {"qwen14b": live_table(l14, ids), "llama3b": live_table(l3, ids)}
json.dump(R, open(O / "outputs/stronger_generator_answers.json", "w"), indent=1)
for k in ("audit", "live"):
    for nm, v in R[k].items():
        for gname, t in v.items():
            if k == "audit": print(k, nm, gname, "n", t["n"], {p: (t[p]["correct"], t[p].get("WAI"), t[p]["abstain"], t[p]["unresolved"]) for p in ("fresh", "A", "B", "D")}, "B->D", t["B->D"])
            else: print(k, nm, gname, "n", t["n"], {p: (t[p]["correct"], t[p]["abstain"], t[p]["lost_gained_vs_always"], t[p]["p"]) for p in POL})
