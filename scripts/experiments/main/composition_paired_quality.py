"""Paired quality analysis of the four-way composition (held-out audit, frozen two-judge rule; sufficiency = frozen SUFFV two-judge rule).
Populations: full audit; baseline answer-cache misses; requests whose route, evidence, or output differs between the pair; baseline-correct;
baseline-wrong/abstaining. Transitions per pair (baseline -> composition). Uncertainty: exact McNemar, Wald CI on discordant pairs, and a
cluster bootstrap (10,000 resamples of query clusters) for paired differences; Wilson/rule-of-three upper bounds when no losses are observed.
Also counts route/evidence changes over the full held-out stream. Writes outputs/composition_paired_quality.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, math, pickle, random, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import composition_helpers as X
_argv = sys.argv; sys.argv = ["composition_ablation_replay.py", "noop"]; from composition_ablation_replay import suff_pool; sys.argv = _argv
O = X.O; ST = pickle.load(open(O / "work/composition_ablation_replay.pkl", "rb")); P = ST["policies"]; hl = X.Held(); SL = suff_pool()
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
import csv
FRESH = {r["request_id"]: r["fresh_label"] for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv")) if r["point"] == "theta_0.75"}
FCN = sum(v == "CORRECT" for v in FRESH.values()); assert REF.check("heldout_audit/fresh_correct", FCN)
REC = {r["query_id"]: r for r in X.engine()["recs"]}; CLU = {q: REC[q].get("cluster_id") or q for q in IDS}
L = {}
for p in P:
    for q in IDS:
        a = P[p]["audit"][q]; lab, ans, _ = hl.labels(SQ[q]["query"], SQ[q]["gold"], a["key"])
        s = SL.get((q, a["evidence_sha"])) if a["evidence_sha"] else None
        suf = None if s is None else (s[0] if s[0] == s[1] else "JUDGE_DISAGREEMENT")
        L[(p, q)] = {"label": lab or "UNRESOLVED", "answer": ans, "suff": suf or ("NO_EVIDENCE" if not a["evidence_sha"] else "UNRESOLVED"), "tier": a["tier"], "ev": a["evidence_sha"], "key": tuple(a["key"])}
def mcn(b, c): n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def wald(b, c, n):
    if not n: return None
    d = (c - b) / n; se = math.sqrt(max((b + c) - (c - b) ** 2 / n, 0)) / n; return [round(100 * d, 2), round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)]
def boot(ids, f, B=10000, seed=7):
    cl = collections.defaultdict(list)
    for q in ids: cl[CLU[q]].append(q)
    keys = list(cl); rng = random.Random(seed); vals = []
    for _ in range(B):
        samp = [q for _ in keys for q in cl[keys[rng.randrange(len(keys))]]]; vals.append(f(samp))
    vals.sort(); return [round(100 * vals[int(.025 * B)], 2), round(100 * vals[int(.975 * B) - 1], 2)]
def upper95(k, n):   # exact one-sided 95% upper bound for a binomial proportion (Clopper-Pearson)
    if n == 0: return None
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2; cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > 0.05 else (lo, mid)
    return round(100 * hi, 2)
def metrics(p):
    c = collections.Counter(L[(p, q)]["label"] for q in IDS); s = collections.Counter(L[(p, q)]["suff"] for q in IDS)
    w = sum(1 for q in IDS if FRESH[q] == "CORRECT" and L[(p, q)]["label"] == "WRONG")
    return {"correct": c["CORRECT"], "accuracy_pct": round(100 * c["CORRECT"] / len(IDS), 2), "wrong": c["WRONG"], "abstain": c["ABSTAIN"], "unresolved": c["UNRESOLVED"], "WAI": w, "C_WAI_pct": round(100 * w / FCN, 2),
            "sufficient": s["SUFFICIENT"], "insufficient": s["INSUFFICIENT"], "suff_disagreement": s["JUDGE_DISAGREEMENT"], "suff_unresolved": s["UNRESOLVED"] + s["NO_EVIDENCE"]}
def changed(a, b, q): x, y = L[(a, q)], L[(b, q)]; return x["tier"] != y["tier"] or x["ev"] != y["ev"] or x["answer"] != y["answer"]
def transitions(a, b, ids):   # a = baseline, b = composition
    T = collections.Counter(); ok = [q for q in ids if L[(a, q)]["label"] != "UNRESOLVED" and L[(b, q)]["label"] != "UNRESOLVED"]
    for q in ok:
        x, y = L[(a, q)]["label"], L[(b, q)]["label"]
        if x == "CORRECT" and y == "WRONG": T["CORRECT->WRONG"] += 1
        if x == "CORRECT" and y == "ABSTAIN": T["CORRECT->ABSTAIN"] += 1
        if x in ("WRONG", "ABSTAIN") and y == "CORRECT": T["WRONG/ABSTAIN->CORRECT"] += 1
        if x == "WRONG" and y == "ABSTAIN": T["WRONG->ABSTAIN"] += 1
        if x == "ABSTAIN" and y == "WRONG": T["ABSTAIN->WRONG"] += 1
    sk = [q for q in ids if L[(a, q)]["suff"] in ("SUFFICIENT", "INSUFFICIENT", "JUDGE_DISAGREEMENT") and L[(b, q)]["suff"] in ("SUFFICIENT", "INSUFFICIENT", "JUDGE_DISAGREEMENT")]
    sl = sum(1 for q in sk if L[(a, q)]["suff"] == "SUFFICIENT" and L[(b, q)]["suff"] != "SUFFICIENT"); sg = sum(1 for q in sk if L[(a, q)]["suff"] != "SUFFICIENT" and L[(b, q)]["suff"] == "SUFFICIENT")
    loss = T["CORRECT->WRONG"] + T["CORRECT->ABSTAIN"]; gain = T["WRONG/ABSTAIN->CORRECT"]
    acc = lambda s: (sum(L[(b, q)]["label"] == "CORRECT" for q in s) - sum(L[(a, q)]["label"] == "CORRECT" for q in s)) / max(len(s), 1)
    suf = lambda s: (sum(L[(b, q)]["suff"] == "SUFFICIENT" for q in s) - sum(L[(a, q)]["suff"] == "SUFFICIENT" for q in s)) / max(len(s), 1)
    return {"N_label_resolved": len(ok), "N_label_unresolved": len(ids) - len(ok), **dict(T), "correct_losses_total": loss, "correct_gains": gain, "correctness_mcnemar_p": mcn(loss, gain),
            "accuracy_diff_pp_wald": wald(loss, gain, len(ok)), "accuracy_diff_pp_cluster_boot": boot(ok, acc) if ok else None,
            "N_suff_evaluable": len(sk), "SUFF->INSUFF": sl, "INSUFF->SUFF": sg, "suff_mcnemar_p": mcn(sl, sg), "suff_diff_pp_wald": wald(sl, sg, len(sk)), "suff_diff_pp_cluster_boot": boot(sk, suf) if sk else None,
            "upper95_new_error_rate_pct": upper95(T["CORRECT->WRONG"] + T["CORRECT->ABSTAIN"], sum(1 for q in ok if L[(a, q)]["label"] == "CORRECT"))}
PAIRS = [("A", "D"), ("B", "D"), ("A", "B"), ("C", "D")]
R = {"metrics": {p: metrics(p) for p in P}, "pairs": {}}
for a, b in PAIRS:
    miss = [q for q in IDS if L[(a, q)]["tier"] not in ("L1", "REAL_TIME")]; aff = [q for q in IDS if changed(a, b, q)]
    bc = [q for q in IDS if L[(a, q)]["label"] == "CORRECT"]; bw = [q for q in IDS if L[(a, q)]["label"] in ("WRONG", "ABSTAIN")]
    wai = lambda p, s: sum(1 for q in s if FRESH[q] == "CORRECT" and L[(p, q)]["label"] == "WRONG")
    full_changed = sum(1 for q in P[a]["sig"] if P[a]["sig"][q] != P[b]["sig"][q])
    R["pairs"][f"{a}->{b}"] = {"populations": {"full_audit": len(IDS), "baseline_answer_cache_misses": len(miss), "affected_route_evidence_or_output": len(aff),
                                                 "affected_output_only": sum(1 for q in IDS if L[(a, q)]["answer"] != L[(b, q)]["answer"]), "affected_among_answer_cache_hits": sum(1 for q in aff if L[(a, q)]["tier"] == "L1"),
                                                 "baseline_correct": len(bc), "baseline_wrong_or_abstain": len(bw), "full_heldout_stream_requests_changed": full_changed, "full_heldout_stream": ST["n"]},
                               "full_audit": transitions(a, b, IDS), "answer_cache_misses": transitions(a, b, miss), "affected": transitions(a, b, aff),
                               "baseline_correct": transitions(a, b, bc), "baseline_wrong_or_abstain": transitions(a, b, bw),
                               "WAI_vs_fresh": {"baseline": wai(a, IDS), "composition": wai(b, IDS), "baseline_only": sum(1 for q in IDS if FRESH[q] == "CORRECT" and L[(a, q)]["label"] == "WRONG" and L[(b, q)]["label"] != "WRONG"),
                                                "composition_only": sum(1 for q in IDS if FRESH[q] == "CORRECT" and L[(b, q)]["label"] == "WRONG" and L[(a, q)]["label"] != "WRONG")}}
json.dump(R, open(O / "outputs/composition_paired_quality.json", "w"), indent=1)
print(json.dumps(R["metrics"], indent=0)); 
for k, v in R["pairs"].items(): print(k, json.dumps(v["populations"]), "\n  full:", json.dumps(v["full_audit"]), "\n  affected:", json.dumps(v["affected"]), "\n  WAI:", v["WAI_vs_fresh"])
