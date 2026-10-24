r"""Held-out scripts/evaluation of D+gamma, D+gate and D+AV (answer verifier). Frame U' = U (the common-sample frame) plus the held-out requests with a reference
(not REAL_TIME) that only the new arms change relative to B. Sample: the common 1,500-request sample from U plus a simple random sample of U' \ U at the
same sampling fraction (seed 20261016), so the sample is self-weighting over U'. Every arm and B, D are labelled on every sampled request
(frozen 3B generator, frozen two-judge rule). Modes: build | items | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, math, pickle, random, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "natural_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); MODE = sys.argv[1]
import output_workspace; output_workspace.use("natural_workload"); import composition_study_helpers as X, live_stream_labeller as CL   # noqa: E402
E = X.engine(); p2 = E["p2"]; REC = {r["query_id"]: r for r in E["recs"]}; GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid):
    qt = REC[qid]["query"]; g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
TRACES = pickle.load(open(CS_DIR / "work/traces.pkl", "rb")); TC = pickle.load(open(O / "work/list_control_traces.pkl", "rb")); TA = pickle.load(open(O / "work/av_traces.pkl", "rb"))
T = {"B": TRACES["B"], "D": TRACES["D"], "Dgamma": TC["Dgamma"], "Dgate": TC["Dgate"], "DAV": TA["DAV"]}; ARMS = ("D", "Dgamma", "Dgate", "DAV")
C = pickle.load(open(CS_DIR / "work/common.pkl", "rb")); U = set(C["U"]); Nref = C["N_ref"]
if MODE == "build":
    ref = [q for q in T["B"] if goldof(q) and T["B"][q]["tier"] != "REAL_TIME"]
    extra = sorted(q for q in ref if q not in U and any(sig(T[a][q]) != sig(T["B"][q]) for a in ARMS[1:]))
    frac = len(C["S"]) / len(U); k = round(frac * len(extra)); S2 = sorted(random.Random(20261016).sample(extra, k))
    rows = {}
    for q in list(C["S"]) + S2:
        for p in ("B",) + ARMS:
            r = T[p][q]; key, ctx, gq = X.key_for({"ev": r["ev"], "gen_query": r["gen_query"]}, REC[q]["query"]); rows[(q, p)] = {"key": key, "ctx": ctx, "gq": gq, "tier": r["tier"], "root": r.get("root"), "root_route": r.get("root_route")}
    pickle.dump({"S": list(C["S"]) + S2, "S_extra": S2, "U_prime": len(U) + len(extra), "extra_frame": len(extra), "fraction": frac, "rows": rows}, open(O / "work/admission_control_evaluation.pkl", "wb"))
    print("U'", len(U) + len(extra), "extra frame", len(extra), "extra sampled", len(S2)); sys.exit(0)
W = pickle.load(open(O / "work/admission_control_evaluation.pkl", "rb")); rows, S = W["rows"], W["S"]; lab = CL.Lab()
import answer_verifier as AVM  # noqa: E402  (Pool only)
if MODE == "items":
    pool = AVM.Pool(); G, J = {}, {}
    for (q, p), a in rows.items():
        ans = pool.answer(tuple(a["key"]))
        if ans is None: G.setdefault(tuple(a["key"]), {"key": a["key"], "question": a["gq"], "context": a["ctx"]}); continue
        l, _, s = lab.label_answer(REC[q]["query"], goldof(q), ans)
        if l is None: J.setdefault(s, {"sig": list(s), "q": REC[q]["query"], "gold": goldof(q), "cand": ans})
    X.wjl(O / "gpu/in/gen_tasks.jsonl", list(G.values())); X.wjl(O / "gpu/in/judge2_items.jsonl", list(J.values())); print("gen", len(G), "judge2", len(J)); sys.exit(0)
if MODE == "score":
    pool = AVM.Pool(); L = {(q, p): lab.label_answer(REC[q]["query"], goldof(q), pool.answer(tuple(a["key"])))[0] for (q, p), a in rows.items()}
    n, Up = len(S), W["U_prime"]; f = 100 * Up / Nref; RP = json.load(open(O / "outputs/list_controls_replay.json"))["policies"]; RA = json.load(open(O / "outputs/av_replay.json"))
    CHECK = 30.0; nh = json.load(open(OUTPUT_ROOT / "validation_split/split.json"))["test"]["requests"]   # held-out requests
    def cost(p):
        if p == "DAV": return RA["cost_ms"]
        m = RP[p]; return round(m["cost_ms"] - m["inspection_ms_per_request"] + CHECK * m["l2_checks"] / nh, 1)   # uniform 30 ms per check
    R = {"n": n, "U_prime": Up, "extra_frame": W["extra_frame"], "extra_sampled": len(W["S_extra"]), "unresolved": sum(v is None for v in L.values()), "arms": {}}
    for p in ARMS:
        d = [int(L[(q, p)] == "CORRECT") - int(L[(q, "B")] == "CORRECT") for q in S]; mu = sum(d) / n; var = sum((x - mu) ** 2 for x in d) / (n - 1); se = math.sqrt(var / n * (1 - n / Up))
        lo = sum(x < 0 for x in d); ga = sum(x > 0 for x in d)
        R["arms"][p] = {"losses": lo, "gains": ga, "net": ga - lo, "stream_pp": round(f * mu, 3), "ci_pp": [round(f * (mu - 1.96 * se), 3), round(f * (mu + 1.96 * se), 3)],
                        "expanded_losses": round(Up * lo / n, 1), "cost_ms": cost(p)}
    D = R["arms"]["D"]
    for p in ARMS[1:]:
        a = R["arms"][p]; dc = a["cost_ms"] - D["cost_ms"]; dl = D["expanded_losses"] - a["expanded_losses"]
        a["losses_removed_vs_D_stream"] = round(dl, 1); a["extra_cost_ms_vs_D"] = round(dc, 1); a["losses_removed_per_ms"] = round(dl / dc, 3) if dc > 0 else None
    json.dump(R, open(O / "outputs/admission_control_evaluation.json", "w"), indent=1); print(json.dumps(R, indent=1))
