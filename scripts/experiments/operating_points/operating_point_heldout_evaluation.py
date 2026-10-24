"""Operating-point comparison, held-out stage (scored once after selection_frozen.json). Modes: replay | items | score.
Points: NoCache; FreshCache primary; FreshCache theta_L2 0.80; every validation-selected SemanticTTL and CategoryAware point (quality-matched,
relaxed, cost-matched). Operations + common cost model on the held-out replay; accuracy / WAI / C-WAI / abstention on the frozen
audit (2-judge rule); evidence sufficiency on the evidence each policy gives the generator (frozen SUFFV two-judge
rule; FreshCache arms from the frozen L2 operating-point audit). Nothing is selected or changed after this point."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, hashlib, json, math, os, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/operating_points"))
import operating_point_helpers as X; import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # run-time reference values (outputs/reference_values.json)
from fractions import Fraction
MODE = sys.argv[1]; W = X.O / "work"
SEL = json.load(open(X.O / "selection_frozen.json")); assert SEL["heldout_consulted"] is False
CO = json.load(open(X.COST_DIR / "component_costs.json"))["coefficients"]; LK = json.load(open(X.O / "design_frozen.json"))["lookup_ms"]
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
POINTS = {}
for fam in ("SemanticTTL", "CategoryAware"):
    for rule, c in SEL["selection"][fam].items():
        if isinstance(c, str): POINTS.setdefault(c, {"family": fam, "rules": []})["rules"].append(rule)
if MODE == "replay":
    E = X.engine(); H = X.C.streams(E)["heldout"]; n = len(H); S = E["support_of"](H, E["rounds"]); nonrt = sum(1 for _, r in H if r["freshness_class"] != "REAL_TIME")
    CL = ["TIMELESS", "SLOW", "MEDIUM", "FAST"]; OUT, AUD = {}, {}
    def cost(s, f, g, emb, lk):
        return {f"{sc}_ms_per_request": round((s * CO["c_search_ms"][sc] + f * CO["c_fetch_ms"][sc] + g * CO["c_gen_ms"]["value"] + emb * CO["c_embed_ms"]["value"]) / n + lk, 3) for sc in ("local", "live_p50")}
    for pid, p in POINTS.items():
        c = SEL["cells"][pid]
        th = c["theta_by_class"] if p["family"] == "CategoryAware" else {k: c["theta"] for k in CL}; ka = c["kappa_by_class"] if p["family"] == "CategoryAware" else {k: c["kappa"] for k in CL}
        if p["family"] == "CategoryAware" and any(v is None for v in list(th.values()) + list(ka.values())): th, ka = X.ca_params(c["dtheta"], c["m_ttl"])   # the frontier's own reconstruction
        th["REAL_TIME"] = 1.1; ka["REAL_TIME"] = 0.0
        assert all(v is not None for v in list(th.values()) + list(ka.values())), (pid, th, ka)
        m, rows = X.ca_replay(H, th, ka); per, _, dr = X.NS_drift(E, H, rows, S) if hasattr(X, "NS_drift") else (None, None, {})
        OUT[pid] = {"family": p["family"], "rules": p["rules"], "search_saved_pct": m["search_saved_pct"], "search_calls": m["search_calls"], "fetches": m["fetches"], "generations": m["generations"],
                    "fetch_per_1k": m["fetch_per_1k"], "gen_per_1k": m["gen_per_1k"], "l1_hits": m["l1_hits"], **cost(m["search_calls"], m["fetches"], m["generations"], nonrt, LK["CategoryAware" if p["family"] == "CategoryAware" else "SemanticTTL"])}
        AUD[pid] = {}
        for q in IDS:
            k, ctx, gq = X.key_for(rows[q], SQ[q]["query"]); AUD[pid][q] = {"tier": rows[q]["tier"], "key": k, "gen_question": gq, "context": ctx, "ev": rows[q].get("ev")}
        X.log("heldout", pid, json.dumps({x: OUT[pid][x] for x in ("search_saved_pct", "fetch_per_1k", "gen_per_1k", "live_p50_ms_per_request")}))
    # FreshCache arms (frozen engine) and NoCache
    FR = json.load(open(X.CRR / "results/replays_reported.json"))["heldout/FreshCache"]
    fsc = round(n * (1 - FR["search_saved_pct"] / 100)); OUT["FreshCache_primary"] = {"family": "FreshCache", "rules": ["primary (frozen)"], **{x: FR[x] for x in ("search_saved_pct", "fetches", "generations", "fetch_per_1k", "gen_per_1k", "l1_hits")}, "search_calls": fsc, **cost(fsc, FR["fetches"], FR["generations"], nonrt, LK["FreshCache"])}
    o = X.engine()["exp"]._L2_EQ_SIM_FLOOR; X.engine()["exp"]._L2_EQ_SIM_FLOOR = 0.80
    try: m80, _, t80 = X.C.metrics(E, H, "FreshCache", S)
    finally: X.engine()["exp"]._L2_EQ_SIM_FLOOR = o; assert o == 0.75
    s80 = sum(1 for q in t80 if t80[q]["tier"] in ("miss", "REAL_TIME_BYPASS"))
    OUT["FreshCache_thetaL2_0.80"] = {"family": "FreshCache", "rules": ["theta_L2 0.80 sensitivity"], **{x: m80[x] for x in ("search_saved_pct", "fetches", "generations", "fetch_per_1k", "gen_per_1k", "l1_hits")}, "search_calls": s80, **cost(s80, m80["fetches"], m80["generations"], nonrt, LK["FreshCache"])}
    ncf = sum(len(r["urls"]) for _, r in H); OUT["NoCache"] = {"family": "NoCache", "rules": ["reference"], "search_saved_pct": 0.0, "search_calls": n, "fetches": ncf, "generations": n, "fetch_per_1k": round(1000 * ncf / n, 1), "gen_per_1k": 1000.0, "l1_hits": 0, **cost(n, ncf, n, 0, 0.0)}
    G = {"FreshCache primary live cost equals the recorded reference": REF.check("cost/FreshCache_heldout/live_p50_ms_per_request_2dp", round(OUT["FreshCache_primary"]["live_p50_ms_per_request"], 2)), "FreshCache theta 0.80 saved reproduces the L2-sweep point": m80["search_saved_pct"] == float(next(r for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_points.csv")) if r["point"] == "theta_0.80")["search_saved_pct"])}
    for k, v in G.items(): X.log("GATE", k, v)
    if not all(G.values()): X.log("STOP"); sys.exit(1)
    hl = X.Held(); todo = {}; SUFT = {}
    for pid, A in AUD.items():
        for q, a in A.items():
            if hl.answer(a["key"]) is None: todo.setdefault(tuple(a["key"]), {"key": a["key"], "question": a["gen_question"], "context": a["context"]})
            eb = X.evblock([tuple(x) for x in (a["ev"] or [])]); a["evidence_sha"] = hashlib.sha256(eb.encode()).hexdigest()
            SUFT.setdefault((q, a["evidence_sha"]), {"query_id": q, "arm": a["evidence_sha"][:16], "query": SQ[q]["query"], "evidence": eb})
    X.wjl(X.O / "gpu/in/gen_tasks.jsonl", list(todo.values())); X.wjl(W / "heldout_suff_candidates.jsonl", list(SUFT.values()))
    pickle.dump({"ops": OUT, "audit": {p: {q: {k: v for k, v in a.items() if k not in ("context", "ev")} for q, a in A.items()} for p, A in AUD.items()}, "gates": G}, open(W / "stage3.pkl", "wb"))
    X.log("held-out new answers needed:", len(todo), "distinct sufficiency evidence blocks:", len(SUFT)); sys.exit(0)
ST = pickle.load(open(W / "stage3.pkl", "rb")); hl = X.Held()
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
SL = {}   # (qid, sha256(evidence block)) -> (llama8b, qwen7b), from every frozen sufficiency source + this folder
def addsuff(tasks, l8, q7):
    for t in tasks:
        a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
        if a is not None and b is not None: SL.setdefault((t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()), (a, b))
L8 = lambda f: {(r["query_id"], r["arm"]): r["label"] for r in jl(f)}
addsuff(jl(X.OV / "audit_suff_tasks.jsonl"), L8(X.OV / "audit_suff_llama8b.jsonl"), L8(X.OV / "audit_suff_qwen7b.jsonl"))
RU = json.load(open(X.CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
cl8.update(L8(X.CRR / "gpu/out/suffv_llama8b.jsonl")); cq7.update(L8(X.CRR / "gpu/out/suffv_qwen7b.jsonl")); addsuff(jl(X.CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7)
for D in (X.L2_POINTS_DIR, X.CA_FRONTIER_DIR, X.CA_COMPOSITION_DIR, X.O):
    addsuff(jl(D / "gpu/in/suffv_tasks.jsonl"), {**L8(D / "gpu/out/suffv_llama8b.jsonl"), **{k: v for s in range(8) for k, v in L8(D / f"gpu/out/suffv_llama8b_shard{s}.jsonl").items()}},
            {**L8(D / "gpu/out/suffv_qwen7b.jsonl"), **{k: v for s in range(8) for k, v in L8(D / f"gpu/out/suffv_qwen7b_shard{s}.jsonl").items()}})
if MODE == "items":
    items, miss = X.judge_items(hl, [(SQ[q]["query"], SQ[q]["gold"], a["key"]) for A in ST["audit"].values() for q, a in A.items()])
    if miss: X.log(f"STOP: {len(miss)} answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items)
    need = [t for t in jl(W / "heldout_suff_candidates.jsonl") if (t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()) not in SL]
    X.wjl(X.O / "gpu/in/suffv_tasks.jsonl", need); X.log("judge2 items:", len(items), "sufficiency tasks:", len(need)); sys.exit(0)
# ---------------- score
L2_REQUESTS = {}
for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")):
    if r["point"] in ("theta_0.75", "theta_0.80"): L2_REQUESTS.setdefault(r["point"], {})[r["request_id"]] = r
FRESH = {q: L2_REQUESTS["theta_0.75"][q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values()); assert REF.check("heldout_audit/fresh_correct", FCN)
def suff(q, sha):
    x = SL.get((q, sha)); return None if x is None else (x[0] if x[0] == x[1] else "JUDGE_DISAGREEMENT")
LAB = {"FreshCache_primary": {q: (L2_REQUESTS["theta_0.75"][q]["answer_label"], L2_REQUESTS["theta_0.75"][q]["sufficiency"]) for q in IDS},
       "FreshCache_thetaL2_0.80": {q: (L2_REQUESTS["theta_0.80"][q]["answer_label"], L2_REQUESTS["theta_0.80"][q]["sufficiency"]) for q in IDS},
       "NoCache": {q: (FRESH[q], None) for q in IDS}}
for pid, A in ST["audit"].items(): LAB[pid] = {q: (hl.labels(SQ[q]["query"], SQ[q]["gold"], a["key"])[0] or "UNRESOLVED", suff(q, a["evidence_sha"]) or "UNRESOLVED") for q, a in A.items()}
def mcn(b, c):
    n = b + c; return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [round(100 * (c - m) / d, 2), round(100 * (c + m) / d, 2)]
TAB = []
for pid, o in ST["ops"].items():
    L = LAB[pid]; c = collections.Counter(a for a, _ in L.values()); s = collections.Counter(x for _, x in L.values()); w = sum(1 for q in IDS if FRESH[q] == "CORRECT" and L[q][0] == "WRONG")
    fc = LAB["FreshCache_primary"]; wa = sum(1 for q in IDS if FRESH[q] == "CORRECT" and fc[q][0] == "WRONG" and L[q][0] != "WRONG"); wb = sum(1 for q in IDS if FRESH[q] == "CORRECT" and L[q][0] == "WRONG" and fc[q][0] != "WRONG")
    sa = sum(1 for q in IDS if fc[q][1] == "SUFFICIENT" and L[q][1] != "SUFFICIENT"); sb = sum(1 for q in IDS if L[q][1] == "SUFFICIENT" and fc[q][1] != "SUFFICIENT")
    TAB.append({"point": pid, "family": o["family"], "selection_rule": "; ".join(o["rules"]), "search_saved_pct": o["search_saved_pct"], "fetch_per_1k": o["fetch_per_1k"], "gen_per_1k": o["gen_per_1k"],
                "live_cost_s": round(o["live_p50_ms_per_request"] / 1000, 3), "local_cost_s": round(o["local_ms_per_request"] / 1000, 3), "accuracy_pct": round(100 * c["CORRECT"] / len(IDS), 2), "abstain_pct": round(100 * c["ABSTAIN"] / len(IDS), 2),
                "WAI": w, "C_WAI_pct": round(100 * w / FCN, 2), "C_WAI_ci95": wil(w, FCN), "sufficient": s["SUFFICIENT"] if pid != "NoCache" else None, "sufficient_pct": round(100 * s["SUFFICIENT"] / len(IDS), 2) if pid != "NoCache" else None,
                "unresolved_answer": c["UNRESOLVED"], "unresolved_suff": s["UNRESOLVED"] if pid != "NoCache" else None,
                "vs_FreshCache_WAI_FConly_otheronly_p": [wa, wb, mcn(wa, wb)], "vs_FreshCache_SUFF_FConly_otheronly_p": [sa, sb, mcn(sa, sb)] if pid != "NoCache" else None})
json.dump({"decision_table": TAB, "selection_sha256": SEL["sha256"]}, open(X.O / "heldout_decision_table.json", "w"), indent=1)
with open(X.O / "heldout_decision_table.csv", "w", newline="") as f: w_ = csv.DictWriter(f, fieldnames=list(TAB[0])); w_.writeheader(); w_.writerows(TAB)
for r in TAB: print(json.dumps(r))
