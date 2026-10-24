"""Policy D-prime = D, except answers generated from a reused URL list are not written to the answer cache (pre-registered rule).
Modes: replay | items | score.
replay: gate (default switch reproduces the frozen D trace exactly: answer hits, savings, every signature), held-out and validation
        operation counts / modeled cost for B, D, D'; B-vs-D' changed population; frozen audit sample (seed 20261006) -> work/dprime_sample.json.
items:  generation tasks (Llama-3.2-3B FORCED, each policy's own evidence) for answers not already in the frozen pools; then judge/SUFFV items.
score:  losses/gains, loss given B correct, sufficiency, fixed-stream estimate (M_e/N_ref)(gains-losses)/n with exact hypergeometric CI."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, math, pickle, random, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "write_controls")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("write_controls"); import composition_study_helpers as X, composition_engine_reuse_write_suppression as RE   # noqa: E402
MODE = sys.argv[1]; E = X.engine(); p2 = E["p2"]; REC = {r["query_id"]: r for r in E["recs"]}
GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid, qt):
    g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
TH, KA = X.ca_params(0.0, 1.0); CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # component_costs.json (serving_cost_summary.py)
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
if MODE == "replay":
    S = X.C.streams(E); FZ = pickle.load(open(X.RCD / "work/composition_ablation_replay.pkl", "rb"))["policies"]
    def run(stream, **kw):
        m, rows = RE.replay(E, stream, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **kw); n = m["n_requests"]
        nonrt = sum(1 for _, r in stream if r["freshness_class"] != "REAL_TIME") / n
        m["cost_ms"] = round((m["search_calls"] * CS + m["fetches"] * CF + m["generations"] * CG) / n + nonrt * CE, 1); return m, rows
    R = {}
    mD, rD = run(S["heldout"])
    gate = {"answer_hits": mD["l1_hits"], "saved": round(mD["search_saved_pct"], 2), "signatures_equal_frozen_D": all(sig(rD[q]) == (FZ["D"]["sig"][q][0], tuple(tuple(x) for x in FZ["D"]["sig"][q][1]), FZ["D"]["sig"][q][2]) for q in rD)}
    assert gate["answer_hits"] == FZ["D"]["metrics"]["l1_hits"] and gate["saved"] == round(FZ["D"]["metrics"]["search_saved_pct"], 2) and gate["signatures_equal_frozen_D"], gate
    R["gate"] = gate
    for split in ("heldout", "validation"):
        R[split] = {}
        for p, kw in (("B", dict(l2_on=False)), ("D", dict()), ("Dprime", dict(l1_write_from_l2=False))):
            m, rows = run(S[split], **kw); n = m["n_requests"]
            R[split][p] = {"saved_pct": round(m["search_saved_pct"], 2), "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "search_per_1k": round(1000 * m["search_calls"] / n, 1),
                           "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1), "cost_ms": m["cost_ms"]}
            if split == "heldout": R.setdefault("_rows", {})[p] = rows
        for a, b in (("B", "D"), ("B", "Dprime"), ("D", "Dprime")):
            R[split][f"cost_cut_{a}_to_{b}_pct"] = round(100 * (1 - R[split][b]["cost_ms"] / R[split][a]["cost_ms"]), 2)
    rows = R.pop("_rows"); B, Dp, Dd = rows["B"], rows["Dprime"], rows["D"]
    M = [q for q in B if sig(B[q]) != sig(Dp[q])]; Me = [q for q in M if goldof(q, REC[q]["query"])]
    MD = [q for q in B if sig(B[q]) != sig(Dd[q])]
    Nref = sum(1 for _, r in S["heldout"] if goldof(r["query_id"], r["query"]))
    route = collections.Counter(f"{B[q]['tier']}->{Dp[q]['tier']}" for q in M)
    samp = sorted(random.Random(20261006).sample(sorted(Me), 400))
    R["changed"] = {"N": len(B), "N_ref": Nref, "M_B_vs_Dprime": len(M), "M_e": len(Me), "M_u": len(M) - len(Me), "route_transitions": dict(route.most_common()),
                    "M_B_vs_D": len(MD), "share_of_D_changed_removed_pct": round(100 * (1 - len(M) / len(MD)), 1)}
    json.dump({"seed": 20261006, "n": len(samp), "requests": samp, "rows": {q: {p: {"tier": rows[p][q]["tier"], "ev": [list(x) for x in rows[p][q]["ev"]], "gen_query": rows[p][q].get("gen_query")} for p in ("B", "Dprime")} for q in samp},
               "frame": len(Me), "M": len(M), "N_ref": Nref}, open(O / "work/dprime_sample.json", "w"))
    json.dump(R, open(O / "outputs/dprime_replay.json", "w"), indent=1); print(json.dumps(R, indent=1)); sys.exit(0)
F = json.load(open(O / "work/dprime_sample.json")); hl = X.Held()
A = {}
for q in F["requests"]:
    for p in ("B", "Dprime"):
        r = F["rows"][q][p]; k, ctx, gq = X.key_for({"ev": r["ev"], "gen_query": r["gen_query"]}, REC[q]["query"]); ev = [tuple(x) for x in r["ev"]]; blk = X.evblock(ev) if ev else None
        A[(q, p)] = {"key": k, "ctx": ctx, "gq": gq, "blk": blk, "sha": hashlib.sha256(blk.encode()).hexdigest() if blk else None}
if MODE == "items":
    gen = {tuple(a["key"]): {"key": a["key"], "question": a["gq"], "context": a["ctx"]} for a in A.values() if hl.answer(a["key"]) is None}
    X.wjl(O / "gpu/in/gen_tasks.jsonl", list(gen.values())); print("gen", len(gen))
    if not gen:
        items, miss = X.judge_items(hl, [(REC[q]["query"], goldof(q, REC[q]["query"]), A[(q, p)]["key"]) for q, p in A]); assert not miss
        X.wjl(O / "gpu/in/judge2_items.jsonl", items)
        X.wjl(O / "gpu/in/suffv_tasks.jsonl", [{"query_id": q, "arm": a["sha"][:16], "query": REC[q]["query"], "evidence": a["blk"]} for (q, p), a in A.items() if a["blk"]])
        print("judge2", len(items))
    sys.exit(0)
if MODE == "score":
    jl = X.jl; S = {j: {} for j in ("llama8b", "qwen7b")}
    for j in S:
        for f in [O / f"gpu/out/suffv_{j}.jsonl"] + sorted((O / "gpu/out").glob(f"suffv_{j}_shard*.jsonl")):
            for r in jl(f): S[j][(r["query_id"], r["arm"])] = r["label"]
    for j in S:
        for d in (X.RCD, ((OUTPUT_ROOT / "composition_checks"))):
            for f in [d / f"gpu/out/suffv_{j}.jsonl"]:
                for r in jl(f): S[j].setdefault((r["query_id"], r["arm"]), r["label"])
    L = {}
    for (q, p), a in A.items():
        lab = hl.labels(REC[q]["query"], goldof(q, REC[q]["query"]), a["key"])[0] or "UNRESOLVED"
        s = (S["llama8b"].get((q, a["sha"][:16])), S["qwen7b"].get((q, a["sha"][:16]))) if a["sha"] else (None, None)
        L[(q, p)] = (lab, "UNRESOLVED" if None in s else (s[0] if s[0] == s[1] else "JUDGE_DISAGREEMENT"))
    ids = F["requests"]; n = len(ids); T = collections.Counter()
    for q in ids:
        b, d = L[(q, "B")], L[(q, "Dprime")]
        T["unresolved"] += "UNRESOLVED" in (b[0], d[0]) or "UNRESOLVED" in (b[1], d[1])
        if b[0] == "CORRECT" and d[0] == "WRONG": T["to_wrong"] += 1
        if b[0] == "CORRECT" and d[0] == "ABSTAIN": T["to_abstain"] += 1
        if b[0] != "CORRECT" and d[0] == "CORRECT": T["gains"] += 1
        if b[1] == "SUFFICIENT" and d[1] != "SUFFICIENT": T["suff_loss"] += 1
        if b[1] != "SUFFICIENT" and d[1] == "SUFFICIENT": T["suff_gain"] += 1
    loss = T["to_wrong"] + T["to_abstain"]; gain = T["gains"]; bc = sum(1 for q in ids if L[(q, "B")][0] == "CORRECT")
    Me, Nref = F["frame"], F["N_ref"]
    def wil(k, n, z=1.96):
        p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [round(100 * (c - m) / d, 2), round(100 * (c + m) / d, 2)]
    def hyper_ci(x, n, Np, a=0.05):
        pm = lambda K, i: math.comb(K, i) * math.comb(Np - K, n - i) / math.comb(Np, n)
        lc = lambda K: sum(pm(K, i) for i in range(0, min(x, K) + 1)); uc = lambda K: 1 - sum(pm(K, i) for i in range(0, x))
        return next(K for K in range(x, Np + 1) if uc(K) > a / 2), max(K for K in range(x, Np - n + x + 1) if lc(K) > a / 2)
    net = gain - loss
    R = {"n": n, "transitions": dict(T), "B_correct": bc, "losses": loss, "gains": gain, "loss_rate_pct": round(100 * loss / n, 2), "loss_rate_ci": wil(loss, n),
         "loss_given_B_correct_pct": round(100 * loss / bc, 1) if bc else None, "loss_given_B_correct_ci": wil(loss, bc) if bc else None,
         "M_e": Me, "N_ref": Nref, "delta_acc_pp": round(100 * Me / Nref * net / n, 3), "formula": f"({Me}/{Nref})*({gain}-{loss})/{n}",
         "delta_suff_pp": round(100 * Me / Nref * (T["suff_gain"] - T["suff_loss"]) / n, 3)}
    if gain == 0 and loss > 0:
        lo, hi = hyper_ci(loss, n, Me); R["losses_in_M_e_ci"] = [lo, hi]; R["delta_acc_ci_pp"] = [round(-100 * hi / Nref, 3), round(-100 * lo / Nref, 3)]; R["ci_method"] = "exact hypergeometric (no gains)"
    else:   # per-request net d in {-1,0,+1}; Wald interval with finite-population correction (SRS without replacement from M_e)
        d = [(1 if (L[(q, "B")][0] != "CORRECT" and L[(q, "Dprime")][0] == "CORRECT") else 0) - (1 if (L[(q, "B")][0] == "CORRECT" and L[(q, "Dprime")][0] != "CORRECT") else 0) for q in ids]
        mu = sum(d) / n; var = sum((x - mu) ** 2 for x in d) / (n - 1); se = math.sqrt(var / n * (1 - n / Me))
        R["delta_acc_ci_pp"] = [round(100 * Me / Nref * (mu - 1.96 * se), 3), round(100 * Me / Nref * (mu + 1.96 * se), 3)]; R["ci_method"] = "Wald with finite-population correction" 
    json.dump(R, open(O / "outputs/dprime_quality.json", "w"), indent=1); print(json.dumps(R, indent=1))
