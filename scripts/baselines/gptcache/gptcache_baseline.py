"""GPTCache baseline. Uses the official GPTCache package (gptcache==0.1.44 from PyPI, not vendored) with BGE-M3 embeddings and
FreshCache-matched answer admission; only the adapter is part of this repository.

GPTCache (official package, gptcache==0.1.44 from PyPI, unmodified) as a STANDALONE answer-cache baseline in the frozen FreshCache
replay harness. Variants:
  GPTCache_native     official init_similar_cache recipe: sqlite+faiss data manager, top_k=1, SearchDistanceEvaluation(max_distance=4.0,
                      positive=False), Config(similarity_threshold=0.8) [GPTCache default; with unit-norm BGE-M3 this admits cos >= 0.60],
                      temperature_softmax post-processing (temperature 0 = most similar). No threshold was selected or tuned.
  GPTCache_matchedFC  same GPTCache pipeline; evaluator replaced (documented extension point) by FreshCache's frozen primary answer admission,
                      top_k=40 candidates from GPTCache's faiss search (proven non-truncating: gptcache_topk_bound.py; held-out equal to a top_k=100 run), Config(similarity_threshold=0.80) on a (0, 1) score range.
Harness (identical to the frozen engine's answer-only scripts/baselines): empty cache per replay; REAL_TIME requests bypass the cache (rt_bypass=True, as
for every Table 2 baseline); miss = search + fetch every listed URL + generate, then GPTCache put(); hit = serve the stored answer (no search,
no fetch, no generation). No URL-list reuse, no page reuse, no TTL, no freshness-risk model.
Modes: replay_one VARIANT STREAM | emulate | collect | items | score"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, json, os, pickle, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
HERE = ((OUTPUT_ROOT / "baselines/gptcache")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/gptcache"))
import gptcache_helpers as X, gptcache_adapter as A
import numpy as np
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); MODE = sys.argv[1]; W = X.O / "work"
VARS = {"GPTCache_native": {"top_k": 1, "threshold": 0.8, "evaluator": "official SearchDistanceEvaluation(max_distance=4.0, positive=False)"},
        "GPTCache_matchedFC": {"top_k": 40, "threshold": 0.80, "evaluator": "local FCAdmissionEvaluation (FreshCache primary answer admission)"}}
LAB = {"GPTCache_native": "GPTCache-native (BGE-M3, official evaluator, threshold 0.8)", "GPTCache_matchedFC": "GPTCache-matched-FC-admission"}
MAX_SIZE = 1_000_000   # > any stream length: GPTCache's LRU eviction never triggers (every Table 2 method is capacity-unbounded)
K5 = ("search_saved_pct", "drift_pct", "coverage_pct", "fetch_per_1k", "gen_per_1k")
NPZ = (OUTPUT_ROOT / "l2_verifier/out/emb_queries.npz")
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/method_replay.py", "freshcache/replay_engine/engine_loader.py",
        "outputs/replay/results/replays_reported.json", "scripts/evaluation/generation_judging_stage.py", "freshcache/method/tiered_cache_model.py", "data/derived/round_table.jsonl", "data/sim_matrix_bgem3.npy",
        "outputs/l2_verifier/out/emb_queries.npz", "outputs/general_audit/sample.json", "outputs/l2_operating_points/l2_threshold_request_level.csv",
        "outputs/concurrency_cost/component_costs.json", "outputs/baselines/matched_admission/baseline_results.csv", "configs/integrity/FILE_HASHES.csv"]
if not (W / "read_only_before.json").exists(): json.dump({p: X.fstat(inpath(p)) for p in READ}, open(W / "read_only_before.json", "w"), indent=1)
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)


def harness(E, stream, lookup, store):
    """Answer-only harness, identical accounting to mixed_age_replay.replay for an l2='none', l3='none', rt_bypass=True configuration."""
    me, RT, exp = E["me"], E["rounds"], E["exp"]; SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    per, tr, ENT = [], {}, []; search = fetch = gen = hits = 0; lk = 0.0
    for t, r in stream:
        q, fc, qid = r["query"], r["freshness_class"], r["query_id"]; own = [u["url_hash"] for u in r["urls"]]
        if fc == "REAL_TIME":
            search += 1; fetch += len(own); gen += 1; per.append((qid, None)); tr[qid] = {"tier": "REAL_TIME_BYPASS", "ev": [(x, t) for x in own], "gen_query": q}; continue
        t0 = time.perf_counter(); e = lookup(q, fc); lk += time.perf_counter() - t0
        if e is not None:
            en = ENT[e]; hits += 1; o = me.aggregate([me.outcome(x, tc, t, RT) for x, tc in en["ev"]]); per.append((qid, me.UNOBS if o is None else o))
            tr[qid] = {"tier": "L1", "ev": en["ev"], "gen_query": en["q"], "entry": e, "entry_fc": en["fc"], "fc": fc, "sim": float(SIM[IDX[q], IDX[en["q"]]]), "age": t - en["t"]}
        else:
            search += 1; fetch += len(own); gen += 1; ev = [(x, t) for x in own]; eid = len(ENT); ENT.append({"q": q, "fc": fc, "t": t, "ev": ev}); store(q, eid, fc)
            per.append((qid, me.aggregate([]))); tr[qid] = {"tier": "miss", "ev": ev, "gen_query": q}
    n = len(stream); S = E["support_of"](stream, RT); P = dict(per)
    ch = sum(1 for q in S if P.get(q) == me.CHANGED); un = sum(1 for q in S if P.get(q) == me.UNCHANGED)
    m = {"n": n, "search_calls": search, "search_saved_pct": round(100 * (1 - search / n), 4), "drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None,
         "coverage_pct": round(100 * (ch + un) / len(S), 4), "fetches": fetch, "generations": gen, "fetch_per_1k": round(1000 * fetch / n, 1), "gen_per_1k": round(1000 * gen / n, 1),
         "cache_hits": hits, "l1_hits": hits, "l2_hits": 0, "l3_hits": 0, "changed": ch, "unchanged": un, "nonrt": sum(1 for _, r in stream if r["freshness_class"] != "REAL_TIME"),
         "lookup_s_total": lk, "lookup_ms_per_request": 1000 * lk / n}
    return m, P, tr


if MODE == "replay_one":
    v, sname = sys.argv[2], sys.argv[3]; A.block_auto_install()
    import gptcache
    from gptcache.adapter import api
    from gptcache.similarity_evaluation import SearchDistanceEvaluation
    E = X.engine(); exp = E["exp"]; stream = X.C.streams(E)[sname]; emb = A.BGEM3Embedding(NPZ)
    ev = SearchDistanceEvaluation() if v == "GPTCache_native" else A.make_fc_evaluation(exp)
    c, dm = A.make_cache(str(X.O / "run" / f"{v}_{sname}"), emb, ev, VARS[v]["top_k"], VARS[v]["threshold"], MAX_SIZE)
    TRUNC = [0]
    def lookup(q, fc):
        if hasattr(ev, "req_ge080"): ev.req_ge080 = 0
        a = api.get(q, cache_obj=c)
        if hasattr(ev, "req_ge080") and ev.req_ge080 >= VARS[v]["top_k"]: TRUNC[0] += 1
        return None if a is None else int(a)
    store = lambda q, eid, fc: api.put(q, str(eid), cache_obj=c)
    X.log(f"[{now()}] replay {v} {sname} gptcache {gptcache.__version__}")
    m, P, tr = harness(E, stream, lookup, store)
    m["faiss_entries_at_end"] = dm.v.count(); m["truncation_possible_requests"] = TRUNC[0]; m["embedding_calls"] = emb.calls; m["gptcache_version"] = gptcache.__version__
    if hasattr(ev, "n_eval"): m["fc_evaluator_calls"] = ev.n_eval; m["fc_candidates_cos_ge_080"] = ev.n_ge080
    pickle.dump({"m": m, "per": P, "tr": tr}, open(W / f"rep_{v}_{sname}.pkl", "wb"))
    X.log(sname, v, json.dumps({k: m[k] for k in K5 + ("cache_hits", "lookup_ms_per_request", "faiss_entries_at_end", "truncation_possible_requests")})); sys.exit(0)

if MODE == "emulate":   # harness gate: SCALM no-TTL + FreshCache admission through THIS harness must equal the Baselines engine result exactly
    E = X.engine(); exp = E["exp"]; SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX; ST = X.C.streams(E); out = {}
    BL = {(r["stream"], r["config"]): r for r in csv.DictReader(open((OUTPUT_ROOT / "baselines/matched_admission/baseline_results.csv"), encoding="utf-8"))}
    for sname in ("primary", "heldout"):
        st = {"qi": [], "q": [], "fc": [], "fr": []}
        def lookup(q, fc):
            if not st["qi"]: return None
            sims = np.asarray(SIM[IDX[q]])[np.asarray(st["qi"])]; same = np.fromiter((f == fc for f in st["fc"]), bool, len(st["fc"]))
            cand = np.nonzero((sims >= exp.L1_SIM_THRESHOLD) & same)[0]; cand = cand[sims[cand] >= exp._EQ_SIM_FLOOR]
            keep = [int(j) for j in cand if exp.semantic_equivalent(q, st["q"][int(j)], float(sims[j])) and exp._entity_match(q, st["q"][int(j)])]
            if not keep: return None
            kk = np.asarray(keep); fr = np.asarray([st["fr"][j] for j in kk]); h = int(kk[np.lexsort((-sims[kk], -fr))[0]]); st["fr"][h] += 1; return h
        def store(q, eid, fc): st["qi"].append(IDX[q]); st["q"].append(q); st["fc"].append(fc); st["fr"].append(0)
        m, _, _ = harness(E, ST[sname], lookup, store); b = BL[(sname, "SCALM_FCadmit_noTTL")]
        out[sname] = {"harness": {k: m[k] for k in K5 + ("l1_hits",)}, "baselines_engine": {k: b[k] for k in K5 + ("l1_hits",)},
                      "equal": all(abs(float(m[k]) - float(b[k])) < 1e-9 for k in K5 + ("l1_hits",))}
        X.log("EMULATE", sname, json.dumps(out[sname]))
    json.dump(out, open(W / "harness_emulation_gate.json", "w"), indent=1); sys.exit(0)

if MODE == "collect":
    E = X.engine(); exp = E["exp"]; G = {}
    em = json.load(open(W / "harness_emulation_gate.json"))
    for s in ("primary", "heldout"): G[f"{s}: harness reproduces the frozen-engine SCALM no-TTL + FC admission result exactly"] = em[s]["equal"]
    R = {(v, s): pickle.load(open(W / f"rep_{v}_{s}.pkl", "rb")) for v in VARS for s in ("primary", "heldout")}
    emb = A.BGEM3Embedding(NPZ); REQ = {s: {r["query_id"]: (t, r) for t, r in X.C.streams(E)[s]} for s in ("primary", "heldout")}; INV = {}
    for (v, s), d in R.items():
        hits = [q for q, x in d["tr"].items() if x["tier"] == "L1"]; m = d["m"]
        cosn = [float(emb.emb[emb.idx[REQ[s][q][1]["query"]]] @ emb.emb[emb.idx[d["tr"][q]["gen_query"]]]) for q in hits]
        fail = sum(1 for q in hits if not (exp.semantic_equivalent(REQ[s][q][1]["query"], d["tr"][q]["gen_query"], d["tr"][q]["sim"]) and exp._entity_match(REQ[s][q][1]["query"], d["tr"][q]["gen_query"])))
        INV[f"{s}/{v}"] = {"cache_hits": len(hits), "min_hit_cosine_normalised_embedding": min(cosn, default=None), "min_hit_cosine_frozen_SIM": min((d["tr"][q]["sim"] for q in hits), default=None),
                           "hits_failing_FC_admission": fail, "hits_cross_class": sum(1 for q in hits if d["tr"][q]["entry_fc"] != d["tr"][q]["fc"]),
                           "faiss_entries_at_end": m["faiss_entries_at_end"], "entries_expected (misses excl. REAL_TIME)": m["search_calls"] - (m["n"] - m["nonrt"]),
                           "truncation_possible_requests": m["truncation_possible_requests"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"], "lookup_ms_per_request": m["lookup_ms_per_request"]}
        G[f"{s}/{v}: answer-cache only (0 L2, 0 L3; fetches = all listed pages of every search)"] = m["l2_hits"] == 0 and m["l3_hits"] == 0 and m["fetches"] == sum(len(r["urls"]) for t, r in REQ[s].values() if d["tr"][r["query_id"]]["tier"] != "L1")
        G[f"{s}/{v}: no eviction (faiss entries = stored misses)"] = INV[f"{s}/{v}"]["faiss_entries_at_end"] == INV[f"{s}/{v}"]["entries_expected (misses excl. REAL_TIME)"]
        if v == "GPTCache_matchedFC":
            G[f"{s}/{v}: every hit passes FreshCache answer admission"] = fail == 0
            G[f"{s}/{v}: candidate list never truncated (fewer than top_k candidates with cos >= 0.80 at every request)"] = m["truncation_possible_requests"] == 0
        else:
            G[f"{s}/{v}: every hit within GPTCache native threshold (normalised cos >= 0.60 - 1e-4)"] = min(cosn, default=1) >= 0.6 - 1e-4
    TB = json.load(open(W / "topk_bound.json"))
    for s in ("primary", "heldout"): G[f"{s}: matched top_k={VARS['GPTCache_matchedFC']['top_k']} exceeds max #entries with embedding cos >= 0.79 ({TB[s]['max_entries_emb_cos_ge_0.79']}); max |emb cos - SIM| {TB[s]['max_abs_emb_cos_minus_SIM']:.4f} < 0.01"] = TB[s]["max_entries_emb_cos_ge_0.79"] < VARS["GPTCache_matchedFC"]["top_k"] and TB[s]["max_abs_emb_cos_minus_SIM"] < 0.01
    for s in ("primary", "heldout"): G[f"{s}: matched GPTCache hits equal the numpy decision-rule emulation ({TB[s]['emulated_hits']})"] = R[("GPTCache_matchedFC", s)]["m"]["cache_hits"] == TB[s]["emulated_hits"]
    if (W / "rep_GPTCache_matchedFC_heldout_topk100.pkl").exists():   # optional: a held-out replay rerun with top_k=100
     T100 = pickle.load(open(W / "rep_GPTCache_matchedFC_heldout_topk100.pkl", "rb")); T40 = R[("GPTCache_matchedFC", "heldout")]
     G["heldout: matched top_k=40 run identical to top_k=100 run (every route, served entry and drift outcome)"] = T100["per"] == T40["per"] and all(T100["tr"][q]["tier"] == T40["tr"][q]["tier"] and T100["tr"][q]["gen_query"] == T40["tr"][q]["gen_query"] and T100["tr"][q]["ev"] == T40["tr"][q]["ev"] for q in T40["tr"])
    FR = json.load(open(X.CRR / "results/replays_reported.json"))
    for k, v in G.items(): X.log("GATE", k, v)
    if not all(G.values()): X.log("STOP: gate failed"); sys.exit(1)
    ROWS = []
    for s in ("primary", "heldout"):
        for v in VARS:
            m = R[(v, s)]["m"]; ROWS.append({"stream": s, "method": LAB[v], "config": v, **{k: m[k] for k in K5 + ("cache_hits", "l1_hits", "l2_hits", "l3_hits", "changed", "unchanged", "search_calls", "fetches", "generations", "nonrt", "lookup_ms_per_request")}, "source": "this run (official GPTCache 0.1.44)"})
        f = FR[f"{s}/FreshCache"]; n = f["n"]; nonrt = sum(1 for t, r in REQ[s].values() if r["freshness_class"] != "REAL_TIME")
        ROWS.append({"stream": s, "method": "FreshCache primary (frozen, reference)", "config": "FreshCache", **{k: f[k] for k in K5 + ("l1_hits", "l2_hits", "l3_hits", "changed", "unchanged", "fetches", "generations")},
                     "cache_hits": f["l1_hits"] + f["l2_hits"], "search_calls": round(n * (1 - f["search_saved_pct"] / 100)), "nonrt": nonrt, "lookup_ms_per_request": X.LOOKUP["FreshCache"], "source": "frozen replays_reported.json (not rerun)"})
    for r in ROWS: r.update(X.cost(r["search_calls"], r["fetches"], r["generations"], len(REQ[r["stream"]]), r["nonrt"], r["lookup_ms_per_request"]))
    fcH = next(r for r in ROWS if r["stream"] == "heldout" and r["config"] == "FreshCache")
    G2 = {"FreshCache held-out modeled cost reproduces cost-model/CategoryAware-frontier (local / live ms/request, whole ms; run-time reference)": (round(fcH["local_ms_per_request"]), round(fcH["live_p50_ms_per_request"])) == tuple(round(REF.value(f"cost/FreshCache_heldout/{k}_ms_per_request_2dp")) for k in ("local", "live_p50"))}
    for k, v in G2.items(): X.log("GATE", k, v)
    if not all(G2.values()): X.log("STOP: cost gate failed"); sys.exit(1)
    G.update(G2); AUD = {}; hl = X.Held(); todo = {}
    for v in VARS:
        tr = R[(v, "heldout")]["tr"]; AUD[v] = {}
        for q in IDS:
            x = tr[q]; k, ctx, gq2 = X.key_for({"ev": x["ev"], "gen_query": x["gen_query"]}, SQ[q]["query"]); AUD[v][q] = {"tier": x["tier"], "key": k}
            if hl.answer(k) is None: todo.setdefault(tuple(k), {"key": k, "question": gq2, "context": ctx})
    X.wjl(X.O / "gpu/in/gen_tasks.jsonl", list(todo.values()))
    (X.O / "request_level").mkdir(exist_ok=True)
    for (v, s), d in R.items():
        with open(X.O / f"request_level/{s}_{v}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["request_id", "t", "freshness_class", "tier", "drift_outcome", "hit_matched_query", "hit_cosine_frozen_SIM", "hit_entry_age_s", "n_evidence_pages"])
            for t, r in REQ[s].values():
                q = r["query_id"]; x = d["tr"][q]; o = d["per"].get(q)
                w.writerow([q, t, r["freshness_class"], x["tier"], {E["me"].CHANGED: "CHANGED", E["me"].UNCHANGED: "UNCHANGED"}.get(o, "UNOBSERVABLE" if o is not None else ""),
                            x["gen_query"] if x["tier"] == "L1" else "", x.get("sim", ""), x.get("age", ""), len(x["ev"])])
    pickle.dump({"rows": ROWS, "gates": G, "inv": INV, "audit": AUD}, open(W / "collect.pkl", "wb")); X.log("new answers needed:", len(todo)); sys.exit(0)

ST_ = pickle.load(open(W / "collect.pkl", "rb")); hl = X.Held()
if MODE == "items":
    items, miss = X.judge_items(hl, [(SQ[q]["query"], SQ[q]["gold"], a["key"]) for A_ in ST_["audit"].values() for q, a in A_.items()])
    if miss: X.log(f"STOP: {len(miss)} answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("judge2 items:", len(items)); sys.exit(0)
L2_REQUESTS = {r["request_id"]: r for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")) if r["point"] == "theta_0.75"}
FRESH = {q: L2_REQUESTS[q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values())
L = {"FreshCache": {q: L2_REQUESTS[q]["answer_label"] for q in IDS}}
for v, A_ in ST_["audit"].items(): L[v] = {q: (hl.labels(SQ[q]["query"], SQ[q]["gold"], A_[q]["key"])[0] or "UNRESOLVED") for q in IDS}
wa = lambda L_, q: FRESH[q] == "CORRECT" and L_[q] == "WRONG"
HC = {r["config"]: r for r in ST_["rows"] if r["stream"] == "heldout"}; QR = []
for v in list(VARS) + ["FreshCache"]:
    c = collections.Counter(L[v].values()); w = sum(1 for q in IDS if wa(L[v], q))
    row = {"method": LAB.get(v, "FreshCache primary (frozen, reference)"), "config": v, "N": len(IDS), "fresh_correct": FCN, "correct": c["CORRECT"], "accuracy_pct": round(100 * c["CORRECT"] / len(IDS), 2),
           "wrong": c["WRONG"], "abstain": c["ABSTAIN"], "unresolved": c["UNRESOLVED"], "WAI": w, "WAI_pct": round(100 * w / len(IDS), 2), "C_WAI_pct": round(100 * w / FCN, 2),
           "heldout_cost_local_ms_per_request": HC[v]["local_ms_per_request"], "heldout_cost_live_p50_ms_per_request": HC[v]["live_p50_ms_per_request"],
           "audit_tiers": json.dumps(dict(collections.Counter(ST_["audit"][v][q]["tier"] for q in IDS))) if v in ST_["audit"] else ""}
    if v != "FreshCache":
        b = sum(1 for q in IDS if wa(L[v], q) and not wa(L["FreshCache"], q)); a_ = sum(1 for q in IDS if wa(L["FreshCache"], q) and not wa(L[v], q))
        cb = sum(1 for q in IDS if L[v][q] == "CORRECT" and L["FreshCache"][q] != "CORRECT"); ca = sum(1 for q in IDS if L["FreshCache"][q] == "CORRECT" and L[v][q] != "CORRECT")
        row.update({"WAI_FreshCache_only": a_, "WAI_GPTCache_only": b, "WAI_mcnemar_p": X.mcnemar(a_, b), "correct_FreshCache_only": ca, "correct_GPTCache_only": cb, "correct_mcnemar_p": X.mcnemar(ca, cb)})
    QR.append(row)
cols = ["stream", "method", "config"] + list(K5) + ["cache_hits", "l1_hits", "l2_hits", "l3_hits", "changed", "unchanged", "search_calls", "fetches", "generations", "lookup_ms_per_request", "local_ms_per_request", "live_p50_ms_per_request", "source"]
with open(X.O / "gptcache_results.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(ST_["rows"])
with open(X.O / "gptcache_answer_quality.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in QR for k in r))); w.writeheader(); w.writerows(QR)
with open(X.O / "request_level/heldout_audit_labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["request_id", "fresh_label", "FreshCache_label"] + [x for v in VARS for x in (f"{v}_tier", f"{v}_label")])
    for q in IDS: w.writerow([q, FRESH[q], L["FreshCache"][q]] + [x for v in VARS for x in (ST_["audit"][v][q]["tier"], L[v][q])])
BEF = json.load(open(W / "read_only_before.json")); UNCH = {p: X.fstat(inpath(p)) == v for p, v in BEF.items()}
json.dump({"generated_utc": now(), "gates": ST_["gates"], "invariants": ST_["inv"], "quality": QR, "read_only_inputs_unchanged": UNCH}, open(W / "score_summary.json", "w"), indent=1, default=str)
X.log("scored; read-only inputs unchanged:", all(UNCH.values()))
for r in ST_["rows"]: print(r["stream"], r["config"], [r[k] for k in K5], r["cache_hits"], round(r["local_ms_per_request"], 1), round(r["live_p50_ms_per_request"], 1))
for r in QR: print(json.dumps(r))
