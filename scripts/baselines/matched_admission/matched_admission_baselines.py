"""Matched-admission baselines. SCALM-style is our reimplementation of SCALM's admission policy (Li et al., IWQoS 2024), not the
complete published system; it omits SCALM's hierarchical cache structure and its eviction policy.

Fair STANDALONE scripts/baselines with FreshCache's frozen primary answer-admission rule (no FreshCache L2/L3, no FreshCache temporal rule):
  SCALM_FCadmit_noTTL          SCALM-style answer cache only; same-class candidates (cos >= 0.40), FreshCache L1 admission, SCALM frequency-then-similarity
                               selection, NO TTL.
  TieredFixedTTL_FCadmit_L2075 TieredFixedTTL three tiers with its own fixed TTLs (ttl='halflife': age <= HALF_LIFE[class] x TIER_MULT[tier]) on every tier;
                               L1 = FreshCache answer admission; L2 raw similarity floor 0.75 (theta_L2); L3 exact-URL fixed-TTL refetch.
Modes: replay | items | score. All writes under Baselines. FreshCache is NOT rerun (its rows are read from frozen outputs)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, json, os, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/matched_admission"))
import matched_admission_helpers as X, matched_admission_engine_patch as PE
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); MODE = sys.argv[1]
SC, TT = "SCALM_FCadmit_noTTL", "TieredFixedTTL_FCadmit_L2075"
LAB = {SC: "SCALM-style (no TTL) + FreshCache answer admission", TT: "TieredFixedTTL (fixed TTLs) + FreshCache answer admission + theta_L2 0.75"}
REF = {"SCALM": "SCALM-style (Table 2, reference)", "TieredFixedTTL": "TieredFixedTTL (Table 2, reference)", "FreshCache": "FreshCache primary (frozen, reference)"}
K5 = ("search_saved_pct", "drift_pct", "coverage_pct", "fetch_per_1k", "gen_per_1k")
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/method_replay.py", "freshcache/replay_engine/engine_loader.py",
        "freshcache/replay_engine/answer_pools.py", "outputs/replay/results/replays_reported.json", "scripts/evaluation/generation_judging_stage.py", "freshcache/method/tiered_cache_model.py",
        "data/derived/round_table.jsonl", "data/sim_matrix_bgem3.npy", "outputs/general_audit/sample.json",
        "outputs/l2_operating_points/l2_threshold_request_level.csv", "configs/integrity/FILE_HASHES.csv"]
W = X.O / "work"
if not (W / "read_only_before.json").exists(): json.dump({p: X.fstat(inpath(p)) for p in READ}, open(W / "read_only_before.json", "w"), indent=1)
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
if MODE == "replay":
    X.log(f"[{now()}] Baselines replay")
    E = X.engine(); me, ea, RT, exp = E["me"], E["ea"], E["rounds"], E["exp"]; ST = X.C.streams(E); mod = PE.build(me)
    ea.CONFIG.setdefault(SC, dict(ea.CONFIG["SCALM"], l1="scalm_eqgate")); ea.CONFIG.setdefault(TT, dict(ea.CONFIG["TieredFixedTTL"], l1="eqgate_risk"))
    for v in (SC, TT): mod.VARIANTS[v] = (v, "risk")
    mod.NO_TTL_VARIANTS.add(SC); mod.L2_FLOOR_BY_VARIANT[TT] = 0.75
    G = {}
    G["no FreshCache answer-risk window or FreshCache L1 timestamp sets used (RISK_L1_VARIANTS and OLDEST_EVIDENCE_VARIANTS empty)"] = not mod.RISK_L1_VARIANTS and not mod.OLDEST_EVIDENCE_VARIANTS
    G["baseline names do not trigger FreshCache-family rules (l1_te_oldest_evidence False)"] = not any(me.l1_te_oldest_evidence(v) or mod.l1_te_oldest_evidence(v) for v in (SC, TT))
    G["SCALM_FCadmit_noTTL: l2='none', l3='none', ttl=None"] = (ea.CONFIG[SC]["l2"], ea.CONFIG[SC]["l3"], ea.CONFIG[SC]["ttl"]) == ("none", "none", None)
    G["TieredFixedTTL_FCadmit_L2075: l2='raw_ttl', l3='ttl_refetch', ttl='halflife' (fixed TTL), l2_reg_on_hit False"] = (ea.CONFIG[TT]["l2"], ea.CONFIG[TT]["l3"], ea.CONFIG[TT]["ttl"], ea.CONFIG[TT]["l2_reg_on_hit"]) == ("raw_ttl", "ttl_refetch", "halflife", False)
    CFG = {"generated_utc": now(), "configs": {v: {**ea.CONFIG[v], "temporal_mode": mod.VARIANTS[v][1], "no_ttl": v in mod.NO_TTL_VARIANTS, "l2_floor": mod.L2_FLOOR_BY_VARIANT.get(v)} for v in (SC, TT)},
           "constants_at_runtime": {"L1_SIM_THRESHOLD (candidate scan)": exp.L1_SIM_THRESHOLD, "_EQ_SIM_FLOOR (reuse cosine)": exp._EQ_SIM_FLOOR, "_EQ_JACCARD_MIN": exp._EQ_JACCARD_MIN,
                                    "L2_SIM_THRESHOLD (engine default, overridden to 0.75 for TT)": exp.L2_SIM_THRESHOLD, "_L2_EQ_SIM_FLOOR (FreshCache theta_L2)": exp._L2_EQ_SIM_FLOOR,
                                    "HALF_LIFE (s)": dict(exp.HALF_LIFE), "TIER_MULT": dict(exp.TIER_MULT),
                                    "TieredFixedTTL TTL (s) = HALF_LIFE x TIER_MULT": {c: {t: exp.HALF_LIFE.get(c, 86400.0) * m for t, m in exp.TIER_MULT.items()} for c in exp.HALF_LIFE}}}
    json.dump(CFG, open(X.O / "baseline_configs.json", "w"), indent=1, default=str)
    FR = json.load(open(X.CRR / "results/replays_reported.json"))
    ROWS, AUD, INV = [], {}, {}
    (X.O / "request_level").mkdir(exist_ok=True)
    for sname in ("primary", "heldout"):
        stream = ST[sname]; n = len(stream); S = E["support_of"](stream, RT); SS = set(S)
        for v in ("SCALM", "TieredFixedTTL"):   # baseline patcher sanity: the copy reproduces the frozen engine for the Table 2 scripts/baselines
            m0, p0 = me.replay(stream, RT, v); m1, p1 = mod.replay(stream, RT, v)
            G[f"{sname}: patched copy reproduces frozen engine for {v}"] = (m0 == m1 and p0 == p1)
            ch = sum(1 for q, o in p1 if q in SS and o == me.CHANGED); un = sum(1 for q, o in p1 if q in SS and o == me.UNCHANGED); f = FR[f"{sname}/{v}"]
            r = {"stream": sname, "method": REF[v], "config": v, **{k: f[k] for k in K5 + ("l1_hits", "l2_hits", "l3_hits", "changed", "unchanged")}, "source": "frozen replays_reported.json"}
            G[f"{sname}: {v} changed/unchanged equal frozen"] = (ch, un) == (f["changed"], f["unchanged"]); ROWS.append(r)
        for v in (SC, TT):
            tr = {}; m, per = mod.replay(stream, RT, v, trace=tr); per = dict(per)
            ch = sum(1 for q in S if per.get(q) == me.CHANGED); un = sum(1 for q in S if per.get(q) == me.UNCHANGED); sc = sum(1 for q in tr if tr[q]["tier"] in ("miss", "REAL_TIME_BYPASS"))
            ROWS.append({"stream": sname, "method": LAB[v], "config": v, "search_saved_pct": round(100 * (1 - sc / n), 4), "drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None,
                         "coverage_pct": round(100 * (ch + un) / len(S), 4), "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1),
                         "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"], "changed": ch, "unchanged": un, "source": "this run (in-memory baseline engine copy)"})
            # invariants re-checked on every admitted reuse
            REQ = {r["query_id"]: (t, r) for t, r in stream}; l1 = [q for q in tr if tr[q]["tier"] == "L1"]; l2 = [q for q in tr if tr[q]["tier"] == "L2"]
            inv = {"L1 hits": len(l1), "L2 hits": len(l2), "min admitted L1 cosine": min((tr[q]["l1_sim"] for q in l1), default=None), "min admitted L2 cosine": min((tr[q]["l2_sim"] for q in l2), default=None),
                   "L1 hits failing semantic_equivalent (cos>=0.80, Jaccard>=0.30, answer type, no strict subset)": sum(1 for q in l1 if not exp.semantic_equivalent(REQ[q][1]["query"], tr[q]["gen_query"], tr[q]["l1_sim"])),
                   "L1 hits failing entity match": sum(1 for q in l1 if not exp._entity_match(REQ[q][1]["query"], tr[q]["gen_query"])),
                   "L1 hits across freshness classes": sum(1 for q in l1 if tr[q]["entry_fc"] != tr[q]["fc"])}
            if v == TT:
                ttl = lambda fc, tier: exp.HALF_LIFE.get(fc, 86400.0) * exp.TIER_MULT[tier]
                inv["L1 hits older than the fixed answer TTL"] = sum(1 for q in l1 if tr[q]["l1_age"] > ttl(tr[q]["entry_fc"], "answer"))
                G[f"{sname}: {TT} min admitted L2 cosine >= 0.75"] = inv["min admitted L2 cosine"] is None or inv["min admitted L2 cosine"] >= 0.75
                G[f"{sname}: {TT} every L1 hit within its fixed TTL"] = inv["L1 hits older than the fixed answer TTL"] == 0
            else:
                G[f"{sname}: {SC} has no L2/L3 activity"] = m["l2_hits"] == 0 and m["l3_hits"] == 0 and len(l2) == 0
                G[f"{sname}: {SC} same-class only"] = inv["L1 hits across freshness classes"] == 0
            G[f"{sname}: {v} every L1 hit passes FreshCache answer admission"] = inv["L1 hits failing semantic_equivalent (cos>=0.80, Jaccard>=0.30, answer type, no strict subset)"] == 0 and inv["L1 hits failing entity match"] == 0 and (inv["min admitted L1 cosine"] is None or inv["min admitted L1 cosine"] >= 0.80)
            INV[f"{sname}/{v}"] = inv
            with open(X.O / f"request_level/{sname}_{v}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["request_id", "t", "freshness_class", "tier", "in_support_S", "drift_outcome", "admitted_l1_cosine", "admitted_l2_cosine", "l1_matched_query", "n_evidence_pages"])
                for t, r in stream:
                    q = r["query_id"]; x = tr[q]; o = per.get(q)
                    w.writerow([q, t, r["freshness_class"], x["tier"], q in SS, {me.CHANGED: "CHANGED", me.UNCHANGED: "UNCHANGED"}.get(o, "UNOBSERVABLE" if o is not None else ""),
                                x.get("l1_sim", ""), x.get("l2_sim", ""), x["gen_query"] if x["tier"] == "L1" else "", len(x.get("ev") or [])])
            if sname == "heldout":
                AUD[v] = {}
                for q in IDS:
                    x = tr[q]; t, r = REQ[q]
                    ev = x.get("ev") if x["tier"] != "REAL_TIME_BYPASS" else [(u["url_hash"], t) for u in r["urls"]]; gq = x.get("gen_query", SQ[q]["query"])
                    k, ctx, gq2 = X.key_for({"ev": ev, "gen_query": gq}, SQ[q]["query"]); AUD[v][q] = {"tier": x["tier"], "key": k, "gen_question": gq2, "context": ctx}
            X.log(sname, v, json.dumps({k: ROWS[-1][k] for k in K5 + ("l1_hits", "l2_hits", "l3_hits")}))
        f = FR[f"{sname}/FreshCache"]; ROWS.append({"stream": sname, "method": REF["FreshCache"], "config": "FreshCache", **{k: f[k] for k in K5 + ("l1_hits", "l2_hits", "l3_hits", "changed", "unchanged")}, "source": "frozen replays_reported.json (not rerun)"})
    for k, v in G.items(): X.log("GATE", k, v)
    if not all(G.values()): X.log("STOP: gate failed"); sys.exit(1)
    hl = X.Held(); todo = {}
    for A in AUD.values():
        for a in A.values():
            if hl.answer(a["key"]) is None: todo.setdefault(tuple(a["key"]), {"key": a["key"], "question": a["gen_question"], "context": a["context"]})
    X.wjl(X.O / "gpu/in/gen_tasks.jsonl", list(todo.values()))
    pickle.dump({"rows": ROWS, "gates": G, "inv": INV, "audit": {l: {q: {k: v for k, v in a.items() if k != "context"} for q, a in A.items()} for l, A in AUD.items()}}, open(W / "replay.pkl", "wb"))
    X.log("new answers needed:", len(todo)); sys.exit(0)
ST_ = pickle.load(open(W / "replay.pkl", "rb")); hl = X.Held()
if MODE == "items":
    items, miss = X.judge_items(hl, [(SQ[q]["query"], SQ[q]["gold"], a["key"]) for A in ST_["audit"].values() for q, a in A.items()])
    if miss: X.log(f"STOP: {len(miss)} answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("judge2 items:", len(items)); sys.exit(0)
# ---------------- score
L2_REQUESTS = {r["request_id"]: r for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")) if r["point"] == "theta_0.75"}
FRESH = {q: L2_REQUESTS[q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values())
L = {"FreshCache": {q: L2_REQUESTS[q]["answer_label"] for q in IDS}}
for v, A in ST_["audit"].items(): L[v] = {q: (hl.labels(SQ[q]["query"], SQ[q]["gold"], A[q]["key"])[0] or "UNRESOLVED") for q in IDS}
wa = lambda L_, q: FRESH[q] == "CORRECT" and L_[q] == "WRONG"
QR = []
for v in (SC, TT, "FreshCache"):
    c = collections.Counter(L[v].values()); w = sum(1 for q in IDS if wa(L[v], q)); row = {"method": LAB.get(v, REF["FreshCache"]), "config": v, "N": len(IDS), "fresh_correct": FCN, "correct": c["CORRECT"],
         "accuracy_pct": round(100 * c["CORRECT"] / len(IDS), 2), "wrong": c["WRONG"], "abstain": c["ABSTAIN"], "unresolved": c["UNRESOLVED"], "WAI": w, "WAI_pct": round(100 * w / len(IDS), 2), "C_WAI_pct": round(100 * w / FCN, 2),
         "audit_tiers": json.dumps(dict(collections.Counter(ST_["audit"][v][q]["tier"] for q in IDS))) if v in ST_["audit"] else ""}
    if v != "FreshCache":
        b = sum(1 for q in IDS if wa(L[v], q) and not wa(L["FreshCache"], q)); a_ = sum(1 for q in IDS if wa(L["FreshCache"], q) and not wa(L[v], q))
        cb = sum(1 for q in IDS if L[v][q] == "CORRECT" and L["FreshCache"][q] != "CORRECT"); ca = sum(1 for q in IDS if L["FreshCache"][q] == "CORRECT" and L[v][q] != "CORRECT")
        row.update({"WAI_FreshCache_only": a_, "WAI_baseline_only": b, "WAI_mcnemar_p_vs_FreshCache": X.mcnemar(a_, b), "correct_FreshCache_only": ca, "correct_baseline_only": cb, "correct_mcnemar_p_vs_FreshCache": X.mcnemar(ca, cb)})
    QR.append(row)
ROWS = ST_["rows"]; cols = ["stream", "method", "config"] + list(K5) + ["l1_hits", "l2_hits", "l3_hits", "changed", "unchanged", "source"]
with open(X.O / "baseline_results.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(ROWS)
with open(X.O / "answer_quality_results.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in QR for k in r))); w.writeheader(); w.writerows(QR)
with open(X.O / "request_level/heldout_audit_labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["request_id", "fresh_label", "FreshCache_label", f"{SC}_tier", f"{SC}_label", f"{TT}_tier", f"{TT}_label"])
    for q in IDS: w.writerow([q, FRESH[q], L["FreshCache"][q], ST_["audit"][SC][q]["tier"], L[SC][q], ST_["audit"][TT][q]["tier"], L[TT][q]])
BEF = json.load(open(W / "read_only_before.json")); UNCH = {p: X.fstat(inpath(p)) == v for p, v in BEF.items()}
json.dump({"generated_utc": now(), "gates": ST_["gates"], "invariants": ST_["inv"], "quality": QR, "read_only_inputs_unchanged": UNCH}, open(W / "score_summary.json", "w"), indent=1, default=str)
X.log("scored; read-only inputs unchanged:", all(UNCH.values()))
fm = lambda v, d=2: "—" if v in (None, "") else f"{float(v):,.{d}f}"
for sname in ("primary", "heldout"):
    print(f"\n{sname}"); [print(f"  {r['method']:<80} {fm(r['search_saved_pct'])} {fm(r['drift_pct'])} {fm(r['coverage_pct'])} {fm(r['fetch_per_1k'],1)} {fm(r['gen_per_1k'],1)}  L1 {r['l1_hits']} L2 {r['l2_hits']} L3 {r['l3_hits']}") for r in ROWS if r["stream"] == sname]
for r in QR: print(json.dumps(r))
