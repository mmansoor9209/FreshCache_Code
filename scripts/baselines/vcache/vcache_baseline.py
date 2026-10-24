"""vCache-style baseline: our reimplementation of vCache's per-entry learned-threshold policy (Schroeder et al., ICLR 2026), not
the complete published system; it uses proxy correctness feedback and does not test the published error guarantee.

Held-out metrics for the existing vCache comparator ("vCache (proxy feedback)"), run unchanged.
Uses scripts/baselines/vcache/vcache_reference_replay.py reference_replay(stream, rounds=..., delta=0.02) exactly as called by
scripts/experiments/main/exact_ttl_vcache_replay.py: released vCache decision math (vcache_policy._select_action_explore), RNG seed 42,
unobservable verdicts not recorded ("no_update"), argmax over every entry (no floor), REAL_TIME = full pipeline + insert. The held-out stream is
replayed once with that code; the call is gated by first reproducing the frozen primary row.
Modes: replay | items | score. All writes under outputs/baselines/vcache."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, json, os, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache"))
import vcache_helpers as X
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); MODE = sys.argv[1]; W = X.O / "work"; DELTA = 0.02
READ = ["scripts/baselines/vcache/vcache_reference_replay.py", "scripts/experiments/main/exact_ttl_vcache_replay.py", "scripts/baselines/vcache/vcache_policy.py",
        "outputs/replay/results/baseline_measurement_reported.json", "freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/engine_loader.py",
        "outputs/replay/results/replays_reported.json", "freshcache/method/tiered_cache_model.py", "data/derived/round_table.jsonl", "data/sim_matrix_bgem3.npy",
        "outputs/general_audit/sample.json", "outputs/l2_operating_points/l2_threshold_request_level.csv", "outputs/concurrency_cost/component_costs.json",
        "scripts/experiments/cost_and_latency/cold_lookup_timing.py", "configs/integrity/FILE_HASHES.csv"]
if not (W / "read_only_before.json").exists(): json.dump({p: X.fstat(inpath(p)) for p in READ}, open(W / "read_only_before.json", "w"), indent=1)
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)


def load_rf(E):
    """Import the frozen vCache module exactly as exact_ttl_vcache_replay.py does (same paths, identity assertion)."""
    sys.path.insert(1, str(REPO_ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/exact_ttl_and_l1_only")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache"))
    import vcache_reference_replay as RF
    assert "scripts/baselines/vcache/vcache_reference_replay.py" in RF.__file__ and RF.me is E["me"], (RF.__file__, RF.me)
    return RF


def entries_from(stream, events):
    """Rebuild the cache entry list from the reference events: an entry is inserted on miss_realtime, miss_cold, and miss_explore with c == 0."""
    ent = []
    for (t, r), e in zip(stream, events):
        assert e["query_id"] == r["query_id"]
        if e["action"] in ("miss_realtime", "miss_cold") or (e["action"] == "miss_explore" and e["feedback_label"] == 0):
            ent.append({"q": r["query"], "qid": r["query_id"], "t": t, "urls": [u["url_hash"] for u in r["urls"]]})
    return ent


if MODE == "replay":
    X.log(f"[{now()}] vCache held-out replay (frozen reference_replay, delta={DELTA})")
    E = X.engine(); me, RT = E["me"], E["rounds"]; ST = X.C.streams(E); RF = load_rf(E); G = {}
    G["vCache module constants unchanged (RNG seed 42, unobservable policy 'no_update')"] = RF.RNG_SEED == 42 and RF.UNOBS_POLICY == "no_update"
    REF = json.load(open(X.CRR / "results/baseline_measurement_reported.json"))["vCache_delta0.02"]; OUT = {}
    for sname in ("primary", "heldout"):
        S = ST[sname]; SUP = E["support_of"](S, RT); n = len(S)
        m, o, ev = RF.reference_replay(S, rounds=RT, delta=DELTA); sc = RF.score(o, SUP)
        row = {"search_saved_pct": round(m["search_saved_pct"], 4), "drift_pct": None if sc["drift_pct"] is None else round(sc["drift_pct"], 4), "coverage_pct": round(sc["coverage_pct"], 4),
               "determinable": sc["determinable"], "changed": sc["changed"], "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1)}
        if sname == "primary":
            G["primary: this call reproduces the frozen paper vCache row exactly (baseline_measurement_reported.json)"] = row == REF
        ent = entries_from(S, ev)
        G[f"{sname}: rebuilt entry list size equals the replay's cache_entries ({m['cache_entries']})"] = len(ent) == m["cache_entries"]
        # rebuilt hit evidence must reproduce the replay's own hit outcomes (validates the audit-key construction)
        P = dict(o); bad = 0; nh = 0
        for (t, r), e in zip(S, ev):
            if e["action"] != "hit": continue
            nh += 1; en = ent[e["matched_entry"]]; labs = [me.outcome(u, en["t"], t, RT) for u in en["urls"]]
            lab = me.CHANGED if me.CHANGED in labs else (me.UNCHANGED if me.UNCHANGED in labs else me.UNOBS)
            bad += lab != P[r["query_id"]]
        G[f"{sname}: rebuilt hit entries reproduce every hit's replay outcome ({nh} hits)"] = bad == 0 and nh == m["hits"]
        G[f"{sname}: operation identities (generations = searches = misses; hits + misses = n)"] = m["generations"] == m["search_calls"] == m["misses"] and m["hits"] + m["misses"] == n
        OUT[sname] = {"metrics": m, "score": sc, "row": row, "events": ev, "entries": ent, "n": n, "nonrt": sum(1 for _, r in S if r["freshness_class"] != "REAL_TIME")}
        X.log(sname, json.dumps({**row, "hits": m["hits"], "feedback_updates": m["feedback_updates"], "cache_entries": m["cache_entries"]}))
    for k, v in G.items(): X.log("GATE", k, v)
    if not all(G.values()): X.log("STOP: gate failed"); sys.exit(1)
    (X.O / "request_level").mkdir(exist_ok=True)
    for sname in ("primary", "heldout"):
        d = OUT[sname]; cols = list(d["events"][0]) + ["matched_entry_query_id", "matched_entry_time"]
        with open(X.O / f"request_level/{sname}_vCache_delta0.02_events.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for e in d["events"]:
                en = d["entries"][e["matched_entry"]] if e["action"] == "hit" else None
                w.writerow({**e, "matched_entry_query_id": en["qid"] if en else "", "matched_entry_time": en["t"] if en else ""})
    H = ST["heldout"]; REQ = {r["query_id"]: (t, r, e) for (t, r), e in zip(H, OUT["heldout"]["events"])}; AUD = {}; hl = X.Held(); todo = {}
    for q in IDS:
        t, r, e = REQ[q]
        if e["action"] == "hit": en = OUT["heldout"]["entries"][e["matched_entry"]]; ev, gq, tier = [(u, en["t"]) for u in en["urls"]], en["q"], "L1"
        else: ev, gq, tier = [(u["url_hash"], t) for u in r["urls"]], r["query"], ("REAL_TIME" if r["freshness_class"] == "REAL_TIME" else "miss")
        k, ctx, gq2 = X.key_for({"ev": ev, "gen_query": gq}, SQ[q]["query"]); AUD[q] = {"tier": tier, "action": e["action"], "key": k}
        if hl.answer(k) is None: todo.setdefault(tuple(k), {"key": k, "question": gq2, "context": ctx})
    X.wjl(X.O / "gpu/in/gen_tasks.jsonl", list(todo.values()))
    pickle.dump({"gates": G, "out": {s: {k: v for k, v in d.items() if k not in ("events", "entries")} for s, d in OUT.items()}, "audit": AUD}, open(W / "replay.pkl", "wb"))
    X.log("new answers needed:", len(todo)); sys.exit(0)

ST_ = pickle.load(open(W / "replay.pkl", "rb")); hl = X.Held()
if MODE == "items":
    items, miss = X.judge_items(hl, [(SQ[q]["query"], SQ[q]["gold"], a["key"]) for q, a in ST_["audit"].items()])
    if miss: X.log(f"STOP: {len(miss)} answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("judge2 items:", len(items)); sys.exit(0)
# ---------------- score
G = dict(ST_["gates"]); H = ST_["out"]["heldout"]; m, n, nonrt = H["metrics"], H["n"], H["nonrt"]
cold = json.load(open(W / "local_cold_vCache.json")); ctl = json.load(open((OUTPUT_ROOT / "baselines/scalm/work/local_cold_FreshCache_control.json")))
CO = json.load(open(X.COST_DIR / "component_costs.json"))["coefficients"]
c = {"c_search_local": CO["c_search_ms"]["local"], "c_search_live_p50": CO["c_search_ms"]["live_p50"], "c_fetch_local": CO["c_fetch_ms"]["local"], "c_fetch_live_p50": CO["c_fetch_ms"]["live_p50"],
     "c_gen": CO["c_gen_ms"]["value"], "c_embed": CO["c_embed_ms"]["value"]}
def breakdown(name, s, f, g, emb, lk):
    out = []
    for sc in ("live_p50", "local"):
        terms = {"search": s / n * c[f"c_search_{sc}"], "fetch": f / n * c[f"c_fetch_{sc}"], "generation": g / n * c["c_gen"], "embedding": emb / n * c["c_embed"], "local_lookup_overhead": lk}
        tot = sum(terms.values()); chk = X.cost(s, f, g, n, emb, lk)[f"{sc}_ms_per_request"]
        out.append({"method": name, "scenario": sc, "searches": s, "fetches": f, "generations": g, "embeddings": emb, "n": n, **{f"{k}_ms_per_request": round(v, 4) for k, v in terms.items()},
                    "total_ms_per_request": round(tot, 3), "total_s_per_request": round(tot / 1000, 4), "matches_shared_cost_function": abs(round(tot, 3) - chk) < 1e-3})
    return out
FC = json.load(open(X.CRR / "results/replays_reported.json"))["heldout/FreshCache"]; fs = round(n * (1 - FC["search_saved_pct"] / 100))
CB = breakdown("vCache [4] (frozen comparator, delta 0.02)", m["search_calls"], m["fetches"], m["generations"], nonrt, cold["lookup_ms_per_request_used"]) + \
     breakdown("FreshCache primary (frozen; cost-model lookup)", fs, FC["fetches"], FC["generations"], nonrt, X.LOOKUP["FreshCache"])
fl = next(r for r in CB if r["method"].startswith("FreshCache") and r["scenario"] == "live_p50"); fo = next(r for r in CB if r["method"].startswith("FreshCache") and r["scenario"] == "local")
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); G["FreshCache live modeled cost reproduces the recorded reference with the same calculation"] = REF.check("cost/FreshCache_heldout/live_p50_ms_per_request_2dp", round(fl["total_ms_per_request"], 2))
G["FreshCache local modeled cost equals the recorded reference"] = REF.check("cost/FreshCache_heldout/local_ms_per_request_2dp", round(fo["total_ms_per_request"], 2))
G["every cost row equals the shared cost-model cost function"] = all(r["matches_shared_cost_function"] for r in CB)
G["vCache cold-timing pass reproduces the same held-out operation counts"] = cold["ops_during_timed_pass"] == {k: m[k] for k in ("search_calls", "fetches", "generations", "hits")}
L2_REQUESTS = {r["request_id"]: r for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")) if r["point"] == "theta_0.75"}
FRESH = {q: L2_REQUESTS[q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values()); LF = {q: L2_REQUESTS[q]["answer_label"] for q in IDS}
LV = {q: (hl.labels(SQ[q]["query"], SQ[q]["gold"], ST_["audit"][q]["key"])[0] or "UNRESOLVED") for q in IDS}
G["audit: frozen held-out audit requests, all with L2-sweep fresh + FreshCache labels; fresh-CORRECT denominator (run-time reference)"] = DS.check("audit_sample_requests", len(IDS)) and all(q in L2_REQUESTS for q in IDS) and REF.check("heldout_audit/fresh_correct", FCN)
G["audit: no UNRESOLVED vCache labels"] = "UNRESOLVED" not in LV.values()
wa = lambda L, q: FRESH[q] == "CORRECT" and L[q] == "WRONG"; cV = collections.Counter(LV.values()); cF = collections.Counter(LF.values())
wV = sum(1 for q in IDS if wa(LV, q)); wF = sum(1 for q in IDS if wa(LF, q)); fo_ = sum(1 for q in IDS if wa(LF, q) and not wa(LV, q)); vo = sum(1 for q in IDS if wa(LV, q) and not wa(LF, q))
QR = [{"method": "vCache [4]", "N": len(IDS), "fresh_CORRECT_denominator": FCN, "correct": cV["CORRECT"], "correct_pct": round(100 * cV["CORRECT"] / len(IDS), 2), "wrong": cV["WRONG"], "abstain": cV["ABSTAIN"],
       "unresolved": cV["UNRESOLVED"], "WAI": wV, "WAI_pct": round(100 * wV / len(IDS), 2), "C_WAI_pct": round(100 * wV / FCN, 2), "FreshCache_only_WAI": fo_, "vCache_only_WAI": vo, "WAI_exact_mcnemar_p": X.mcnemar(fo_, vo),
       "audit_routes": json.dumps(dict(collections.Counter(a["action"] for a in ST_["audit"].values())))},
      {"method": "FreshCache primary (frozen)", "N": len(IDS), "fresh_CORRECT_denominator": FCN, "correct": cF["CORRECT"], "correct_pct": round(100 * cF["CORRECT"] / len(IDS), 2), "wrong": cF["WRONG"], "abstain": cF["ABSTAIN"],
       "unresolved": cF["UNRESOLVED"], "WAI": wF, "WAI_pct": round(100 * wF / len(IDS), 2), "C_WAI_pct": round(100 * wF / FCN, 2)}]
G["FreshCache audit reproduces the frozen correct / WAI / C-WAI (run-time reference)"] = REF.check("heldout_audit/FreshCache_primary/CORRECT", cF["CORRECT"]) and REF.check("heldout_audit/FreshCache_primary/WAI", wF) and QR[1]["C_WAI_pct"] == round(100 * REF.value("heldout_audit/FreshCache_primary/WAI") / REF.value("heldout_audit/fresh_correct"), 2)
BEF = json.load(open(W / "read_only_before.json")); UNCH = {p: X.fstat(inpath(p)) == v for p, v in BEF.items()}; G["all read-only source files unchanged"] = all(UNCH.values())
for k, v in G.items(): X.log("GATE", k, v)
lv = next(r for r in CB if r["method"].startswith("vCache") and r["scenario"] == "live_p50"); lo = next(r for r in CB if r["method"].startswith("vCache") and r["scenario"] == "local")
TABLE_ROWS = [{"method": "vCache [4]", "search_saved_pct": round(m["search_saved_pct"], 2), "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1), "cache_hits": m["hits"],
       "cost_s_per_request_live": lv["total_s_per_request"], "cost_s_per_request_local": lo["total_s_per_request"], "C_WAI_pct": QR[0]["C_WAI_pct"], "correct": cV["CORRECT"], "WAI": wV, "WAI_pct": QR[0]["WAI_pct"],
       "drift_pct": H["row"]["drift_pct"], "coverage_pct": H["row"]["coverage_pct"], "searches": m["search_calls"], "fetches": m["fetches"], "generations": m["generations"], "feedback_updates": m["feedback_updates"], "cache_entries": m["cache_entries"]}]
for fn, rows in (("vcache_table3_metrics.csv", TABLE_ROWS), ("vcache_answer_quality.csv", QR), ("vcache_cost_breakdown.csv", CB)):
    with open(X.O / fn, "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
with open(X.O / "request_level/heldout_audit_labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["request_id", "fresh_label", "FreshCache_label", "vCache_action", "vCache_label"])
    for q in IDS: w.writerow([q, FRESH[q], LF[q], ST_["audit"][q]["action"], LV[q]])
json.dump({"generated_utc": now(), "gates": G, "table3": TABLE_ROWS, "quality": QR, "cost": CB, "coefficients": c, "cold_timing": cold, "freshcache_same_day_control": ctl, "primary_row": ST_["out"]["primary"]["row"],
           "heldout": {k: v for k, v in H.items()}, "read_only_unchanged": UNCH}, open(W / "table3_summary.json", "w"), indent=1, default=str)
X.log("ALL GATES", all(G.values())); print(json.dumps({"table3": TABLE_ROWS, "quality": QR}, indent=1)); [print(json.dumps(r)) for r in CB]
