"""SCALM-style baseline: our reimplementation of SCALM's admission policy (Li et al., IWQoS 2024), not the complete recorded
system; it omits SCALM's hierarchical cache structure and its eviction policy.

Held-out Table-3 metrics for the SCALM-style baseline (no TTL) with matched FreshCache answer admission (config SCALM_FCadmit_noTTL).
The policy itself is not recomputed; only (a) the fresh-process cold lookup timing needed for the cost model and (b) a re-derivation of the
stored audit labels from the frozen answer/judge pools (consistency check) are computed. All writes under outputs/baselines/scalm."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, json, os, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/scalm"))
import scalm_helpers as X
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); B = (OUTPUT_ROOT / "baselines/matched_admission"); V = "SCALM_FCadmit_noTTL"
SRC = {"held-out operation counts (row)": "outputs/baselines/matched_admission/baseline_results.csv", "held-out request-level routes": "outputs/baselines/matched_admission/request_level/heldout_SCALM_FCadmit_noTTL.csv",
       "baseline configuration": "outputs/baselines/matched_admission/baseline_configs.json", "baseline gates + invariants": "outputs/baselines/matched_admission/work/score_summary.json", "audit keys (replay)": "outputs/baselines/matched_admission/work/replay.pkl",
       "stored audit labels": "outputs/baselines/matched_admission/request_level/heldout_audit_labels.csv", "stored audit summary": "outputs/baselines/matched_admission/answer_quality_results.csv", "baseline engine patcher": "scripts/baselines/matched_admission/matched_admission_engine_patch.py",
       "cost coefficients + FreshCache lookup": "outputs/concurrency_cost/component_costs.json", "cost-model cold-timing protocol": "scripts/experiments/cost_and_latency/cold_lookup_timing.py",
       "fresh labels + FreshCache audit labels": "outputs/l2_operating_points/l2_threshold_request_level.csv", "audit population": "outputs/general_audit/sample.json",
       "FreshCache held-out counts": "outputs/replay/results/replays_reported.json"}
BEF = {p: X.fstat(inpath(p)) for p in SRC.values()}
G = {}
# ---------------- 1. operation counts (re-read, not recomputed)
row = next(r for r in csv.DictReader(open(B / "baseline_results.csv", encoding="utf-8")) if r["stream"] == "heldout" and r["config"] == V)
E = X.engine(); H = X.C.streams(E)["heldout"]; n = len(H); URLS = {r["query_id"]: len(r["urls"]) for t, r in H}; FCL = {r["query_id"]: r["freshness_class"] for t, r in H}
RL = list(csv.DictReader(open(B / "request_level/heldout_SCALM_FCadmit_noTTL.csv", encoding="utf-8"))); tiers = collections.Counter(r["tier"] for r in RL)
search = sum(1 for r in RL if r["tier"] in ("miss", "REAL_TIME_BYPASS")); fetch = sum(URLS[r["request_id"]] for r in RL if r["tier"] in ("miss", "REAL_TIME_BYPASS")); gen = search
nonrt = sum(1 for q in FCL if FCL[q] != "REAL_TIME")
G["request-level routes cover the held-out stream in frozen order"] = [r["request_id"] for r in RL] == [r["query_id"] for t, r in H]
G["request-level counts reproduce the stored held-out row (saved, fetch/1k, gen/1k)"] = (round(100 * (1 - search / n), 4), round(1000 * fetch / n, 1), round(1000 * gen / n, 1)) == (float(row["search_saved_pct"]), float(row["fetch_per_1k"]), float(row["gen_per_1k"]))
G["answer-cache only: no L2/L3 tiers in the request-level routes"] = set(tiers) <= {"L1", "miss", "REAL_TIME_BYPASS"} and int(row["l2_hits"]) == 0 and int(row["l3_hits"]) == 0
cold = json.load(open(X.O / "work/local_cold_SCALM.json")); ctl = json.load(open(X.O / "work/local_cold_FreshCache_control.json"))
G["cold-timing pass reproduces the same SCALM operation counts"] = (cold["ops_during_timed_pass"]["search_calls"], cold["ops_during_timed_pass"]["fetches"], cold["ops_during_timed_pass"]["generations"], cold["ops_during_timed_pass"]["l1_hits"]) == (search, fetch, gen, tiers["L1"])
# ---------------- 2. cost model (component-cost model, identical formula and coefficients)
CO = json.load(open(X.COST_DIR / "component_costs.json"))["coefficients"]; c = {"c_search_local": CO["c_search_ms"]["local"], "c_search_live_p50": CO["c_search_ms"]["live_p50"], "c_fetch_local": CO["c_fetch_ms"]["local"],
     "c_fetch_live_p50": CO["c_fetch_ms"]["live_p50"], "c_gen": CO["c_gen_ms"]["value"], "c_embed": CO["c_embed_ms"]["value"]}
def breakdown(name, s, f, g, emb, lk):
    out = []
    for sc in ("local", "live_p50"):
        terms = {"search": s / n * c[f"c_search_{sc}"], "fetch": f / n * c[f"c_fetch_{sc}"], "generation": g / n * c["c_gen"], "embedding": emb / n * c["c_embed"], "local_lookup_overhead": lk}
        tot = sum(terms.values()); chk = X.cost(s, f, g, n, emb, lk)[f"{sc}_ms_per_request"]
        out.append({"method": name, "scenario": sc, "searches": s, "fetches": f, "generations": g, "embeddings": emb, "n": n, **{f"{k}_ms_per_request": round(v, 4) for k, v in terms.items()},
                    "total_ms_per_request": round(tot, 3), "total_s_per_request": round(tot / 1000, 4), "check_vs_shared_cost_function": abs(round(tot, 3) - chk) < 1e-3})
    return out
FC = json.load(open(X.CRR / "results/replays_reported.json"))["heldout/FreshCache"]; fs = round(n * (1 - FC["search_saved_pct"] / 100))
CB = breakdown("SCALM-style (no TTL) + matched FC admission", search, fetch, gen, nonrt, cold["cold_ms_per_request"]) + breakdown("FreshCache primary (frozen; cost-model lookup)", fs, FC["fetches"], FC["generations"], nonrt, X.LOOKUP["FreshCache"])
G["FreshCache held-out cost reproduces cost-model / CategoryAware-frontier (local, live ms/request; run-time reference)"] = [round(r["total_ms_per_request"], 2) for r in CB if r["method"].startswith("FreshCache")] == [REF.value(f"cost/FreshCache_heldout/{k}_ms_per_request_2dp") for k in ("local", "live_p50")]
G["every cost row equals the shared cost-model cost function"] = all(r["check_vs_shared_cost_function"] for r in CB)
G[f"same-session FreshCache cold control within 10% of cost-model's frozen value ({ctl['cold_ms_per_request']} vs {X.LOOKUP['FreshCache']} ms)"] = abs(ctl["cold_ms_per_request"] / X.LOOKUP["FreshCache"] - 1) < 0.10
# ---------------- 3. audit (stored labels, re-derived from frozen pools)
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
L2_REQUESTS = {r["request_id"]: r for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")) if r["point"] == "theta_0.75"}
FRESH = {q: L2_REQUESTS[q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values()); LF = {q: L2_REQUESTS[q]["answer_label"] for q in IDS}
ST = {r["request_id"]: r for r in csv.DictReader(open(B / "request_level/heldout_audit_labels.csv", encoding="utf-8"))}
AUD = pickle.load(open(B / "work/replay.pkl", "rb"))["audit"][V]; hl = X.Held(); missing = [q for q in IDS if hl.answer(AUD[q]["key"]) is None]
LS = {q: (hl.labels(SQ[q]["query"], SQ[q]["gold"], AUD[q]["key"])[0] or "UNRESOLVED") for q in IDS}
G["audit population = the frozen held-out audit requests (sample.json), same as FreshCache's L2-sweep audit"] = DS.check("audit_sample_requests", len(IDS)) and set(IDS) == set(ST) and all(q in L2_REQUESTS for q in IDS)
G["no missing SCALM audit answers (no generation needed)"] = not missing
G["re-derived labels equal the stored SCALM labels (all audit requests)"] = all(LS[q] == ST[q][f"{V}_label"] for q in IDS)
G["FreshCache labels in the stored file equal L2-sweep (all audit requests)"] = all(ST[q]["FreshCache_label"] == LF[q] and ST[q]["fresh_label"] == FRESH[q] for q in IDS)
wa = lambda L, q: FRESH[q] == "CORRECT" and L[q] == "WRONG"; cS = collections.Counter(LS.values()); cF = collections.Counter(LF.values())
wS = sum(1 for q in IDS if wa(LS, q)); wF = sum(1 for q in IDS if wa(LF, q)); fo = sum(1 for q in IDS if wa(LF, q) and not wa(LS, q)); so = sum(1 for q in IDS if wa(LS, q) and not wa(LF, q))
QR = [{"method": "SCALM-style (no TTL) + matched FC admission", "N": len(IDS), "fresh_CORRECT_denominator": FCN, "correct": cS["CORRECT"], "correct_pct": round(100 * cS["CORRECT"] / len(IDS), 2), "wrong": cS["WRONG"],
       "abstain": cS["ABSTAIN"], "unresolved": cS["UNRESOLVED"], "WAI": wS, "WAI_pct": round(100 * wS / len(IDS), 2), "C_WAI_pct": round(100 * wS / FCN, 2), "FreshCache_only_WAI": fo, "SCALM_only_WAI": so, "WAI_exact_mcnemar_p": X.mcnemar(fo, so),
       "audit_tiers": json.dumps(dict(collections.Counter(AUD[q]["tier"] for q in IDS)))},
      {"method": "FreshCache primary (frozen)", "N": len(IDS), "fresh_CORRECT_denominator": FCN, "correct": cF["CORRECT"], "correct_pct": round(100 * cF["CORRECT"] / len(IDS), 2), "wrong": cF["WRONG"], "abstain": cF["ABSTAIN"],
       "unresolved": cF["UNRESOLVED"], "WAI": wF, "WAI_pct": round(100 * wF / len(IDS), 2), "C_WAI_pct": round(100 * wF / FCN, 2)}]
SA = next(r for r in csv.DictReader(open(B / "answer_quality_results.csv", encoding="utf-8")) if r["config"] == V)
G["re-derived audit summary equals the stored SCALM summary (correct, WAI, C-WAI, McNemar)"] = (QR[0]["correct"], QR[0]["WAI"], QR[0]["C_WAI_pct"], fo, so) == (int(SA["correct"]), int(SA["WAI"]), float(SA["C_WAI_pct"]), int(SA["WAI_FreshCache_only"]), int(SA["WAI_baseline_only"]))
# ---------------- 4. baseline logic unchanged
CFG = json.load(open(B / "baseline_configs.json"))["configs"][V]; SS = json.load(open(B / "work/score_summary.json"))
G["config unchanged: l1=scalm_eqgate, l2=none, l3=none, ttl=None, no_ttl=True, no L2 floor"] = (CFG["l1"], CFG["l2"], CFG["l3"], CFG["ttl"], CFG["no_ttl"], CFG["l2_floor"]) == ("scalm_eqgate", "none", "none", None, True, None)
G["earlier run's SCALM gates all True (no L2/L3, same-class, every hit passes FC admission, equals matched-admission no-TTL row)"] = all(v for k, v in SS["gates"].items() if V in k or "FreshCache answer-risk" in k or "FreshCache-family" in k)
inv = SS["invariants"][f"heldout/{V}"]
G["held-out invariants: every L1 hit cos >= 0.80 and passes semantic_equivalent + entity match; 0 cross-class hits"] = inv["min admitted L1 cosine"] >= 0.80 and inv["L1 hits failing semantic_equivalent (cos>=0.80, Jaccard>=0.30, answer type, no strict subset)"] == 0 and inv["L1 hits failing entity match"] == 0 and inv["L1 hits across freshness classes"] == 0
UNCH = {p: X.fstat(inpath(p)) == BEF[p] for p in SRC.values()}; G["all reused source files unchanged (sha256, mtime, size)"] = all(UNCH.values())
for k, v in G.items(): X.log("GATE", k, v)
# ---------------- outputs
cl = next(r for r in CB if r["method"].startswith("SCALM") and r["scenario"] == "local"); cv = next(r for r in CB if r["method"].startswith("SCALM") and r["scenario"] == "live_p50")
TABLE_ROWS = [{"method": "SCALM-style [3] (no TTL) + matched FC admission", "search_saved_pct": round(100 * (1 - search / n), 2), "fetch_per_1k": round(1000 * fetch / n, 1), "gen_per_1k": round(1000 * gen / n, 1),
       "cost_s_per_request_local": cl["total_s_per_request"], "cost_s_per_request_live_p50": cv["total_s_per_request"], "C_WAI_pct": QR[0]["C_WAI_pct"], "WAI": wS, "correct": cS["CORRECT"], "fresh_CORRECT_denominator": FCN}]
for fn, rows in (("scalm_table3_metrics.csv", TABLE_ROWS), ("scalm_answer_quality.csv", QR), ("scalm_cost_breakdown.csv", CB)):
    with open(X.O / fn, "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
json.dump({"generated_utc": now(), "gates": G, "sources": SRC, "source_hashes": {p: {**BEF[p], "unchanged": UNCH[p]} for p in SRC.values()}, "counts": {"n": n, "nonrt": nonrt, "tiers": dict(tiers), "searches": search, "fetches": fetch, "generations": gen},
           "cold_timing": cold, "freshcache_control_timing": ctl, "coefficients": c, "table3": TABLE_ROWS, "quality": QR, "cost": CB}, open(X.O / "work/table3_summary.json", "w"), indent=1, default=str)
X.log("ALL GATES", all(G.values())); print(json.dumps({"table3": TABLE_ROWS, "quality": QR}, indent=1)); [print(json.dumps(r)) for r in CB]
