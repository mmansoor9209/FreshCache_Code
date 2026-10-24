"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

SemanticTTL frontier, stage 1 (validation only; no held-out data is touched).
  - integrity of every reused input against the frozen hashes in configs/integrity/FILE_HASHES.csv
  - reproduction gates: FreshCache validation replay, the existing (0.60, 1/2) replay, and the reported (0.60, 1/2) validation answer keys
  - 20 SemanticTTL configs: operational replay (mixed_age_replay 'SemanticTTL') + provenance replay (quality_selection_sample.sttl_replay), cross-checked per request
  - answer keys for the frozen screened validation set; generation tasks for keys absent from every read-only pool
  - selection rules written and hashed BEFORE any answer is generated or scored
Writes under outputs/semanticttl_frontier/ (logs/, frozen_selection/selection_rules.json and the per-config outputs).
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import csv, datetime, gzip, json, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
sys.path.insert(0, __import__("os").path.dirname(__file__))
import semanticttl_frontier_helpers as X

now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
LOG = {"stage": "validation_replay", "started_utc": now()}
# ---------------------------------------------------------------- integrity
INT = X.integrity()
bad = [p for p, v in INT.items() if v["frozen_sha256"] is not None and not v["match"]]
LOG["integrity"] = INT; LOG["integrity_mismatches"] = bad
if bad:
    json.dump(LOG, open(X.STTL_FRONTIER_DIR / "logs/validation_replay_STOPPED_integrity.json", "w"), indent=1); sys.exit(f"STOP: frozen input changed: {bad}")

E = X.engine(); exp, p2, me = E["exp"], E["p2"], E["me"]; Q2 = X.q2()
ST = X.C.streams(E); V = ST["validation"]; assert DS.check("validation_requests", len(V)), len(V)
assert not any((r.get("cluster_id") or r["query_id"]) in set(E["split"]["test_clusters"]) for _, r in V)
S = E["support_of"](V, E["rounds"])

def opmetrics(variant, theta=None, kappa=None):
    if theta is None:
        row, per, tr = X.C.metrics(E, V, variant, S)
    else:
        with X.patched(theta, kappa):
            row, per, tr = X.C.metrics(E, V, variant, S)
    row["search_calls"] = sum(1 for q in tr if tr[q]["tier"] in ("miss", "REAL_TIME_BYPASS"))
    row["searches_per_1k"] = round(1000 * row["search_calls"] / len(V), 1)
    row["determinable"] = row["changed"] + row["unchanged"]
    return row, per, tr

# ---------------------------------------------------------------- gates
G = {}
fc_row, fc_per, fc_tr = opmetrics("FreshCache")
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); G["freshcache_validation_saved_equals_reference"] = REF.check("reported_engine/validation/FreshCache/search_saved_pct", fc_row["search_saved_pct"])
ref_row, ref_per, ref_tr = opmetrics("SemanticTTL", 0.60, 0.5)
pre = json.load(open(OUTPUT_ROOT / "quality_selection/quality_selection_prereg.json"))
pre060 = next(c for c in pre["candidates"] if c["arm"] == "sttl_t0.60_k1_2")
G["sttl_060_k05_saved_equals_prereg_grid"] = ref_row["search_saved_pct"] == pre060["saved"]
QS = [json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))]
EVSET = {r["query_id"] for r in csv.DictReader(open((OUTPUT_ROOT / "quality_selection/q_verification.csv"))) if r.get("evidence_valid") == "1"}
QK = {r["query_id"]: r for r in json.load(open(X.CRR / "results/quality_selection_keys_reported.json"))["records"]}
byq = {r["query_id"]: r for _, r in V}
h, ctx_for = p2.h, p2.ctx_for

def sttl_rows(theta, kappa, stream):
    with X.patched(theta, kappa) as pt:
        m, rows = Q2.sttl_replay(stream, theta, kappa, pt.orig_ttl)
    return m, rows
def key_for(p, query):
    ctx = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""
    gq = p.get("gen_query", query)
    return ["gen", h(ctx), h(gq)], ctx, gq

m060, rows060 = sttl_rows(0.60, 0.5, V)
G["sttl_replay_060_saved_equals_engine"] = m060["search_saved_pct"] == ref_row["search_saved_pct"]
k_eq = sum(1 for d in QS if key_for(rows060[d["query_id"]], d["query"])[0] == QK[d["query_id"]]["sttl_t0.60_k1_2_key"])
G["recorded_060_validation_keys_reproduced_all"] = k_eq == len(QS) and DS.check("quality_selection_sample_requests", len(QS))
G["screened_set_size"] = REF.check("validation_quality/N", len(EVSET)) and EVSET <= {d["query_id"] for d in QS}

def tier_agree(tr, rows):
    mp = {"L1": "L1", "miss": "miss", "REAL_TIME_BYPASS": "REAL_TIME"}
    return sum(1 for _, r in V if mp[tr[r["query_id"]]["tier"]] != rows[r["query_id"]]["tier"])
G["sttl_060_engine_vs_transcription_tier_disagreements"] = tier_agree(ref_tr, rows060)
LOG["gates"] = G
ok = all(v is True for k, v in G.items() if not k.endswith("disagreements")) and G["sttl_060_engine_vs_transcription_tier_disagreements"] == 0
if not ok:
    json.dump(LOG, open(X.STTL_FRONTIER_DIR / "logs/validation_replay_STOPPED_gates.json", "w"), indent=1, default=str); sys.exit(f"STOP: reproduction gate failed: {G}")

# ---------------------------------------------------------------- selection rules, frozen before any answer exists
RULES = {
    "written_utc": now(), "written_before_any_generation_or_scoring": True, "heldout_consulted": False,
    "freshcache_validation_reference": {"search_saved_pct": fc_row["search_saved_pct"], "source": "mixed_age_replay FreshCache on the validation replay (gated against the run-time reference)",
                                        "c_wai": "FreshCache validation WAI / fresh-correct (frozen validation quality selection, selection_and_8b.json); recomputed and gated in stage 3"},
    "grid": X.CONFIGS,
    "A_quality_matched": "Among the 20 configs with validation C-WAI <= FreshCache validation C-WAI (FreshCache WAI / fresh-correct, as a 2-decimal percentage; "
                         "equivalently WAI <= FreshCache WAI on the fresh-correct requests), select the highest validation search savings. Ties: lower C-WAI, then more CORRECT, "
                         "then grid order. If no config qualifies, report NONE and do not substitute. The literal 2-decimal-percentage reading is recorded alongside.",
    "B_cost_matched": "Select the config whose validation search savings is closest to FreshCache validation savings. Ties: lower C-WAI, then higher savings, then grid order.",
    "quality_population": "screened validation requests (q_verification.csv evidence_valid=1) of the recorded sample; fresh-correct denominator from the frozen fresh arm.",
    "scoring": "two-judge rule: ABSTAIN if the answer contains a PRIM abstention pattern; else CORRECT iff llama8b and qwen7b both say CORRECT; else WRONG.",
}
RULES["sha256"] = X.hashlib.sha256(json.dumps({k: v for k, v in RULES.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
json.dump(RULES, open(X.STTL_FRONTIER_DIR / "frozen_selection/selection_rules.json", "w"), indent=1)

# ---------------------------------------------------------------- 20 configs
GA = X.P.AnswerLookup(extra=[X.CRR / "gpu/out/answers_llama3b.jsonl", X.CRR / "gpu/out3/answers_llama3b.jsonl"])
QA = {tuple(r["key"]): r["answer"] for r in X.jl((OUTPUT_ROOT / "quality_selection/q_answers.jsonl"))}
def have(key): return tuple(key) in QA or GA.get(list(key)) is not None
metrics, keys, tasks, reqrows = {"FreshCache": fc_row, "sttl_t0.60_k0.5_reference": ref_row}, {}, {}, []
for cfg in X.CONFIGS:
    row, per, tr = opmetrics("SemanticTTL", cfg["theta"], cfg["kappa"])
    m, rows = sttl_rows(cfg["theta"], cfg["kappa"], V)
    dis = tier_agree(tr, rows)
    if dis or m["search_saved_pct"] != row["search_saved_pct"]:
        LOG["inconsistency"] = {"config": cfg, "tier_disagreements": dis, "engine_saved": row["search_saved_pct"], "transcription_saved": m["search_saved_pct"]}
        json.dump(LOG, open(X.STTL_FRONTIER_DIR / "logs/validation_replay_STOPPED_inconsistency.json", "w"), indent=1, default=str); sys.exit(f"STOP: engine/transcription inconsistency {LOG['inconsistency']}")
    row["engine_vs_transcription_tier_disagreements"] = dis
    metrics[cfg["id"]] = {**cfg, **row}
    for _, r in V:
        q = r["query_id"]; p = rows[q]
        reqrows.append({"config": cfg["id"], "query_id": q, "freshness_class": r["freshness_class"], "tier": p["tier"],
                        "similarity": p.get("similarity"), "age_s": p.get("age"), "matched_query": p.get("matched_query"),
                        "drift_outcome": per.get(q), "in_support": q in S})
    keys[cfg["id"]] = {}
    for d in QS:
        if d["query_id"] not in EVSET:
            continue
        k, ctx, gq = key_for(rows[d["query_id"]], d["query"])
        keys[cfg["id"]][d["query_id"]] = {"key": k, "tier": rows[d["query_id"]]["tier"]}
        if not have(k):
            tasks.setdefault(tuple(k), {"key": k, "question": gq, "context": ctx})
    print(cfg["id"], {x: row[x] for x in ("search_saved_pct", "l1_hits", "drift_pct", "coverage_pct")}, flush=True)
for name, (pr, tr_) in {"FreshCache": (fc_per, fc_tr), "sttl_t0.60_k0.5_reference": (ref_per, ref_tr)}.items():
    for _, r in V:
        q = r["query_id"]
        reqrows.append({"config": name, "query_id": q, "freshness_class": r["freshness_class"], "tier": tr_[q]["tier"], "similarity": None,
                        "age_s": None, "matched_query": None, "drift_outcome": pr.get(q), "in_support": q in S})

json.dump(metrics, open(X.STTL_FRONTIER_DIR / "validation/replay_metrics.json", "w"), indent=1)
json.dump({"screened_set": sorted(EVSET), "frozen_fresh_and_freshcache_keys": {q: {"fresh": QK[q]["fresh_key"], "fc": QK[q]["fc_key"], "sttl_t0.60_k1_2": QK[q]["sttl_t0.60_k1_2_key"]} for q in EVSET},
           "configs": keys}, open(X.STTL_FRONTIER_DIR / "validation/quality_keys.json", "w"))
with gzip.open(X.STTL_FRONTIER_DIR / "validation/request_level_replay.csv.gz", "wt", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(reqrows[0])); w.writeheader(); w.writerows(reqrows)
X.wjl(X.STTL_FRONTIER_DIR / "gpu/in/val_gen_tasks.jsonl", list(tasks.values()))
LOG.update({"finished_utc": now(), "new_generation_tasks": len(tasks), "selection_rules_sha256": RULES["sha256"]})
json.dump(LOG, open(X.STTL_FRONTIER_DIR / "logs/validation_replay.json", "w"), indent=1, default=str)
print("GATES", G, "\nnew generation tasks:", len(tasks))
