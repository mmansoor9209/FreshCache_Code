"""CategoryAware-style comparator: our reimplementation of the per-category similarity threshold and TTL policy of
category-aware semantic caching (Wang et al., 2025), not the complete published system; it omits quotas, the storage tier,
indexing and load adaptation.

CategoryAware frontier, selection stage (validation only).
`items`: two-judge items for validation answers that lack judgments. `select`: score all candidates on the validation set, apply
the frozen constraints, select CA-Strict / CA-Relaxed by minimum validation modeled cost, record the dominance check, write
frontier_validation_grid.csv and freeze frontier_selection_manifest.json (read-only) before any held-out run."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, json, os, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/category_aware_frontier"))
import frontier_helpers as X
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
ST = json.load(open(X.O / "work/stage1.json")); G0 = json.load(open(X.O / "grid_frozen.json")); EV = ST["EV"]; lab = X.Val()
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
QK = {r["query_id"]: r for r in json.load(open(X.CRR / "results/quality_selection_keys_reported.json"))["records"]}
ALL = [(q, QK[q]["fresh_key"]) for q in EV] + [(q, QK[q]["fc_key"]) for q in EV] + [(q, v["key"]) for c in ST["keys"].values() for q, v in c.items()]
if sys.argv[1] == "items":
    items, miss = X.judge_items(lab, [(QS[q]["query"], QS[q]["gold"], k) for q, k in ALL])
    if miss: X.log(f"STOP: {len(miss)} validation answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("validation judge2 items:", len(items)); sys.exit(0)
MAN = X.O / "frontier_selection_manifest.json"
if MAN.exists(): sys.exit("STOP: selection already frozen")
L = {}
for q, k in ALL:
    l, a, _ = lab.labels(QS[q]["query"], QS[q]["gold"], k)
    if l is None: X.log("STOP: unlabelled validation key", q); sys.exit(1)
    L[(q, tuple(k))] = l
fresh = {q: L[(q, tuple(QK[q]["fresh_key"]))] for q in EV}; FC = sum(v == "CORRECT" for v in fresh.values())
fcw = sum(1 for q in EV if fresh[q] == "CORRECT" and L[(q, tuple(QK[q]["fc_key"]))] == "WRONG")
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); G = {"validation N / fresh-correct / FreshCache WAI equal the recorded references": all(REF.check("validation_quality/" + n, v) for n, v in (("N", len(EV)), ("fresh_correct", FC), ("FreshCache_WAI", fcw)))}
def q_stats(keys):
    labs = {q: L[(q, tuple(keys[q]))] for q in EV}; w = sum(1 for q in EV if fresh[q] == "CORRECT" and labs[q] == "WRONG"); c = collections.Counter(labs.values())
    return {"N": len(EV), "fresh_correct": FC, "CORRECT": c["CORRECT"], "WRONG": c["WRONG"], "ABSTAIN": c["ABSTAIN"], "accuracy_pct": round(100 * c["CORRECT"] / len(EV), 3), "WAI": w, "C_WAI_pct": round(100 * w / FC, 4)}
STRICT, RELAX = 100 * fcw / FC, 100 * fcw / FC + 5.0   # FreshCache validation C-WAI; +5 pp relaxed
ROWS = []
for i, g in enumerate(G0["grid"]):
    o = ST["ops"][g["id"]]; q = q_stats({k: v["key"] for k, v in ST["keys"][g["id"]].items()})
    ROWS.append({"order": i, "config_id": g["id"], "dtheta": g["dtheta"], "m_ttl": g["m_ttl"], **{f"theta_{c}": g["theta"][c] for c in X.CLASSES}, **{f"kappa_{c}": g["kappa"][c] for c in X.CLASSES},
                 **{k: o[k] for k in ("search_saved_pct", "fetch_per_1k", "gen_per_1k", "l1_hits", "cross_category_hits", "live_p50_ms_per_request", "local_ms_per_request")}, **q,
                 "meets_STRICT": q["C_WAI_pct"] <= STRICT + 1e-9, "meets_RELAXED": q["C_WAI_pct"] <= RELAX + 1e-9})
fo = ST["ops"]["FreshCache_primary"]; fq = q_stats({q: QK[q]["fc_key"] for q in EV})
FCROW = {"config_id": "FreshCache_primary", **{k: fo[k] for k in ("search_saved_pct", "fetch_per_1k", "gen_per_1k", "l1_hits", "live_p50_ms_per_request", "local_ms_per_request")}, **fq}
CA0_WAI = next(r for r in ROWS if r["config_id"] == "ca_d+0.000_m1.00")["WAI"]; G["candidate (0, 1.0) validation WAI equals the recorded baseline-selection reference"] = REF.check("validation_quality/CategoryAware_original_WAI", CA0_WAI)
def pick(flag, cost_key):
    f = [r for r in ROWS if r[flag]]
    return min(f, key=lambda r: (r[cost_key], r["C_WAI_pct"], r["order"]))["config_id"] if f else "NONE"
SEL = {"CA_Strict": pick("meets_STRICT", "live_p50_ms_per_request"), "CA_Relaxed": pick("meets_RELAXED", "live_p50_ms_per_request")}
SEL_LOCAL = {"CA_Strict": pick("meets_STRICT", "local_ms_per_request"), "CA_Relaxed": pick("meets_RELAXED", "local_ms_per_request")}
DOM = [r["config_id"] for r in ROWS if r["live_p50_ms_per_request"] <= FCROW["live_p50_ms_per_request"] and r["C_WAI_pct"] <= FCROW["C_WAI_pct"]]
DOM_LOCAL = [r["config_id"] for r in ROWS if r["local_ms_per_request"] <= FCROW["local_ms_per_request"] and r["C_WAI_pct"] <= FCROW["C_WAI_pct"]]
for k, v in G.items(): X.log("GATE", k, v)
if not all(G.values()): X.log("STOP"); sys.exit(1)
with open(X.O / "frontier_validation_grid.csv", "w", newline="") as f:
    cols = list(ROWS[0]); w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(ROWS); w.writerow({k: FCROW.get(k, "") for k in cols})
M = {"frozen_before_heldout_frontier": True, "heldout_consulted": False, "grid_sha256": G0["sha256"], "constraints_pct": {"STRICT": round(STRICT, 4), "RELAXED": round(RELAX, 4)},
     "selection_cost_regime": "live_p50 (primary, predeclared)", "selected": SEL, "CA_Relaxed_distinct_from_CA_Strict": SEL["CA_Relaxed"] != SEL["CA_Strict"], "selection_under_local_regime_for_reference": SEL_LOCAL,
     "selected_params": {k: next((g for g in G0["grid"] if g["id"] == v), None) for k, v in SEL.items()}, "n_meeting_STRICT": sum(r["meets_STRICT"] for r in ROWS), "n_meeting_RELAXED": sum(r["meets_RELAXED"] for r in ROWS),
     "freshcache_validation": FCROW, "dominance_check": {"rule": "validation modeled cost <= FreshCache AND validation C-WAI <= FreshCache", "live_p50": DOM, "local": DOM_LOCAL, "any": bool(DOM)}}
M["sha256"] = X.sha_obj(M); M["frozen_utc"] = now()
json.dump(M, open(MAN, "w"), indent=1); os.chmod(MAN, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
X.log(f"[{M['frozen_utc']}] selection manifest frozen sha256 {M['sha256']} BEFORE held-out: {json.dumps(SEL)}; distinct {M['CA_Relaxed_distinct_from_CA_Strict']}; dominance {DOM}")
