"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

SemanticTTL/CategoryAware baseline selection, stage 2 (validation only). Modes:
`items`: two-judge items for validation answers lacking judgments. `select`: score the shared-parameter cells and the per-category cells
on the validation set, apply the frozen rules, and write frozen_selection.json (read-only) before any held-out work."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, datetime, hashlib, json, os, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_category_aware"))
import baseline_selection_helpers as X
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
ST = json.load(open(X.O / "work/stage1.json")); EV = ST["EV"]; lab = X.Val()
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
QK = {r["query_id"]: r for r in json.load(open(X.CRR / "results/quality_selection_keys_reported.json"))["records"]}
CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST"]
allkeys = [(q, QK[q]["fresh_key"]) for q in EV] + [(q, QK[q]["fc_key"]) for q in EV] + [(q, v["key"]) for c in ST["A_keys"].values() for q, v in c.items()] + [(q, v["key"]) for c in ST["B1_keys"].values() for q, v in c.items()]
if sys.argv[1] == "items":
    items, miss = X.judge_items(lab, [(QS[q]["query"], QS[q]["gold"], k) for q, k in allkeys])
    if miss: X.log(f"STOP: {len(miss)} validation answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("validation judge2 items:", len(items)); sys.exit(0)
SELP = X.O / "frozen_selection.json"
if SELP.exists(): sys.exit("STOP: selection already frozen")
L = {}
for q, k in allkeys:
    l, a, _ = lab.labels(QS[q]["query"], QS[q]["gold"], k)
    if l is None: X.log("STOP: unlabelled validation key", q, k); sys.exit(1)
    L[(q, tuple(k))] = l
fresh = {q: L[(q, tuple(QK[q]["fresh_key"]))] for q in EV}; fc = {q: L[(q, tuple(QK[q]["fc_key"]))] for q in EV}
fcn = sum(v == "CORRECT" for v in fresh.values()); fcw = sum(1 for q in EV if fresh[q] == "CORRECT" and fc[q] == "WRONG")
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); G = {"validation N / fresh-correct equal the recorded references": REF.check("validation_quality/N", len(EV)) and REF.check("validation_quality/fresh_correct", fcn)}
REF.record("validation_quality/FreshCache_WAI", fcw); REF.record("validation_quality/FreshCache_correct", sum(v == "CORRECT" for v in fc.values()))
if not all(G.values()): X.log("STOP", G); sys.exit(1)
def q_stats(keys, ids):
    labs = {q: L[(q, tuple(keys[q]["key"]))] for q in ids}; w = sum(1 for q in ids if fresh[q] == "CORRECT" and labs[q] == "WRONG"); c = collections.Counter(labs.values())
    nfc = sum(1 for q in ids if fresh[q] == "CORRECT"); return {"N": len(ids), "fresh_correct": nfc, "WAI": w, "C_WAI_pct": round(100 * w / nfc, 4) if nfc else None, "CORRECT": c["CORRECT"], "ABSTAIN": c["ABSTAIN"], "WRONG": c["WRONG"]}
LIM, REL = 100 * fcw / fcn, 100 * fcw / fcn + 5.0
A = {cid: {**ST["A_ops"][cid], **q_stats(ST["A_keys"][cid], EV)} for cid in ST["A_ops"]}
order = list(A)
feas = [c for c in order if A[c]["C_WAI_pct"] <= LIM + 1e-9]; rel = [c for c in order if A[c]["C_WAI_pct"] <= REL + 1e-9]
if feas: a1, tier = max(feas, key=lambda c: (A[c]["search_saved_pct"], -order.index(c))), "primary limit (C-WAI <= FreshCache)"
elif rel: a1, tier = max(rel, key=lambda c: (A[c]["search_saved_pct"], -order.index(c))), "relaxed limit (FreshCache + 5 pp)"
else: a1, tier = min(order, key=lambda c: (A[c]["WAI"], -A[c]["search_saved_pct"], order.index(c))), "no feasible cell: lowest WAI"
FCS = ST["freshcache_validation"]["search_saved_pct"]
a2 = min(order, key=lambda c: (abs(A[c]["search_saved_pct"] - FCS), A[c]["WAI"], -A[c]["search_saved_pct"], order.index(c)))
# ---------------- Part B per-category selection
fc_class_wai = {c: sum(1 for q in EV if QS[q]["fc"] == c and fresh[q] == "CORRECT" and fc[q] == "WRONG") for c in CLASSES}
B = {}; SEL = {}
for c in CLASSES:
    ids = [q for q in EV if QS[q]["fc"] == c]; cells = [cid for cid in ST["B1_ops"] if cid.startswith(c + "|")]
    for cid in cells: B[cid] = {**ST["B1_ops"][cid], **q_stats(ST["B1_keys"][cid], ids), "freshcache_class_WAI": fc_class_wai[c]}
    f = [cid for cid in cells if B[cid]["WAI"] <= fc_class_wai[c]]
    pick = max(f, key=lambda x: (B[x]["search_saved_pct_class"], -cells.index(x))) if f else min(cells, key=lambda x: (B[x]["WAI"], -B[x]["search_saved_pct_class"], cells.index(x)))
    SEL[c] = {"cell": pick, "theta": B[pick]["theta"], "kappa": B[pick]["kappa"], "feasible": bool(f), "class_savings": B[pick]["search_saved_pct_class"], "class_WAI": B[pick]["WAI"], "freshcache_class_WAI": fc_class_wai[c]}
FROZEN = {"frozen_utc": now(), "heldout_consulted": False, "design_sha256": json.load(open(X.O / "design_frozen.json"))["sha256"],
          "validation_reference": {"freshcache_savings_pct": FCS, "freshcache_WAI": fcw, "fresh_correct": fcn, "C_WAI_limit_pct": round(LIM, 4), "relaxed_limit_pct": round(REL, 4)},
          "A1": {"id": a1, "selection_tier": tier, **{k: A[a1][k] for k in ("theta", "kappa", "search_saved_pct", "WAI", "C_WAI_pct", "CORRECT")}},
          "A2": {"id": a2, **{k: A[a2][k] for k in ("theta", "kappa", "search_saved_pct", "WAI", "C_WAI_pct", "CORRECT")}, "savings_gap_pp": round(abs(A[a2]["search_saved_pct"] - FCS), 4)},
          "CategoryAware_style": {"per_category": SEL, "REAL_TIME": "bypass (as SemanticTTL)"}}
FROZEN["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in FROZEN.items() if k != "frozen_utc"}, sort_keys=True).encode()).hexdigest()
json.dump({"gates": G, "partA": A, "partB_cells": B}, open(X.O / "work/validation_scores.json", "w"), indent=1)
json.dump(FROZEN, open(SELP, "w"), indent=1); os.chmod(SELP, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
X.log(f"[{FROZEN['frozen_utc']}] selection frozen sha256 {FROZEN['sha256']}"); X.log(json.dumps(FROZEN, indent=1))
