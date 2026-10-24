"""Operating-point selection, stage 2 (validation only). Modes:
  items   -> two-judge items for validation answers still unjudged (after gen)
  select  -> validation quality per cell (screened set, two-judge rule), apply the frozen rules of design_frozen.json, write selection_frozen.json.
Held-out data is never read here."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, datetime, hashlib, json, os, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/operating_points"))
import operating_point_helpers as X
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(); MODE = sys.argv[1]
ST = pickle.load(open(X.O / "work/stage1.pkl", "rb")); CELLS, KEYS, EV = ST["cells"], ST["keys"], ST["EV"]
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
lab = X.Val()
QK = {r["query_id"]: r for r in json.load(open(X.CRR / "results/quality_selection_keys_reported.json"))["records"]}   # as baseline_validation_selection.py
FRESH = {q: lab.labels(QS[q]["query"], QS[q]["gold"], QK[q]["fresh_key"])[0] for q in EV}; FCN = sum(v == "CORRECT" for v in FRESH.values()); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); assert REF.check("validation_quality/fresh_correct", FCN), FCN
if MODE == "items":
    items, miss = X.judge_items(lab, [(QS[q]["query"], QS[q]["gold"], KEYS[c][q]["key"]) for c in KEYS if c != "NoCache" for q in EV])
    if miss: X.log(f"STOP: {len(miss)} validation answers missing"); sys.exit(1)
    X.wjl(X.O / "gpu/in/judge2_items.jsonl", items); X.log("judge2 items:", len(items)); sys.exit(0)
D = json.load(open(X.O / "design_frozen.json"))   # LIMIT / RELAX: FreshCache primary validation C-WAI (set below)
Q = {}
for c in CELLS:
    if c == "NoCache": Q[c] = {"N": len(EV), "fresh_correct": FCN, "CORRECT": FCN, "WRONG": sum(v == "WRONG" for v in FRESH.values()), "ABSTAIN": sum(v == "ABSTAIN" for v in FRESH.values()), "UNRESOLVED": 0, "WAI": 0, "C_WAI_pct": 0.0, "accuracy_pct": round(100 * FCN / len(EV), 3), "abstain_pct": round(100 * sum(v == "ABSTAIN" for v in FRESH.values()) / len(EV), 3)}; continue
    if c not in KEYS:   # CA cells: quality already in the cell (frontier validation grid)
        x = CELLS[c]; Q[c] = {"N": x["N"], "fresh_correct": x["fresh_correct"], "CORRECT": x["CORRECT"], "WRONG": x["WRONG"], "ABSTAIN": x["ABSTAIN"], "UNRESOLVED": 0, "WAI": x["WAI"], "C_WAI_pct": x["C_WAI_pct"], "accuracy_pct": x["accuracy_pct"], "abstain_pct": round(100 * x["ABSTAIN"] / x["N"], 3)}; continue
    L = {q: (lab.labels(QS[q]["query"], QS[q]["gold"], KEYS[c][q]["key"])[0] or "UNRESOLVED") for q in EV}; cnt = collections.Counter(L.values()); w = sum(1 for q in EV if FRESH[q] == "CORRECT" and L[q] == "WRONG")
    Q[c] = {"N": len(EV), "fresh_correct": FCN, "CORRECT": cnt["CORRECT"], "WRONG": cnt["WRONG"], "ABSTAIN": cnt["ABSTAIN"], "UNRESOLVED": cnt["UNRESOLVED"], "WAI": w, "C_WAI_pct": round(100 * w / FCN, 4), "accuracy_pct": round(100 * cnt["CORRECT"] / len(EV), 3), "abstain_pct": round(100 * cnt["ABSTAIN"] / len(EV), 3)}
assert all(Q[c]["UNRESOLVED"] == 0 for c in Q), {c: Q[c]["UNRESOLVED"] for c in Q if Q[c]["UNRESOLVED"]}
FC = CELLS["FreshCache_primary"]; fcq = Q["FreshCache_primary"]; LIMIT = 100 * fcq["WAI"] / FCN; RELAX = LIMIT + 5; G = {"FreshCache primary validation WAI equals the recorded reference": REF.check("validation_quality/FreshCache_WAI", fcq["WAI"]), "FreshCache keys rebuilt here equal baseline-selection/quality-selection fc_key": all(KEYS["FreshCache_primary"][q]["key"] == QK[q]["fc_key"] for q in EV)}
fc_cost = FC["live_p50_ms_per_request"]
def pick(family):
    cells = [c for c in CELLS if CELLS[c]["family"] == family]
    qm = [c for c in cells if Q[c]["C_WAI_pct"] <= LIMIT]; qm_r = [c for c in cells if Q[c]["C_WAI_pct"] <= RELAX]
    key_qm = lambda c: (-CELLS[c]["search_saved_pct"], CELLS[c]["live_p50_ms_per_request"], Q[c]["WAI"])
    band = [c for c in cells if abs(CELLS[c]["live_p50_ms_per_request"] / fc_cost - 1) <= 0.05]
    cm = sorted(band, key=lambda c: (Q[c]["C_WAI_pct"], CELLS[c]["live_p50_ms_per_request"]))[0] if band else sorted(cells, key=lambda c: abs(CELLS[c]["live_p50_ms_per_request"] - fc_cost))[0]
    return {"quality_matched": sorted(qm, key=key_qm)[0] if qm else None, "quality_matched_relaxed": sorted(qm_r, key=key_qm)[0] if qm_r else None, "cost_matched": cm, "cost_matched_within_band": bool(band),
            "n_cells": len(cells), "n_meeting_strict": len(qm), "n_meeting_relaxed": len(qm_r)}
SEL = {fam: pick(fam) for fam in ("SemanticTTL", "CategoryAware")}
for fam, s in SEL.items():
    for k, c in s.items():
        if isinstance(c, str): X.log("SELECT", fam, k, c, json.dumps({**{x: CELLS[c][x] for x in ("search_saved_pct", "live_p50_ms_per_request")}, **{x: Q[c][x] for x in ("WAI", "C_WAI_pct")}}))
OUT = {"frozen_utc": now(), "design_sha256": D["sha256"], "constraint_C_WAI_limit_pct": round(LIMIT, 4), "relaxed_limit_pct": round(RELAX, 4), "freshcache_validation": {**{k: FC[k] for k in ("search_saved_pct", "live_p50_ms_per_request", "local_ms_per_request")}, **fcq},
       "selection": SEL, "cells": {c: {**CELLS[c], **{f"q_{k}": v for k, v in Q[c].items()}} for c in CELLS}, "gates": G, "heldout_consulted": False}
OUT["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in OUT.items() if k != "frozen_utc"}, sort_keys=True, default=str).encode()).hexdigest()
if not (X.O / "selection_frozen.json").exists(): json.dump(OUT, open(X.O / "selection_frozen.json", "w"), indent=1, default=str)
import csv
with open(X.O / "validation_grid.csv", "w", newline="") as f:
    cols = ["cell", "family", "theta", "kappa", "theta_L2", "config_id", "search_saved_pct", "fetch_per_1k", "gen_per_1k", "local_ms_per_request", "live_p50_ms_per_request", "drift_pct", "coverage_pct"] + [f"q_{k}" for k in ("CORRECT", "WRONG", "ABSTAIN", "accuracy_pct", "abstain_pct", "WAI", "C_WAI_pct")]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for c in CELLS: w.writerow({"cell": c, **CELLS[c], **{f"q_{k}": v for k, v in Q[c].items()}})
X.log("selection frozen", OUT["sha256"][:16], json.dumps(SEL))
