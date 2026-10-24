"""SemanticTTL savings-matched operating point (the frozen protocol had quality- and cost-matched rules only).
Rule (pre-registered, VALIDATION ONLY, written after the primary held-out look and BEFORE any held-out replay of the selected cell):
  savings_matched = the SemanticTTL cell whose validation search savings are closest to FreshCache primary's validation savings;
  ties -> lower validation C-WAI. The selected cell is then run through the UNCHANGED operating_point_heldout_evaluation.py pipeline (in-memory copy; only the point
list and output file names differ), so FreshCache and every other point are untouched. Reported separately as an added operating point.
Usage: savings_matched_operating_point.py replay | items | score"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, datetime, hashlib, json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "standalone_operating_points"))
AF = O / "selection_addendum_frozen.json"
if not AF.exists():
    SEL = json.load(open(O / "selection_frozen.json")); fc = SEL["freshcache_validation"]["search_saved_pct"]
    R = [r for r in csv.DictReader(open(O / "validation_grid.csv")) if r["family"] == "SemanticTTL"]
    R.sort(key=lambda r: (abs(float(r["search_saved_pct"]) - fc), float(r["q_C_WAI_pct"])))
    c = R[0]
    A = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "selection_frozen_sha256": SEL["sha256"],
         "rule": "SemanticTTL cell with validation search savings closest to FreshCache primary validation savings; ties -> lower validation C-WAI",
         "target_validation_savings": fc, "selected": c["cell"], "validation": {k: c[k] for k in ("search_saved_pct", "live_p50_ms_per_request", "q_WAI", "q_C_WAI_pct")},
         "status": "savings-matched addendum; validation-only selection; held-out replay of this cell not yet run when written", "heldout_consulted_for_this_cell": False}
    A["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in A.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
    json.dump(A, open(AF, "w"), indent=1)
A = json.load(open(AF)); print("addendum", A["selected"], A["sha256"][:12], flush=True)
src = open((REPO_ROOT / "scripts/experiments/operating_points/operating_point_heldout_evaluation.py")).read()
anchor = 'if MODE == "replay":\n'; assert src.count(anchor) == 1
src = src.replace(anchor, f'POINTS = {{{A["selected"]!r}: {{"family": "SemanticTTL", "rules": ["savings_matched (addendum)"]}}}}\n' + anchor)
for a, b in (('W / "stage3.pkl"', 'W / "stage3_addendum.pkl"'), ('"heldout_decision_table.json"', '"heldout_decision_table_addendum.json"'), ('"heldout_decision_table.csv"', '"heldout_decision_table_addendum.csv"')):
    assert a in src, a; src = src.replace(a, b)
g = {"__file__": str((REPO_ROOT / "scripts/experiments/operating_points/operating_point_heldout_evaluation.py")), "__name__": "__main__"}
exec(compile(src, "<operating_point_heldout_evaluation.py addendum copy>", "exec"), g)
