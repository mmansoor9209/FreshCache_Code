"""Audit sample of the risk-budget arms: the 400 audit records of risk_budget_audit_records.py, wrapped with the sample target and seed.
Writes outputs/parameter_calibration/answer_quality/audit_sample.json (read by risk_budget_audit_generation_judging.py and risk_budget_answer_keys.py)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, os, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
D = (OUTPUT_ROOT / "parameter_calibration/answer_quality")
records = json.load(open(D / "audit_records.json"))
json.dump({"target": 400, "seed": 42, "source": "l1_precision_gate/heldout_audit_sample.json (verbatim)", "requests": records}, open(D / "audit_sample.json", "w"), indent=1)
print(len(records), "requests")
