#!/usr/bin/env python3
"""
heldout_sensitivity_protocol.py -- fix the sensitivity configuration before any held-out
evaluation. Writes outputs/quality_selection/heldout_sensitivity/freeze.json with a hash;
the downstream held-out sensitivity scripts require it.

The configuration is SemanticTTL theta=0.60, kappa=1/2, the single candidate admitted by
the pre-registered 5 pp sensitivity constraint in outputs/quality_selection/quality_selection_prereg.json.
It did not satisfy the primary no-slack constraint; no candidate did, and that primary
result is reported unchanged.
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import hashlib, json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
HERE = ((OUTPUT_ROOT / "quality_selection/heldout_sensitivity"))
QAB = ((OUTPUT_ROOT / "quality_selection"))
pre = json.load(open((OUTPUT_ROOT / "quality_selection/quality_selection_prereg.json"), encoding="utf-8"))
q6 = json.load(open(QAB / "quality_selection_results.json", encoding="utf-8"))
assert q6["n_satisfying_primary"] == 0, "primary result changed — stop"
assert q6["selected"] is None
sat5 = [c for c in q6["candidates"] if c["satisfies_secondary_5pp"]]
assert len(sat5) == 1, sat5
c = sat5[0]
freeze = {
    "frozen_before_heldout_evaluation": True,
    "config": {"policy": "SemanticTTL", "theta": c["theta"], "kappa": c["kappa"],
               "kappa_frac": c["kappa_frac"], "arm": c["arm"]},
    "admitted_by": "PRE-REGISTERED 5 pp SENSITIVITY constraint only",
    "primary_constraint_result": "FAILED — 0 of 9 candidates satisfied the "
                                 "no-slack constraint; that result is unchanged",
    "validation_evidence": {k: c[k] for k in
                            ("saved", "drift", "coverage", "accuracy_pct", "wai",
                             "conditional_wai_pct", "conditional_wai_ci95",
                             "fresh_correct_n", "N")},
    "freshcache_validation_conditional_wai_pct":
        q6["constraint_threshold_conditional_wai_pct"],
    "selection_used_heldout_data": False,
    "prereg_sha256": pre["prereg_sha256"],
    "heldout_protocol": "the existing held-out answer audit sample; "
                        "FreshCache answers, fresh reference, judgments and revised "
                        "gold verification are reused unchanged; only the new "
                        "SemanticTTL arm is generated and judged",
}
freeze["freeze_sha256"] = hashlib.sha256(
    json.dumps(freeze, sort_keys=True).encode()).hexdigest()
json.dump(freeze, open(HERE / "freeze.json", "w"), indent=2)
print(f"  FROZEN: SemanticTTL theta={c['theta']:.2f} kappa={c['kappa_frac']}")
print(f"    validation: saved {c['saved']:.4f}%  cond.WAI {c['conditional_wai_pct']}% "
      f"(FreshCache {q6['constraint_threshold_conditional_wai_pct']}%)")
print(f"    admitted by the 5 pp sensitivity only; primary constraint satisfied by 0 candidates")
print(f"    freeze sha256 {freeze['freeze_sha256'][:32]}...")
