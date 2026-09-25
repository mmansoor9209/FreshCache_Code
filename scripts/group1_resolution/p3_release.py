#!/usr/bin/env python3
"""N20 -- release scaffolding. Builds a real manifest from files on disk;
nothing is invented."""
from __future__ import annotations
import hashlib, json, pathlib, subprocess, sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
REL = HERE / "release"
V3 = HERE.parent
ROOT = V3.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


FROZEN = [
    V3 / "parameter_calibration" / "04_budgets" / "frozen_config.json",
    V3 / "parameter_calibration" / "04_budgets" / "prereg.json",
    V3 / "l1_precision_gate" / "frozen_l1_precision_gate.json",
    V3 / "l1_precision_gate" / "l1_precision_prereg.json",
    V3 / "l1_precision_gate" / "l1_guards.py",
    V3 / "remaining_critical_issues" / "05_prospective_temporal" / "freeze_manifest.json",
    V3 / "remaining_critical_issues" / "00_audit" / "audit_inventory.json",
]
CODE = ["experiment.py", "collect.py", "calibrate.py", "v16_exp12/schedules.py",
        "v14_baselines/engine_all.py", "v16_exp12/mixed_engine.py",
        "v13_corrected/corrected_engine.py", "v9/mixed_age_v2.py",
        "vcache_implementation.py",
        "validation/mixed_age_full_policy_audit/prep2.py"]
ROWLEVEL = sorted(
    [p for p in V3.rglob("*per_request*.csv")] +
    [p for p in V3.rglob("*rows*.csv")] +
    [ROOT / "validation" / "fresh_path_failure_audit" / "fresh_path_failure_per_request.csv",
     ROOT / "validation" / "actual_l2_artifact_audit" / "a_per_case.csv",
     ROOT / "validation" / "l2_independent_audit_300" / "results_canonical" / "per_case_overlap.csv",
     ROOT / "v16_validation_wai_depth" / "wai_by_tier.json"])


def main():
    REL.mkdir(exist_ok=True)
    man = {"generated_utc": datetime.now(timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"),
           "frozen_configs": {}, "core_code": {}, "row_level_artifacts": {}}
    for p in FROZEN:
        if p.exists():
            man["frozen_configs"][str(p.relative_to(ROOT))] = {
                "sha256": sha(p), "bytes": p.stat().st_size}
    for c in CODE:
        p = ROOT / c
        if p.exists():
            man["core_code"][c] = {"sha256": sha(p), "bytes": p.stat().st_size}
    seen = set()
    for p in ROWLEVEL:
        if p.exists() and p not in seen:
            seen.add(p)
            man["row_level_artifacts"][str(p.relative_to(ROOT))] = {
                "sha256": sha(p), "bytes": p.stat().st_size}
    json.dump(man, open(REL / "ARTIFACT_MANIFEST.json", "w"), indent=2)
    print(f"  manifest: {len(man['frozen_configs'])} frozen configs, "
          f"{len(man['core_code'])} code files, "
          f"{len(man['row_level_artifacts'])} row-level artifacts")

    # environment
    try:
        pf = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, timeout=120).stdout
    except Exception:
        pf = ""
    keep = ("numpy", "scipy", "scikit-learn", "torch", "transformers",
            "sentence-transformers", "FlagEmbedding", "pandas", "tqdm",
            "setproctitle", "accelerate", "tokenizers", "safetensors")
    req = [l for l in pf.splitlines()
           if any(l.lower().startswith(k.lower() + "==") for k in keep)]
    (REL / "requirements.txt").write_text(
        "# Pinned versions of the packages the FreshCache pipeline imports.\n"
        "# Generated from the live environment; the full freeze is in\n"
        "# pip-freeze-full.txt.\n" + "\n".join(sorted(req)) + "\n",
        encoding="utf-8")
    (REL / "pip-freeze-full.txt").write_text(pf, encoding="utf-8")
    ver = sys.version.split()[0]
    (REL / "environment.yml").write_text(
        f"""name: freshcache
channels: [conda-forge, defaults]
dependencies:
  - python={'.'.join(ver.split('.')[:2])}
  - pip
  - pip:
{chr(10).join('      - ' + r for r in sorted(req))}
""", encoding="utf-8")
    print(f"  environment: python {ver}, {len(req)} pinned packages")


if __name__ == "__main__":
    main()
