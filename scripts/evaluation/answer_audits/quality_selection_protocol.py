#!/usr/bin/env python3
"""
quality_selection_protocol.py -- pre-registration. Must run, and its output must exist,
before any answer is generated. Fixes the candidate set, the quality constraint, the
selection rule and the sampling specification using only operational metrics already
computed on the validation split (outputs/semanticttl_joint_grid/).

Nothing here depends on answer quality, and no answer-quality result exists at this point.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, hashlib, json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction

HERE = ((OUTPUT_ROOT / "quality_selection"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
GRID42 = (OUTPUT_ROOT / "semanticttl_joint_grid")

grid = [{"theta": float(r["theta"]), "kappa": float(r["kappa"]),
         "saved": float(r["search_saved_pct"]), "drift": float(r["drift_pct"]),
         "coverage": float(r["coverage_pct"]),
         "determinable": int(r["determinable"])}
        for r in csv.DictReader(open(GRID42 / "joint_validation_grid.csv",
                                     encoding="utf-8"))]
FC = json.load(open(GRID42 / "joint_validation_selection.json",
                    encoding="utf-8"))["freshcache_validation_reference"]

# ---- candidate rule (operational only) ----
cands = sorted([c for c in grid if c["saved"] >= FC["search_saved_pct"]],
               key=lambda c: -c["saved"])
for c in cands:
    c["kappa_frac"] = str(Fraction(c["kappa"]).limit_denominator(64))
    c["arm"] = f"sttl_t{c['theta']:.2f}_k{c['kappa_frac'].replace('/', '_')}"

prereg = {
    "preregistered": True,
    "written_before_any_answer_generation": True,
    "freshcache_configuration": "unchanged frozen FreshCache "
                                "(theta=0.40, eps 0.10/0.20/0.35, half-life risk gate)",
    "freshcache_validation_reference": FC,
    "candidate_rule": "all cells of the existing 42-cell SemanticTTL validation grid whose "
                      "validation search savings are at least FreshCache's "
                      f"validation search savings ({FC['search_saved_pct']}%). Both the similarity "
                      "threshold theta and the TTL multiplier kappa are tunable.",
    "candidates": cands,
    "n_candidates": len(cands),
    "quality_constraint": {
        "primary": "SemanticTTL conditional WAI on the VALIDATION evidence-valid "
                   "subset must be LESS THAN OR EQUAL TO frozen FreshCache's "
                   "conditional WAI on the IDENTICAL request set. No slack.",
        "rationale": "A slack margin would be arbitrary. The no-slack form asks "
                     "the question this selection addresses directly: is there a "
                     "SemanticTTL setting that is at least as safe as FreshCache "
                     "on answer quality while saving more search?",
        "secondary_sensitivity": "the same rule with a 5 pp slack, reported "
                                 "separately and never used for the headline",
        "denominator": "requests whose fresh reference is CORRECT, within the "
                       "validation evidence-valid subset",
        "identical_requests_across_policies": True},
    "selection_rule": {
        "objective": "among candidates satisfying the primary constraint, choose "
                     "the highest VALIDATION search savings",
        "tie_breaks": ["lower validation drift", "higher validation coverage",
                       "higher theta", "smaller kappa"],
        "if_none_satisfy": "report that no configuration satisfies the constraint. "
                           "The constraint is NOT relaxed after seeing results; "
                           "the secondary sensitivity is reported as a separate, "
                           "clearly-labelled row."},
    "gold_verification": {
        "procedure": "timestamp-validity procedure (outputs/gold_verification/timestamp_validity): expanded snapshot evidence (5,000 chars/page, no minimum-length floor, 12,000-char budget), question-type-aware prompts (AS_STATED vs CURRENT), Llama-3.1-8B + Qwen2.5-7B, Mistral-7B-Instruct-v0.3 breaking ties (2-of-3 MAJORITY)",
        "evidence_valid_rule": "(paraphrase fidelity SAME under both judges OR "
                               "original base question) AND unit VALID_AT_T",
        "note": "VALID_AT_T is LLM-supported temporal validity, NOT independently "
                "verified factual correctness."},
    "sample": {
        "frame": "validation clusters only, from outputs/validation_split/split.json",
        "target_n": 700, "seed": 42,
        "stratification": "freshness class x FreshCache serving tier, "
                          "policy-neutral (strata defined before any policy is run)",
        "test_clusters_used": 0,
        "frozen_before_generation": True},
    "generation_protocol": "identical to both existing audits: "
                           "Llama-3.2-3B-Instruct, FORCED prompt, greedy, "
                           "max_new_tokens=80, 1024-token truncation, answer cut at "
                           "the first '.' or newline",
    "judging_protocol": "Llama-3.1-8B-Instruct + Qwen2.5-7B-Instruct, chat template, "
                        "greedy, max_new_tokens=6, both must agree",
    "heldout_use": "the selected configuration is replayed ONCE on the untouched "
                   "held-out test split; no tuning occurs there",
}
blob = json.dumps(prereg, sort_keys=True).encode()
prereg["prereg_sha256"] = hashlib.sha256(blob).hexdigest()
HERE.mkdir(parents=True, exist_ok=True); json.dump(prereg, open(HERE / "quality_selection_prereg.json", "w"), indent=2)   # definitions: configs/protocols/quality_selection_prereg.json
print(f"  candidates: {len(cands)} (validation savings >= {FC['search_saved_pct']}%)")
for c in cands:
    print(f"    {c['arm']:<24} theta={c['theta']:.2f} kappa={c['kappa_frac']:>4}  "
          f"saved={c['saved']:8.4f}  drift={c['drift']:7.4f}")
print(f"  constraint : conditional WAI <= FreshCache conditional WAI (no slack)")
print(f"  objective  : max validation search savings among satisfiers")
print(f"  sample     : {prereg['sample']['target_n']} validation-cluster requests, seed {prereg['sample']['seed']}")
print(f"  prereg sha256: {prereg['prereg_sha256'][:32]}...")
print(f"  wrote prereg.json")
