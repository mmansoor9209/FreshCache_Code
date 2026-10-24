#!/usr/bin/env python3
"""
Pre-registration for the L1-Precision gate. MUST be run before
l1_precision_validation_grid.py. Writes l1_precision_prereg.json and prints its SHA-256.

Nothing here depends on any grid result, because no grid has been run.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import hashlib, json, pathlib, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()

HERE = ((OUTPUT_ROOT / "l1_precision"))


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


SIM_FLOORS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97]
JAC_FLOORS = [0.30, 0.40, 0.50, 0.60]
ANSWER_TYPE_MODES = ["as_implemented", "required"]
GUARD_MODES = ["off", "on"]

PREREG = {
  "written_before_any_grid_cell_was_evaluated": True,
  "scope": "The L1 semantic admission condition ONLY. L2 similarity threshold, "
           "L2 temporal eligibility, L2 re-registration, all L3 logic, "
           "half-lives, risk budgets, timestamps, cache-update semantics, "
           "request order and the freshness-class source are unchanged and are "
           "asserted unchanged by a file-hash check before and after every run.",
  "preserved_from_the_recorded_gate": [
      "BGE-M3 cosine similarity floor",
      "content-word Jaccard floor",
      "answer-type agreement",
      "entity agreement (experiment._entity_match, retained verbatim)",
      "strict content-token subset rejection",
  ],
  "grid": {
      "sim_floor": SIM_FLOORS,
      "jaccard_floor": JAC_FLOORS,
      "answer_type": {
          "as_implemented": "reject only when BOTH answer types are detectable "
                            "and differ (the recorded rule)",
          "required": "both answer types must be detectable AND equal",
      },
      "structural_guards": {
          "off": "recorded gate only",
          "on": "recorded gate AND the three deterministic guards in l1_structural_guards.py (numeric/date, negation, comparative/superlative)",
      },
      "n_cells": len(SIM_FLOORS) * len(JAC_FLOORS) * len(ANSWER_TYPE_MODES)
                 * len(GUARD_MODES),
  },
  "guards_are_frozen_by_hash": "l1_structural_guards.py",
  "guards_contain_no_model": "pure string operations; no LLM, no embedding, no "
                             "network, no randomness, in the online L1 path",
  "replay": "TRUE SEQUENTIAL replay of the full validation stream for every "
            "cell. Changing an L1 decision changes later cache state, so no "
            "cell is produced by filtering another cell's L1 hits.",
  "data": "VALIDATION clusters only. Held-out test results are not computed, "
          "inspected or used at any point during selection.",

  "wai_measurability": {
      "finding": "No validation-side answer audit exists before this study: "
                 "every earlier audit (the general held-out audit, the "
                 "stronger-generator audit and the mixed-age audit) samples "
                 "TEST clusters only; none of their requests falls in a "
                 "validation cluster.",
      "consequence": "Validation WAI is not available for all "
                     f"{len(SIM_FLOORS)*len(JAC_FLOORS)*len(ANSWER_TYPE_MODES)*len(GUARD_MODES)} "
                     "cells, and generating 96 validation audits is not "
                     "feasible. Selection is therefore TWO-STAGE, declared here "
                     "before any cell is run.",
  },
  "selection": {
      "stage_1": {
          "description": "Run all grid cells on validation. Keep the cells that "
                         "satisfy the hard constraints, rank by the primary "
                         "objective, and carry the top K forward.",
          "K": 5,
          "hard_constraints": [
              "C1: validation search_saved_pct >= (recorded FreshCache "
              "validation search_saved_pct - 0.10 percentage points)",
              "C3: at least 100 validation L1 hits",
          ],
          "primary_objective": "MINIMISE validation L1 semantic mismatch "
                               "under the ties->DIFFERENT convention",
      },
      "stage_2": {
          "description": "Build ONE new policy-neutral answer audit "
                         "sampled from VALIDATION clusters (same 28-stratum "
                         "design, same seed 42, same evidence builder and "
                         "context limits as the recorded audit), and measure "
                         "WAI for the recorded gate and for each of the K "
                         "carried-forward cells. Then apply C2 and the "
                         "tie-breaks.",
          "hard_constraint": "C2: validation WAI must not exceed the recorded "
                             "FreshCache validation WAI. If WAI is not "
                             "measurable for a cell, C2 is recorded as "
                             "NOT_EVALUABLE for that cell and the cell is NOT "
                             "eliminated by it.",
      },
      "tie_breaks_in_this_exact_order": [
          "1. lower mismatch (ties->DIFFERENT)",
          "2. lower WAI",
          "3. more L1 hits",
          "4. fewer additional generations",
          "5. lower similarity threshold",
      ],
      "explicitly_not_an_objective": "L1 hit rate is NOT maximised. Semantic "
                                     "precision is the objective and hit count "
                                     "enters only as constraint C3 and "
                                     "tie-break 3.",
      "exactly_one_configuration_is_frozen": True,
      "frozen_to": "frozen_l1_precision_gate.json, written BEFORE l1_precision_heldout_replay.py",
  },

  "equivalence_jury": {
      "models": ["meta-llama/Llama-3.2-3B-Instruct",
                 "meta-llama/Llama-3.1-8B-Instruct",
                 "Qwen/Qwen2.5-7B-Instruct",
                 "mistralai/Mistral-7B-Instruct-v0.3"],
      "prompt": "identical to the paper's equivalence prompt "
                "(scripts/experiments/ablations/l1_precision_gate/"
                "l1_hit_generation_judging_protocol.py)",
      "decoding": "greedy, do_sample=False, max_new_tokens=6, chat template",
      "blinding": "the judge sees the two query strings only, never the policy "
                  "or arm",
      "tie_rule": "a 2-2 split is a TIE and is reported as its own category; "
                  "all three conventions (ties->DIFFERENT, ties excluded, "
                  "ties->SAME) are reported and none is hidden",
      "used_online": False,
  },

  "answer_audit": {
      "held_out_sample": "the SAME held-out audit "
                         "(outputs/general_audit/sample.json), request IDs asserted identical",
      "freshness_class_source": "COLLECTION CLASS -- the class carried by "
                                "tiered_cache_model.build_query_records, which is what "
                                "the manuscript headline held-out FreshCache "
                                "arm uses. The "
                                "predicted-class arm is NOT "
                                "used for the primary comparison.",
      "labelling_rule": "the manuscript rule, verbatim from "
                        "scripts/evaluation/answer_audits/general_audit_analysis.py: an answer matching the "
                        "abstention pattern is ABSTAIN; otherwise CORRECT only "
                        "if BOTH judges (Llama-3.1-8B and Qwen2.5-7B) say "
                        "CORRECT, else WRONG. Judge disagreement therefore "
                        "collapses to WRONG, exactly as in the manuscript.",
      "secondary_rule": "the three-way rule that separates JUDGE_DISAGREEMENT "
                        "is reported alongside, clearly labelled, and is never "
                        "substituted for the primary.",
      "reproduction_gate": "the recorded gate must reproduce the recorded held-out WAI "
                           "under the primary rule before any comparison "
                           "is believed.",
  },

  "success_criteria": {
      "FULLY_RESOLVED_requires_all": [
          "held-out ties->DIFFERENT L1 mismatch <= 10%",
          "overall FreshCache search savings fall by <= 0.10 percentage points",
          "held-out WAI does not increase materially",
          "no test-set tuning occurred",
          "the result holds on the ENTIRE realised L1 population, not only "
          "shared hits",
      ],
      "STRONGLY_ADDRESSED": "mismatch substantially below the held-out mismatch of the stricter 0.90 similarity floor but above 10%",
      "PARTIALLY_RESOLVED": "otherwise",
      "no_threshold_manipulation": "the grid and the selection rule are fixed "
                                   "here; neither will be altered to reach the "
                                   "10% criterion.",
  },

  "optional_second_stage_verifier": {
      "trigger": "ONLY if the best deterministic validation-selected cell still "
                 "has poor VALIDATION mismatch (declared here as > 15% under "
                 "ties->DIFFERENT on validation).",
      "constraint": "no LLM; inspect locally available cross-encoder / "
                    "reranker models first; a new model download must be "
                    "reported explicitly. Any pairwise threshold is tuned on "
                    "validation only.",
  },
  "seed": 42,
}


def main():
    PREREG["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    PREREG["l1_guards_sha256"] = sha((REPO_ROOT / "freshcache/method/l1_structural_guards.py"))
    out = (OUTPUT_ROOT / "l1_precision/l1_precision_prereg.json"); out.parent.mkdir(parents=True, exist_ok=True)   # definitions: configs/protocols/l1_precision_prereg.json
    if out.exists():
        sys.exit(f"REFUSING to overwrite an existing pre-registration: {out}\n"
                 f"  sha256 {sha(out)}\n"
                 f"  A pre-registration may be written once. Delete it "
                 f"deliberately if you truly intend to re-register.")
    out.write_text(json.dumps(PREREG, indent=2), encoding="utf-8")
    print("PRE-REGISTRATION WRITTEN (before any grid cell was evaluated)")
    print(f"  file           {out}")
    print(f"  sha256         {sha(out)}")
    print(f"  l1_structural_guards.py   {PREREG['l1_guards_sha256']}")
    print(f"  grid cells     {PREREG['grid']['n_cells']}")
    print(f"  objective      {PREREG['selection']['stage_1']['primary_objective']}")
    print(f"  constraints    C1 savings >= recorded - 0.10 pp; "
          f"C3 >= 100 validation L1 hits; C2 WAI via a NEW validation audit "
          f"(stage 2, K={PREREG['selection']['stage_1']['K']})")
    print(f"  WAI note       {PREREG['wai_measurability']['finding']}")
    print(f"  audit class    COLLECTION CLASS (manuscript headline)")


if __name__ == "__main__":
    main()
