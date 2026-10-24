"""Write and hash the validation-only pre-registration (outputs/l1_validation/config/prereg.json and prereg.sha256) BEFORE any replay, generation or judging in scripts/experiments/ablations/l1_validation/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, hashlib, pathlib, itertools, datetime; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
W = ((OUTPUT_ROOT / "l1_validation"))
GUARDS = ["numeric", "negation", "comparative"]
subsets = [list(c) for n in range(4) for c in itertools.combinations(GUARDS, n)]
P = {
 "written_utc": datetime.datetime.utcnow().isoformat() + "Z",
 "scope": "L1 admission only; validation clusters only (outputs/validation_split/split.json). Held-out test clusters are never loaded.",
 "fixed_rules": {"jaccard_floor": 0.30, "answer_type": "as_implemented (reject only when both types detectable and different)", "strict_subset_rule": "on",
                 "entity_match": "unchanged (engine-level)", "temporal_rules_budgets_multipliers_half_lives": "unchanged", "L2_L3": "unchanged",
                 "stream": "zipf_uniform, seed 42", "replay": "outputs/mixed_age_audit/audit_replay.replay, FreshCache variant, true sequential replay per cell"},
 "grid": {"cosine_floor": [0.80, 0.85, 0.88, 0.90, 0.92, 0.95], "guard_subsets": subsets,
          "guards_source": "l1_structural_guards.py copied from outputs/l1_precision (sha256 must equal 952e8c9cb59a796328481bc8cc23d7dc9042db09cd96980d6be9fecc972c3873)",
          "n_cells": 6 * len(subsets)},
 "reproduction_gates": ["cell (0.80, no guards) reproduces the recorded rule's validation replay request for request",
                        "cell (0.90, all guards) reproduces the earlier study's validation cell s0.9_j0.3_as_implemented_G (L1 hits and savings)"],
 "metrics": {
   "l1_hits": "realized L1 hits in the sequential validation replay",
   "mismatch_pct": "four-juror query-equivalence mismatch among realized L1 hits, 2-2 ties counted as DIFFERENT (paper convention); jurors Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, verbatim equivalence prompt, greedy, 6 tokens",
   "search_saved_pct": "1 - search calls / validation requests", "generation_saved_pct": "1 - (search calls + L2 hits) / validation requests",
   "hit_harm": "on every realized L1 hit: served stored answer WRONG and fresh-path answer CORRECT for the incoming question (manuscript rule: abstention pattern -> ABSTAIN; CORRECT only if Llama-3.1-8B and Qwen2.5-7B both say CORRECT; else WRONG); dataset reference answers",
   "audit_wai": "existing validation answer audit (outputs/l1_precision/validation_audit_sample.json): fresh CORRECT and cell WRONG",
   "drift_coverage": "reported, not used for selection"},
 "selection": {
   "reference": "the recorded rule (0.80, no guards) evaluated on the same validation replay",
   "hard_constraints": ["C1 retention: L1 hits >= 50% of the recorded rule's validation L1 hits",
                        "C2 search savings >= recorded - 0.10 percentage points",
                        "C3 hit-level harm count <= recorded hit-level harm count",
                        "C4 audit WAI <= recorded audit WAI"],
   "objective": "minimise mismatch_pct among cells satisfying C1-C4",
   "tie_breaks_in_order": ["lower hit-level harm rate (harm / L1 hits)", "more L1 hits", "lower cosine floor", "fewer guards", "guard subset in grid order"],
   "if_no_cell_passes": "freeze nothing and report",
   "no_held_out_use": "the selection reads validation outputs only; the held-out L1 hits and all held-out results are not read by any script in this folder"},
 "disclosure": ("This grid and criterion were written after earlier work had already evaluated a (0.90, all guards) rule on the held-out split "
                "and after held-out L1 diagnostics; the design is therefore not blind to held-out results even though selection uses validation data only."),
 "seed": 42}
json.dump(P, open(W/"config/prereg.json", "w"), indent=1)
open(W/"config/prereg.sha256", "w").write(hashlib.sha256((W/"config/prereg.json").read_bytes()).hexdigest() + "\n")
print("cells", P["grid"]["n_cells"], open(W/"config/prereg.sha256").read().strip())
