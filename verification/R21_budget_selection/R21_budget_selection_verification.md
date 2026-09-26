# R21 (budget selection): verification of the validation-selected FreshCache budgets

Audit date: 2026-09-24. Read-only: no code, data, manuscript or existing output was modified. New files live only in this folder: `r21_budget_recompute.py`, `R21_budget_results.json`, `candidate_grid_48.csv` (all 48 rows), `code_excerpts.md`, `run.log`.

**[EXECUTED]** = recomputed now from saved row-level records. **[RECORD]** = file written by the study on 2026-09-22. **[UNVERIFIED]** = not checkable from local files.

## 0. Where the claims come from

Manuscript (WWW V3): "Validation-only budget selection raises held-out avoidance from 60.58% to 70.24% with no detected drift penalty (−0.11 pp, p = 0.33)" (§5, Parameter selection); "On the validation split, 10 of 48 budget triples satisfy the pre-defined savings, drift and coverage constraints; ε_L2 is the dominant lever (42.08% to 67.49% savings from 0.10 to 0.35), ε_L1 moves savings by 0.14 points, and ε_L3 leaves savings at 56.87% while drift rises from 2.47% to 4.16%" (Appendix B, budget sensitivity). The selected triple (0.10, 0.35, 0.25) itself is not printed in the PDF text I extracted; it appears in the study records below.

Producing study: `validation_V3/parameter_calibration/` (all files 2026-09-22 05:02 to 05:46 UTC). Key files: `04_budgets/prereg.json` (+ `.sha256`), `04_budgets/sweep_validation.py`, `04_budgets/validation_sweep.json`, `04_budgets/frozen_config.json`, `05_heldout/fit_and_evaluate.py`, `05_heldout/heldout_results.json`, `05_heldout/heldout_per_request.csv`, `07_answer_quality/answer_quality.json` and `answer_quality_rows.csv`, `logs/t4.log`, `logs/heldout.log`.

Note on wording: the manuscript sentence says "savings, drift and coverage constraints". The recorded constraints are drift, L1 hits and coverage; savings is the objective, not a constraint (§3).

## 1. Candidate grid [RECORD, re-derived EXECUTED]

`prereg.json` `search_space` (verbatim in `code_excerpts.md`):

| parameter | values | count |
|---|---|---|
| ε_L1 (`eps_answer`) | 0.05, 0.10, 0.15, 0.20 | 4 |
| ε_L2 (`eps_url_list`) | 0.10, 0.20, 0.30, 0.35 | 4 |
| ε_L3 (`eps_content`) | 0.25, 0.35, 0.45 | 3 |

Cartesian product 4 × 4 × 3 = 48 (`sweep_validation.py:41-44` builds it with `itertools.product` and asserts 48). The 48 keys in `validation_sweep.json["cells"]` equal this product exactly [EXECUTED]. Multipliers were held at (1.5, 1.2, 1.0) and half-lives at their published values; the prereg states that (m, ε) enter the gate only through k_t = −ln(1−ε)/(m ln 2), so varying m would be redundant. All 48 triples produce 48 distinct policies (no decision-equivalent pairs, `t4.log`).

## 2. Validation population [RECORD, EXECUTED for the split file]

- Split: `validation/heldout_baseline_tuning/split.json`, cluster-disjoint, stratified by freshness class, seed 42, 30/70. Validation = 1,877 clusters / 9,353 requests; test = 4,381 clusters / 21,848 requests; overlap 0.
- Stream: the full 31,201-request stream built once with schedule `zipf_uniform`, seed 42 (`sweep_validation.py:24, 63-65`), then filtered to validation clusters; `assert len(val) == 9353` and a leakage assertion against test clusters (`:66-68`).
- Denominators (`sweep_validation.py:73-87`): every metric is taken from one `mixed_engine.replay(val, rounds, "FreshCache")` call over the 9,353 requests. `search_saved_pct` is the engine's savings over all 9,353 requests. Coverage = determinate outcomes (CHANGED or UNCHANGED) / 9,353. Drift = CHANGED / determinate. L1 hits = the engine's count over the 9,353 requests. The published configuration evaluated on this population gives savings 56.8695%, drift 3.3987%, coverage 43.0985%, L1 285 [RECORD], which matches the published-arm entry of the independent stricter-gate study (`l1_precision_gate/stage1_selection.json`: 56.8695 / 285) [EXECUTED cross-check].

## 3. Hard constraints [RECORD, values recomputed EXECUTED]

From `prereg.json` and `sweep_validation.py:112-120`, all relative to the published configuration's validation run:

| constraint | rule | numerical value |
|---|---|---|
| C1 drift | drift ≤ published + 0.25 pp | ≤ 3.6487% |
| C2 L1 hits | L1 hits ≤ published | ≤ 285 |
| C3 coverage | coverage ≥ 0.95 × published | ≥ 40.9436% (0.95 × 43.0985) |
| savings | none (objective only) | — |
| answer quality | none ("WAI is NOT part of the selection objective", prereg) | — |

No other eligibility criterion exists in the prereg or the code. No threshold is inferred; the three values are read from `validation_sweep.json["constraints"]` and recomputed from the published validation row.

## 4. Selection objective and tie-breaks [RECORD]

`prereg.json`: primary = maximise validation `search_saved_pct` among feasible triples; tie-breaks in order T1 lower drift, T2 fewer L1 hits, T3 lexicographically smallest (ε_L1, ε_L2, ε_L3). As coded, `sweep_validation.py:122-124` sorts feasible cells by `(-search_saved_pct, drift_pct, l1_hits, eps_answer, eps_url_list, eps_content)` and takes the first, which implements exactly that order.

## 5. Recomputation of the selection [EXECUTED]

Applying the constraints to the 48 saved cells: **10 eligible** (recorded 10). Ranking:

| rank | (ε_L1, ε_L2, ε_L3) | savings | drift | coverage | L1 |
|---|---|---|---|---|---|
| 1 | **(0.10, 0.35, 0.25)** | 67.4864 | 2.6802 | 41.0884 | 285 |
| 2 | (0.05, 0.35, 0.25) | 67.4864 | 2.6809 | 41.0777 | 167 |
| 3 | (0.10, 0.30, 0.25) | 64.5354 | 2.6739 | 41.1846 | 285 |
| 4 | (0.05, 0.30, 0.25) | 64.5141 | 2.6781 | 41.1205 | 167 |
| 5 | (0.10, 0.20, 0.35) published | 56.8695 | 3.3987 | 43.0985 | 285 |
| 6 | (0.05, 0.20, 0.35) | 56.8267 | 3.4003 | 43.0771 | 167 |
| 7 | (0.10, 0.10, 0.35) | 42.0828 | 2.5475 | 42.8098 | 285 |
| 8 | (0.10, 0.10, 0.45) | 42.0828 | 3.0780 | 44.8092 | 285 |
| 9 | (0.05, 0.10, 0.35) | 41.9545 | 2.5475 | 42.8098 | 167 |
| 10 | (0.05, 0.10, 0.45) | 41.9545 | 3.0780 | 44.8092 | 167 |

The primary objective is **tied** at 67.4864% between (0.10, 0.35, 0.25) and (0.05, 0.35, 0.25). Tie-break T1 decides: drift 2.6802% versus 2.6809%, a 0.0007 pp difference (one CHANGED outcome). T2 and T3 were not reached. Under T2 alone the other triple would have won (167 < 285 L1 hits), so the selection depends on the pre-registered order of the tie-breaks. The selected triple, its metrics (67.4864 / 2.6802 / 41.0884 / 285) and the published validation values (56.8695 / 3.3987 / 43.0985 / 285) all match the records and the manuscript's 67.49% and 56.87%.

Why 38 triples fail: every ε_L1 ∈ {0.15, 0.20} triple fails C2 (385 or 457 L1 hits > 285); ε_L3 = 0.25 with ε_L2 ≤ 0.20 fails C3 (coverage 40.37 to 40.81%); ε_L3 ∈ {0.35, 0.45} with ε_L2 ≥ 0.30, and ε_L3 = 0.45 with ε_L2 = 0.20, fail C1. Per-cell flags are in `candidate_grid_48.csv`.

## 6. Held-out evaluation of the frozen triple (configuration C) [RECORD; drift, coverage and WAI EXECUTED]

`fit_and_evaluate.py` reads `frozen_config.json` and its SHA-256 (`a05df970…`), replays the 21,848 held-out requests / 4,381 clusters for five configurations, and aborts unless configuration A reproduces the published anchor 60.5776% / 806 / 12,429 (`:34, :152-155`); `heldout.log` records "ANCHOR GATE PASSED".

| quantity | manuscript / claim | record (`heldout_results.json`) | recomputed from `heldout_per_request.csv` |
|---|---|---|---|
| N | 21,848 | 21,848 | 21,848 rows |
| search savings | 70.24% | 70.2444% | not recomputable (file holds outcomes only, not tiers or search calls) |
| drift | 3.05% | 3.0543% (286 / 9,364) | **3.0543%** |
| coverage | 42.86% | 42.8598% (9,364 / 21,848) | **42.8598%** |
| L1 / L2 / L3 hits | — | 806 / 14,541 / 22,395 | not recomputable from this file |
| WAI | 7/400 | 7 (`answer_quality.json`) | **7/400**; all seven are L2-tier requests |

Drift here is CHANGED / (CHANGED + UNCHANGED) over the whole held-out stream; the 4,950 (A) and 5,629 (C) requests with a None outcome (fresh misses, REAL_TIME) and the UNOBSERVABLE ones are outside the denominator. The paired contrast (−0.11 pp, p = 0.33) is a [RECORD] from `paired_contrasts.json` and was not recomputed. WAI detail [EXECUTED]: A and C share six WAI request IDs; A's seventh is `p_q_5abb871e_2`, C's is `p_q_3cae62d9_1`, so "WAI stays at 7/400" is a count equality, not an event-for-event identity. The 400 audit IDs equal the published held-out audit (study's gate, [RECORD]).

## 7. Was the triple frozen before held-out evaluation?

Evidence, all revision-stage:
- `prereg.json` written 05:06:54 UTC, hashed to `prereg.sha256` at the same second; `sweep_validation.py:35-39` refuses to run if the hash differs.
- `frozen_config.json` and `logs/t4.log` written 05:16:41 (end of the validation sweep, which touched only the 9,353 validation requests and asserted zero test-cluster leakage).
- `heldout_results.json`, `heldout_per_request.csv`, `logs/heldout.log` written 05:23:25; the script records the frozen file's hash and `heldout_touched_during_selection: false`.

So within this study the ordering prereg → sweep → freeze → held-out replay is documented by file timestamps, hashes and in-script assertions [RECORD, timestamps EXECUTED]. Two limits must be stated. First, all of this dates from 2026-09-22, after the published configuration existed; the prereg itself says it "is NOT a pre-registration of the original paper's choices". There is no historical record of how (0.10, 0.20, 0.35) was chosen, and none is claimed. Second, the same author process ran both steps a few minutes apart; the freeze is an internal commitment, not an externally timestamped one.

## 8. Remaining blockers

- Held-out savings 70.2444% and the tier hit counts rest on the study's replay record plus the anchor gate; the per-request file cannot regenerate them. Re-running the engine would be needed for an independent check and was outside this audit's no-replay scope.
- The selection hinges on a tie broken by a 0.0007 pp drift difference under a pre-declared tie-break order; the manuscript does not mention the tie.
- The manuscript's "savings, drift and coverage constraints" misnames the constraint set (drift, L1 hits, coverage).
