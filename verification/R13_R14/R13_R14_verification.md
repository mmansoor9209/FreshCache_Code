# R13 / R14 verification: later-round information in half-life calibration through domain-volatility filtering

Scope: the actual implementation and executed experiment, not comments or manuscript text. Nothing outside this folder was modified; `calibrate.main()` was never invoked (it patches `risk_model.py` and overwrites `data/calibration_report.json`). No live-Web or paid calls. Files here: `results.json`, `comparison.csv`, `run.log`, `code_excerpts.md`, `isolation_test.py`, `replay_control.py`, `replay_control.json`.

Every statement below is tagged **[executed]** (produced by scripts in this folder on the current repository), **[historical record]** (an artifact or comment that already existed), or **[unverified]** (asserted somewhere but not reproducible from what exists).

## 1. Entry points, implementation, inputs, consumer

| Item | File:line | Status |
|---|---|---|
| Calibration entry point | `calibrate.py:597-780` (`main`) | [historical record] current file dated 2026-09-09 |
| Fit windows | `calibrate.py:109-122` `FIT_WINDOWS_BY_CLASS` / `HOLDOUT_WINDOWS_BY_CLASS` | matches manuscript: MEDIUM/FAST 1h+12h; TIMELESS/SLOW 1h+12h+24h; REAL_TIME not fitted |
| Estimator | `calibrate.py:248-290` | age-weighted mean of closed-form λ from per-(round, class) aggregate change rates, rate clamped to [0.01, 0.99]; **not** a per-observation MLE and there are no per-URL weights; the only "weights" are the window ages |
| Domain filter at fit time | `calibrate.py:301-316` imports `is_noise_url` and applies it to the **denominator** (tracked URLs) only | executed below |
| Filter implementation | `freshcache/risk_model.py:169-186` `is_noise_url`, reading `_DEFAULT_DOMAIN_VOLATILITY` (`:96-153`) and `CLASS_VOL_CEILING` (`:158-164`) | the dictionary is **hand-curated**; comments at `:136-153` record values set by hand, e.g. "Volatility set just above the TIMELESS ceiling (0.45) so they are excluded from TIMELESS calibration" |
| Observed volatility (what the manuscript describes) | `calibrate.py:493-517` `compute_observed_domain_volatility`: uses `changed_1h` and `changed_24h` only, ≥2 observations, 3 d.p.; `risk_model.py:411-425` `RuleBasedRiskModel.calibrate` fits a second version from every non-baseline round | neither is what `is_noise_url` reads; see §2 |
| Patch step | `calibrate.py:525-585`, called at `:745` **after** the fit; overwrites only dictionary keys that already exist | [historical record] |
| Inputs | `data/url_manifest.jsonl` (89,494 rows; run_00 20,413; rerun_1h/12h/24h 19,326/19,326/19,377; rerun_48h 270; rerun_7d 10,782), `data/change_log.jsonl` (3,246 rows: 1h 522, 12h 675, 24h 752, 7d 1,297; 251 `noise=True`), both dated 2026-06-19 | [executed] counts |
| Numerator filter | `change_log.noise` was written by `recompute_changes.py:25,48-63` using the **same** hand dictionary; agrees with today's `is_noise_url` on 3,246/3,246 rows | [executed] |
| Replay consumer | `experiment.py:127-133` `HALF_LIFE` (720/276/127.2/124.8 h), comment at `:124-125`: "Synced manually from data/calibration_report.json"; `p_stale` at `:422-437` uses only `HALF_LIFE` and `TIER_MULT`, never domain volatility | [historical record] |
| TIMELESS floor | `risk_model.py:440-452` `MIN_HALF_LIVES`, TIMELESS 2,592,000 s = 720 h | [historical record] |

**Manuscript-producing version versus current code.** Three half-life sets coexist on disk, and no artifact records the run that produced the deployed one:
- `data/calibration_report.json` (2026-06-30 09:23) records a **1h-only** fit: MEDIUM 21.5 h, FAST 24.2 h, SLOW 28.3 h, TIMELESS 69.0 h. Its `fit_details.fit_window` is `['rerun_1h']` for every class, so it was produced by an earlier `calibrate.py` than the one on disk (2026-09-09). [historical record]
- `freshcache/risk_model.py.bak` (2026-06-30 05:59) holds FAST 214.7 h, MEDIUM 217.7 h, SLOW 276.5 h, TIMELESS 713.8 h, consistent with the "uniform 1h+12h+24h fit that did not separate MEDIUM from FAST" the docstring says was rejected. [historical record]
- `freshcache/risk_model.py` (2026-06-30 09:39) holds FAST 3.2 h, MEDIUM 15.1 h, SLOW 374.8 h, TIMELESS 518.6 h, labelled "calibrated from observed change rates", close to `experiment.py`'s `HALF_LIFE_PRE_W2_FIT_ON_TEST`. [historical record]
- `experiment.py` `HALF_LIFE` holds the deployed 720/276/127.2/124.8 h, synced by hand. **The run that produced it is not on disk.** [unverified provenance]

## 2. Trace: raw observations → lookup → mask → estimator → half-lives → replay

1. Raw observations: manifest rows with `snapshot_available` and `run_id != run_00`, per (round, class). [executed counts in `run.log` §1]
2. Domain lookup: `is_noise_url` reads the **hand dictionary** (43 entries), exact match then suffix match, default 0.40. The observed lookups (`calibrate.py:493`, `risk_model.py:411`) are computed but not consulted by the filter.
3. Inclusion mask: a URL is removed from the **denominator** of its (round, class) cell when its domain's dictionary value exceeds the class ceiling. Removed over the fit windows [executed]: TIMELESS 2,553 of 11,770; SLOW 1,138 of 14,130; MEDIUM 482 of 8,345; FAST 52 of 9,430. The numerator excludes `noise=True` rows, written with the same dictionary.
4. Estimator: per class, λ from each in-window rate, averaged with window-age weights, half-life = ln2/λ̄.
5. Half-lives: TIMELESS 713.81 h → 720 h by the floor; SLOW 276.47 h; MEDIUM 126.06 h; FAST 124.94 h. [executed]
6. Replay: `experiment.HALF_LIFE` only. The filter does not act at replay.

**Does the all-round lookup affect calibration?** No, because it never became the filter. `compute_observed_domain_volatility` yields 3,291 domains using 1h+24h (828 without 24h; 92 values change, 2,463 domains lost) [executed], so that lookup *does* depend on the 24h round, but only 5 of the 34 dictionary entries that also have observed values equal the observed value, and the `.bak` and v8-backup dictionaries are byte-identical to the current one. It is a separate, descriptive quantity in the code as it exists.

**Does the filter touch evaluation populations?** The mixed-age labels used for every primary result come from `v13_corrected/corrected_round_table.jsonl`, built in `a1_dataset_audit.py:160-172` from snapshot content hashes and substantiveness only; the `noise` flag is stored there as `previous_label` for comparison and not applied, and `mixed_engine.outcome` (`v16_exp12/mixed_engine.py:87-93`) reads only those labels. **Not filtered.** The legacy fixed-age path `experiment.build_stale_sets` (`experiment.py:566-578`) does drop `noise=True` rows. **Filtered**, via the same hand dictionary.

## 3. Isolation test [executed], `isolation_test.py`

Replication check: my re-implementation of `compute_run_class_rates` matches `calibrate.py`'s own function on all 25 (round, class) cells.

| Condition | TIMELESS | SLOW | MEDIUM | FAST |
|---|---|---|---|---|
| Baseline: current code, current inputs, hand-dict filter | 713.81 h | 276.47 h | 126.06 h | 124.94 h |
| All 24h/7d/48h records removed from manifest and change log | 419.16 h | 159.30 h | **126.06 h** | **124.94 h** |
| 24h/7d change events deleted, 24h/7d denominators halved | 419.16 h | 159.30 h | **126.06 h** | **124.94 h** |
| Positive control: in-window 12h round removed | 707.83 h | 285.38 h | 21.47 h | 24.24 h |

MEDIUM and FAST are bit-identical under removal and perturbation of every later-round record, and move by a factor of five when an in-window round is removed, so the fit is genuinely recomputed each time and carries no later-round dependence through the pipeline as it exists. TIMELESS and SLOW move because 24h is inside their fit window by design; that is documented in-sample use, not leakage.

Filter variants, same estimator and inputs (removed / tracked over each class's fit windows):

| Filter | TIMELESS | SLOW | MEDIUM | FAST |
|---|---|---|---|---|
| hand dictionary (in force) | 713.81 (2,553/9,217) | 276.47 (1,138/12,992) | 126.06 (482/7,863) | 124.94 (52/9,378) |
| no filter | 817.21 | 301.11 | 133.93 | 125.64 |
| observed 1h+24h (patch intent) | 805.94 | 289.52 | 128.80 | 122.03 |
| fitting rounds only, 1h+12h (control) | 811.65 | 291.99 | 128.44 | 121.51 |
| all four rounds (manuscript wording) | 811.26 | 287.31 | 127.36 | 119.03 |

## 4. Recomputation of the reported results

**Half-lives.** Against the deployed 720 / 276 / 127.2 / 124.8 h: TIMELESS reproduces through the floor; SLOW reproduces to the printed precision (276.47 vs 276.0); **MEDIUM (126.06 vs 127.2, 0.9%) and FAST (124.94 vs 124.8, 0.1%) do not reproduce exactly** under any tested variant: rate rounding at 2–5 d.p., numerator with noise rows kept, unfiltered numerator, unweighted λ mean, or no filter [executed, `run.log`]. The September provenance check (`artifacts/calibration/discrepancy_report.md`) obtained the same 276.47/126.06/124.94 and described the deployed set as reproduced "without the observability restriction"; that is approximate, not exact.

**Calibration error 12.75 pp** [executed]: it is the unweighted mean of |denoised observed − model| over 19 class-by-age cells of validation-only L3 page-change observations (`validation_V3/parameter_calibration/02_targets/targets_summary.json`), with model = 1 − 2^(−age/h) at the deployed half-lives and multiplier 1.0; every cell's model value recomputes exactly. Recomputed MAE 12.7496 pp; n-weighted 10.0783 pp. The cells sum to 4,548 observations; the stated 4,568 includes a REAL_TIME 7d cell of 20 observations that is in `l3_observations.csv` but absent from the 19-cell table. The five largest cells: MEDIUM 7d 37.07 pp (observed 22.90% vs model 59.97%), REAL_TIME 1h 36.36, FAST 7d 29.12, REAL_TIME 24h 25.81, REAL_TIME 12h 23.53. Full table in `results.json`. Note this population (validation clusters, body-observable, denoised) is not the population `calibrate.py` fits on.

## 5. Fitting-round-only control and replay [executed], `replay_control.py`

Later-round dependence was **not** confirmed for MEDIUM/FAST, so the control is reported as sensitivity, not as a correction. Full 31,201-request mixed-age replay of FreshCache, `exp.HALF_LIFE` patched in memory and restored:

| Half-life set | saved % | drift % (this run's determinable) | L1/L2/L3 hits | fetches | requests routed differently vs deployed | of the 400 audited |
|---|---|---|---|---|---|---|
| deployed | 62.732 | 3.527 (14,263) | 1,175/18,398/33,907 | 10,959 | 0 | 0 |
| recomputed, hand dict | 62.681 | 3.533 (14,211) | 1,175/18,382/33,825 | 10,988 | 172 | 2 |
| observed 1h+24h lookup | 62.860 | 3.494 (14,194) | 1,203/18,410/33,780 | 10,947 | 719 | 10 |
| fitting-rounds-only lookup | 62.851 | 3.523 (14,193) | 1,204/18,406/33,787 | 10,965 | 783 | 9 |
| no filter | 63.399 | 3.600 (14,250) | 1,230/18,551/33,855 | 10,820 | 1,111 | 18 |

Drift here is over each run's own determinable requests, not the paper's 17,877-request policy-independent common support, so it is not comparable to the printed 3.42%. Search savings move by at most 0.67 points across all variants. Routing changes are small but non-zero: even the recomputed set alters 172 requests, 2 of them inside the 400-request answer audit, so **answer-audit results cannot be reused unchanged for any variant**; a matched rerun of the audit would be required before any WAI claim for a refitted set, and none was run here.

## 6. Manuscript claims, verified or not

| Claim | Finding |
|---|---|
| MEDIUM/FAST fit on 1h and 12h; TIMELESS/SLOW on 1h, 12h, 24h; REAL_TIME 30 s prior | **verified** in `calibrate.py:109-122` and by execution |
| TIMELESS 720 h minimum | **verified**, `risk_model.py:440-452`; the unclamped fit is 713.81 h |
| Domain volatility uses all four rerun rounds and is measured, not assigned | **not supported.** The filter in force reads a hand-curated dictionary; the observed lookup in `calibrate.py` uses 1h+24h; the all-round version exists only in `RuleBasedRiskModel.calibrate`, which the primary replay does not call. The manuscript's Appendix B wording and `domain_filter_ablation.json`'s volatility column (which echoes the hand values) misdescribe the quantity that gates the fit |
| Half-lives 720/276/127.2/124.8 h | TIMELESS and SLOW reproduce; MEDIUM and FAST are within 1% but **unreproduced exactly**, and the producing run is not on disk |
| Calibration error 12.75 pp | **verified exactly**, with the population and the 20-observation gap stated above |

## 7. Remaining blockers

1. The run that produced the deployed MEDIUM/FAST values is not recorded; only a new, documented calibration run would give a reproducible pair. Not done here, since that would change deployed constants.
2. The domain filter's provenance is the in-code comments alone: no date, no data-driven derivation, no selection log. The comment "was 0.25, raised so unknown domains (default 0.40) pass through" is the only record of the TIMELESS ceiling change.
3. `calibrate.py`'s patch mechanism means any future full run *would* write 1h+24h observed values into the filter and thereby introduce the leakage the manuscript denies; the code path should be removed or fixed before any recalibration.
4. `risk_model.py` currently ships a third, pre-W2-like half-life set and its `calibrate()` no longer excludes 24h/7d (the `.bak` did); the learned-variant path that calls it at import fits on all rounds.
5. Answer audits for any refitted half-life set require a matched rerun (routing differs on audited requests).
6. All of §5 is a revision-stage check executed today; it is not, and must not be described as, a historical preregistration.
