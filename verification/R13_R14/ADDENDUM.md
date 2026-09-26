# Addendum to R13_R14_verification.md

Everything here was executed today on the current repository (`addendum_tests.py`, `sandbox/`); nothing outside `audit/R13_R14/` was modified. Tags as before: **[executed]**, **[historical record]**, **[unverified]**.

## 1. Half-life reconciliation: the display-rounding convention

**[executed]** Full-precision refit → TIMELESS floor → hours to days → one decimal → back to hours reproduces all four deployed values exactly:

| class | full-precision fit | days, 1 dp | back to hours | deployed |
|---|---|---|---|---|
| TIMELESS | 713.81 h | 30.0 (see note) | 720.0 | 720 |
| SLOW | 276.47 h | 11.5 | 276.0 | 276 |
| MEDIUM | 126.06 h | 5.3 | 127.2 | 127.2 |
| FAST | 124.94 h | 5.2 | 124.8 | 124.8 |

**[historical record] The convention exists in the actual entry point.** `calibrate.py:784-795` `_fmt_hl` prints fitted half-lives as `f"{seconds/86400:.1f}d"` below 14 days and `f"{seconds/86400:.0f}d"` above (`calibrate.py:779` prints the final table with it). `experiment.py:129-132` carries exactly those labels as comments ("5.2d", "5.3d", "11.5d", "30d"), its seconds equal the labels times 86,400 (5.3 × 86,400 = 457,920; 5.2 × 86,400 = 449,280; 11.5 × 86,400 = 993,600; 30 × 86,400 = 2,592,000), and `experiment.py:124-125` says the values were "synced manually". Running the entry point in the sandbox (§3) prints "30d / 11.5d / 5.3d / 5.2d".

Note on TIMELESS: the "30d" arises from the `.0f` display branch (2,569,717 s = 29.74 d), which is independent of the 30-day floor in `risk_model.py:440-452`; `calibrate.py` itself applies no floor. Either mechanism yields 720 h.

**What this does and does not establish.** It establishes numerical compatibility and a documented display-then-transcribe path that reproduces every deployed value from the current inputs. It does not establish that the historical run used identical inputs: the only on-disk report (`data/calibration_report.json`, 2026-06-30) records a different, 1h-only run, and the run behind `experiment.py` was never saved. The fitting was not altered. Section 4 of the verification report, which called MEDIUM/FAST unreproduced, is superseded by this section; blocker 1 in §7 reduces to "historical inputs unrecorded", not "values unreproduced".

## 2. TIMELESS/SLOW isolation, completed

**[executed]** With 1h/12h/24h retained and only the excluded rounds (7d, 48h) removed, then perturbed (their change events deleted, their denominators halved):

| class | condition | lookup | mask removed / tracked (fit windows) | weights | unclamped | operational | L3 equivalent TTL |
|---|---|---|---|---|---|---|---|
| TIMELESS | baseline / removed / perturbed | hand dictionary, constant | 2,553 / 9,217 in all three | window ages 3,600 / 43,200 / 86,400 s | 713.81 h in all three | 720 h (floor) | 447.47 h |
| SLOW | baseline / removed / perturbed | hand dictionary, constant | 1,138 / 12,992 in all three | same | 276.47 h in all three | 276.47 h | 171.82 h |

Nothing changes, so the excluded rounds do not enter the TIMELESS/SLOW fit either. Combined with the earlier MEDIUM/FAST test, no class's fit depends on any round outside its declared window.

**Lookup scope clarification.** The "fitting-rounds-only" control in the verification report used a **common 1h+12h cutoff** for every class. A class-specific variant (each class's own fit rounds, so 1h+12h+24h for TIMELESS/SLOW) gives TIMELESS 811.41 h vs 811.65, SLOW 287.44 h vs 291.99 (631 vs 421 removed), and identical MEDIUM/FAST (same rounds). Both are sensitivity numbers under a lookup that is not in force.

## 3. Full entry point in a disposable copy, two fresh processes

**[executed]** `sandbox/` holds copies of `calibrate.py`, `freshcache/`, `data/url_manifest.jsonl`, `data/change_log.jsonl`; `python calibrate.py` was run twice in fresh processes (`sandbox/run1.out`, `run2.out`).

- Both runs print identical half-lives (30d / 11.5d / 5.3d / 5.2d) and write identical reports: fit windows per class exactly as `FIT_WINDOWS_BY_CLASS`, half-lives 713.81 / 276.47 / 126.06 / 124.94 h.
- `_DEFAULT_HALF_LIFE` in the sandbox `risk_model.py` is patched to the full-precision seconds (449,768.7 / 453,810.3 / 995,303.1 / 2,569,717.4).
- **The domain dictionary changed by 0 entries in both runs.** `patch_risk_model` (`calibrate.py:575-585`) rewrites only the 20 most volatile observed domains that already exist in the dictionary; on current data all 20 have observed volatility 1.0 and their hand values are already 1.0, so the write is a no-op. This is why the leakage path in the code has never produced an effect; it would on different data.
- Stored `noise` flags disagree with the post-run `is_noise_url` on 0 of 3,246 rows: numerator and denominator filtering remain consistent across runs.

**Auxiliary all-round fitting.** `experiment.py:172` calls `_MLP_RISK.calibrate()` at import; that is `RuleBasedRiskModel.calibrate` (`risk_model.py:309-316`), whose `excluded_runs` defaults to none in the current code, so every non-baseline round including 24h/48h/7d feeds that instance's `_half_life` and `_domain_vol`. Consumers: `LearnedRiskModel._mlp_features` reads `self._domain_vol` (`risk_model.py:620`), and `p_stale_mlp` is reached only by the learned-variant path (`experiment.py:1585`). No primary result uses it; the manuscript names the learned variant as auxiliary (`freshcache_pro.tex:434`, `:729`) and cites it once for the earlier single-dataset inert-transfer observation (`:1374`).

## 4. The 19 calibration-error cells

**[executed]** Inclusion rule: `build_targets.py:116-117`, `if len(g) < 30: continue`. The REAL_TIME 7d cell has n = 20 and is therefore omitted; it is present in `l3_observations.csv`. Raw records 4,568; records inside the 19 retained cells 4,548. Per class, fitting rounds are those of `FIT_WINDOWS_BY_CLASS`; the cells that lie in each class's excluded rounds are FAST/MEDIUM at 24h and 7d, TIMELESS/SLOW at 7d, REAL_TIME everywhere (not fitted).

| class | age | n | observed raw | denoised | model | abs err (pp) | round status |
|---|---|---|---|---|---|---|---|
| REAL_TIME | 1h | 33 | 63.64 | 63.64 | 100.00 | 36.36 | not fitted |
| REAL_TIME | 12h | 34 | 76.47 | 76.47 | 100.00 | 23.53 | not fitted |
| REAL_TIME | 24h | 31 | 74.19 | 74.19 | 100.00 | 25.81 | not fitted |
| FAST | 1h | 286 | 13.29 | 13.29 | 0.55 | 12.73 | fit |
| FAST | 12h | 286 | 19.58 | 19.58 | 6.45 | 13.13 | fit |
| FAST | 24h | 285 | 19.30 | 19.30 | 12.48 | 6.82 | excluded |
| FAST | 7d | 279 | 31.90 | 31.54 | 60.67 | 29.12 | excluded |
| MEDIUM | 1h | 267 | 9.36 | 8.99 | 0.54 | 8.45 | fit |
| MEDIUM | 12h | 266 | 12.41 | 11.28 | 6.33 | 4.95 | fit |
| MEDIUM | 24h | 264 | 14.77 | 13.26 | 12.26 | 1.00 | excluded |
| MEDIUM | 7d | 262 | 23.66 | 22.90 | 59.97 | 37.07 | excluded |
| SLOW | 1h | 281 | 7.47 | 6.41 | 0.25 | 6.15 | fit |
| SLOW | 12h | 280 | 11.43 | 9.64 | 2.97 | 6.67 | fit |
| SLOW | 24h | 284 | 14.44 | 12.32 | 5.85 | 6.47 | fit |
| SLOW | 7d | 271 | 22.14 | 20.66 | 34.42 | 13.76 | excluded |
| TIMELESS | 1h | 289 | 4.15 | 2.77 | 0.10 | 2.67 | fit |
| TIMELESS | 12h | 288 | 3.82 | 2.43 | 1.15 | 1.28 | fit |
| TIMELESS | 24h | 289 | 4.15 | 2.77 | 2.28 | 0.48 | fit |
| TIMELESS | 7d | 273 | 13.19 | 9.16 | 14.93 | 5.78 | excluded |

Unweighted MAE 12.7496 pp; n-weighted 10.0783 pp; every model value recomputes from the deployed half-lives at multiplier 1.0. This population (validation-cluster, body-observable, denoised page observations) is not the population `calibrate.py` fits on, and the two are not mixed here.

## Provenance still missing

The inputs and code state of the run that produced `experiment.py`'s `HALF_LIFE`, and any dated record of the hand edits to the volatility dictionary and the TIMELESS ceiling. Both remain absent; neither is reconstructed.
