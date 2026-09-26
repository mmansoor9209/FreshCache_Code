# Observability audit: addendum

Date: 2026-09-24. Read-only extension of `Observability_verification.md`. Nothing in the manuscript, data, implementation or original results was changed. All new numbers come from `addendum_recompute.py` (output `run_addendum.log`, `addendum_results.json`, `class_count_tables_addendum.csv`). Code cited below is reproduced verbatim in `code_excerpts_addendum.md`. The original audit's `observability_results.json` and `class_count_tables.csv` are attached unchanged in this folder.

**[EXECUTED]** = recomputed now from raw files. **[RECORD]** = existing study output. **[UNVERIFIED]** = not checkable locally.

## 1. Snapshot percentages, mutually exclusive categories [EXECUTED]

Rule applied per snapshot, in order: status ≠ 200 → non-200; stripped text ≥400 chars and no block-page regex in the first 400 chars → substantive; ≥400 chars with the regex → block page; 0 chars → empty; otherwise → short text. This is the order `v9/build_version_timeline.py:46-49` implies (length is tested before the regex, so a short block page counts as short text).

| round | snapshots | substantive | short text (<400) | block page (≥400) | non-200 | empty | non-substantive % | short-text % |
|---|---|---|---|---|---|---|---|---|
| run_00 (baseline) | 8,635 | 4,559 | 4,058 | 18 | 0 | 0 | **47.2032** | 46.9948 |
| rerun_1h | 8,429 | 4,394 | 4,001 | 34 | 0 | 0 | 47.8704 | 47.4671 |
| rerun_12h | 8,358 | 4,394 | 3,930 | 34 | 0 | 0 | 47.4276 | 47.0208 |
| rerun_24h | 8,348 | 4,387 | 3,927 | 34 | 0 | 0 | 47.4485 | 47.0412 |
| rerun_7d | 8,168 | 4,228 | 3,903 | 37 | 0 | 0 | 48.2370 | 47.7840 |
| five named rounds | 41,938 | 21,962 | 19,819 | 157 | 0 | 0 | **47.6322** | 47.2579 |
| rerun_48h (extra, unused) | 160 | 90 | 70 | 0 | 0 | 0 | 43.7500 | 43.7500 |
| six rounds incl. 48h | 42,098 | 22,052 | 19,889 | 157 | 0 | 0 | **47.6175** | **47.2445** |

All four stated figures verify: baseline 47.20 %, five-round 47.63 %, six-round 47.62 %, six-round short-text-only 47.24 %. The study's taxonomy (22,052 / 19,889 / 157) [RECORD] is the six-round total. Short text is reported as short text; whether it is a title is not recorded anywhere and is not claimed here. The manuscript's "about 42 %" matches none of these rows.

## 2. "At most two URLs per search" versus 1 to 5 URLs per request [EXECUTED]

**Retrieval.** Every run_00 manifest row carries rank 1 or 2 (10,216 rank-1, 10,197 rank-2 rows; no other rank value). So each Serper call did return at most two URLs. The manuscript sentence is true per search call.

**Why a request can hold more.** 1,032 questions have two run_00 searches with different retrieval timestamps (1,030 with 4 rows, 2 with 5), one on 2026-05-15 and one on 2026-06-01. `experiment.build_query_records:503-518` takes every run_00 row with `snapshot_available` for the question, sorts by rank and applies no cap and no deduplication; paraphrases inherit the base list (:541-556). Rows whose snapshot failed are dropped at that point, which is why lists of 1 and 3 exist.

| URLs in record | base questions | all 31,201 requests | distinct url_hash (base) |
|---|---|---|---|
| 1 | 3,906 | 19,474 | 4,036 |
| 2 | 1,883 | 9,386 | 1,925 |
| 3 | 183 | 912 | 220 |
| 4 | 285 | 1,424 | 77 |
| 5 | 1 | 5 | 0 |

434 base records repeat a `url_hash` (the rank-1 URL returned by both searches). After deduplication the maximum is 4 distinct URLs and only 297 questions have more than two. The repeated entry is harmless for observability (set membership) but is a real list entry for fetch and evidence counts in the replay; that effect is outside this audit.

**Examples with more than two URLs** (run_00 rows in timestamp order):
- `q_000001` (4 URLs) "Who holds the all-time record at the Grammys for the most wins in the album of…": rank 1 `015ecb20`, rank 2 `344b4f27` at 2026-05-15 07:30; rank 1 `015ecb20`, rank 2 `eb4a52dd` at 2026-06-01 09:38. Record list = [015ecb20, 015ecb20, 344b4f27, eb4a52dd].
- `q_000013` (3 URLs): rank 2 of the first search (`349037b0`) has no snapshot and is dropped; the second search adds `49ebb5e3`.
- `q_000015` (4 URLs) and `q_000021` (3 URLs): same pattern.

**Fetching and aggregation.** Reruns fetch the URLs by hash (`data/snapshots/<round>/<hash>.json`), so a URL retrieved twice is fetched once per round; the panel of 8,635 unique URLs is unaffected. Which of the two searches was the "one Serper search" the manuscript refers to is not recorded [UNVERIFIED]. Recommended wording: "each search returns at most two URLs; 1,032 questions were searched twice at baseline, so a request's evidence list holds one to four distinct URLs".

## 3. 16,813 paired held-out requests versus 12,568 common-support requests [EXECUTED]

Both sets were rebuilt on the same held-out stream (zipf_uniform, seed 42, 21,848 test-cluster requests) and the same `v13_corrected/corrected_round_table.jsonl`. The paired set was read from the recovery study's per-request file; the support set was recomputed with `e1_robustness.support_of`.

| set | rule | n | MEDIUM | SLOW | TIMELESS | FAST |
|---|---|---|---|---|---|---|
| common support S | non-REAL_TIME request with ≥1 URL substantive at run_00 **and** at `version_at(arrival t)` (`e1_robustness.py:50-61`) | 12,568 | 3,039 | 3,233 | 2,715 | 3,581 |
| paired | both arms' `mixed_engine.replay` emit a non-None outcome; the outcome is CHANGED / UNCHANGED / UNOBSERVABLE (`recover_and_recompute.py:152-153`; `mixed_engine.py:223-228, 288-290`) | 16,813 | 3,913 | 4,747 | 3,963 | 4,190 |
| S ∩ paired | | 9,864 | | | | |
| paired \ S | | 6,949 | 1,655 | 2,035 | 1,729 | 1,530 |
| S \ paired | | 2,704 | 781 | 521 | 481 | 921 |
| jointly observable at 400 chars | paired and neither outcome UNOBSERVABLE | 9,184 | | | | |

Neither set contains the other, and the jointly observable set (9,184) is not a subset of S either. The 6,949 paired-only requests have **no** URL substantive at run_00 and at their arrival round (0 of 6,949), yet 5,938 carry UNOBSERVABLE in both arms and 1,011 carry a determinate label in at least one arm. That is consistent with the two definitions: `support_of` looks at the request's own URL list against run_00, while the replay outcome compares the version cached at `t_cached` with the version at `t` for the URLs actually served, which can be another request's L1/L2 URLs and need not involve run_00. The 2,704 support-only requests got a None outcome in at least one arm, which `mixed_engine.py:288-290` emits when no served URL produced a label (for example a fresh miss). Same stream, same round table, but different replay configuration (one policy versus a two-arm comparison) and different unit of observation. The two numbers are not interchangeable and no equivalence is inferred.

## 4. Corrected deduplicated-table arithmetic [EXECUTED]

The original audit rounded from 2-dp percentages. Exact values:

| unit | N | χ² | Cramér's V | mean(FAST, REAL_TIME) − mean(TIMELESS, SLOW) | TIMELESS | SLOW | MEDIUM | FAST | REAL_TIME |
|---|---|---|---|---|---|---|---|---|---|
| (a) (query, URL) rows, the study's unit | 17,351 | 121.5301 | 0.083691 | **8.258638 pp** | 58.1895 | 57.0864 | 60.8190 | 58.5251 | 73.2681 |
| (b) unique (URL, class) pairs | 7,958 | 15.0270 | **0.043454** | **5.834447 pp** | 55.4348 | 53.2629 | 55.0879 | 53.3395 | 67.0270 |
| (c) unique single-class URLs (228 multi-class excluded) | 7,495 | 12.8947 | 0.041478 | **4.888290 pp** | 55.2191 | 53.4146 | 55.5685 | 52.8736 | 65.5367 |

Corrections to `Observability_verification.md` §3: "about 5.9 points (unit b) or 5.1 points (unit c)" should read 5.83 and 4.89 points; "V = 0.044" for unit b should read 0.0435. The row-level 8.26 points and V = 0.084 (manuscript) stand as arithmetic on unit (a). The three analyses remain distinct and are not averaged.

## Files added

`Observability_addendum.md`, `addendum_recompute.py`, `addendum_results.json`, `class_count_tables_addendum.csv`, `code_excerpts_addendum.md`, `run_addendum.log`. Attached unchanged from the first pass: `observability_results.json`, `class_count_tables.csv`.
