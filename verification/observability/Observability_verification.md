# Observability population accounting: verification

Audit date: 2026-09-24. Read-only. No manuscript, code, data or original result was modified. No live-Web call, new collection or model generation. All recomputation is in `observability_recompute.py` (this folder); console output in `run.log`; structured output in `observability_results.json`; count tables in `class_count_tables.csv`. Verbatim, line-numbered code for every rule cited below is in `code_excerpts.md`, so no other file is needed to follow the findings.

Legend: **[EXECUTED]** check I ran now; **[RECORD]** existing file produced by the original study, not re-run; **[UNVERIFIED]** stated but not checkable locally.

## 1. Summary table (the required format)

| metric | unit | rounds / split | numerator | denominator | reported | recomputed | filtering | source file:line |
|---|---|---|---|---|---|---|---|---|
| Body-observable URLs at 24h | unique URL (`url_hash`), one run_00→rerun_24h transition | run_00 and rerun_24h; full panel, no split | 4,382 | 8,635 | 4,382 / 8,635 | **4,382 / 8,635** [EXECUTED, rebuilt from the snapshot files] | substantive at run_00 AND at rerun_24h; substantive = stripped `extracted_text` ≥400 chars and no block-page regex in the first 400 chars | `v9/build_version_timeline.py:40-49`; `v10_remaining_feedback/e1_c1_observable_refit.py:85-87` |
| Body-observable request pool | replay request (base query or paraphrase) | fixed 24h horizon; full 31,201 stream, no split; REAL_TIME included | 18,216 | 31,201 | 18,216 / 31,201 | **18,216 / 31,201** [EXECUTED] | request kept if **at least one** of its URLs is in the 4,382 set | `e1_c1_observable_refit.py:194-195`; `validation/observability_by_class/analyze.py:256-262` |
| Class observability (5 values) | manifest row = (query, URL) pair with a rerun_24h snapshot | rerun_24h rows only; full panel | 1,787 / 2,457 / 2,406 / 2,746 / 973 | 3,071 / 4,304 / 3,956 / 4,692 / 1,328 (sum 17,351) | 58.19 / 57.09 / 60.82 / 58.53 / 73.27 % | **identical** [EXECUTED] | `snapshot_available`, `run_id == rerun_24h`, `is_noise_url` domain filter; URL observable iff in the 4,382 set | `analyze.py:169-178` |
| Class association | same 17,351 (query, URL) rows | same | — | 5×2 table above | χ²=121.53, df 4, p=2.5e-25, V=0.084 | **χ²=121.5301, df 4, p=2.52e-25, V=0.0837** [EXECUTED, scipy] | same | `analyze.py:233-247` |
| Same test on unique (URL, class) pairs | unique (URL, class) | same | 867 / 1,159 / 1,034 / 1,150 / 124 | 1,564 / 2,176 / 1,877 / 2,156 / 185 (sum 7,958) | not reported | χ²=15.03, p=0.0046, V=0.044; 55.43 / 53.26 / 55.09 / 53.34 / 67.03 % [EXECUTED] | same filter, deduplicated | this audit |
| Same test on unique single-class URLs | unique URL (228 multi-class URLs excluded) | same | 857 / 1,095 / 953 / 1,058 / 116 | 1,552 / 2,050 / 1,715 / 2,001 / 177 (sum 7,495) | not reported | χ²=12.89, p=0.012, V=0.041; 55.22 / 53.41 / 55.57 / 52.87 / 65.54 % [EXECUTED] | same, deduplicated | this audit |
| Paired observability, 400-char rule | held-out replay request, paired (both arms emit an outcome) | held-out test split, seed 42, mixed-age stream of 21,848; 16,813 paired | 9,184 | 16,813 | 54.62 % | **54.62 %** (9,184 / 16,813) [EXECUTED from `per_request_recovered.csv`] | non-REAL_TIME requests with a comparison snapshot in both the FreshCache and the AlwaysPassTemporal arm; jointly observable = neither arm's outcome is UNOBSERVABLE | `recover_and_recompute.py:152-161,176-178` |
| Paired observability, 50-char rule | same | same | 13,396 | 16,813 | 79.68 % | **79.68 %** [EXECUTED] | short texts (<400 chars, status 200, no block pattern) re-admitted at ≥50 chars; nothing re-fetched | `recover_and_recompute.py:68-107` |

Every reported number reproduces. The problems are in labelling and in the unit of the class analysis, not in arithmetic.

## 2. What each measurement actually counts (items 1 and 2)

**Panel and rounds.** 8,635 URLs have a run_00 snapshot. Rerun snapshots: 1h 8,429; 12h 8,358; 24h 8,348; 7d 8,168; plus 160 at a 48h round that no observability figure uses. Total (URL, round) observations: 41,938 over the five named rounds, 42,098 including 48h [EXECUTED]. No train/validation split is involved in any URL-level or class-level figure. Only the paired-missingness figures live on the held-out test split.

**The substantive flag, as coded.** `v9/build_version_timeline.py:46-49` tests text length after strip ≥400 and the absence of the block-page regex in the first 400 characters. It does **not** test the HTTP status. The later round table (`v13_corrected/a1b_round_table.py:53`) adds `status_code == 200`. The two flag sets agree on all 41,938 entries and give the same 4,382 URLs, because every stored snapshot has status 200 [EXECUTED: status histogram shows only 200 in every round]. So the manuscript's "carry HTTP 200" clause is true but vacuous; the operative checks are the length and the block regex. Observation taxonomy [EXECUTED]: 21,962 substantive, 19,819 short text (<400), 157 block page, 0 non-200, 0 empty. The study's figures (22,052 / 19,889 / 157) include the 48h round.

**"About 42% of snapshots are title-only."** Not supported by these records. The non-substantive share is 47.20 % at run_00 and 47.24 % over all observations [EXECUTED]. The "42 %" wording originates in the reviewer's own text quoted in `v9/build_version_timeline.py:15` and `v9/observable_rescore.py:5`, not in a measurement. Recommended: "about 47 %".

**4,382 / 8,635.** Unit is the unique URL. A URL qualifies iff substantive at run_00 and at rerun_24h. Denominator is all 8,635 panel URLs, including the 287 with no 24h snapshot at all (8,635 − 8,348). Deduplication is inherent (one row per `url_hash`); no class, no query, no split. Rebuilt from the raw snapshot files with zero mismatches against the saved timeline [EXECUTED].

**18,216 / 31,201.** Unit is the replay request (6,258 base questions + 24,943 paraphrases; every request has 1–5 URLs, none has zero, none references a URL without a snapshot). The rule is **any**: a request is retained if at least one of its URLs is in the 4,382 set (`e1_c1_observable_refit.py:194-195`). Under an **all-URLs** rule the pool would be 12,813 [EXECUTED]. REAL_TIME is included (344 of 354 kept). Per class, any-rule retention is TIMELESS 56.89 %, SLOW 56.24 %, MEDIUM 56.64 %, FAST 61.74 %, REAL_TIME 97.18 % [EXECUTED]. Missing snapshots are handled by exclusion: a URL without a 24h snapshot is simply not observable. Note this pool is a trace-construction filter used by the v10 refit; it is not the drift denominator (those are 17,872 fixed-24h and 17,877 mixed-age; see `v18_validation_p01_p04_p05/p1_support_audit.py:10-28` [RECORD]). Appendix Table row "Body-observable pool (24h) 18,216, keep requests with a body-observable transition" is consistent with the any-rule.

**Class percentages and χ².** The unit is the **manifest row**: one row per (query, URL) retrieved for that query at rerun_24h, filtered by `snapshot_available` and the domain noise filter (`analyze.py:169-173`). The 17,351 rows cover only 7,723 distinct URLs; a URL retrieved by k queries contributes k rows, and 228 URLs appear under more than one class [EXECUTED]. Observability is a property of the URL, so the rows are repeated observations of the same URL, not independent trials. The study's own log says so ("a URL tracked under k classes contributes k observations", `analyze.py:229-231`) but the manuscript presents the percentages without stating the unit.

## 3. Per-class count tables and the partition question (item 3)

Full tables for all three units are in `class_count_tables.csv`. The study's unit (a) and the two deduplicated units (b, c):

| class | (a) rows obs / not | (a) % | (b) unique (URL,class) obs / not | (b) % | (c) single-class URLs obs / not | (c) % |
|---|---|---|---|---|---|---|
| TIMELESS | 1,787 / 1,284 | 58.19 | 867 / 697 | 55.43 | 857 / 695 | 55.22 |
| SLOW | 2,457 / 1,847 | 57.09 | 1,159 / 1,017 | 53.26 | 1,095 / 955 | 53.41 |
| MEDIUM | 2,406 / 1,550 | 60.82 | 1,034 / 843 | 55.09 | 953 / 762 | 55.57 |
| FAST | 2,746 / 1,946 | 58.53 | 1,150 / 1,006 | 53.34 | 1,058 / 943 | 52.87 |
| REAL_TIME | 973 / 355 | 73.27 | 124 / 61 | 67.03 | 116 / 61 | 65.54 |
| total | 10,369 / 6,982 (N 17,351) | 59.76 | 4,334 / 3,624 (N 7,958) | 54.46 | 4,079 / 3,416 (N 7,495) | 54.43 |
| χ², p, V | 121.53, 2.5e-25, 0.084 | | 15.03, 0.0046, 0.044 | | 12.89, 0.012, 0.041 | |

**Do the class groups partition the URL pool? No** [EXECUTED]. Of the 8,635 panel URLs, 912 have no rerun_24h manifest row after the noise filter and so are in no class group; 228 are in two or more classes. The class rows cover 4,204 of the 4,382 observable URLs. The five groups therefore neither cover nor disjointly divide the pool. The 251 multi-class figure in `observability_stats.json` counts multi-class URLs across all rounds before the noise filter; 228 is the count among the rerun_24h rows actually tabulated.

**Consequences for the reported test.** The χ² of 121.53 and p = 2.5e-25 are correct arithmetic on the 17,351-row table but overstate the evidence: each URL is counted once per retrieving query, inflating N by a factor of about 2.2 and treating dependent rows as independent. On deduplicated units the association is still present but much weaker (p ≈ 0.005 to 0.012) and the class percentages shift by 3 to 8 points (REAL_TIME 73.27 → 65.5 to 67.0 %). The qualitative claim "volatile classes are on average more observable" survives at every unit (REAL_TIME stays highest), but the specific figure "8.26 points more observable" is the row-level mean(FAST, REAL_TIME) − mean(TIMELESS, SLOW) = 65.90 − 57.64 and becomes about 5.9 points (unit b) or 5.1 points (unit c). Cramér's V = 0.084 → 0.04.

## 4. Reconciliation of the four populations (item 4)

| population | unit | N | rounds | split | relation to the others |
|---|---|---|---|---|---|
| URL pool | unique URL | 8,635 → 4,382 observable | run_00 vs rerun_24h | none | base set for everything below |
| Request pool | replay request | 31,201 → 18,216 | fixed 24h | none | request kept if any URL in the 4,382 |
| Class tables | (query, URL) manifest row | 17,351 rows over 7,723 URLs | rerun_24h rows | none | rows of URLs from the 8,635; multi-counting; not a partition |
| Paired missingness | held-out request | 21,848 → 16,813 paired → 9,184 (400 chars) / 13,396 (50 chars) jointly observable | mixed-age arrival round per request | held-out test clusters, seed 42 | different stream, different horizon; not comparable with 18,216 |

The 54.62 % and 79.68 % are shares of the 16,813 paired held-out requests, computed from the outcomes the published `mixed_engine.replay` emits for the FreshCache and AlwaysPassTemporal arms, and reproduce from the saved per-request file [EXECUTED]. The 50-char recovery is a threshold change on locally stored historical text with the block filter kept; 19,868 short texts were re-read and 21 rejected as block pages [RECORD, `recovery_results.json`]. Whether the drift estimates (+3.40 → +2.88) are correct was reproduced by the study's own gate against the published replay [RECORD] and is outside this audit's scope.

## 5. Manuscript claims that need correction

1. **"About 42 % of snapshots are title-only"** (§ Observability). Measured share of non-substantive snapshots is 47 %. The 42 % is the reviewer's phrase.
2. **Class observability percentages and the χ² test** (Limitations). State the unit: (query, URL) manifest rows at 24h, 17,351 rows over 7,723 distinct URLs, so URLs retrieved by several queries or filed under several classes are counted repeatedly, and the rows are not independent. Either report the deduplicated table (unit b or c above) or keep the row-level figures with that caveat and note that on unique URLs the association weakens to p ≈ 0.005–0.012, V ≈ 0.04. The "8.26 points" figure should be tied to the row-level unit or replaced.
3. **"Both snapshots carry HTTP 200"** is not a coded condition in the flag that produced 4,382; it holds trivially because every stored snapshot is status 200. Safer wording: "every stored snapshot has HTTP 200; a transition is body observable when both snapshots pass the block-page pattern and hold at least 400 characters".
4. **18,216** should be described as requests with at least one observable URL (the any-rule); the appendix row wording "a body-observable transition" is acceptable only with that reading. It should not be called a drift denominator.
5. **Class groups are not a partition** of the 8,635 URLs (912 uncovered, 228 in several classes); any sentence implying the five percentages divide the pool should be avoided.

## 6. What could not be verified

- The original run that first produced 4,382 / 8,635 and 18,216 (`v10_remaining_feedback/e1_c1_observable_refit.py`) was not re-executed; I re-derived both numbers from the raw snapshot files and the manifest with independent code, which is stronger than re-running it.
- The mixed-age replay behind the paired figures was not re-run (it needs the full engine); the shares were recomputed from the study's saved per-request outcomes, and the study's own reproduction gate against the published bound is a [RECORD].
- Whether the 400-character rule was chosen before or after seeing results is not recorded anywhere I could find [UNVERIFIED].

## Files in this folder

`Observability_verification.md`, `observability_recompute.py`, `observability_results.json`, `class_count_tables.csv`, `code_excerpts.md`, `run.log`.
