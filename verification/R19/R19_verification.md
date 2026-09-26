# R19 verification: external-benchmark experiment behind `tab:external`

Read-only on the repository; outputs in `audit/R19/` (`r19_recompute.py`, `R19_recompute.json`, `R19_results.json`, `comparison.csv`, `run.log`). Tags: **[executed]** today, **[historical record]** existing file or code, **[unverified]** claimed but not evidenced.

## 1. What produced the printed table

**[historical record]** Loader, evaluator and driver are one file: `validation_V3/remaining_critical_issues/09_external_generalization/run_external.py` (mtime 2026-09-21T12:00). Its outputs `external_results.json` (`utc` 2026-09-21T12:00:44Z), `external_results.csv` and `run.log` (`utc 2026-09-21T12:00:12Z`) carry the same minute, so the script on disk is the version that produced the stored result. The copy under `Reframe_Paper_Pro/artifacts/external/external_results.json` is byte-identical (sha256 `e91183304a2994fb…`). The printed table takes the `EvolvingQA` and `DailyQA` blocks of that file (common 24 h age); the file also holds an `EvolvingQA@natural` block at the dataset's own 30-day interval, which the table does not print.

The older experiment behind the earlier draft's numbers is `cise_eval.py` with `data/cise_summary.json` (2026-06-17): EvolvingQA only, 43,453 pairs, its own SemanticTTL TTL schedule (7 d / 3 d / 48 h / 6 h / 5 min) and strict `<` expiry (`cise_eval.py:73-79`, `:29`). It is a different program with different baselines, not an earlier run of the same one.

## 2. Exact runtime parameters, traced to the consumer [executed]

Printed by the script itself at `run_external.py:299-302` and re-read from the imported module today:

- `AGE = 24 * 3600` s (`:63`), applied to **both** datasets; `NATURAL_AGE` = 30 d (EvolvingQA), 1 d (DailyQA) (`:64`).
- **SemanticTTL** (`:186-195`): reuse iff `theta_ok and age <= FIXED_TTL[class]`. θ is never applied: `evaluate` calls `reuse_decision(pol, fc, AGE)` (`:229`) with the default `theta_ok=True`, and the script never computes a similarity. κ does not exist in this code; the TTL is the bare class schedule `FIXED_TTL` = TIMELESS 720 h, SLOW 168 h, MEDIUM 24 h, FAST 1 h, REAL_TIME 0 (`experiment.py`). Expiry uses `<=` (`:195`), so MEDIUM at exactly 24 h **is** reused. Effective TTLs are the schedule itself. Unused parameters: θ (not evaluated), κ (absent).
- **ExactTTL** is the identical branch with `theta_ok` defaulting to True; the "ExactTTL = SemanticTTL" statement is true **by construction of the code**, not an empirical finding.
- **FreshCache / FreshCache-L1Only** (`:196-197`): reuse iff `p_stale(class, 24 h, "answer") <= EPS_ANSWER`, with `HALF_LIFE` = {REAL_TIME 30 s, FAST 124.8 h, MEDIUM 127.2 h, SLOW 276 h, TIMELESS 720 h}, `TIER_MULT["answer"] = 1.5`, `EPS_ANSWER = 0.10`. Effective admission age T = −h·ln(1−ε)/(m·ln 2): FAST 12.6 h, MEDIUM 12.9 h, SLOW 28.0 h, TIMELESS 73.0 h; at 24 h it admits SLOW and TIMELESS only ({'REAL_TIME': False, 'FAST': False, 'MEDIUM': False, 'SLOW': True, 'TIMELESS': True}). No semantic condition is exercised: every pair re-asks the same question, and the L1 equivalence gate is never called. The two FreshCache rows are identical by construction.
- **ExactNoTTL** (`:189-190`): always reuse the cached answer. **NoCache** (`:187-188`): never.
- **Class assignment** (`:108-110`, `:210`): `_classify_freshness(question, temporal keywords, answer type)` on the question text only; it reads neither answer nor the change label. **[executed]** confirmed by code path; the class distribution reproduces exactly.

**Freezing.** The script prints "frozen published parameters" and reads them from `experiment.py` at run time; nothing in the folder records that these values were fixed *before* the external evaluation (no preregistration file, no hash). The values equal the deployed constants, which is consistency, not evidence of freezing. **[unverified]**

## 3. Inputs, pair construction, counts [executed]

**EvolvingQA.** Loaded from HuggingFace `kat-research/EvolvingQA` (`:129-143`), configs `edited` (question, answer1 → cached, answer2 → fresh; consecutive monthly snapshots) and `unchanged` (answer → both). The local cache (fingerprint `5eb679ac6cead846…`, cached 2026-06-09) was used offline today. Pool: 18,968 edited and 129,497 unchanged rows; the loader shuffles each with `random.Random(42)`, keeps **10,000 edited + 33,456 unchanged** (`:114`, `:144-146`), shuffles again. **43,456 is therefore a seeded capped sample, not the dataset size**; the manuscript's "43,456 consecutive-snapshot pairs" omits the cap. Recount: 43,456 pairs, 40,824 unique questions (some questions recur across splits), 9,963 changed (edited rows whose two answers still match under `answers_match` count as unchanged). The 24 h age is **imposed**: the real interval is one month, and at the dataset's own 30-day age (`EvolvingQA@natural` block) FreshCache reuses 0 and SemanticTTL 7,025; the table prints the imposed-age numbers.

**DailyQA.** Local files `external/DailyQA/data/qa/qa_250104.jsonl … qa_250201.jsonl`, 29 daily files (sha256 prefixes in `R19_results.json`), schema `{id, query, answer}`. Pairs are the same question on consecutive day files (`:173-181`); within a day a repeated question keeps the **last** answer (`:170-171`); a question missing on either day yields no pair (`:178-179`). Recount: **227,065 pairs over 8,346 unique questions**, matching the stored 227,065 and the printed 8,346. Here the imposed 24 h age equals the actual one-day interval.

## 4. All six rows recomputed [executed]

Recomputation reran the module's own `evaluate` on the loaded rows. Every integer matches the stored file (`comparison.csv`, `R19_recompute.json`).

| dataset | policy | n | reuse | errors | correct | Reuse % | Error/query % | Acc. % | printed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| EvolvingQA | FreshCache | 43,456 | 8,606 | 2,411 | 41,045 | 19.80 | 5.55 | 94.45 | 19.80 / 5.55 / 94.45 ✓ |
| EvolvingQA | SemanticTTL | 43,456 | 40,009 | 9,057 | 34,399 | 92.07 | 20.84 | 79.16 | 92.07 / 20.84 / 79.16 ✓ |
| EvolvingQA | ExactNoTTL | 43,456 | 43,456 | 9,963 | 33,493 | 100.00 | 22.93 | 77.07 | 100.00 / 22.93 / 77.07 ✓ |
| DailyQA | FreshCache | 227,065 | 43,341 | 1,848 | 225,217 | 19.09 | 0.81 | 99.19 | 19.09 / 0.81 / 99.19 ✓ |
| DailyQA | SemanticTTL | 227,065 | 177,621 | 10,778 | 216,287 | 78.22 | 4.75 | 95.25 | 78.22 / 4.75 / 95.25 ✓ |
| DailyQA | ExactNoTTL | 227,065 | 227,065 | 13,336 | 213,729 | 100.00 | 5.87 | 94.13 | 100.00 / 5.87 / 94.13 ✓ |

**Scoring rule, traced.** Class from the question (`:210`); `changed = not answers_match(cached, fresh)` (`:211`); served answer = cached if reused else the fresh (next-snapshot) answer (`:230`); correct = `answers_match(served, fresh)` (`:231`). On a miss the served answer *is* the fresh answer, so it is correct by construction; there is no generation, no LLM judge, and no gold substitution beyond the dataset's own next-snapshot answer. Hence `accuracy = corr/n` (`:242`) equals **exactly** 1 − Error/query, and NoCache scores 100% by definition. `answers_match` (`:91-105`) is lowercase, punctuation-stripped equality, or containment when the shorter string is at most 40 characters; it is lenient and asymmetric to length, and 37 of the 10,000 edited EvolvingQA rows count as unchanged under it. "Accuracy" in the table is therefore a string-match agreement rate with the next snapshot, not answer correctness.

## 5. Change record versus the artifacts

**[historical record]** The earlier draft's EvolvingQA numbers (43,453 pairs; 19.77% / 5.55% / 94.45%; SemanticTTL 92.03% / 20.84%) are `data/cise_summary.json` from `cise_eval.py`. The three-pair difference (43,453 vs 43,456) is a different shuffle/cap realisation of the same 10,000 + 33,456 design (`cise_eval.py` uses `random.seed(42)` with global shuffles; `run_external.py` uses `random.Random(42)` instance shuffles), not a filtering change. The CISE SemanticTTL used its own TTL schedule and strict `<` expiry, so its 92.03% and the symmetric run's 92.07% are not the same policy. **No CISE artifact for DailyQA with 43,456 pairs / 25.05% / 4.08% exists in the repository**; the source of those earlier DailyQA numbers was not located and the change record's attribution of them to "separate single-dataset evaluations" is unverified for DailyQA. **[unverified]** The revised numbers reproduce exactly from the stored file and from a fresh offline run.

## 6. Blockers and residual risks

1. No preregistration or freeze record for the external run; parameter identity with `experiment.py` is all that exists.
2. The earlier DailyQA figures (43,456 / 25.05 / 4.08) have no locatable artifact.
3. The manuscript describes 43,456 as the pair count without stating the seeded caps (10,000 edited + 33,456 unchanged of 148,465 available) or that the 24 h age is imposed on a 30-day dataset; and "Acc." should be described as next-snapshot string agreement with misses correct by construction. Not edited here.
4. EvolvingQA reproduction depended on the local HuggingFace cache; a machine without it would need network access.
