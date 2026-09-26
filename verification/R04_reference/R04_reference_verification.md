# R04 / V35-N03: reference validity, the 57/107/41 crosswalk, and Table 15

Audit date: 2026-09-24. Read-only. No manuscript, data, reference answer, label, page or judgment was changed or regenerated. Deliverables in this folder: this report, `r04_recompute.py`, `R04_results.json`, `heldout_400_reference_records.csv` (one row per held-out request with its unit, both screens' raw judge decisions, evidence sizes and evidence fetch times), `code_excerpts.md`, `run.log`.

Tags: **[EXECUTED]** recomputed now from row-level files; **[RECORD]** existing study output; **[UNVERIFIED]** asserted but not checkable locally.

## 0. Records located

| record | path | date |
|---|---|---|
| held-out audit (400 IDs, keys, fresh evidence versions) | `validation/heldout_baseline_tuning/answer_audit_k1_16/sample.json`, `answers.jsonl`, `judge_{llama8b,qwen7b}.jsonl`, `gen_tasks.jsonl` | 2026-09-1x [RECORD] |
| quality-selected SemanticTTL (0.60, ½) held-out arm | `validation/quality_aware_baseline/heldout_sensitivity/h_sample.jsonl`, `h_answers.jsonl`, `h_judge_*.jsonl`, `h4_analyze.py` | [RECORD] |
| gold verification, original stage 3b (two judges) | `validation/gold_verification_400/`: `verification_units.csv` (715 units), `validity_tasks.jsonl`, `validity_labels_{llama8b,qwen7b}.jsonl`, `gold_validity_at_t.csv`, `paraphrase_fidelity.csv`, `combined_request_status.csv` (786 requests) | 2026-09-20 16:24–17:04 |
| gold verification, revised 3b (three judges, expanded evidence) | `validation/gold_verification_400/revised_3b/`: `r_validity_tasks.jsonl`, `r_validity_{llama8b,qwen7b,mistral7b}.jsonl`, `r_gold_validity.csv`, `r4_summary.json` | 2026-09-20 17:17–17:27 |
| the four-way classification of the 400 | `validation_V3/strong_reference_evaluation/02_reference_verification/verify_and_diagnose.py`, `reference_and_diagnosis.csv/.json`, `human_reference_review.csv` | 2026-09-22 02:44 |
| crosswalk and Table 15 | `Reframe_Paper_Pro/scripts/crosswalk_57_107.py` → `Reframe_Paper_Pro/artifacts/crosswalk_57_107.json` | 2026-09-23 10:54 |

The manuscript says these ship as `crosswalk_57_107.json` and `gold_verification_400/`. The crosswalk file is in `Reframe_Paper_Pro/artifacts/`; `gold_verification_400/` is not under `Reframe_Paper_Pro/artifacts/` or the release bundle [EXECUTED listing], so the "screening pipeline ships" statement is not met as staged.

## 1. The 400 IDs and the four-way classification [EXECUTED]

- 400 IDs, 400 unique, no duplicates; all 400 have a gold-verification record and map to exactly one verification unit (base query at the request's snapshot round).
- Recomputed from `combined_request_status.csv` plus the rule in `verify_and_diagnose.py:39-43,125-129,153-156`: **SUPPORTED 57, CONTRADICTED 101, TEMPORALLY_AMBIGUOUS 178, UNVERIFIABLE 64**; identical to `reference_and_diagnosis.csv` (0 mismatches).
- What the labels actually are. The underlying two-judge status on the 400 is VALID_AT_T 57, SUPERSEDED_AT_T 101, INDETERMINATE_AT_T 74, JUDGE_DISAGREEMENT 168. SUPPORTED = VALID_AT_T; CONTRADICTED = SUPERSEDED_AT_T. **TEMPORALLY_AMBIGUOUS is not a judge outcome**: it is the 178 requests that were INDETERMINATE (60) or JUDGE_DISAGREEMENT (118) *and* whose dataset volatility tag is VOLATILE *and* whose question has no four-digit year (regex). UNVERIFIABLE (64) is the remainder of the indeterminate/disagreement pool. The manuscript presents the four counts as if they were judged categories; the 178/64 split is a deterministic relabelling.

## 2. The 57, the 107 and the 41 [EXECUTED]

Rebuilt from the row files with the original screening rules (`crosswalk_57_107.py:19-39`):

| set | rule | n |
|---|---|---|
| supported 57 | original stage 3b: Llama-3.1-8B and Qwen2.5-7B both VALID_AT_T on the 2,000-char-per-page / 2,800-char evidence, disagreement excluded (`stage3b_validity.py:26-31, 366-379`) | 57 |
| screened 107 | paraphrase fidelity SAME (two judges agree) or base question, AND revised-3b status VALID_AT_T by 2-of-3 majority (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B tiebreak) on expanded evidence (5,000 chars per page, 12,000 total) with question-type-aware prompts (`r1_build.py:33-36,60-61,126-128`; `r4_analyze.py:104-123,172-178`) | 107 |
| intersection | | **41** (57-only 16, 107-only 66) |

All three ID lists equal `crosswalk_57_107.json` exactly.

**The two screens are not the same test on different thresholds.** They use different evidence texts, different prompts and different aggregation. Cross-tabulated on the 400 [EXECUTED, `heldout_400_reference_records.csv`]:
- Of the 57 supported, 56 stay VALID under the revised majority and 1 becomes SUPERSEDED. 16 are not in the 107: 14 because the paraphrase-fidelity judges disagreed, 1 fidelity DIFFERENT, 1 revised SUPERSEDED.
- Of the 101 contradicted, 86 stay SUPERSEDED, **8 become VALID** under the revised majority, 6 have no majority, 1 becomes INDETERMINATE.
- Of the 107 screened, only 41 were VALID under the original pair; 58 were original disagreements, **6 were original SUPERSEDED** and 2 INDETERMINATE.
- Under the revised pair rule (same two judges, expanded evidence) only 54 of the 400 are VALID; under the revised majority 159.
So "supported" and "screened" disagree outright on 15 requests (1 + 8 + 6). The manuscript states that the criteria differ but not that they contradict each other, nor that the 57 rest on the earlier, smaller evidence build that the revised study was created to replace (STAGE4_DIAGNOSTIC.md documents five confirmed false SUPERSEDED labels in the original run).

## 3. Reference, evidence, time, prompt, decisions, aggregation [EXECUTED unless noted]

- **Reference answer** per request: `sample.json["gold"]` = dataset gold (FreshQA / TemporalAlignQA / TriviaQA), with aliases and annotation year carried in the unit (`validity_tasks.jsonl`). No gold was rewritten.
- **Evidence presented to the judges**: the request's own retrieved pages at the snapshot round `version_at(t)` of the request's replay time, assembled with the audits' loader (2,000-char page cap, 200-char minimum, 2,800 budget) in the original run; 5,000 / 12,000 in the revised run. Every one of the 400 units had evidence (no NO_EVIDENCE units; `stage3b_summary.json` no_evidence 0). Eight of the 400 requests' page slots have no snapshot at their round, so those pages were absent from the evidence [EXECUTED, `fresh_versions_without_snapshot_entry`]. The 57 supported units' original evidence ranges from 224 chars upward (five smallest: 224, 249, 271, 438, 609), i.e. some supports rest on a single short page.
- **Evidence observation time**: the unit round equals `version_at(request t)` for all 400 (rounds: rerun_24h 365, rerun_12h 18, rerun_1h 16, run_00 1). The actual `fetched_at` of the evidence pages, taken from `corrected_round_table.jsonl`, spans 2026-05-17 to 2026-06-13 for rerun_24h pages, 05-23 to 06-13 for rerun_12h, 06-01 to 06-12 for rerun_1h, and 06-12 for run_00; per-request values are in the records CSV. These are snapshot capture times, not file modification times. "Valid at request time" in the manuscript therefore means "valid according to pages captured at the snapshot round the replay maps the request to", which can be up to a day before the replay's t; the request timestamps themselves are simulated stream times.
- **Prompt**: original system prompt verbatim in `stage3b_validity.py:72-92` (VALID / SUPERSEDED / INDETERMINATE; absence of the gold is never SUPERSEDED; judge only from evidence at capture time); user template adds question, gold with aliases, annotation year and the round label. Revised run: two system prompts, CURRENT ("still current?") and AS_STATED ("correct as stated for the period?"), chosen by a regex question type (`r2_judge.py:34-96`). Decoding: chat template, greedy, `max_new_tokens=6`, bfloat16 [RECORD].
- **Raw decisions**: present for all 400 units from both original judges (raw strings recorded, e.g. "VALID"); for the 57 supported, both raw decisions are VALID_AT_T in every case. Revised decisions from three judges are in `r_validity_*.jsonl` (Mistral only for the 331 tie-break units).
- **Aggregation**: original, both judges must agree else JUDGE_DISAGREEMENT; revised, pair agreement else Mistral majority else NO_MAJORITY. Fidelity: two judges must agree SAME.

## 4. Table 15 recomputed from the row files [EXECUTED]

Scoring rule reproduced from `crosswalk_57_107.py:24-30,49-52` (the published h4 path): abstention pattern → ABSTAIN (never counted correct, never a WAI event); CORRECT only when both answer judges say CORRECT; WAI = fresh CORRECT and policy WRONG. All 400 requests have all three labels (0 dropped).

| subset | N | fresh corr. | FreshCache acc. | FC WAI | SemanticTTL (.60, ½) acc. | STTL WAI | manuscript |
|---|---|---|---|---|---|---|---|
| all held-out | 400 | 81 | 75 = 18.75% | 7 | 64 = 16.00% | 25 | same |
| screened | 107 | 49 | 45 = 42.06% | 4 | 38 = 35.51% | 15 | same |
| supported | 57 | 40 | 37 = 64.91% | 3 | 28 = 49.12% | 13 | same |
| screened ∩ supported | 41 | 30 | 27 = 65.85% | 3 | 22 = 53.66% | 9 | same |

Every Table 15 cell reproduces from the same row-specific IDs. Abstentions: 2 (FreshCache) and 3 (SemanticTTL) on the 400, none inside the three subsets.

## 5. Missing records, duplicates, timestamps, accessibility

- No duplicate IDs; no request maps to more than one unit; no missing answer or judge label on the 400; both original judge labels present for all 400 units [EXECUTED].
- Timestamps: unit round and `version_at(t)` agree for all 400; the evidence fetch times are recorded per page and are consistent with the round labels [EXECUTED]. The 400 replay timestamps are simulated stream times, not wall-clock request times.
- Accessibility: the crosswalk JSON is staged in `Reframe_Paper_Pro/artifacts/`; the gold-verification pipeline, its judge outputs, the revised-3b outputs and the human-review package are **not** in `Reframe_Paper_Pro/artifacts/` nor in the release bundle [EXECUTED]. The human-review file (242 uncertain cases) has all labels empty; no human validation exists [RECORD].
- The reference statuses are LLM-assisted; the strong-reference study itself labels them "NOT independent human validation" [RECORD]. No statement in the manuscript claims human validation.

## 6. Discrepancies to fix (not applied)

1. State that TEMPORALLY_AMBIGUOUS / UNVERIFIABLE are a rule-based split of the judges' indeterminate-or-disagreement pool, not judged categories.
2. State that the 57 use the original two-judge run on 2,000-char evidence while the 107 use the revised three-judge run on 5,000-char evidence with different prompts, and that the two disagree on 15 requests (8 "contradicted" references are VALID under the revised majority; 6 screened requests were SUPERSEDED under the original pair).
3. Replace "valid at request time" with "valid according to pages captured at the request's snapshot round" (capture dates 2026-05-17 to 06-13).
4. Ship `gold_verification_400/` (including `revised_3b/`) or drop the sentence that it ships.
