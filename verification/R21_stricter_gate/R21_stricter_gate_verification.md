# R21 audit: stricter-gate study (Appendix "Stricter L1 admission")

Audit date: 2026-09-23. Scope: the stricter-gate claims only. Nothing in the manuscript, the study directory, or any original result was modified. No live-Web or paid calls were made. All checks run offline against local files with `r21_recompute.py`; the console output is in `run.log` and the structured output in `R21_results.json`.

Legend used throughout: **[EXECUTED]** = a check I ran now; **[RECORD]** = an existing file written by the study on 2026-09-21, not re-executed; **[UNVERIFIED]** = stated somewhere but not checkable from local files.

## 1. What the manuscript claims

`Reframe_Paper_Pro/appendix_pro.tex:81-91` (same text as `qa/sources_merged_from/supplementary_35.tex:939-964`):

> On the held-out replay the published gate produces 806 L1 hits. Raising the similarity floor to 0.90 leaves 450, and a stricter precision gate (similarity ≥0.90, Jaccard ≥0.30, type agreement, the deterministic guards) leaves 314. In the gate-specific precision audit, mismatch falls from 43.30% to 18.22% and 14.97%. Search savings move only from 60.58% to 60.56%, because L2 supplies most of the avoidance, and WAI stays at 7/400. These gate-specific values use a different audit protocol from the four-juror estimate in Table 3 and serve only within-ablation comparison.

## 2. Producing study

`validation_V3/l1_precision_gate/` (all files dated 2026-09-21 17:38 to 18:22 UTC+local). The earlier two-arm experiment `validation_V3/final_three_issues/02_l1_strict_gate/` (2026-09-21 14:46 to 14:53) reported 43.55% for the published arm and 450 hits for the 0.90 arm. Both directories are revision-stage work, not the original submission's evidence. No older record of a stricter-gate study exists in the repository as far as the local files show.

## 3. Exact gate definitions (item 1)

Side-by-side table in `gate_definitions.csv`; verbatim code in `code_excerpts.md`.

| Rule | published | strict090 | precision (frozen cell `s0.9_j0.3_as_implemented_G`) |
|---|---|---|---|
| Similarity floor | 0.80, reject if `sim < floor` (`experiment.py:298`, `l1_gate.py:33`) | 0.90 | 0.90 |
| Content-token Jaccard floor | 0.30, reject if `< floor` (`experiment.py:299`, `l1_gate.py:40`) | 0.30 | 0.30 |
| Answer-type rule | as_implemented: reject only when both types are detectable and differ (`l1_gate.py:46-47`) | same | same (the stricter "required" mode was in the grid but was **not** selected) |
| Strict-subset rule | on (`experiment.py:384`, `l1_gate.py:48`) | on | on |
| Structural guards | off | off | on: numeric/date ∧ negation ∧ comparative (`l1_guards.py:117,152,203,211`; `l1_gate.py:49-50`) |
| Entity match | applied by the engine outside the gate (`experiment._entity_match`, `experiment.py:231`); unchanged in every arm; all 806/450/314 judged hits have `entity_match=True` [EXECUTED] | same | same |
| Where defined | `run_heldout.py:114` reads `exp._EQ_SIM_FLOOR`, `exp._EQ_JACCARD_MIN` | `run_heldout.py:115`, literals | `run_heldout.py:116-117`, read from `frozen_l1_precision_gate.json` |

Boundary operators: both floors are inclusive (`<` rejects). Realised minima confirm it: minimum similarity 0.8003 (published) and 0.9003 (both strict arms); minimum Jaccard exactly 0.30 in every arm [EXECUTED].

Everything else is identical across arms [EXECUTED from `run_heldout.py:97-126`]: same held-out stream (schedule `zipf_uniform`, seed 42, 21,848 requests in 4,381 test clusters, `split.json` of `validation/heldout_baseline_tuning`, zero validation leakage asserted), same engine `prep2.replay(stream, {}, "FreshCache")` with an empty initial cache, same temporal parameters (the engine's own half-lives and budgets; the study never touches them), same write rules. The only thing swapped is `exp.semantic_equivalent`, restored after each arm. Each arm is a true sequential replay, so later cache state differs between arms; that is why 2 of the 379 published hits lost under strict090 have similarity ≥0.90 (the entry they had matched no longer exists in that arm) [EXECUTED].

Wording problems in the manuscript sentence:
- "type agreement" suggests a stricter answer-type rule. The frozen cell keeps the published `as_implemented` rule. The only added condition is the three deterministic guards.
- "Raising the similarity floor to 0.90" is exact for strict090 (Jaccard stays 0.30).

Concrete differing candidates [EXECUTED, from `heldout_l1_hits_judged.csv`]:
- Rejected by strict090 (sim 0.880 < 0.90), jury DIFFERENT: request `q_631acf06`, "Who holds the position of Second Counselor in the general presidency of the Young Women organization?" vs cached "Who is the First Counselor in the general presidency of the Young Women organization?" (Jaccard 0.60, passes the published gate).
- Rejected by precision via the comparative guard: `p_q_3f0409f7_3`, "What was the name of the team that received his recent WNBA Coach of Year Award?" vs "Which team was led by the most recent WNBA Coach of the Year Award winner?" (sim 0.941; relation signatures `age-` vs `age-|quality+|quantity+`; jury said SAME, so this rejection is a false rejection by the guard).
- Rejected by precision via the negation guard: `p_q_97c477a6_0`, "How many inches is recommended as the standard length of a cricket bat…" vs "…must be no more than how many inches?" (sim 0.931; jury SAME).

Of the 139 strict090 hits that are not precision hits, 136 fail only the comparative guard, 1 only the numeric guard, 1 only the negation guard, 1 all three [EXECUTED]. So the 450 → 314 step is almost entirely the comparative/superlative guard.

## 4. Reconstructed results (item 2)

Full table in `results_comparison.csv`. Everything below was recomputed from row-level files [EXECUTED] and agrees with the study's own records [RECORD] and the manuscript.

| | published | strict090 | precision |
|---|---|---|---|
| L1 hits (per-request tiers and judged file agree) | 806 | 450 | 314 |
| Non-hit requests (miss + REAL_TIME) | 8,363 + 250 | 8,363 + 250 | 8,367 + 250 |
| Search savings, full precision | 60.5776 % | 60.5776 % | 60.5593 % |
| Savings delta vs published | 0 | 0 | −0.0183 pp |
| Published L1 hits displaced | | 357 → L2, 22 → fresh search, 427 stay L1; 23 newly L1 | 480 → L2, 31 → fresh search, 295 stay L1; 19 newly L1 |

The manuscript's 60.58 → 60.56 is the 2-dp rounding of 60.5776 → 60.5593. The reference is the primary held-out FreshCache replay: `run_heldout.py:46` hard-codes the anchor (60.5776 %, L1 806, L2 12,429) and `run_heldout.py:126-131` aborts if the published arm does not reproduce it; `logs/06_heldout.log` records "ANCHOR GATE PASSED" [RECORD]. Generation counts (963.11 / 979.40 / 985.63 per 1k) are [RECORD] only; the per-request file carries tiers, not the search-call and L2 counts needed to recompute them.

## 5. Gate-specific mismatch audit (item 3)

Protocol, from `l1_precision_prereg.json`, `run_heldout.py`, `gen_judge_l1p.py`, `prompts_verbatim.json`, `analyze_heldout.py` [RECORD, code read now]:
- **Unit**: every realised L1 hit (incoming query, cached query) of each arm. It is a census, not a sample: 806 + 450 + 314 = 1,570 hit rows, 844 distinct query pairs judged once each and joined back to rows. No strata, no weights.
- **Populations and IDs**: each arm's own realised hit set; request IDs are unique within each arm; overlaps 427 (strict∩published), 295 (precision∩published), 311 (precision∩strict) [EXECUTED].
- **Judges**: Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3; greedy, `do_sample=False`, `max_new_tokens=6`, chat template (`gen_judge_l1p.py:76,166`). Prompt is the paper's equivalence prompt verbatim ("Do these two queries ask for the SAME fact… Reply with exactly one word: SAME or DIFFERENT"). Judges see only the two strings, never the arm.
- **Aggregation**: majority of four; 2–2 = JURY_TIE (`analyze_heldout.py:101-107`). Primary metric counts ties as DIFFERENT; ties-excluded and ties-as-SAME are also reported. CI = Wilson on the raw count (`analyze_heldout.py:38-46`).
- **Missing labels**: none. All 844 pairs carry four labels; verdicts SAME 484, DIFFERENT 248, TIE 112 [EXECUTED].

Recomputed from the `equivalence` column [EXECUTED]:

| arm | SAME | DIFFERENT | TIE | ties→DIFF | Wilson 95 % | ties excluded | ties→SAME |
|---|---|---|---|---|---|---|---|
| published | 457 | 244 | 105 | 349/806 = **43.30 %** | [39.92, 46.74] | 34.81 % | 30.27 % |
| strict090 | 368 | 45 | 37 | 82/450 = **18.22 %** | [14.93, 22.05] | 10.90 % | 10.00 % |
| precision | 267 | 23 | 24 | 47/314 = **14.97 %** | [11.45, 19.34] | 7.93 % | 7.32 % |

Fleiss κ 0.2316 [RECORD]; the 3B juror votes DIFFERENT on 89 % of pairs while the other three vote 24–40 %, so the ties→DIFFERENT number is driven by disagreement, not by consensus.

**Why 43.30 % here versus 19.1 % in Table 3.** These are different measurements of different populations:
1. Table 3's L1 row is the full-stream fixed-24h replay (1,164 hits) audited on a stratified sample of 140 judgments (two similarity bands × 70) reweighted to the population, with FPC Wald intervals (see `audit/R17`). The stricter-gate row is a census of all 806 hits of the held-out test replay (21,848 requests), unweighted Wilson.
2. Same four judges and same prompt, but a different judging run, so per-pair labels are not shared.
3. Both use ties→DIFFERENT.
The manuscript's phrase "a different audit protocol from the four-juror estimate" is therefore right about the protocol (census vs stratified sample; different replay) but misleading about the jurors, which are the same four models. The two numbers should not be compared, as the manuscript already says.

**43.55 % in the earlier experiment.** `02_l1_strict_gate/summary.json` records 351/806 with 108 ties over 839 pairs [RECORD]. The current run judged 844 pairs (the precision arm adds 5) and, on the 839 shared pairs, 21 of 3,356 juror labels differ from the earlier run despite greedy decoding [EXECUTED]. Batch-composition nondeterminism is the plausible cause; the records do not state one. The 0.25 pp gap is within this noise.

## 6. Selection status (item 4)

All records are revision-stage (2026-09-21). They form an internally consistent pre-registration, but nothing predates the revision.

- `l1_precision_prereg.json` (17:39): grid of 96 cells (sim ∈ {0.90,…,0.97} × Jaccard ∈ {0.3,0.4,0.5,0.6} × answer_type {as_implemented, required} × guards {off,on}); two-stage selection on validation clusters only; C1 savings ≥ published − 0.10 pp, C3 ≥ 100 validation L1 hits, objective minimise ties→DIFFERENT mismatch; C2 validation WAI ≤ published; guards frozen by hash; success criteria fixed before running, including "no threshold manipulation". [RECORD]
- `stage1_selection.json` (18:05): 96 pass C1, **4** pass C1 ∧ C3; top-K carried: `s0.9_j0.3_as_implemented_G` (115 hits, 10.43 %), `s0.92_j0.3_g` (109, 11.01 %), `s0.9_j0.4_g` (111, 15.32 %), `s0.9_j0.3_g` (155, 15.48 %). [RECORD]
- `frozen_l1_precision_gate.json` (18:09:53): validation WAI 1/400 for every carried cell and for the published gate (C2 non-discriminating), so the cell with the lowest validation mismatch was frozen. Held-out run starts after the freeze (`06_heldout.log`; `sanity_checks.json` "freeze → test ordering verified by mtime"). [RECORD]
- The "strict090" arm is not a selected cell. It is a fixed comparison arm carried from the earlier experiment (`run_heldout.py:115`).

Note that the frozen cell's validation mismatch (10.43 %) sat just above the prereg's "≤10 % on held-out" bar for FULLY_RESOLVED, and the held-out result (14.97 %) also misses it. By the study's own success criteria the outcome is STRONGLY_ADDRESSED, not FULLY_RESOLVED. The manuscript does not claim otherwise.

## 7. WAI evidence (item 5)

- Same 400 IDs: `heldout_audit_sample.json` lists 400 requests whose IDs equal, in order, `validation/heldout_baseline_tuning/answer_audit_k1_16/sample.json` [EXECUTED]; `06_heldout.log` recorded "SAMPLE GATE PASSED" [RECORD].
- Variants were scored, not assumed: per arm the audit records the tier and answer key each request actually received (`published_key`, `strict090_key`, `precision_key`). 388 requests have identical keys in all arms, 12 differ; 1,979 generations were reused byte-for-byte and 8 new ones generated (`06_heldout.log`) [RECORD]; labels in `answer_audit.csv` differ across arms for exactly one request [EXECUTED].
- Labelling rule: ABSTAIN pattern, else CORRECT only if both Llama-3.1-8B and Qwen2.5-7B say CORRECT, else WRONG (`analyze_heldout.py:120-135`); WAI = fresh CORRECT ∧ policy WRONG (`:158`).
- Recomputed [EXECUTED]: WAI = 7/400 in all three arms, identical IDs (`q_96543752`, `p_q_e5616ef7_3`, `p_q_3055ab76_1`, `p_q_5abb871e_2`, `q_a8f577f5`, `q_f93213af`, `q_e2f9f525`). Fresh CORRECT 81/400.
- **All seven WAI events are L2 hits in every arm.** None is an L1 hit, so no L1 gate could have changed them. Only 15 / 5 / 4 of the 400 audited requests are L1 hits under published / strict090 / precision. The single discordance is `p_q_3ec48396_1`: L1 CORRECT under published, L2 WRONG under both stricter arms, fresh WRONG (this is the published arm's one "reverse" event, which disappears). McNemar published vs either stricter arm: 1 vs 0, p = 1 [RECORD].

So "WAI stays at 7/400" is exactly true, but it is weak evidence about the gate: the audit sample contains almost no L1 traffic, and the stricter gates could not have moved the seven events. The earlier experiment's 6/400 used the predicted-class arm; the current study uses the collection-class arm, which is the manuscript headline arm (prereg `answer_audit.freshness_class_source`) [RECORD].

## 8. Integrity records [RECORD]

`sanity_checks.json` (13 checks, 0 failures): no validation/test overlap (1,877 / 4,381 clusters), 34 original FreshCache files unchanged by hash, guards hash matches prereg, single frozen cell, anchor reproduces, WAI 7/400 reproduces, all aggregates regenerate from rows, savings within budget, no output outside the directory. I re-derived the "aggregates regenerate" check independently (Section 4, 5, 7) and it holds.

## 9. Findings for the manuscript (not applied)

1. Replace "type agreement" in the precision-gate description: the answer-type rule is unchanged; the added condition is the three deterministic guards (numeric/date, negation, comparative).
2. "Different audit protocol from the four-juror estimate" should say the jurors and prompt are the same and the difference is census-of-held-out-hits versus stratified sample of the fixed-24h replay.
3. Consider stating that the seven WAI events are all L2 reuses, so WAI invariance is expected and is not evidence that the stricter gate is harmless at L1 (only 15 L1 requests are in the audit).
4. Optionally state that the frozen cell was chosen on validation by a revision-stage pre-registration, and that the held-out 14.97 % misses the pre-registered 10 % target.

## 10. Blockers

- No pre-revision record of a stricter-gate study exists; the selection history is genuine but dated 2026-09-21. This cannot be changed after the fact and should not be presented as older.
- Generation counts per arm can only be taken from `summary.json`; the per-request file does not carry the fields needed to recompute them.
- The 43.55 % vs 43.30 % difference is explained by jury-label instability across runs, but the cause of the instability is not recorded.

## Deliverables in this folder

`R21_stricter_gate_verification.md` (this file), `gate_definitions.csv`, `results_comparison.csv`, `code_excerpts.md`, `r21_recompute.py`, `R21_results.json`, `run.log`.
