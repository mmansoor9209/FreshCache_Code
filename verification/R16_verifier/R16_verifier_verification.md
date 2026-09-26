# R16 / V-02–V-10: optional L2 evidence verifier (FreshCache-L2Verify)

Audit date: 2026-09-24. Read-only: no code, data, manuscript or existing output was modified. New files, all in this folder: `R16_verifier_verification.md`, `request_accounting.csv`, `paired_400_transitions.csv`, `gamma_recompute.csv`, `r16_recompute.py`, `R16_results.json`, `code_excerpts.md`, `run.log`.

Tags: **[EXECUTED]** recomputed now from saved row-level files; **[RECORD]** existing study output of 2026-09-21; **[UNVERIFIED]** not checkable locally.

## 0. Where the study lives and the bundle-path inconsistency

- Producing study: `validation_V2/l2_evidence_verification/` (scripts `s1_embed.py` … `s6_answer_audit.py`, outputs in `out/`, all 2026-09-21 10:21–11:06 UTC) [RECORD].
- Manuscript (Appendix E, `Reframe_Paper_Pro/appendix_pro.tex:395-396`; WWW V3 PDF p. 15) says the records "ship under `artifacts/verifier/`".
- Actual bundles [EXECUTED, directory listings]:
  - `Reframe_Paper_Pro/artifacts/verifier/`: `prereg_gamma.json`, `gamma_selection.json`, `s4_select_gamma.py`, `test_operational.json`, `val_labels.jsonl` (5 files).
  - `Reframe_Paper_Pro/release/bundle/artifacts/l2_verify/`: `prereg_gamma.json`, `gamma_selection.json` only; `release/MANIFEST.json:739-740` lists them under `artifacts/l2_verify/`.
- **Inconsistency.** The release archive uses the name `l2_verify` while the manuscript cites `verifier`. Neither bundle contains the "held-out per-request outcomes" the manuscript promises (`test_rows.pkl`, `test_scores_L2Verify_g0.55.jsonl`, `audit_sample.jsonl`, `audit_results.json`, the sufficiency and judge files); those exist only in the study directory. The "excluded identifiers" are also not shipped as a list; they are derivable from `validation/l2_independent_audit_300/selected_cases.csv`, which the bundle does not include under either name. The statement about what ships is therefore not met by either bundle as it stands.

## 1. Threshold selection (V-02–V-04)

**Pre-registration.** `out/prereg_gamma.json`, mtime 2026-09-21 10:34:11 UTC [EXECUTED]. Its embedded `prereg_sha256` = `18c7e0a965846311…` is the SHA-256 of `json.dumps(P, sort_keys=True)` with the hash field removed (`s3a_prereg.py:30`); recomputed now and identical [EXECUTED]. The plain file hash is `29c01922b9134cac…` [EXECUTED], as the manuscript states. The copy under `artifacts/verifier/` is byte-identical to the study file [EXECUTED].

**Order of operations** [EXECUTED, mtimes]: prereg 10:34:11 → label tasks built 10:35:52 → judge outputs 10:39:10 / 10:41:29 → `val_labels.jsonl` 10:42:23 → `gamma_selection.json` 10:42:43 → held-out score files 10:53:34–10:54:36 → `test_operational.json` 10:54:36 → audit 10:56–11:04. `s5_test_replay.py:11` reads γ from `gamma_selection.json`, and `gamma_selection.json` records `selection_used_test_data: false` [RECORD]. Selection preceded held-out evaluation within this workstream. All of it is revision-stage (2026-09-21); no earlier record exists.

**Validation population** [EXECUTED]: the 600 labelled requests are all in validation clusters (600/600; 0 in test clusters), spanning 534 distinct validation clusters. `s3b_build.json` [RECORD]: 5,034 validation L2 hits, 75 excluded because they appear in the 300-case audit, 4,959 eligible, 600 sampled over 20 strata (class × similarity band, seed 42). Exclusion check [EXECUTED]: 0 of the 600 labelled IDs occur among the 300 `incoming_query_id` values of `selected_cases.csv` (`s3b_label.py:77-80`).

**Labels** [EXECUTED from `val_labels.jsonl`]: 600 rows = INSUFFICIENT 298, SUFFICIENT 171, JUDGE_DISAGREEMENT 131. Six agreed rows (all INSUFFICIENT) have no support score (no page embedding) and are dropped by `s4_select_gamma.py:9-10`, giving **463 usable = 292 INSUFFICIENT + 171 SUFFICIENT**. Confirmed.

**Gamma sweep** [EXECUTED, `gamma_recompute.csv`]: 91 values 0.00–0.90 step 0.01; TPR = share of INSUFFICIENT with V < γ, FPR = share of SUFFICIENT with V < γ, J = TPR − FPR; tie-break smaller γ then lower FPR (`s4_select_gamma.py:27-28`). Result: **γ = 0.55, TPR 0.6952, FPR 0.2398, J 0.4554**, reject rate 0.527; J is maximised at 0.55 alone (no tie). The whole sweep equals the recorded one. **AUC = 0.7787** with SUFFICIENT scoring higher, and identically 0.7787 for the rejection score −V with INSUFFICIENT positive (the two orientations are the same number) [EXECUTED]. All manuscript figures (0.695, 0.240, 0.455, 0.779) confirmed.

## 2. Held-out replay denominator (V-05)

**Configuration** [RECORD, `s5_test_replay.py:30-36, 63-67`]: zipf_uniform stream, seed 42, filtered to the 4,381 test clusters, asserted free of validation clusters; engine `s2_engine.replay` with γ = 0.55 and precomputed BGE-M3 embeddings; FreshCache arm = same engine with γ = None, which `s2_verify_fidelity.py` asserts reproduces `prep2.replay` exactly [RECORD]. N = 21,848 in both `test_operational.json` and `test_rows.pkl` [EXECUTED].

**Request accounting from `test_rows.pkl`** [EXECUTED; full table in `request_accounting.csv`]:

| quantity | FreshCache | L2Verify γ=0.55 |
|---|---|---|
| L1 hits | 806 | 806 |
| L2 hits kept | 12,429 | 7,110 (6,832 ACCEPT + 278 ACCEPT_UNSCORED) |
| L2 hits rejected (fallback search) | 0 | 6,261 |
| plain misses | 8,363 | 7,421 |
| REAL_TIME bypass | 250 | 250 |
| search calls = misses + rejected + REAL_TIME | **8,613** | **13,932** |
| search saved = 1 − calls/21,848 | **60.5776%** | **36.2321%** |
| generations = search calls + L2 hits | 21,042 (963.11/1k) | 21,042 (963.11/1k) |
| fetches, engine rule | 8,076 (369.64/1k) | 8,596 (393.45/1k) |

Every figure equals `test_operational.json` [RECORD]. The engine's fetch rule counts every REAL_TIME URL position, including 285 duplicate positions across the 250 REAL_TIME requests; counting distinct URLs gives 7,791 / 8,311 instead. The published 369.64 / 393.45 use the engine rule.

**Scoring coverage.** 13,371 L2 hits reached the post-L3 verification point (rows with tier L2 or miss_rejected), and `test_scores_L2Verify_g0.55.jsonl` has exactly 13,371 rows with 13,371 distinct IDs [EXECUTED]. Of these, 13,093 received a score (6,832 accepted, 6,261 rejected) and **278 received no score** because no page embedding existed for any served page; `s2_engine.py:163-164` accepts them unverified (`ACCEPT_UNSCORED`, fail-open). So **"scores all 13,093 L2 hits" is inaccurate**: 13,093 is the scored subset of 13,371, and 278 hits (2.1%) were accepted without a score. The study's FINAL_REPORT discloses this; the manuscript sentence does not.

**Why 13,371 rather than 12,429 or 14,541.** The verifier arm is its own sequential replay. A rejected request performs a fresh search and registers its own URL list as a new L2 entry with discovery time t (`s2_engine.py:180-184`), and its generated answer as a new L1 entry, exactly as a miss would. Later requests therefore see a different cache. Tier transitions FreshCache → L2Verify [EXECUTED]: L2→L2 6,749; L2→rejected 5,534; L2→plain miss 146; miss→L2 361; miss→rejected 727; miss→miss 7,275; L1 and REAL_TIME unchanged. So 1,088 requests that missed under FreshCache became L2 candidates under the verifier (361 kept, 727 rejected), which is why the candidate count (13,371) exceeds FreshCache's 12,429. The budget-selected 14,541 belongs to a different configuration (ε_L2 = 0.35) and is unrelated.

## 3. Savings and cost reconciliation (V-06)

Confirmed from search-call outcomes [EXECUTED]: 8,613/21,848 → 60.5776%; 13,932/21,848 → 36.2321%; difference 24.3455 pp. The difference is **not** 6,261/21,848 = 28.66 pp because the extra search calls are 13,932 − 8,613 = 5,319, not 6,261: 727 of the rejections fall on requests that were already misses in FreshCache (no new search), and 361 FreshCache misses became accepted L2 hits in the verifier arm (one search saved each); 146 FreshCache L2 hits became plain misses. Net: 5,534 + 146 − 361 = 5,319 extra searches. Generations are unchanged (963.11/1k) because a rejected hit still generates once. Fetches rise from 369.64 to 393.45 per 1k.

**Verifier overhead.** No timing of the embedding or scoring step exists anywhere in the study (`s1_embed.py`, `s5_test_replay.py`, FINAL_REPORT, logs: no latency or duration record) [EXECUTED grep]. Query and page embeddings were precomputed offline. The online cost is therefore **unmeasured** and must not be called negligible; the manuscript does not quantify it [UNVERIFIED].

## 4. The 400-request verifier audit (V-07–V-09)

- **Population** [EXECUTED]: `audit_sample.jsonl` holds 400 IDs identical, in order, to `validation/heldout_baseline_tuning/answer_audit_k1_16/sample.json`; both arms use the same 400 (`s6_answer_audit.py:94, 116-139`). All 400 have fresh, FreshCache and verifier labels (0 dropped). Serving changed on 166 of 400; verifier verdicts within the 400: 130 ACCEPT, 116 REJECT, 2 ACCEPT_UNSCORED, 152 not L2.
- **Sufficiency counts** [EXECUTED from `audit_suff_llama8b.jsonl` and `audit_suff_qwen7b.jsonl`]: FreshCache 129 SUFFICIENT / 180 INSUFFICIENT / 91 judge disagreement; L2Verify 145 / 161 / 94. Confirmed 129/400 and 145/400.
- **Paired binary test, exact rule** (`s6_answer_audit.py:380-386`): each arm's label is "SUFFICIENT" only when both judges say so; INSUFFICIENT and JUDGE_DISAGREEMENT are pooled as "not sufficient"; all 400 pairs enter. Transition table [EXECUTED]: INSUFF→SUFF 17, DISAGREE→SUFF 2 (together the **19**), SUFF→INSUFF 3 (the **3**), SUFF→DISAGREE 0, plus 126 SUFF→SUFF, 157 INSUFF→INSUFF, 88 DISAGREE→DISAGREE, 6 INSUFF→DISAGREE, 1 DISAGREE→INSUFF. Exact McNemar on 19 vs 3: **p = 8.55 × 10⁻⁴**. Restricting to the 303 pairs judge-agreed in both arms gives 17 vs 3, p = 2.6 × 10⁻³ (sensitivity, not reported in the manuscript).
- **WAI** [EXECUTED]: fresh correct 81/400; FreshCache WAI 7, L2Verify WAI 4; the four are a subset of the seven (both 4, FreshCache-only 3, L2Verify-only 0); exact McNemar p = 0.25. Correctness: FreshCache-only 0, L2Verify-only 4, p = 0.125. All equal `audit_results.json` [RECORD].
- **Conclusion allowed.** The sufficiency shift is significant; the WAI and correctness shifts are not. With 3 and 4 discordant pairs, no equivalence or non-inferiority conclusion follows, and the manuscript correctly claims "no harm detected, not equivalence".

## 5. Serving semantics (V-10) [EXECUTED from code]

`s2_engine.py:146-200`:
1. L2 candidate chosen as in FreshCache (cosine ≥ floor, temporal gate, argmax similarity), lines 147-156.
2. Tentative L3 processing is simulated without cache mutation (`l3_pure`, lines 90-99), producing the evidence list C at the versions L3 would serve, line 161.
3. Score V = max cosine over C's pages (`Support.score`, lines 52-64); missing query or page embeddings give None.
4. None → ACCEPT_UNSCORED; V ≥ γ → ACCEPT; V < γ → REJECT (lines 163-168). On REJECT the candidate is discarded (`hit2 = -1`, line 173) and the tentative evidence is never committed.
5. A rejected request then takes the miss path: search counted (line 181), its own URL list becomes a new L2 entry with discovery time t (lines 182-184), normal L3 handling with commit (lines 186-189), tier recorded as `miss_rejected` (line 192), and a new L1 entry is written from its generation (lines 198-200).
6. The rejected L2 entry is not touched: nothing writes to `l2_qi`, `l2_t` or `l2_urls` at the matched index, so its list and discovery timestamp remain. Accepted hits re-register the list under the incoming query with the discovery time copied (lines 177-179), as in `prep2.py:179-180`.

"Alias chain" is not a data structure in this engine: L2 re-registration copies the discovery time into a new entry and keeps no link back, so the only chain state is the copied timestamp, which rejection leaves unchanged.

## 6. Summary of discrepancies

| item | status |
|---|---|
| γ = 0.55, TPR/FPR/J, AUC 0.779, 463 = 292 + 171, grid 0.00–0.90, exclusion of the 300 audited requests, prereg hash | all confirmed [EXECUTED] |
| 60.58% → 36.23%, 6,261 rejections, N = 21,848, 129/145, 19/3, p = 8.6e-4, WAI 7 vs 4 | all confirmed [EXECUTED] |
| "scores all 13,093 L2 hits" | wrong: 13,371 reached the gate, 13,093 scored, 278 accepted unscored |
| "outcomes ship under artifacts/verifier/" | release bundle uses `artifacts/l2_verify/` with 2 files; neither bundle holds the per-request outcomes, labels' excluded IDs, or audit files |
| verifier overhead | not measured [UNVERIFIED] |
| selection before held-out use | documented within the 2026-09-21 workstream by mtimes, hash and flags; no earlier record exists |
