# FreshCache-L2Verify — optional L2 evidence-verification ablation

All work and all outputs are confined to
`<PROJECT_ROOT>/validation_V2/l2_evidence_verification/`. No existing
FreshCache result, script, dataset, judgment or manuscript file was modified.

## The ablation

FreshCache is unchanged except for one additional check:

```
L2 hit  ->  normal L3 processing  ->  evidence C
V(q,C) = max_c cosine(E(q), E(c))          BGE-M3 dense, precomputed offline
V >= gamma : keep the L2 reuse
V <  gamma : REJECT -> fresh search -> normal L3 -> generation
```

L1, the L2 threshold/floor and temporal gate, half-lives, epsilons, L3 logic,
timestamps, query order and cache-update semantics are untouched. On rejection
the request registers exactly what an L2 miss registers; the rejected entry stays
in the cache. **No LLM runs in the online verification path** — BGE-M3 is an
embedding model and all 31,201 query and 39,403 page embeddings were computed
offline (`s1_embed.py`). Only evidence available at serving time is used: each
page at the version L3 actually served, read with the audits' own 2,000-char
loader.

**Engine fidelity.** `s2_engine.replay` is a transcription of
`validation/mixed_age_full_policy_audit/prep2.py::replay`. With `gamma=None` it
reproduces that function **exactly** on both splits — identical aggregates and
**0 per-request tier/evidence/served mismatches** on 9,353 validation and 21,848
test requests (`s2_verify_fidelity.py`). The replay is genuinely sequential, so
rejections change later cache state.

---

## 1. Selected gamma

Pre-registered before any sufficiency label existed (`out/prereg_gamma.json`,
sha256 `18c7e0a965846311e4576eb7ab331feb`): candidates 0.00–0.90 step 0.01,
objective **maximise Youden's J = TPR − FPR** with POSITIVE = INSUFFICIENT,
tie-breaks smaller gamma then lower FPR.

Labels: 600 **validation-split** L2 hits, stratified by freshness class ×
similarity band, seed 42, with **all 75 requests appearing in the existing
300-case audit excluded** (5,034 validation L2 hits → 4,959 eligible). Judged by
Llama-3.1-8B + Qwen2.5-7B on the **actual served evidence**, both must agree.

| | |
|---|---|
| Judge-agreed usable labels | **463** (INSUFFICIENT 292, SUFFICIENT 171) |
| Judge disagreement (excluded) | 131 |
| Mean V: INSUFFICIENT / SUFFICIENT | 0.4817 / 0.5879 |
| Discrimination AUC | **0.7787** |
| **Selected gamma** | **0.55** |
| At gamma=0.55 on validation | J = 0.4554, TPR 0.6952, FPR 0.2398, reject rate 52.7% |

The gate separates the two classes well above chance but far from cleanly: at the
selected operating point it rejects 69.5% of insufficient evidence while also
rejecting 24.0% of sufficient evidence.

## 2–5. Held-out test operational results (21,848 requests, |S| = 12,568)

| Arm | search savings | L1 | L2 | L3 | fetch/1k | gen/1k | drift | coverage | accepted / rejected |
|---|---|---|---|---|---|---|---|---|---|
| **FreshCache** | **60.5776%** | 806 | 12,429 | 23,446 | 369.64 | 963.11 | 3.2798% (296/9,025) | 71.8094% | — |
| **L2Verify γ=0.55** | **36.2321%** | 806 | 7,110 | 23,015 | 393.45 | 963.11 | 3.2359% (299/9,240) | 73.5201% | **6,832 / 6,261** |
| L2Verify γ=0.50 | 44.4892% | 806 | 8,914 | 23,195 | 385.98 | 963.11 | 3.3034% (305/9,233) | 73.4644% | 8,636 / 4,112 |
| L2Verify γ=0.60 | 25.3204% | 806 | 4,726 | 22,802 | 401.91 | 963.11 | 3.3460% (309/9,235) | 73.4803% | 4,435 / 9,071 |

FreshCache reproduces the published held-out savings of **60.5776%** exactly.
278 L2 hits (2.1%) had no page embedding and were **accepted unverified**
(fail-open); this is disclosed, not hidden.

**Cost.** The gate rejects **6,261 of 13,093 scored L2 hits (47.8%)** and costs
**−24.35 pp of search savings** — roughly 40% of FreshCache's headline benefit —
while raising fetch/1k from 369.64 to 393.45. Generations are unchanged (963.11)
because a rejected L2 hit still generates, just from freshly searched evidence.

**Drift and coverage.** Drift moves −0.044 pp and coverage +1.71 pp; both are
essentially flat. *Caveat, stated plainly:* these absolute values come from my own
scorer (drift labels over L3-reused pages, applied identically to every arm) and
are **not identical** to the published `evaluate_test` figures (3.0678%, 73.4007%),
which use `mixed_engine`'s `risk_cget` labels. The arm-to-arm comparison is valid;
the absolute level should not be quoted against the published table.

## 6. Actual-artifact evidence sufficiency (held-out 400, both arms)

Judged on the evidence each arm actually served, same two judges, both must agree.

| Arm | SUFFICIENT | INSUFFICIENT | judge disagreement | of all 400 | 95% CI | of judge-agreed | 95% CI |
|---|---|---|---|---|---|---|---|
| FreshCache | 129 | 180 | 91 | **32.25%** | 27.86–36.98 | **41.75%** | 36.38–47.32 |
| **L2Verify γ=0.55** | **145** | 161 | 94 | **36.25%** | 31.69–41.07 | **47.39%** | 41.86–52.98 |

Paired: **19 requests become SUFFICIENT, 3 become INSUFFICIENT**, exact McNemar
**p = 8.55 × 10⁻⁴**. This is the clearest positive result in the experiment: the
gate significantly improves the sufficiency of what gets served.

**The original stored-artifact result is not superseded and is not hidden.** The
300-case actual-artifact audit found **81/300 = 27.00% sufficient** (81/242 =
33.47% among judge-agreed) for unmodified FreshCache on a different, paraphrase-only
sample from the fixed-24 h replay. The 32.25% here is the same *kind* of
measurement on the mixed-age held-out 400; the two are not interchangeable.

## 7–8. Answer quality on the held-out 400

Fresh reference identical across arms: 81/400 correct. L2Verify changed serving on
**166 of 400** requests (116 rejections, 130 accepts, 2 unscored-accepts, 152 not
L2). 55 new generations; all other answers and judgments reused byte-for-byte.

| Arm | accuracy | 95% CI | WAI | 95% CI | conditional WAI | 95% CI |
|---|---|---|---|---|---|---|
| FreshCache | 75/400 = **18.75%** | 15.23–22.87 | **7** (1.75%) | 0.85–3.57 | **8.642%** | 4.25–16.78 |
| **L2Verify γ=0.55** | 79/400 = **19.75%** | 16.14–23.93 | **4** (1.00%) | 0.39–2.54 | **4.938%** | 1.94–12.02 |

| Paired test | Result |
|---|---|
| WAI: both 4, FreshCache-only **3**, L2Verify-only **0** | exact McNemar **p = 0.25** |
| Correctness: FreshCache-only 0, L2Verify-only **4** | exact McNemar **p = 0.125** |

Every directional movement favours L2Verify — +1.00 pp accuracy, 3 fewer WAI
events, conditional WAI roughly halved, and **not a single WAI event created** —
but with 3 and 4 discordant pairs neither test is significant. **This is "no harm
detected and a consistent favourable direction", not a demonstrated improvement.**

## 9. Sensitivity

γ ± 0.05 is in the table above. The gate is steep: ±0.05 moves savings from
44.49% to 25.32% (a 19-point swing) and rejections from 4,112 to 9,071, while
drift stays within 0.11 pp and coverage within 0.06 pp. The operating point is
therefore **highly sensitive on cost and insensitive on drift**, which is a real
weakness of the design.

## 10. Sanity checks

| Check | Result |
|---|---|
| Engine reproduces `prep2.replay` at gamma=None | **exact**, 0/9,353 and 0/21,848 mismatches, aggregates identical |
| FreshCache held-out savings anchor | **60.5776%**, matches the published value |
| Validation clusters present in the test stream | **0** (asserted at runtime) |
| Gamma selected using test data | **No** — validation labels only; `selection_used_test_data: false` |
| Trained on the 300 audited examples | **No** — all 75 overlapping requests excluded before sampling |
| LLM in the online verification path | **No** — embeddings precomputed offline |
| Original files modified | **None.** Spot-checked md5: `answer_audit_k1_16/answers.jsonl` `955a1cea…`, `l2_three_path/three_path_results.csv` `e885c58b…`, `actual_l2_artifact_audit/a_per_case.csv` `d0c9be28…` |
| Outputs outside this folder | **None** — `validation_V2/` contains only `l2_evidence_verification/` |

---

## Verdict: **B) PARTIALLY RESOLVES** the L2 reviewer concern

**What it does resolve.** The concern is that FreshCache reuses L2 URL lists whose
evidence does not support the incoming question. L2Verify demonstrably detects and
removes a large share of exactly those cases: it rejects 69.5% of
judge-agreed-insufficient evidence on validation, and on the held-out 400 it moves
19 requests from insufficient to sufficient against 3 in the other direction
(**p = 8.6 × 10⁻⁴**). It also never creates a WAI event (0 L2Verify-only) while
removing 3, and halves conditional WAI. The mechanism is cheap, uses only
serving-time evidence, and keeps FreshCache's methodology otherwise intact.

**What it does not resolve.** Three things.

1. **Absolute sufficiency stays low.** Even after verification, only **36.25%** of
   served evidence is judge-agreed sufficient (47.39% of agreed cases). The
   reviewer's underlying observation — that most reused L2 evidence does not
   clearly support the question — survives the fix. The unmodified 300-case
   figure of **81/300 = 27.00%** likewise stands.
2. **The cost is severe.** −24.35 pp of search savings, a ~40% relative loss of
   FreshCache's headline benefit, and 47.8% of scored L2 hits rejected. At
   36.23% savings FreshCache-L2Verify is a materially different system, not a free
   safety net.
3. **The answer-quality gain is not demonstrated.** p = 0.25 (WAI) and p = 0.125
   (correctness) on 3 and 4 discordant pairs. The direction is consistent and
   never adverse, but 400 requests cannot establish the improvement, and
   non-significance here must not be read as equivalence in either direction.

**Why not A.** A strong resolution would show that verification restores served
evidence to a high sufficiency level and that answer quality improves
significantly, at acceptable cost. None of the three holds.

**Why not C.** The sufficiency improvement is real and statistically significant,
the gate discriminates well above chance (AUC 0.779), and it strictly reduces
WAI. It is a genuine, working mitigation with a measured price.

**Recommendation.** Report L2Verify as an **optional ablation** with its full cost
disclosed, not as a default configuration, and keep the unmodified 27.00% /
32.25% sufficiency figures in the paper. The honest framing is that evidence
sufficiency can be improved significantly by a cheap serving-time check, but only
by surrendering a large fraction of the savings that motivate L2 reuse in the
first place — which is itself an informative result about the L2 mechanism.

## Files

`s1_embed.py`, `s2_engine.py`, `s2_verify_fidelity.py`, `s3a_prereg.py`,
`s3b_label.py`, `s4_select_gamma.py`, `s5_test_replay.py`, `s6_answer_audit.py`,
`run_audit.sh`, `lib/` (copied reference scripts), and `out/` containing
`emb_*.npz`, `prereg_gamma.json`, `val_label_tasks.jsonl`, `val_support_*.jsonl`,
`val_labels.jsonl`, `gamma_selection.json`, `test_operational.json`,
`test_scores_*.jsonl`, `test_rows.pkl`, `audit_sample.jsonl`,
`audit_answers.jsonl`, `audit_judge_*.jsonl`, `audit_suff_*.jsonl`,
`audit_results.json`, `run_audit.log`.
