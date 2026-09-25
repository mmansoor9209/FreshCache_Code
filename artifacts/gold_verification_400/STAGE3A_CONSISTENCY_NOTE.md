# Stage 3a — reporting-consistency check

No judgment, label file or experimental result was changed. This note corrects
summary descriptions only. `paraphrase_labels_*.jsonl`, `paraphrase_fidelity.csv`
and `stage3a_summary.json` are unmodified.

## 1. Why raw 412 SAME / 203 disagreements → final 411 SAME / 200 disagreements

The confusion matrix reports **raw** dual-judge outcomes for all 668 requests.
The final classification applies the approved rule that the **12 requests with an
existing fidelity judgment keep their stored verdict**, which overrides the new
judges for those rows only.

Among the 12 reused requests: Llama-3.1-8B said SAME on all 12; Qwen2.5-7B said
SAME on 9 and DIFFERENT on 3. So raw gives 9 SAME/SAME agreements and 3
disagreements there. Their stored verdicts are 8 SAME and 4 DIFFERENT.

| Final label | Reconciliation | Result |
|---|---|---|
| SAME | raw 412 SAME/SAME − 9 reused-overridden + 8 stored-SAME | **411** |
| JUDGE_DISAGREEMENT | raw 203 − 3 reused-overridden | **200** |
| DIFFERENT | raw 53 DIFFERENT/DIFFERENT + 4 stored-DIFFERENT | **57** |

411 + 200 + 57 = 668. Verified directly from `paraphrase_fidelity.csv`.

## 2. Newly judged vs reused, stated separately

| Set | n | SAME | DIFFERENT | UNCLEAR | JUDGE_DISAGREEMENT | raw agreement |
|---|---|---|---|---|---|---|
| **Newly judged** (dual-judge rule) | **656** | 403 | 53 | 0 | 200 | 456/656 = 69.5% |
| **Reused** (stored verdict is final) | **12** | 8 | 4 | 0 | 0 | 9/12 raw, not used |
| All | 668 | 411 | 57 | 0 | 200 | 465/668 = 69.6% |

The reused rows cannot contribute a `JUDGE_DISAGREEMENT` by construction, because
their final label comes from the stored verdict. Every one of the 200
disagreements is therefore in the newly judged set.

## 3. Correction — Llama-3.1-8B SAME count

**The earlier report said Llama-3.1-8B labelled 614/668 SAME (91.9%). That is
wrong.** The correct marginals, from the confusion matrix and verified against the
label file:

| Judge | SAME | DIFFERENT | UNCLEAR | Total |
|---|---|---|---|---|
| Llama-3.1-8B | **608 (91.0%)** | 54 | 6 | 668 |
| Qwen2.5-7B | 413 (61.8%) | 255 | 0 | 668 |

Llama row of the matrix: 196 + 412 + 0 = 608. The 614 figure was an arithmetic
slip in the narrative; no data was affected, and the Qwen figure (413, 61.8%) was
correct.

The substantive finding is unchanged: 199 of the 203 raw disagreements are
Llama SAME / Qwen DIFFERENT, a one-sided leniency gap, which is why Cohen's
kappa is 0.252 despite 69.6% raw agreement.

## Files affected

None. This note is additive.
