# Gold-verification diagnostic (Stages 0–3b outputs, read-only)

No LLM judgment was run, no original experiment touched, no gold label changed,
no manuscript edited. Every number below comes from files already in
`validation/gold_verification_400/` plus read-only reads of `data/snapshots/`.

---

## 1. Why only 81 of 786 requests pass both conditions

| Condition | Passing | Rate |
|---|---|---|
| (A) fidelity SAME or base question | **529 / 786** | 67.3% |
| (C) unit VALID_AT_T under both judges | **110 / 786** | 14.0% |
| **Both** | **81** | 10.3% |

If the two filters were independent, 74 would pass; 81 do, so they are near-independent
and the attrition is essentially multiplicative. **Condition C is the binding
constraint** — it alone removes 86% of requests.

Where the 529 fidelity-passers are lost:

| Validity outcome among A-passers | n |
|---|---|
| JUDGE_DISAGREEMENT | **235** |
| SUPERSEDED_AT_T | 122 |
| INDETERMINATE_AT_T | 91 |
| VALID_AT_T | **81** |

**The single largest cause of attrition is judge disagreement, not evidence of a bad
gold.** Across both filters, 435 of the 705 exclusions (61.7%) are unresolved
disagreement (200 fidelity + 235 validity), not a negative finding.

---

## 2. The 175 SUPERSEDED_AT_T units — are they really superseded?

Composition: temporal_align_qa 157, freshqa 13, triviaqa 5. By freshness class:
FAST 70, MEDIUM 60, SLOW 40, TIMELESS 5. By volatility: VOLATILE 157, SINGLE_YEAR 13,
NOT_TIME_KEYED 5.

Question-type classification (regex over the base question, no LLM):

| Type | n | Superseding appropriate? |
|---|---|---|
| recency cue, undated ("current", "latest", "most recent") | **130** | yes — the question asks for the present state |
| unmarked recurring-event phrasing | 36 | usually yes — TemporalAlignQA questions are year-indexed by construction, so "In which city was the ILCA 7 World Championship held?" implicitly means the most recent |
| historical phrasing ("was", "first", "founded") | 6 | **at risk** |
| explicitly dated ("in 2012", "from 2004 to 2011") | 3 | **at risk** |

### Confirmed false positives

The 5 TIMELESS units and the 5 triviaqa units are **the same 5 rows**, and all 5 look
wrong:

| Question | Gold | Why SUPERSEDED is incorrect |
|---|---|---|
| "Appointed in 2012, who is the President of the Fifth French Republic?" | François Hollande | explicitly dated to 2012 — Hollande is correct; 2026 evidence naming Macron does not supersede it |
| "Who was appointed captain of Great Britain's Davis Cup team in 2006?" | John Lloyd | explicitly dated to 2006 |
| "Which architect, President of the Royal Academy from 2004 to 2011…" | Sir Nicholas Grimshaw | explicitly dated range |
| "'Red Lion' was for many years accepted as the most popular pub name…" | 'The Crown' | historical framing |
| "What is kitchen tin foil made from?" | aluminium | timeless fact; cannot be superseded |

This is exactly the failure mode flagged in the brief: a historically correct answer
marked superseded because newer information exists. **All 5 are in the TriviaQA /
TIMELESS stratum**, which makes them cheap to isolate and exclude.

The other 170 are dominated by genuinely current-seeking questions, where SUPERSEDED
is a defensible reading.

---

## 3. Llama vs Qwen on the 321 validity disagreements

Marginals over all 715 judged units:

| Judge | VALID_AT_T | SUPERSEDED_AT_T | INDETERMINATE_AT_T |
|---|---|---|---|
| Llama-3.1-8B | **302** | 219 | 194 |
| Qwen2.5-7B | **104** | **402** | 209 |

Disagreement pattern:

| Llama → Qwen | n |
|---|---|
| VALID → SUPERSEDED | **164** |
| INDETERMINATE → SUPERSEDED | 63 |
| VALID → INDETERMINATE | 43 |
| SUPERSEDED → INDETERMINATE | 42 |
| INDETERMINATE → VALID | 7 |
| SUPERSEDED → VALID | 2 |

**Qwen says SUPERSEDED in 227 of 321 disagreements (70.7%).** The disagreement is
overwhelmingly one-directional: Qwen defaults to "superseded", Llama to "valid". This
mirrors Stage 3a, where Qwen called DIFFERENT and Llama called SAME. The two models
are not interchangeable on either task.

The gap widens exactly where the content moves:

| Class | units | disagree | Llama VALID | Qwen VALID |
|---|---|---|---|---|
| TIMELESS | 171 | 72 (42%) | 129 | 82 |
| SLOW | 179 | 82 (46%) | 79 | **20** |
| MEDIUM | 180 | 83 (46%) | 49 | **1** |
| FAST | 185 | 84 (45%) | 45 | **1** |

Qwen finds essentially nothing valid in MEDIUM/FAST (2 of 365); Llama finds 94. By
source, disagreement is 40% (triviaqa), 46% (temporal_align_qa), 50% (freshqa).

**Evidence size is not the driver**: agreeing units have median 1,999 evidence chars
and 1.18 pages; disagreeing units 2,000 and 1.21. The split is behavioural, not
informational.

---

## 4. Is the TIMELESS / TriviaQA skew retrieval coverage or the verification rules?

**It is the verification rules, not retrieval coverage.** Coverage is flat across
classes:

| Class | units | URLs/unit | pages kept | median chars | zero-evidence |
|---|---|---|---|---|---|
| TIMELESS | 171 | 1.41 | 1.19 | 2000 | 0 |
| SLOW | 179 | 1.42 | 1.12 | 1999 | 0 |
| MEDIUM | 180 | 1.47 | 1.23 | 2000 | 0 |
| FAST | 185 | 1.45 | 1.23 | 2000 | 0 |

Yet VALID rates are 42.7% / 11.2% / 0.6% / 0.5%. Exclusion reasons per class:

| Class | in subset | paraphrase DIFF | paraphrase DISAGREE | validity INDET | validity DISAGREE | validity SUPERSEDED |
|---|---|---|---|---|---|---|
| TIMELESS | **61** | 10 | 50 | 20 | 50 | **4** |
| SLOW | 18 | 16 | 52 | 25 | 62 | 27 |
| MEDIUM | **1** | 14 | 55 | 26 | 57 | **42** |
| FAST | **1** | 17 | 43 | 20 | 66 | **49** |

The skew is produced by two mechanisms acting together: (i) fast-changing golds are
genuinely superseded more often (49 + 42 vs 4), which is a true finding; and (ii) Qwen
almost never returns VALID for MEDIUM/FAST, converting most remaining units into
JUDGE_DISAGREEMENT. Mechanism (ii) is a verifier artifact, not a property of the data.

---

## 5. Unused snapshot evidence

Measured over the 1,015 snapshot pages backing the 715 units:

| Quantity | Value |
|---|---|
| Median stored `extracted_text` | **1,502 chars** (max 5,000) |
| Pages longer than the 2,000-char per-page cap | **453 (44.6%)** |
| Pages shorter than the 200-char MINCHARS floor, dropped entirely | **160 (15.8%)** |
| **Characters stored but never shown to the judge** | **1,007,784 — 44.7% of all stored text** |
| Units where pages_kept < n_urls | 167 |
| Units truncated at the 2,800-char budget | 64 |
| Units sitting at exactly 2,000 chars (one page at the cap) | 253 |

So nearly half the captured evidence never reached the verifier. The caps were
inherited from `prep2.ctx_for` for comparability with the audits, but the verification
task has no reason to be bound by the *generator's* context budget.

The same URLs are also captured at other rounds (run_00 1,015; 1h 1,010; 12h 1,007;
24h 1,012; 7d 985). Those are **not** additional evidence for this question — they
describe different times — but they are usable as a corroboration signal (see §6).

### Lexical corroboration already available

Gold/alias presence in the *same* evidence the judge saw:

| Status | gold present | absent | % present |
|---|---|---|---|
| VALID_AT_T | 77 | 18 | **81%** |
| SUPERSEDED_AT_T | 23 | 152 | 13% |
| INDETERMINATE_AT_T | 1 | 123 | 1% |
| JUDGE_DISAGREEMENT | **102** | 219 | 32% |

**126 units have the gold lexically present in the evidence but are not labelled
VALID_AT_T — 102 of them purely because the judges disagreed.** Representative cases
(all Llama VALID / Qwen SUPERSEDED, gold string present in the page text): "principal
currently in office at St Clare's College" → *Ann Freeman*; "senator representing the
Lagos West Senatorial District" → *Oluranti Adebule*; "Daytona 300 winner" → *Austin
Hill*. The INDETERMINATE bucket, by contrast, is genuinely evidence-poor (1/124 with
the gold present), so it is correctly named.

---

## 6. Feasible improvements — without relaxing correctness or changing golds

Ordered by expected recovery per unit of cost. None alters a gold answer, none
weakens the dual-judge requirement.

1. **Lift the evidence caps for verification** (no new retrieval). Raise the per-page
   cap from 2,000 to the stored 5,000 and drop or lower the 200-char MINCHARS floor.
   Recovers 44.7% of captured text and 160 wholly discarded pages. Recomputation cost:
   re-judging only, 1,430 calls. Highest expected yield, since 32% of the disagreement
   bucket already contains the gold within the *truncated* text.

2. **Add a third independent judge and take a 2-of-3 majority** on the 321
   disagreements. Keeps the standard strict (still no single-judge decision) but
   resolves the one-directional Llama/Qwen split. Cost: 321 calls, one model.

3. **Add a question-type pre-classifier** so explicitly dated and one-time historical
   questions are routed to a different prompt that asks "was this correct *as stated*",
   not "is this still current". This alone fixes the 5 confirmed false positives and
   protects the 9 at-risk rows. Cost: 715 cheap classifications, or a regex pre-pass at
   zero cost.

4. **Report a lexical-corroboration column** alongside the judge labels (already
   computed above, zero cost). It does not decide anything, but it lets the write-up
   separate "judges disagreed and the gold is in the evidence" (102 units) from
   "judges disagreed and it is not" (219).

5. **Use other-round captures as a consistency check** on SUPERSEDED calls: if the same
   URL yields the same superseding answer at every round, the call is stable; if it
   flips, flag it. Zero new retrieval, 715 units × up to 5 rounds.

6. **Calibrate the judges on the 81 already-VALID units** and report per-judge
   precision, rather than treating agreement as truth.

### Recommended next action

Run (1) and (3) together as a Stage 3b-revised pass — expanded evidence plus
question-type-aware prompting — then (2) on whatever disagreement remains. That is
~2,150 LLM calls, no new retrieval, no relaxation of the dual-judge rule, and no change
to any gold answer. Expected effect: a materially larger evidence-valid subset drawn
from the same strict criteria, and removal of the 5 identified false-positive
SUPERSEDED labels.

**What this diagnostic does not claim:** nothing here establishes that any gold answer
is factually correct. The recoverable cases are cases where *temporal validity at the
request time* was under-detected, not cases of verified correctness.
