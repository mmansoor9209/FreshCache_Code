# R17 verification: mismatch-sampling confidence intervals (Table tab:mismatch)

Read-only on the repository; outputs in `audit/R17/` (`r17_ci.py`, `R17_results.json`, `run.log`). Tags: **[executed]** today, **[historical record]** existing artifact or code.

## 1. Where the table comes from

| Row | Judged items | 3B incumbent judgments | Other jurors | Population strata and N | Aggregation |
|---|---|---|---|---|---|
| SCALM-style | `data/scalm_hit_judgments_24h.jsonl` (600) | same file | `v8/jury/{llama8b,qwen7b,mistral7b}__scalm.jsonl` | `data/scalm_hit_judgment_summary_24h.json` (N 30,067; six bands) | `v8/jury_aggregate.py` |
| SemanticTTL permissive | `semanticttl_hit_judgments_24h.jsonl` (600) | same | `…__semanticttl.jsonl` | `semanticttl_hit_judgment_summary_24h.json` (N 22,561; six bands) | same |
| FreshCache L1 | `l1_hit_judgments_24h.jsonl` (**140**) | same | `…__fc_l1.jsonl` | `l1_hit_judgment_summary_24h.json` (N 1,164; **two** bands) | same |
| FreshCache L2 | `l2_hit_judgments_24h.jsonl` (**590**) | same | `…__fc_l2.jsonl` | `l2_hit_judgment_summary_24h.json` (N 24,021; **three** bands) | same |
| SemanticTTL quality-selected | `Reframe_ResearchPaper/artifacts/q16/jury_selected/` (400, four jurors) | n/a | same folder | `semanticttl_selected_hits_t24h.jsonl` (N 12,795; four bands) | `scripts/q16/aggregate_selected_jury.py` |

Majority of four jurors, 2–2 ties to DIFFERENT, in both aggregators. [historical record]

## 2. Per-row accounting [executed]

Full strata (N_h, n_h, y_h, weight N_h/n_h) are in `R17_results.json` and `run.log`. Summary:

| Row | N | bands | n_h per band | judged n | weighted mismatch |
|---|---|---|---|---|---|
| SCALM-style | 30,067 | 6 | 100 each | 600 | 97.66% |
| SemanticTTL permissive | 22,561 | 6 | 100 each | 600 | 95.40% |
| FreshCache L1 | 1,164 | 2 (0.80–0.90, 0.90–1.01) | 70, 70 | 140 | 19.11% |
| FreshCache L2 | 24,021 | 3 (0.70–0.80, 0.80–0.90, 0.90–1.01) | 150, 220, 220 | 590 | 24.78% |
| SemanticTTL quality-selected | 12,795 | 4 | 100 each | 400 | 53.38% |

All four published estimates and their printed intervals reproduce: 97.7 [97.5, 97.8], 95.4 [94.0, 96.8], 19.1 [14.0, 24.2], 24.8 [21.5, 28.0].

## 3. The CI formulas actually used

**Published rows** (`v8/jury_aggregate.py:81-97`, `:194-195`): estimate $\hat p=\sum_h (N_h/N)\,p_h$ with $p_h=y_h/n_h$; variance $\sum_h (N_h/N)^2\,\frac{p_h(1-p_h)}{n_h}\cdot\frac{N_h-n_h}{N_h-1}$ (finite-population correction); CI $\hat p\pm1.96\,\mathrm{se}$, clipped to [0, 1]. This is a stratified normal-approximation interval on the **judged** counts $n_h$. It does **not** treat weighted pseudo-counts or the full population $N$ as judged observations; "effective judged count" is not a concept in this code.

**Quality-selected row** (`aggregate_selected_jury.py:48-51`): a Wilson interval on $n_{\text{eff}}=\sum_h n_h=400$ applied to the weighted estimate. Here "effective judged count" means only the total number of judged items; the interval ignores the stratification and weights entirely. That is the misstatement in the current implementation, and it is mine.

**Defect in the published formula.** Per-stratum Wald variance $p_h(1-p_h)/n_h$ is exactly zero when a band is judged 100% (or 0%) mismatch. SCALM-style has two such bands covering 27,877 of 30,067 hits (92.7% of the weight), SemanticTTL permissive one (5,255 hits). Those strata contribute no uncertainty at all, so the published SCALM interval [97.5, 97.8] is too narrow.

## 4. Corrected intervals [executed]

Stratified bootstrap: 10,000 replicates, `random.Random(42)`, judged items resampled within each band, $N_h$-weighted rate recomputed. Cluster-aware variant: clusters (`cluster_incoming` in the judgment rows; `query_id → cluster_id` for the quality-selected sample) resampled within band.

| Policy | weighted mismatch | old CI | corrected stratified-bootstrap CI | cluster-aware CI | judged n | population N |
|---|---|---|---|---|---|---|
| SCALM-style | 97.66% | [97.48, 97.84] | [97.45, 97.85] | [97.44, 97.87] | 600 | 30,067 |
| SemanticTTL permissive (0.40) | 95.40% | [94.03, 96.76] | [93.90, 96.68] | [93.86, 96.66] | 600 | 22,561 |
| FreshCache L1 | 19.11% | [13.97, 24.24] | [13.63, 24.71] | [13.88, 24.69] | 140 | 1,164 |
| FreshCache L2 | 24.78% | [21.53, 28.02] | [21.55, 28.13] | [21.53, 28.12] | 590 | 24,021 |
| SemanticTTL quality-selected (0.60, ½) | 53.38% | [48.48, 58.21] (Wilson, unstratified) | [50.14, 56.65] | [50.10, 56.67] | 400 | 12,795 |

Reading: for the four published rows the bootstrap confirms the printed widths, because their design was already stratified. For the quality-selected row the corrected interval is **narrower** than the one I printed (the unstratified Wilson ignored the variance reduction from stratification). Clustering changes nothing material: at most 90 of 504 clusters carry more than one judged item.

**Limitation that the bootstrap shares with the Wald formula.** A band judged 100% mismatch resamples to 100% in every replicate, so the bootstrap also assigns it zero uncertainty. For SCALM-style the honest statement is that the bootstrap interval is a lower bound on the true width. A conservative alternative that does not collapse, the $N_h$-weighted sum of per-stratum Wilson bounds, is recorded in `R17_results.json` (`conservative_bound_pct_sum_of_stratum_wilson`): SCALM-style [93.79, 98.00], SemanticTTL permissive [90.20, 97.23]; it is a bound, not a calibrated interval, and is reported as such.

## 5. Accounting [executed]

| Set | judged items |
|---|---|
| SCALM-style | 600 |
| SemanticTTL permissive | 600 |
| FreshCache L1 | 140 (two bands × 70) |
| FreshCache L2 | 590 (150 + 220 + 220) |
| **four published rows** | **1,930, not 2,400** |
| SemanticTTL quality-selected | 400 additional |
| L2 evidence-sufficiency audit | separate (300 realized reuses, 345 judged rows over five bands); not part of the above |

The manuscript's statement that the 2,400 judged items are "four populations × 600" (Appendix D, Table 3 caption context) is **wrong** and must be corrected to the counts above; the FreshCache L1 row rests on 140 judged items in two bands, and L2 on 590 in three. The "2,400" figure matches no combination of these files. Not edited here.

## 6. What remains unverified

The per-band population counts $N_h$ come from the four `*_hit_judgment_summary_24h.json` files [historical record]; for SCALM-style and permissive SemanticTTL they agree with recounting the released hit files by similarity band, for FreshCache L1/L2 no per-hit population file with similarities was located, so their $N_h$ rest on the summary files alone.
