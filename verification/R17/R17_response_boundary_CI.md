# R17 response: boundary-aware intervals for Table 3

Computed 2026-09-24 by `r17_boundary_ci.py` (output `R17_boundary_ci.json`, appended to `run.log`) from the per-band counts in `R17_results.json` (N_h, n_h, y_h for every band of every row). No manuscript file was edited.

## Method

Each band h is an independent binomial sample of n_h judged hits with y_h mismatches, weighted by W_h = N_h / N. The point estimate is unchanged: p̂ = Σ_h W_h y_h / n_h.

For the interval, each band's proportion gets the Jeffreys posterior Beta(y_h + ½, n_h − y_h + ½). Because the bands are independent, the posterior of the weighted total p = Σ_h W_h p_h is the convolution of these Betas, obtained by Monte Carlo (200,000 draws, seed 42); the 2.5th and 97.5th percentiles give the 95% interval. This is a joint interval for the stratified estimator, not a union of per-band intervals, and it is boundary-aware: a band judged 100% mismatch out of 100 contributes Beta(100.5, 0.5), whose 2.5th percentile is 0.963, instead of the zero variance that the Wald and bootstrap formulas assign. The Jeffreys interval has near-nominal frequentist coverage for binomial proportions including at the boundary (Brown, Cai and DasGupta 2001), so the combined interval inherits calibrated coverage band by band. A uniform-prior (Clopper–Pearson-type) variant is reported as a sensitivity check.

## Result

| Reuse policy | Hits N | judged n (bands) | Mism. | printed 95% CI (V3) | stratified bootstrap | **boundary-aware 95% (Jeffreys)** | uniform-prior variant | bands at 100% (weight) |
|---|---|---|---|---|---|---|---|---|
| SCALM-style | 30,067 | 600 (6×100) | 97.7% | [97.4, 97.9] | [97.4, 97.9] | **[95.6, 97.7]** | [94.6, 97.6] | 2 (92.7%) |
| SemanticTTL (θ=.40) | 22,561 | 600 (6×100) | 95.4% | [93.9, 96.7] | [93.9, 96.7] | **[93.2, 96.3]** | [92.6, 96.0] | 1 (23.3%) |
| SemanticTTL (θ=.60, κ=½) | 12,795 | 400 (4×100) | 53.4% | [50.1, 56.7] | [50.1, 56.7] | **[50.0, 56.6]** | [49.9, 56.6] | 0 |
| FreshCache L1 | 1,164 | 140 (2×70) | 19.1% | [13.9, 24.7] | [13.6, 24.7] | **[14.3, 25.3]** | [14.7, 25.8] | 0 |
| FreshCache L2 | 24,021 | 590 (150/220/220) | 24.8% | [21.5, 28.1] | [21.6, 28.1] | **[21.7, 28.3]** | [21.8, 28.4] | 0 |

Reading. For the three rows with no saturated band the boundary-aware interval agrees with the bootstrap to within 0.6 points, so the printed widths were correct there. For SCALM-style the interval widens from 0.5 to 2.1 points and becomes asymmetric below the estimate, because 92.7% of its hit population sits in two bands whose 100/100 judgments carry real but one-sided uncertainty. For permissive SemanticTTL the lower limit moves from 93.9 to 93.2. The sum-of-Wilson bound ([93.8, 98.0] for SCALM-style) is dropped: it has no joint coverage interpretation and is superseded. Every ordering claim in Section 5.3 survives: the SCALM-style and permissive SemanticTTL intervals do not overlap the FreshCache L1 or L2 intervals, and the quality-selected SemanticTTL interval [50.0, 56.6] stays far above L1's [14.3, 25.3].

## Replacement text

**Table 3 caption (replace the CI clause):** "95% intervals are boundary-aware stratified intervals: independent Jeffreys posteriors per similarity band combined with the population weights N_h/N (Appendix D)."

**Table 3, 95% CI column:** SCALM-style [95.6, 97.7]; SemanticTTL [93.2, 96.3]; SemanticTTL (θ=.60, κ=½) [50.0, 56.6]; FreshCache L1 [14.3, 25.3]; FreshCache L2 [21.7, 28.3].

**Appendix D, replace the two paragraphs from "Its 95% interval is obtained…" through "…calibrated confidence interval":**

"Its 95% interval treats each band as an independent binomial sample and gives the band proportion the Jeffreys posterior Beta(y_h + ½, n_h − y_h + ½). The posterior of the weighted total Σ_h W_h p_h is obtained by Monte Carlo over the bands (200,000 draws, seed 42) and its central 95% region is reported. The interval is joint, not a union of per-band limits, and it is boundary-aware: a band judged 100% mismatch out of 100 contributes an interval of [96.3%, 100%] rather than zero variance. Jeffreys intervals have near-nominal coverage for binomial proportions including at the boundary, so the combined interval inherits calibrated coverage. For SCALM-style, two bands holding 92.7% of the hit population are judged 100% mismatch; the interval is therefore asymmetric, [95.6, 97.7] around 97.7%, whereas the earlier cluster-bootstrap interval [97.4, 97.9] understated the uncertainty. For rows without a saturated band the two methods agree within 0.6 points. A uniform-prior variant moves every limit by at most 1.0 point and changes no comparison."

**Table 18 (change record), Mismatch sampling / CIs row, Revised column:** add "boundary-aware stratified Jeffreys intervals replace the bootstrap; SCALM-style [95.6, 97.7], permissive SemanticTTL [93.2, 96.3]".
