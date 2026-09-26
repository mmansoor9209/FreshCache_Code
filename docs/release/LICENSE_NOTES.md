# License notes

**No licence is asserted here.** This file records what a licence must
account for; the authors must choose and add the actual licence text before
release.

## Code

Written for this project: `experiment.py`, `collect.py`, `calibrate.py`,
`schedules.py`, the engines under `v13_corrected/`, `v14_baselines/`,
`v16_exp12/`, `v9/`, and all of `validation*/`. A permissive licence
(MIT/Apache-2.0) is appropriate.

## Third-party code

- `vcache_implementation.py` is a **reimplementation ported against** the
  vCache reference (`vcache-project/vCache`,
  `vcache_policy/strategies/verified.py`) and the paper
  (arXiv:2502.03771). It reproduces the algorithm, including a variance map
  transcribed from the reference. **Check the upstream licence before
  redistributing this file**; attribution is required regardless.
- The SCALM baselines implement the algorithm described in
  arXiv:2406.00025. No upstream code was copied.

## Models

Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct are governed by the Llama
Community License; Qwen2.5-7B and Mistral-7B by their own licences; BGE-M3 by
MIT. Weights are not redistributed here.

## Data

See `DATA_AVAILABILITY.md`. Web page text is third-party content and is not
redistributed. Derived hashes and labels are ours.

## Attribution required

Any release must cite SCALM (arXiv:2406.00025) and vCache (arXiv:2502.03771)
as the sources of the baseline algorithms.
