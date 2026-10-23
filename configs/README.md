# Configuration

* `freshcache/` – the standalone FreshCache policy (thresholds, TTL budgets eps, tier multipliers, half-lives, L1 checks) and compositions A-D.
* `baselines/` – one file per baseline and control (NoCache, GPTCache, SCALM-style, vCache-style, SemanticTTL, TieredFixedTTL,
  EquivalentTTL, tier ablations) and the comparator grid (36 SemanticTTL and 20 CategoryAware-style cells).
* `experiments/` – workload and split, generators and judges (models, prompts, budgets), learned L2 gate, answer verifier.
* `protocols/` – protocols and pre-registrations frozen before the corresponding runs (reference-validity protocol, tier-quality study,
  budget, L1-precision and quality-selection pre-registrations, live-stream collection protocols, L2-sweep criteria), the 300 request IDs
  excluded from verifier labelling, and the 24-test Holm family. `*.sha256` files pin the frozen files.
* `data/dataset_sizes.yaml` – sizes of the full benchmark and of the fixed-size audit samples, used by the size checks of `scripts/analysis/dataset_sizes.py` (skipped on `data/sample`).
* `integrity/FILE_HASHES.csv` – hashes of the released code, configuration and `data/sample/` files; stages stop only when released code or configuration differs (run outputs and data inputs are logged, not enforced). The stage integrity checks read only the rows with `mode` = `copied`; rows with `mode` = `release` cover the remaining released `.py`, `.sh`, `.yaml`, `.yml`, `.json`, `.csv`, `.md` and `data/sample/` files.
* `output_dirs.txt` – run-time output directories (created by `scripts/prepare_output_dirs.py`).

Every entry in the YAML files records its value and the code location it mirrors (`source`). Run
`python scripts/analysis/check_config_consistency.py` to verify that the configuration and the code agree (it is the first step of
`run_all.sh`). Values that are selected on validation data at run time (for example the CategoryAware-style cell used in composition D,
the verifier threshold gamma, the gate thresholds) are written by the selection stages to `outputs/` and read from there.
