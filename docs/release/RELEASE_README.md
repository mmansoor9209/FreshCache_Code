# FreshCache — artifact release

Three-tier freshness-aware semantic caching for open-web RAG.
This directory is the reproducibility package for the FreshCache submission.

## What is here

| File | Purpose |
|---|---|
| `ARTIFACT_MANIFEST.json` | SHA-256 and size for every frozen config, core source file and row-level artifact |
| `RUN_ORDER.md` | The order in which the pipeline stages must be executed |
| `requirements.txt` / `environment.yml` | Pinned environment (Python 3.10.20) |
| `pip-freeze-full.txt` | Complete freeze of the environment actually used |
| `DATA_AVAILABILITY.md` | What can and cannot be redistributed |
| `LICENSE_NOTES.md` | Licensing of code, data and third-party models |

## Verifying the release

```bash
python validation_V3/group1_resolution/p1_f08_f10_f26.py   # 75 row-level checks
python validation_V3/remaining_critical_issues/final/integrity_check.py
```

The first re-derives every manuscript-relevant aggregate of the four most
recent workstreams from its row-level file. The second re-hashes the 34
original implementation files and re-runs the anchor replays.

## Reproduction anchors

Any change to the pipeline must still reproduce these exactly:

| Anchor | Value |
|---|---|
| Mixed-age FreshCache (31,201 requests) | 62.7320% saved, 3.4154% drift, 73.3736% coverage, L1 1,175, L2 18,398 |
| Held-out FreshCache (21,848 requests) | 60.5776% saved, L1 806, L2 12,429 |
| Held-out FreshCache WAI | 7/400 |

## Determinism

All replays are deterministic given `schedule=zipf_uniform`, `seed=42`.
Bootstraps use 10,000 resamples at `seed=42`. GPU generation uses greedy
decoding (`do_sample=False`), so answers are reproducible up to kernel-level
floating-point non-determinism.
