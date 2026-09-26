# Data availability

## Redistributable

- Query sets, freshness-class annotations, cluster assignments and the
  replay schedule.
- Derived per-URL metadata: `url_hash`, `content_hash`, `n_chars`,
  HTTP status, `substantive` flag, and per-round change labels
  (`v13_corrected/corrected_round_table.jsonl`). These are hashes and
  counters, not page text.
- All row-level experimental outputs listed in `ARTIFACT_MANIFEST.json`.
- Generated answers and model judgments produced by us.
- Calibrated parameters, frozen configs and pre-registrations with hashes.

## Not redistributable

- **Web page text** (`data/snapshots/`). Extracted text from third-party
  sites is retained locally for reproducibility but is not ours to
  redistribute. Snapshots store `extracted_text` only; **no raw HTML was
  retained**, so re-extraction from stored bytes is not possible.
- **Search API responses** in raw form (provider terms).
- Model weights (obtain from the original providers).

## Consequence for reproduction

Stages 1–3 of `RUN_ORDER.md` cannot be re-executed to yield the same Web
state: the pages have changed. Reproduction from stage 4 onward is exact,
because every later stage consumes stored hashes and labels. A third party can
therefore verify every number in the paper without re-crawling, but cannot
reconstruct the original page text from this release.

## Historical limitation, stated plainly

The prospective +24h experiment is frozen against a specific wall-clock window
(T0 at 2026-09-21T12:05–12:28Z). It cannot be re-run retroactively; only its
stored outputs can be verified.
