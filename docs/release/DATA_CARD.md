# FreshCache-Bench — data card

**Version** `freshcache-bench-v1` · manifest `MANIFEST.json` (SHA-256 per file)

## What this is
31,201 open-web QA requests replayed over a simulated seven-day stream, paired
with a five-round longitudinal panel of 8,635 URLs carrying per-round content
hashes, character counts, HTTP status and observability flags. It exists to
study **temporal reuse decisions**: whether a cached answer, URL list or page
may still be served, scored against what the Web actually did.

## Composition
| | Count |
|---|---|
| Collected base questions | 8,072 |
| Retained (≥1 successful baseline fetch) | 6,258 (6,012 en / 246 ko) |
| Paraphrases | 24,943 |
| Replay requests | 31,201 |
| Panel URLs with baseline snapshot | 8,635 (4,182 domains) |
| Snapshot rounds | baseline, ≈1h, 12h, 24h, 7d |
| Cluster-disjoint split | 1,877 validation / 4,381 test clusters |

Sources: SealQA, FreshQA, TemporalAlignQA, plus hand-authored Korean and
REAL_TIME seed queries.

## Intended use
Evaluating cache-reuse policies where evidence changes over time; comparing
freshness, evidence sufficiency and answer correctness as **separate**
properties.

## Known biases and limits — read before using
- **Repeat-heavy by construction.** Every base question has ~4 paraphrases.
  Savings measured here do not transfer to arbitrary production traffic.
- **Paraphrases inherit the base question's gold answer *and* URL list.**
  Within-cluster URL overlap is true by construction and carries no
  evidential weight.
- **Roughly half the panel is body observable** (4,382/8,635 at 24h), and
  observability is not random. At the class level, volatile classes are in
  fact *more* observable (Cramér's V = 0.084).
- **English-dominant.** Korean (246 base queries) is a diagnostic only; the
  entity guard is English-only.
- **REAL_TIME is small** (72 base questions against a 200 target) and bypasses
  the cache, so it never produces a hit.
- **Reference validity is the binding limit on answer metrics.** Of 400
  held-out references, 57 are supported, 101 contradicted, 178 temporally
  ambiguous, 64 unverifiable. All adjudication is model-based; there are **no
  human-adjudicated labels**.
- **Retrospective.** The replay scores stored snapshots; it does not evaluate
  prospective online prediction.

## Redistribution
Page **text** is third-party content and is not redistributed — the archive
ships hashes, counts, status codes and change labels. No raw HTML was ever
retained, so parsing cannot be redone. The Web has since changed, so the
collection stages cannot be re-executed to the same state; everything from
the round table onward reproduces exactly. Model weights come from their
original providers.

## Maintenance
Single frozen version tied to the manuscript. Anonymity audit reports 0
residual author identifiers across the 1,038 released files.
