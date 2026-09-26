# FreshCache-Bench — dataset provenance

Five files. Together they let a reader rebuild the request stream, the
cluster-disjoint split and every temporal quantity in the paper without
refetching the web.

| File | Rows | What it is |
|---|---|---|
| `queries_base.jsonl` | 8,072 | Every base question collected, with `source` dataset, assigned `freshness_class`, `original_freshness`, `topic`, `language`, and `retained` |
| `paraphrase_clusters.jsonl` | 24,943 | Every request in the benchmark, with `cluster_id`, `base_query_id`, `is_paraphrase` and the class it inherits |
| `split.json` | — | The cluster-disjoint split: `seed` 42, `validation_fraction` 0.3, stratified by freshness class, with both cluster id lists and per-class counts |
| `longitudinal_panel.jsonl` | 43,175 | One row per (URL, re-observation round): content hashes and character counts at baseline and at the round, HTTP statuses, fetch timestamps, the observability `label` and its `reason` |
| `validators.csv` | 42,098 | Per (URL, round) HTTP caching validators: `etag`, `last_modified`, `content_type`, `cache_control`, alongside status, content hash and length |

## Reconstructing the headline counts

- 8,072 base questions, of which **6,258 are retained** and 1,814 excluded.
  `retained` is derived: a base question is retained exactly when it heads a
  cluster in `paraphrase_clusters.jsonl`.
- 6,258 clusters over 24,943 requests. Adding the 6,258 base questions
  themselves gives the **31,201-request** stream.
- The split gives **21,848 held-out** and 9,353 validation requests
  (4,381 and 1,877 clusters), disjoint by construction — assert it yourself
  from the two id lists.
- The panel covers **8,635 URLs** across a baseline round `run_00` and five
  re-observation rounds at 1 h, 12 h, 24 h, 48 h and 7 d. The 48 h round is
  partial (160 URLs); it is present and flagged rather than dropped.
- 13,169 of the 42,098 panel observations carry an HTTP validator. Most servers
  send none, which is the reason the paper does not rely on them.

## What is deliberately not here

**Page bodies are not redistributed.** We do not own the crawled text. The
panel ships content *hashes* and character counts instead, which is what every
change/volatility number in the paper is actually computed from, so each result
remains checkable without the copyrighted text.

**Per-question exclusion reasons were not logged at collection time.** The
1,814 excluded base questions are identifiable (`retained: false`), but the
individual reason each was dropped was never recorded. The governing rules are
stated in the supplement and implemented in the shipped `build_queries.py`
(FreshQA false-premise items dropped; TemporalAlignQA items with a unique
answer ratio below 0.10 dropped as too stable). We state this rather than
reconstruct reasons after the fact.
