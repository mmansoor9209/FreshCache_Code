# FreshCache-Bench data

This repository includes a representative sample of FreshCache-Bench (data/sample/) for running the pipeline end to end. Third-party Web page content is referenced by URLs and content hashes rather than redistributed.

All scripts read the data from one setting, `DATA_DIR`, and write results to `OUTPUT_DIR` (default `outputs/`, git-ignored). Set them in
`.env` (copy `.env.example`) or in the shell; `scripts/run_sample_quickstart.sh` sets `DATA_DIR=data/sample`:

```bash
export DATA_DIR=data/sample
```

## Sources

* **SealQA**, **FreshQA** (Hugging Face `natyou/freshqa_10_06`) and **TemporalAlignQA** (`ROIM/temporal-alignment-qa`) questions, built by
  `scripts/data/query_construction.py`; FreshQA false-premise items are excluded.
* Hand-authored **Korean** questions and Korean paraphrase templates (`scripts/data/korean_query_generation.py`) and **REAL_TIME** questions.
* English paraphrases generated with `humarin/chatgpt_paraphraser_on_T5_base` (`scripts/data/paraphrase_generation.py`).
* A five-round web panel (baseline, about 1 h, 12 h, 24 h and 7 d) collected with `scripts/data/web_collection.py`.

Freshness classes: SealQA/FreshQA never- and slow-changing questions map to TIMELESS/SLOW, a tech/science keyword rule splits faster items
into MEDIUM/FAST; TemporalAlignQA score cut-offs 0.10, 0.40 and 0.75 assign SLOW, MEDIUM and FAST. The 31,201-request replay is split
cluster-disjointly into 9,353 validation and 21,848 held-out requests (`scripts/data/validation_heldout_split.py`).

## Files in `data/sample/`

| File | Content |
|---|---|
| `queries.jsonl` | base questions: `query_id`, `query`, `freshness_class`, `language`, `source` |
| `paraphrase_clusters.jsonl` | paraphrase requests: `query_id`, `cluster_id` (= base question), `base_query_id`, `query`, `freshness_class`, `language`, `is_paraphrase` |
| `url_manifest.jsonl` | ranked search results per base question and collection round: `run_id`, `query_id`, `rank`, `url`, `url_hash`, `domain`, `content_hash`, `snapshot_available`, `retrieval_timestamp` (titles and snippets are blank) |
| `change_log.jsonl` | detected content changes between rounds: `url_hash`, `run_id`, `baseline_hash`, `new_hash`, `noise`, `domain_vol` |
| `derived/round_table.jsonl` | per URL and round: `content_hash`, `text_hash`, `n_chars`, `status_code`, `fetched_at`, `substantive` |
| `requests.jsonl` | one row per request: `request_id`, `query`, `cluster_id`, `freshness_class`, `language`, `split`, arrival time in the 31,201-request stream (`arrival_t_primary_stream`), reference answer and aliases, ranked URL list with URL and content hashes |
| `sample_summary.json` | 60 whole clusters, 60 requests per freshness class (300 requests), 148 URLs, 220 requests with a reference |

Generated files (not tracked): `query_embeddings_bgem3.npy` and `sim_matrix_bgem3.npy`, the BGE-M3 embeddings and pairwise cosine
similarities of all requests in the record order of `freshcache/method/tiered_cache_model.py` (`build_query_records`). They are written into
`DATA_DIR` by `scripts/data/query_embeddings.py`, the first data step of `scripts/run_sample_quickstart.sh` and `run_all.sh`.

The same file layout is produced by the dataset-construction block of `run_all.sh` (`scripts/data/`), which builds the benchmark from the
source datasets and a new web collection. `configs/integrity/FILE_HASHES.csv` records the hashes of the released code, configuration and
`data/sample/` files; data inputs that differ from the paper's collection (the sample, re-fetched pages, a new collection) produce a warning,
not a stop.

## Page text

Stages that read page text use `scripts/data/page_refetch.py`, which re-fetches pages by URL and checks them against the stored content
hashes (same fetcher, text extraction and content hash as `scripts/data/web_collection.py`; no API key). It reports each URL as UNCHANGED,
CHANGED, UNAVAILABLE or NO_STORED_HASH (`refetch_report.csv`, `refetch_summary.json` under `OUTPUT_DIR/page_refetch/`). With
`--build-data DIR` it writes a data directory with every file of `DATA_DIR` plus the re-fetched pages, which any stage reads through
`DATA_DIR=DIR`; `REFETCH=1 bash scripts/run_sample_quickstart.sh` runs the sample this way.

```bash
python scripts/data/page_refetch.py --dry-run                          # what would be fetched; no network access
DATA_DIR=data/sample python scripts/data/page_refetch.py --build-data outputs/sample/refetched_data
```

Results on re-fetched pages may differ from the paper because of web content drift: pages change or disappear after collection.

External workloads used by the live studies are placed by hand: the QReCC test split (`qrecc_test.json` in `outputs/natural_workload/data/`),
MS MARCO train queries (`queries.train.tsv` in `outputs/write_controls/data/`) and the FreshQA public sheet of 2026-04-21
(`freshqa_2026-04-21.csv` in `outputs/live_query_block/data/`).
