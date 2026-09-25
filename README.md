# FreshCache: Freshness-Aware Caching for Open-Web Retrieval-Augmented LLMs

Code, benchmark metadata, frozen configurations, row-level results and verification
procedures for the paper *FreshCache: Freshness-Aware Caching for Open-Web
Retrieval-Augmented LLMs* (WWW '27, anonymous submission). 

FreshCache is a three-tier cache for open-web RAG. It reuses **answers (L1)**,
**URL lists (L2)** and **page content (L3)** under separate semantic and temporal
eligibility rules, so that query similarity alone never decides whether a stored
artifact may be served. On the full 31,201-request replay it avoids 62.73% of
searches at 3.42% observed content drift; on the cluster-disjoint held-out replay
it avoids 60.58% of searches against 3.69% for its answer-only variant, and it
induces 7 wrong answers on 400 audited requests where the quality-selected
SemanticTTL baseline induces 25.

## Repository map

| Path | Contents |
|---|---|
| `paper/` | The paper PDF, the conceptual figure source, bibliography files |
| `freshcache/` | The FreshCache package: tiered caches (`l1_cache.py`, `l2_cache.py`, `l3_cache.py`), the temporal risk model (`risk_model.py`), entity/answer-type validators, embeddings, workload models |
| `scripts/core/` | The replay and evaluation engines used for every reported number: `experiment.py` (query records, semantic gates, similarity matrix), `mixed_engine.py` / `engine_all.py` / `corrected_engine.py` (mixed-age and fixed-age replays), `schedules.py` (arrival schedules), `calibrate.py` (half-life calibration), `collect.py` (Web collection), `lookup_cost.py`, `deployment_cost.py`, `vcache_implementation.py` (ported vCache baseline) |
| `scripts/` (other folders) | Every workstream runner behind the paper's tables: baselines, stricter L1 gate, budget selection, observability, verifier, latency, external benchmarks, audits. Each folder has a `FINAL_REPORT.md` or log describing what it produced |
| `data/` | FreshCache-Bench metadata: 8,072 base questions, 24,943 paraphrases with cluster ids, the cluster-disjoint split (seed 42), the five-round longitudinal panel of 8,635 URLs (content hashes, lengths, HTTP status, observability labels, fetch timestamps) and per-round HTTP validators. See `data/README.md` |
| `artifacts/` | Row-level outputs: answer audits (generated answers and judge labels), mismatch juries, gold-reference verification, verifier study, budget selection, stricter-gate study, observability analyses, timing traces and stage pools, external-benchmark diagnostics, the 57/107 reference crosswalk |
| `verification/` | Independent re-derivation audits of the paper's numbers from the row-level files (reports, recompute scripts, tables). See `verification/README.md` |
| `docs/release/` | The frozen release records: `MANIFEST.json` (SHA-256 for 1,129 files), `ARTIFACT_MANIFEST.json`, `RUN_ORDER.md`, `DATA_AVAILABILITY.md`, `DATA_CARD.md`, `LICENSE_NOTES.md`, sanitisation and anonymity audits, full `pip freeze` |
| `verify_release.py` | Standard-library script that checks every manifest hash and re-derives headline numbers from the row-level records |

## Quick verification (no GPU, no network)

```bash
python verify_release.py
```

This checks the SHA-256 of all 1,129 manifest files and recomputes, from row-level
records only: the held-out answer audit (fresh correct 81/400, FreshCache WAI 7,
SemanticTTL WAI 64); the stricter-gate mismatch rates (43.30% / 18.22% / 14.97% on
806 / 450 / 314 hits); the validation budget selection (10 of 48 eligible triples,
selected (0.10, 0.35, 0.25) at 67.4864% validation savings); the optional verifier
(gamma 0.55, 463 labels, AUC 0.7787, 13,093 scored L2 hits, 6,261 rejections,
60.5776% to 36.2321% savings, joint sufficiency 129 to 145); the 40-request timing
table; and the 57 / 107 / 41 reference crosswalk.

## Installation

```bash
conda env create -f environment.yml      # Python 3.10, pinned versions
conda activate freshcache
python -m spacy download en_core_web_sm  # entity guard
```

The exact environment used for the paper is recorded in
`docs/release/pip-freeze-full.txt`. Models are downloaded from their providers:
BAAI/bge-m3 (embeddings), meta-llama/Llama-3.2-3B-Instruct (generator),
meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct and
mistralai/Mistral-7B-Instruct-v0.3 (judges and jurors). Set `SERPER_API_KEY` only
if you intend to re-run live search stages.

## How the method is implemented

* **Semantic gates** (`scripts/core/experiment.py`): L1 reuse requires cosine
  similarity at or above 0.80 on L2-normalised BGE-M3 vectors, content-token Jaccard
  at or above 0.30, answer-type agreement when both types are detectable, no
  strict-subset relation between content-token sets, and matching entity spans.
  L2 reuse requires cosine at or above 0.75 and the temporal test only. L3 is keyed
  by exact URL.
* **Temporal eligibility** (`freshcache/risk_model.py`, `scripts/core/mixed_engine.py`):
  a tier serves a stored object only while the estimated staleness probability
  1 − 2^(−age / (m·h)) stays within the tier budget, with class half-lives h from
  `calibrate.py`, multipliers m = (1.5, 1.2, 1.0) and budgets ε = (0.10, 0.20, 0.35).
  Multiplier and budget enter the decision only through k = −ln(1−ε)/(m ln 2), so
  the rule is equivalent to a per-class TTL (the EquivalentTTL control).
* **Replay** (`scripts/core/mixed_engine.py`, `prep2.py`): a seven-day request stream
  (`zipf_uniform`, seed 42) over stored Web snapshots. A fetch at replay time t
  returns the most recent snapshot no later than t; drift is measured on
  body-observable transitions only.
* **Selection discipline** (`docs/release/RUN_ORDER.md`): every selected parameter
  (budgets, stricter gate, verifier threshold, quality-selected baseline) was chosen
  on the 1,877 validation clusters and frozen with a hash before the 4,381 held-out
  test clusters were replayed. The pre-registration files and hashes ship under
  `artifacts/`.

## Reproducing the paper's stages

`docs/release/RUN_ORDER.md` lists the pipeline in dependency order. Stages 1–3
(query construction and Web collection) contact Web services and cannot reproduce
the original Web state. Stage 4 onward (round table, embeddings, calibration,
replays, generation, judging, analysis) is deterministic given the released
metadata, seed 42 and greedy decoding.

Two inputs are not shipped in this repository because of size or licensing:

* **Page text** (`data/snapshots/`): third-party content, not redistributed.
  All change and volatility results are computed from the released content hashes
  and labels, so every replay-level number reproduces without the text. Stages that
  need page text (answer generation, evidence sufficiency judging) can be
  re-executed only against the stored outputs we release.
* **Embedding matrices** (`data/query_embeddings_bgem3.npy`, `data/sim_matrix_bgem3.npy`,
  3.7 GB): regenerate with `experiment.precompute_similarity_matrix` from the
  released questions and paraphrases (BGE-M3, L2-normalised, float32).

To run the core scripts, put the repository root and `scripts/core` on the path:

```bash
export PYTHONPATH=$PWD:$PWD/scripts/core
```

Workstream scripts were sanitised for release: the placeholder `<PROJECT_ROOT>`
stands for the repository root and `<HOME>` for the user's home directory.

## Data availability and licence

`docs/release/DATA_AVAILABILITY.md` states what is redistributed. In short:
question sets, paraphrases, cluster split, replay schedule, per-URL hashes,
counters, HTTP metadata, change labels, our generated answers and model
judgments, prompts, frozen configurations and pre-registrations are released;
page text, raw search responses and model weights are not. Code is released
under the MIT licence (`LICENSE`); `scripts/core/vcache_implementation.py` is a
port of the vCache reference implementation and follows that project's licence.
Cite SCALM (arXiv:2406.00025) and vCache (arXiv:2502.03771) when using the
baseline implementations.

## Citation

See `CITATION.cff`.
