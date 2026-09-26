# Retrieval accounting: two remaining points

Date: 2026-09-24. Read-only; no replay, collection or generation. Output of the generating script is in `run_retrieval_reconciliation.log`. All figures **[EXECUTED]** from `data/url_manifest.jsonl`, the queries and paraphrase files, `v9/version_timeline.jsonl` and the `data/snapshots/run_00/` file list. "Inferred retrieval group" below means rows of one question within a 120 s timestamp gap; the manifest records no call ID.

## 1. q_000299: the sentence was wrong, the records were right

Actual `build_query_records` list for q_000299 (rank, hash), in the order the function returns it (sorted by rank only, `experiment.py:517-518`):

```
(1, f8c04b002d0425f6ebbe85fa)   A
(1, f8c04b002d0425f6ebbe85fa)   A
(2, f48deeec5d7be2fa36e342a6)   B
(2, 7d6d1eb83645663b92ac0116)   C
(2, f48deeec5d7be2fa36e342a6)   B
```

5 positions, **3 distinct hashes**, 2 repeats. The correction's sentence "5 list entries with 4 distinct hashes" was wrong; the displayed records (A, B, C, A, B) were right. q_000299 is one of the 78 double-repeat records. The related sentence "the 78 double-repeat records are questions whose two calls returned the same two URLs" is also too strong: q_000299 has three inferred groups, and a double repeat only requires two hashes to recur. Nothing was deduplicated; the list above is what the replay consumes.

Histograms recomputed from the actual 6,258 base records (unchanged from the addendum):

| list size | raw positions (records) | distinct hashes (records) |
|---|---|---|
| 1 | 3,906 | 4,036 |
| 2 | 1,883 | 1,925 |
| 3 | 183 | 220 |
| 4 | 285 | 77 |
| 5 | 1 | 0 |
| total positions | **9,366** | **8,854** |

Repeated positions 512; affected records 434 (356 with one repeat, 78 with two). All confirmed.

## 2. Question-ID reconciliation and the URL panel

Sets: **M** = question IDs with at least one run_00 manifest row; **Mav** = those with at least one run_00 row flagged `snapshot_available`; **Q** = IDs in the queries file (8,072 rows, 8,072 unique IDs); **B** = retained base records.

| set | unique IDs |
|---|---|
| M (manifest, run_00) | 9,187 |
| Mav | 7,250 |
| Q (queries file) | 8,072 |
| M ∩ Q | 8,005 |
| M \ Q (manifest only) | 1,182 |
| Q \ M (queries file only, no run_00 row) | 67 |
| Q \ Mav | 1,814 (67 with no row + 1,747 with rows but no available snapshot) |
| B = Q ∩ Mav | **6,258** (equality verified) |

**Join and filter rule** (`experiment.py:503-528`): iterate the queries file in order; for each ID take every run_00 manifest row with `snapshot_available`; drop the question if that list is empty. So retained = queries-file IDs that have at least one available run_00 row, and nothing from the manifest-only side enters. All 6,258 paraphrase base IDs are retained IDs.

**Text check.** No manifest question ID carries more than one distinct query text, and for all 8,005 shared IDs the queries-file text equals the manifest text (0 mismatches).

**Panel.** The 8,635-URL panel equals the set of `data/snapshots/run_00/` files exactly, and every panel URL appears in at least one run_00 manifest row. Of the 8,635: 8,124 belong to a queries-file question (and all 8,124 to a retained base record); **511 belong only to the 1,182 manifest questions outside the queries file**. Those 511 URLs are fetched in every round and count in the 8,635 / 4,382 URL-level figures and in the 17,351-row class table, but no replay request references them. Manifest run_00 rows: 20,413 total, 10,782 available; every available row has a run_00 file; 44 rows flagged unavailable nevertheless have a file for their hash (retrieved for another question). Why 1,182 questions have manifest rows but no queries-file entry is not recorded and is not inferred here.

## Consequences for the earlier documents

- `Observability_addendum_correction.md` §1: replace "4 distinct hashes" with "3 distinct hashes (A, A, B, C, B)"; qualify the sentence about the 78 double-repeat records as above.
- `Observability_verification.md` §2 and the addendum: the URL-level denominators include 511 URLs tied only to non-request questions; the request-level and support figures do not. This does not change any number but should be stated wherever the 8,635 panel and the 31,201 requests are put side by side.

## Code excerpt

`experiment.py:503-528`, verbatim in `code_excerpts_addendum.md` (rows 503-518) plus:

```python
 524      for q in queries:
 525          qid  = q["query_id"]
 526          urls = query_urls.get(qid, [])
 527          if not urls:
 528              continue
```
