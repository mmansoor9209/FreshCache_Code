# Correction to the observability addendum, section 2 (retrieval details)

Date: 2026-09-24. Read-only. Executed with the manifest and query files only; console output in `run_retrieval_correction.log`. Everything below is **[EXECUTED]** unless marked.

## 1. The two questions with five run_00 manifest rows

The manifest has no search or call identifier. Its keys are `run_id, query_id, query, freshness_class, retrieval_timestamp, rank, url, canonical_url, url_hash, title, snippet, domain, snapshot_available, content_hash, changed, etag, last_modified, content_length, snapshot_path`. `retrieval_timestamp` is per row and rows of one search differ by a few seconds, so a "call" can only be inferred from timestamp proximity. I grouped rows whose timestamps are within 120 s.

| query_id | retrieval_timestamp | rank | url_hash | snapshot_available | inferred call |
|---|---|---|---|---|---|
| q_000299 | 2026-05-15T07:13:35.683058+00:00 | 1 | f8c04b002d0425f6ebbe85fa | True | call 1 |
| q_000299 | 2026-05-15T07:13:38.253953+00:00 | 2 | f48deeec5d7be2fa36e342a6 | True | call 1 |
| q_000299 | 2026-05-15T07:27:43.454031+00:00 | 2 | 7d6d1eb83645663b92ac0116 | True | call 2 (14 min later, one row) |
| q_000299 | 2026-06-01T09:30:11.906032+00:00 | 1 | f8c04b002d0425f6ebbe85fa | True | call 3 |
| q_000299 | 2026-06-01T09:30:11.906552+00:00 | 2 | f48deeec5d7be2fa36e342a6 | True | call 3 |
| q_000323 | 2026-05-15T07:14:06.536425+00:00 | 1 | e71142c5381ed8044b300cba | False | call 1 |
| q_000323 | 2026-05-15T07:14:08.257890+00:00 | 2 | 07fed7a3643e9396329ff6c8 | False | call 1 |
| q_000323 | 2026-05-15T07:17:49.947862+00:00 | 1 | 379fcc9ecf7c1816a5eb13b8 | True | call 2 (3.5 min later, one row) |
| q_000323 | 2026-06-01T09:30:22.685671+00:00 | 1 | 07fed7a3643e9396329ff6c8 | False | call 3 |
| q_000323 | 2026-06-01T09:30:24.352255+00:00 | 2 | e71142c5381ed8044b300cba | True | call 3 |

The fifth row is neither a duplicate of another row nor part of the two-row calls: it is a single-row record from a **third** call on the first collection day (one new URL at rank 2 for q_000299, rank 1 for q_000323). Why that extra call happened is not recorded [UNVERIFIED]. Both questions are in the retained set; q_000299 keeps 5 list entries with 4 distinct hashes, q_000323 keeps 2 (the three unavailable rows are dropped).

**Rows per inferred call, all 9,187 run_00 questions:** 10,192 calls have 2 rows, 29 calls have 1 row, none has more than 2, and no call repeats a rank. Under the 120 s grouping the "at most two rows per call" statement holds for every call; it is an inference from timestamps, not from a recorded call ID. The addendum's earlier field `distinct_retrieval_timestamps_per_query_at_run_00` was uninformative (each row has its own timestamp, so it equals the row count) and should be disregarded.

## 2. Which population the "1,032 questions searched twice" refers to

It counted **all 9,187 questions with run_00 rows**, not the 6,258 retained base questions. Corrected counts (120 s grouping):

| population | questions | 1 call | 2 calls | 3 calls |
|---|---|---|---|---|
| all run_00 questions in the manifest | 9,187 | 8,155 | 1,030 | 2 |
| retained base questions (`build_query_records`) | 6,258 | 5,330 | 926 | 2 |

The manifest holds run_00 rows for 9,187 question IDs, the queries file has 8,072, and 6,258 survive the snapshot filter. Also correcting the addendum's date statement: the second call is on 2026-06-01 for every twice-searched question, but the first call falls on 2026-05-15, 05-20 or 05-23 (509 + 398 + 125 among all; 470 + 336 + 122 among retained). Single-call questions were mostly fetched on 2026-06-12.

## 3. The last table column and the list-position arithmetic

Confirmed: the column 4,036 / 1,925 / 220 / 77 / 0 counts **retained base questions by the number of distinct `url_hash` values** in their record list (1, 2, 3, 4, 5). The other two columns count raw list positions.

| quantity | value |
|---|---|
| base records | 6,258 |
| raw list positions | **9,366** |
| distinct positions after hash deduplication | **8,854** |
| repeated positions | **512** |
| affected records | **434** (356 with one repeat, 78 with two) |
| all 31,201 requests, raw / distinct positions | 46,703 / 44,148 |

The 78 double-repeat records are questions whose two calls returned the same two URLs; the 356 single-repeat records share one URL across calls.

## Code excerpts

`experiment.py:503-518` (no cap, no deduplication; sort by rank only) and `:541-556` (paraphrases inherit the base list) are reproduced in `code_excerpts_addendum.md`. The grouping rule used here is the 120 s gap in `run_retrieval_correction.log`'s generating script, reproduced verbatim at the top of that log; no project code implements a call identifier.
