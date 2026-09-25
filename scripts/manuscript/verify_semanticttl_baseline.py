"""NEW-357-01: establish the identity of the published SemanticTTL baseline.

Table 3 reports 22,561 realized hits for SemanticTTL at the fixed-24h
operating point. An earlier attempt of ours to reproduce that through the
mixed-age replay engine returned 30,169 / 29,378 and failed its gate. This
script finds out why, by running the ACTUAL published generator.

The two are different experiments, not two attempts at one experiment:

  * replay_baseline_hits.py (the published path) sweeps the records ONCE in
    a fixed order with NO arrival times, and assigns every cached entry the
    SAME simulated age - 24 h - which is then compared against the entry's
    per-class FIXED_TTL. "t=24h" names that simulated age.
  * mixed_engine.replay (what we mistakenly used) drives a timestamped
    arrival stream where each entry carries its own real age.

Read-only: writes only under Reframe_ResearchPaper/artifacts/q16/.
"""
import json, pathlib, sys
import numpy as np

SRC = pathlib.Path("<PROJECT_ROOT>")
OUT = pathlib.Path(__file__).resolve().parent.parent.parent/"artifacts/q16"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))
import os
os.chdir(SRC)                      # the generator uses paths relative to the repo root
import replay_baseline_hits as rb

print("loading records exactly as the published generator does ...")
records = rb.build_query_records(rb.load_jsonl(rb.QUERIES_FILE),
                                 rb.load_jsonl(rb.MANIFEST_FILE),
                                 rb.load_jsonl(rb.PARA_FILE))
stale_map = rb.build_stale_sets(rb.load_jsonl(rb.CHANGE_FILE))
rb._SIM_MATRIX = np.load(str(rb.SIM_MATRIX), mmap_mode="r")
rb._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
assert rb._SIM_MATRIX.shape[0] == len(records)
print(f"  records: {len(records):,}")
print(f"  SIM_AGES: {rb.SIM_AGES}   (a simulated AGE, not a TTL)")
print(f"  FIXED_TTL: {rb.FIXED_TTL}")
print(f"  L1_SIM_THRESHOLD: {rb.L1_SIM_THRESHOLD}")

tmp = OUT/"_semanticttl_hits_t24h_recomputed.jsonl"
res = rb.replay_semantic_ttl(records, stale_map["rerun_24h"],
                             rb.SIM_AGES["24h"], tmp)
print(f"\n  recomputed l1_hits: {res['l1_hits']:,}")

PUB = 22561
frozen = SRC/"data/semanticttl_hits_t24h.jsonl"
n_frozen = sum(1 for _ in open(frozen, encoding="utf-8"))
print(f"  frozen artifact rows: {n_frozen:,}")
print(f"  Table 3 published   : {PUB:,}")

# row-for-row identity, not just the count
new = [json.loads(l) for l in open(tmp, encoding="utf-8")]
old = [json.loads(l) for l in open(frozen, encoding="utf-8")]
identical = (len(new) == len(old)) and all(a == b for a, b in zip(new, old))
print(f"  row-for-row identical to the frozen artifact: {identical}")

ok = res["l1_hits"] == PUB == n_frozen and identical
rec = {
 "finding": ("The published SemanticTTL baseline is reproduced exactly. The earlier "
             "30,169 / 29,378 figures came from a DIFFERENT experiment, not from a "
             "failure of the published one."),
 "published_generator": "replay_baseline_hits.py :: replay_semantic_ttl",
 "operating_point": {
   "name": "fixed-24h",
   "meaning": ("every cached entry is given the SAME simulated age of 24 h, which is "
               "compared against that entry's per-class FIXED_TTL; it is NOT a 24 h TTL"),
   "sim_age_seconds": rb.SIM_AGES["24h"],
   "FIXED_TTL": rb.FIXED_TTL,
   "cosine_threshold": rb.L1_SIM_THRESHOLD,
   "selection": "nearest neighbour over the whole cache (highest cosine), no equivalence check",
   "order": "records swept once, sorted by (cluster_id, base-before-paraphrase, query_id); no arrival times",
 },
 "why_the_earlier_attempt_differed": (
   "mixed_engine.replay drives a timestamped arrival stream in which every entry carries "
   "its own real age, and we varied FIXED_TTL rather than holding a uniform simulated age. "
   "That is a different estimand, so it could not have reproduced this number."),
 "recomputed_l1_hits": res["l1_hits"],
 "frozen_artifact_rows": n_frozen,
 "published_in_table3": PUB,
 "row_for_row_identical": identical,
 "reproduced": bool(ok),
}
(OUT/"semanticttl_baseline_identity.json").write_text(json.dumps(rec, indent=1))
tmp.unlink()
print("\n" + ("REPRODUCED EXACTLY" if ok else "NOT REPRODUCED"))
sys.exit(0 if ok else 1)
