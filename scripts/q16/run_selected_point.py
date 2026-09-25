"""Q-16, done properly: replay SemanticTTL at its quality-selected operating
point (theta=0.60, kappa=1/2) using the PUBLISHED generator.

Now that the published baseline's identity is established
(replay_baseline_hits.py, uniform simulated age vs per-class FIXED_TTL), the
matched run is a two-line change to that same path: raise the cosine
threshold to 0.60 and halve the TTLs.

Gate: the same path at the published setting must return 22,561 hits.
Read-only; writes only under Reframe_ResearchPaper/artifacts/q16/.
"""
import json, os, pathlib, sys
import numpy as np

SRC = pathlib.Path("<PROJECT_ROOT>")
OUT = pathlib.Path(__file__).resolve().parent.parent.parent/"artifacts/q16"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC)); os.chdir(SRC)
import replay_baseline_hits as rb

records = rb.build_query_records(rb.load_jsonl(rb.QUERIES_FILE),
                                 rb.load_jsonl(rb.MANIFEST_FILE),
                                 rb.load_jsonl(rb.PARA_FILE))
stale = rb.build_stale_sets(rb.load_jsonl(rb.CHANGE_FILE))["rerun_24h"]
rb._SIM_MATRIX = np.load(str(rb.SIM_MATRIX), mmap_mode="r")
rb._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
ORIG_TTL = dict(rb.FIXED_TTL); ORIG_THETA = rb.L1_SIM_THRESHOLD

def run(theta, kappa, tag):
    rb.L1_SIM_THRESHOLD = theta
    rb.FIXED_TTL = {k: v*kappa for k, v in ORIG_TTL.items()}
    tmp = OUT/f"_hits_{tag}.jsonl"
    try:
        res = rb.replay_semantic_ttl(records, stale, rb.SIM_AGES["24h"], tmp)
        rows = [json.loads(l) for l in open(tmp, encoding="utf-8")]
    finally:
        rb.L1_SIM_THRESHOLD = ORIG_THETA; rb.FIXED_TTL = dict(ORIG_TTL)
    print(f"  [{tag}] theta={theta} kappa={kappa}  hits={res['l1_hits']:,}")
    return res, rows, tmp

print("[GATE] published operating point must return 22,561")
g, grows, gtmp = run(0.40, 1.0, "gate")
gtmp.unlink()
if g["l1_hits"] != 22561:
    print(f"  GATE FAILED ({g['l1_hits']:,}); not proceeding."); sys.exit(1)
print("  GATE PASSED\n")

print("[TARGET] quality-selected point")
sel, rows, tmp = run(0.60, 0.5, "selected")
tmp.rename(OUT/"semanticttl_selected_hits_t24h.jsonl")

pub = [json.loads(l) for l in open(SRC/"data/semanticttl_hits_t24h.jsonl", encoding="utf-8")]
key = lambda r: (r["query_id"], r["matched_query"])
pubset = {key(r) for r in pub}
shared = sum(1 for r in rows if key(r) in pubset)
print(f"\n  selected-point hits : {len(rows):,}")
print(f"  published hits      : {len(pub):,}")
print(f"  of the selected-point hits, {shared:,} ({100*shared/max(len(rows),1):.1f}%) "
      f"are also realized in the published run")

rec = {"gate": {"expected": 22561, "observed": g["l1_hits"], "passed": True},
       "published": {"theta": 0.40, "kappa": 1.0, "hits": len(pub)},
       "selected":  {"theta": 0.60, "kappa": 0.5, "hits": len(rows),
                     "stale_error_rate": round(sel["cache_stale_error_rate"], 4)},
       "overlap_with_published_hits": shared,
       "generator": "replay_baseline_hits.py :: replay_semantic_ttl (published path, two constants changed)"}
(OUT/"semanticttl_selected_point.json").write_text(json.dumps(rec, indent=1))
print("\nwrote artifacts/q16/semanticttl_selected_point.json")
