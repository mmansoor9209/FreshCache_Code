"""NOT USED FOR ANY REPORTED NUMBER.

This was an attempt to replay SemanticTTL at its quality-selected point
(theta=0.60, kappa=1/2) directly. Its reproduction gate FAILS: this
reconstruction of the published permissive run yields 29,378 realized L1 hits
against the published 22,561, so the operating point it reconstructs is not
the one Table 'mismatch' reports. We kept the script and this notice rather
than delete the failed attempt, and the Q-16 number in the paper comes
instead from theta_restricted_mismatch.py, which re-analyses the already
judged sample and whose own gate does pass.
"""

import json, copy, os, pathlib, sys, time
import os
import numpy as np

ROOT = pathlib.Path("<PROJECT_ROOT>")
os.chdir(ROOT)   # experiment.py resolves data paths relative to the repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"v16_exp12"))
sys.path.insert(0, str(ROOT/"v14_baselines"))
sys.path.insert(0, str(ROOT/"v13_corrected"))
sys.path.insert(0, str(ROOT/"v9"))
import experiment as exp
import engine_all as ea
import schedules as sc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mixed_engine_q16 as me

OUT = pathlib.Path(__file__).resolve().parent.parent.parent/"artifacts/q16"
OUT.mkdir(parents=True, exist_ok=True)

print("building records and stream ...")
queries  = exp.load_jsonl(exp.QUERIES_FILE)
manifest = exp.load_jsonl(exp.MANIFEST_FILE)
paras    = exp.load_jsonl(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else []
records  = exp.build_query_records(queries, manifest, paras)
exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
exp._SIM_MATRIX   = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
ea._rich_feats(); ea.set_cluster_base(records)
rounds = {}
with open(ROOT/"v13_corrected"/"corrected_round_table.jsonl", encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
stream = sc.build_stream(records, "zipf_uniform", 42)
assert len(stream) == 31201, len(stream)
print(f"  stream {len(stream):,} requests")

ORIG = {"FIXED_TTL": copy.deepcopy(exp.FIXED_TTL), "L1T": exp.L1_SIM_THRESHOLD}

UNIFORM_24H = 86400.0

def run(theta, kappa, tag, uniform24=True):
    if theta is not None: exp.L1_SIM_THRESHOLD = theta
    # Table 'mismatch' is the fixed-24h operating point: one uniform 24 h TTL
    # for every class, not the per-class FIXED_TTL ladder.
    if uniform24:
        for k in exp.FIXED_TTL:
            exp.FIXED_TTL[k] = 0.0 if k == "REAL_TIME" else UNIFORM_24H
    if kappa is not None:
        for k in exp.FIXED_TTL: exp.FIXED_TTL[k] = exp.FIXED_TTL[k] * kappa
    try:
        t0 = time.time()
        m, per = me.replay(stream, rounds, "SemanticTTL")
        pairs = [dict(p) for p in me.PAIRS]
        print(f"  [{tag}] l1_hits={m['l1_hits']:,}  pairs={len(pairs):,}  ({time.time()-t0:.0f}s)")
    finally:
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG["FIXED_TTL"])
        exp.L1_SIM_THRESHOLD = ORIG["L1T"]
    return m, pairs

print("\n[GATE] SemanticTTL at the published permissive setting")
m0, p0 = run(None, None, "published")
PUB = 22561
if m0["l1_hits"] != PUB or len(p0) != PUB:
    print(f"  GATE FAILED: expected {PUB:,} realized hits, got {m0['l1_hits']:,}")
    print("  Not proceeding: a selected-point number from a path that cannot")
    print("  reproduce the published one would not be trustworthy.")
    sys.exit(1)
print(f"  GATE PASSED: reproduces the published {PUB:,} realized hits exactly")

print("\n[TARGET] SemanticTTL at the quality-selected point (theta=0.60, kappa=1/2)")
m1, p1 = run(0.60, 0.5, "selected")
json.dump({"policy": "SemanticTTL", "theta": 0.60, "kappa": 0.5,
           "l1_hits": m1["l1_hits"], "n_pairs": len(p1),
           "gate": {"published_hits_expected": PUB, "reproduced": m0["l1_hits"]}},
          open(OUT/"selected_point_summary.json", "w"), indent=1)
with open(OUT/"semanticttl_selected_hits.jsonl", "w", encoding="utf-8") as f:
    for p in p1: f.write(json.dumps(p, ensure_ascii=False)+"\n")
print(f"\nwrote {len(p1):,} realized pairs -> artifacts/q16/semanticttl_selected_hits.jsonl")
