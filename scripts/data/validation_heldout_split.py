#!/usr/bin/env python3
"""
scripts/data/validation_heldout_split.py

Deterministic cluster-disjoint validation/test split for held-out baseline
tuning.

  validation = 30% of clusters, test = 70%, seed 42, stratified by freshness
  class. Every request of a cluster stays in the same split.

Stratification is exact because clusters are freshness-class homogeneous: zero
clusters span more than one class (verified and asserted below).

The split is written ONCE. If split.json already exists the script refuses to
overwrite it and exits, so the frozen assignment can never be regenerated.

Read-only on all existing project files. No web calls.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
import os
import pathlib
import random
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = ((OUTPUT_ROOT / "validation_split"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
from schedule_robustness import support_of  # noqa: E402

SCHEDULE, SEED, VAL_FRAC = "zipf_uniform", 42, 0.30
CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]
OUT = HERE / "split.json"


def main():
    if OUT.exists():
        print(f"  split.json already exists at {OUT}")
        print("  refusing to regenerate a frozen split. Nothing written.")
        return

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    rounds = {}
    with open((DATA_ROOT / "derived/round_table.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]

    # the ORIGINAL complete stream, exactly as the recorded reference replay
    stream = sc.build_stream(records, SCHEDULE, SEED)
    assert DS.check("primary_requests", len(stream)), f"unexpected stream size {len(stream)}"
    S = support_of(stream, rounds)

    by_cluster = defaultdict(list)
    cls_of = {}
    for t, r in stream:
        c = r.get("cluster_id") or r["query_id"]
        by_cluster[c].append((t, r))
        cls_of.setdefault(c, set()).add(r["freshness_class"])

    mixed = [c for c, v in cls_of.items() if len(v) > 1]
    assert not mixed, f"{len(mixed)} clusters span >1 freshness class"
    cls_of = {c: next(iter(v)) for c, v in cls_of.items()}

    rng = random.Random(SEED)
    val = set()
    per_class = {}
    for k in CLASSES:
        cs = sorted([c for c in by_cluster if cls_of[c] == k])
        rng.shuffle(cs)
        n_val = int(round(VAL_FRAC * len(cs)))
        val.update(cs[:n_val])
        per_class[k] = {"clusters_total": len(cs), "clusters_validation": n_val,
                        "clusters_test": len(cs) - n_val}
    test = set(by_cluster) - val

    # ---- assertions ----
    assert val & test == set(), "validation and test cluster sets intersect"
    assert val | test == set(by_cluster), "some cluster is in neither split"
    for c in by_cluster:
        assert (c in val) ^ (c in test), f"cluster {c} not in exactly one split"

    def stats(cs):
        reqs = [x for c in cs for x in by_cluster[c]]
        return {
            "clusters": len(cs), "requests": len(reqs),
            "support": sum(1 for _, r in reqs if r["query_id"] in S),
            "requests_by_class": dict(Counter(r["freshness_class"]
                                              for _, r in reqs)),
            "support_by_class": dict(Counter(
                r["freshness_class"] for _, r in reqs if r["query_id"] in S)),
        }

    v, t = stats(val), stats(test)
    assert v["requests"] + t["requests"] == len(stream)
    assert v["support"] + t["support"] == len(S)

    payload = {
        "schedule": SCHEDULE, "seed": SEED, "validation_fraction": VAL_FRAC,
        "stratified_by": "freshness_class",
        "clusters_are_class_homogeneous": True,
        "split_unit": "cluster_id (fallback query_id)",
        "stream_construction": ("Option 1: the full stream is "
                                "built once with schedule=zipf_uniform, seed=42, "
                                "then filtered by cluster assignment. Original "
                                "timestamps, ages and snapshot rounds are "
                                "preserved."),
        "totals": {"clusters": len(by_cluster), "requests": len(stream),
                   "support": len(S)},
        "validation": v, "test": t, "per_class_clusters": per_class,
        "validation_clusters": sorted(val), "test_clusters": sorted(test),
    }
    json.dump(payload, open(OUT, "w"), indent=2)
    write_summary(payload)

    print(f"  clusters  total {len(by_cluster):,}  validation {v['clusters']:,}"
          f"  test {t['clusters']:,}")
    print(f"  requests  total {len(stream):,}  validation {v['requests']:,}"
          f"  test {t['requests']:,}")
    print(f"  support   total {len(S):,}  validation {v['support']:,}"
          f"  test {t['support']:,}")
    print(f"  assertions passed: disjoint, exhaustive, one-split-per-cluster")
    print(f"  wrote {OUT} and split_summary.md")


def write_summary(p):
    v, t, pc = p["validation"], p["test"], p["per_class_clusters"]
    rows = "\n".join(
        f"| {k} | {pc[k]['clusters_total']:,} | {pc[k]['clusters_validation']:,} | "
        f"{pc[k]['clusters_test']:,} | "
        f"{v['requests_by_class'].get(k, 0):,} | {t['requests_by_class'].get(k, 0):,} | "
        f"{v['support_by_class'].get(k, 0):,} | {t['support_by_class'].get(k, 0):,} |"
        for k in ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"])
    md = f"""# Validation / test split summary

Deterministic, cluster-disjoint, stratified by freshness class.
Validation = {int(p['validation_fraction']*100)}% of clusters, seed {p['seed']}.

Split unit: `{p['split_unit']}`. Every request of a cluster is in the same
split. Clusters are freshness-class homogeneous, so stratification is exact.

**Stream construction.** {p['stream_construction']}

## Totals

| | clusters | requests | in common observable support |
|---|---|---|---|
| validation | {v['clusters']:,} | {v['requests']:,} | {v['support']:,} |
| test | {t['clusters']:,} | {t['requests']:,} | {t['support']:,} |
| **total** | **{p['totals']['clusters']:,}** | **{p['totals']['requests']:,}** | **{p['totals']['support']:,}** |

## Per freshness class

| class | clusters total | clusters val | clusters test | requests val | requests test | support val | support test |
|---|---|---|---|---|---|---|---|
{rows}

## Assertions passed

- validation and test cluster sets are disjoint
- their union is every cluster
- every cluster belongs to exactly one split
- request counts sum to the stream length
- support counts sum to the full common observable support
- zero clusters span more than one freshness class

The split is frozen. `validation_heldout_split.py` refuses to overwrite `split.json`.
"""
    (HERE / "split_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
