#!/usr/bin/env python3
"""
scripts/experiments/calibration_and_measurement/freshness_class_prediction.py

Regenerates the inference-time freshness-classifier predictions using the
EXISTING classifier code (freshness_classifier_evaluation.py) unchanged, and materialises them
per record so the mixed-age replay can consume them.

Why regenerate rather than load: freshness_classifier_evaluation.py wrote only the aggregate
metrics (outputs/class_predictor/classifier_eval_results.json) and recorded
"per_record_predictions_file": null, so the per-record vector was never saved.
The model is fully deterministic -- fixed SEED=42 cluster permutation,
full-batch gradient descent, no sampling -- so re-running the same functions on
the same inputs reproduces the same predictions. This script PROVES that by
asserting the regenerated 5-way accuracy, confusion matrix, binary accuracy and
harmful/conservative counts match outputs/class_predictor/classifier_eval_results.json exactly. If
any of them differ the script aborts and nothing downstream runs.

Nothing is modified: freshness_classifier_evaluation.py is imported, not edited, and
outputs/class_predictor/classifier_eval_results.json is read only.

Writes: outputs/class_predictor/predicted_classes.jsonl  (one row per record, in
        build_query_records order: idx, query_id, cluster_id, is_paraphrase,
        true_class, pred_class)

Run: python3 -u scripts/experiments/calibration_and_measurement/freshness_class_prediction.py
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()

os.environ.setdefault("OMP_NUM_THREADS", "8")
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
HERE = ((OUTPUT_ROOT / "class_predictor"))
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method"))

import numpy as np                      # noqa: E402
import tiered_cache_model as exp                # noqa: E402
import freshness_classifier_evaluation as clf           # noqa: E402

OUT = HERE/"predicted_classes.jsonl"
REF = (OUTPUT_ROOT / "class_predictor/classifier_eval_results.json")


def main():
    ref = json.load(open(REF, encoding="utf-8"))
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    n = len(records)
    assert n == ref["n_records"], f"record count {n} != {ref['n_records']}"

    E = np.asarray(np.load(clf.EMB_FILE, mmap_mode="r"), dtype=np.float64)
    assert E.shape[0] == n
    E = E/np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)

    y_true = np.array([clf.CLASSES.index(r["freshness_class"]) for r in records])
    is_base = np.array([not r.get("is_paraphrase") for r in records])
    clusters = [clf.cluster_of(r["query_id"]) for r in records]

    rng = np.random.RandomState(clf.SEED)
    uniq = sorted(set(clusters))
    perm = rng.permutation(len(uniq))
    fold_of = {c: int(perm[i]) % clf.N_FOLDS for i, c in enumerate(uniq)}
    fold = np.array([fold_of[c] for c in clusters])

    y_pred = np.full(n, -1, dtype=np.int64)
    for k in range(clf.N_FOLDS):
        tr = (fold != k) & is_base
        te = (fold == k)
        W = clf.fit_multinomial(E[tr], y_true[tr], len(clf.CLASSES))
        y_pred[te] = clf.predict(W, E[te])
        print(f"  fold {k}: trained {int(tr.sum()):,} base -> predicted "
              f"{int(te.sum()):,} records", flush=True)
    assert (y_pred >= 0).all()

    bt, bp = y_true[is_base], y_pred[is_base]
    acc5 = float((bt == bp).mean())
    conf = np.zeros((5, 5), dtype=int)
    for a, b in zip(bt, bp):
        conf[a, b] += 1
    stable = {clf.CLASSES.index(c) for c in clf.STABLE}
    ts, ps = np.isin(bt, list(stable)), np.isin(bp, list(stable))
    accb = float((ts == ps).mean())
    harmful = int(((~ts) & ps).sum())
    conserv = int((ts & (~ps)).sum())

    # ---- exact reproduction check against the recorded artifact ----
    checks = [
        ("acc_5way_base", acc5, ref["acc_5way_base"]),
        ("acc_binary", accb, ref["acc_binary_stable_vs_volatile"]),
        ("harmful", harmful, ref["harmful_volatile_to_stable"]["count"]),
        ("harmful_of", int((~ts).sum()), ref["harmful_volatile_to_stable"]["of"]),
        ("conservative", conserv, ref["conservative_stable_to_volatile"]),
        ("n_base", int(is_base.sum()), ref["n_base"]),
        ("confusion", conf.tolist(), ref["confusion_base"]),
    ]
    bad = [(k, a, b) for k, a, b in checks
           if (a != b if isinstance(a, list) else abs(a-b) > 1e-12)]
    print(f"\n  reproduction check against {REF}:")
    for k, a, b in checks:
        ok = "OK " if (k, a, b) not in bad else "MISMATCH"
        s = "" if isinstance(a, list) else f"  regenerated={a}  recorded={b}"
        print(f"    [{ok}] {k}{s}")
    if bad:
        raise SystemExit(f"\n  ABORT: {len(bad)} mismatch(es); predictions are "
                         f"not the recorded classifier's. Nothing written.")

    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            f.write(json.dumps({
                "idx": i, "query_id": r["query_id"],
                "cluster_id": r.get("cluster_id") or r["query_id"],
                "is_paraphrase": bool(r.get("is_paraphrase")),
                "true_class": clf.CLASSES[y_true[i]],
                "pred_class": clf.CLASSES[y_pred[i]]})+"\n")

    from collections import Counter
    print(f"\n  class distribution over all {n:,} records")
    print(f"    {'class':<12}{'dataset':>10}{'predicted':>12}")
    ct, cp = Counter(y_true.tolist()), Counter(y_pred.tolist())
    for i, c in enumerate(clf.CLASSES):
        print(f"    {c:<12}{ct[i]:>10,}{cp[i]:>12,}")
    print(f"    {'agree':<12}{int((y_true == y_pred).sum()):>10,}"
          f"  ({100*(y_true == y_pred).mean():.1f}% of all records)")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
