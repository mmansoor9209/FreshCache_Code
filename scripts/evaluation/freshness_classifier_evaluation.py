"""
freshness_classifier_evaluation.py — freshness classification and its end-to-end effect.

Question this answers: FreshCache's scripts/experiments supply the freshness class
as a collection-time annotation. What happens when the class must be
predicted from the query instead?

Design
------
1. A multinomial logistic regression over the precomputed BGE-M3 query
   embeddings (data/query_embeddings_bgem3.npy, row i = record i, verified
   against the similarity matrix). Embeddings rather than surface keywords,
   since topical signal (finance, live sports, civics) is what carries
   freshness information.
2. Grouped 5-fold cross-validation at the query-cluster level: a paraphrase
   is always in the same fold as its base query, so no model ever predicts
   a cluster it trained on. Same leakage discipline as the risk model's
   cluster split. Training uses base queries only; prediction covers every
   record out-of-fold.
3. Metrics at three granularities:
     a. 5-way accuracy and confusion matrix (reported, not the headline).
     b. Decision-relevant binary: stable {TIMELESS, SLOW} vs volatile
        {MEDIUM, FAST, REAL_TIME}. Under the deployed budgets the answer
        tier admits exactly the stable group (stable-class risks below,
        volatile-class risks above eps_L1 = 0.10), so this binary is what the
        cache decision consumes.
     c. The harmful direction alone: volatile predicted stable. The other
        direction (stable predicted volatile) only routes queries to a
        lower tier, which costs savings, never staleness.
4. End-to-end: sim_freshcache at t=24h with every record's class replaced
   by its out-of-fold prediction, against the oracle-label operating point
   (staleness and savings with the true classes). This is the number the system claim needs:
   not "the classifier is accurate" but "the system tolerates its errors."

Writes: outputs/class_predictor/classifier_eval_results.json
Reads only; does not touch experiment_results.json or pareto_data.csv.

Run:
    python -u freshness_classifier_evaluation.py
"""

from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys; sys.path.insert(0, str(REPO_ROOT / "freshcache/method")); import json; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tiered_cache_model as exp

DATA_DIR = (DATA_ROOT)
RESULTS_FILE = OUTPUT_ROOT / "class_predictor/classifier_eval_results.json"
EMB_FILE = DATA_DIR / "query_embeddings_bgem3.npy"

CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]
STABLE = {"TIMELESS", "SLOW"}
N_FOLDS = 5
SEED = 42


def cluster_of(query_id: str) -> str:
    """p_q_XXХ_N -> q_XXX ; q_XXX -> q_XXX"""
    m = re.match(r"^p_(.+)_\d+$", query_id)
    return m.group(1) if m else query_id


def fit_multinomial(X: np.ndarray, y: np.ndarray, n_classes: int,
                    l2: float = 1.0, epochs: int = 300,
                    lr: float = 0.5) -> np.ndarray:
    """
    Multinomial logistic regression by full-batch gradient descent with
    balanced class weights. Returns W of shape (d+1, n_classes).
    Self-contained so the only dependency is numpy.
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
    W = np.zeros((d + 1, n_classes), dtype=np.float64)

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    cw = n / (n_classes * np.clip(counts, 1, None))
    sw = cw[y]
    sw = sw / sw.mean()

    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    for _ in range(epochs):
        Z = Xb @ W
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = Xb.T @ ((P - Y) * sw[:, None]) / n + l2 * W / n
        W -= lr * G
    return W


def predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
    return np.argmax(Xb @ W, axis=1)


def main() -> None:
    print(f"\n{'='*76}")
    print(f"  S7 — Freshness classification and end-to-end effect")
    print(f"{'='*76}\n", flush=True)

    queries     = exp.load_jsonl(exp.QUERIES_FILE)
    manifest    = exp.load_jsonl(exp.MANIFEST_FILE)
    changes     = exp.load_jsonl(exp.CHANGE_FILE)
    paraphrases = exp.load_jsonl(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else []

    records = exp.build_query_records(queries, manifest, paraphrases)
    n = len(records)

    E = np.asarray(np.load(EMB_FILE, mmap_mode="r"), dtype=np.float64)
    assert E.shape[0] == n, f"embedding rows {E.shape[0]} != records {n}"
    # L2-normalise so the linear model works on the same geometry the
    # cache's cosine lookup uses.
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)

    y_true = np.array([CLASSES.index(r["freshness_class"]) for r in records])
    is_base = np.array([not r.get("is_paraphrase") for r in records])
    clusters = [cluster_of(r["query_id"]) for r in records]

    # ── Grouped folds: hash each cluster id into one of N_FOLDS buckets ──
    rng = np.random.RandomState(SEED)
    uniq = sorted(set(clusters))
    perm = rng.permutation(len(uniq))
    fold_of_cluster = {c: int(perm[i]) % N_FOLDS for i, c in enumerate(uniq)}
    fold = np.array([fold_of_cluster[c] for c in clusters])

    print(f"  records={n}  base={int(is_base.sum())}  "
          f"clusters={len(uniq)}  folds={N_FOLDS}", flush=True)

    # ── Out-of-fold predictions ──────────────────────────────────────────
    y_pred = np.full(n, -1, dtype=np.int64)
    for k in range(N_FOLDS):
        tr = (fold != k) & is_base
        te = (fold == k)
        W = fit_multinomial(E[tr], y_true[tr], len(CLASSES))
        y_pred[te] = predict(W, E[te])
        print(f"  fold {k}: trained on {int(tr.sum())} base queries, "
              f"predicted {int(te.sum())} records", flush=True)
    assert (y_pred >= 0).all()

    # ── Metrics (base queries, the unit the taxonomy labels) ─────────────
    bt, bp = y_true[is_base], y_pred[is_base]
    acc5 = float((bt == bp).mean())

    conf = np.zeros((5, 5), dtype=int)
    for t, p in zip(bt, bp):
        conf[t, p] += 1

    stable_idx = {CLASSES.index(c) for c in STABLE}
    t_stab = np.isin(bt, list(stable_idx))
    p_stab = np.isin(bp, list(stable_idx))
    acc_bin = float((t_stab == p_stab).mean())
    n_volatile = int((~t_stab).sum())
    harmful = int(((~t_stab) & p_stab).sum())
    conservative = int((t_stab & (~p_stab)).sum())

    print(f"\n  5-way accuracy (base queries): {acc5*100:.1f}%")
    print(f"  Per-class recall:")
    for i, c in enumerate(CLASSES):
        row = conf[i].sum()
        r = conf[i, i] / row if row else 0.0
        print(f"    {c:<10} {r*100:5.1f}%   (n={row})")
    print(f"\n  Confusion matrix (rows=true, cols=pred, order {CLASSES}):")
    for i, c in enumerate(CLASSES):
        print(f"    {c:<10} {conf[i].tolist()}")
    print(f"\n  Decision-relevant binary (stable vs volatile): {acc_bin*100:.1f}%")
    print(f"  Harmful direction, volatile->stable: {harmful}/{n_volatile} "
          f"({100*harmful/max(n_volatile,1):.1f}%)")
    print(f"  Conservative direction, stable->volatile: {conservative}", flush=True)

    # ── End-to-end under predicted classes ───────────────────────────────
    print(f"\n  End-to-end: sim_freshcache at t=24h with predicted classes...",
          flush=True)
    exp.precompute_similarity_matrix(records)
    stale_map = exp.build_stale_sets(changes)
    stale_24h = stale_map["rerun_24h"]
    sim_age = exp.SIM_AGES["24h"]

    nc = exp.sim_nocache(records, stale_24h, sim_age)

    pred_records = []
    for i, r in enumerate(records):
        r2 = dict(r)
        r2["freshness_class"] = CLASSES[y_pred[i]]
        pred_records.append(r2)

    oracle = exp.sim_freshcache(records, stale_24h, sim_age)
    predicted = exp.sim_freshcache(pred_records, stale_24h, sim_age)

    def summ(m):
        return {
            "stale_pct": round(m["cache_stale_error_rate"] * 100, 2),
            "saved_pct": round(100 * (1 - m["search_calls"] /
                                      max(nc["search_calls"], 1)), 2),
            "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
            "l3_hits": m["l3_hits"],
        }

    o, p = summ(oracle), summ(predicted)
    print(f"\n  {'':<22} {'Stale':>8} {'Saved':>8} {'L1':>8} {'L2':>8}")
    print(f"  {'Annotated classes':<22} {o['stale_pct']:>7.2f}% "
          f"{o['saved_pct']:>7.2f}% {o['l1_hits']:>8,} {o['l2_hits']:>8,}")
    print(f"  {'Predicted classes':<22} {p['stale_pct']:>7.2f}% "
          f"{p['saved_pct']:>7.2f}% {p['l1_hits']:>8,} {p['l2_hits']:>8,}",
          flush=True)

    results = {
        "n_records": n,
        "n_base": int(is_base.sum()),
        "n_clusters": len(uniq),
        "acc_5way_base": acc5,
        "confusion_base": conf.tolist(),
        "classes": CLASSES,
        "acc_binary_stable_vs_volatile": acc_bin,
        "harmful_volatile_to_stable": {"count": harmful, "of": n_volatile},
        "conservative_stable_to_volatile": conservative,
        "end_to_end_24h": {"annotated": o, "predicted": p},
        "per_record_predictions_file": None,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_FILE}")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
