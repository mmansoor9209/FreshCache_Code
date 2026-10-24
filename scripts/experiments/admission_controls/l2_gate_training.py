"""Learned L2 compatibility gate, stage 2 (validation only): train the gate and freeze it. Labels: two-judge SUFFV on each validation L2
candidate's evidence (agree -> label; disagreement / unclear excluded from training). Cluster-grouped 5-fold CV (seed 42): logistic regression vs
HistGradientBoosting by out-of-fold ROC-AUC; the single-feature scripts/baselines (support_max = the frozen gamma verifier's score; qq_sim = FreshCache's
admission similarity) are scored on the same folds. Threshold tau = Youden J on the selected model's out-of-fold scores. Writes gate_frozen.pkl /
gate_frozen.json. Held-out data is not read."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, datetime, hashlib, json, os, pathlib, pickle; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
O = ((OUTPUT_ROOT / "learned_l2_gate")); PRE = json.load(open(O / "prereg.json"))
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
R = jl(O / "work/val_candidates.jsonl")
def merged(j):
    out = {}
    for f in [O / "gpu/out" / f"suffv_{j}.jsonl"] + sorted((O / "gpu/out").glob(f"suffv_{j}_shard*.jsonl")):
        for r in jl(f): out.setdefault((r["query_id"], r["arm"]), r["label"])
    return out
L8, L7 = merged("llama8b"), merged("qwen7b")
FEATS = [f for f in R[0] if f in ("qq_sim", "qq_jaccard", "entity_match", "answer_type_match", "strict_subset", "support_max", "support_mean", "support_min", "n_pages", "n_pages_text", "evidence_chars",
                                  "qtok_coverage_all", "qtok_coverage_max_page", "num_coverage", "cap_coverage", "entry_age_frac_of_window") or f.startswith("class_")]
lab, cnt = [], collections.Counter()
for r in R:
    k = (r["query_id"], r["evidence_sha"][:16]); a, b = L8.get(k), L7.get(k)
    if a is None or b is None: cnt["unlabelled"] += 1; continue
    if a != b: cnt["judge_disagreement"] += 1; continue
    if a not in ("SUFFICIENT", "INSUFFICIENT"): cnt[f"other_{a}"] += 1; continue
    cnt[a] += 1; lab.append((r, 1 if a == "SUFFICIENT" else 0))
X = np.array([[r[f] for f in FEATS] for r, _ in lab], float); y = np.array([v for _, v in lab]); g = np.array([r["cluster_id"] for r, _ in lab])
folds = list(GroupKFold(n_splits=5).split(X, y, g))
def oof(make):
    p = np.zeros(len(y))
    for tr, te in folds: m = make(); m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
MODELS = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
          "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42)}
P = {k: oof(m) for k, m in MODELS.items()}; AUC = {k: round(roc_auc_score(y, p), 4) for k, p in P.items()}
for single in ("support_max", "qq_sim", "qtok_coverage_all"): AUC[f"single_feature:{single}"] = round(roc_auc_score(y, X[:, FEATS.index(single)]), 4)
best = max(MODELS, key=lambda k: AUC[k]); fpr, tpr, thr = roc_curve(y, P[best]); j = np.argmax(tpr - fpr); tau = float(thr[j])
final = MODELS[best](); final.fit(X, y)
acc = P[best] >= tau
OUT = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "prereg_sha256": PRE["sha256"], "n_candidates": len(R), "label_counts": dict(cnt), "n_train": int(len(y)), "positive_rate": round(float(y.mean()), 4),
       "features": FEATS, "oof_auc": AUC, "selected_model": best, "tau_youden": round(tau, 6), "oof_at_tau": {"accept_rate": round(float(acc.mean()), 4), "tpr_accept_sufficient": round(float(tpr[j]), 4),
       "fpr_accept_insufficient": round(float(fpr[j]), 4), "sufficient_rate_among_accepted": round(float(y[acc].mean()), 4), "sufficient_rate_among_rejected": round(float(y[~acc].mean()), 4)},
       "heldout_consulted": False}
if hasattr(final, "named_steps"):
    coef = final.named_steps["logisticregression"].coef_[0]; OUT["logreg_standardized_coefficients"] = {f: round(float(c), 4) for f, c in sorted(zip(FEATS, coef), key=lambda z: -abs(z[1]))}
pickle.dump({"model": final, "features": FEATS, "tau": tau}, open(O / "gate_frozen.pkl", "wb"))
OUT["model_sha256"] = hashlib.sha256(open(O / "gate_frozen.pkl", "rb").read()).hexdigest()
json.dump(OUT, open(O / "gate_frozen.json", "w"), indent=1); print(json.dumps(OUT, indent=1))
