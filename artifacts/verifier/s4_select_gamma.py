#!/usr/bin/env python3
"""s4_select_gamma.py — apply the PRE-REGISTERED rule on validation labels only."""
import json, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
P = json.load(open(OUT / "prereg_gamma.json"))
assert P["preregistered"]
rows = [json.loads(l) for l in open(OUT / "val_labels.jsonl") if l.strip()]
use = [r for r in rows if r["label"] in ("SUFFICIENT", "INSUFFICIENT")
       and r["support_score"] is not None]
pos = [r for r in use if r["label"] == "INSUFFICIENT"]      # should be rejected
neg = [r for r in use if r["label"] == "SUFFICIENT"]        # should be kept
print(f"  prereg sha {P['prereg_sha256'][:24]}...")
print(f"  usable labels {len(use)}  (INSUFFICIENT {len(pos)}, SUFFICIENT {len(neg)})")
print(f"  support score: INSUF mean {np.mean([r['support_score'] for r in pos]):.4f} "
      f"| SUF mean {np.mean([r['support_score'] for r in neg]):.4f}")
best, tab = None, []
for g in [round(x / 100, 2) for x in range(0, 91)]:
    tpr = sum(1 for r in pos if r["support_score"] < g) / len(pos)
    fpr = sum(1 for r in neg if r["support_score"] < g) / len(neg)
    j = tpr - fpr
    tab.append({"gamma": g, "tpr_reject_insufficient": round(tpr, 4),
                "fpr_reject_sufficient": round(fpr, 4), "youden_j": round(j, 4),
                "reject_rate": round(sum(1 for r in use
                                         if r["support_score"] < g) / len(use), 4)})
    if best is None or (j, -g, -fpr) > (best["youden_j"], -best["gamma"],
                                        -best["fpr_reject_sufficient"]):
        best = tab[-1]
print(f"\n  SELECTED gamma = {best['gamma']}  (Youden J {best['youden_j']}, "
      f"TPR {best['tpr_reject_insufficient']}, FPR {best['fpr_reject_sufficient']}, "
      f"validation reject rate {best['reject_rate']})")
print("\n  neighbourhood:")
for r in tab:
    if abs(r["gamma"] - best["gamma"]) <= 0.06:
        m = " <-- selected" if r["gamma"] == best["gamma"] else ""
        print(f"    gamma {r['gamma']:.2f}  J {r['youden_j']:+.4f}  "
              f"TPR {r['tpr_reject_insufficient']:.4f}  FPR "
              f"{r['fpr_reject_sufficient']:.4f}  reject {r['reject_rate']:.4f}{m}")
auc = sum((1 if a["support_score"] > b["support_score"] else
           0.5 if a["support_score"] == b["support_score"] else 0)
          for a in neg for b in pos) / (len(neg) * len(pos))
print(f"\n  discrimination AUC (SUFFICIENT scores higher) = {auc:.4f}")
json.dump({"prereg_sha256": P["prereg_sha256"], "usable_labels": len(use),
           "n_insufficient": len(pos), "n_sufficient": len(neg),
           "selected_gamma": best["gamma"], "selected": best, "auc": round(auc, 4),
           "sweep": tab, "selection_used_test_data": False},
          open(OUT / "gamma_selection.json", "w"), indent=2)
print("  wrote gamma_selection.json")
