#!/usr/bin/env python3
"""
Recompute every human-dependent number once REAL annotators have filled in
human_annotation_sheet.csv.

Refuses to run on an unfilled sheet. Refuses to run on a single annotator.
Never invents, imputes or model-fills a missing label.

  python recompute_after_human_labels.py --sheets human_annotation_sheet__*.csv
"""
from __future__ import annotations
import argparse, csv, glob, json, math, pathlib, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
LABELS = ["paraphrase_fidelity_label", "temporal_validity_label",
          "evidence_sufficiency_label", "answer_correctness_label"]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 2),
            round(100 * min(1.0, (c + m) / d), 2))


def cohen(a, b):
    pr = [(x, y) for x, y in zip(a, b) if x and y]
    if not pr:
        return None
    n = len(pr)
    po = sum(1 for x, y in pr if x == y) / n
    labs = {x for x, _ in pr} | {y for _, y in pr}
    pe = sum((sum(1 for x, _ in pr if x == l) / n) *
             (sum(1 for _, y in pr if y == l) / n) for l in labs)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--adjudicated", default=None,
                    help="optional adjudicator sheet resolving disagreements")
    a = ap.parse_args()
    paths = [p for g in a.sheets for p in glob.glob(g)]
    if len(paths) < 2:
        sys.exit("ERROR: at least two independent annotator sheets are "
                 "required. Human verification cannot be established from one "
                 "annotator, and never from a model.")
    sheets = {}
    for p in paths:
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        filled = sum(1 for r in rows if any(r.get(l, "").strip() for l in LABELS))
        if filled == 0:
            sys.exit(f"ERROR: {p} contains no human labels. Nothing is "
                     f"recomputed and the human-verification criticism remains "
                     f"UNRESOLVED.")
        print(f"  {p}: {len(rows):,} rows, {filled:,} with at least one label")
        sheets[p] = {r["case_id"]: r for r in rows}

    ids = sorted(set.intersection(*[set(s) for s in sheets.values()]))
    print(f"  common cases across annotators: {len(ids):,}")
    print("\n  inter-annotator agreement (Cohen's kappa)")
    names = list(sheets)
    agree = {}
    for lab in LABELS:
        ka = cohen([sheets[names[0]][i].get(lab, "") for i in ids],
                   [sheets[names[1]][i].get(lab, "") for i in ids])
        n_ag = sum(1 for i in ids
                   if sheets[names[0]][i].get(lab, "")
                   and sheets[names[0]][i].get(lab) == sheets[names[1]][i].get(lab))
        agree[lab] = {"kappa": ka, "n_agree": n_ag, "n": len(ids)}
        print(f"    {lab:<32} kappa = {ka}   agree {n_ag}/{len(ids)}")

    adj = {}
    if a.adjudicated:
        adj = {r["case_id"]: r for r in
               csv.DictReader(open(a.adjudicated, encoding="utf-8"))}
        print(f"  adjudicator sheet: {len(adj):,} rows")

    def final(i, lab):
        v = [sheets[n][i].get(lab, "").strip() for n in names]
        if v[0] and v[0] == v[1]:
            return v[0]
        return (adj.get(i, {}).get(lab, "").strip() or None)

    key = {}
    kp = HERE / "case_key.csv"
    if kp.exists():
        key = {r["case_id"]: r for r in csv.DictReader(open(kp, encoding="utf-8"))}

    out = {"n_cases": len(ids), "annotator_agreement": agree,
           "unresolved_after_adjudication": 0, "by_arm": {}}
    per_arm = defaultdict(lambda: Counter())
    unres = 0
    for i in ids:
        c = final(i, "answer_correctness_label")
        if c is None:
            unres += 1
            continue
        arm = key.get(i, {}).get("arm", "unknown")
        per_arm[arm][c] += 1
    out["unresolved_after_adjudication"] = unres
    print(f"\n  cases still unresolved after adjudication: {unres} "
          f"(reported, never imputed)")
    print(f"\n  human-labelled correctness by arm")
    for arm, c in sorted(per_arm.items()):
        det = c["CORRECT"] + c["INCORRECT"]
        acc = 100 * c["CORRECT"] / det if det else None
        out["by_arm"][arm] = {**dict(c), "determinate": det,
                              "accuracy_pct": round(acc, 4) if acc else None,
                              "accuracy_ci": wilson(c["CORRECT"], det) if det else None}
        print(f"    {arm:<14}{dict(c)}  acc "
              f"{f'{acc:.2f}%' if acc is not None else '-'}")
    json.dump(out, open(HERE / "human_recomputed.json", "w"), indent=2)
    print("\n  wrote human_recomputed.json")
    print("  NOTE: these numbers are valid only if the labels were produced by "
          "real human annotators. No model may fill this sheet.")


if __name__ == "__main__":
    main()
