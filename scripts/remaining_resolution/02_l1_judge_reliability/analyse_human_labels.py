#!/usr/bin/env python3
"""
TASK 2D -- analysis of REAL human labels. Refuses to run before they exist.

Computes human-human agreement, the adjudicated human mismatch rate, and the
LLM jury's precision/recall/F1 against the human consensus, with intervals.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, pathlib, sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
LP = HERE.parents[1] / "l1_precision_gate"
JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
OK = {"SAME", "DIFFERENT", "UNCERTAIN"}


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 2),
            round(100 * min(1.0, (c + m) / d), 2))


def kappa(pairs):
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    labs = {x for x, _ in pairs} | {y for _, y in pairs}
    pe = sum((sum(1 for x, _ in pairs if x == l) / n) *
             (sum(1 for _, y in pairs if y == l) / n) for l in labs)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def load(p, need=True):
    if not p:
        return {}
    rs = list(csv.DictReader(open(p, encoding="utf-8")))
    filled = [r for r in rs if r["human_label"].strip()]
    if need and not filled:
        sys.exit(f"ERROR: {p} contains no human labels. NOTHING is computed. "
                 f"The human-verification component stays PENDING.")
    bad = {r["human_label"] for r in filled} - OK
    if bad:
        sys.exit(f"ERROR: {p} has invalid labels {bad}")
    return {r["pair_id"]: r["human_label"].strip() for r in filled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--adjudicated", default=None)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    ADJ = load(a.adjudicated, need=False) if a.adjudicated else {}
    ids = sorted(set(A) & set(B))
    if len(ids) < 2:
        sys.exit("ERROR: fewer than two commonly labelled pairs.")
    print(f"  commonly labelled pairs: {len(ids)}")

    pairs = [(A[i], B[i]) for i in ids]
    agree = sum(1 for x, y in pairs if x == y)
    print(f"\n  human-human agreement {agree}/{len(ids)} = "
          f"{100*agree/len(ids):.2f}%  CI{wilson(agree, len(ids))}")
    print(f"  Cohen kappa (3 labels) {kappa(pairs)}")
    sub = [(x, y) for x, y in pairs if x != "UNCERTAIN" and y != "UNCERTAIN"]
    if sub:
        ag2 = sum(1 for x, y in sub if x == y)
        print(f"  excluding UNCERTAIN: {ag2}/{len(sub)} = "
              f"{100*ag2/len(sub):.2f}%   kappa {kappa(sub)}")

    cons, unres = {}, []
    for i in ids:
        if A[i] == B[i]:
            cons[i] = A[i]
        elif i in ADJ:
            cons[i] = ADJ[i]
        else:
            unres.append(i)
    print(f"\n  consensus resolved {len(cons)}/{len(ids)}; unresolved "
          f"{len(unres)} (reported, never imputed)")
    det = {i: v for i, v in cons.items() if v in ("SAME", "DIFFERENT")}
    nd = sum(1 for v in det.values() if v == "DIFFERENT")
    nu = sum(1 for v in cons.values() if v == "UNCERTAIN")
    print(f"  HUMAN adjudicated mismatch {nd}/{len(det)} = "
          f"{100*nd/max(1,len(det)):.4f}%  CI{wilson(nd, len(det))}   "
          f"(UNCERTAIN {nu}, excluded from this denominator)")

    eq = {}
    for j in JUDGES:
        for l in open(LP / f"heldout_equiv_{j}.jsonl", encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                eq.setdefault(d["pair_id"], {})[j] = d["label"]

    def jury(pid, js):
        v = [eq.get(pid, {}).get(j) for j in js]
        v = [x for x in v if x in ("SAME", "DIFFERENT")]
        if len(v) < 2:
            return None
        c = Counter(v)
        return "TIE" if c["SAME"] == c["DIFFERENT"] else c.most_common(1)[0][0]

    out = {"n_common": len(ids), "human_agreement_pct": round(100*agree/len(ids), 4),
           "cohen_kappa": kappa(pairs), "consensus_resolved": len(cons),
           "unresolved": len(unres),
           "human_mismatch": {"k": nd, "n": len(det),
                              "pct": round(100*nd/max(1, len(det)), 4),
                              "ci": wilson(nd, len(det))},
           "uncertain": nu, "jury_vs_human": {}}
    print(f"\n  LLM jury vs human consensus (positive class = DIFFERENT)")
    print(f"    {'jury':<34}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
          f"{'prec':>8}{'rec':>8}{'F1':>8}")
    for tag, js, ties in (("four-model, ties->DIFFERENT", JUDGES, "DIFFERENT"),
                          ("four-model, ties excluded", JUDGES, "drop"),
                          ("three-model (no 3B) SENSITIVITY",
                           [j for j in JUDGES if j != "llama3b"], "DIFFERENT")):
        tp = fp = fn = tn = 0
        for i, hv in det.items():
            v = jury(i, js)
            if v is None:
                continue
            if v == "TIE":
                if ties == "drop":
                    continue
                v = ties
            if v == "DIFFERENT" and hv == "DIFFERENT":
                tp += 1
            elif v == "DIFFERENT":
                fp += 1
            elif hv == "DIFFERENT":
                fn += 1
            else:
                tn += 1
        pr = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        f1 = (2 * pr * rc / (pr + rc)) if pr and rc else None
        out["jury_vs_human"][tag] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(pr, 4) if pr is not None else None,
            "precision_ci": wilson(tp, tp + fp) if tp + fp else None,
            "recall": round(rc, 4) if rc is not None else None,
            "recall_ci": wilson(tp, tp + fn) if tp + fn else None,
            "f1": round(f1, 4) if f1 is not None else None}
        print(f"    {tag:<34}{tp:>5}{fp:>5}{fn:>5}{tn:>5}"
              + (f"{pr:>8.3f}" if pr is not None else f"{'-':>8}")
              + (f"{rc:>8.3f}" if rc is not None else f"{'-':>8}")
              + (f"{f1:>8.3f}" if f1 is not None else f"{'-':>8}"))

    ex = [i for i, hv in det.items()
          if jury(i, JUDGES) not in (None, hv)][:20]
    print(f"\n  disagreement examples (jury vs human), first {len(ex)}")
    rows = {r["pair_id"]: r for r in csv.DictReader(
        open(a.a, encoding="utf-8"))}
    for i in ex:
        r = rows.get(i, {})
        print(f"    human {det[i]:<9} jury {str(jury(i, JUDGES)):<9} "
              f"{r.get('incoming_query','')[:48]!r} vs "
              f"{r.get('cached_query','')[:48]!r}")
    out["disagreement_examples"] = [
        {"pair_id": i, "human": det[i], "jury": jury(i, JUDGES),
         "incoming": rows.get(i, {}).get("incoming_query"),
         "cached": rows.get(i, {}).get("cached_query")} for i in ex]
    json.dump(out, open(HERE / "human_vs_jury.json", "w"), indent=2)
    print(f"\n  wrote human_vs_jury.json")
    print(f"  These numbers are valid ONLY if the labels were produced by real "
          f"human annotators.")


if __name__ == "__main__":
    main()
