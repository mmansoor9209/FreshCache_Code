#!/usr/bin/env python3
"""Build the third-party adjudication form from two completed annotator files.
Refuses to run on unfilled forms."""
from __future__ import annotations
import argparse, csv, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
OK = {"SAME", "DIFFERENT", "UNCERTAIN"}


def load(p):
    rs = list(csv.DictReader(open(p, encoding="utf-8")))
    filled = [r for r in rs if r["human_label"].strip()]
    if not filled:
        sys.exit(f"ERROR: {p} contains no human labels. Nothing is adjudicated.")
    bad = {r["human_label"] for r in filled} - OK
    if bad:
        sys.exit(f"ERROR: {p} has invalid labels {bad}; allowed {sorted(OK)}")
    print(f"  {p}: {len(filled)}/{len(rs)} labelled")
    return {r["pair_id"]: r for r in rs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    ids = sorted(set(A) & set(B))
    dis = [i for i in ids
           if A[i]["human_label"].strip() and B[i]["human_label"].strip()
           and A[i]["human_label"] != B[i]["human_label"]]
    out = HERE / "human_adjudication.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "incoming_query",
                                           "cached_query",
                                           "annotation_question",
                                           "human_label", "notes"])
        w.writeheader()
        for i in dis:
            w.writerow({"pair_id": i,
                        "incoming_query": A[i]["incoming_query"],
                        "cached_query": A[i]["cached_query"],
                        "annotation_question": A[i]["annotation_question"],
                        "human_label": "", "notes": ""})
    print(f"  {len(dis)} disagreements of {len(ids)} common pairs "
          f"({100*len(dis)/max(1,len(ids)):.1f}%)")
    print(f"  wrote {out.name} -- give this to a THIRD person. The two "
          f"annotators' labels are NOT shown in it.")


if __name__ == "__main__":
    main()
