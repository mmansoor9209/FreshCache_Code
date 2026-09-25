#!/usr/bin/env python3
"""
Section 7D -- blinded human-annotation package.

Claude Code CANNOT create human judgments, and none are created here. This
script only assembles the cases a human must look at, in a blinded sheet, and
provides the script that recomputes every affected number once real annotators
have filled it in.

Included, by construction:
  * every WAI case (cached arm WRONG while the fresh reference is CORRECT),
  * every correctness-discordant case (any arm disagreeing with the fresh
    reference in either direction),
  * every L2 / evidence judge-disagreement case available in the repository,
  * a reproducible random sample (seed 42) of cases where the judges AGREE,
    so agreement itself can be audited rather than assumed.

Policy identity is withheld: the sheet shows an anonymised case id, the
question, the reference answer, the serving evidence and the candidate answer.
The arm each row came from is stored separately in case_key.csv, which
annotators must not be given.
"""
from __future__ import annotations
import csv, hashlib, json, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
AUD = BASE / "06_stronger_generator"
SEED = 42
ARMS = ["fresh", "fc", "sttl", "l1only", "exactttl"]
JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    t = (t or "").strip()
    if t == "INSUFFICIENT_CONTEXT":
        return True
    tl = t.lower()
    return any(p in tl for p in ABSTAIN_PAT)


def main():
    sample = json.load(open(AUD / "sample.json", encoding="utf-8"))
    reqs = {d["query_id"]: d for d in sample["requests"]}
    rows, key = [], []
    seen = set()

    for gen in ("llama3b", "llama8b_forced", "llama8b_abstain"):
        af = AUD / f"answers_{gen}.jsonl"
        if not af.exists():
            continue
        ans = {tuple(r["key"]): r["answer"] for r in jl(af)}
        jud = {j: {tuple(r["sig"]): r["label"]
                   for r in jl(AUD / f"judge_{gen}__{j}.jsonl")}
               for j in JUDGES}
        jud = {j: v for j, v in jud.items() if v}
        if len(jud) < 2:
            continue

        def lab(d, arm):
            a = ans.get(tuple(d[f"{arm}_key"]))
            if a is None:
                return None, None, {}
            if is_abstain(a):
                return "ABSTAIN", a, {}
            sig = (h(d["query"]), h(d["gold"]), h(a))
            ls = {j: jud[j].get(sig) for j in jud}
            vals = [v for v in ls.values() if v and v != "UNPARSED"]
            if not vals:
                return "UNJUDGED", a, ls
            if len(set(vals)) > 1:
                return "JUDGE_DISAGREEMENT", a, ls
            return vals[0], a, ls

        agree_pool = []
        for qid, d in reqs.items():
            fl, fa, fls = lab(d, "fresh")
            for arm in ARMS:
                L, a, ls = lab(d, arm)
                if L is None:
                    continue
                reason = None
                if L == "JUDGE_DISAGREEMENT":
                    reason = "judge_disagreement"
                elif arm != "fresh" and fl == "CORRECT" and L == "INCORRECT":
                    reason = "WAI (fresh CORRECT, cached WRONG)"
                elif arm != "fresh" and fl and L and fl != L and \
                        {fl, L} <= {"CORRECT", "INCORRECT"}:
                    reason = "correctness_discordant"
                sig = (gen, qid, arm)
                if reason:
                    if sig in seen:
                        continue
                    seen.add(sig)
                    rows.append({"case_id": h(f"{gen}|{qid}|{arm}"),
                                 "question": d["query"],
                                 "reference_answer": d["gold"],
                                 "candidate_answer": a,
                                 "serving_evidence_sha": d.get(f"{arm}_ctx_sha"),
                                 "request_timestamp_s": d["t"],
                                 "cached_answer_timestamp_s":
                                     d.get(f"{arm}_stored_answer_t"),
                                 "age_seconds": d.get(f"{arm}_age"),
                                 "freshness_class": d["fc"],
                                 "inclusion_reason": reason,
                                 "paraphrase_fidelity_label": "",
                                 "temporal_validity_label": "",
                                 "evidence_sufficiency_label": "",
                                 "answer_correctness_label": "",
                                 "annotator_id": "", "notes": ""})
                    key.append({"case_id": h(f"{gen}|{qid}|{arm}"),
                                "generator": gen, "arm": arm, "query_id": qid,
                                "model_label": L,
                                **{f"judge_{j}": ls.get(j) for j in JUDGES}})
                elif L in ("CORRECT", "INCORRECT"):
                    agree_pool.append((gen, qid, arm, a, L, ls))

        rng = random.Random(SEED)
        rng.shuffle(agree_pool)
        for gen_, qid, arm, a, L, ls in agree_pool[:120]:
            d = reqs[qid]
            cid = h(f"{gen_}|{qid}|{arm}")
            if (gen_, qid, arm) in seen:
                continue
            seen.add((gen_, qid, arm))
            rows.append({"case_id": cid, "question": d["query"],
                         "reference_answer": d["gold"], "candidate_answer": a,
                         "serving_evidence_sha": d.get(f"{arm}_ctx_sha"),
                         "request_timestamp_s": d["t"],
                         "cached_answer_timestamp_s": d.get(f"{arm}_stored_answer_t"),
                         "age_seconds": d.get(f"{arm}_age"),
                         "freshness_class": d["fc"],
                         "inclusion_reason": f"random agreeing control "
                                             f"(seed {SEED})",
                         "paraphrase_fidelity_label": "",
                         "temporal_validity_label": "",
                         "evidence_sufficiency_label": "",
                         "answer_correctness_label": "",
                         "annotator_id": "", "notes": ""})
            key.append({"case_id": cid, "generator": gen_, "arm": arm,
                        "query_id": qid, "model_label": L,
                        **{f"judge_{j}": ls.get(j) for j in JUDGES}})

    rng = random.Random(SEED)
    rng.shuffle(rows)
    with open(HERE / "human_annotation_sheet.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(HERE / "case_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader(); w.writerows(key)
    from collections import Counter
    c = Counter(r["inclusion_reason"] for r in rows)
    print(f"  human_annotation_sheet.csv: {len(rows):,} blinded cases")
    for k, v in c.most_common():
        print(f"    {k:<44}{v:>6,}")
    print("  case_key.csv holds the arm/generator/model labels and must NOT be "
          "given to annotators")
    print("  ALL FOUR LABEL COLUMNS ARE EMPTY. No model filled them and none "
          "may. Two independent annotators plus adjudication are required.")


if __name__ == "__main__":
    main()
