#!/usr/bin/env python3
"""
scripts/evaluation/answer_audits/semanticttl_audit_analysis.py

Analysis stage of the SemanticTTL answer audit (after
scripts/evaluation/answer_audits/semanticttl_audit_sample.py and
scripts/evaluation/answer_audits/semanticttl_audit_generation_judging.py): paired
outcomes for FreshCache vs SemanticTTL(kappa=1/8) against one shared fresh
reference, on the same requests as the mixed-age answer audit
(outputs/mixed_age_audit/mixed_age_sample.jsonl).

Resolution rules are the same as in the mixed-age audit: abstention is detected
from the answer text first, then an answer is CORRECT only if both judges say
CORRECT. Results are written to outputs/semanticttl_audit/.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import hashlib
import json
import math
import pathlib
import sys
from collections import Counter

HERE = ((OUTPUT_ROOT / "semanticttl_audit"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
CORR = (OUTPUT_ROOT / "mixed_age_audit")

ABSTAIN_PAT = ("i don't know", "i dont know", "cannot determine",
               "not enough information", "insufficient", "no information",
               "unclear from the context")


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    t = (t or "").strip().lower()
    return any(p in t for p in ABSTAIN_PAT)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def main():
    sample = jl(HERE / "sttl_mixed_age_sample.jsonl")
    ans = {tuple(r["key"]): r["answer"] for r in jl(CORR / "answers2.jsonl")}
    ans.update({tuple(r["key"]): r["answer"]
                for r in jl(HERE / "sttl_answers.jsonl")})
    j1 = {tuple(r["sig"]): r["label"] for r in jl(CORR / "judge2_llama8b.jsonl")}
    j1.update({tuple(r["sig"]): r["label"]
               for r in jl(HERE / "sttl_judge_llama8b.jsonl")})
    j2 = {tuple(r["sig"]): r["label"] for r in jl(CORR / "judge2_qwen7b.jsonl")}
    j2.update({tuple(r["sig"]): r["label"]
               for r in jl(HERE / "sttl_judge_qwen7b.jsonl")})
    man = json.load(open(HERE / "sttl_manifest.json", encoding="utf-8"))

    def outcome(q, gold, a):
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h(q), h(gold), h(a))
        l1, l2 = j1.get(sig), j2.get(sig)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    rows, dropped = [], 0
    fresh_keys = {}
    for d in sample:
        fk = ("fresh", d["fresh_ctx_sha"], h(d["query"]))
        fresh_keys[d["query_id"]] = fk
        got = {}
        for arm, key in (("fresh", fk), ("fc", tuple(d["fc_key"])),
                         ("sttl", tuple(d["sttl_key"]))):
            got[arm] = outcome(d["query"], d["gold"], ans.get(tuple(key)))
        if any(v is None for v in got.values()):
            dropped += 1
            continue
        rows.append({"qid": d["query_id"], "fc_tier": d["fc_tier"],
                     "sttl_tier": d["sttl_tier"], **got})
    n = len(rows)

    def block(arm):
        c = Counter(r[arm] for r in rows)
        f = Counter(r["fresh"] for r in rows)
        wai = sum(1 for r in rows if r[arm] == "WRONG" and r["fresh"] == "CORRECT")
        rev = sum(1 for r in rows if r[arm] == "CORRECT" and r["fresh"] == "WRONG")
        lo, hi = wilson(wai, n)
        return {"N": n, "correct": c["CORRECT"], "wrong": c["WRONG"],
                "abstain": c["ABSTAIN"],
                "correct_pct": round(100 * c["CORRECT"] / n, 4),
                "fresh_correct": f["CORRECT"], "fresh_wrong": f["WRONG"],
                "fresh_abstain": f["ABSTAIN"],
                "fresh_correct_pct": round(100 * f["CORRECT"] / n, 4),
                "wai": wai, "wai_pct": round(100 * wai / n, 4),
                "wai_ci95_pct": [round(100 * lo, 4), round(100 * hi, 4)],
                "reverse": rev, "reverse_pct": round(100 * rev / n, 4)}

    fc, st = block("fc"), block("sttl")
    only_st = [r["qid"] for r in rows
               if (r["sttl"] == "WRONG" and r["fresh"] == "CORRECT")
               and not (r["fc"] == "WRONG" and r["fresh"] == "CORRECT")]
    only_fc = [r["qid"] for r in rows
               if (r["fc"] == "WRONG" and r["fresh"] == "CORRECT")
               and not (r["sttl"] == "WRONG" and r["fresh"] == "CORRECT")]
    p = mcnemar(len(only_st), len(only_fc))

    paths = {}
    for arm, tk in (("fc", "fc_tier"), ("sttl", "sttl_tier")):
        d = {}
        for tier in ("L1", "L2", "miss"):
            sub = [r for r in rows if r[tk] == tier]
            if not sub:
                continue
            cc = Counter(r[arm] for r in sub)
            d[tier] = {"N": len(sub), "correct": cc["CORRECT"],
                       "wrong": cc["WRONG"], "abstain": cc["ABSTAIN"],
                       "correct_pct": round(100 * cc["CORRECT"] / len(sub), 4),
                       "wai": sum(1 for r in sub if r[arm] == "WRONG"
                                  and r["fresh"] == "CORRECT"),
                       "reverse": sum(1 for r in sub if r[arm] == "CORRECT"
                                      and r["fresh"] == "WRONG")}
        paths[arm] = d

    keys = set(j1) & set(j2)
    agree = sum(1 for k in keys if j1[k] == j2[k])

    # integrity confirmations
    corr_ids = [json.loads(l)["query_id"] for l in
                open(CORR / "mixed_age_sample.jsonl", encoding="utf-8") if l.strip()]
    same_ids = ([d["query_id"] for d in sample] == corr_ids)
    shared_fresh = all(
        ans.get(tuple(fresh_keys[d["query_id"]])) is not None for d in sample)

    res = {"kappa": man["kappa"], "N": n, "dropped_incomplete": dropped,
           "same_ids_as_reported_audit": same_ids,
           "shared_fresh_reference_identical": shared_fresh,
           "freshcache": fc, "semanticttl_k8": st,
           "wai_difference_pp": round(st["wai_pct"] - fc["wai_pct"], 4),
           "paired": {"wai_only_semanticttl": len(only_st),
                      "wai_only_freshcache": len(only_fc),
                      "mcnemar_exact_two_sided_p": p,
                      "wai_only_semanticttl_ids": only_st,
                      "wai_only_freshcache_ids": only_fc},
           "path_composition": {"freshcache": dict(
               Counter(r["fc_tier"] for r in rows)),
               "semanticttl_k8": dict(Counter(r["sttl_tier"] for r in rows))},
           "by_path": paths,
           "judge_agreement_pct": round(100 * agree / len(keys), 2),
           "judge_items": len(keys),
           "new_generations": man["new_generations_required"],
           "web_calls": 0}
    json.dump(res, open(HERE / "sttl_vs_freshcache_results.json", "w"), indent=2)

    with open(HERE / "sttl_vs_freshcache_per_request.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "fc_tier", "sttl_tier", "freshcache",
                    "semanticttl_k8", "fresh", "fc_wai", "sttl_wai"])
        for r in rows:
            w.writerow([r["qid"], r["fc_tier"], r["sttl_tier"], r["fc"],
                        r["sttl"], r["fresh"],
                        int(r["fc"] == "WRONG" and r["fresh"] == "CORRECT"),
                        int(r["sttl"] == "WRONG" and r["fresh"] == "CORRECT")])
    emit(res)


def emit(r):
    fc, st = r["freshcache"], r["semanticttl_k8"]
    print()
    print("PAIRED AUDIT — FreshCache vs SemanticTTL (kappa = 1/8)")
    print(f"N: {r['N']}   dropped: {r['dropped_incomplete']}   "
          f"same 400 IDs: {r['same_ids_as_reported_audit']}   "
          f"shared fresh: {r['shared_fresh_reference_identical']}")
    print(f"path composition  FreshCache {r['path_composition']['freshcache']}"
          f"   SemanticTTL {r['path_composition']['semanticttl_k8']}")
    for nm, b in (("FreshCache", fc), ("SemanticTTL k=1/8", st)):
        print()
        print(nm)
        print(f"  policy correct/wrong/abstain: {b['correct']}/{b['wrong']}"
              f"/{b['abstain']}  ({b['correct_pct']:.2f}%)")
        print(f"  fresh  correct/wrong/abstain: {b['fresh_correct']}"
              f"/{b['fresh_wrong']}/{b['fresh_abstain']}  "
              f"({b['fresh_correct_pct']:.2f}%)")
        print(f"  WAI {b['wai']} ({b['wai_pct']:.2f}%, CI "
              f"[{b['wai_ci95_pct'][0]:.2f}, {b['wai_ci95_pct'][1]:.2f}])   "
              f"reverse {b['reverse']} ({b['reverse_pct']:.2f}%)")
    pr = r["paired"]
    print()
    print(f"WAI difference (SemanticTTL - FreshCache): "
          f"{r['wai_difference_pp']:+.2f} pp")
    print(f"  WAI only SemanticTTL: {pr['wai_only_semanticttl']}   "
          f"WAI only FreshCache: {pr['wai_only_freshcache']}   "
          f"exact McNemar p = {pr['mcnemar_exact_two_sided_p']:.4f}")
    print()
    print("BY PATH")
    for arm in ("fc", "sttl"):
        for tier, v in r["by_path"][arm].items():
            print(f"  {arm:<5}{tier:<6}N {v['N']:>4}  correct {v['correct']:>3}"
                  f"  wrong {v['wrong']:>3}  abstain {v['abstain']:>2}"
                  f"  WAI {v['wai']:>2}  reverse {v['reverse']:>2}")
    print()
    print(f"judge agreement {r['judge_agreement_pct']}% over {r['judge_items']} "
          f"items   new generations {r['new_generations']}   "
          f"web calls {r['web_calls']}")


if __name__ == "__main__":
    main()
