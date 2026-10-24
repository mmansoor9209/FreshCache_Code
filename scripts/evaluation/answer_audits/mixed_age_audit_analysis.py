#!/usr/bin/env python3
"""
scripts/evaluation/answer_audits/mixed_age_audit_analysis.py

Stage 4: outcomes, WAI, reverse transitions, McNemar, and the report.

Resolution rules are the previous mixed-age audit's, unchanged: abstention from
the answer text, then CORRECT only if both judges say CORRECT.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import hashlib
import json
import math
import pathlib
from collections import Counter

HERE = ((OUTPUT_ROOT / "mixed_age_audit"))

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
    sample = jl(HERE / "mixed_age_sample.jsonl")
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "answers2.jsonl")}
    j1 = {tuple(r["sig"]): r["label"] for r in jl(HERE / "judge2_llama8b.jsonl")}
    j2 = {tuple(r["sig"]): r["label"] for r in jl(HERE / "judge2_qwen7b.jsonl")}
    man = json.load(open(HERE / "sample_manifest.json", encoding="utf-8"))

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
    for d in sample:
        fk = ("fresh", d["fresh_ctx_sha"], h(d["query"]))
        o = {"qid": d["query_id"], "fc": d["fc"], "in_S": d["in_S"],
             "on_tier": d["on_tier"], "off_tier": d["off_tier"],
             "on_mode": d["on_mode"], "off_mode": d["off_mode"],
             "on_age": d["on_age"], "off_age": d["off_age"]}
        got = {}
        for arm, key in (("fresh", fk), ("on", tuple(d["on_key"])),
                         ("off", tuple(d["off_key"]))):
            got[arm] = outcome(d["query"], d["gold"], ans.get(tuple(key)))
        if any(v is None for v in got.values()):
            dropped += 1
            continue
        o.update(got)
        rows.append(o)
    n = len(rows)

    def block(arm):
        c = Counter(r[arm] for r in rows)
        f = Counter(r["fresh"] for r in rows)
        wai = sum(1 for r in rows if r[arm] == "WRONG" and r["fresh"] == "CORRECT")
        rev = sum(1 for r in rows if r[arm] == "CORRECT" and r["fresh"] == "WRONG")
        lo, hi = wilson(wai, n)
        return {"N": n, "correct": c["CORRECT"], "wrong": c["WRONG"],
                "abstain": c["ABSTAIN"],
                "correct_pct": round(100 * c["CORRECT"] / n, 4) if n else None,
                "fresh_correct": f["CORRECT"], "fresh_wrong": f["WRONG"],
                "fresh_abstain": f["ABSTAIN"],
                "fresh_correct_pct": round(100 * f["CORRECT"] / n, 4) if n else None,
                "wai": wai, "wai_pct": round(100 * wai / n, 4) if n else None,
                "wai_ci95_pct": [round(100 * lo, 4), round(100 * hi, 4)],
                "reverse": rev,
                "reverse_pct": round(100 * rev / n, 4) if n else None}

    on, off = block("on"), block("off")
    b = sum(1 for r in rows if (r["off"] == "WRONG" and r["fresh"] == "CORRECT")
            and not (r["on"] == "WRONG" and r["fresh"] == "CORRECT"))
    c_ = sum(1 for r in rows if (r["on"] == "WRONG" and r["fresh"] == "CORRECT")
             and not (r["off"] == "WRONG" and r["fresh"] == "CORRECT"))
    p = mcnemar(b, c_)

    # by path
    per_path = {}
    for arm in ("on", "off"):
        d = {}
        for tier in ("L1", "L2", "miss"):
            sub = [r for r in rows if r[f"{arm}_tier"] == tier]
            if not sub:
                continue
            cc = Counter(r[arm] for r in sub)
            wai = sum(1 for r in sub
                      if r[arm] == "WRONG" and r["fresh"] == "CORRECT")
            rev = sum(1 for r in sub
                      if r[arm] == "CORRECT" and r["fresh"] == "WRONG")
            d[tier] = {"N": len(sub), "correct": cc["CORRECT"],
                       "wrong": cc["WRONG"], "abstain": cc["ABSTAIN"],
                       "wai": wai, "reverse": rev,
                       "correct_pct": round(100 * cc["CORRECT"] / len(sub), 4)}
        per_path[arm] = d

    keys = set(j1) & set(j2)
    agree = sum(1 for k in keys if j1[k] == j2[k])

    res = {"design": "policy-neutral sampling; real execution path per tier",
           "N": n, "dropped_incomplete": dropped,
           "frame_size": man["frame_size"], "seed": man["seed"],
           "n_strata": man["n_strata"],
           "composition": man["composition"],
           "unique_generation_tasks": man["unique_generation_tasks"],
           "gate_on": on, "gate_off": off,
           "paired_wai_difference_pp": round(off["wai_pct"] - on["wai_pct"], 4),
           "mcnemar": {"discordant_off_only": b, "discordant_on_only": c_,
                       "exact_two_sided_p": p},
           "by_path": per_path,
           "judge_agreement_pct": round(100 * agree / len(keys), 2) if keys else None,
           "judge_items": len(keys)}
    json.dump(res, open(HERE / "recomputed_results.json", "w"), indent=2)

    with open(HERE / "path_breakdown.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "path", "N", "correct", "wrong", "abstain",
                    "correct_pct", "wai", "reverse"])
        for arm in ("on", "off"):
            for tier, v in per_path[arm].items():
                w.writerow([arm, tier, v["N"], v["correct"], v["wrong"],
                            v["abstain"], v["correct_pct"], v["wai"],
                            v["reverse"]])
    emit(res)
    return res


def emit(r):
    on, off, mc = r["gate_on"], r["gate_off"], r["mcnemar"]
    print()
    print("REPORTED MIXED-AGE FULL-POLICY AUDIT")
    print()
    print(f"N: {r['N']}   dropped incomplete: {r['dropped_incomplete']}")
    print(f"policy-neutral frame: {r['frame_size']:,}   strata {r['n_strata']}"
          f"   seed {r['seed']}")
    print(f"composition on : {r['composition']['on']}")
    print(f"composition off: {r['composition']['off']}")
    print()
    for nm, blk in (("GATE ON", on), ("GATE OFF", off)):
        print(nm)
        print(f"  policy correct/wrong/abstain: {blk['correct']}/{blk['wrong']}"
              f"/{blk['abstain']}  ({blk['correct_pct']:.2f}% correct)")
        print(f"  fresh  correct/wrong/abstain: {blk['fresh_correct']}"
              f"/{blk['fresh_wrong']}/{blk['fresh_abstain']}  "
              f"({blk['fresh_correct_pct']:.2f}% correct)")
        print(f"  WAI: {blk['wai']} ({blk['wai_pct']:.2f}%, CI "
              f"[{blk['wai_ci95_pct'][0]:.2f}, {blk['wai_ci95_pct'][1]:.2f}])"
              f"   reverse: {blk['reverse']} ({blk['reverse_pct']:.2f}%)")
    print()
    print(f"PAIRED WAI DIFFERENCE (off - on): "
          f"{r['paired_wai_difference_pp']:+.2f} pp")
    print(f"McNemar discordant {mc['discordant_off_only']} vs "
          f"{mc['discordant_on_only']}, exact two-sided p = "
          f"{mc['exact_two_sided_p']:.4f}")
    print()
    print("BY PATH")
    for arm in ("on", "off"):
        for tier, v in r["by_path"][arm].items():
            print(f"  {arm:<4}{tier:<6}N {v['N']:>4}  correct {v['correct']:>3}"
                  f"  wrong {v['wrong']:>3}  abstain {v['abstain']:>2}"
                  f"  WAI {v['wai']:>2}  reverse {v['reverse']:>2}")
    print()
    print(f"judge agreement: {r['judge_agreement_pct']}%  "
          f"over {r['judge_items']} distinct items")


if __name__ == "__main__":
    main()
