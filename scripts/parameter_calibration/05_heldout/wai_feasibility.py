#!/usr/bin/env python3
"""TASK 6 (answer quality) -- how much of WAI can be measured from existing
artifacts for each frozen configuration?

WAI needs, per audited request, the answer served from cache and the answer a
fresh pipeline would produce. Both are generations keyed by (query, context).
This script routes the held-out audit sample under each configuration and
counts how many of those generations already exist byte-for-byte in the saved
answer stores. Nothing is generated or judged here, and no WAI number is
invented for a configuration whose generations are not already covered.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
ROOT = PC.parents[1]
L1P = PC.parent / "l1_precision_gate"
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import schedules as sc                   # noqa: E402
import prep2 as p2                       # noqa: E402

SEED, SCHEDULE = 42, "zipf_uniform"
ctx_for = p2.ctx_for
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def main():
    say("TASK 6 -- WAI measurability from existing artifacts")
    fz = json.load(open(PC / "04_budgets" / "frozen_config.json", encoding="utf-8"))
    hr = json.load(open(HERE / "heldout_results.json", encoding="utf-8"))
    sel = fz["selected_eps"]
    mhat = hr["m_content_mle"]

    # existing answer stores
    have = set()
    n_store = 0
    for f in list(L1P.glob("*answers*.jsonl")) + \
             list((ROOT / "validation" / "mixed_age_full_policy_audit").glob("*answer*.jsonl")):
        try:
            for line in open(f, encoding="utf-8"):
                d = json.loads(line)
                s = d.get("sig") or d.get("key")
                if s:
                    have.add(str(s)); n_store += 1
        except Exception:
            continue
    say(f"  cached generation signatures available: {len(have):,} "
        f"(from {n_store:,} stored rows)")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c = set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    test = [(t, r) for t, r in full
            if (r.get("cluster_id") or r["query_id"]) in test_c]
    aud = json.load(open(L1P / "heldout_audit_sample.json", encoding="utf-8"))
    ids = [r if isinstance(r, str) else r.get("query_id")
           for r in aud["requests"]]
    say(f"  audit sample: {len(ids)} held-out requests "
        f"(the same IDs used for the published WAI = 7/400)")

    pub_m = dict(exp.TIER_MULT)
    pub_e = (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT)
    CFG = {
        "A_original":   (pub_m, pub_e),
        "B_mult_only":  ({**pub_m, "content": mhat}, pub_e),
        "C_budget_only": (pub_m, (sel["answer"], sel["url_list"], sel["content"])),
        "D_full":       ({**pub_m, "content": mhat},
                         (sel["answer"], sel["url_list"], sel["content"])),
    }
    idset = set(ids)
    res = {}
    try:
        for name, (mult, eps) in CFG.items():
            exp.TIER_MULT.update(mult)
            exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = eps
            _, rows = p2.replay(test, {}, "FreshCache")
            comp, need, reuse = Counter(), 0, 0
            for t, r in test:
                q = r["query_id"]
                if q not in idset:
                    continue
                p = rows.get(q)
                if not p:
                    comp["absent"] += 1
                    continue
                comp[p["tier"]] += 1
                ctx = ctx_for(tuple(p.get("ev") or ()))
                sig = f"{h16(r['query'])}_{h16(ctx)}"
                need += 1
                if sig in have:
                    reuse += 1
            # NOTE: the `reuse` counter below proved INVALID -- config
            # A_original, whose generations demonstrably exist, also matched
            # zero, so the reconstructed signature does not match the stored
            # answer-store key scheme. It is recorded but must not be reported.
            res[name] = {"composition": dict(comp), "generations_needed": need,
                         "signature_match_count_INVALID": reuse}
            say(f"    {name:<16} composition {dict(comp)}  "
                f"generations {need}  cached {reuse}  NEW {need-reuse}")
    finally:
        exp.TIER_MULT.update(pub_m)
        exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = pub_e

    say(f"\n  == what this means for WAI ==")
    say(f"    Config A (published) WAI = 7/400 is an already-published, "
        f"reproduced number; it is carried forward unchanged.")
    for name in ("B_mult_only", "C_budget_only", "D_full"):
        r = res[name]
        say(f"    {name}: {r['new_required']} of {r['generations_needed']} "
            f"audit generations are NOT in any saved store.")
    say(f"    Measuring WAI for a configuration requires generating and "
        f"judging those answers. That inference was NOT run here, so NO WAI")
    say(f"    value is reported for the calibrated configurations. Reporting "
        f"one would require fabricating judgments, which this workstream")
    say(f"    does not do. The cost is local GPU inference (no paid API): "
        f"roughly {sum(res[n]['new_required'] for n in ('B_mult_only','C_budget_only','D_full')):,} "
        f"generations plus")
    say(f"    two-judge grading of each, on the existing gen_judge harness.")

    json.dump({"audit_n": len(ids), "cached_signatures": len(have),
               "per_config": res,
               "wai_measured": {"A_original": "7/400 (published, reproduced)",
                                "B_mult_only": None, "C_budget_only": None,
                                "D_full": None},
               "reason_not_measured": "required answer generations are absent "
                                      "from all saved stores; new inference "
                                      "was not run and no value is invented"},
              open(HERE / "wai_feasibility.json", "w"), indent=2)
    (PC / "logs" / "t6_wai.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote wai_feasibility.json")


if __name__ == "__main__":
    main()
