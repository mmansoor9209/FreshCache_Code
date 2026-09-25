#!/usr/bin/env python3
"""TASK 1/2 -- build the 400-request answer audit for configurations A-D.

The earlier signature check was wrong: it invented a key
f"{h(query)}_{h(ctx)}" while the answer stores are keyed by the TUPLE
(role, h(context), h(question)) with role in {stored, fresh, gen}, exactly as
run_heldout.py builds it. This script uses the real scheme and VERIFIES it by
requiring that configuration A -- the published FreshCache -- reproduces the
existing 400-request audit: identical request IDs in order, identical tier
composition, and every one of its answer keys already present in the saved
stores. Only if that gate passes is any cached generation reused.

Nothing is generated here. This step reports exactly what new inference the
calibrated configurations would need, so the cost is known before it is spent.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
V3 = PC.parent
ROOT = V3.parent
L1P = V3 / "l1_precision_gate"
RCI = V3 / "remaining_critical_issues"
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import schedules as sc                   # noqa: E402
import prep2 as p2                       # noqa: E402

ctx_for, h = p2.ctx_for, p2.h
SEED, SCHEDULE = 42, "zipf_uniform"
ANSWER_STORES = [
    RCI / "06_stronger_generator" / "answers_llama3b.jsonl",
    RCI / "01_l1_analysis" / "l1_answers.jsonl",
    V3 / "final_three_issues" / "02_l1_strict_gate" / "answers_llama3b.jsonl",
    L1P / "heldout_answers_llama3b.jsonl",
    L1P / "validation_answers_llama3b.jsonl",
]
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def main():
    say("TASK 1/2 -- 400-request answer audit for configurations A-D")
    fz = json.load(open(PC / "04_budgets" / "frozen_config.json", encoding="utf-8"))
    hr = json.load(open(PC / "05_heldout" / "heldout_results.json", encoding="utf-8"))
    sel, mhat = fz["selected_eps"], hr["m_content_mle"]
    say(f"  frozen budgets {sel}; m_content MLE {mhat:.4f} (parameters NOT changed here)")

    ans = {}
    for f in ANSWER_STORES:
        if f.exists():
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    ans[tuple(d["key"])] = d["answer"]
    say(f"  cached answers available: {len(ans):,} from "
        f"{sum(1 for f in ANSWER_STORES if f.exists())} stores")

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

    pub = json.load(open(L1P / "heldout_audit_sample.json", encoding="utf-8"))
    sample = pub["requests"]
    ids = [d["query_id"] for d in sample]
    say(f"  audit sample: {len(ids)} held-out requests, reused verbatim "
        f"(no re-sampling, no re-selection)")

    pub_m = dict(exp.TIER_MULT)
    pub_e = (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT)
    CFG = {
        "A_original":    (pub_m, pub_e),
        "B_mult_only":   ({**pub_m, "content": mhat}, pub_e),
        "C_budget_only": (pub_m, (sel["answer"], sel["url_list"], sel["content"])),
        "D_full":        ({**pub_m, "content": mhat},
                          (sel["answer"], sel["url_list"], sel["content"])),
    }
    rowsets, tasks = {}, {}
    try:
        for name, (mult, eps) in CFG.items():
            exp.TIER_MULT.update(mult)
            exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = eps
            _, rows = p2.replay(test, {}, "FreshCache")
            rowsets[name] = rows
    finally:
        exp.TIER_MULT.update(pub_m)
        exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = pub_e

    tmap = {r["query_id"]: t for t, r in test}
    arows, comp = [], {n: Counter() for n in CFG}
    for d in sample:
        q, t = d["query_id"], d["t"]
        rec = {"query_id": q, "cluster_id": d["cluster_id"], "query": d["query"],
               "gold": d["gold"], "fc": d["fc"], "t": t}
        # fresh answer: identical across configs by construction (own URLs at t)
        fctx = ctx_for(tuple((u, t) for u in rowsets["A_original"][q]["own"]))
        fk = ("fresh", h(fctx), h(d["query"]))
        tasks[fk] = (d["query"], fctx)
        rec["fresh_key"] = list(fk)
        assert list(fk) == d["fresh_key"], \
            f"fresh key mismatch on {q}: {list(fk)} vs {d['fresh_key']}"
        for name in CFG:
            p = rowsets[name][q]
            comp[name][p["tier"]] += 1
            rec[f"{name}_tier"] = p["tier"]
            if p["tier"] == "L1":
                k = ("stored", h(ctx_for(tuple(p["stored_answer_ev"]))),
                     h(p["stored_answer_query"]))
                tasks[k] = (p["stored_answer_query"],
                            ctx_for(tuple(p["stored_answer_ev"])))
            else:
                pctx = ctx_for(tuple(p["ev"]))
                k = ("gen", h(pctx), h(d["query"]))
                tasks[k] = (d["query"], pctx)
            rec[f"{name}_key"] = list(k)
        arows.append(rec)

    # ---------------- VERIFICATION GATE on configuration A ----------------
    say(f"\n  == verification gate: does configuration A reproduce the "
        f"existing audit? ==")
    same_ids = [d["query_id"] for d in arows] == ids
    say(f"    identical request IDs, in order: {same_ids}")
    same_comp = dict(comp["A_original"]) == pub["composition"]["published"]
    say(f"    tier composition {dict(comp['A_original'])} vs published "
        f"{pub['composition']['published']}: {same_comp}")
    a_keys = [tuple(r["A_original_key"]) for r in arows] + \
             [tuple(r["fresh_key"]) for r in arows]
    same_keys = [tuple(r["A_original_key"]) == tuple(s[f"published_key"])
                 for r, s in zip(arows, sample)]
    say(f"    per-request keys equal to the published arm's: "
        f"{sum(same_keys)}/{len(same_keys)}")
    miss = [k for k in set(a_keys) if k not in ans]
    say(f"    A's answer keys absent from the saved stores: {len(miss)} "
        f"of {len(set(a_keys))}")
    gate = same_ids and same_comp and all(same_keys) and not miss
    say(f"    GATE {'PASSED' if gate else 'FAILED'}")
    if not gate:
        say(f"    Not reusing any cached generation until this gate passes.")
        json.dump({"gate_passed": False, "missing_examples": [list(k) for k in miss[:10]]},
                  open(HERE / "audit_build.json", "w"), indent=2)
        sys.exit(2)

    # ---------------- cost of the calibrated configurations ----------------
    need = {k: v for k, v in tasks.items() if k not in ans}
    say(f"\n  == generation cost ==")
    say(f"    distinct generation tasks across A-D and fresh: {len(tasks):,}")
    say(f"    already cached byte-for-byte: {len(tasks)-len(need):,}")
    say(f"    NEW generations required:     {len(need):,}")
    for name in CFG:
        ks = {tuple(r[f"{name}_key"]) for r in arows}
        say(f"      {name:<15} keys {len(ks):>4}  new {len(ks - set(ans)):>4}  "
            f"composition {dict(comp[name])}")
    say(f"    judging: each new answer needs 2 primary judges "
        f"(llama8b, qwen7b) = {2*len(need):,} judge calls;")
    say(f"    already-judged pairs are reused by (h(query), h(gold), h(answer)).")
    say(f"    All inference is LOCAL GPU (Llama-3.2-3B generate, "
        f"Llama-3.1-8B + Qwen2.5-7B judge). No paid API calls.")

    json.dump({"gate_passed": True,
               "audit_n": len(arows),
               "composition": {k: dict(v) for k, v in comp.items()},
               "tasks_total": len(tasks), "tasks_cached": len(tasks)-len(need),
               "tasks_new": len(need),
               "per_config_new": {n: len({tuple(r[f"{n}_key"]) for r in arows}
                                         - set(ans)) for n in CFG}},
              open(HERE / "audit_build.json", "w"), indent=2)
    json.dump(arows, open(HERE / "audit_records.json", "w"), indent=1)
    with open(HERE / "gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    with open(HERE / "cached_answers.jsonl", "w", encoding="utf-8") as fh:
        for k in tasks:
            if k in ans:
                fh.write(json.dumps({"key": list(k), "answer": ans[k]}) + "\n")
    (PC / "logs" / "aq1.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote audit_build.json, audit_records.json, gen_tasks.jsonl, "
        f"cached_answers.jsonl")


if __name__ == "__main__":
    main()
