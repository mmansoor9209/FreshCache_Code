#!/usr/bin/env python3
"""Risk-budget calibration: build the held-out answer audit for configurations A-D.

Inputs are the frozen budgets (outputs/parameter_calibration/budgets/frozen_config.json)
and the validation-fitted content multiplier
(outputs/parameter_calibration/heldout/heldout_results.json); no parameter is
changed here.

Answer stores are keyed by the TUPLE (role, h(context), h(question)) with role in
{stored, fresh, gen}, exactly as
scripts/experiments/ablations/l1_precision_gate/l1_precision_heldout_replay.py builds it.
The key scheme is VERIFIED by requiring that configuration A -- the reported
FreshCache -- reproduces the existing audit: identical request IDs
in order, identical tier composition, and every one of its answer keys already
present in the saved stores. Only if that gate passes is any cached generation
reused.

Nothing is generated here. This step reports exactly what new inference the
calibrated configurations would need, so the cost is known before it is spent.
Writes audit_build.json, audit_records.json, gen_tasks.jsonl and
cached_answers.jsonl to outputs/parameter_calibration/answer_quality/.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import hashlib, json, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = ((OUTPUT_ROOT / "parameter_calibration/answer_quality"))
PC = ((OUTPUT_ROOT / "parameter_calibration"))
PRIOR = ((OUTPUT_ROOT / "archived_runs"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
L1P = ((OUTPUT_ROOT / "l1_precision"))
RCI = ((OUTPUT_ROOT / "l1only_exactttl"))
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import tiered_cache_model as exp                 # noqa: E402
import method_replay as ea                  # noqa: E402
import request_schedules as sc                   # noqa: E402
import audit_replay as p2                       # noqa: E402

ctx_for, h = p2.ctx_for, p2.h
SEED, SCHEDULE = 42, "zipf_uniform"
ANSWER_STORES = [
    RCI / "stronger_generator" / "answers_llama3b.jsonl",
    RCI / "l1_hit_analysis" / "l1_answers.jsonl",
    ((OUTPUT_ROOT / "l1_strict_gate/strict_gate_audit/answers_llama3b.jsonl")),
    L1P / "heldout_answers_llama3b.jsonl",
    L1P / "validation_answers_llama3b.jsonl",
]
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def main():
    say("TASK 1/2 -- answer audit for configurations A-D")
    fz = json.load(open(PC / "budgets" / "frozen_config.json", encoding="utf-8"))
    hr = json.load(open(PC / "heldout" / "heldout_results.json", encoding="utf-8"))
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
    split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"), encoding="utf-8"))
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
    same_comp = dict(comp["A_original"]) == pub["composition"]["recorded"]
    say(f"    tier composition {dict(comp['A_original'])} vs recorded "
        f"{pub['composition']['recorded']}: {same_comp}")
    a_keys = [tuple(r["A_original_key"]) for r in arows] + \
             [tuple(r["fresh_key"]) for r in arows]
    same_keys = [tuple(r["A_original_key"]) == tuple(s[f"recorded_key"])
                 for r, s in zip(arows, sample)]
    say(f"    per-request keys equal to the recorded arm's: "
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
