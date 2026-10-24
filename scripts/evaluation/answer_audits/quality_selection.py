#!/usr/bin/env python3
"""
quality_selection.py — apply the PRE-REGISTERED constraint and selection rule on
VALIDATION data only, freeze the result, then evaluate the frozen configuration
on the untouched held-out test split (operational metrics).

The constraint is never relaxed. If no candidate satisfies it, that is reported
as the outcome and the secondary sensitivity is shown as a separate labelled row.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import copy, csv, hashlib, json, math, os, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter
from fractions import Fraction

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = ((OUTPUT_ROOT / "quality_selection"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import tiered_cache_model as exp                 # noqa: E402
import execution_semantics as ce            # noqa: E402
import method_replay as ea                  # noqa: E402
import request_schedules as sc                   # noqa: E402
import mixed_age_replay as me                # noqa: E402
from schedule_robustness import support_of     # noqa: E402

ABST = ("i don't know", "i dont know", "cannot determine",
        "not enough information", "insufficient", "no information",
        "unclear from the context")
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]   # noqa
LOG = []


def say(s=""):
    print(s, flush=True); LOG.append(s)


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0, (c - m) / d), 3), round(100 * min(1, (c + m) / d), 3)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(Fraction(1), Fraction(
        sum(math.comb(n, i) for i in range(k + 1)) * 2, 2 ** n)))


def main():
    pre = json.load(open((OUTPUT_ROOT / "quality_selection/quality_selection_prereg.json"), encoding="utf-8"))
    say("Q6 — pre-registered selection on VALIDATION, then frozen held-out eval")
    say(f"  prereg sha256 {pre['prereg_sha256'][:24]}...")
    say(f"  constraint: {pre['quality_constraint']['primary']}")
    say(f"  objective : {pre['selection_rule']['objective']}")

    S = jl(HERE / "q_sample.jsonl")
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "q_answers.jsonl")}
    j1 = {tuple(r["sig"]): r["label"] for r in jl(HERE / "q_judge_llama8b.jsonl")}
    j2 = {tuple(r["sig"]): r["label"] for r in jl(HERE / "q_judge_qwen7b.jsonl")}
    ver = {r["query_id"]: r for r in csv.DictReader(
        open(HERE / "q_verification.csv", encoding="utf-8"))}
    arms = json.load(open(HERE / "quality_sample_summary.json", encoding="utf-8"))["arms"]
    ops = json.load(open(HERE / "quality_sample_summary.json", encoding="utf-8"))["operational"]

    def outc(d, key):
        a = ans.get(tuple(key))
        if a is None:
            return None
        if any(p in a.strip().lower() for p in ABST):
            return "ABSTAIN"
        s = (h(d["query"]), h(d["gold"]), h(a))
        l1, l2 = j1.get(s), j2.get(s)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    rows, dropped = [], 0
    for d in S:
        r = {"qid": d["query_id"], "fc": d["fc"],
             "fresh": outc(d, ("fresh", d["fresh_ctx_sha"], h(d["query"])))}
        for a in arms:
            r[a] = outc(d, tuple(d[f"{a}_key"]))
        if any(v is None for v in [r["fresh"]] + [r[a] for a in arms]):
            dropped += 1
            continue
        v = ver.get(d["query_id"], {})
        r["evidence_valid"] = int(v.get("evidence_valid", "0") == "1")
        rows.append(r)
    say(f"\n  scored {len(rows)} of {len(S)} sampled requests (dropped {dropped})")
    EV = [r for r in rows if r["evidence_valid"]]
    say(f"  VALIDATION evidence-valid subset: {len(EV)} requests")
    fc_corr = sum(1 for r in EV if r["fresh"] == "CORRECT")
    say(f"  fresh-correct on that subset (the conditional-WAI denominator): {fc_corr}")
    if fc_corr == 0:
        say("  STOP: no fresh-correct requests; the constraint cannot be evaluated")
        sys.exit(4)

    def stats(sub, a):
        n = len(sub)
        c = Counter(r[a] for r in sub)
        fcn = sum(1 for r in sub if r["fresh"] == "CORRECT")
        w = sum(1 for r in sub if r[a] == "WRONG" and r["fresh"] == "CORRECT")
        return {"N": n, "correct": c["CORRECT"], "abstain": c["ABSTAIN"],
                "accuracy_pct": round(100 * c["CORRECT"] / n, 3) if n else None,
                "wai": w, "wai_pct": round(100 * w / n, 3) if n else None,
                "fresh_correct_n": fcn,
                "conditional_wai_pct": round(100 * w / fcn, 3) if fcn else None,
                "conditional_wai_ci95": wilson(w, fcn)}

    fcq = stats(EV, "fc")
    say(f"\n  FROZEN FRESHCACHE on the validation evidence-valid subset:")
    say(f"    accuracy {fcq['correct']}/{fcq['N']} ({fcq['accuracy_pct']}%)  "
        f"WAI {fcq['wai']}  conditional WAI {fcq['conditional_wai_pct']}% "
        f"CI{fcq['conditional_wai_ci95']}")
    THRESH = fcq["conditional_wai_pct"]

    say(f"\n  CANDIDATE EVALUATION (constraint: conditional WAI <= {THRESH}%)")
    say(f"    {'candidate':<20}{'saved%':>10}{'acc%':>8}{'WAI':>6}"
        f"{'cond.WAI%':>11}{'CI95':>18}  constraint")
    cand_rows = []
    for c in pre["candidates"]:
        a = c["arm"]
        st = stats(EV, a)
        ok = (st["conditional_wai_pct"] is not None
              and st["conditional_wai_pct"] <= THRESH)
        ok5 = (st["conditional_wai_pct"] is not None
               and st["conditional_wai_pct"] <= THRESH + 5.0)
        cand_rows.append({**c, **st, "satisfies_primary": int(ok),
                          "satisfies_secondary_5pp": int(ok5)})
        say(f"    {a:<20}{c['saved']:>10.4f}{st['accuracy_pct']:>8.2f}"
            f"{st['wai']:>6}{st['conditional_wai_pct']:>11.2f}"
            f"{str(st['conditional_wai_ci95']):>18}  "
            f"{'SATISFIES' if ok else 'fails'}")

    sat = [c for c in cand_rows if c["satisfies_primary"]]
    sat5 = [c for c in cand_rows if c["satisfies_secondary_5pp"]]
    say(f"\n  candidates satisfying the PRIMARY constraint: {len(sat)}")
    say(f"  candidates satisfying the 5 pp SENSITIVITY   : {len(sat5)} "
        f"(reported separately, never used for the headline)")

    selected = None
    if sat:
        selected = sorted(sat, key=lambda c: (-c["saved"], c["drift"],
                                              -c["coverage"], -c["theta"],
                                              c["kappa"]))[0]
        say(f"\n  SELECTED (pre-registered rule): theta={selected['theta']:.2f} "
            f"kappa={selected['kappa_frac']}  validation saved "
            f"{selected['saved']:.4f}%  conditional WAI "
            f"{selected['conditional_wai_pct']:.2f}%")
    else:
        say("\n  NO CONFIGURATION SATISFIES THE PRE-REGISTERED QUALITY CONSTRAINT.")
        say("  The constraint is not relaxed. No configuration is selected, and")
        say("  no held-out scripts/evaluation of a 'quality-aware SemanticTTL' is performed,")
        say("  because no such configuration exists in the candidate set.")

    # drift-selected incumbent, for comparison
    inc = next((c for c in cand_rows
                if abs(c["theta"] - 0.40) < 1e-9 and abs(c["kappa"] - 0.0625) < 1e-9),
               None)
    if inc:
        say(f"\n  ORIGINAL drift-selected SemanticTTL (theta=0.40, kappa=1/16):")
        say(f"    validation saved {inc['saved']:.4f}%  accuracy {inc['accuracy_pct']}%"
            f"  WAI {inc['wai']}  conditional WAI {inc['conditional_wai_pct']}%"
            f"  -> {'satisfies' if inc['satisfies_primary'] else 'FAILS'} the constraint")

    # paired tests vs FreshCache on the validation evidence-valid subset
    say(f"\n  PAIRED WAI vs FreshCache on the validation evidence-valid subset")
    paired = {}
    for c in cand_rows:
        a = c["arm"]
        wa = lambda r, x: r[x] == "WRONG" and r["fresh"] == "CORRECT"    # noqa
        both = sum(1 for r in EV if wa(r, a) and wa(r, "fc"))
        ao = sum(1 for r in EV if wa(r, a) and not wa(r, "fc"))
        bo = sum(1 for r in EV if wa(r, "fc") and not wa(r, a))
        paired[a] = {"both": both, "sttl_only": ao, "freshcache_only": bo,
                     "exact_mcnemar_p": mcnemar(ao, bo)}
        say(f"    {a:<20} both {both:>3}  sttl-only {ao:>3}  fc-only {bo:>3}  "
            f"p {paired[a]['exact_mcnemar_p']:.4g}")

    # ---- held-out operational scripts/evaluation ----
    heldout = {}
    say(f"\n  HELD-OUT TEST EVALUATION (untouched split)")
    if selected is None:
        say("    skipped: no configuration was selected under the constraint")
    else:
        split = json.load(open((OUTPUT_ROOT / "validation_split/split.json"), encoding="utf-8"))
        TEST = set(split["test_clusters"])
        queries = exp.load_jsonl(exp.QUERIES_FILE)
        manifest = exp.load_jsonl(exp.MANIFEST_FILE)
        paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
                 if exp.PARAPHRASE_FILE.exists() else [])
        records = exp.build_query_records(queries, manifest, paras)
        exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
        exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
        rich = ea._rich_feats(); ea.set_cluster_base(records)
        rounds = {}
        with open((DATA_ROOT / "derived/round_table.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line); rounds[d["url_hash"]] = d["rounds"]
        full = sc.build_stream(records, "zipf_uniform", 42)
        ts = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in TEST]
        Sv = support_of(ts, rounds)
        ORIG = copy.deepcopy(exp.FIXED_TTL); L1T0 = exp.L1_SIM_THRESHOLD

        def score(m, per):
            ins = [o for qid, o in per if o is not None and qid in Sv]
            ch = sum(1 for o in ins if o == ce.CHANGED)
            un = sum(1 for o in ins if o == ce.UNCHANGED)
            return {"search_saved_pct": m["search_saved_pct"],
                    "drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None,
                    "changed": ch, "determinable": ch + un,
                    "coverage_pct": round(100 * (ch + un) / len(Sv), 4),
                    "fetches": m["fetches"], "generations": m["generations"],
                    "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
                    "l3_hits": m["l3_hits"], "n_requests": len(ts)}
        try:
            mfc, pfc = me.replay(ts, rounds, "FreshCache", rich=rich)
            heldout["FreshCache"] = score(mfc, pfc)
            exp.L1_SIM_THRESHOLD = selected["theta"]
            exp.FIXED_TTL.clear()
            exp.FIXED_TTL.update({c: v * selected["kappa"] for c, v in ORIG.items()})
            ms, ps = me.replay(ts, rounds, "SemanticTTL", rich=rich)
            heldout["SemanticTTL_selected"] = score(ms, ps)
        finally:
            exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
            exp.L1_SIM_THRESHOLD = L1T0
        assert exp.FIXED_TTL == ORIG and exp.L1_SIM_THRESHOLD == L1T0
        for k, v in heldout.items():
            say(f"    {k:<24} saved {v['search_saved_pct']:8.4f}%  drift "
                f"{v['drift_pct']:7.4f}% ({v['changed']}/{v['determinable']})  "
                f"cov {v['coverage_pct']:7.4f}%")

    json.dump({"prereg_sha256": pre["prereg_sha256"],
               "validation_scored": len(rows), "dropped": dropped,
               "validation_evidence_valid": len(EV),
               "fresh_correct_denominator": fc_corr,
               "freshcache_validation_quality": fcq,
               "constraint_threshold_conditional_wai_pct": THRESH,
               "candidates": cand_rows,
               "n_satisfying_primary": len(sat),
               "n_satisfying_secondary_5pp": len(sat5),
               "selected": selected,
               "no_configuration_satisfies_constraint": selected is None,
               "constraint_relaxed_after_seeing_results": False,
               "drift_selected_incumbent": inc,
               "paired_vs_freshcache": paired,
               "operational_validation": ops,
               "heldout": heldout,
               "test_clusters_used_for_selection": 0},
              open(HERE / "quality_selection_results.json", "w"), indent=2, ensure_ascii=False)
    (HERE / "q6.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote quality_selection_results.json, q6.log")


if __name__ == "__main__":
    main()
