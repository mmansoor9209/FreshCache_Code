#!/usr/bin/env python3
"""
Two-stage, validation-only selection, exactly as pre-registered.

  stage1  apply C1 (savings) and C3 (>=100 L1 hits), rank by the primary
          objective, carry the top K forward, and build ONE new policy-neutral
          400-request answer audit sampled from VALIDATION clusters so that C2
          (WAI) becomes measurable for those K cells.
  stage2  apply C2 and the pre-registered tie-breaks, and freeze EXACTLY ONE
          configuration to frozen_l1_precision_gate.json.

Held-out test results are never read by either stage.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, random, sys, time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-l1precision")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
sys.path.insert(0, str(HERE))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import engine_all as ea               # noqa: E402
import schedules as sc                # noqa: E402
import prep2 as p2                    # noqa: E402
import l1_gate                        # noqa: E402

SCHEDULE, SEED, TARGET = "zipf_uniform", 42, 400
band, ctx_for, h = p2.band, p2.ctx_for, p2.h
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def is_abstain(t):
    tl = (t or "").strip().lower()
    return any(p in tl for p in ABSTAIN_PAT)


def build_stream():
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
    val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in val_c]
    assert len(stream) == split["validation"]["requests"]
    assert not [1 for _, r in stream
                if (r.get("cluster_id") or r["query_id"]) in test_c]
    return records, stream


def stage1(args):
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    P = json.load(open(HERE / "l1_precision_prereg.json", encoding="utf-8"))
    K = P["selection"]["stage_1"]["K"]
    rows = json.load(open(HERE / "validation_grid_scored.json", encoding="utf-8"))
    pub = [r for r in rows if r["cell"] == "published"][0]
    say("L1-PRECISION -- STAGE 1 (validation only)")
    say(f"  prereg sha256 {sha(HERE / 'l1_precision_prereg.json')}")
    say(f"  published validation: L1 {pub['l1_hits']:,}  "
        f"saved {pub['search_saved_pct']:.4f}%  "
        f"mismatch(ties->DIFF) {pub['mismatch_ties_diff_pct']:.2f}%")

    C1 = pub["search_saved_pct"] - 0.10
    say(f"\n  C1  validation search savings >= {C1:.4f}%")
    say(f"  C3  at least 100 validation L1 hits")
    cand = [r for r in rows if r["cell"] != "published"
            and r["mismatch_ties_diff_pct"] is not None]
    c1 = [r for r in cand if r["search_saved_pct"] >= C1]
    c13 = [r for r in c1 if r["l1_hits"] >= 100]
    say(f"  cells: {len(cand)} scored -> {len(c1)} pass C1 -> "
        f"{len(c13)} pass C1 and C3")
    if not c13:
        say("  NO cell satisfies the pre-registered constraints. Reported as "
            "such; the constraints are NOT relaxed after seeing results.")
        json.dump({"feasible": 0}, open(HERE / "stage1_selection.json", "w"),
                  indent=2)
        return

    ranked = sorted(c13, key=lambda r: r["mismatch_ties_diff_pct"])
    top = ranked[:K]
    say(f"\n  top {K} by the primary objective (mismatch, ties->DIFFERENT):")
    say(f"    {'cell':<34}{'L1':>6}{'mm_tD%':>9}{'mm_noT%':>9}{'saved%':>10}"
        f"{'gen/1k':>9}")
    for r in top:
        say(f"    {r['cell']:<34}{r['l1_hits']:>6,}"
            f"{r['mismatch_ties_diff_pct']:>8.2f}%"
            f"{r['mismatch_no_ties_pct']:>8.2f}%"
            f"{r['search_saved_pct']:>9.4f}%{r['gen_per_1k']:>9.2f}")

    # ---- build ONE validation-side answer audit so C2 becomes measurable ----
    records, stream = build_stream()
    gold = json.load(open(p2.GOLD, encoding="utf-8"))

    def goldof(qid, qt):
        g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower())
        return (g.get("answer") if isinstance(g, dict) else g)

    base = ea._CLUSTER_BASE
    IDX, SIM = exp._QUERY_TO_IDX, exp._SIM_MATRIX
    simbase = {}
    for _, r in stream:
        bq = base.get(r["query_id"], "")
        simbase[r["query_id"]] = (float(SIM[IDX[r["query"]], IDX[bq]])
                                  if r.get("is_paraphrase") and bq and bq in IDX
                                  else None)
    frame = []
    for t, r in stream:
        q = r["query_id"]
        if r["freshness_class"] == "REAL_TIME":
            continue
        g = goldof(q, r["query"])
        if not g:
            continue
        own = [u["url_hash"] for u in r["urls"]]
        if not ctx_for(tuple((u, t) for u in own)):
            continue
        frame.append({"qid": q, "t": t, "query": r["query"], "gold": g,
                      "fc": r["freshness_class"], "simbase": simbase.get(q),
                      "cluster": r.get("cluster_id") or q})
    strata = defaultdict(list)
    for d in frame:
        strata[(d["fc"], band(d["simbase"]))].append(d)
    rng = random.Random(SEED)
    per = max(1, TARGET // max(len(strata), 1))
    sample = []
    for k in sorted(strata):
        g = strata[k][:]
        rng.shuffle(g)
        sample.extend(g[:per])
    if len(sample) < TARGET:
        chosen = {d["qid"] for d in sample}
        rest = [d for d in frame if d["qid"] not in chosen]
        rng.shuffle(rest)
        sample.extend(rest[:TARGET - len(sample)])
    sample = sample[:TARGET]
    say(f"\n  VALIDATION answer-audit frame {len(frame):,}; strata "
        f"{len(strata)}; sampled {len(sample)} (seed {SEED}, same 28-stratum "
        f"design, same evidence builder and context limits as the published "
        f"audit)")
    say(f"  the sample is built from the frame, before any policy is consulted")

    ORIG = exp.semantic_equivalent
    arms = {"published": (exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN,
                          "as_implemented", False)}
    for r in top:
        arms[r["cell"]] = (r["sim_floor"], r["jaccard_floor"],
                           r["answer_type"], bool(r["guards"]))
    rowsets = {}
    for name, (sf, jf, at, gd) in arms.items():
        exp.semantic_equivalent = l1_gate.make_gate(exp, sf, jf, at, gd)
        try:
            m_, rows_ = p2.replay(stream, {}, "FreshCache")
        finally:
            exp.semantic_equivalent = ORIG
        rowsets[name] = rows_
        say(f"    arm {name:<34} L1 {m_['l1_hits']:>5,}  "
            f"saved {m_['search_saved_pct']:.4f}%")
    assert exp.semantic_equivalent is ORIG

    out, tasks = [], {}
    comp = {a: Counter() for a in arms}
    for d in sample:
        q, t = d["qid"], d["t"]
        rec = {"query_id": q, "cluster_id": d["cluster"], "query": d["query"],
               "gold": d["gold"], "fc": d["fc"], "t": t}
        fctx = ctx_for(tuple((u, t) for u in rowsets["published"][q]["own"]))
        fk = ("fresh", h(fctx), h(d["query"]))
        tasks[fk] = (d["query"], fctx)
        rec["fresh_key"] = list(fk)
        for name in arms:
            p = rowsets[name][q]
            comp[name][p["tier"]] += 1
            rec[f"{name}_tier"] = p["tier"]
            if p["tier"] == "L1":
                sctx = ctx_for(tuple(p["stored_answer_ev"]))
                sq = p["stored_answer_query"]
                k = ("stored", h(sctx), h(sq))
                tasks[k] = (sq, sctx)
            else:
                pctx = ctx_for(tuple(p["ev"]))
                k = ("gen", h(pctx), h(d["query"]))
                tasks[k] = (d["query"], pctx)
            rec[f"{name}_key"] = list(k)
        out.append(rec)
    json.dump({"target": TARGET, "seed": SEED, "split": "VALIDATION clusters",
               "arms": list(arms), "arm_params": {k: list(v) for k, v in arms.items()},
               "composition": {k: dict(v) for k, v in comp.items()},
               "requests": out},
              open(HERE / "validation_audit_sample.json", "w"), indent=2)
    with open(HERE / "validation_gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    json.dump({"K": K, "C1_threshold": C1, "published": pub,
               "n_pass_c1": len(c1), "n_pass_c1_c3": len(c13),
               "top_k": top},
              open(HERE / "stage1_selection.json", "w"), indent=2)
    say(f"\n  unique generation tasks {len(tasks):,}")
    say(f"  wrote validation_audit_sample.json, validation_gen_tasks.jsonl, "
        f"stage1_selection.json")
    (HERE / "logs" / "stage1.log").write_text("\n".join(log) + "\n",
                                              encoding="utf-8")


def stage2(args):
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    P = json.load(open(HERE / "l1_precision_prereg.json", encoding="utf-8"))
    S1 = json.load(open(HERE / "stage1_selection.json", encoding="utf-8"))
    A = json.load(open(HERE / "validation_audit_sample.json", encoding="utf-8"))
    ans = {tuple(r["key"]): r["answer"] for r in
           (json.loads(l) for l in
            open(HERE / "validation_answers_llama3b.jsonl", encoding="utf-8")
            if l.strip())}
    PRIM = ["llama8b", "qwen7b"]
    jud = {}
    for j in PRIM:
        p = HERE / f"validation_judge_{j}.jsonl"
        jud[j] = {tuple(r["sig"]): r["label"] for r in
                  (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}
    say("L1-PRECISION -- STAGE 2 (validation only): C2 and the tie-breaks")
    say(f"  labelling rule: the manuscript rule -- ABSTAIN on the abstention "
        f"pattern, else CORRECT only if BOTH {PRIM} say CORRECT, else WRONG")

    def lab(d, key):
        a = ans.get(tuple(d[key]))
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h16(d["query"]), h16(d["gold"]), h16(a))
        l1 = jud[PRIM[0]].get(sig)
        l2 = jud[PRIM[1]].get(sig)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    reqs = A["requests"]
    fresh = {d["query_id"]: lab(d, "fresh_key") for d in reqs}
    fc_ok = [q for q, v in fresh.items() if v == "CORRECT"]
    say(f"  validation audit {len(reqs)} requests; fresh CORRECT {len(fc_ok)}")
    wai = {}
    for name in A["arms"]:
        L = {d["query_id"]: lab(d, f"{name}_key") for d in reqs}
        w = sum(1 for q in L if L[q] == "WRONG" and fresh.get(q) == "CORRECT")
        acc = sum(1 for v in L.values() if v == "CORRECT")
        wai[name] = {"wai": w, "n": len(reqs),
                     "wai_pct": round(100 * w / len(reqs), 4),
                     "correct": acc,
                     "accuracy_pct": round(100 * acc / len(reqs), 4),
                     "conditional_wai_pct": (round(100 * w / len(fc_ok), 4)
                                             if fc_ok else None)}
        say(f"    {name:<34} WAI {w}/{len(reqs)} = "
            f"{100*w/len(reqs):.4f}%   accuracy {acc}/{len(reqs)} = "
            f"{100*acc/len(reqs):.4f}%")
    pub_wai = wai["published"]["wai"]
    say(f"\n  C2  validation WAI must not exceed the published gate's "
        f"{pub_wai}/{len(reqs)}")

    top = S1["top_k"]
    kept, dropped = [], []
    for r in top:
        w = wai.get(r["cell"])
        if w is None:
            r["_wai"] = None
            r["_c2"] = "NOT_EVALUABLE"
            kept.append(r)
            continue
        r["_wai"] = w["wai"]
        r["_c2"] = "PASS" if w["wai"] <= pub_wai else "FAIL"
        (kept if r["_c2"] != "FAIL" else dropped).append(r)
    for r in dropped:
        say(f"    DROPPED by C2: {r['cell']}  WAI {r['_wai']} > {pub_wai}")
    if not kept:
        say("  NO carried-forward cell satisfies C2. Reported as such; the "
            "constraint is NOT relaxed.")
        json.dump({"frozen": None, "reason": "no cell satisfies C2",
                   "wai": wai}, open(HERE / "frozen_l1_precision_gate.json", "w"),
                  indent=2)
        return

    # pre-registered tie-breaks, in order
    pubgen = S1["published"]["generations"]
    kept.sort(key=lambda r: (r["mismatch_ties_diff_pct"],
                             (r["_wai"] if r["_wai"] is not None else 10 ** 9),
                             -r["l1_hits"],
                             r["generations"] - pubgen,
                             r["sim_floor"]))
    win = kept[0]
    say(f"\n  tie-breaks applied in the pre-registered order "
        f"(mismatch, WAI, more L1 hits, fewer extra generations, lower sim)")
    for r in kept:
        say(f"    {r['cell']:<34} mm {r['mismatch_ties_diff_pct']:>6.2f}%  "
            f"WAI {str(r['_wai']):>4}  L1 {r['l1_hits']:>5,}  "
            f"+gen {r['generations']-pubgen:>+5,}  sim {r['sim_floor']}")
    say(f"\n  FROZEN configuration: {win['cell']}")
    say(f"    sim_floor {win['sim_floor']}  jaccard_floor {win['jaccard_floor']}"
        f"  answer_type {win['answer_type']}  guards {bool(win['guards'])}")
    say(f"    validation: L1 {win['l1_hits']:,}  "
        f"mismatch(ties->DIFF) {win['mismatch_ties_diff_pct']:.4f}%  "
        f"mismatch(no ties) {win['mismatch_no_ties_pct']:.4f}%  "
        f"saved {win['search_saved_pct']:.4f}%  WAI {win['_wai']}")

    frozen = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg_sha256": sha(HERE / "l1_precision_prereg.json"),
        "l1_guards_sha256": sha(HERE / "l1_guards.py"),
        "selected_on": "VALIDATION clusters only; held-out test never consulted",
        "cell": win["cell"],
        "sim_floor": win["sim_floor"], "jaccard_floor": win["jaccard_floor"],
        "answer_type": win["answer_type"], "guards": bool(win["guards"]),
        "validation": {k: win[k] for k in
                       ("l1_hits", "mismatch_ties_diff_pct",
                        "mismatch_no_ties_pct", "mismatch_ties_same_pct",
                        "cross_cluster_mismatch_ties_diff_pct",
                        "search_saved_pct", "generations", "gen_per_1k",
                        "fetches", "drift_pct", "coverage_pct")},
        "validation_wai": wai.get(win["cell"]),
        "published_validation_wai": wai["published"],
        "constraints": {"C1_savings_floor": S1["C1_threshold"],
                        "C2_wai_ceiling": pub_wai,
                        "C3_min_l1_hits": 100},
        "all_validation_wai": wai,
        "candidates_considered": [r["cell"] for r in top],
    }
    json.dump(frozen, open(HERE / "frozen_l1_precision_gate.json", "w"),
              indent=2)
    say(f"\n  wrote frozen_l1_precision_gate.json  "
        f"(sha256 {sha(HERE / 'frozen_l1_precision_gate.json')})")
    say(f"  EXACTLY ONE configuration is frozen. run_heldout.py may now run.")
    (HERE / "logs" / "stage2.log").write_text("\n".join(log) + "\n",
                                              encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1").set_defaults(func=stage1)
    sub.add_parser("stage2").set_defaults(func=stage2)
    a = ap.parse_args()
    a.func(a)
