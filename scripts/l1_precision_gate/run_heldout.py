#!/usr/bin/env python3
"""
Held-out sequential replay: published L1 vs strict-0.90 L1 vs the frozen
L1-Precision gate. Run ONCE, after frozen_l1_precision_gate.json exists.

Only the L1 semantic admission condition differs between arms. L2 similarity
threshold, L2 temporal eligibility, L2 re-registration, all L3 logic,
half-lives, risk budgets, timestamps, cache-update semantics, request order and
the freshness-class source are identical, and the implementation files are
hashed before and after to prove they were not edited.
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
RCI = V3 / "remaining_critical_issues"
sys.path.insert(0, str(HERE))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import engine_all as ea               # noqa: E402
import mixed_engine as me             # noqa: E402
import schedules as sc                # noqa: E402
import prep2 as p2                    # noqa: E402
from e1_robustness import support_of  # noqa: E402
import l1_gate                        # noqa: E402
import l1_guards as G                 # noqa: E402

SCHEDULE, SEED, TARGET = "zipf_uniform", 42, 400
CHANGED, UNCHANGED = me.CHANGED, me.UNCHANGED
band, ctx_for, h = p2.band, p2.ctx_for, p2.h
WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
         "v14_baselines/engine_all.py", "v16_exp12/schedules.py",
         "validation/mixed_age_full_policy_audit/prep2.py"]
ANCHOR = {"search_saved_pct": 60.5776, "l1_hits": 806, "l2_hits": 12429}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    F = HERE / "frozen_l1_precision_gate.json"
    if not F.exists():
        sys.exit("REFUSING to run: frozen_l1_precision_gate.json does not "
                 "exist. Selection must complete on validation first.")
    FR = json.load(open(F, encoding="utf-8"))
    if FR.get("cell") is None:
        sys.exit("REFUSING to run: no configuration was frozen (no cell "
                 "satisfied the pre-registered constraints).")
    say("L1-PRECISION GATE -- held-out sequential replay (ONE run)")
    say(f"  frozen file sha256 {sha(F)}")
    say(f"  prereg sha256      {FR['prereg_sha256']}")
    say(f"  guards sha256      {FR['l1_guards_sha256']}  "
        f"(current {sha(HERE / 'l1_guards.py')})")
    assert FR["l1_guards_sha256"] == sha(HERE / "l1_guards.py"), \
        "l1_guards.py changed after freezing"
    say(f"  FROZEN gate: sim>={FR['sim_floor']}  jaccard>={FR['jaccard_floor']}"
        f"  answer_type={FR['answer_type']}  guards={FR['guards']}")
    say(f"  selected on: {FR['selected_on']}")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c, val_c = set(split["test_clusters"]), set(split["validation_clusters"])
    assert not (test_c & val_c)
    full = sc.build_stream(records, SCHEDULE, SEED)
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(stream) == split["test"]["requests"]
    assert not [1 for _, r in stream
                if (r.get("cluster_id") or r["query_id"]) in val_c]
    S = support_of(stream, rounds)
    say(f"\n  held-out stream {len(stream):,} requests, {len(test_c):,} "
        f"clusters, |S| = {len(S):,}; validation leakage 0")

    ORIG = exp.semantic_equivalent
    ARMS = {
        "published": (exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN, "as_implemented", False),
        "strict090": (0.90, 0.30, "as_implemented", False),
        "precision": (FR["sim_floor"], FR["jaccard_floor"],
                      FR["answer_type"], bool(FR["guards"])),
    }
    mets, rowsets = {}, {}
    for name, (sf, jf, at, gd) in ARMS.items():
        exp.semantic_equivalent = l1_gate.make_gate(exp, sf, jf, at, gd)
        try:
            m_, rows_ = p2.replay(stream, {}, "FreshCache")
        finally:
            exp.semantic_equivalent = ORIG
        m_["generations"] = m_["search_calls"] + m_["l2_hits"]
        mets[name], rowsets[name] = m_, rows_
    assert exp.semantic_equivalent is ORIG

    bad = [k for k, v in ANCHOR.items()
           if abs(mets["published"][k] - v) > (1e-4 if isinstance(v, float) else 0)]
    if bad:
        say(f"  ANCHOR GATE FAILED on {bad}"); sys.exit(2)
    say(f"  ANCHOR GATE PASSED: published arm reproduces the held-out anchor "
        f"(60.5776%, L1 806, L2 12,429)")
    s = mets["strict090"]
    say(f"  strict-0.90 cross-check: L1 {s['l1_hits']:,} "
        f"(the earlier experiment reported 450)")

    # ---------------- drift / coverage per arm ----------------
    def drift_cov(rows_):
        ch = un = 0
        for t, r in stream:
            q = r["query_id"]
            p = rows_.get(q)
            if not p or q not in S:
                continue
            if p["tier"] == "L1":
                labs = [me.outcome(x, p["t_cached"], t, rounds)
                        for x in p.get("served", [])]
            else:
                labs = [me.outcome(u, tc, t, rounds)
                        for u, tc in p.get("ev", []) if tc < t]
            if not labs:
                continue
            o = (CHANGED if CHANGED in labs else
                 (UNCHANGED if UNCHANGED in labs else None))
            if o == CHANGED:
                ch += 1
            elif o == UNCHANGED:
                un += 1
        det = ch + un
        return ((round(100 * ch / det, 4) if det else None),
                (round(100 * det / len(S), 4) if S else None), det, ch)

    n = len(stream)
    say(f"\n  {'arm':<12}{'L1':>7}{'L2':>8}{'L3':>8}{'saved%':>10}"
        f"{'gen':>9}{'gen/1k':>9}{'fetch/1k':>10}{'drift%':>9}{'cov%':>9}")
    summ = {}
    for name in ARMS:
        m_ = mets[name]
        d, c, det, chg = drift_cov(rowsets[name])
        summ[name] = {**m_, "drift_pct": d, "coverage_pct": c,
                      "determinable": det, "changed": chg,
                      "gen_per_1k": round(1000 * m_["generations"] / n, 3),
                      "fetch_per_1k": round(1000 * m_["fetches"] / n, 3),
                      "params": list(ARMS[name])}
        say(f"  {name:<12}{m_['l1_hits']:>7,}{m_['l2_hits']:>8,}"
            f"{m_['l3_hits']:>8,}{m_['search_saved_pct']:>9.4f}%"
            f"{m_['generations']:>9,}{summ[name]['gen_per_1k']:>9.2f}"
            f"{summ[name]['fetch_per_1k']:>10.2f}"
            + (f"{d:>8.4f}%" if d is not None else f"{'-':>9}")
            + (f"{c:>8.4f}%" if c is not None else f"{'-':>9}"))

    # ---------------- where do rejected L1 hits go? ----------------
    say(f"\n  == displacement of published-gate L1 hits ==")
    disp = {}
    for name in ("strict090", "precision"):
        moved = Counter()
        for q, v in rowsets["published"].items():
            if v["tier"] != "L1":
                continue
            tb = rowsets[name].get(q, {}).get("tier")
            if tb != "L1":
                moved["L2" if tb == "L2" else
                      "fresh_search" if tb == "miss" else
                      "REAL_TIME_or_other"] += 1
        gained = sum(1 for q, v in rowsets[name].items()
                     if v["tier"] == "L1"
                     and rowsets["published"].get(q, {}).get("tier") != "L1")
        disp[name] = {"rejected_total": sum(moved.values()),
                      "absorbed_by_L2": moved["L2"],
                      "require_fresh_search": moved["fresh_search"],
                      "other": moved["REAL_TIME_or_other"],
                      "newly_L1_from_cache_divergence": gained}
        say(f"    {name}: rejected {sum(moved.values()):,} -> "
            f"L2 {moved['L2']:,}, fresh search {moved['fresh_search']:,}, "
            f"other {moved['REAL_TIME_or_other']:,}; newly-L1 {gained:,}")

    # ---------------- per-request trace ----------------
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    with open(HERE / "heldout_per_request.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "cluster_id", "freshness_class", "t",
                    "tier_published", "tier_strict090", "tier_precision",
                    "sim_published", "sim_precision"])
        for t, r in stream:
            q = r["query_id"]
            w.writerow([q, cid[q], r["freshness_class"], t,
                        rowsets["published"].get(q, {}).get("tier"),
                        rowsets["strict090"].get(q, {}).get("tier"),
                        rowsets["precision"].get(q, {}).get("tier"),
                        rowsets["published"].get(q, {}).get("similarity"),
                        rowsets["precision"].get(q, {}).get("similarity")])

    # ---------------- every realised L1 hit, all three arms ----------------
    gold = json.load(open(p2.GOLD, encoding="utf-8"))
    def goldof(qid, qt):
        g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower())
        return (g.get("answer") if isinstance(g, dict) else g)
    cl_of_q = {}
    for r in records:
        cl_of_q.setdefault(r["query"], r.get("cluster_id") or r["query_id"])

    hits, tasks, pairs = [], {}, {}
    for name in ARMS:
        for t, r in stream:
            q = r["query_id"]
            p = rowsets[name].get(q)
            if not p or p["tier"] != "L1":
                continue
            inc, mq = r["query"], p["matched_query"]
            simv = p.get("similarity")
            t1, t2 = exp._eq_content_tokens(inc), exp._eq_content_tokens(mq)
            jac = len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0
            sctx = ctx_for(tuple(p["stored_answer_ev"]))
            sq = p["stored_answer_query"]
            own = [u["url_hash"] for u in r["urls"]]
            fctx = ctx_for(tuple((u, t) for u in own))
            sk = ("stored", h(sctx), h(sq))
            fk = ("fresh", h(fctx), h(inc))
            tasks[sk] = (sq, sctx)
            if fctx:
                tasks[fk] = (inc, fctx)
            pid = f"{h16(inc)}_{h16(mq)}"
            pairs[pid] = (inc, mq)
            hits.append({
                "arm": name, "request_id": q, "cluster_id": cid[q],
                "incoming_query": inc, "cached_query": mq,
                "stored_answer_query": sq,
                "similarity": simv, "jaccard": round(jac, 4),
                "entities_incoming": "|".join(sorted(exp._entities(inc))),
                "entities_cached": "|".join(sorted(exp._entities(mq))),
                "entity_match": bool(exp._entity_match(inc, mq)),
                "answer_type_incoming": exp._eq_answer_type(inc),
                "answer_type_cached": exp._eq_answer_type(mq),
                **G.guard_detail(inc, mq),
                "freshness_class": r["freshness_class"],
                "cache_age_seconds": p.get("age"),
                "reference_answer": goldof(q, inc),
                "cross_cluster": int(cl_of_q.get(mq) != cid[q]),
                "pair_id": pid,
                "stored_key": "|".join(sk),
                "fresh_key": "|".join(fk) if fctx else "",
            })
    with open(HERE / "heldout_l1_hits.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hits[0]))
        w.writeheader(); w.writerows(hits)
    with open(HERE / "jury_tasks.jsonl", "w", encoding="utf-8") as fh:
        for pid, (a, b) in sorted(pairs.items()):
            fh.write(json.dumps({"pair_id": pid, "incoming": a,
                                 "cached": b}) + "\n")
    say(f"\n  realised L1 hits captured: "
        f"{ {k: sum(1 for x in hits if x['arm']==k) for k in ARMS} }")
    say(f"  distinct (incoming, cached) pairs to judge: {len(pairs):,}")

    # ---------------- the SAME held-out 400-request audit ----------------
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
                      "cluster": cid[q]})
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
    pub = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                         / "answer_audit_k1_16" / "sample.json",
                         encoding="utf-8"))
    assert [d["qid"] for d in sample] == [r["query_id"] for r in pub["requests"]], \
        "SAMPLE GATE FAILED: not the published 400-request held-out audit"
    say(f"  SAMPLE GATE PASSED: the 400 audit request IDs are identical to the "
        f"published held-out audit (answer_audit_k1_16), in the same order")
    say(f"  freshness-class source: COLLECTION CLASS "
        f"(experiment.build_query_records) -- the manuscript headline arm")

    arows, comp = [], {a: Counter() for a in ARMS}
    for d in sample:
        q, t = d["qid"], d["t"]
        rec = {"query_id": q, "cluster_id": d["cluster"], "query": d["query"],
               "gold": d["gold"], "fc": d["fc"], "t": t}
        fctx = ctx_for(tuple((u, t) for u in rowsets["published"][q]["own"]))
        fk = ("fresh", h(fctx), h(d["query"]))
        tasks[fk] = (d["query"], fctx)
        rec["fresh_key"] = list(fk)
        for name in ARMS:
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
        arows.append(rec)
    json.dump({"target": TARGET, "seed": SEED, "arms": list(ARMS),
               "freshness_class_source": "collection class "
                                         "(experiment.build_query_records)",
               "sample_gate": "identical to answer_audit_k1_16",
               "composition": {k: dict(v) for k, v in comp.items()},
               "requests": arows},
              open(HERE / "heldout_audit_sample.json", "w"), indent=2)
    for k, v in comp.items():
        say(f"  audit composition {k:<12} {dict(v)}")

    # reuse every answer already produced by earlier runs, byte-for-byte
    prev = {}
    for f in (RCI / "06_stronger_generator" / "answers_llama3b.jsonl",
              RCI / "01_l1_analysis" / "l1_answers.jsonl",
              V3 / "final_three_issues" / "02_l1_strict_gate" / "answers_llama3b.jsonl"):
        if f.exists():
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    d = json.loads(l)
                    prev[tuple(d["key"])] = d["answer"]
    reuse = {k: v for k, v in prev.items() if k in tasks}
    with open(HERE / "heldout_answers_llama3b.jsonl", "w", encoding="utf-8") as fh:
        for k, v in reuse.items():
            fh.write(json.dumps({"key": list(k), "answer": v}) + "\n")
    with open(HERE / "heldout_gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    say(f"  generation tasks {len(tasks):,}; reusable byte-for-byte "
        f"{len(reuse):,}; new {len(tasks)-len(reuse):,}")

    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post, "implementation files changed"
    say(f"  implementation files unchanged: True")

    json.dump({"frozen": FR, "arms": {k: summ[k] for k in ARMS},
               "displacement": disp,
               "n_requests": n, "support": len(S), "hashes": pre},
              open(HERE / "heldout_replay.json", "w"), indent=2)
    (HERE / "logs" / "heldout_replay.log").write_text("\n".join(log) + "\n",
                                                      encoding="utf-8")
    say("  wrote heldout_per_request.csv, heldout_l1_hits.csv, "
        "jury_tasks.jsonl, heldout_audit_sample.json, heldout_replay.json")


if __name__ == "__main__":
    main()
