#!/usr/bin/env python3
"""
Issue 2 -- stricter L1 gate, FULL held-out evaluation.

The stricter gate was selected on VALIDATION ONLY in a pre-registered sweep
(validation_V3/remaining_critical_issues/01_l1_analysis/stricter_prereg.json,
sha 4e5cd495...). It is read from that artifact and asserted here; no threshold
is chosen using held-out results.

This script runs a GENUINE SEQUENTIAL REPLAY of FreshCache over the complete
held-out test traffic under the stricter gate -- the cache state that later
requests see is the state the stricter policy actually produced. The effect is
NOT estimated by filtering the original gate's L1 hits.

Everything except the L1 equivalence floors is identical between the two arms:
L2, L3, temporal eligibility, half-lives, epsilons, timestamps, cache-update
semantics, request order and freshness classes. The stricter gate is realised
by setting exp._EQ_SIM_FLOOR and exp._EQ_JACCARD_MIN, which
exp.semantic_equivalent reads at call time, so the published replay code runs
verbatim.

Outputs the per-request trace of both arms, every realised stricter-gate L1 hit
with the twelve required fields, the jury and answer-generation task files, and
the 400-request audit arms for both gates.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, random, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-final3")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
V3 = BASE.parent
ROOT = V3.parent
RCI = V3 / "remaining_critical_issues"
sys.path.insert(0, str(RCI / "lib"))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import engine_all as ea               # noqa: E402
import schedules as sc                # noqa: E402
import mixed_engine as me             # noqa: E402
import prep2 as p2                    # noqa: E402

SCHEDULE, SEED, TARGET = "zipf_uniform", 42, 400
band, ctx_for, h = p2.band, p2.ctx_for, p2.h
WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
         "v14_baselines/engine_all.py", "v16_exp12/schedules.py",
         "validation/mixed_age_full_policy_audit/prep2.py"]
ANCHOR = {"search_saved_pct": 60.5776, "l1_hits": 806, "l2_hits": 12429}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jaccard(a, b):
    ta, tb = set(exp._eq_content_tokens(a)), set(exp._eq_content_tokens(b))
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def ents(q):
    try:
        return sorted(exp._entities(q))
    except Exception:
        try:
            return sorted(exp._entity_set(q))
        except Exception:
            return []


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("ISSUE 2 -- stricter L1 gate, full held-out sequential replay")

    # ---------------- the frozen gate, read not chosen ----------------
    S = json.load(open(RCI / "01_l1_analysis" / "stricter_l1_results.json",
                       encoding="utf-8"))
    fr = S["frozen"]
    pj = RCI / "01_l1_analysis" / "stricter_prereg.json"
    assert sha(pj) == S["prereg_sha256"], "pre-registration file has changed"
    SIMF, JACF, REQT = fr["sim_floor"], fr["jac_floor"], fr["require_answer_type"]
    assert abs(SIMF - 0.90) < 1e-12 and abs(JACF - 0.30) < 1e-12 and REQT is False, \
        f"frozen gate is not the expected one: {SIMF}, {JACF}, {REQT}"
    say(f"  FROZEN stricter gate (validation-selected, pre-registered "
        f"{S['prereg_sha256'][:16]}...)")
    say(f"    similarity floor            {SIMF}   (published {exp._EQ_SIM_FLOOR})")
    say(f"    Jaccard floor               {JACF}   (published {exp._EQ_JACCARD_MIN})")
    say(f"    answer-type agreement required  {REQT}")
    say(f"    selected on validation only; validation L1 {fr['l1_hits']} of "
        f"{S['validation_baseline']['l1_hits']} "
        f"({100*fr['l1_hits']/S['validation_baseline']['l1_hits']:.1f}%)")
    say(f"    NO threshold is chosen here using held-out results.")

    # ---------------- data ----------------
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
    assert len(full) == 31201
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(stream) == split["test"]["requests"]
    assert not [1 for _, r in stream
                if (r.get("cluster_id") or r["query_id"]) in val_c]
    say(f"\n  held-out stream {len(stream):,} requests, {len(test_c):,} "
        f"clusters, validation leakage 0")

    # ---------------- arm A: published gate ----------------
    mA, rowsA = p2.replay(stream, {}, "FreshCache")
    bad = [k for k, v in ANCHOR.items()
           if abs(mA[k] - v) > (1e-4 if isinstance(v, float) else 0)]
    if bad:
        say(f"  ANCHOR GATE FAILED on {bad}: { {k: mA[k] for k in ANCHOR} }")
        sys.exit(2)
    say(f"  arm A (published gate) saved {mA['search_saved_pct']:.4f}%  "
        f"L1 {mA['l1_hits']:,}  L2 {mA['l2_hits']:,}  -> reproduces the "
        f"published held-out anchor")

    # ---------------- arm B: stricter gate, genuine sequential replay -------
    OS, OJ = exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN
    exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN = SIMF, JACF
    try:
        mB, rowsB = p2.replay(stream, {}, "FreshCache")
    finally:
        exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN = OS, OJ
    assert exp._EQ_SIM_FLOOR == OS and exp._EQ_JACCARD_MIN == OJ
    say(f"  arm B (stricter gate)  saved {mB['search_saved_pct']:.4f}%  "
        f"L1 {mB['l1_hits']:,}  L2 {mB['l2_hits']:,}")
    sl = json.load(open(RCI / "01_l1_analysis" / "stricter_l1_results.json",
                        encoding="utf-8"))["heldout_test"]
    assert mB["l1_hits"] == sl["l1_hits"], \
        f"sequential replay L1 {mB['l1_hits']} != the ablation's {sl['l1_hits']}"
    say(f"    cross-check: matches the pre-registered ablation's held-out "
        f"L1 count ({sl['l1_hits']})")

    # prep2 does not return a generation count; it is search_calls + l2_hits,
    # exactly as mixed_engine defines it (an L2 hit regenerates, an L1 hit
    # does not).
    for _m in (mA, mB):
        _m["generations"] = _m["search_calls"] + _m["l2_hits"]

    # ---------------- where do the rejected hits go? ----------------
    tierA = {q: v["tier"] for q, v in rowsA.items()}
    tierB = {q: v["tier"] for q, v in rowsB.items()}
    moved = Counter()
    for q, ta in tierA.items():
        tb = tierB.get(q)
        if ta == "L1" and tb != "L1":
            moved[tb] += 1
    gainedL1 = [q for q in tierB if tierB[q] == "L1" and tierA.get(q) != "L1"]
    say(f"\n  == displacement ==")
    say(f"    L1 hits under A {sum(1 for v in tierA.values() if v=='L1'):,}   "
        f"under B {sum(1 for v in tierB.values() if v=='L1'):,}")
    say(f"    requests that were L1 under A but NOT under B: {sum(moved.values()):,}")
    for k, v in moved.most_common():
        say(f"      -> {str(k):<12}{v:>6,}")
    say(f"    requests that are L1 under B but were NOT under A: "
        f"{len(gainedL1):,}  (cache-state divergence, not filtering)")
    say(f"    searches   A {mA['search_calls']:,}   B {mB['search_calls']:,}   "
        f"delta {mB['search_calls']-mA['search_calls']:+,}")
    say(f"    fetches    A {mA['fetches']:,}   B {mB['fetches']:,}   "
        f"delta {mB['fetches']-mA['fetches']:+,}")
    say(f"    generations A {mA['generations']:,}   B {mB['generations']:,}   "
        f"delta {mB['generations']-mA['generations']:+,}   "
        f"<- the exact generation-cost penalty")

    # ---------------- per-request trace ----------------
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    fcof = {r["query_id"]: r["freshness_class"] for r in records}
    with open(HERE / "per_request.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "cluster_id", "freshness_class", "t",
                    "tier_published_gate", "tier_strict_gate",
                    "sim_published", "sim_strict", "moved"])
        for t, r in stream:
            q = r["query_id"]
            a, b = rowsA.get(q, {}), rowsB.get(q, {})
            w.writerow([q, cid[q], fcof.get(q), t, a.get("tier"), b.get("tier"),
                        a.get("similarity"), b.get("similarity"),
                        int(a.get("tier") != b.get("tier"))])

    # ---------------- realised L1 hits, both arms ----------------
    gold = json.load(open(p2.GOLD, encoding="utf-8"))
    def goldof(qid, qt):
        g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower())
        return (g.get("answer") if isinstance(g, dict) else g)
    qtext = {r["query_id"]: r["query"] for r in records}
    qpara = {r["query_id"]: bool(r.get("is_paraphrase")) for r in records}
    clus_of_query = {}
    for r in records:
        clus_of_query.setdefault(r["query"], r.get("cluster_id") or r["query_id"])

    tasks, jury, hits = {}, [], []
    for arm, rows_ in (("published", rowsA), ("strict", rowsB)):
        for t, r in stream:
            q = r["query_id"]
            p = rows_.get(q)
            if not p or p["tier"] != "L1":
                continue
            inc, mq = r["query"], p["matched_query"]
            sim = p.get("similarity")
            jac = jaccard(inc, mq)
            at_i = exp._eq_answer_type(inc)
            at_c = exp._eq_answer_type(mq)
            cl_i = cid[q]
            cl_c = clus_of_query.get(mq)
            sctx = ctx_for(tuple(p["stored_answer_ev"]))
            sq = p["stored_answer_query"]
            own = [u["url_hash"] for u in r["urls"]]
            fctx = ctx_for(tuple((u, t) for u in own))
            skey = ("stored", h(sctx), h(sq))
            fkey = ("fresh", h(fctx), h(inc))
            tasks[skey] = (sq, sctx)
            if fctx:
                tasks[fkey] = (inc, fctx)
            rec = {"arm": arm, "query_id": q, "cluster_id": cl_i,
                   "incoming_query": inc, "serving_cached_query": mq,
                   "stored_answer_query": sq,
                   "similarity": sim, "jaccard": round(jac, 4),
                   "answer_type_incoming": at_i, "answer_type_cached": at_c,
                   "answer_type_match": (at_i == at_c),
                   "entities_incoming": "|".join(ents(inc)),
                   "entities_cached": "|".join(ents(mq)),
                   "entity_match": bool(exp._entity_match(inc, mq)),
                   "freshness_class": r["freshness_class"],
                   "cache_age_seconds": p.get("age"),
                   "cross_cluster": int(cl_c != cl_i),
                   "is_paraphrase": int(qpara.get(q, False)),
                   "reference_answer": goldof(q, inc),
                   "stored_key": "|".join(skey),
                   "fresh_key": "|".join(fkey) if fctx else "",
                   "fresh_context_available": int(bool(fctx))}
            hits.append(rec)
            jury.append({"arm": arm, "query_id": q, "incoming": inc,
                         "cached": mq})
    with open(HERE / "l1_hits.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hits[0]))
        w.writeheader(); w.writerows(hits)
    # jury tasks: dedupe by (incoming, cached) pair, judged once, reused by arm
    seen, jt = set(), []
    for j in jury:
        k = (h(j["incoming"]), h(j["cached"]))
        if k in seen:
            continue
        seen.add(k)
        jt.append({"pair_id": f"{k[0]}_{k[1]}", "incoming": j["incoming"],
                   "cached": j["cached"]})
    with open(HERE / "jury_tasks.jsonl", "w", encoding="utf-8") as fh:
        for j in jt:
            fh.write(json.dumps(j) + "\n")
    with open(HERE / "gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    nA = sum(1 for x in hits if x["arm"] == "published")
    nB = sum(1 for x in hits if x["arm"] == "strict")
    say(f"\n  realised L1 hits captured: published {nA:,}   strict {nB:,}")
    say(f"  distinct (incoming, cached) pairs to judge: {len(jt):,}")
    say(f"  unique generation tasks: {len(tasks):,}")

    # ---------------- 400-request audit, both arms ----------------
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
    per_s = max(1, TARGET // max(len(strata), 1))
    sample = []
    for k in sorted(strata):
        g = strata[k][:]
        rng.shuffle(g)
        sample.extend(g[:per_s])
    if len(sample) < TARGET:
        chosen = {d["qid"] for d in sample}
        rest = [d for d in frame if d["qid"] not in chosen]
        rng.shuffle(rest)
        sample.extend(rest[:TARGET - len(sample)])
    sample = sample[:TARGET]
    pub = json.load(open(RCI / "06_stronger_generator" / "sample.json",
                         encoding="utf-8"))
    assert [d["qid"] for d in sample] == [r["query_id"] for r in pub["requests"]], \
        "SAMPLE GATE FAILED: the 400-request sample differs from the V3 audit"
    say(f"  SAMPLE GATE PASSED: the 400 audit request IDs are identical to the "
        f"existing held-out audit")

    audit, atasks = [], {}
    comp = {"published": Counter(), "strict": Counter()}
    for d in sample:
        q, t = d["qid"], d["t"]
        rec = {"query_id": q, "cluster_id": d["cluster"], "query": d["query"],
               "gold": d["gold"], "fc": d["fc"], "t": t}
        fctx = ctx_for(tuple((u, t) for u in rowsA[q]["own"]))
        fk = ("fresh", h(fctx), h(d["query"]))
        atasks[fk] = (d["query"], fctx)
        rec["fresh_key"] = list(fk)
        for arm, rows_ in (("published", rowsA), ("strict", rowsB)):
            p = rows_[q]
            comp[arm][p["tier"]] += 1
            rec[f"{arm}_tier"] = p["tier"]
            rec[f"{arm}_age"] = p.get("age", 0.0)
            rec[f"{arm}_similarity"] = p.get("similarity")
            if p["tier"] == "L1":
                sctx = ctx_for(tuple(p["stored_answer_ev"]))
                sq = p["stored_answer_query"]
                k = ("stored", h(sctx), h(sq))
                atasks[k] = (sq, sctx)
                rec[f"{arm}_mode"] = "stored_answer"
                rec[f"{arm}_gen_query"] = sq
                rec[f"{arm}_key"] = list(k)
            else:
                pctx = ctx_for(tuple(p["ev"]))
                k = ("gen", h(pctx), h(d["query"]))
                atasks[k] = (d["query"], pctx)
                rec[f"{arm}_mode"] = ("regenerate_l2" if p["tier"] == "L2"
                                      else "regenerate_miss")
                rec[f"{arm}_gen_query"] = d["query"]
                rec[f"{arm}_key"] = list(k)
        audit.append(rec)
    for k, v in atasks.items():
        tasks.setdefault(k, v)
    with open(HERE / "gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    say(f"  audit composition published {dict(comp['published'])}")
    say(f"  audit composition strict    {dict(comp['strict'])}")
    diff = sum(1 for r in audit if r["published_key"] != r["strict_key"])
    say(f"  audit requests whose EXECUTION PATH differs between arms: {diff}")
    json.dump({"target": TARGET, "seed": SEED,
               "composition": {k: dict(v) for k, v in comp.items()},
               "paths_differ": diff, "requests": audit},
              open(HERE / "audit_sample.json", "w"), indent=2)

    # reuse answers already generated by the existing V3 audit
    prev = {}
    for f in ("answers_llama3b.jsonl",):
        p_ = RCI / "06_stronger_generator" / f
        if p_.exists():
            for l in open(p_, encoding="utf-8"):
                if l.strip():
                    d = json.loads(l)
                    prev[tuple(d["key"])] = d["answer"]
    p_ = RCI / "01_l1_analysis" / "l1_answers.jsonl"
    if p_.exists():
        for l in open(p_, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                prev[tuple(d["key"])] = d["answer"]
    reuse = {k: v for k, v in prev.items() if k in tasks}
    with open(HERE / "answers_llama3b.jsonl", "w", encoding="utf-8") as fh:
        for k, v in reuse.items():
            fh.write(json.dumps({"key": list(k), "answer": v}) + "\n")
    say(f"  generation tasks {len(tasks):,}; reusable from existing runs "
        f"{len(reuse):,}; new {len(tasks)-len(reuse):,}")

    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post, "implementation files changed"
    say(f"  implementation files unchanged: True")

    json.dump({"frozen_gate": {"sim_floor": SIMF, "jac_floor": JACF,
                               "require_answer_type": REQT,
                               "prereg_sha256": S["prereg_sha256"]},
               "arm_published": mA, "arm_strict": mB,
               "displacement": {"lost_l1": dict(moved),
                                "gained_l1": len(gainedL1),
                                "delta_searches": mB["search_calls"]-mA["search_calls"],
                                "delta_fetches": mB["fetches"]-mA["fetches"],
                                "delta_generations": mB["generations"]-mA["generations"]},
               "l1_hits_published": nA, "l1_hits_strict": nB,
               "jury_pairs": len(jt), "generation_tasks": len(tasks),
               "reused_answers": len(reuse), "audit_paths_differ": diff,
               "hashes": pre},
              open(HERE / "prep_summary.json", "w"), indent=2)
    (HERE / "prep.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote per_request.csv, l1_hits.csv, jury_tasks.jsonl, "
        "gen_tasks.jsonl, audit_sample.json, prep_summary.json")


if __name__ == "__main__":
    main()
