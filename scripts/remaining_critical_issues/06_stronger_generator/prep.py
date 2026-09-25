#!/usr/bin/env python3
"""
Unified held-out 400-request answer audit -- CPU stage.

Extends the published held-out audit
(validation/heldout_baseline_tuning/answer_audit_k1_16/prep.py) with the two
new policies (FreshCache-L1Only, ExactTTL) while keeping the sampling frame,
the 28-stratum design, the seed, the evidence builder and the context limits
byte-identical. prep2's helpers are imported directly.

REPRODUCTION GATE
  * FreshCache and SemanticTTL replays must reproduce Stage 2 exactly.
  * The 400 sampled query_ids must be IDENTICAL to the published sample.json.
  * Every (fresh / fc / sttl) generation key must be identical to the published
    one, so the published Llama-3.2-3B answers can be reused byte-for-byte and
    the stronger-generator comparison is strictly paired.

Writes only into 06_stronger_generator/. No web calls, no GPU.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, random, sys, time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
PUB = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"
sys.path.insert(0, str(BASE / "lib"))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import engine_all as ea               # noqa: E402
import schedules as sc                # noqa: E402
import mixed_engine as me             # noqa: E402
from e1_robustness import support_of  # noqa: E402
import prep2 as p2                    # noqa: E402
import v3_engine as v3                # noqa: E402

SCHEDULE, SEED, TARGET = "zipf_uniform", 42, 400
band, ctx_for, h = p2.band, p2.ctx_for, p2.h
ARMS = ["fresh", "fc", "sttl", "l1only", "exactttl"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("V3 UNIFIED HELD-OUT ANSWER AUDIT -- prep (CPU)")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    v3.register()

    STAGE = ROOT / "validation" / "heldout_baseline_tuning"
    split = json.load(open(STAGE / "split.json", encoding="utf-8"))
    sel = json.load(open(STAGE / "validation_selection.json", encoding="utf-8"))
    test_res = json.load(open(STAGE / "test_results.json", encoding="utf-8"))
    c = sel["selection"]["SemanticTTL"]["savings_matched"]
    theta, kappa = c["theta"], c["kappa"]
    assert abs(kappa - 1/16) < 1e-12
    say(f"  frozen SemanticTTL: theta={theta} kappa={kappa}")

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

    test_c, val_c = set(split["test_clusters"]), set(split["validation_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    assert len(full) == 31201
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(stream) == split["test"]["requests"]
    assert not [1 for _, r in stream
                if (r.get("cluster_id") or r["query_id"]) in val_c]
    S = support_of(stream, rounds)
    say(f"  test stream {len(stream):,} requests, |S| = {len(S):,}, leakage 0")

    # ---------- policy traces ----------
    ORIG_TTL, ORIG_T = dict(exp.FIXED_TTL), exp.L1_SIM_THRESHOLD
    mf, fcrows = p2.replay(stream, {}, "FreshCache")
    e = test_res["results"]["FreshCache"]
    assert abs(mf["search_saved_pct"] - e["search_saved_pct"]) < 1e-6 \
        and mf["l1_hits"] == e["l1_hits"] and mf["l2_hits"] == e["l2_hits"]
    say(f"  FreshCache        saved {mf['search_saved_pct']:.4f}%  "
        f"L1 {mf['l1_hits']:,}  L2 {mf['l2_hits']:,}  -> reproduces Stage 2")

    v3.assert_fidelity(stream, rounds)
    say("  fidelity gate PASSED: transcribed L1 cache reproduces mixed_engine "
        "SemanticTTL and FreshCache_L1Only exactly")
    ml, _mlper, l1rows = v3.replay_l1cache(stream, rounds, "c18", "risk")
    say(f"  FreshCache_L1Only saved {ml['search_saved_pct']:.4f}%  "
        f"L1 {ml['l1_hits']:,}  L2 {ml['l2_hits']:,}")
    assert ml["l2_hits"] == 0 and ml["l3_hits"] == 0

    ttl = {k: v * kappa for k, v in ORIG_TTL.items()}
    exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ttl)
    exp.L1_SIM_THRESHOLD = theta
    try:
        ms, strows, _ = None, None, None
        _m, _p, strows = v3.replay_l1cache(stream, rounds, "sim", "fixed")
        ms = _m
        _me, _pe, exrows = v3.replay_l1cache(stream, rounds, "exact", "fixed")
    finally:
        exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG_TTL)
        exp.L1_SIM_THRESHOLD = ORIG_T
    s2 = [v for v in test_res["results"].values()
          if v["method"] == "SemanticTTL" and abs(v["kappa"] - kappa) < 1e-12
          and abs(v["theta"] - theta) < 1e-12][0]
    assert abs(ms["search_saved_pct"] - s2["search_saved_pct"]) < 1e-6 \
        and ms["l1_hits"] == s2["l1_hits"], f"{ms} vs {s2}"
    say(f"  SemanticTTL       saved {ms['search_saved_pct']:.4f}%  "
        f"L1 {ms['l1_hits']:,}  -> reproduces Stage 2")
    say(f"  ExactTTL          saved {_me['search_saved_pct']:.4f}%  "
        f"L1 {_me['l1_hits']:,}")

    # ---------- sampling frame (identical construction) ----------
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
        qid = r["query_id"]
        if r["freshness_class"] == "REAL_TIME":
            continue
        g = goldof(qid, r["query"])
        if not g:
            continue
        own = [u["url_hash"] for u in r["urls"]]
        if not ctx_for(tuple((u, t) for u in own)):
            continue
        frame.append({"qid": qid, "t": t, "query": r["query"], "gold": g,
                      "fc": r["freshness_class"], "simbase": simbase.get(qid),
                      "cluster": r.get("cluster_id") or qid, "in_S": qid in S})
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
    assert len(sample) == len({d["qid"] for d in sample}) == TARGET

    pub = json.load(open(PUB / "sample.json", encoding="utf-8"))
    pub_ids = [r["query_id"] for r in pub["requests"]]
    assert [d["qid"] for d in sample] == pub_ids, \
        "SAMPLE GATE FAILED: sampled request set differs from the published audit"
    say(f"\n  frame {len(frame):,}  strata {len(strata)}  cap {per}  "
        f"sampled {len(sample)}")
    say("  SAMPLE GATE PASSED: the 400 request IDs are identical to the "
        "published held-out audit, in the same order")

    # ---------- arms ----------
    rowsets = {"fc": fcrows, "sttl": strows, "l1only": l1rows, "exactttl": exrows}
    out, tasks = [], {}
    comp = {a: Counter() for a in rowsets}
    for d, pubrec in zip(sample, pub["requests"]):
        qid, t = d["qid"], d["t"]
        fctx = ctx_for(tuple((u, t) for u in fcrows[qid]["own"]))
        rec = {"query_id": qid, "cluster_id": d["cluster"], "query": d["query"],
               "gold": d["gold"], "fc": d["fc"], "t": t, "in_S": d["in_S"],
               "simbase": d["simbase"], "simbase_band": band(d["simbase"]),
               "fresh_ctx_sha": h(fctx), "fresh_pages": len(fcrows[qid]["own"])}
        tasks[("fresh", h(fctx), h(d["query"]))] = (d["query"], fctx)
        rec["fresh_key"] = ["fresh", h(fctx), h(d["query"])]
        assert rec["fresh_key"] == pubrec["fresh_key"], f"fresh key drift {qid}"
        for tag, rowset in rowsets.items():
            p = rowset[qid]
            tier = p["tier"]
            comp[tag][tier] += 1
            rec[f"{tag}_tier"] = tier
            rec[f"{tag}_age"] = p.get("age", 0.0)
            rec[f"{tag}_similarity"] = p.get("similarity")
            if tier in ("L1", "HIT"):
                sctx = ctx_for(tuple(p["stored_answer_ev"]))
                sq = p["stored_answer_query"]
                key = ("stored", h(sctx), h(sq))
                tasks[key] = (sq, sctx)
                rec[f"{tag}_mode"] = "stored_answer"
                rec[f"{tag}_gen_query"] = sq
                rec[f"{tag}_matched_query"] = p["matched_query"]
                rec[f"{tag}_ctx_sha"] = h(sctx)
                rec[f"{tag}_key"] = list(key)
                rec[f"{tag}_stored_answer_t"] = p["stored_answer_t"]
            else:
                pctx = ctx_for(tuple(p["ev"]))
                key = ("gen", h(pctx), h(d["query"]))
                tasks[key] = (d["query"], pctx)
                rec[f"{tag}_mode"] = ("regenerate_l2" if tier == "L2"
                                      else "regenerate_miss")
                rec[f"{tag}_gen_query"] = d["query"]
                rec[f"{tag}_ctx_sha"] = h(pctx)
                rec[f"{tag}_key"] = list(key)
                rec[f"{tag}_l3_reused_pages"] = p.get("l3_reused_pages", 0)
                rec[f"{tag}_ctx_equals_fresh"] = (h(pctx) == h(fctx))
            if tag in ("fc", "sttl"):
                assert rec[f"{tag}_key"] == pubrec[f"{tag}_key"], \
                    f"{tag} key drift at {qid}"
        out.append(rec)
    say("  KEY GATE PASSED: every fresh/fc/sttl generation key matches the "
        "published audit -> published Llama-3.2-3B answers are reusable verbatim")

    pubans = {tuple(json.loads(l)["key"]): json.loads(l)["answer"]
              for l in open(PUB / "answers.jsonl", encoding="utf-8") if l.strip()}
    reuse = sum(1 for k in tasks if k in pubans)
    say(f"\n  unique generation tasks {len(tasks):,}  "
        f"({reuse:,} already answered by the published Llama-3.2-3B run, "
        f"{len(tasks)-reuse:,} new)")
    for a in rowsets:
        say(f"  composition {a:<10} {dict(comp[a])}")

    json.dump({"target": TARGET, "seed": SEED, "arms": ARMS,
               "sample_gate": "identical to published answer_audit_k1_16",
               "frozen_semanticttl": {"theta": theta, "kappa": kappa},
               "policy_metrics": {"FreshCache": mf, "FreshCache_L1Only": ml,
                                  "SemanticTTL": ms, "ExactTTL": _me},
               "composition": {k: dict(v) for k, v in comp.items()},
               "unique_generation_tasks": len(tasks),
               "reusable_published_answers": reuse,
               "max_context_chars": p2.MAXCHARS, "min_page_chars": p2.MINCHARS,
               "requests": out},
              open(HERE / "sample.json", "w"), indent=2)
    with open(HERE / "gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")
    # seed the 3B answer file with the published answers, byte-for-byte
    with open(HERE / "answers_llama3b.jsonl", "w", encoding="utf-8") as fh:
        for k in tasks:
            if k in pubans:
                fh.write(json.dumps({"key": list(k), "answer": pubans[k]}) + "\n")
    (HERE / "prep.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote sample.json, gen_tasks.jsonl, answers_llama3b.jsonl, prep.log")


if __name__ == "__main__":
    main()
