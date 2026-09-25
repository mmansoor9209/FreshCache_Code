#!/usr/bin/env python3
"""
validation/mixed_age_full_policy_audit/prep2.py

Stage 1 of the CORRECTED paired mixed-age full-policy answer audit.

Fixes the two defects found in the previous 400-request audit:

  1. Sampling no longer conditions on the policy outcome. Requests are drawn
     from a policy-neutral frame (non-REAL_TIME, gold present, non-empty fresh
     context) BEFORE any gate-on/gate-off hit or miss is consulted. Misses are
     kept.
  2. The real execution path is followed per tier:
       L1 hit  -> the STORED answer is returned. That answer is the one produced
                  when the entry was created, from the creating request's own
                  query and the evidence served to it at that time. It is never
                  regenerated against the incoming query.
       L2 hit  -> regenerate from the incoming query on the actual per-page L3
                  evidence (reused pages at their L3 cache time, refetched pages
                  at the request time).
       miss    -> the engine searches, then still reuses any L3 page that is
                  cached and passes the content temporal test. The evidence is
                  therefore the request's own URLs at their ACTUAL content
                  versions, which equals the pure fresh context only when no
                  page was reused. Both cases are handled exactly.

Because an L1 write-back happens only on requests that were NOT L1 hits, every
stored answer is grounded in evidence and never in another stored answer. The
recursion terminates at depth one, so no iterative generation is needed.

Timestamp discipline: every page is read at version_at of its own content time,
which is always at or before the request time. No future snapshot can be seen.

Read-only on all existing artifacts. Writes only into this folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import re
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import corrected_engine as ce         # noqa: E402
import engine_all as ea               # noqa: E402
import schedules as sc                # noqa: E402
import mixed_engine as me             # noqa: E402
import mixed_age_v2 as ma             # noqa: E402
import e2e_answer_grading as e2e      # noqa: E402
from e1_robustness import support_of  # noqa: E402

SCHEDULE, SEED = "zipf_uniform", 42
TARGET, MAXCHARS, MINCHARS = 400, 2800, 200
GOLD = ROOT / "v10_remaining_feedback" / "c2_gold_answers.json"
ARMS = {"FreshCache": "on", "FreshCache_AlwaysPassTemporal": "off"}
EXPECT = {"FreshCache": {"saved": 62.7320, "l1": 1175, "l2": 18398},
          "FreshCache_AlwaysPassTemporal": {"saved": 80.3500, "l1": 2895,
                                            "l2": 22175}}


def band(s):
    if s is None:
        return "base"
    for a, b in [(0.0, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
                 (0.90, 0.95), (0.95, 1.01)]:
        if a <= s < b:
            return f"{a:.2f}-{b:.2f}"
    return "other"


def replay(stream, rounds, variant):
    """mixed_engine.replay's FreshCache path with full per-request provenance.

    Additionally records, for every L1 entry, the (query, evidence) pair that
    produced the answer stored in it, so an L1 hit can return that stored answer
    rather than regenerating.
    """
    cfg = ea.CONFIG["FreshCache_Full"]
    temporal = me.VARIANTS[variant][1]
    L1T = exp.L1_SIM_THRESHOLD
    L2FLOOR = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
    EPS = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
           "content": exp.EPS_CONTENT}
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    n_max = len(stream)

    def ok(fc, age, tier):
        if temporal == "always":
            return True
        return exp.p_stale(fc, age, tier) <= EPS[tier]

    l1_qi = np.empty(n_max, np.int64); l1_t = np.empty(n_max, np.float64)
    l1_q, l1_fc, l1_urls = [], [], []
    l1_src = []           # per entry: {"query","ev","t"} of the creating request
    n1 = 0
    l2_qi = np.empty(n_max, np.int64); l2_t = np.empty(n_max, np.float64)
    l2_fc, l2_q, l2_urls = [], [], []
    n2 = 0
    l3_cache = {}
    l1h = l2h = l3h = search = fetches = 0
    rows = {}

    for t, r in stream:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        qi = IDX[q]
        row = SIM[qi]
        own = [u["url_hash"] for u in urls]

        if fc == "REAL_TIME":
            search += 1
            for h in own:
                fetches += 1
                l3_cache[h] = t
            rows[qid] = {"tier": "REAL_TIME", "t": t,
                         "ev": [(u, t) for u in own], "own": own}
            continue

        hit1 = -1
        sim1 = None
        if n1:
            sims = row[l1_qi[:n1]]
            cand = np.nonzero(sims >= L1T)[0]
            if cand.size:
                okm = np.fromiter((ok(l1_fc[int(j)], t - l1_t[j], "answer")
                                   for j in cand), bool, cand.size)
                keep = cand[okm]
                for j in keep[np.argsort(-sims[keep], kind="stable")]:
                    j = int(j)
                    if not (exp._entity_match(q, l1_q[j])
                            and exp.semantic_equivalent(q, l1_q[j],
                                                        float(sims[j]))):
                        continue
                    hit1 = j
                    sim1 = float(sims[j])
                    break
        if hit1 >= 0:
            l1h += 1
            src = l1_src[hit1]
            rows[qid] = {"tier": "L1", "t": t, "similarity": sim1,
                         "served": list(l1_urls[hit1]),
                         "t_cached": float(l1_t[hit1]),
                         "age": t - float(l1_t[hit1]),
                         "matched_query": l1_q[hit1],
                         "stored_answer_query": src["query"],
                         "stored_answer_ev": src["ev"],
                         "stored_answer_t": src["t"],
                         "own": own}
            continue

        hit2 = -1
        sim2 = None
        if n2:
            s2 = row[l2_qi[:n2]]
            cand = np.nonzero(s2 >= L2FLOOR)[0]
            if cand.size:
                okm = np.fromiter((ok(l2_fc[int(j)], t - l2_t[j], "url_list")
                                   for j in cand), bool, cand.size)
                keep = cand[okm]
                if keep.size:
                    hit2 = int(keep[np.argmax(s2[keep])])
                    sim2 = float(s2[hit2])

        if hit2 >= 0:
            l2h += 1
            served = list(l2_urls[hit2])
            disc = float(l2_t[hit2])
            l2_qi[n2] = qi; l2_t[n2] = disc
            l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
        else:
            search += 1
            served = list(own)
            disc = t
            l2_qi[n2] = qi; l2_t[n2] = t
            l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1

        # L3: a page is reused at its own cache time, otherwise refetched at t.
        # This happens on BOTH the L2-hit and the L2-miss path.
        ev, n_reused = [], 0
        for x in served:
            if x in l3_cache and ok(fc, t - l3_cache[x], "content"):
                l3h += 1
                n_reused += 1
                ev.append((x, l3_cache[x]))
                continue
            fetches += 1
            l3_cache[x] = t
            ev.append((x, t))

        rows[qid] = {"tier": ("L2" if hit2 >= 0 else "miss"), "t": t,
                     "similarity": sim2, "served": served, "t_cached": disc,
                     "age": (t - disc) if hit2 >= 0 else 0.0,
                     "ev": ev, "l3_reused_pages": n_reused, "own": own}

        l1_qi[n1] = qi; l1_t[n1] = t
        l1_q.append(q); l1_fc.append(fc); l1_urls.append(served)
        l1_src.append({"query": q, "ev": ev, "t": t})
        n1 += 1

    m = {"search_saved_pct": round(100 * (1 - search / len(stream)), 4),
         "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "fetches": fetches,
         "search_calls": search}
    return m, rows


_CTX = {}


def ctx_for(pairs):
    key = tuple(pairs)
    if key in _CTX:
        return _CTX[key]
    parts, total = [], 0
    for uh, tv in pairs:
        txt = e2e.load_snapshot_text(uh, ma.version_at(tv)) or ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) < MINCHARS:
            continue
        room = MAXCHARS - total
        if room <= 0:
            break
        parts.append(txt[:room])
        total += min(len(txt), room)
    out = "\n\n".join(parts)
    _CTX[key] = out
    return out


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def main():
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
    stream = sc.build_stream(records, SCHEDULE, SEED)
    S = support_of(stream, rounds)
    gold = json.load(open(GOLD, encoding="utf-8"))

    def goldof(qid, qt):
        g = gold.get(qid) or gold.get(qt) or gold.get((qt or "").strip().lower())
        return (g.get("answer") if isinstance(g, dict) else g)

    # policy-neutral similarity anchor: query vs its own cluster base
    base = ea._CLUSTER_BASE
    IDX, SIM = exp._QUERY_TO_IDX, exp._SIM_MATRIX
    simbase = {}
    for _, r in stream:
        qid = r["query_id"]
        bq = base.get(qid, "")
        simbase[qid] = (float(SIM[IDX[r["query"]], IDX[bq]])
                        if r.get("is_paraphrase") and bq and bq in IDX else None)

    arms = {}
    for variant in ARMS:
        m, rows = replay(stream, rounds, variant)
        e = EXPECT[variant]
        bad = [k for k, v in (("saved", m["search_saved_pct"]),
                              ("l1", m["l1_hits"]), ("l2", m["l2_hits"]))
               if abs(v - e[k]) > 1e-6]
        print(f"  {variant:<32} saved {m['search_saved_pct']:.4f}%  "
              f"L1 {m['l1_hits']:,}  L2 {m['l2_hits']:,}  "
              f"{'reproduces published' if not bad else 'MISMATCH ' + str(bad)}",
              flush=True)
        if bad:
            raise SystemExit("  ABORT: replay does not reproduce the published "
                             "aggregates.")
        arms[variant] = rows

    # ---------- policy-neutral sampling frame ----------
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
                      "in_S": qid in S})
    print(f"\n  policy-neutral frame: {len(frame):,} "
          f"(non-REAL_TIME, gold present, non-empty fresh context)", flush=True)

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
    print(f"  strata {len(strata)}  per-stratum cap {per}  sampled {len(sample)}"
          f"  unique {len(set(d['qid'] for d in sample))}", flush=True)

    # ---------- build the three arms ----------
    out, tasks = [], {}
    comp = {"on": Counter(), "off": Counter()}
    for d in sample:
        qid, t = d["qid"], d["t"]
        fresh_pairs = tuple((u, t) for u in arms["FreshCache"][qid]["own"])
        fctx = ctx_for(fresh_pairs)
        rec = {"query_id": qid, "query": d["query"], "gold": d["gold"],
               "fc": d["fc"], "t": t, "in_S": d["in_S"],
               "simbase_band": band(d["simbase"]),
               "fresh_ctx_sha": h(fctx), "fresh_ctx": fctx}
        tasks[("fresh", h(fctx), h(d["query"]))] = (d["query"], fctx)
        for variant, tag in ARMS.items():
            p = arms[variant][qid]
            tier = p["tier"]
            comp[tag][tier] += 1
            rec[f"{tag}_tier"] = tier
            rec[f"{tag}_age"] = p.get("age", 0.0)
            rec[f"{tag}_similarity"] = p.get("similarity")
            if tier == "L1":
                # real path: return the STORED answer, generated at entry
                # creation from the creating request's query and evidence
                sctx = ctx_for(tuple(p["stored_answer_ev"]))
                sq = p["stored_answer_query"]
                key = ("stored", h(sctx), h(sq))
                tasks[key] = (sq, sctx)
                rec[f"{tag}_mode"] = "stored_answer"
                rec[f"{tag}_gen_query"] = sq
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
        out.append(rec)

    with open(HERE / "sample400.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    with open(HERE / "gen_tasks.jsonl", "w", encoding="utf-8") as fh:
        for k, (qq, cc) in tasks.items():
            fh.write(json.dumps({"key": list(k), "question": qq,
                                 "context": cc}) + "\n")

    eqfresh = {tag: sum(1 for r in out if r.get(f"{tag}_ctx_equals_fresh"))
               for tag in ("on", "off")}
    json.dump({"frame_size": len(frame), "sampled": len(sample),
               "target": TARGET, "seed": SEED, "n_strata": len(strata),
               "per_stratum_cap": per,
               "sampling_is_policy_neutral": True,
               "composition": {t: dict(c) for t, c in comp.items()},
               "unique_generation_tasks": len(tasks),
               "in_support_S": sum(1 for r in out if r["in_S"]),
               "policy_ctx_equals_fresh_ctx": eqfresh,
               "max_context_chars": MAXCHARS, "min_page_chars": MINCHARS},
              open(HERE / "sample_manifest.json", "w"), indent=2)

    print(f"\n  composition on : {dict(comp['on'])}")
    print(f"  composition off: {dict(comp['off'])}")
    print(f"  unique generation tasks: {len(tasks):,} "
          f"(vs {3*len(sample):,} naive)")
    print(f"  policy context identical to fresh context: {eqfresh}")
    print(f"  wrote sample400.jsonl, gen_tasks.jsonl, sample_manifest.json")


if __name__ == "__main__":
    main()
