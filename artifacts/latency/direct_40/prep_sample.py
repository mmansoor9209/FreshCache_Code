#!/usr/bin/env python3
"""
PREP for the direct end-to-end latency validation.

Picks ONE shared sample of representative requests and records, for each of the
four policies, exactly what the engine would do at that point in the trace:

  path            L1 hit / L2 hit / miss / realtime_bypass
  served URLs     with a per-URL decision: served from the L3 cache, or fetched
  l1/l2 index     the live index size at that moment, so the direct harness can
                  search a real index of the right size
  matched query   the cached query an L1 hit would serve (for the entity gate)

The profile pass mirrors mixed_engine.replay and is asserted to reproduce its
aggregates for every policy, exactly as stage A did. CPU only, no web, no GPU.
Reads existing artifacts read-only; writes only into this folder.
"""
from __future__ import annotations

import csv, json, os, pathlib, random, sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import corrected_engine as ce         # noqa: E402
import engine_all as ea               # noqa: E402
import schedules as sc                # noqa: E402
import mixed_engine as me             # noqa: E402

SCHEDULE, SEED, N_SAMPLE = "zipf_uniform", 42, 40
CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]
NOL1, STTL = "FreshCache_NoL1", "SemanticTTL_k1_16"
POLICIES = ["NoCache", STTL, NOL1, "FreshCache"]


def profile(stream, rounds, want, use_l1, use_l2, use_l3, temporal,
            ttl=None, theta=None):
    """Replay; capture full per-request detail for the query_ids in `want`."""
    EPS = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
           "content": exp.EPS_CONTENT}
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = theta if theta is not None else exp.L1_SIM_THRESHOLD
    L2FLOOR = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
    n_max = len(stream)

    def ok(fc, age, tier):
        if temporal == "ttl":
            v = ttl.get(fc, 0)
            return bool(v and age <= v)
        return exp.p_stale(fc, age, tier) <= EPS[tier]

    l1_qi = np.empty(n_max, np.int64); l1_t = np.empty(n_max, np.float64)
    l1_q, l1_fc, l1_urls = [], [], []
    n1 = 0
    l2_qi = np.empty(n_max, np.int64); l2_t = np.empty(n_max, np.float64)
    l2_fc, l2_q, l2_urls = [], [], []
    n2 = 0
    l3_cache = {}
    l1h = l2h = l3h = search = fetches = 0
    cap = {}

    for pos, (t, r) in enumerate(stream):
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        qi = IDX[q]
        row = SIM[qi]
        own = [u["url_hash"] for u in urls]
        want_it = qid in want
        rec = {"query_id": qid, "pos": pos, "query": q, "fc": fc,
               "own_urls": own, "l1_index": n1, "l2_index": n2,
               "matched_query": None, "path": "", "served": []}

        if fc == "REAL_TIME":
            search += 1
            for h in own:
                fetches += 1; l3_cache[h] = t
            rec["path"] = "realtime_bypass"
            rec["served"] = [{"url_hash": h, "from_cache": False} for h in own]
            if want_it:
                cap[qid] = rec
            continue

        hit1 = -1
        if use_l1 and n1:
            sims = row[l1_qi[:n1]]
            cand = np.nonzero(sims >= L1T)[0]
            if cand.size:
                okm = np.fromiter((ok(l1_fc[int(j)], t - l1_t[j], "answer")
                                   for j in cand), bool, cand.size)
                keep = cand[okm]
                for j in keep[np.argsort(-sims[keep], kind="stable")]:
                    j = int(j)
                    if temporal == "risk" and not (
                            exp._entity_match(q, l1_q[j])
                            and exp.semantic_equivalent(q, l1_q[j],
                                                        float(sims[j]))):
                        continue
                    hit1 = j
                    break
        if hit1 >= 0:
            l1h += 1
            rec["path"] = "L1"; rec["matched_query"] = l1_q[hit1]
            if want_it:
                cap[qid] = rec
            continue

        hit2 = -1
        if use_l2 and n2:
            s2 = row[l2_qi[:n2]]
            cand = np.nonzero(s2 >= L2FLOOR)[0]
            if cand.size:
                okm = np.fromiter((ok(l2_fc[int(j)], t - l2_t[j], "url_list")
                                   for j in cand), bool, cand.size)
                keep = cand[okm]
                if keep.size:
                    hit2 = int(keep[np.argmax(s2[keep])])
        if use_l2:
            if hit2 >= 0:
                l2h += 1
                served = list(l2_urls[hit2]); disc = float(l2_t[hit2])
                rec["path"] = "L2"; rec["matched_query"] = l2_q[hit2]
                l2_qi[n2] = qi; l2_t[n2] = disc
                l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
            else:
                search += 1; served = list(own); rec["path"] = "miss"
                l2_qi[n2] = qi; l2_t[n2] = t
                l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
        else:
            search += 1; served = list(own); rec["path"] = "miss"

        det = []
        if use_l3:
            for x in served:
                if x in l3_cache and ok(fc, t - l3_cache[x], "content"):
                    l3h += 1; det.append({"url_hash": x, "from_cache": True})
                    continue
                fetches += 1; l3_cache[x] = t
                det.append({"url_hash": x, "from_cache": False})
        else:
            fetches += len(served)
            det = [{"url_hash": x, "from_cache": False} for x in served]
        rec["served"] = det
        if want_it:
            cap[qid] = rec

        if use_l1:
            l1_qi[n1] = qi; l1_t[n1] = t
            l1_q.append(q); l1_fc.append(fc); l1_urls.append(served); n1 += 1

    m = {"search_calls": search, "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h,
         "fetches": fetches, "generations": search + l2h, "n_requests": n_max,
         "search_saved_pct": round(100 * (1 - search / n_max), 4)}
    return m, cap


def main():
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea._rich_feats(); ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    stream = sc.build_stream(records, SCHEDULE, SEED)
    assert len(stream) == 31201

    # sample from the warm second half, stratified by freshness class,
    # and require a real URL so every policy has something to serve/fetch
    rng = random.Random(SEED)
    half = len(stream) // 2
    by = defaultdict(list)
    url_of = {}
    for m in manifest:
        if m.get("run_id") == "run_00" and m.get("url"):
            url_of.setdefault(m["url_hash"], m["url"])
    seen = set()
    for pos in range(half, len(stream)):
        t, r = stream[pos]
        if r["query_id"] in seen or not r["urls"]:
            continue
        if not all(u["url_hash"] in url_of for u in r["urls"]):
            continue
        seen.add(r["query_id"])
        by[r["freshness_class"]].append(r["query_id"])
    per = N_SAMPLE // len(CLASSES)
    want = []
    for c in CLASSES:
        pool = sorted(by[c]); rng.shuffle(pool)
        want.extend(pool[:per])
    want = set(want[:N_SAMPLE])
    print(f"  sample {len(want)} requests, stratified, from the warm second half",
          flush=True)

    base = dict(ea.CONFIG["FreshCache_Full"])
    ea.CONFIG[NOL1] = {**base, "l1": "none"}
    me.VARIANTS[NOL1] = (NOL1, "risk")
    ORIG = dict(exp.FIXED_TTL)
    ttl16 = {k: v / 16 for k, v in ORIG.items()}

    specs, aggr = {}, {}
    cfgs = {
        "FreshCache": dict(use_l1=True, use_l2=True, use_l3=True, temporal="risk"),
        NOL1: dict(use_l1=False, use_l2=True, use_l3=True, temporal="risk"),
        STTL: dict(use_l1=True, use_l2=False, use_l3=False, temporal="ttl",
                   ttl=ttl16, theta=0.40),
    }
    ref_variant = {"FreshCache": "FreshCache", NOL1: NOL1, STTL: "SemanticTTL"}
    for pol, kw in cfgs.items():
        if pol == STTL:
            exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ttl16)
        try:
            ref, _ = me.replay(stream, rounds, ref_variant[pol])
        finally:
            exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG)
        m, cap = profile(stream, rounds, want, **kw)
        for k in ("search_calls", "l1_hits", "l2_hits", "l3_hits", "fetches",
                  "generations"):
            assert m[k] == ref[k], f"{pol}: {k} {m[k]} != {ref[k]}"
        specs[pol] = cap; aggr[pol] = m
        paths = defaultdict(int)
        for v in cap.values():
            paths[v["path"]] += 1
        print(f"  {pol:<20} verified vs mixed_engine   sample paths {dict(paths)}",
              flush=True)

    # NoCache: every request searches and fetches its own URLs
    cap = {}
    for pos, (t, r) in enumerate(stream):
        if r["query_id"] in want and r["query_id"] not in cap:
            cap[r["query_id"]] = {
                "query_id": r["query_id"], "pos": pos, "query": r["query"],
                "fc": r["freshness_class"],
                "own_urls": [u["url_hash"] for u in r["urls"]],
                "l1_index": 0, "l2_index": 0, "matched_query": None,
                "path": "nocache",
                "served": [{"url_hash": u["url_hash"], "from_cache": False}
                           for u in r["urls"]]}
    specs["NoCache"] = cap
    aggr["NoCache"] = {"search_calls": len(stream), "l1_hits": 0, "l2_hits": 0,
                       "l3_hits": 0, "generations": len(stream)}
    print(f"  {'NoCache':<20} analytic", flush=True)

    ids = sorted(want)
    out = {"schedule": SCHEDULE, "seed": SEED, "n_sample": len(ids),
           "sample_query_ids": ids, "url_of": {h: url_of[h] for h in
                                               {u for p in specs.values()
                                                for v in p.values()
                                                for u in v["own_urls"]}
                                               if h in url_of},
           "aggregates": aggr,
           "specs": {p: [specs[p][i] for i in ids if i in specs[p]]
                     for p in POLICIES}}
    json.dump(out, open(HERE / "direct_sample_spec.json", "w"), indent=2)
    for p in POLICIES:
        n = len(out["specs"][p])
        print(f"    {p:<20} {n} request specs captured", flush=True)
    print("  wrote direct_sample_spec.json", flush=True)


if __name__ == "__main__":
    main()
