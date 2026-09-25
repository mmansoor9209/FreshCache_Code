#!/usr/bin/env python3
"""
v9/mixed_age_v2.py — corrected mixed-age replay.

CLOSES FOUR REVIEW ITEMS

N1 (Critical) "The mixed-age rejection rates appear incompatible with the
   reported search savings." The reviewer computes 0.219 + 0.781*0.289 = 44.5%
   maximum savings from the reported 78.1% / 71.1% rejection rates, against a
   reported 62.5%. The reported rates are ENTRY-LEVEL: v7/mixed_age_replay.py
   increments its candidate counter once per cached entry scanned inside the
   lookup loop, so 83,737 L2 candidates arise from 31,201 requests. The
   reviewer's inequality assumes one candidate per request, which is the
   QUERY-LEVEL reading. This run reports both, with every denominator named, so
   the two accounts reconcile.

N3 (High) "The Web-state model for arbitrary replay times is unspecified."
   Defined here as a step function over the five snapshot rounds:
       version_at(t) = the snapshot whose collection age is the largest <= t
       t in [0,1h) -> run_00; [1h,12h) -> rerun_1h; [12h,24h) -> rerun_12h;
       [24h,7d) -> rerun_24h; >=7d -> rerun_7d
   A fetch at t stores version_at(t). A reuse of an entry cached at t_c and read
   at t_r is stale iff hash(version_at(t_c)) != hash(version_at(t_r)), taken from
   v9/version_timeline.jsonl. This is exact per URL and replaces the previous
   cumulative-from-baseline approximation, which is only correct when the cached
   version is the baseline one.

N2 (High) "The mixed-age experiment changes more than the temporal gate."
   Four policies now run on the IDENTICAL stream, so any difference is the
   policy: freshcache (semantic + temporal), no_temporal (semantic only,
   temporal always passes), equivalent_ttl (the temporal condition expressed as
   its derived per-class TTL), and l2only (URL-list reuse alone).

H4 (High) "Semantic drift through repeated L2 aliasing." Every L2 entry carries
   the query whose real search produced the list and a chain depth incremented
   on each re-registration, so reuse-chain length can be crossed against
   evidence suitability.

SAFETY  Read-only on data/ and v7/. Writes only v9/.
"""
from __future__ import annotations

import json, math, pathlib, random, statistics, sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import experiment as exp   # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
SEED, HORIZON, ZIPF_A = 42, 7*86_400, 1.0
RUNS = [("run_00", 0.0), ("rerun_1h", 3_600.0), ("rerun_12h", 43_200.0),
        ("rerun_24h", 86_400.0), ("rerun_7d", 604_800.0)]


def version_at(t):
    """Step function over the five snapshot rounds. See N3 above."""
    v = RUNS[0][0]
    for name, age in RUNS:
        if t >= age:
            v = name
        else:
            break
    return v


def build_stream(records):
    """Arrival model copied from v7/mixed_age_replay.py so the stream matches."""
    rng = random.Random(SEED)
    by_c = defaultdict(list)
    for r in records:
        by_c[r.get("cluster_id") or r["query_id"]].append(r)
    clusters = sorted(by_c)
    weights = [1.0/((i+1)**ZIPF_A) for i in range(len(clusters))]
    tot = sum(weights)
    stream = []
    for cid, w in zip(clusters, weights):
        rows = sorted(by_c[cid], key=lambda r: (1 if r.get("is_paraphrase") else 0,
                                                r["query_id"]))
        base = rng.uniform(0, HORIZON/2)*(1-w/tot) + rng.uniform(0, HORIZON/2)*(w/tot)
        for i, r in enumerate(rows):
            stream.append((base if i == 0 else rng.uniform(base, HORIZON), r))
    stream.sort(key=lambda x: x[0])
    return stream


def ttl_for(fc, tier):
    """EquivalentTTL: the age at which the exponential gate would first refuse."""
    eps = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
           "content": exp.EPS_CONTENT}[tier]
    h = exp.HALF_LIFE.get(fc, 86_400.0)
    m = exp.TIER_MULT[tier]
    return -h*math.log(1-eps)/(m*math.log(2))


def replay(stream, hashes, policy, observable=None):
    """
    policy: 'freshcache' | 'no_temporal' | 'equivalent_ttl' | 'l2only'
    hashes: url_hash -> {run -> content_hash}
    observable: optional set; when given, drift is scored on observable URLs only
    """
    use_l1 = policy != "l2only"
    l1_cache, l2_cache, l3_cache = [], [], {}
    n = len(stream)
    q = Counter()                     # QUERY-level funnel
    e = Counter()                     # ENTRY-level, the v7 semantics
    chain_hits = defaultdict(int)     # chain depth -> L2 hits
    ages = defaultdict(list)
    stale_errors = hits_scored = 0
    l1h = l2h = l3h = search = 0

    def changed(uh, tc, tr):
        h = hashes.get(uh)
        if not h:
            return False
        a, b = h.get(version_at(tc)), h.get(version_at(tr))
        return bool(a and b and a != b)

    def temporal_ok(fc, age, tier):
        if policy == "no_temporal":
            return True
        if policy == "equivalent_ttl":
            return age <= ttl_for(fc, tier)
        eps = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
               "content": exp.EPS_CONTENT}[tier]
        return exp.p_stale(fc, age, tier) <= eps

    for t, r in stream:
        qq, fc, urls = r["query"], r["freshness_class"], r["urls"]
        uh = [u["url_hash"] for u in urls]
        q["requests"] += 1
        if fc == "REAL_TIME":
            q["realtime_bypass"] += 1
            search += 1
            for h in uh:
                l3_cache[h] = t
            continue

        # ---------- L1 ----------
        served = None
        if use_l1:
            q["reached_l1"] += 1
            best, cand_sem, rej_tmp = 0.0, 0, 0
            for en in l1_cache:
                s = exp.cosine_sim(qq, en["query"])
                if s < exp.L1_SIM_THRESHOLD or s <= best:
                    continue
                if not (exp._entity_match(qq, en["query"])
                        and exp.semantic_equivalent(qq, en["query"], s)):
                    continue
                cand_sem += 1
                e["l1_cand"] += 1
                if not temporal_ok(en["fc"], t-en["t"], "answer"):
                    rej_tmp += 1
                    e["l1_rej"] += 1
                    continue
                best, served = s, en
            if cand_sem:
                q["l1_sem_candidate"] += 1
            if served:
                q["l1_served"] += 1
                l1h += 1
                age = t-served["t"]
                ages["l1"].append(age)
                su = [x for x in served["url_hashes"]
                      if observable is None or x in observable]
                if su:
                    hits_scored += 1
                    if any(changed(x, served["t"], t) for x in su):
                        stale_errors += 1
                continue
            if cand_sem and rej_tmp == cand_sem:
                q["l1_temporal_rejected_all"] += 1

        # ---------- L2 ----------
        q["reached_l2"] += 1
        hit2, best, cand_sem, rej_tmp = None, 0.0, 0, 0
        for en in l2_cache:
            s = exp.cosine_sim(qq, en["query"])
            if (s < exp.L2_SIM_THRESHOLD or s < exp._L2_EQ_SIM_FLOOR or s <= best):
                continue
            cand_sem += 1
            e["l2_cand"] += 1
            if not temporal_ok(en["fc"], t-en["t"], "url_list"):
                rej_tmp += 1
                e["l2_rej"] += 1
                continue
            best, hit2 = s, en
        if cand_sem:
            q["l2_sem_candidate"] += 1
        if hit2:
            q["l2_served"] += 1
            l2h += 1
            ages["l2"].append(t-hit2["t"])
            chain_hits[hit2["chain"]] += 1
            serve = hit2["url_hashes"]
            l2_cache.append({"query": qq, "fc": fc, "url_hashes": serve,
                             "t": hit2["t"], "origin": hit2["origin"],
                             "chain": hit2["chain"]+1})
            su = [x for x in serve if observable is None or x in observable]
            if su:
                hits_scored += 1
                if any(changed(x, hit2["t"], t) for x in su):
                    stale_errors += 1
        else:
            if cand_sem and rej_tmp == cand_sem:
                q["l2_temporal_rejected_all"] += 1
            q["searched"] += 1
            search += 1
            serve = uh
            l2_cache.append({"query": qq, "fc": fc, "url_hashes": uh, "t": t,
                             "origin": qq, "chain": 0})

        # ---------- L3 ----------
        for x in serve:
            if x in l3_cache:
                e["l3_cand"] += 1
                if temporal_ok(fc, t-l3_cache[x], "content"):
                    l3h += 1
                    ages["l3"].append(t-l3_cache[x])
                else:
                    e["l3_rej"] += 1
                    l3_cache[x] = t
            else:
                l3_cache[x] = t
        if use_l1:
            l1_cache.append({"query": qq, "fc": fc, "url_hashes": uh, "t": t})

    med = lambda k: statistics.median(ages[k])/3600 if ages[k] else 0.0
    p95 = lambda k: (sorted(ages[k])[int(.95*len(ages[k]))]/3600) if ages[k] else 0.0
    return {"policy": policy, "n": n,
            "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "search": search,
            "saved_pct": round(100*(1-search/n), 2),
            "drift": round(stale_errors/max(hits_scored, 1), 5),
            "stale_errors": stale_errors, "hits_scored": hits_scored,
            "query_funnel": dict(q), "entry_counters": dict(e),
            "median_age_h": {k: round(med(k), 2) for k in ("l1", "l2", "l3")},
            "p95_age_h": {k: round(p95(k), 2) for k in ("l1", "l2", "l3")},
            "l2_chain_hits": dict(chain_hits)}


def main():
    tl = {r["url_hash"]: r for r in
          (json.loads(l) for l in open(OUT/"version_timeline.jsonl"))}
    hashes = {u: r["hashes"] for u, r in tl.items()}
    observable = {u for u, r in tl.items() if r["observable"]}

    queries     = exp.load_jsonl(exp.QUERIES_FILE)
    manifest    = exp.load_jsonl(exp.MANIFEST_FILE)
    paraphrases = (exp.load_jsonl(exp.PARAPHRASE_FILE)
                   if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paraphrases)
    exp.precompute_similarity_matrix(records)
    stream = build_stream(records)
    print(f"  stream: {len(stream):,} requests over {HORIZON/86400:.0f} days, "
          f"Zipf(a={ZIPF_A}) cluster arrivals, seed {SEED}")
    print(f"  web-state model: version_at(t) = last snapshot with age <= t")
    for name, age in RUNS:
        print(f"      t >= {age/3600:>6.1f}h  ->  {name}")

    res = {}
    print(f"\n{'='*104}\n  FOUR POLICIES ON THE IDENTICAL STREAM (review item N2)\n{'='*104}")
    print(f"  {'policy':<16}{'L1':>8}{'L2':>9}{'L3':>9}{'search':>9}{'saved':>9}"
          f"{'drift(all)':>12}{'drift(obs)':>12}")
    for pol in ("freshcache", "no_temporal", "equivalent_ttl", "l2only"):
        a = replay(stream, hashes, pol)
        b = replay(stream, hashes, pol, observable=observable)
        a["drift_observable"] = b["drift"]
        a["hits_scored_observable"] = b["hits_scored"]
        a["stale_errors_observable"] = b["stale_errors"]
        res[pol] = a
        print(f"  {pol:<16}{a['l1_hits']:>8,}{a['l2_hits']:>9,}{a['l3_hits']:>9,}"
              f"{a['search']:>9,}{a['saved_pct']:>8.1f}%"
              f"{100*a['drift']:>11.2f}%{100*b['drift']:>11.2f}%")

    f = res["freshcache"]
    qf, ec, n = f["query_funnel"], f["entry_counters"], f["n"]
    print(f"\n{'='*104}\n  N1: RECONCILING REJECTION RATES WITH SEARCH SAVINGS\n{'='*104}")
    print(f"\n  ENTRY-LEVEL (what V9 reported: one count per cached entry scanned)")
    for t in ("l1", "l2", "l3"):
        c, r = ec.get(f"{t}_cand", 0), ec.get(f"{t}_rej", 0)
        print(f"    {t.upper():<4} rejected {r:>8,} of {c:>8,} entries scanned "
              f"= {100*r/max(c,1):>5.1f}%   ({c/n:.2f} entries per request)")
    print(f"\n  QUERY-LEVEL (one count per request; this is what the savings identity uses)")
    print(f"    {'requests':<38}{n:>9,}")
    for k, lbl in (("realtime_bypass", "REAL_TIME bypass (never cached)"),
                   ("reached_l1", "reached the L1 test"),
                   ("l1_sem_candidate", "had >=1 semantically eligible L1 entry"),
                   ("l1_temporal_rejected_all", "  ... all of them refused on age"),
                   ("l1_served", "SERVED BY L1"),
                   ("reached_l2", "reached the L2 test"),
                   ("l2_sem_candidate", "had >=1 semantically eligible L2 entry"),
                   ("l2_temporal_rejected_all", "  ... all of them refused on age"),
                   ("l2_served", "SERVED BY L2"),
                   ("searched", "fell through to a live search")):
        print(f"    {lbl:<38}{qf.get(k,0):>9,}   {100*qf.get(k,0)/n:>6.2f}%")
    p1 = qf["l1_served"]/n
    p2 = qf["l2_served"]/max(qf["reached_l2"], 1)
    print(f"\n    P(L1 hit)                       = {qf['l1_served']:,}/{n:,} = {p1:.4f}")
    print(f"    P(L2 hit | not served by L1)    = {qf['l2_served']:,}/{qf['reached_l2']:,} = {p2:.4f}")
    print(f"    P(search avoided) = {p1:.4f} + (1-{p1:.4f})x{p2:.4f} = "
          f"{p1+(1-p1)*p2:.4f} = {100*(p1+(1-p1)*p2):.1f}%")
    print(f"    reported search savings                                 = {f['saved_pct']:.1f}%")
    print(f"    identity holds: {abs(100*(p1+(1-p1)*p2)-f['saved_pct'])<0.05}")
    print(f"\n    The reviewer's bound used the entry-level rates as if they were")
    print(f"    query-level. At query level the L1 refusal rate is")
    print(f"    {qf.get('l1_temporal_rejected_all',0):,}/{qf.get('l1_sem_candidate',0):,} = "
          f"{100*qf.get('l1_temporal_rejected_all',0)/max(qf.get('l1_sem_candidate',1),1):.1f}% "
          f"of requests WITH a candidate, and at L2 "
          f"{100*qf.get('l2_temporal_rejected_all',0)/max(qf.get('l2_sem_candidate',1),1):.1f}%.")

    print(f"\n{'='*104}\n  H4: L2 ALIASING — REUSE-CHAIN DEPTH\n{'='*104}")
    ch = f["l2_chain_hits"]
    tot = sum(ch.values())
    print(f"  {'chain depth':<14}{'L2 hits':>10}{'share':>9}   depth 0 = served by an entry from a real search")
    for d in sorted(ch)[:12]:
        print(f"  {d:<14}{ch[d]:>10,}{100*ch[d]/tot:>8.1f}%")
    if len(ch) > 12:
        print(f"  {'>=12':<14}{sum(v for k,v in ch.items() if k>=12):>10,}")
    mx = max(ch) if ch else 0
    wmean = sum(k*v for k, v in ch.items())/max(tot, 1)
    print(f"\n  max chain depth {mx}, hit-weighted mean {wmean:.2f}")

    json.dump(res, open(OUT/"mixed_age_v2_results.json", "w"), indent=2)
    print(f"\n  wrote {OUT/'mixed_age_v2_results.json'}")


if __name__ == "__main__":
    main()
