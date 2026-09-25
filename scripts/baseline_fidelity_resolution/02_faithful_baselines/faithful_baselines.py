#!/usr/bin/env python3
"""TASK 2/4 -- faithful(er) SCALM and vCache on the primary mixed-age workload.

Two new baselines, in separate scripts from the originals, preserving each
original method's essential algorithm:

  vCache-Reference   the policy of vcache_implementation.py (a port of
                     vcache/vcache_policy/strategies/verified.py) driven by the
                     MIXED-AGE stream instead of a single fixed age. Logistic
                     threshold fit, Eq-11 tau, explore/exploit sampling, and
                     update-on-explore are used verbatim.

  SCALM-HSC          SCALM's semantics-oriented cache: DBSCAN semantic patterns
                     over the real BGE-M3 query embeddings, token-saving-ratio
                     pattern ranking, high/mid/low ranks at the top 25/50/75%,
                     rank-adaptive storage, and the improved-LFU eviction with
                     priorities 3/2/1 incremented on hit. Evaluated unbounded
                     (as the benchmark is) and at a bounded capacity, because
                     SCALM's storage and eviction contributions cannot act
                     unless capacity binds.

Neither baseline is given FreshCache's temporal gate, risk model or tier
structure, and neither is given information unavailable at serving time: every
correctness signal is computed from the snapshot state at the serving round and
fed back only after the request has been served.
"""
from __future__ import annotations
import csv, json, math, os, pathlib, random, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
BFR = HERE.parent
V3 = BFR.parent
ROOT = V3.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import mixed_age_v2 as ma                # noqa: E402
import schedules as sc                   # noqa: E402
import vcache_implementation as VC       # noqa: E402
from e1_robustness import support_of     # noqa: E402

CHANGED, UNCHANGED, UNOBS = me.CHANGED, me.UNCHANGED, me.UNOBS
SEED, SCHEDULE = 42, "zipf_uniform"
PAPER_DELTA = 0.10
SCALM_SIM = 0.90          # SCALM paper threshold (Table I), on its own scale
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def score(m, per, S, n):
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == CHANGED)
    det = ch + sum(1 for o in ins if o == UNCHANGED)
    return {"search_saved_pct": round(m["search_saved_pct"], 4),
            "drift_pct": (round(100 * ch / det, 4) if det else None),
            "coverage_pct": (round(100 * det / len(S), 4) if S else None),
            "determinable": det, "changed": ch,
            "searches": m["search_calls"], "fetches": m["fetches"],
            "generations": m["generations"],
            "fetch_per_1k": round(1000 * m["fetches"] / n, 2),
            "gen_per_1k": round(1000 * m["generations"] / n, 2),
            "l1_hits": m["l1_hits"], "n_requests": n}


def outcome(entry_urls, t_cached, t_serve, rounds):
    labs = [me.outcome(u, t_cached, t_serve, rounds) for u in entry_urls]
    if not labs:
        return None
    return (CHANGED if CHANGED in labs else
            UNCHANGED if UNCHANGED in labs else UNOBS)


# ----------------------------------------------------------------- vCache
def vcache_mixed(stream, rounds, delta=PAPER_DELTA, seed=SEED):
    """vcache_implementation.sim_vcache_reference, re-driven by the mixed-age
    stream. The POLICY calls are unchanged; only the staleness label is now
    computed per (entry, serving time) from the round table, because in a
    mixed-age replay there is no single global stale set."""
    rng = random.Random(seed)
    cache, per = [], []
    searches = fetches = gens = l1 = 0
    n_dec = n_fit = n_exp = 0
    c1 = c0 = same_q = 0
    for t, r in stream:
        q, fc = r["query"], r["freshness_class"]
        uh = [u["url_hash"] for u in r["urls"]]
        uset = frozenset(uh)

        def full(add):
            nonlocal searches, fetches, gens
            searches += 1; fetches += len(uh); gens += 1
            if add:
                cache.append({"query": q, "qi": exp._QUERY_TO_IDX[q],
                              "fc": fc, "url_hashes": uh,
                              "url_set": uset, "observations": [], "t": t})
        if fc == "REAL_TIME":
            full(True); per.append((r["query_id"], None)); continue
        if cache:
            ci = np.fromiter((e["qi"] for e in cache), dtype=np.int64, count=len(cache))
            sims = exp._SIM_MATRIX[exp._QUERY_TO_IDX[q], ci]
            j = int(np.argmax(sims)); best, nn = float(sims[j]), cache[j]
        else:
            best, nn = -1.0, None
        if nn is None:
            full(True); per.append((r["query_id"], None)); continue
        explore, _fit, _tau = VC._select_action_explore(
            best, nn["observations"], delta, rng)
        n_dec += 1
        n_fit += int(bool(_fit))
        n_exp += int(bool(explore))
        if not explore:                      # Alg 1 line 5: exploit
            l1 += 1
            per.append((r["query_id"], outcome(nn["url_hashes"], nn["t"], t, rounds)))
            continue
        # Alg 1 lines 7-12: explore. The label is post-serve feedback.
        o = outcome(nn["url_hashes"], nn["t"], t, rounds)
        c = 1 if (nn["url_set"] == uset and o != CHANGED) else 0
        same_q += int(nn["url_set"] == uset)
        c1 += c; c0 += (1 - c)
        nn["observations"].append((round(best, 3), c))
        full(add=(c == 0))
        per.append((r["query_id"], None))
    nc = len(stream)
    say(f"    vCache feedback labels: c=1 {c1:,} ({100*c1/max(1,c1+c0):.2f}%), "
        f"c=0 {c0:,}; nearest neighbour was the SAME question "
        f"{same_q:,} times ({100*same_q/max(1,c1+c0):.2f}%)")
    say(f"    vCache: {n_dec:,} policy decisions, {n_fit:,} with a fitted "
        f"threshold ({100*n_fit/max(1,n_dec):.1f}%), {n_exp:,} explore / "
        f"{l1:,} exploit; final cache {len(cache):,} entries")
    return ({"search_calls": searches, "fetches": fetches, "generations": gens,
             "l1_hits": l1, "n_requests": nc,
             "search_saved_pct": 100 * (1 - searches / nc),
             "decisions": n_dec, "decisions_with_fit": n_fit,
             "explores": n_exp, "label_c1": c1, "label_c0": c0,
             "nn_same_question": same_q}, per)


# ------------------------------------------------------------------ SCALM
def scalm_hsc(stream, rounds, records, capacity=None, seed=SEED):
    """SCALM: semantic patterns -> token-saving rank -> adaptive storage and
    improved-LFU eviction. Alg 2 (SE-HSC) with R = 1 (see fidelity notes)."""
    from sklearn.cluster import DBSCAN
    emb = np.load(str(ROOT / "data" / "query_embeddings_bgem3.npy"),
                  mmap_mode="r")
    idx = exp._QUERY_TO_IDX
    # ---- SE-HSC: cluster the query embeddings into semantic patterns ----
    X = np.asarray(emb, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    lab = DBSCAN(eps=0.35, min_samples=5, metric="cosine",
                 n_jobs=8).fit_predict(X)
    # ---- token-saving ratio per pattern (Eq. for R_ts), query tokens as the
    #      observable proxy: answers are not stored for every query ----
    toks = np.array([len(r["query"].split()) for r in records], dtype=np.float64)
    tot = toks.sum()
    pat_tok = defaultdict(float); pat_n = Counter()
    for i, p in enumerate(lab):
        if p < 0:
            continue
        pat_tok[p] += toks[i]; pat_n[p] += 1
    ranked = sorted(pat_tok, key=lambda p: -pat_tok[p] / tot)
    hi = set(ranked[:max(1, int(0.25 * len(ranked)))])
    mid = set(ranked[:max(1, int(0.50 * len(ranked)))]) - hi
    prio = lambda p: 3 if p in hi else 2 if p in mid else 1   # noqa: E731
    say(f"    SE-HSC: {len(ranked)} patterns over "
        f"{sum(pat_n.values()):,} clustered queries "
        f"({100*sum(pat_n.values())/len(lab):.1f}% assigned; "
        f"{int((lab<0).sum()):,} noise); high {len(hi)}, mid {len(mid)}")

    cache, per = {}, []
    searches = fetches = gens = l1 = 0
    order = inserted = evictions = refused = 0
    for t, r in stream:
        q, fc = r["query"], r["freshness_class"]
        uh = [u["url_hash"] for u in r["urls"]]
        i = idx.get(q, -1)
        pat = int(lab[i]) if i >= 0 else -1

        def full(add):
            nonlocal searches, fetches, gens, order, inserted, evictions, refused
            searches += 1; fetches += len(uh); gens += 1
            if not add:
                return
            # ---- adaptive storage (SCALM IV-B): once the cache is full only
            #      mid/high-rank patterns are admitted ----
            full_cache = capacity is not None and len(cache) >= capacity
            if full_cache and prio(pat) < 2:
                refused += 1
                return
            if full_cache:
                # ---- improved-LFU eviction (SCALM IV-C) ----
                victim = min(cache, key=lambda k: (cache[k]["value"],
                                                   cache[k]["order"]))
                del cache[victim]; evictions += 1
            order += 1; inserted += 1
            cache[q] = {"query": q, "qi": idx.get(q, -1), "url_hashes": uh,
                        "t": t, "pat": pat, "value": prio(pat), "order": order}
        if fc == "REAL_TIME":
            full(True); per.append((r["query_id"], None)); continue
        if cache:
            ents = list(cache.values())
            ci = np.fromiter((e["qi"] for e in ents), dtype=np.int64, count=len(ents))
            sims = exp._SIM_MATRIX[i, ci] if i >= 0 else np.full(len(ents), -1.0)
            j = int(np.argmax(sims)); best, nn = float(sims[j]), ents[j]
        else:
            best, nn = -1.0, None
        if nn is not None and best >= exp.L1_SIM_THRESHOLD:
            l1 += 1
            nn["value"] += 1                 # hits raise eviction priority
            per.append((r["query_id"], outcome(nn["url_hashes"], nn["t"], t, rounds)))
            continue
        full(True); per.append((r["query_id"], None))
    nc = len(stream)
    return ({"search_calls": searches, "fetches": fetches, "generations": gens,
             "l1_hits": l1, "n_requests": nc,
             "search_saved_pct": 100 * (1 - searches / nc)}, per,
            {"patterns": len(ranked), "capacity": capacity,
             "inserted": inserted, "evictions": evictions,
             "refused": refused, "final_cache_size": len(cache)})


def main():
    say("TASK 2/4 -- faithful SCALM and vCache on the PRIMARY mixed-age workload")
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    for line in open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
                     encoding="utf-8"):
        d = json.loads(line)
        rounds[d["url_hash"]] = d["rounds"]
    full = sc.build_stream(records, SCHEDULE, SEED)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c = set(split["test_clusters"])
    test = [(t, r) for t, r in full
            if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(test) == split["test"]["requests"]

    out = {}
    for pop, stream in (("full_mixed_age", full), ("heldout_test", test)):
        S = support_of(stream, rounds)
        n = len(stream)
        say(f"\n  ======== {pop}: {n:,} requests, |S| = {len(S):,} ========")
        rows = {}
        m, pr = me.replay(stream, rounds, "SCALM", rich=rich)
        rows["SCALM-style (published approximation)"] = score(m, pr, S, n)
        for cap in (None, 400, 200):
            m, pr, info = scalm_hsc(stream, rounds, records, capacity=cap)
            key = ("SCALM-HSC unbounded" if cap is None
                   else f"SCALM-HSC capacity={cap:,}")
            say(f"      -> inserted {info['inserted']:,}, evictions "
                f"{info['evictions']:,}, refused by rank {info['refused']:,}")
            rows[key] = score(m, pr, S, n); rows[key]["scalm_info"] = info
        m, pr = vcache_mixed(stream, rounds, PAPER_DELTA)
        k = f"vCache-Reference (proxy feedback, delta={PAPER_DELTA})"
        rows[k] = score(m, pr, S, n)
        rows[k]["vcache_info"] = {x: m[x] for x in
                                  ("decisions", "decisions_with_fit", "explores",
                                   "label_c1", "label_c0", "nn_same_question")}
        say(f"\n  {'baseline':<46}{'saved%':>9}{'drift%':>9}{'cov%':>8}"
            f"{'fetch/1k':>10}{'gen/1k':>9}{'L1':>8}{'searches':>10}")
        for kk, v in rows.items():
            d = f"{v['drift_pct']:.4f}" if v["drift_pct"] is not None else "   -  "
            c = f"{v['coverage_pct']:.4f}" if v["coverage_pct"] is not None else "  -  "
            say(f"  {kk:<46}{v['search_saved_pct']:>8.4f}%{d:>8}%{c:>7}%"
                f"{v['fetch_per_1k']:>10.2f}{v['gen_per_1k']:>9.2f}"
                f"{v['l1_hits']:>8,}{v['searches']:>10,}")
        out[pop] = rows

    json.dump({"schedule": SCHEDULE, "seed": SEED,
               "delta": PAPER_DELTA, "populations": out},
              open(HERE / "faithful_baselines_mixed.json", "w"), indent=2)
    (BFR / "logs" / "t2.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote faithful_baselines_mixed.json")


if __name__ == "__main__":
    main()
