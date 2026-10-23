#!/usr/bin/env python3
"""
verifier_replay.py — FreshCache-L2Verify sequential replay engine.

The body is a verbatim transcription of
freshcache/selection_replay_engine/audit_replay.py::replay (FreshCache path with
per-request provenance), with EXACTLY ONE behavioural addition:

    L2 hit -> normal L3 processing -> evidence C
    V(q,C) = max_c cosine(E(q), E(c))        (BGE-M3, precomputed offline)
    V >= gamma : keep the L2 reuse
    V <  gamma : REJECT -> fresh search, then normal L3 and generation

Unchanged: L1 (entity + semantic-equivalence + temporal), L2 threshold/floor and
temporal gate, half-lives, epsilons, L3 reuse rule, timestamps, query order and
cache-update semantics. On rejection the request proceeds and registers exactly
as an L2 miss would; the rejected L2 entry is left in the cache untouched.

gamma=None disables the gate and must reproduce audit_replay.replay bit-for-bit
(asserted by the fidelity check).

The L3 step is factored into a PURE simulation plus an explicit commit so the
tentative evidence can be scored before any cache mutation. The simulation walks
the served list in order over a copy-on-write view, so a URL repeated within one
request behaves exactly as in audit_replay (first occurrence fetches, later occurrences
reuse at age 0).
"""
from __future__ import annotations
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import pathlib, sys
import numpy as np

HERE = ((OUTPUT_ROOT / "replay/verifier"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()   # copy location
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
import tiered_cache_model as exp                      # noqa: E402
import method_replay as ea                       # noqa: E402
import page_version_replay as ma                     # noqa: E402


class Support:
    """Precomputed BGE-M3 support scorer; no model runs during the replay."""

    def __init__(self, outdir):
        q = np.load(pathlib.Path(outdir) / "emb_queries.npz", allow_pickle=True)
        p = np.load(pathlib.Path(outdir) / "emb_pages.npz", allow_pickle=True)
        self.qi = {k: i for i, k in enumerate(q["keys"])}
        self.qe = q["emb"]
        self.pi = {k: i for i, k in enumerate(p["keys"])}
        self.pe = p["emb"]

    def score(self, query, ev):
        qi = self.qi.get(query)
        if qi is None:
            return None
        best = None
        for uh, tv in ev:
            j = self.pi.get(f"{uh}|{ma.version_at(tv)}")
            if j is None:
                continue
            s = float(self.qe[qi] @ self.pe[j])
            if best is None or s > best:
                best = s
        return best


def replay(stream, gamma=None, support=None):
    cfg = ea.CONFIG["FreshCache_Full"]
    L1T = exp.L1_SIM_THRESHOLD
    L2FLOOR = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
    EPS = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST,
           "content": exp.EPS_CONTENT}
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    n_max = len(stream)

    def ok(fc, age, tier):
        return exp.p_stale(fc, age, tier) <= EPS[tier]

    l1_qi = np.empty(n_max, np.int64); l1_t = np.empty(n_max, np.float64)
    l1_q, l1_fc, l1_urls, l1_src = [], [], [], []
    n1 = 0
    l2_qi = np.empty(n_max, np.int64); l2_t = np.empty(n_max, np.float64)
    l2_fc, l2_q, l2_urls = [], [], []
    n2 = 0
    l3_cache = {}
    l1h = l2h = l3h = search = fetches = 0
    accepted = rejected = unscored = 0
    rows, scores = {}, []

    def l3_pure(served, t, fc):
        """audit_replay's L3 loop, simulated without mutating l3_cache."""
        local, ev, reused, tofetch = {}, [], 0, []
        for x in served:
            ct = local.get(x, l3_cache.get(x))
            if ct is not None and ok(fc, t - ct, "content"):
                reused += 1; ev.append((x, ct))
            else:
                tofetch.append(x); local[x] = t; ev.append((x, t))
        return ev, reused, tofetch

    for t, r in stream:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]; qi = IDX[q]; row = SIM[qi]
        own = [u["url_hash"] for u in urls]

        if fc == "REAL_TIME":
            search += 1
            for h in own:
                fetches += 1          # REAL_TIME writes no L3 page
            rows[qid] = {"tier": "REAL_TIME", "t": t,
                         "ev": [(u, t) for u in own], "own": own,
                         "gen_query": q, "verified": None, "support_score": None}
            continue

        # ---- L1 (unchanged) ----
        hit1 = -1; sim1 = None
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
                    hit1 = j; sim1 = float(sims[j]); break
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
                         "stored_answer_t": src["t"], "own": own,
                         "gen_query": src["query"], "ev": src["ev"],
                         "verified": None, "support_score": None}
            continue

        # ---- L2 (unchanged selection) ----
        hit2 = -1; sim2 = None
        if n2:
            s2 = row[l2_qi[:n2]]
            cand = np.nonzero(s2 >= L2FLOOR)[0]
            if cand.size:
                okm = np.fromiter((ok(l2_fc[int(j)], t - l2_t[j], "url_list")
                                   for j in cand), bool, cand.size)
                keep = cand[okm]
                if keep.size:
                    hit2 = int(keep[np.argmax(s2[keep])]); sim2 = float(s2[hit2])

        verdict = None; vscore = None
        if hit2 >= 0 and gamma is not None and support is not None:
            tent_served = list(l2_urls[hit2])
            tent_ev, _, _ = l3_pure(tent_served, t, fc)
            vscore = support.score(q, tent_ev)
            if vscore is None:
                unscored += 1; verdict = "ACCEPT_UNSCORED"
            elif vscore >= gamma:
                accepted += 1; verdict = "ACCEPT"
            else:
                rejected += 1; verdict = "REJECT"
            scores.append({"query_id": qid, "support_score": vscore,
                           "verdict": verdict, "l2_similarity": sim2,
                           "freshness_class": fc, "t": t})
            if verdict == "REJECT":
                hit2 = -1; sim2 = None

        if hit2 >= 0:
            l2h += 1
            served = list(l2_urls[hit2]); disc = float(l2_t[hit2])
            l2_qi[n2] = qi; l2_t[n2] = disc
            l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
        else:
            search += 1
            served = list(own); disc = t
            l2_qi[n2] = qi; l2_t[n2] = t
            l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1

        ev, n_reused, tofetch = l3_pure(served, t, fc)
        l3h += n_reused; fetches += len(tofetch)
        for x in tofetch:
            l3_cache[x] = t

        rows[qid] = {"tier": ("L2" if hit2 >= 0 else
                              ("miss_rejected" if verdict == "REJECT" else "miss")),
                     "t": t, "similarity": sim2, "served": served,
                     "t_cached": disc, "age": (t - disc) if hit2 >= 0 else 0.0,
                     "ev": ev, "l3_reused_pages": n_reused, "own": own,
                     "gen_query": q, "verified": verdict, "support_score": vscore}

        l1_qi[n1] = qi; l1_t[n1] = min(float(tc) for _, tc in ev) if ev else t   # oldest supporting page
        l1_q.append(q); l1_fc.append(fc); l1_urls.append(served)
        l1_src.append({"query": q, "ev": ev, "t": t}); n1 += 1

    n = len(stream)
    m = {"n_requests": n, "search_calls": search,
         "search_saved_pct": round(100 * (1 - search / n), 4),
         "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "fetches": fetches,
         "generations": search + l2h,
         "fetches_per_1k": round(1000 * fetches / n, 2),
         "generations_per_1k": round(1000 * (search + l2h) / n, 2),
         "l2_accepted": accepted, "l2_rejected": rejected,
         "l2_unscored": unscored}
    return m, rows, scores


def ma_outcome(uh, tc, tr, rounds):
    """mixed_age_replay.outcome: CHANGED/UNCHANGED/None for a page cached at tc,
    read at request time tr, against the snapshot round table."""
    import execution_semantics as _ce
    import mixed_age_replay as _me                                 # latest stored snapshot not later than t
    per = rounds.get(uh)
    if not per:
        return None
    ra, rb = _me.round_at(per, tc), _me.round_at(per, tr)
    if ra is None or rb is None:
        return None
    a, b = per[ra], per[rb]
    if not a.get("substantive") or not b.get("substantive"):
        return None
    return _ce.CHANGED if a["content_hash"] != b["content_hash"] else _ce.UNCHANGED
