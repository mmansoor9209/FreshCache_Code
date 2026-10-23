"""Composition replay engine (composition_engine) with optional candidate-retriever hooks (retr) for the top-k index parity check; built on
freshcache/composition/categoryaware_freshcache_engine.py with two added switches (defaults reproduce categoryaware_freshcache_engine exactly):
  l2_on=False -> no URL-list reuse (every answer-cache miss searches its own list) ; l3_on=False -> no page reuse (every served URL fetched at serving time).
A = l2l3=False ; B = l2_on=False ; C = l3_on=False ; D = both on. The base engine's docstring follows."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
"""composition hybrid replay engine: a transcription of outputs/replay/verifier/verifier_replay.replay (itself a verbatim transcription of the
reported audit_replay FreshCache path plus the gamma verifier) with exactly three switches:
  l1     "freshcache" -> FreshCache L1 unchanged (used ONLY to gate the transcription against frozen replays)
         "ca"         -> the frozen baseline-selection/CategoryAware-frontier CategoryAware-style answer cache: threshold from the REQUEST's class, TTL = base FIXED_TTL[class] x
                         kappa[class] from the ENTRY's class, age = request time - entry write time, highest-similarity kept entry (stable)
  floor  L2 equivalence floor (theta_L2); L2 threshold = max(L2_SIM_THRESHOLD, floor) exactly as frozen
  l2l3   False -> no L2/L3 (every L1 miss searches and fetches its own pages; used ONLY to gate the CA L1 against baseline-selection ca_replay)
Everything else (L2 selection and temporal gate, verifier, L3 reuse rule incl. duplicate-URL handling, L2 registration, REAL_TIME full bypass,
timestamps, sequential order) is unchanged. Hybrid L1 entries keep the evidence pages actually used to generate the answer."""
import numpy as np
BASE_TTL = {"TIMELESS": 2592000, "SLOW": 604800, "MEDIUM": 86400, "FAST": 3600, "REAL_TIME": 0}

def replay(E, stream, l1="ca", floor=0.75, gamma=None, support=None, theta_by_class=None, kappa_by_class=None, l2l3=True, l2_on=True, l3_on=True, retr=None):
    # retr: optional candidate retriever object with .l1(qi, n1, sims) / .l2(qi, n2, sims) returning candidate positions, and .add1/.add2 hooks (top-k parity study)
    exp = E["exp"]; L1T = exp.L1_SIM_THRESHOLD; L2FLOOR = max(exp.L2_SIM_THRESHOLD, floor)
    EPS = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST, "content": exp.EPS_CONTENT}
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX; n_max = len(stream)
    ok = lambda fc, age, tier: exp.p_stale(fc, age, tier) <= EPS[tier]
    if l1 == "ca":
        ttl = {c: BASE_TTL[c] * kappa_by_class.get(c, 0.0) for c in BASE_TTL}
        ca_ok = lambda efc, age: bool(ttl.get(efc, 0) and age <= ttl.get(efc, 0))
    l1_qi = np.empty(n_max, np.int64); l1_t = np.empty(n_max, np.float64); l1_q, l1_fc, l1_src = [], [], []; n1 = 0
    l2_qi = np.empty(n_max, np.int64); l2_t = np.empty(n_max, np.float64); l2_fc, l2_q, l2_urls = [], [], []; n2 = 0
    l3_cache = {}; l1h = l2h = l3h = search = fetches = 0; accepted = rejected = unscored = 0; rows = {}
    def l3_pure(served, t, fc):
        local, ev, reused, tofetch = {}, [], 0, []
        for x in served:
            ct = local.get(x, l3_cache.get(x))
            if ct is not None and ok(fc, t - ct, "content"): reused += 1; ev.append((x, ct))
            else: tofetch.append(x); local[x] = t; ev.append((x, t))
        return ev, reused, tofetch
    def fetch_all(served, t):   # C (L2 only): every served URL is fetched at serving time; duplicates within one request fetched once
        seen, ev, tofetch = set(), [], []
        for x in served:
            ev.append((x, t))
            if x not in seen: seen.add(x); tofetch.append(x)
        return ev, 0, tofetch
    for t, r in stream:
        q, fc, qid = r["query"], r["freshness_class"], r["query_id"]; qi = IDX[q]; row = SIM[qi]; own = [u["url_hash"] for u in r["urls"]]
        if fc == "REAL_TIME":
            search += 1; fetches += len(own)
            rows[qid] = {"tier": "REAL_TIME", "t": t, "ev": [(u, t) for u in own], "own": own, "gen_query": q, "fetched": len(own), "l3_reused_pages": 0}; continue
        hit1 = -1
        if n1:
            sims = row[l1_qi[:n1]]
            if l1 == "freshcache":
                cand = np.nonzero(sims >= L1T)[0]
                if cand.size:
                    okm = np.fromiter((ok(l1_fc[int(j)], t - l1_t[j], "answer") for j in cand), bool, cand.size); keep = cand[okm]
                    for j in keep[np.argsort(-sims[keep], kind="stable")]:
                        j = int(j)
                        if not (exp._entity_match(q, l1_q[j]) and exp.semantic_equivalent(q, l1_q[j], float(sims[j]))): continue
                        hit1 = j; break
            else:
                cand = np.nonzero(sims >= theta_by_class[fc])[0]
                if retr is not None: cand = np.intersect1d(cand, retr.l1(qi, n1, sims))
                if cand.size:
                    keep = cand[np.fromiter((ca_ok(l1_fc[int(j)], t - l1_t[j]) for j in cand), bool, cand.size)]
                    if keep.size: hit1 = int(keep[np.argsort(-sims[keep], kind="stable")][0])
        if hit1 >= 0:
            l1h += 1; src = l1_src[hit1]
            rows[qid] = {"tier": "L1", "t": t, "similarity": float(row[l1_qi[hit1]]), "ev": src["ev"], "gen_query": src["query"], "stored_answer_t": src["t"], "entry_class": l1_fc[hit1],
                         "entry_index": hit1, "own": own, "fetched": 0, "l3_reused_pages": 0}; continue
        if not l2l3:
            search += 1; fetches += len(own); ev = [(u, t) for u in own]
            rows[qid] = {"tier": "miss", "t": t, "ev": ev, "own": own, "gen_query": q, "fetched": len(own), "l3_reused_pages": 0}
            l1_qi[n1] = qi; l1_t[n1] = t; l1_q.append(q); l1_fc.append(fc); l1_src.append({"query": q, "ev": ev, "t": t}); n1 += 1
            if retr is not None: retr.add1(qi)
            continue
        hit2 = -1; sim2 = None
        if n2 and l2_on:
            s2 = row[l2_qi[:n2]]; cand = np.nonzero(s2 >= L2FLOOR)[0]
            if retr is not None: cand = np.intersect1d(cand, retr.l2(qi, n2, s2))
            if cand.size:
                okm = np.fromiter((ok(l2_fc[int(j)], t - l2_t[j], "url_list") for j in cand), bool, cand.size); keep = cand[okm]
                if keep.size: hit2 = int(keep[np.argmax(s2[keep])]); sim2 = float(s2[hit2])
        verdict = None; vscore = None
        if hit2 >= 0 and gamma is not None and support is not None:
            tent_ev, _, _ = l3_pure(list(l2_urls[hit2]), t, fc); vscore = support.score(q, tent_ev)
            if vscore is None: unscored += 1; verdict = "ACCEPT_UNSCORED"
            elif vscore >= gamma: accepted += 1; verdict = "ACCEPT"
            else: rejected += 1; verdict = "REJECT"
            if verdict == "REJECT": hit2 = -1; sim2 = None
        if hit2 >= 0:
            l2h += 1; served = list(l2_urls[hit2]); disc = float(l2_t[hit2])
        else:
            search += 1; served = list(own); disc = t
        l2_qi[n2] = qi; l2_t[n2] = disc; l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
        if retr is not None: retr.add2(qi)
        if l3_on: ev, n_reused, tofetch = l3_pure(served, t, fc)
        else: ev, n_reused, tofetch = fetch_all(served, t)
        l3h += n_reused; fetches += len(tofetch)
        if l3_on:
            for x in tofetch: l3_cache[x] = t
        rows[qid] = {"tier": "L2" if hit2 >= 0 else ("miss_rejected" if verdict == "REJECT" else "miss"), "t": t, "similarity": sim2, "served": served, "t_cached": disc, "ev": ev,
                     "l3_reused_pages": n_reused, "fetched": len(tofetch), "own": own, "gen_query": q, "verified": verdict, "support_score": vscore}
        l1_qi[n1] = qi; l1_q.append(q); l1_fc.append(fc); l1_src.append({"query": q, "ev": ev, "t": t})
        if retr is not None: retr.add1(qi)
        l1_t[n1] = (min(float(tc) for _, tc in ev) if ev else t) if l1 == "freshcache" else t   # FreshCache: oldest supporting page; CA: write time (frozen CA policy)
        n1 += 1
    n = len(stream)
    return {"n_requests": n, "search_calls": search, "search_saved_pct": round(100 * (1 - search / n), 4), "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "fetches": fetches,
            "generations": n - l1h, "l2_accepted": accepted, "l2_rejected": rejected, "l2_unscored": unscored}, rows

class Support:   # verifier_replay.Support, verbatim logic (precomputed BGE-M3 embeddings; no model runs)
    def __init__(self, outdir, ma):
        q = np.load(outdir / "emb_queries.npz", allow_pickle=True); p = np.load(outdir / "emb_pages.npz", allow_pickle=True)
        self.qi = {k: i for i, k in enumerate(q["keys"])}; self.qe = q["emb"]; self.pi = {k: i for i, k in enumerate(p["keys"])}; self.pe = p["emb"]; self.ma = ma
    def score(self, query, ev):
        qi = self.qi.get(query)
        if qi is None: return None
        best = None
        for uh, tv in ev:
            j = self.pi.get(f"{uh}|{self.ma.version_at(tv)}")
            if j is None: continue
            s = float(self.qe[qi] @ self.pe[j]); best = s if best is None or s > best else best
        return best
