"""Provenance-recording transcription of the frozen FreshCache path (same logic as outputs/systems_timing/
freshcache/simulation/reference_replay.py at unbounded capacity, which reproduces the frozen replay exactly). Adds only bookkeeping: which request created each
L1 entry and L2 entry, the request that originally discovered each L2 URL list (aliases inherit it), and the request that last wrote each L3 page.
No decision depends on the bookkeeping. The caller gates the decisions against the frozen per-request tiers."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import numpy as np

def run(E, stream):
    exp, p2 = E["exp"], E["p2"]; SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = exp.L1_SIM_THRESHOLD; L2F = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
    EPS = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST, "content": exp.EPS_CONTENT}
    ok = lambda fc, age, tier: exp.p_stale(fc, age, tier) <= EPS[tier]
    n = len(stream)
    L1 = {"qi": np.empty(n, np.int64), "t": np.empty(n, np.float64), "fc": [], "q": [], "creator": [], "n": 0}
    L2 = {"qi": np.empty(n, np.int64), "t": np.empty(n, np.float64), "fc": [], "urls": [], "creator": [], "origin": [], "n": 0}
    L3, L3w = {}, {}
    rows = {}
    for t, r in stream:
        q, fc, qid = r["query"], r["freshness_class"], r["query_id"]; own = [u["url_hash"] for u in r["urls"]]
        if fc == "REAL_TIME":
            rows[qid] = {"tier": "REAL_TIME_BYPASS", "own": len(own), "fetched": len(own)}; continue
        row = SIM[IDX[q]]; hit = -1; n1 = L1["n"]
        if n1:
            sims = row[L1["qi"][:n1]]; cand = np.nonzero(sims >= L1T)[0]
            if cand.size:
                okm = np.fromiter((ok(L1["fc"][int(j)], t - L1["t"][j], "answer") for j in cand), bool, cand.size); keep = cand[okm]
                for j in keep[np.argsort(-sims[keep], kind="stable")]:
                    j = int(j)
                    if exp._entity_match(q, L1["q"][j]) and exp.semantic_equivalent(q, L1["q"][j], float(sims[j])): hit = j; break
        if hit >= 0:
            rows[qid] = {"tier": "L1", "own": len(own), "fetched": 0, "l1_entry_creator": L1["creator"][hit]}; continue
        hit2 = -1; n2 = L2["n"]
        if n2:
            s2 = row[L2["qi"][:n2]]; cand = np.nonzero(s2 >= L2F)[0]
            if cand.size:
                okm = np.fromiter((ok(L2["fc"][int(j)], t - L2["t"][j], "url_list") for j in cand), bool, cand.size); keep = cand[okm]
                if keep.size: hit2 = int(keep[np.argmax(s2[keep])])
        if hit2 >= 0: served, disc, tier, origin = list(L2["urls"][hit2]), float(L2["t"][hit2]), "L2", L2["origin"][hit2]
        else: served, disc, tier, origin = list(own), t, "miss", qid
        ev, reuse, fetched = [], [], 0
        for x in served:
            tc = L3.get(x)
            if tc is not None and ok(fc, t - tc, "content"): ev.append((x, tc)); reuse.append(L3w[x]); continue
            fetched += 1; L3[x] = t; L3w[x] = qid; ev.append((x, t))
        rows[qid] = {"tier": tier, "own": len(own), "served": len(served), "fetched": fetched, "l3_reuse_writers": reuse,
                     **({"l2_entry_creator": L2["creator"][hit2], "l2_list_origin": origin} if hit2 >= 0 else {})}
        j = L2["n"]; L2["qi"][j] = IDX[q]; L2["t"][j] = disc; L2["fc"].append(fc); L2["urls"].append(served); L2["creator"].append(qid); L2["origin"].append(origin); L2["n"] += 1
        k = L1["n"]; L1["qi"][k] = IDX[q]
        L1["t"][k] = (min(float(tc) for _, tc in ev) if (p2.L1_TE_PAGES_ONLY and ev) else min([float(tc) for _, tc in ev] + [float(disc)]))
        L1["fc"].append(fc); L1["q"].append(q); L1["creator"].append(qid); L1["n"] += 1
    return rows
