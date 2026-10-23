"""Step-wise (online) form of reference_replay.Replay for the end-to-end diagnostic: identical decision logic, but the caller receives each
request's decision (tier, served pages, evidence, pages to fetch, matched L1 entry) and performs the real work (embedding, fetch, generation)
between steps. Unbounded capacity only. Decisions are gated by the caller against reference_replay.Replay on the same sub-stream."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import time
import numpy as np

class Online:
    def __init__(self, E):
        self.exp, self.p2 = E["exp"], E["p2"]; ex = self.exp
        self.SIM, self.IDX = ex._SIM_MATRIX, ex._QUERY_TO_IDX
        self.L1T = ex.L1_SIM_THRESHOLD; self.L2F = max(ex.L2_SIM_THRESHOLD, ex._L2_EQ_SIM_FLOOR)
        self.EPS = {"answer": ex.EPS_ANSWER, "url_list": ex.EPS_URL_LIST, "content": ex.EPS_CONTENT}
        self.L1 = {"qi": [], "t": [], "fc": [], "q": [], "ev": []}; self.L2 = {"qi": [], "t": [], "fc": [], "urls": []}; self.L3 = {}
    def ok(self, fc, age, tier): return self.exp.p_stale(fc, age, tier) <= self.EPS[tier]
    def decide(self, t, r):
        ex, ok, tm = self.exp, self.ok, time.perf_counter_ns; q, fc = r["query"], r["freshness_class"]; own = [u["url_hash"] for u in r["urls"]]; T = {}
        if fc == "REAL_TIME": return {"tier": "REAL_TIME_BYPASS", "served": own, "ev": [(u, t) for u in own], "fetch": own, "l1_entry": None, "timers_ms": T}
        row = self.SIM[self.IDX[q]]; L1, L2 = self.L1, self.L2
        t0 = tm(); hit = -1
        if L1["qi"]:
            qi = np.asarray(L1["qi"]); lt = np.asarray(L1["t"]); sims = row[qi]; cand = np.nonzero(sims >= self.L1T)[0]
            if cand.size:
                okm = np.fromiter((ok(L1["fc"][int(j)], t - lt[j], "answer") for j in cand), bool, cand.size); keep = cand[okm]
                for j in keep[np.argsort(-sims[keep], kind="stable")]:
                    j = int(j)
                    if ex._entity_match(q, L1["q"][j]) and ex.semantic_equivalent(q, L1["q"][j], float(sims[j])): hit = j; break
        T["l1_lookup"] = (tm() - t0) / 1e6
        if hit >= 0: return {"tier": "L1", "served": [], "ev": L1["ev"][hit], "fetch": [], "l1_entry": hit, "timers_ms": T}
        t0 = tm(); hit2 = -1
        if L2["qi"]:
            qi = np.asarray(L2["qi"]); lt = np.asarray(L2["t"]); s2 = row[qi]; cand = np.nonzero(s2 >= self.L2F)[0]
            if cand.size:
                okm = np.fromiter((ok(L2["fc"][int(j)], t - lt[j], "url_list") for j in cand), bool, cand.size); keep = cand[okm]
                if keep.size: hit2 = int(keep[np.argmax(s2[keep])])
        T["l2_lookup"] = (tm() - t0) / 1e6
        if hit2 >= 0: served, disc, tier = list(L2["urls"][hit2]), float(L2["t"][hit2]), "L2"
        else: served, disc, tier = list(own), t, "miss"
        t0 = tm(); ev, fetch = [], []
        for x in served:
            tc = self.L3.get(x)
            if tc is not None and ok(fc, t - tc, "content"): ev.append((x, tc)); continue
            self.L3[x] = t; fetch.append(x); ev.append((x, t))
        T["l3_lookup"] = (tm() - t0) / 1e6
        t0 = tm()
        L2["qi"].append(self.IDX[q]); L2["t"].append(disc); L2["fc"].append(fc); L2["urls"].append(served)
        L1["qi"].append(self.IDX[q]); L1["t"].append(min(float(tc) for _, tc in ev) if (self.p2.L1_TE_PAGES_ONLY and ev) else min([float(tc) for _, tc in ev] + [float(disc)]))
        L1["fc"].append(fc); L1["q"].append(q); L1["ev"].append(ev)
        T["cache_write_update"] = (tm() - t0) / 1e6
        return {"tier": tier, "served": served, "ev": ev, "fetch": fetch, "l1_entry": None, "new_l1_entry": len(L1["qi"]) - 1, "timers_ms": T}
