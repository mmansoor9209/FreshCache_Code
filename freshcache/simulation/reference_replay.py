"""Standalone FreshCache replay: a transcription of the frozen audit_replay.replay / mixed_age_replay FreshCache_Full path (L1 eqgate gate + risk
temporal test, L2 eq_risk with alias registration, L3 risk content test, L1 t_e from supporting page versions, full REAL_TIME bypass), with
  - optional per-component timers (perf_counter_ns), and
  - optional per-tier capacity (entries/pages) with a FIXED least-recently-used eviction rule (a hit or write refreshes recency).
At unbounded capacity it must reproduce the frozen FreshCache replay exactly (gated by the caller). No threshold or setting is changed."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, time
import numpy as np

class Replay:
    def __init__(self, E, cap=None, timers=False):
        self.exp, self.p2 = E["exp"], E["p2"]; ex = self.exp
        self.SIM, self.IDX = ex._SIM_MATRIX, ex._QUERY_TO_IDX
        self.L1T = ex.L1_SIM_THRESHOLD; self.L2F = max(ex.L2_SIM_THRESHOLD, ex._L2_EQ_SIM_FLOOR)
        self.EPS = {"answer": ex.EPS_ANSWER, "url_list": ex.EPS_URL_LIST, "content": ex.EPS_CONTENT}
        self.cap = cap or {}; self.timers = timers
    def ok(self, fc, age, tier): return self.exp.p_stale(fc, age, tier) <= self.EPS[tier]
    def run(self, stream):
        n = len(stream); ex = self.exp; ok = self.ok; tm = time.perf_counter_ns
        L1 = {"qi": np.empty(n, np.int64), "t": np.empty(n, np.float64), "alive": np.zeros(n, bool), "fc": [], "q": [], "ev": [], "n": 0}
        L2 = {"qi": np.empty(n, np.int64), "t": np.empty(n, np.float64), "alive": np.zeros(n, bool), "fc": [], "urls": [], "n": 0}
        L3 = collections.OrderedDict(); r1, r2 = collections.OrderedDict(), collections.OrderedDict()
        c1, c2, c3 = self.cap.get("L1"), self.cap.get("L2"), self.cap.get("L3")
        cnt = collections.Counter(); tiers = {}; T = collections.defaultdict(list); peak = {"L1": 0, "L2": 0, "L3": 0}; ev_all = {}
        def evict(store, rec, cap):
            while cap is not None and len(rec) > cap:
                j, _ = rec.popitem(last=False); store["alive"][j] = False; cnt["evictions_" + ("L1" if store is L1 else "L2")] += 1
        for t, r in stream:
            q, fc, qid = r["query"], r["freshness_class"], r["query_id"]; own = [u["url_hash"] for u in r["urls"]]
            if fc == "REAL_TIME":
                cnt["search"] += 1; cnt["fetches"] += len(own); tiers[qid] = "REAL_TIME_BYPASS"; continue
            row = self.SIM[self.IDX[q]]
            # ---- L1 read
            t0 = tm(); hit = -1; n1 = L1["n"]
            if n1:
                sims = row[L1["qi"][:n1]]; cand = np.nonzero((sims >= self.L1T) & L1["alive"][:n1])[0]
                if cand.size:
                    okm = np.fromiter((ok(L1["fc"][int(j)], t - L1["t"][j], "answer") for j in cand), bool, cand.size); keep = cand[okm]
                    for j in keep[np.argsort(-sims[keep], kind="stable")]:
                        j = int(j)
                        if ex._entity_match(q, L1["q"][j]) and ex.semantic_equivalent(q, L1["q"][j], float(sims[j])): hit = j; break
            if self.timers: T["l1_lookup"].append((tm() - t0) / 1e6)
            if hit >= 0:
                cnt["l1_hits"] += 1; tiers[qid] = "L1"; r1.move_to_end(hit); ev_all[qid] = L1["ev"][hit]; continue
            # ---- L2 lookup
            t0 = tm(); hit2 = -1; n2 = L2["n"]
            if n2:
                s2 = row[L2["qi"][:n2]]; cand = np.nonzero((s2 >= self.L2F) & L2["alive"][:n2])[0]
                if cand.size:
                    okm = np.fromiter((ok(L2["fc"][int(j)], t - L2["t"][j], "url_list") for j in cand), bool, cand.size); keep = cand[okm]
                    if keep.size: hit2 = int(keep[np.argmax(s2[keep])])
            if self.timers: T["l2_lookup"].append((tm() - t0) / 1e6)
            if hit2 >= 0: cnt["l2_hits"] += 1; served, disc, tier = list(L2["urls"][hit2]), float(L2["t"][hit2]), "L2"; r2.move_to_end(hit2)
            else: cnt["search"] += 1; served, disc, tier = list(own), t, "miss"
            # ---- L3 per page
            t0 = tm(); ev, writes = [], []
            for x in served:
                tc = L3.get(x)
                if tc is not None and ok(fc, t - tc, "content"): cnt["l3_hits"] += 1; ev.append((x, tc)); L3.move_to_end(x); continue
                cnt["fetches"] += 1; L3[x] = t; L3.move_to_end(x); writes.append(x); ev.append((x, t))
                if c3 is not None:
                    while len(L3) > c3: L3.popitem(last=False); cnt["evictions_L3"] += 1
            if self.timers: T["l3_lookup_per_request"].append((tm() - t0) / 1e6); T["l3_pages_per_request"].append(len(served))
            # ---- writes: L2 registration + L1 insert
            t0 = tm()
            j = L2["n"]; L2["qi"][j] = self.IDX[q]; L2["t"][j] = disc; L2["alive"][j] = True; L2["fc"].append(fc); L2["urls"].append(served); L2["n"] += 1; r2[j] = True; evict(L2, r2, c2)
            k = L1["n"]; L1["qi"][k] = self.IDX[q]
            L1["t"][k] = (min(float(tc) for _, tc in ev) if (self.p2.L1_TE_PAGES_ONLY and ev) else min([float(tc) for _, tc in ev] + [float(disc)]))
            L1["alive"][k] = True; L1["fc"].append(fc); L1["q"].append(q); L1["ev"].append(ev); L1["n"] += 1; r1[k] = True; evict(L1, r1, c1)
            if self.timers: T["cache_write_update"].append((tm() - t0) / 1e6)
            tiers[qid] = tier; ev_all[qid] = ev
            peak["L1"] = max(peak["L1"], int(L1["alive"][:L1["n"]].sum()) if c1 is not None else L1["n"]); peak["L2"] = max(peak["L2"], int(L2["alive"][:L2["n"]].sum()) if c2 is not None else L2["n"]); peak["L3"] = max(peak["L3"], len(L3))
        m = {"n": n, "search_calls": cnt["search"], "search_saved_pct": round(100 * (1 - cnt["search"] / n), 4), "l1_hits": cnt["l1_hits"], "l2_hits": cnt["l2_hits"],
             "l3_hits": cnt["l3_hits"], "fetches": cnt["fetches"], "generations": cnt["search"] + cnt["l2_hits"],
             "evictions": {k: v for k, v in cnt.items() if k.startswith("evictions")}, "peak_entries": peak}
        self.state = {"L1": L1, "L2": L2, "L3": L3}
        return m, tiers, T, ev_all
