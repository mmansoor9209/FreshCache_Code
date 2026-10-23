"""composition_engine_admission_hooks: composition_engine_resumable plus three optional hooks (all default off; with defaults it reproduces composition_engine_resumable request for request):
  l2_check(q, tent_ev, meta) -> bool   list-level admission (frozen gamma verifier or learned gate); a rejection falls back to the own search
  serve_check(q, src, t) -> bool       answer verifier before an answer-cache serve; a rejection makes the request an answer-cache miss
  write_check(q, ev, t) -> bool        answer verifier before an answer-cache write; a rejection withholds the write
Inspection counts are kept in self.c["l2_checks"], ["serve_checks"], ["write_checks"]. The composition_engine_resumable description follows.
composition_engine_resumable: the composition_engine replay logic (CategoryAware-style answer cache + L2 URL-list reuse + L3 page cache) recast as a resumable state
object so that a cache state can be cloned at a root request (2x2 root diagnostic). Decision logic is copied unchanged from composition_engine_write_mask /
composition_engine_reuse_write_suppression; replay gates require exact signature equality with the frozen B, D traces and with the D' and B-thin replays.

Write rules (what is written to the answer cache after a generation):
  "all"   every generated answer (B, D)
  "no_l2" not when the answer was generated from a reused URL list (D')
  "mask"  not when the request id is in `mask` (B-minus matched-mask diagnostic; = B-thin when mask = D' L2 set)
  "elig"  not when an L2 lookup on the policy's own URL-list registry finds an eligible list (B-minus, past-information rule; with l2_on=False
          the lookup is computed exactly as D would compute it, but the list is not used)
Every row carries lineage: root id of the answer-cache entry served (L1), write decision, and write opportunity."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import copy
import numpy as np
BASE_TTL = {"TIMELESS": 2592000, "SLOW": 604800, "MEDIUM": 86400, "FAST": 3600, "REAL_TIME": 0}


class State:
    def __init__(self, E, n_max, floor=0.80, theta_by_class=None, kappa_by_class=None, l2_on=True, write_rule="all", mask=frozenset(), l2_check=None, serve_check=None, write_check=None):
        exp = E["exp"]; self.exp = exp; self.L2FLOOR = max(exp.L2_SIM_THRESHOLD, floor)
        self.EPS = {"answer": exp.EPS_ANSWER, "url_list": exp.EPS_URL_LIST, "content": exp.EPS_CONTENT}
        self.SIM, self.IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
        self.th = theta_by_class; self.ttl = {c: BASE_TTL[c] * kappa_by_class.get(c, 0.0) for c in BASE_TTL}
        self.l2_on, self.write_rule, self.mask = l2_on, write_rule, mask
        self.l1_qi = np.empty(n_max, np.int64); self.l1_t = np.empty(n_max, np.float64); self.l1_fc, self.l1_src = [], []; self.n1 = 0
        self.l2_qi = np.empty(n_max, np.int64); self.l2_t = np.empty(n_max, np.float64); self.l2_fc, self.l2_urls, self.l2_root, self.l2_q = [], [], [], []; self.n2 = 0
        self.l3 = {}; self.c = dict(l1h=0, l2h=0, l3h=0, search=0, fetches=0, n=0, gens=0, l2_checks=0, l2_rejected=0, serve_checks=0, serve_rejected=0, write_checks=0, write_rejected=0)
        self.l2_check, self.serve_check, self.write_check = l2_check, serve_check, write_check

    def clone(self):
        s = copy.copy(self)
        for a in ("l1_qi", "l1_t", "l2_qi", "l2_t"): setattr(s, a, getattr(self, a).copy())
        for a in ("l1_fc", "l1_src", "l2_fc", "l2_urls", "l2_root", "l2_q"): setattr(s, a, list(getattr(self, a)))
        s.l3 = dict(self.l3); s.c = dict(self.c); return s

    def ok(self, fc, age, tier): return self.exp.p_stale(fc, age, tier) <= self.EPS[tier]

    def l3_pure(self, served, t, fc):
        local, ev, reused, tofetch = {}, [], 0, []
        for x in served:
            ct = local.get(x, self.l3.get(x))
            if ct is not None and self.ok(fc, t - ct, "content"): reused += 1; ev.append((x, ct))
            else: tofetch.append(x); local[x] = t; ev.append((x, t))
        return ev, reused, tofetch

    def l2_lookup(self, row, t):
        if not self.n2: return -1
        s2 = row[self.l2_qi[:self.n2]]; cand = np.nonzero(s2 >= self.L2FLOOR)[0]
        if not cand.size: return -1
        keep = cand[np.fromiter((self.ok(self.l2_fc[int(j)], t - self.l2_t[j], "url_list") for j in cand), bool, cand.size)]
        return int(keep[np.argmax(s2[keep])]) if keep.size else -1

    def step(self, t, r, force_evidence=None, force_write=None):
        """force_evidence in {None, 'fresh', 'reuse'} and force_write in {None, True, False} are used only at a diagnostic root (2x2 root diagnostic)."""
        q, fc, qid = r["query"], r["freshness_class"], r["query_id"]; qi = self.IDX[q]; row = self.SIM[qi]; own = [u["url_hash"] for u in r["urls"]]; c = self.c; c["n"] += 1
        if fc == "REAL_TIME":
            c["search"] += 1; c["fetches"] += len(own); c["gens"] += 1
            return {"tier": "REAL_TIME", "t": t, "ev": [(u, t) for u in own], "gen_query": q, "fetched": len(own), "searched": 1, "write_opp": False, "written": False, "root": qid}
        hit1 = -1
        if self.n1:
            sims = row[self.l1_qi[:self.n1]]; cand = np.nonzero(sims >= self.th[fc])[0]
            if cand.size:
                keep = cand[np.fromiter((bool(self.ttl.get(self.l1_fc[int(j)], 0) and t - self.l1_t[j] <= self.ttl.get(self.l1_fc[int(j)], 0)) for j in cand), bool, cand.size)]
                if keep.size: hit1 = int(keep[np.argsort(-sims[keep], kind="stable")][0])
        if hit1 >= 0 and force_evidence is None and self.serve_check is not None:
            c["serve_checks"] += 1
            if not self.serve_check(q, self.l1_src[hit1], t): c["serve_rejected"] += 1; hit1 = -1
        if hit1 >= 0 and force_evidence is None:
            c["l1h"] += 1; src = self.l1_src[hit1]
            return {"tier": "L1", "t": t, "similarity": float(row[self.l1_qi[hit1]]), "ev": src["ev"], "gen_query": src["query"], "stored_answer_t": src["t"], "entry_index": hit1,
                    "root": src["root"], "root_route": src["route"], "fetched": 0, "searched": 0, "write_opp": False, "written": False}
        cand2 = self.l2_lookup(row, t)
        hit2 = cand2 if (self.l2_on and force_evidence is None) else (cand2 if force_evidence == "reuse" else -1)
        if hit2 >= 0 and force_evidence is None and self.l2_check is not None:
            tent_ev, _, _ = self.l3_pure(list(self.l2_urls[hit2]), t, fc); c["l2_checks"] += 1
            meta = {"eq": self.l2_q[hit2], "sim": float(row[self.l2_qi[hit2]]), "fc": fc, "age": t - self.l2_t[hit2], "efc": self.l2_fc[hit2]}
            if not self.l2_check(q, tent_ev, meta): c["l2_rejected"] += 1; hit2 = -1
        if hit2 >= 0: c["l2h"] += 1; served = list(self.l2_urls[hit2]); disc = float(self.l2_t[hit2]); src_root = self.l2_root[hit2]
        else: c["search"] += 1; served = list(own); disc = t; src_root = qid
        self.l2_qi[self.n2] = qi; self.l2_t[self.n2] = disc; self.l2_fc.append(fc); self.l2_urls.append(served); self.l2_root.append(src_root); self.l2_q.append(q); self.n2 += 1
        ev, n_reused, tofetch = self.l3_pure(served, t, fc); c["l3h"] += n_reused; c["fetches"] += len(tofetch); c["gens"] += 1
        for x in tofetch: self.l3[x] = t
        route = "L2" if hit2 >= 0 else "miss"
        if force_write is None and self.write_check is not None:
            c["write_checks"] += 1
            if not self.write_check(q, ev, t): c["write_rejected"] += 1; force_write = False
        if force_write is not None: write = force_write
        elif self.write_rule == "all": write = True
        elif self.write_rule == "no_l2": write = hit2 < 0
        elif self.write_rule == "mask": write = qid not in self.mask
        elif self.write_rule == "elig": write = cand2 < 0
        else: raise ValueError(self.write_rule)
        if write:
            self.l1_qi[self.n1] = qi; self.l1_fc.append(fc); self.l1_src.append({"query": q, "ev": ev, "t": t, "root": qid, "route": route}); self.l1_t[self.n1] = t; self.n1 += 1
        return {"tier": route, "t": t, "similarity": float(row[self.l2_qi[hit2]]) if hit2 >= 0 else None, "served": served, "t_cached": disc, "list_root": src_root, "ev": ev,
                "l3_reused_pages": n_reused, "fetched": len(tofetch), "searched": int(hit2 < 0), "gen_query": q, "write_opp": True, "written": write, "l2_eligible": cand2 >= 0, "root": qid}

    def metrics(self):
        c = self.c; n = c["n"]
        return {"n_requests": n, "search_calls": c["search"], "search_saved_pct": round(100 * (1 - c["search"] / n), 4), "l1_hits": c["l1h"], "l2_hits": c["l2h"], "l3_hits": c["l3h"],
                "fetches": c["fetches"], "generations": c["gens"], "l2_checks": c["l2_checks"], "l2_rejected": c["l2_rejected"], "serve_checks": c["serve_checks"],
                "serve_rejected": c["serve_rejected"], "write_checks": c["write_checks"], "write_rejected": c["write_rejected"]}


def replay(E, stream, **kw):
    s = State(E, len(stream), **kw); rows = {}
    for t, r in stream: rows[r["query_id"]] = s.step(t, r)
    return s.metrics(), rows
