#!/usr/bin/env python3
"""
scripts/baselines/exact_ttl_and_l1_only/exact_ttl_l1_only_policies.py

Two additional baseline policies, added WITHOUT touching any existing
implementation file.

  FreshCache-L1Only   FreshCache's EXACT original L1 (candidate retrieval,
                      similarity floor, semantic equivalence + entity gate, freshness
                      class, exponential temporal eligibility, timestamps,
                      REAL_TIME bypass) with L2 and L3 reuse disabled. On an L1
                      miss it performs a normal fresh search, fresh page
                      processing and generation, and writes L1 normally.

                      Implemented by REGISTERING a new configuration into the
                      in-memory method_replay.CONFIG / mixed_age_replay.VARIANTS
                      dictionaries, so the recorded replay code executes it
                      verbatim. No source file is modified; audit.py's SHA-256
                      list is re-verified by every caller.

  ExactTTL            A conventional cache: reuse a stored answer only when the
                      NORMALISED incoming query string is byte-identical to a
                      previously cached query, subject to the same
                      freshness-class TTL schedule SemanticTTL uses
                      (tiered_cache_model.FIXED_TTL, keyed on the CACHED entry's class,
                      exactly as method_replay's sim_ttl does). No semantic
                      aliasing, no embeddings.
  ExactNoTTL          The same, with the temporal gate removed entirely.

  ExactTTL/ExactNoTTL cannot be expressed in the recorded engine's `l1` modes,
  so replay_l1cache() below is a faithful transcription of mixed_age_replay.replay
  restricted to the L1-only path. FIDELITY GATE: the same function, run with
  match="sim", reproduces mixed_age_replay.replay(..., "SemanticTTL") exactly --
  identical metrics AND identical per-request outcome vector. assert_fidelity()
  enforces this before any Exact* number is produced.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib, re, sys

ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()   # copy relocated
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))

import numpy as np                       # noqa: E402
import tiered_cache_model as exp                 # noqa: E402
import method_replay as ea                  # noqa: E402
import mixed_age_replay as me
_AGG = getattr(me, 'aggregate', None) or (lambda labs: (CHANGED if CHANGED in labs else (UNCHANGED if UNCHANGED in labs else UNOBS)) if labs else None)   # noqa: E402
import execution_semantics as ce            # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS

L1ONLY_CONFIG = dict(l1="eqgate_risk", l2="none", l3="none",
                     l2_reg_on_hit=False, ttl=None, online=False)


def register():
    """Add FreshCache-L1Only to the in-memory policy tables (idempotent)."""
    fc = ea.CONFIG["FreshCache_Full"]
    # the L1 half must be byte-identical to FreshCache's
    assert L1ONLY_CONFIG["l1"] == fc["l1"], "L1 mode diverges from FreshCache"
    assert fc.get("rt_bypass", True) == L1ONLY_CONFIG.get("rt_bypass", True), \
        "REAL_TIME bypass diverges from FreshCache"
    ea.CONFIG.setdefault("FreshCache_L1Only", dict(L1ONLY_CONFIG))
    me.VARIANTS.setdefault("FreshCache_L1Only", ("FreshCache_L1Only", "risk"))
    return ea.CONFIG["FreshCache_L1Only"]


# --------------------------------------------------------------- ExactTTL
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s\.\?\!,;:]+$")


def normalize(q: str) -> str:
    """Query normalisation for the exact-match cache: casefold, collapse
    internal whitespace, drop trailing punctuation. Declared once here and used
    for nothing else."""
    return _PUNCT.sub("", _WS.sub(" ", (q or "").strip().casefold()))


def replay_l1cache(stream, rounds, match="exact", ttl="fixed"):
    """
    mixed_age_replay.replay's L1-only path, transcribed.

      match "sim"   candidate set = {j : cos(q, q_j) >= exp.L1_SIM_THRESHOLD}
                    served = highest similarity passing the temporal test
                    (this is method_replay's sim_ttl -> SemanticTTL)
      match "eqgate"   the same candidate set, then FreshCache's semantic equivalence
                    and entity gate applied in descending-similarity order
                    (this is method_replay's eqgate_risk -> FreshCache-L1Only)
      match "exact" candidate set = {j : normalize(q) == normalize(q_j)}
                    served = the most recently registered passing entry

      ttl   "fixed" age <= exp.FIXED_TTL[cached entry's class], and the class
                    must have a non-zero TTL  (identical to method_replay sim_ttl)
            "risk"  exp.p_stale(class, age, "answer") <= exp.EPS_ANSWER
                    (identical to method_replay's risk gate)
            "none"  no temporal gate at all

    REAL_TIME bypass is on, as it is for every TTL-family policy in method_replay.
    """
    assert match in ("sim", "eqgate", "exact") and ttl in ("fixed", "risk", "none")
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = exp.L1_SIM_THRESHOLD
    n_max = len(stream)

    l1_qi = np.empty(n_max, np.int64)
    l1_t = np.empty(n_max, np.float64)
    l1_fc, l1_urls, l1_key, l1_q = [], [], [], []
    by_key = {}                       # normalised query -> [cache slots]
    n1 = 0
    l1h = search = fetches = 0
    rt_requests = 0
    per_req, rows = [], {}

    def temporal_ok(fcj, age):
        if ttl == "none":
            return True
        if ttl == "risk":
            return exp.p_stale(fcj, age, "answer") <= exp.EPS_ANSWER
        v = exp.FIXED_TTL.get(fcj, 0)
        return bool(v and age <= v)

    for t, r in stream:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        own = [u["url_hash"] for u in urls]

        if fc == "REAL_TIME":                       # rt_bypass=True
            rt_requests += 1
            search += 1
            fetches += len(own)
            per_req.append((qid, None))
            rows[qid] = {"tier": "REAL_TIME", "t": t, "own": own,
                         "ev": [(u, t) for u in own]}
            continue

        hit1 = -1
        sim1 = None
        if match in ("sim", "eqgate"):
            if n1:
                sims = SIM[IDX[q]][l1_qi[:n1]]
                cand = np.nonzero(sims >= L1T)[0]
                if cand.size:
                    okm = np.fromiter(
                        (temporal_ok(l1_fc[int(j)], t - l1_t[j]) for j in cand),
                        bool, cand.size)
                    keep = cand[okm]
                    if keep.size:
                        for j in keep[np.argsort(-sims[keep], kind="stable")]:
                            j = int(j)
                            if match == "eqgate" and not (
                                    exp._entity_match(q, l1_q[j])
                                    and exp.semantic_equivalent(
                                        q, l1_q[j], float(sims[j]))):
                                continue
                            hit1 = j
                            sim1 = float(sims[j])
                            break
        else:
            k = normalize(q)
            for j in reversed(by_key.get(k, ())):
                if temporal_ok(l1_fc[j], t - float(l1_t[j])):
                    hit1 = j
                    sim1 = 1.0
                    break

        if hit1 >= 0:
            l1h += 1
            labs = [me.outcome(x, float(l1_t[hit1]), t, rounds)
                    for x in l1_urls[hit1]]
            o = (_AGG(labs) if _AGG(labs) is not None else UNOBS)   # engine aggregation (keyset rule on the keyset engine, three-valued rule on the replay engine)
            per_req.append((qid, o))
            src = rows[l1_key[hit1]]
            rows[qid] = {"tier": "HIT", "t": t, "similarity": sim1,
                         "t_cached": float(l1_t[hit1]),
                         "age": t - float(l1_t[hit1]),
                         "matched_query": src["query"],
                         "stored_answer_query": src["query"],
                         "stored_answer_ev": src["ev"],
                         "stored_answer_t": src["t"], "own": own}
            continue

        search += 1
        fetches += len(own)
        per_req.append((qid, None))
        rows[qid] = {"tier": "miss", "t": t, "similarity": None, "age": 0.0,
                     "query": q, "ev": [(u, t) for u in own], "own": own}
        l1_qi[n1] = IDX[q]
        l1_t[n1] = t
        l1_fc.append(fc)
        l1_q.append(q)
        l1_urls.append(list(own))
        l1_key.append(qid)
        by_key.setdefault(normalize(q), []).append(n1)
        n1 += 1

    n = len(stream)
    m = {"variant": f"{match}/{ttl}", "n_requests": n, "search_calls": search,
         "search_saved_pct": round(100 * (1 - search / n), 4),
         "search_avoided": n - search, "l1_hits": l1h, "l2_hits": 0,
         "l3_hits": 0, "fetches": fetches, "conditional_gets": 0,
         "generations": search, "realtime_requests": rt_requests}
    return m, per_req, rows


def _cmp(m, per, ref_m, ref_per):
    bad = []
    for k in ("n_requests", "search_calls", "search_avoided", "l1_hits",
              "fetches", "realtime_requests"):
        if m[k] != ref_m[k]:
            bad.append(f"{k}: {m[k]} != {ref_m[k]}")
    if abs(m["search_saved_pct"] - ref_m["search_saved_pct"]) > 1e-9:
        bad.append(f"saved: {m['search_saved_pct']} != {ref_m['search_saved_pct']}")
    if len(per) != len(ref_per):
        bad.append("per-request length")
    else:
        d = sum(1 for a, b in zip(per, ref_per) if a != b)
        if d:
            bad.append(f"{d} per-request outcome mismatches")
    return bad


def assert_fidelity(stream, rounds):
    """The transcription must reproduce the recorded engine exactly, on
    metrics AND on the per-request outcome vector, for BOTH the similarity
    (SemanticTTL) and the semantic (FreshCache-L1Only) L1 modes."""
    register()
    bad = []
    ref_m, ref_per = me.replay(stream, rounds, "SemanticTTL")
    m, per, _ = replay_l1cache(stream, rounds, match="sim", ttl="fixed")
    bad += [f"SemanticTTL/{x}" for x in _cmp(m, per, ref_m, ref_per)]
    ref_m2, ref_per2 = me.replay(stream, rounds, "FreshCache_L1Only")
    m2, per2, _ = replay_l1cache(stream, rounds, match="eqgate", ttl="risk")
    bad += [f"L1Only/{x}" for x in _cmp(m2, per2, ref_m2, ref_per2)]
    assert not bad, "FIDELITY FAILURE: " + "; ".join(bad)
    return {"SemanticTTL": ref_m, "FreshCache_L1Only": ref_m2}
