#!/usr/bin/env python3
"""
Section 4 -- the strongest NON-SYNTHETIC repeated workload available.

WHAT WAS SEARCHED FOR FIRST
  1. A real production request / query log inside the project. NONE EXISTS --
     logs/ contains only collection and training logs, and no file in the
     repository is a served-traffic trace. This is recorded, not worked around.
  2. Naturally repeated / evolving questions in the external benchmarks.
     DailyQA is the only source where the SAME natural-language question is
     genuinely re-asked at successive real timestamps with an independently
     collected answer at each one.

WHAT THIS WORKLOAD IS, STATED PRECISELY
  DailyQA: 8,360 distinct human-written questions, each re-asked on 29
  consecutive real calendar days (2025-01-04 .. 2025-02-01), with that day's
  answer collected independently. 242,440 requests. Repetition, timestamps and
  answer changes are all naturally occurring; NO paraphrase is generated for
  this experiment and none is used.

  It is an evolving-benchmark question set, not sampled production traffic: the
  arrival process is uniform (every question every day) rather than Zipfian, so
  it tests temporal reuse under natural repetition, NOT production popularity
  skew. That limit is reported, not papered over.

STRUCTURAL LIMIT
  DailyQA carries no retrieved URL lists or page bodies, so L2 and L3 are
  undefined and only answer-cache (L1) policies can be compared. Reported.

Policies: NoCache, ExactNoTTL, ExactTTL, SemanticTTL, FreshCache-L1Only,
FreshCache -- all with the published frozen parameters, no retuning.
Real sequential replay: every decision updates the cache state the later
requests see.
"""
from __future__ import annotations
import csv, json, math, os, pathlib, random, re, sys, time
from collections import Counter, defaultdict
from fractions import Fraction

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(BASE / "lib"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import v3_engine as v3                   # noqa: E402
from freshcache.risk_model import (      # noqa: E402
    _detect_answer_type, _detect_temporal_keywords, _classify_freshness)

DQ = ROOT / "external" / "DailyQA" / "data" / "qa"
SEED = 42
DAY = 86400.0
EMB = HERE / "dailyqa_bgem3.npy"
POLICIES = ["NoCache", "ExactNoTTL", "ExactTTL", "SemanticTTL",
            "FreshCache-L1Only", "FreshCache"]
_WS = re.compile(r"\s+")


def normalize(t):
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def answers_match(a, b):
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    if len(nb) <= 40 and nb in na:
        return True
    if len(na) <= 40 and na in nb:
        return True
    return False


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 3),
            round(100 * min(1.0, (c + m) / d), 3))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tot = sum(Fraction(math.comb(n, i)) for i in range(k + 1))
    return float(min(Fraction(1), 2 * tot / Fraction(2) ** n))


def load_days():
    days = []
    for p in sorted(DQ.glob("qa_*.jsonl")):
        by_q = {}
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = (d.get("query") or d.get("question") or "").strip()
            a = (d.get("answer") or "").strip()
            if q and a:
                by_q[q] = a
        days.append((p.stem, by_q))
    return days


def embed(questions, gpu):
    if EMB.exists():
        return np.load(EMB)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
    v = m.encode(questions, batch_size=128, normalize_embeddings=True,
                 show_progress_bar=False).astype(np.float32)
    np.save(EMB, v)
    return v


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    gpu = int(sys.argv[sys.argv.index("--gpu") + 1]) if "--gpu" in sys.argv else 3
    say("V3 SECTION 4 -- non-synthetic / natural repeated workload")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    say("  real production request log in this project: NONE FOUND "
        "(logs/ holds only collection and training logs)")
    say("  strongest available natural source: DailyQA consecutive-day repeats")

    days = load_days()
    qs = sorted({q for _, by in days for q in by})
    say(f"\n  == workload characterisation ==")
    say(f"    days                    {len(days)}  "
        f"({days[0][0]} .. {days[-1][0]})")
    say(f"    distinct questions      {len(qs):,}")
    nreq = sum(len(by) for _, by in days)
    say(f"    requests                {nreq:,}")
    rep = Counter()
    for q in qs:
        rep[sum(1 for _, by in days if q in by)] += 1
    repeated = sum(v for k, v in rep.items() if k > 1)
    say(f"    questions re-asked      {repeated:,} / {len(qs):,} = "
        f"{100*repeated/len(qs):.2f}%  (natural repeat rate)")
    say(f"    requests that are a natural repeat: "
        f"{nreq - len(qs):,} / {nreq:,} = {100*(nreq-len(qs))/nreq:.2f}%")
    say(f"    generated paraphrases   0  (none created for this experiment)")

    fcs = {q: _classify_freshness(q, _detect_temporal_keywords(q),
                                  _detect_answer_type(q)).value for q in qs}
    cd = Counter(fcs.values())
    say(f"    inference-time freshness classes  { {k: cd[k] for k in sorted(cd)} }")

    say(f"\n    embedding {len(qs):,} questions with BAAI/bge-m3 (GPU {gpu}) ...")
    V = embed(qs, gpu)
    say(f"    embeddings {V.shape}")
    idx = {q: i for i, q in enumerate(qs)}
    SIMM = V @ V.T
    np.fill_diagonal(SIMM, -1.0)
    nn = SIMM.max(axis=1)
    for th in (0.40, 0.75, 0.90):
        say(f"    semantic-neighbour rate at cos >= {th:.2f}: "
            f"{100*float((nn >= th).mean()):.2f}% of questions have a DISTINCT "
            f"question above threshold")
    np.fill_diagonal(SIMM, 1.0)

    # arrival construction: day d at t = d * 86400 s, order within a day is one
    # seeded permutation, fixed for every policy.
    rng = random.Random(SEED)
    stream = []
    for d, (name, by) in enumerate(days):
        items = sorted(by.items())
        rng.shuffle(items)
        for q, a in items:
            stream.append((d * DAY, q, a, fcs[q]))
    assert len(stream) == nreq
    say(f"    arrival construction    day d at t = d x 86400 s; within-day "
        f"order is one seed-{SEED} permutation, identical for every policy")
    say(f"    L2 / L3 activity, fetches, content drift: NOT DEFINED "
        f"(DailyQA has no URL lists or page bodies)")

    # ---------------- sequential replay ----------------
    THETA = exp.L1_SIM_THRESHOLD

    def replay(pol):
        """Real sequential replay: a reuse decision leaves the cache entry in
        place, a miss writes a fresh entry, so later requests see the state the
        policy actually produced. The similarity scan is vectorised (one numpy
        row gather per request) exactly as mixed_engine does; selection
        semantics are unchanged -- the served entry is the highest-similarity
        entry that passes the temporal test, ties going to cache order."""
        nmax = len(stream) + 8
        s_t = np.empty(nmax, np.float64)
        s_qi = np.empty(nmax, np.int64)
        s_hl = np.empty(nmax, np.float64)     # half-life of the cached class
        s_ttl = np.empty(nmax, np.float64)    # FIXED_TTL of the cached class
        s_a = []
        by_exact = defaultdict(list)
        n1 = 0
        reuse = corr = wai = 0
        vec = []
        semantic = pol in ("SemanticTTL", "FreshCache", "FreshCache-L1Only")
        risk = pol in ("FreshCache", "FreshCache-L1Only")
        c18 = risk          # FreshCache's L1 also applies the C18 semantic-
                            # equivalence + entity gate; SemanticTTL does not.
        s_q = []
        MUL = exp.TIER_MULT["answer"]
        LN2 = math.log(2.0)
        for t, q, gold, fc in stream:
            hit = -1
            if pol != "NoCache" and n1:
                if semantic:
                    sims = SIMM[idx[q], s_qi[:n1]]
                    # For the C18 arms the equivalence gate returns False
                    # whenever sim < exp._EQ_SIM_FLOOR, so pre-filtering the
                    # candidate set at that floor is EXACT, not an
                    # approximation -- it only skips candidates the gate would
                    # reject on its first comparison.
                    floor = max(THETA, exp._EQ_SIM_FLOOR) if c18 else THETA
                    cand = np.nonzero(sims >= floor)[0]
                    if cand.size:
                        age = t - s_t[cand]
                        if risk:
                            okm = (1.0 - np.exp(-LN2 * MUL * age / s_hl[cand])
                                   ) <= exp.EPS_ANSWER
                        else:
                            ttl = s_ttl[cand]
                            okm = (ttl > 0) & (age <= ttl)
                        keep = cand[okm]
                        if keep.size:
                            order = keep[np.argsort(-sims[keep], kind="stable")]
                            for j in order:
                                j = int(j)
                                if c18 and not (
                                        exp._entity_match(q, s_q[j])
                                        and exp.semantic_equivalent(
                                            q, s_q[j], float(sims[j]))):
                                    continue
                                hit = j
                                break
                else:
                    for j in reversed(by_exact.get(normalize(q), ())):
                        age = t - float(s_t[j])
                        if pol == "ExactNoTTL" or (s_ttl[j] > 0 and age <= s_ttl[j]):
                            hit = j
                            break
            if hit >= 0:
                reuse += 1
                ok = answers_match(s_a[hit], gold)
                corr += ok
                wai += int(not ok)
                vec.append(ok)
                continue
            s_t[n1] = t; s_qi[n1] = idx[q]
            s_hl[n1] = exp.HALF_LIFE.get(fc, 86400.0)
            s_ttl[n1] = exp.FIXED_TTL.get(fc, 0)
            s_a.append(gold); s_q.append(q)
            by_exact[normalize(q)].append(n1)
            n1 += 1
            corr += 1
            vec.append(True)
        return {"reuse": reuse, "correct": corr, "wai": wai}, vec

    out, per = {}, {}
    say(f"\n  {'policy':<20}{'reuse%':>9}{'saved%':>9}{'acc%':>9}"
        f"{'acc 95% CI':>18}{'CIE/query%':>12}{'C-WAI%':>9}{'hits':>10}")
    for pol in POLICIES:
        t0 = time.time()
        m, vec = replay(pol)
        per[pol] = vec
        n = len(stream)
        r = {"n_requests": n, "reuse": m["reuse"],
             "reuse_rate_pct": round(100 * m["reuse"] / n, 4),
             "search_saved_pct": round(100 * m["reuse"] / n, 4),
             "accuracy_pct": round(100 * m["correct"] / n, 4),
             "accuracy_ci": wilson(m["correct"], n),
             "cache_induced_error_n": m["wai"],
             "cache_induced_error_per_query_pct": round(100 * m["wai"] / n, 4),
             "wai_conditional_on_hit_pct": (round(100 * m["wai"] / m["reuse"], 4)
                                            if m["reuse"] else None),
             "cwai_ci": wilson(m["wai"], m["reuse"]) if m["reuse"] else None,
             "l1_hits": m["reuse"], "l2_hits": None, "l3_hits": None,
             "fetches": None, "content_drift_pct": None,
             "seconds": round(time.time() - t0, 1)}
        out[pol] = r
        cw = (f"{r['wai_conditional_on_hit_pct']:.2f}"
              if r["wai_conditional_on_hit_pct"] is not None else "-")
        say(f"  {pol:<20}{r['reuse_rate_pct']:>8.2f}%{r['search_saved_pct']:>8.2f}%"
            f"{r['accuracy_pct']:>8.2f}%{str(r['accuracy_ci']):>18}"
            f"{r['cache_induced_error_per_query_pct']:>11.2f}%{cw:>9}"
            f"{r['l1_hits']:>10,}")

    say(f"\n  exact McNemar vs NoCache (paired, per request)")
    base = per["NoCache"]
    for pol in POLICIES[1:]:
        v = per[pol]
        b = sum(1 for a, x in zip(base, v) if a and not x)
        c = sum(1 for a, x in zip(base, v) if x and not a)
        p = mcnemar_exact(b, c)
        out[pol]["mcnemar_vs_nocache"] = {"b": b, "c": c, "p": p}
        say(f"    {pol:<20} C->W {b:>7,}  W->C {c:>7,}  p = {p:.4g}")

    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source": "DailyQA consecutive-day repeats",
               "production_request_log_available": False,
               "paraphrases_generated": 0,
               "workload": {"days": len(days), "requests": nreq,
                            "unique_questions": len(qs),
                            "natural_repeat_rate_pct": round(100*repeated/len(qs), 2),
                            "repeat_request_fraction_pct": round(100*(nreq-len(qs))/nreq, 2),
                            "freshness_classes": dict(cd),
                            "semantic_neighbour_rate_pct": {
                                str(th): round(100*float((nn >= th).mean()), 2)
                                for th in (0.40, 0.75, 0.90)},
                            "arrival": f"day d at t=d*86400s; within-day order "
                                       f"is one seed-{SEED} permutation"},
               "structural_limits": {"L2_L3": "undefined: no URL lists or page "
                                              "bodies in DailyQA",
                                     "arrival_process": "uniform, not Zipfian; "
                                     "this is an evolving benchmark question "
                                     "set, not sampled production traffic"},
               "frozen_params": {"L1_SIM_THRESHOLD": THETA,
                                 "FIXED_TTL": dict(exp.FIXED_TTL),
                                 "HALF_LIFE": dict(exp.HALF_LIFE),
                                 "EPS_ANSWER": exp.EPS_ANSWER},
               "results": out}, open(HERE / "natural_results.json", "w"), indent=2)
    with open(HERE / "natural_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "n_requests", "reuse_rate_pct", "search_saved_pct",
                    "accuracy_pct", "acc_ci_lo", "acc_ci_hi",
                    "cache_induced_error_per_query_pct",
                    "wai_conditional_on_hit_pct", "l1_hits", "mcnemar_p"])
        for pol in POLICIES:
            r = out[pol]
            w.writerow([pol, r["n_requests"], r["reuse_rate_pct"],
                        r["search_saved_pct"], r["accuracy_pct"],
                        r["accuracy_ci"][0], r["accuracy_ci"][1],
                        r["cache_induced_error_per_query_pct"],
                        r["wai_conditional_on_hit_pct"], r["l1_hits"],
                        r.get("mcnemar_vs_nocache", {}).get("p", "")])
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("\n  wrote natural_results.json/.csv, run.log")


if __name__ == "__main__":
    main()
