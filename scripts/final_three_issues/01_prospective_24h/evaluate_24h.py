#!/usr/bin/env python3
"""
Issue 1 -- prospective +24h evaluation.

Two label sources, one code path:

  --labels prospective   outcomes come from the REAL post-freeze pair
                         T0 -> T0_plus_24h. This is the prospective evaluation.
  --labels retrospective outcomes come from the existing rerun_24h round. This
                         mode exists only to PROVE this harness is correct: it
                         must reproduce the published fixed-24h FreshCache
                         anchor exactly. It is not a prospective result and is
                         never reported as one.

Policies, all with parameters frozen before T0 existed and never refit:
  FreshCache, FreshCache gate-off, SemanticTTL, ExactTTL.

Gate-off is realised by making the risk gate always pass (p_stale -> 0.0) and
nothing else; ExactTTL is a normalised-string exact-match L1 under the same
freshness-class TTL schedule, gate-checked against engine_all's SemanticTTL by
running the identical function in similarity mode.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, math, os, pathlib, random, re, sys
from collections import Counter, defaultdict

try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-final3")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
V3 = BASE.parent
ROOT = V3.parent
PROS = V3 / "remaining_critical_issues" / "05_prospective_temporal"
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import corrected_engine as ce            # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS
AGE = 86_400.0
SEED = 42
# published fixed-24h anchor for the full record set (engine_all, uniform 24h)
ANCHOR = {"search_saved_pct": 80.719, "l1_hits": 1164, "l2_hits": 24021,
          "l3_hits": 37840}
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s\.\?\!,;:]+$")


def norm_q(q):
    return _PUNCT.sub("", _WS.sub(" ", (q or "").strip().casefold()))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 4),
            round(100 * min(1.0, (c + m) / d), 4))


# ---------------------------------------------------------------- labels
def prospective_labels(a="T0", b="T0_plus_24h"):
    """CHANGED / UNCHANGED / UNOBS per url_hash from the real post-freeze pair.
    A URL is observable only when BOTH rounds produced a substantive body."""
    pa, pb = PROS / "snapshots" / f"{a}.jsonl", PROS / "snapshots" / f"{b}.jsonl"
    for p in (pa, pb):
        if not p.exists():
            sys.exit(f"ERROR: {p} does not exist. Collect it first; nothing is "
                     f"estimated in its absence.")
    A = {json.loads(l)["url_hash"]: json.loads(l)
         for l in open(pa, encoding="utf-8") if l.strip()}
    B = {json.loads(l)["url_hash"]: json.loads(l)
         for l in open(pb, encoding="utf-8") if l.strip()}
    lab, rows = {}, []
    for u, ra in A.items():
        rb = B.get(u)
        obs = bool(rb and ra.get("substantive") and rb.get("substantive"))
        if obs:
            lab[u] = (CHANGED if ra.get("content_hash") != rb.get("content_hash")
                      else UNCHANGED)
        else:
            lab[u] = UNOBS
        rows.append({"url_hash": u, "url": ra.get("url"),
                     "freshness_class": ra.get("freshness_class"),
                     "t0_available": bool(ra.get("snapshot_available")),
                     "t0_substantive": bool(ra.get("substantive")),
                     "t24_available": bool(rb and rb.get("snapshot_available")),
                     "t24_substantive": bool(rb and rb.get("substantive")),
                     "t0_hash": ra.get("content_hash"),
                     "t24_hash": rb.get("content_hash") if rb else None,
                     "t0_fetched_at": ra.get("fetched_at"),
                     "t24_fetched_at": rb.get("fetched_at") if rb else None,
                     "outcome": lab[u]})
    return lab, rows, {"a": str(pa.relative_to(ROOT)), "b": str(pb.relative_to(ROOT)),
                       "a_sha256": sha(pa), "b_sha256": sha(pb)}


# ------------------------------------------------------- exact-match L1
def replay_exact_l1(records, labels, sim_age, match="exact"):
    """engine_all.replay's L1 path for CONFIG['SemanticTTL'] (l1=sim_ttl,
    l2=none, l3=none, ttl=fixed, rt_bypass=True), transcribed, with the
    similarity test optionally replaced by normalised-string equality.

    Three details are mirrored exactly, because they are what makes this a
    transcription rather than a re-design:
      * the TTL is looked up on the INCOMING request's freshness class, not on
        the cached entry's class  (engine_all: t = ttl_for(fc, "answer", ...));
      * REAL_TIME requests skip the L1 LOOKUP but are still REGISTERED into L1;
      * registration is unconditional for sim_ttl (l1_ok is True).

    match="sim" reproduces engine_all.CONFIG['SemanticTTL'] exactly and is the
    fidelity gate for match="exact".
    """
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = exp.L1_SIM_THRESHOLD
    n = len(records)
    l1_qi = np.empty(n, np.int64)
    l1_urls = []
    by_key = defaultdict(list)
    n1 = l1h = search = fetches = 0
    per = []
    for r in records:
        q, fc = r["query"], r["freshness_class"]
        own = [u["url_hash"] for u in r["urls"]]
        rt_true = (fc == "REAL_TIME")
        hit = -1
        if not rt_true and n1:
            t = exp.FIXED_TTL.get(fc, 0)          # INCOMING class
            if t and sim_age <= t:
                if match == "sim":
                    sims = SIM[IDX[q]][l1_qi[:n1]]
                    cand = np.nonzero(sims >= L1T)[0]
                    if cand.size:
                        hit = int(cand[np.argsort(-sims[cand], kind="stable")][0])
                else:
                    k = by_key.get(norm_q(q))
                    if k:
                        hit = k[-1]
        if hit >= 0:
            l1h += 1
            per.append((r["query_id"], ea.score(l1_urls[hit], labels)))
        else:
            search += 1
            fetches += len(own)
            per.append((r["query_id"], None))
        # registration is unconditional, and happens on a hit too only if the
        # engine reaches the bottom of the loop -- it does not: engine_all
        # `continue`s on an L1 hit, so only misses register.
        if hit < 0:
            l1_qi[n1] = IDX[q]
            l1_urls.append(own)
            by_key[norm_q(q)].append(n1)
            n1 += 1
    return ({"search_saved_pct": round(100 * (1 - search / len(records)), 3),
             "search_calls": search, "l1_hits": l1h, "l2_hits": 0, "l3_hits": 0,
             "fetches": fetches, "generations": search,
             "n_requests": len(records)}, per)


# ---------------------------------------------------------------- scoring
def score(m, per, S=None):
    ins = [(q, o) for q, o in per if o is not None and (S is None or q in S)]
    ch = sum(1 for _, o in ins if o == CHANGED)
    un = sum(1 for _, o in ins if o == UNCHANGED)
    uo = sum(1 for _, o in ins if o == UNOBS)
    det = ch + un
    n = m["n_requests"]
    return {"search_saved_pct": m["search_saved_pct"],
            "searches": m["search_calls"], "fetches": m.get("fetches"),
            "generations": m.get("generations"),
            "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
            "l3_hits": m["l3_hits"],
            "reused_requests": det + uo, "determinable": det,
            "changed": ch, "unobservable": uo,
            "drift_pct": (100 * ch / det) if det else None,
            "drift_ci": wilson(ch, det) if det else None,
            "coverage_pct": (100 * det / n) if n else None,
            "n_requests": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["prospective", "retrospective"],
                    required=True)
    ap.add_argument("--a", default="T0")
    ap.add_argument("--b", default="T0_plus_24h")
    a = ap.parse_args()

    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    say(f"ISSUE 1 -- +24h evaluation   mode={a.labels}   utc={now}")

    man = json.load(open(PROS / "freeze_manifest.json", encoding="utf-8"))
    fp = man["frozen_parameters"]
    drift = {k: (dict(getattr(exp, k)) if isinstance(getattr(exp, k, None), dict)
                 else getattr(exp, k, None))
             for k, v in fp.items()
             if (dict(getattr(exp, k)) if isinstance(getattr(exp, k, None), dict)
                 else getattr(exp, k, None)) != v}
    if drift:
        sys.exit(f"ERROR: frozen parameters changed since the freeze: {drift}. "
                 f"A prospective evaluation is only valid against the frozen "
                 f"configuration, and nothing may be refit after the freeze.")
    say(f"  frozen parameters unchanged since {man['freeze_time_utc']}: True")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    say(f"  records {len(records):,} (the frozen query set)")

    url_rows, prov = [], {}
    if a.labels == "prospective":
        labels, url_rows, prov = prospective_labels(a.a, a.b)
        c = Counter(url_rows_i["outcome"] for url_rows_i in url_rows)
        att = len(url_rows)
        reach = sum(1 for r in url_rows if r["t0_available"] and r["t24_available"])
        obsv = sum(1 for r in url_rows if r["t0_substantive"] and r["t24_substantive"])
        say(f"\n  == prospective URL population ==")
        say(f"    attempted              {att:,}")
        say(f"    reached in both rounds {reach:,} ({100*reach/att:.2f}%)")
        say(f"    body-observable (both) {obsv:,} ({100*obsv/att:.2f}%)")
        say(f"    CHANGED {c[CHANGED]:,}   UNCHANGED {c[UNCHANGED]:,}   "
            f"UNOBSERVABLE {c[UNOBS]:,}")
        if obsv:
            say(f"    prospective 24 h URL change rate "
                f"{c[CHANGED]:,}/{obsv:,} = {100*c[CHANGED]/obsv:.4f}%  "
                f"CI{wilson(c[CHANGED], obsv)}")
        bycl = defaultdict(lambda: [0, 0])
        for r in url_rows:
            if r["outcome"] in (CHANGED, UNCHANGED):
                d = bycl[r["freshness_class"]]
                d[0] += 1; d[1] += int(r["outcome"] == CHANGED)
        say(f"    class-wise URL change rate")
        for k in sorted(bycl, key=str):
            n_, ch_ = bycl[k]
            say(f"      {str(k):<12}{ch_:>6,}/{n_:<6,} = {100*ch_/n_:>7.4f}%  "
                f"CI{wilson(ch_, n_)}")
    else:
        labels = ce.load_labels("rerun_24h")
        say(f"\n  == retrospective labels (HARNESS VALIDATION ONLY) ==")
        say(f"    rerun_24h labels loaded: {len(labels):,}")

    # ---------------- policies ----------------
    say(f"\n  == policies (all parameters frozen before T0 existed) ==")
    out = {}
    m, per = ea.replay(records, labels, AGE, "FreshCache_Full", rich=rich)
    out["FreshCache"] = score(m, per)
    if a.labels == "retrospective":
        bad = [k for k, v in ANCHOR.items()
               if abs(out["FreshCache"][k] - v) > (1e-3 if isinstance(v, float) else 0)]
        if bad:
            say(f"    HARNESS GATE FAILED on {bad}: "
                f"{ {k: out['FreshCache'][k] for k in ANCHOR} }")
            sys.exit(2)
        say(f"    HARNESS GATE PASSED: reproduces the published fixed-24h "
            f"anchor (saved {ANCHOR['search_saved_pct']}%, L1 {ANCHOR['l1_hits']:,}, "
            f"L2 {ANCHOR['l2_hits']:,}, L3 {ANCHOR['l3_hits']:,})")

    _p = exp.p_stale
    exp.p_stale = lambda fc, age, tier="content": 0.0
    try:
        m, per_off = ea.replay(records, labels, AGE, "FreshCache_Full", rich=rich)
        out["FreshCache_gate_off"] = score(m, per_off)
    finally:
        exp.p_stale = _p

    m, per = ea.replay(records, labels, AGE, "SemanticTTL", rich=rich)
    out["SemanticTTL"] = score(m, per)
    per_st = per

    ms, pers = replay_exact_l1(records, labels, AGE, "sim")
    ref = out["SemanticTTL"]
    dif = sum(1 for (qa, oa), (qb, ob) in zip(pers, per_st) if oa != ob)
    assert ms["l1_hits"] == ref["l1_hits"] \
        and abs(ms["search_saved_pct"] - ref["search_saved_pct"]) < 1e-9 \
        and dif == 0, (
        f"FIDELITY FAILURE: transcribed L1 cache != SemanticTTL "
        f"(L1 {ms['l1_hits']} vs {ref['l1_hits']}, "
        f"saved {ms['search_saved_pct']} vs {ref['search_saved_pct']}, "
        f"{dif} per-request outcome mismatches)")
    say(f"    fidelity gate PASSED: transcribed exact-match engine reproduces "
        f"engine_all SemanticTTL exactly in similarity mode")
    me_, pere = replay_exact_l1(records, labels, AGE, "exact")
    out["ExactTTL"] = score(me_, pere)

    per_on = {q: o for q, o in ea.replay(records, labels, AGE,
                                         "FreshCache_Full", rich=rich)[1]}
    per_offd = {q: o for q, o in per_off}

    say(f"\n  {'policy':<22}{'saved%':>10}{'drift%':>9}{'cov%':>9}{'det':>8}"
        f"{'chg':>7}{'unobs':>8}{'L1':>8}{'L2':>8}{'L3':>8}")
    for k in ("FreshCache", "FreshCache_gate_off", "SemanticTTL", "ExactTTL"):
        r = out[k]
        d = f"{r['drift_pct']:>8.4f}" if r["drift_pct"] is not None else "       -"
        cv = f"{r['coverage_pct']:>8.4f}" if r["coverage_pct"] is not None else "       -"
        say(f"  {k:<22}{r['search_saved_pct']:>9.4f}%{d}%{cv}%"
            f"{r['determinable']:>8,}{r['changed']:>7,}{r['unobservable']:>8,}"
            f"{r['l1_hits']:>8,}{r['l2_hits']:>8,}{r['l3_hits']:>8,}")

    # ---------------- paired common support, gate on vs off ----------------
    comm = [q for q in per_on
            if per_on[q] in (CHANGED, UNCHANGED)
            and per_offd.get(q) in (CHANGED, UNCHANGED)]
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    con = sum(1 for q in comm if per_on[q] == CHANGED)
    coff = sum(1 for q in comm if per_offd[q] == CHANGED)
    delta = (100 * coff / len(comm) - 100 * con / len(comm)) if comm else None
    say(f"\n  == gate ON vs OFF, paired common support ==")
    say(f"    n = {len(comm):,}   gate ON {100*con/max(1,len(comm)):.4f}%   "
        f"gate OFF {100*coff/max(1,len(comm)):.4f}%   "
        f"delta {delta:+.4f} pp" if comm else "    no common support")
    boot = None
    if comm:
        byc = defaultdict(list)
        for q in comm:
            byc[cid[q]].append(q)
        ks = list(byc)
        rng = random.Random(SEED)
        ds = []
        for _ in range(10_000):
            s = [byc[ks[rng.randrange(len(ks))]] for _ in range(len(ks))]
            qs = [q for g in s for q in g]
            if not qs:
                continue
            ds.append(100 * sum(1 for q in qs if per_offd[q] == CHANGED) / len(qs)
                      - 100 * sum(1 for q in qs if per_on[q] == CHANGED) / len(qs))
        ds.sort()
        boot = {"resamples": len(ds), "clusters": len(ks), "seed": SEED,
                "delta_pp": round(delta, 4),
                "ci95": [round(ds[int(.025 * len(ds))], 4),
                         round(ds[int(.975 * len(ds))], 4)]}
        say(f"    paired CLUSTER bootstrap ({len(ks):,} clusters, "
            f"{len(ds):,} resamples, seed {SEED}): "
            f"delta {delta:+.4f} pp  95% CI [{boot['ci95'][0]:+.4f}, "
            f"{boot['ci95'][1]:+.4f}] pp")

    allowed = (a.labels == "prospective")
    say(f"\n  key claim -- \"All model, threshold and calibration parameters "
        f"were frozen before the +24h Web state existed\": "
        f"{'ALLOWED' if allowed else 'NOT ALLOWED (retrospective labels)'}")

    summ = {"utc": now, "mode": a.labels, "age_seconds": AGE, "seed": SEED,
            "freeze_time_utc": man["freeze_time_utc"],
            "frozen_parameters": fp, "label_provenance": prov,
            "policies": out,
            "paired_common_support": (
                {"n": len(comm), "gate_on_drift_pct": 100*con/len(comm),
                 "gate_off_drift_pct": 100*coff/len(comm),
                 "delta_pp": delta} if comm else None),
            "paired_cluster_bootstrap": boot,
            "prospective_claim_allowed": allowed}
    json.dump(summ, open(HERE / f"summary_{a.labels}.json", "w"), indent=2)
    if url_rows:
        with open(HERE / "per_url.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(url_rows[0]))
            w.writeheader(); w.writerows(url_rows)
    with open(HERE / f"per_request_{a.labels}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "cluster_id", "freshness_class",
                    "gate_on_outcome", "gate_off_outcome", "in_common_support"])
        cs = set(comm)
        for r in records:
            q = r["query_id"]
            w.writerow([q, cid[q], r["freshness_class"],
                        per_on.get(q), per_offd.get(q), int(q in cs)])
    (HERE / f"run_{a.labels}.log").write_text("\n".join(log) + "\n",
                                              encoding="utf-8")
    say(f"  wrote summary_{a.labels}.json, per_request_{a.labels}.csv"
        + (", per_url.csv" if url_rows else ""))


if __name__ == "__main__":
    main()
