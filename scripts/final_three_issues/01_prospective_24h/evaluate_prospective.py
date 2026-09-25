#!/usr/bin/env python3
"""
TASKS 3 and 4 -- the frozen prospective +24h evaluation and the temporal-gate
effect, on a Web state that did not exist when the parameters were chosen.

Labels come ONLY from the real post-freeze pair T0 -> T0_plus_24h. A URL is
observable only when BOTH rounds produced a substantive body; missing content
is UNOBSERVABLE and is NEVER counted as unchanged.

Policies, all at their frozen previously specified parameters:
  NoCache, FreshCache, FreshCache without temporal eligibility, SemanticTTL,
  ExactTTL.

Nothing is optimised, refit or selected using the +24h observations. The script
aborts if any frozen parameter has moved since the freeze.

  python evaluate_prospective.py                 # prospective (default)
  python evaluate_prospective.py --labels retrospective   # harness check only
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, math, os, pathlib, random, re, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
F3 = HERE.parent
V3 = F3.parent
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
SEED, B = 42, 10_000
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


def prospective_labels():
    p = HERE / "snapshots" / "T0_plus_24h.jsonl"
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist. Collect it first with "
                 f"collect_plus24h.py. Nothing is estimated in its absence.")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    lab = {}
    for r in rows:
        if r.get("observable"):
            lab[r["url_hash"]] = CHANGED if r["content_changed"] else UNCHANGED
        else:
            lab[r["url_hash"]] = UNOBS
    return lab, rows, sha(p)


def replay_exact_l1(records, labels, sim_age, match="exact"):
    """engine_all's sim_ttl L1 path, transcribed (TTL on the INCOMING class,
    REAL_TIME skips the lookup but still registers, registration only on a
    miss). match='sim' reproduces engine_all SemanticTTL exactly and is the
    fidelity gate for match='exact'."""
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
        hit = -1
        if fc != "REAL_TIME" and n1:
            t = exp.FIXED_TTL.get(fc, 0)
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
            l1_qi[n1] = IDX[q]
            l1_urls.append(own)
            by_key[norm_q(q)].append(n1)
            n1 += 1
    return ({"search_saved_pct": round(100 * (1 - search / len(records)), 3),
             "search_calls": search, "l1_hits": l1h, "l2_hits": 0, "l3_hits": 0,
             "fetches": fetches, "generations": search,
             "n_requests": len(records)}, per)


def score(m, per, records, nocache_fetches=None):
    fc_of = {r["query_id"]: r["freshness_class"] for r in records}
    ins = [(q, o) for q, o in per if o is not None]
    ch = sum(1 for _, o in ins if o == CHANGED)
    un = sum(1 for _, o in ins if o == UNCHANGED)
    uo = sum(1 for _, o in ins if o == UNOBS)
    det = ch + un
    n = m["n_requests"]
    byc = defaultdict(lambda: [0, 0, 0])
    for q, o in ins:
        d = byc[fc_of.get(q, "?")]
        d[0] += int(o == CHANGED)
        d[1] += int(o == UNCHANGED)
        d[2] += int(o == UNOBS)
    cls = {}
    for k, (c_, u_, o_) in byc.items():
        dd = c_ + u_
        cls[k] = {"changed": c_, "unchanged": u_, "unobservable": o_,
                  "determinable": dd,
                  "drift_pct": (round(100 * c_ / dd, 4) if dd else None),
                  "drift_ci": wilson(c_, dd) if dd else None,
                  "coverage_pct": (round(100 * dd / sum(1 for q, _ in ins
                                                        if fc_of.get(q) == k), 4)
                                   if any(fc_of.get(q) == k for q, _ in ins) else None)}
    return {"n_requests": n, "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
            "l3_hits": m["l3_hits"], "searches": m["search_calls"],
            "fetches": m["fetches"], "generations": m.get("generations"),
            "search_saved_pct": m["search_saved_pct"],
            "fetch_saved_pct": (round(100 * (1 - m["fetches"] / nocache_fetches), 4)
                                if nocache_fetches else 0.0),
            "reused_requests": det + uo, "determinable": det,
            "changed": ch, "unobservable": uo,
            "drift_pct": (round(100 * ch / det, 4) if det else None),
            "drift_ci": wilson(ch, det) if det else None,
            "coverage_pct": (round(100 * det / n, 4) if n else None),
            "class_wise": cls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["prospective", "retrospective"],
                    default="prospective")
    a = ap.parse_args()
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    say(f"TASKS 3 & 4 -- frozen prospective +24h evaluation   mode={a.labels}")
    say(f"  utc {now}")

    man = json.load(open(PROS / "freeze_manifest.json", encoding="utf-8"))
    fp = man["frozen_parameters"]
    drift = {k: v for k, v in fp.items()
             if (dict(getattr(exp, k)) if isinstance(getattr(exp, k, None), dict)
                 else getattr(exp, k, None)) != v}
    if drift:
        sys.exit(f"ERROR: frozen parameters changed since the freeze: "
                 f"{list(drift)}. Nothing may be refit after the freeze.")
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
    say(f"  frozen request stream {len(records):,} records "
        f"(same queries, same ordering)")

    urlrows, lab_sha = [], None
    if a.labels == "prospective":
        labels, urlrows, lab_sha = prospective_labels()
        el = [r["elapsed_hours"] for r in urlrows if r["elapsed_hours"] is not None]
        obs = [r for r in urlrows if r["observable"]]
        chg = sum(1 for r in obs if r["content_changed"])
        say(f"\n  == prospective URL population (post-freeze pair) ==")
        say(f"    T0_plus_24h sha256 {lab_sha}")
        say(f"    URLs attempted        {len(urlrows):,}")
        say(f"    min elapsed           {min(el):.4f} h   max {max(el):.4f} h")
        say(f"    reached at T+24h      {sum(1 for r in urlrows if r['snapshot_available']):,}")
        say(f"    substantive at T+24h  {sum(1 for r in urlrows if r['substantive']):,}")
        say(f"    OBSERVABLE (both)     {len(obs):,}")
        say(f"    observed changed      {chg:,} / {len(obs):,} = "
            f"{100*chg/max(1,len(obs)):.4f}%  CI{wilson(chg, len(obs))}")
        byc = defaultdict(lambda: [0, 0])
        for r in obs:
            d = byc[r.get("freshness_class") or "?"]
            d[0] += 1
            d[1] += int(bool(r["content_changed"]))
        say(f"    class-wise URL change rate (observable URLs)")
        for k in sorted(byc, key=str):
            n_, c_ = byc[k]
            say(f"      {str(k):<12}{c_:>6,}/{n_:<6,} = {100*c_/n_:>7.4f}%  "
                f"CI{wilson(c_, n_)}")
    else:
        labels = ce.load_labels("rerun_24h")
        say(f"\n  == retrospective labels (HARNESS VALIDATION ONLY) ==")

    say(f"\n  == policies (frozen parameters; nothing tuned on +24h) ==")
    out = {}
    m, per = ea.replay(records, labels, AGE, "NoCache", rich=rich)
    nc = score(m, per, records)
    NCF = m["fetches"]
    out["NoCache"] = score(m, per, records, NCF)

    m, per_on = ea.replay(records, labels, AGE, "FreshCache_Full", rich=rich)
    out["FreshCache"] = score(m, per_on, records, NCF)
    if a.labels == "retrospective":
        bad = [k for k, v in ANCHOR.items()
               if abs(out["FreshCache"][k] - v) > (1e-3 if isinstance(v, float) else 0)]
        if bad:
            say(f"    HARNESS GATE FAILED on {bad}"); sys.exit(2)
        say(f"    HARNESS GATE PASSED: reproduces the published fixed-24h anchor")

    _p = exp.p_stale
    exp.p_stale = lambda fc, age, tier="content": 0.0
    try:
        m, per_off = ea.replay(records, labels, AGE, "FreshCache_Full", rich=rich)
        out["FreshCache_no_temporal_gate"] = score(m, per_off, records, NCF)
    finally:
        exp.p_stale = _p
    assert exp.p_stale is _p

    m, per_st = ea.replay(records, labels, AGE, "SemanticTTL", rich=rich)
    out["SemanticTTL"] = score(m, per_st, records, NCF)
    ms, pers = replay_exact_l1(records, labels, AGE, "sim")
    dif = sum(1 for (_, oa), (_, ob) in zip(pers, per_st) if oa != ob)
    assert ms["l1_hits"] == out["SemanticTTL"]["l1_hits"] and dif == 0, (
        f"FIDELITY FAILURE: transcribed L1 cache != SemanticTTL "
        f"(L1 {ms['l1_hits']} vs {out['SemanticTTL']['l1_hits']}, {dif} mismatches)")
    say(f"    fidelity gate PASSED: transcribed exact-match engine reproduces "
        f"engine_all SemanticTTL exactly")
    me_, pere = replay_exact_l1(records, labels, AGE, "exact")
    out["ExactTTL"] = score(me_, pere, records, NCF)

    ORDER = ["NoCache", "ExactTTL", "SemanticTTL",
             "FreshCache_no_temporal_gate", "FreshCache"]
    say(f"\n  {'policy':<30}{'reqs':>8}{'L1':>8}{'L2':>8}{'L3':>8}{'srch':>8}"
        f"{'fetch':>8}{'gen':>8}{'saved%':>9}{'fsave%':>9}{'drift%':>9}{'cov%':>9}")
    for k in ORDER:
        r = out[k]
        d = f"{r['drift_pct']:>8.4f}" if r["drift_pct"] is not None else f"{'-':>9}"
        say(f"  {k:<30}{r['n_requests']:>8,}{r['l1_hits']:>8,}{r['l2_hits']:>8,}"
            f"{r['l3_hits']:>8,}{r['searches']:>8,}{r['fetches']:>8,}"
            f"{r['generations']:>8,}{r['search_saved_pct']:>8.4f}%"
            f"{r['fetch_saved_pct']:>8.4f}%{d}%{(r['coverage_pct'] or 0):>8.4f}%")

    say(f"\n  class-wise drift (determinable reuses only; unobservable never "
        f"counted as unchanged)")
    say(f"    {'policy':<30}" + "".join(f"{c:>22}" for c in
                                        ("TIMELESS", "SLOW", "MEDIUM", "FAST")))
    for k in ORDER:
        cw = out[k]["class_wise"]
        cells = []
        for c in ("TIMELESS", "SLOW", "MEDIUM", "FAST"):
            e = cw.get(c)
            cells.append(f"{e['drift_pct']:.3f}% ({e['determinable']:,})"
                         if e and e["drift_pct"] is not None else "-")
        say(f"    {k:<30}" + "".join(f"{c:>22}" for c in cells))

    # ---------------- TASK 4: paired temporal-gate effect ----------------
    ON = dict(per_on)
    OFF = dict(per_off)
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    fco = {r["query_id"]: r["freshness_class"] for r in records}
    pair_rows = []
    for r in records:
        q = r["query_id"]
        aa, bb = ON.get(q), OFF.get(q)
        both = aa is not None and bb is not None
        cell = ("not_paired" if not both else
                "both_observable" if aa != UNOBS and bb != UNOBS else
                "on_only_observable" if aa != UNOBS else
                "off_only_observable" if bb != UNOBS else "neither_observable")
        pair_rows.append({"query_id": q, "cluster_id": cid[q],
                          "freshness_class": fco.get(q),
                          "outcome_gate_on": aa, "outcome_gate_off": bb,
                          "paired": int(both), "cell": cell})
    cells = Counter(r["cell"] for r in pair_rows)
    jo = [r for r in pair_rows
          if r["cell"] == "both_observable"]
    say(f"\n  == TASK 4: paired temporal-gate effect ==")
    say(f"    Delta = drift(no temporal gate) - drift(temporal gate), on "
        f"MATCHED requests where BOTH arms reused content AND both outcomes "
        f"are observable")
    for k in ("both_observable", "on_only_observable", "off_only_observable",
              "neither_observable", "not_paired"):
        say(f"      {k:<22}{cells[k]:>8,}")
    n = len(jo)
    con = sum(1 for r in jo if r["outcome_gate_on"] == CHANGED)
    coff = sum(1 for r in jo if r["outcome_gate_off"] == CHANGED)
    delta = (100 * coff / n - 100 * con / n) if n else None
    say(f"    common support n = {n:,} requests in "
        f"{len({r['cluster_id'] for r in jo}):,} clusters")
    if n:
        say(f"    drift with gate     {100*con/n:.4f}%  CI{wilson(con, n)}")
        say(f"    drift without gate  {100*coff/n:.4f}%  CI{wilson(coff, n)}")
        say(f"    Delta = {delta:+.4f} pp")
    boot = None
    if n:
        byc2 = defaultdict(list)
        for r in jo:
            byc2[r["cluster_id"]].append(r)
        ks = list(byc2)
        rng = random.Random(SEED)
        ds = []
        for _ in range(B):
            s = [byc2[ks[rng.randrange(len(ks))]] for _ in range(len(ks))]
            fl = [x for g in s for x in g]
            if not fl:
                continue
            ds.append(100 * sum(1 for x in fl if x["outcome_gate_off"] == CHANGED) / len(fl)
                      - 100 * sum(1 for x in fl if x["outcome_gate_on"] == CHANGED) / len(fl))
        ds.sort()
        lo, hi = ds[int(.025 * len(ds))], ds[int(.975 * len(ds))]
        boot = {"n": n, "clusters": len(ks), "resamples": len(ds), "seed": SEED,
                "delta_pp": round(delta, 4),
                "ci95_pp": [round(lo, 4), round(hi, 4)]}
        say(f"    paired CLUSTER bootstrap ({len(ks):,} clusters, {len(ds):,} "
            f"resamples, seed {SEED}): 95% CI [{lo:+.4f}, {hi:+.4f}] pp")
        if lo > 0:
            say(f"    -> the gate PROSPECTIVELY REDUCES stale-content reuse by "
                f"{delta:.4f} pp on this common support; the CI excludes zero")
        elif hi < 0:
            say(f"    -> the gate prospectively INCREASES drift; the CI excludes zero")
        else:
            say(f"    -> the CI CROSSES ZERO: the direction is NOT statistically "
                f"resolved on this prospective snapshot")
    bycls = {}
    for c in ("TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"):
        sub = [r for r in jo if r["freshness_class"] == c]
        allc = [r for r in pair_rows if r["freshness_class"] == c]
        if not allc:
            continue
        bycls[c] = {"paired_observable": len(sub), "all_requests": len(allc),
                    "coverage_pct": round(100 * len(sub) / len(allc), 4),
                    "drift_on_pct": (round(100*sum(1 for r in sub if r["outcome_gate_on"]==CHANGED)/len(sub), 4) if sub else None),
                    "drift_off_pct": (round(100*sum(1 for r in sub if r["outcome_gate_off"]==CHANGED)/len(sub), 4) if sub else None)}
    say(f"\n    coverage and gate effect by freshness class")
    say(f"      {'class':<12}{'paired obs':>12}{'of all':>10}{'cov%':>9}"
        f"{'drift on%':>11}{'drift off%':>12}{'delta pp':>10}")
    for c, e in bycls.items():
        dd = ((e["drift_off_pct"] - e["drift_on_pct"])
              if e["drift_on_pct"] is not None else None)
        say(f"      {c:<12}{e['paired_observable']:>12,}{e['all_requests']:>10,}"
            f"{e['coverage_pct']:>8.3f}%"
            + (f"{e['drift_on_pct']:>10.4f}%{e['drift_off_pct']:>11.4f}%"
               f"{dd:>+10.4f}" if e["drift_on_pct"] is not None
               else f"{'-':>11}{'-':>12}{'-':>10}"))

    allowed = (a.labels == "prospective")
    say(f"\n  key claim -- \"all model, threshold and calibration parameters "
        f"were frozen before the +24h Web state existed\": "
        f"{'ALLOWED' if allowed else 'NOT ALLOWED (retrospective labels)'}")

    with open(HERE / f"prospective_per_request_{a.labels}.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0]))
        w.writeheader(); w.writerows(pair_rows)
    json.dump({"utc": now, "mode": a.labels, "age_seconds": AGE, "seed": SEED,
               "freeze_time_utc": man["freeze_time_utc"],
               "labels_sha256": lab_sha,
               "policies": out, "policy_order": ORDER,
               "gate_effect": {"cells": dict(cells), "common_support_n": n,
                               "clusters": len({r['cluster_id'] for r in jo}),
                               "drift_gate_on_pct": (round(100*con/n, 4) if n else None),
                               "drift_gate_off_pct": (round(100*coff/n, 4) if n else None),
                               "delta_pp": delta, "bootstrap": boot,
                               "by_class": bycls},
               "prospective_claim_allowed": allowed},
              open(HERE / f"prospective_results_{a.labels}.json", "w"), indent=2)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"evaluate_{a.labels}.log").write_text("\n".join(log) + "\n",
                                                            encoding="utf-8")
    say(f"\n  wrote prospective_results_{a.labels}.json, "
        f"prospective_per_request_{a.labels}.csv")


if __name__ == "__main__":
    main()
