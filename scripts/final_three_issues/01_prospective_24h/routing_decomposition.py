#!/usr/bin/env python3
"""
TASKS 2 and 3 -- pre-specified routing decomposition and like-for-like content
comparison, explaining why the temporal gate shows MORE measured content drift
at a fixed 24 h horizon.

Nothing about the frozen protocol changes. The original policy-specific drift
metric is preserved verbatim and reported first; everything here is ADDITIONAL
diagnostic decomposition of that same number.

THE MECHANISM THIS TESTS
  engine_all scores a request's outcome from one of two different things:
    * an L1 answer hit  -> score(l1_urls[hit], labels): the URL list STORED in
      the L1 entry, even though an L1 hit returns a stored answer and exposes
      NO page artifact at all;
    * anything else     -> score(reused, labels): the pages actually re-read
      from the L3 content cache; if none were reused the request is not scored.
  So the two arms are scored on different populations. Turning the gate off
  converts requests from content reuse into answer hits, which changes both the
  denominator and what is being measured.

  python routing_decomposition.py --labels retrospective   (diagnostic now)
  python routing_decomposition.py --labels prospective     (after collection)
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, math, os, pathlib, random, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parents[1]
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
AGE, SEED, B = 86_400.0, 42, 10_000


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 4),
            round(100 * min(1.0, (c + m) / d), 4))


def replay_routed(records, labels, sim_age, rich):
    """engine_all.replay for CONFIG['FreshCache_Full'] (l1 c18_risk, l2 eq_risk,
    l3 risk_cget), transcribed, additionally recording the routing of every
    request. Gate-off is produced by the caller patching exp.p_stale, exactly as
    the frozen evaluation does, so this function is identical for both arms.

    assert_matches_engine() checks metrics AND the per-request outcome vector
    against engine_all.replay before any decomposition is believed.
    """
    cfg = ea.CONFIG["FreshCache_Full"]
    SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
    L1T = exp.L1_SIM_THRESHOLD
    L2_EQ = max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
    EA, EU, EC = exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT
    n = len(records)
    l1_qi = np.empty(n, np.int64); l1_ok = np.zeros(n, bool)
    l1_q, l1_urls = [], []
    n1 = 0
    l2_qi = np.empty(n, np.int64); l2_ok = np.zeros(n, bool)
    l2_urls = []
    n2 = 0
    l3_cache = set()
    search = fetches = l1h = l2h = l3h = validated = 0
    outcomes, rows = [], []

    def ps(fc, tier, url="", query=""):
        return exp.p_stale(fc, sim_age, tier)

    for r in records:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        qi = IDX[q]
        row = SIM[qi]
        own = [u["url_hash"] for u in urls]
        uobj = {u["url_hash"]: u for u in urls}
        rt_true = (fc == "REAL_TIME")
        rt = rt_true and cfg.get("rt_bypass", True)

        hit1 = -1
        if not rt_true and n1:
            sims = row[l1_qi[:n1]]
            cand = np.nonzero((sims >= L1T) & l1_ok[:n1])[0]
            if cand.size:
                for j in cand[np.argsort(-sims[cand], kind="stable")]:
                    if exp._entity_match(q, l1_q[j]) and \
                            exp.semantic_equivalent(q, l1_q[j], float(sims[j])):
                        hit1 = int(j); break
        if hit1 >= 0:
            l1h += 1
            out = ea.score(l1_urls[hit1], labels)
            outcomes.append((qid, out))
            rows.append({"query_id": qid, "freshness_class": fc,
                         "routing": "L1", "l1_hit": 1, "l2_hit": 0,
                         "pages_reused": 0, "pages_fetched": 0,
                         "content_exposed": 0, "outcome": out,
                         "outcome_source": "l1_stored_urls",
                         "n_scored_urls": len(l1_urls[hit1])})
            continue

        hit2 = -1
        if not rt_true and n2:
            sims2 = row[l2_qi[:n2]]
            elig = (sims2 >= L2_EQ) & l2_ok[:n2]
            if elig.any():
                hit2 = int(np.where(elig, sims2, -np.inf).argmax())
        if hit2 >= 0:
            l2h += 1
            served = list(l2_urls[hit2])
        else:
            search += 1
            served = list(own)
        l2_qi[n2] = qi
        l2_ok[n2] = ps(fc, "url_list", query=q) <= EU
        l2_urls.append(served); n2 += 1

        reused, nfetch = [], 0
        for uh in served:
            u = uobj.get(uh, {})
            if rt:
                fetches += 1; nfetch += 1; l3_cache.add(uh); continue
            if uh in l3_cache:
                if ps(fc, "content", url=u.get("url", ""), query=q) <= EC:
                    l3h += 1; reused.append(uh)
                else:
                    validated += 1
                    if labels.get(uh) == CHANGED:
                        fetches += 1; nfetch += 1; l3_cache.add(uh)
                    else:
                        l3h += 1; reused.append(uh)
            else:
                fetches += 1; nfetch += 1; l3_cache.add(uh)

        out = ea.score(reused, labels) if reused else None
        outcomes.append((qid, out))
        rows.append({"query_id": qid, "freshness_class": fc,
                     "routing": ("RT_bypass" if rt else
                                 ("L2+reuse" if hit2 >= 0 and reused else
                                  "L2+norreuse" if hit2 >= 0 else
                                  "search+reuse" if reused else "search+nofetchreuse")),
                     "l1_hit": 0, "l2_hit": int(hit2 >= 0),
                     "pages_reused": len(reused), "pages_fetched": nfetch,
                     "content_exposed": int(bool(reused)),
                     "outcome": out,
                     "outcome_source": ("l3_reused_pages" if reused else None),
                     "n_scored_urls": len(reused)})

        l1_qi[n1] = qi
        l1_ok[n1] = ps(fc, "answer", query=q) <= EA
        l1_q.append(q); l1_urls.append(served); n1 += 1

    m = {"n_requests": n, "search_calls": search, "l1_hits": l1h,
         "l2_hits": l2h, "l3_hits": l3h, "fetches": fetches,
         "validated": validated,
         "search_saved_pct": round(100 * (1 - search / max(n, 1)), 3)}
    return m, outcomes, rows


def assert_matches_engine(records, labels, rich, tag):
    m2, out2, rows = replay_routed(records, labels, AGE, rich)
    m1, out1 = ea.replay(records, labels, AGE, "FreshCache_Full", rich=rich)
    bad = [k for k in ("l1_hits", "l2_hits", "l3_hits", "search_calls", "fetches")
           if m1[k] != m2[k]]
    d = sum(1 for (qa, oa), (qb, ob) in zip(out1, out2) if oa != ob or qa != qb)
    assert not bad and d == 0, (
        f"ROUTING TRANSCRIPTION FAILURE [{tag}]: metrics differ on {bad}; "
        f"{d} per-request outcome mismatches "
        f"({ {k: (m1[k], m2[k]) for k in ('l1_hits','l2_hits','l3_hits','fetches')} })")
    return m2, out2, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["prospective", "retrospective"],
                    default="retrospective")
    a = ap.parse_args()
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say(f"TASKS 2 & 3 -- routing decomposition and like-for-like content "
        f"comparison   mode={a.labels}")
    say(f"  utc {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")

    man = json.load(open(PROS / "freeze_manifest.json", encoding="utf-8"))
    drift = {k: v for k, v in man["frozen_parameters"].items()
             if (dict(getattr(exp, k)) if isinstance(getattr(exp, k, None), dict)
                 else getattr(exp, k, None)) != v}
    if drift:
        sys.exit(f"ERROR: frozen parameters changed: {list(drift)}")
    say(f"  frozen parameters unchanged since {man['freeze_time_utc']}: True")
    say(f"  the gate is NOT retuned; the uniform-24h frame is NOT replaced; "
        f"the primary drift metric is preserved and reported first")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)

    if a.labels == "prospective":
        p = HERE / "snapshots" / "T0_plus_24h.jsonl"
        if not p.exists():
            sys.exit(f"ERROR: {p} does not exist. Collect it first.")
        labels = {}
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            labels[r["url_hash"]] = (
                (CHANGED if r["content_changed"] else UNCHANGED)
                if r.get("observable") else UNOBS)
    else:
        labels = ce.load_labels("rerun_24h")
    say(f"  labels: {a.labels}, {len(labels):,} URLs")

    # ---- both arms, routing-aware, each gated against engine_all ----
    m_on, out_on, rows_on = assert_matches_engine(records, labels, rich, "gate ON")
    _p = exp.p_stale
    exp.p_stale = lambda fc, age, tier="content": 0.0
    try:
        m_off, out_off, rows_off = assert_matches_engine(records, labels, rich,
                                                         "gate OFF")
    finally:
        exp.p_stale = _p
    assert exp.p_stale is _p
    say(f"  TRANSCRIPTION GATE PASSED for both arms: metrics and the full "
        f"per-request outcome vector reproduce engine_all exactly")

    R_on = {r["query_id"]: r for r in rows_on}
    R_off = {r["query_id"]: r for r in rows_off}
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}

    # ---------------- 1. the ORIGINAL metric, preserved ----------------
    def orig(rows):
        ins = [r for r in rows if r["outcome"] is not None]
        ch = sum(1 for r in ins if r["outcome"] == CHANGED)
        un = sum(1 for r in ins if r["outcome"] == UNCHANGED)
        uo = sum(1 for r in ins if r["outcome"] == UNOBS)
        return ch, un, uo, ch + un
    say(f"\n  == 1. ORIGINAL policy-specific content drift (unchanged "
        f"definition, reported first) ==")
    say(f"    {'arm':<10}{'scored':>9}{'changed':>9}{'unobs':>8}{'drift%':>10}"
        f"{'L1 hits':>9}{'L3 hits':>9}")
    O = {}
    for tag, rows in (("gate ON", rows_on), ("gate OFF", rows_off)):
        ch, un, uo, det = orig(rows)
        m = m_on if tag == "gate ON" else m_off
        O[tag] = {"changed": ch, "unchanged": un, "unobservable": uo,
                  "determinable": det,
                  "drift_pct": round(100 * ch / det, 4) if det else None,
                  "drift_ci": wilson(ch, det), "l1_hits": m["l1_hits"],
                  "l3_hits": m["l3_hits"]}
        say(f"    {tag:<10}{det:>9,}{ch:>9,}{uo:>8,}"
            f"{100*ch/det:>9.4f}%{m['l1_hits']:>9,}{m['l3_hits']:>9,}")
    say(f"    Delta (OFF - ON) = "
        f"{O['gate OFF']['drift_pct'] - O['gate ON']['drift_pct']:+.4f} pp")

    # ---------------- 2. what is actually being scored ----------------
    say(f"\n  == 2. WHAT IS BEING SCORED in that metric ==")
    say(f"    {'arm':<10}{'from L1 stored URLs':>22}{'from L3 reused pages':>23}"
        f"{'L1 share of scored':>20}")
    for tag, rows in (("gate ON", rows_on), ("gate OFF", rows_off)):
        ins = [r for r in rows if r["outcome"] is not None]
        a1 = [r for r in ins if r["outcome_source"] == "l1_stored_urls"]
        a3 = [r for r in ins if r["outcome_source"] == "l3_reused_pages"]
        d1 = sum(1 for r in a1 if r["outcome"] in (CHANGED, UNCHANGED))
        d3 = sum(1 for r in a3 if r["outcome"] in (CHANGED, UNCHANGED))
        c1 = sum(1 for r in a1 if r["outcome"] == CHANGED)
        c3 = sum(1 for r in a3 if r["outcome"] == CHANGED)
        O[tag].update({"l1_scored": d1, "l1_changed": c1,
                       "l3_scored": d3, "l3_changed": c3,
                       "l1_drift_pct": round(100*c1/d1, 4) if d1 else None,
                       "l3_drift_pct": round(100*c3/d3, 4) if d3 else None})
        say(f"    {tag:<10}{f'{c1:,}/{d1:,} = {100*c1/max(1,d1):.4f}%':>22}"
            f"{f'{c3:,}/{d3:,} = {100*c3/max(1,d3):.4f}%':>23}"
            f"{100*d1/max(1,d1+d3):>19.2f}%")
    say(f"    An L1 hit returns a STORED ANSWER and exposes no page artifact, "
        f"yet the published metric scores it against the URL list saved in the "
        f"L1 entry. Turning the gate off moves requests from L3 content reuse "
        f"into L1 answer hits, so the two arms are scored on different "
        f"populations with different drift rates.")

    # ---------------- TASK 2: pre-specified routing groups ----------------
    def grp(qid):
        on, off = R_on[qid], R_off[qid]
        l1a, l1b = on["l1_hit"], off["l1_hit"]
        if l1a and l1b:
            return "A_L1_both"
        if l1a and not l1b:
            return "B_L1_gate_on_only"
        if l1b and not l1a:
            return "C_L1_gate_off_only"
        if on["l2_hit"] and off["l2_hit"]:
            return ("E_L3_content_both" if on["content_exposed"]
                    and off["content_exposed"] else "D_L2_both")
        if on["routing"] == "RT_bypass" or off["routing"] == "RT_bypass":
            return "H_other"
        if not on["l2_hit"] or not off["l2_hit"]:
            return "G_fresh_search_one_or_both"
        return "H_other"

    groups = defaultdict(list)
    for r in records:
        groups[grp(r["query_id"])].append(r["query_id"])
    say(f"\n  == TASK 2: pre-specified routing decomposition "
        f"(mutually exclusive, applied in the listed order) ==")
    say(f"    A L1 under both | B L1 gate-ON only | C L1 gate-OFF only | "
        f"D L2 both, no content reuse | E content reused under both | "
        f"F is covered by B and C (L1 vs L2/L3 crossing) | "
        f"G fresh search under one or both | H other")
    say(f"\n    {'group':<28}{'n':>7}{'exp_ON':>8}{'exp_OFF':>9}"
        f"{'det_ON':>8}{'chg_ON':>8}{'dr_ON%':>9}"
        f"{'det_OFF':>9}{'chg_OFF':>9}{'dr_OFF%':>10}{'cov_ON%':>9}")
    G = {}
    for g in sorted(groups):
        qs = groups[g]
        on = [R_on[q] for q in qs]; off = [R_off[q] for q in qs]
        eon = sum(r["content_exposed"] for r in on)
        eoff = sum(r["content_exposed"] for r in off)
        don = sum(1 for r in on if r["outcome"] in (CHANGED, UNCHANGED))
        doff = sum(1 for r in off if r["outcome"] in (CHANGED, UNCHANGED))
        con = sum(1 for r in on if r["outcome"] == CHANGED)
        coff = sum(1 for r in off if r["outcome"] == CHANGED)
        uon = sum(1 for r in on if r["outcome"] == UNOBS)
        cls = Counter(r["freshness_class"] for r in on)
        G[g] = {"n": len(qs), "content_exposed_on": eon, "content_exposed_off": eoff,
                "determinable_on": don, "changed_on": con,
                "determinable_off": doff, "changed_off": coff,
                "unobservable_on": uon,
                "drift_on_pct": round(100*con/don, 4) if don else None,
                "drift_off_pct": round(100*coff/doff, 4) if doff else None,
                "coverage_on_pct": round(100*don/len(qs), 4),
                "class_distribution": dict(cls)}
        say(f"    {g:<28}{len(qs):>7,}{eon:>8,}{eoff:>9,}{don:>8,}{con:>8,}"
            + (f"{100*con/don:>8.3f}%" if don else f"{'-':>9}")
            + f"{doff:>9,}{coff:>9,}"
            + (f"{100*coff/doff:>9.3f}%" if doff else f"{'-':>10}")
            + f"{100*don/len(qs):>8.2f}%")
    say(f"\n    contribution to the aggregate difference "
        f"(numerator and denominator moved by each group)")
    tot_on, tot_off = O["gate ON"]["determinable"], O["gate OFF"]["determinable"]
    say(f"      {'group':<28}{'d_det':>9}{'d_chg':>9}"
        f"{'drift if only this group changed':>36}")
    for g in sorted(G):
        e = G[g]
        dd = e["determinable_off"] - e["determinable_on"]
        dc = e["changed_off"] - e["changed_on"]
        alt = ((O["gate ON"]["changed"] + dc) / (tot_on + dd) * 100
               if tot_on + dd else None)
        say(f"      {g:<28}{dd:>+9,}{dc:>+9,}"
            + (f"{alt:>35.4f}%" if alt is not None else f"{'-':>36}"))
    say(f"      {'(all groups together)':<28}"
        f"{tot_off-tot_on:>+9,}{O['gate OFF']['changed']-O['gate ON']['changed']:>+9,}"
        f"{O['gate OFF']['drift_pct']:>35.4f}%")

    # ---------------- TASK 3: like-for-like content comparison ----------
    say(f"\n  == TASK 3: like-for-like content comparison ==")
    say(f"    (a) CONTENT-ONLY drift: scored ONLY on requests that actually "
        f"re-read cached page content. L1 answer hits are excluded because "
        f"they expose no page artifact; requests that expose no artifact are "
        f"NOT given a drift of zero, they are excluded.")
    C = {}
    for tag, rows in (("gate ON", rows_on), ("gate OFF", rows_off)):
        ex = [r for r in rows if r["content_exposed"]]
        det = [r for r in ex if r["outcome"] in (CHANGED, UNCHANGED)]
        ch = sum(1 for r in det if r["outcome"] == CHANGED)
        uo = sum(1 for r in ex if r["outcome"] == UNOBS)
        C[tag] = {"exposed": len(ex), "determinable": len(det), "changed": ch,
                  "unobservable": uo,
                  "drift_pct": round(100*ch/len(det), 4) if det else None,
                  "drift_ci": wilson(ch, len(det))}
        say(f"      {tag:<10} exposed {len(ex):>7,}  scored {len(det):>7,}  "
            f"changed {ch:>6,}  unobservable {uo:>6,}  "
            f"drift {100*ch/max(1,len(det)):>8.4f}%  CI{wilson(ch, len(det))}")
    say(f"      Delta (OFF - ON) on content-only = "
        f"{C['gate OFF']['drift_pct'] - C['gate ON']['drift_pct']:+.4f} pp "
        f"(DESCRIPTIVE: the two populations still differ)")

    say(f"\n    (b) PAIRED MATCHED-CONTENT drift: the same requests, where BOTH "
        f"arms re-read cached content AND both outcomes are observable. This "
        f"is the only comparison here that identifies a paired effect.")
    matched = [r["query_id"] for r in records
               if R_on[r["query_id"]]["content_exposed"]
               and R_off[r["query_id"]]["content_exposed"]
               and R_on[r["query_id"]]["outcome"] in (CHANGED, UNCHANGED)
               and R_off[r["query_id"]]["outcome"] in (CHANGED, UNCHANGED)]
    n = len(matched)
    con = sum(1 for q in matched if R_on[q]["outcome"] == CHANGED)
    coff = sum(1 for q in matched if R_off[q]["outcome"] == CHANGED)
    dm = (100*coff/n - 100*con/n) if n else None
    say(f"      n = {n:,} requests in "
        f"{len({cid[q] for q in matched}):,} clusters")
    boot = None
    if n:
        say(f"      drift gate ON  {100*con/n:>8.4f}%  CI{wilson(con, n)}")
        say(f"      drift gate OFF {100*coff/n:>8.4f}%  CI{wilson(coff, n)}")
        say(f"      Delta (OFF - ON) = {dm:+.4f} pp")
        byc = defaultdict(list)
        for q in matched:
            byc[cid[q]].append(q)
        ks = list(byc); rng = random.Random(SEED); ds = []
        for _ in range(B):
            s = [byc[ks[rng.randrange(len(ks))]] for _ in range(len(ks))]
            fl = [q for g in s for q in g]
            if fl:
                ds.append(100*sum(1 for q in fl if R_off[q]["outcome"] == CHANGED)/len(fl)
                          - 100*sum(1 for q in fl if R_on[q]["outcome"] == CHANGED)/len(fl))
        ds.sort()
        lo, hi = ds[int(.025*len(ds))], ds[int(.975*len(ds))]
        boot = {"n": n, "clusters": len(ks), "delta_pp": round(dm, 4),
                "ci95_pp": [round(lo, 4), round(hi, 4)], "seed": SEED}
        say(f"      paired cluster bootstrap ({len(ks):,} clusters, "
            f"{len(ds):,} resamples): 95% CI [{lo:+.4f}, {hi:+.4f}] pp")
        say(f"      -> " + ("the gate REDUCES drift on matched content reuse"
                            if lo > 0 else
                            "the gate INCREASES drift on matched content reuse"
                            if hi < 0 else
                            "the direction is NOT statistically resolved"))

    say(f"\n    (c) unobservable outcomes, reported separately and NEVER "
        f"counted as unchanged")
    for tag, rows in (("gate ON", rows_on), ("gate OFF", rows_off)):
        ex = [r for r in rows if r["content_exposed"]]
        say(f"      {tag:<10} content-exposed {len(ex):>7,}; unobservable "
            f"{sum(1 for r in ex if r['outcome'] == UNOBS):>7,} "
            f"({100*sum(1 for r in ex if r['outcome']==UNOBS)/max(1,len(ex)):.2f}%)")

    say(f"\n    which comparisons identify an effect:")
    say(f"      1 original policy-specific drift ....... DESCRIPTIVE "
        f"(different populations, different denominators)")
    say(f"      2 content-only drift ................... DESCRIPTIVE "
        f"(removes the L1 artefact, populations still differ)")
    say(f"      3 paired matched-content drift ......... IDENTIFIES a paired "
        f"effect (same requests, both arms exposed content, both observable)")

    with open(HERE / f"routing_per_request_{a.labels}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "cluster_id", "freshness_class", "group",
                    "routing_on", "routing_off", "l1_on", "l1_off",
                    "content_exposed_on", "content_exposed_off",
                    "pages_reused_on", "pages_reused_off",
                    "outcome_on", "outcome_off",
                    "outcome_source_on", "outcome_source_off"])
        for r in records:
            q = r["query_id"]; on = R_on[q]; off = R_off[q]
            w.writerow([q, cid[q], r["freshness_class"], grp(q),
                        on["routing"], off["routing"], on["l1_hit"], off["l1_hit"],
                        on["content_exposed"], off["content_exposed"],
                        on["pages_reused"], off["pages_reused"],
                        on["outcome"], off["outcome"],
                        on["outcome_source"], off["outcome_source"]])
    json.dump({"mode": a.labels, "seed": SEED,
               "original_metric": O, "routing_groups": G,
               "content_only": C,
               "paired_matched_content": boot,
               "engine_metrics": {"gate_on": m_on, "gate_off": m_off}},
              open(HERE / f"routing_decomposition_{a.labels}.json", "w"), indent=2)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"routing_{a.labels}.log").write_text("\n".join(log) + "\n",
                                                           encoding="utf-8")
    say(f"\n  wrote routing_decomposition_{a.labels}.json, "
        f"routing_per_request_{a.labels}.csv")


if __name__ == "__main__":
    main()
