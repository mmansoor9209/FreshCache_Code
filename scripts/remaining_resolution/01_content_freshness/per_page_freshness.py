#!/usr/bin/env python3
"""
EXPERIMENT 1 -- artifact-level (per-page) content freshness.

The request-level metric asks "did ANY reused page change?". That indicator
fires more often when more pages are reused, so it confounds page freshness
with reuse breadth. This computes the per-PAGE change rate instead, on both
workloads, for gate ON and gate OFF, and then matches the arms at four
increasingly strict levels up to identical URL identity AND identical artifact
version.

  fixed-24h   : engine_all FreshCache_Full at a uniform 24 h age
  mixed-age   : prep2/mixed_engine FreshCache on the published 7-day workload

Both transcriptions are fidelity-gated against the originals before any number
is believed. The published request-level metric is preserved and reported
first; nothing here replaces it.
"""
from __future__ import annotations
import argparse, csv, json, math, os, pathlib, random, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
RR = HERE.parent
V3 = RR.parent
ROOT = V3.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import corrected_engine as ce            # noqa: E402
import schedules as sc                   # noqa: E402
import mixed_age_v2 as ma                # noqa: E402
import prep2 as p2                       # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS
AGE, SEED, B = 86_400.0, 42, 10_000
SCHEDULE = "zipf_uniform"


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 4),
            round(100 * min(1.0, (c + m) / d), 4))


def cluster_boot(units, clusters, num, den, b=B, seed=SEED):
    """Cluster bootstrap over request clusters; `units` are (cluster, num, den)."""
    byc = defaultdict(list)
    for cl, nu, de in units:
        byc[cl].append((nu, de))
    ks = list(byc)
    if not ks:
        return (None, None)
    rng = random.Random(seed)
    out = []
    for _ in range(b):
        n_ = d_ = 0
        for _ in range(len(ks)):
            for nu, de in byc[ks[rng.randrange(len(ks))]]:
                n_ += nu; d_ += de
        if d_:
            out.append(100 * n_ / d_)
    out.sort()
    return (round(out[int(.025*len(out))], 4), round(out[int(.975*len(out))], 4))


# ------------------------------------------------------- fixed-24h workload
def replay_pages_fixed(records, labels, sim_age):
    """engine_all FreshCache_Full, transcribed, recording every REUSED page."""
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
    search = fetches = l1h = l2h = l3h = 0
    outcomes, pages, reqs = [], [], {}

    def ps(fc, tier):
        return exp.p_stale(fc, sim_age, tier)

    for r in records:
        q, fc, urls = r["query"], r["freshness_class"], r["urls"]
        qid = r["query_id"]
        qi = IDX[q]; row = SIM[qi]
        own = [u["url_hash"] for u in urls]
        uobj = {u["url_hash"]: u for u in urls}
        rt_true = (fc == "REAL_TIME"); rt = rt_true and cfg.get("rt_bypass", True)

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
            outcomes.append((qid, ea.score(l1_urls[hit1], labels)))
            reqs[qid] = {"tier": "L1", "reused": [], "fc": fc}
            continue

        hit2 = -1
        if not rt_true and n2:
            s2 = row[l2_qi[:n2]]
            el = (s2 >= L2_EQ) & l2_ok[:n2]
            if el.any():
                hit2 = int(np.where(el, s2, -np.inf).argmax())
        served = list(l2_urls[hit2]) if hit2 >= 0 else list(own)
        if hit2 >= 0:
            l2h += 1
        else:
            search += 1
        l2_qi[n2] = qi; l2_ok[n2] = ps(fc, "url_list") <= EU
        l2_urls.append(served); n2 += 1

        reused = []
        for uh in served:
            u = uobj.get(uh, {})
            if rt:
                fetches += 1; l3_cache.add(uh); continue
            if uh in l3_cache:
                if ps(fc, "content") <= EC:
                    l3h += 1; reused.append(uh)
                else:
                    if labels.get(uh) == CHANGED:
                        fetches += 1; l3_cache.add(uh)
                    else:
                        l3h += 1; reused.append(uh)
            else:
                fetches += 1; l3_cache.add(uh)
        for uh in reused:
            pages.append({"request_id": qid, "url_hash": uh,
                          "freshness_class": fc,
                          "artifact_version": "run_00",
                          "reference_version": ma.version_at(sim_age),
                          "artifact_age_seconds": sim_age,
                          "outcome": labels.get(uh, UNOBS)})
        outcomes.append((qid, ea.score(reused, labels) if reused else None))
        reqs[qid] = {"tier": "L2" if hit2 >= 0 else "miss",
                     "reused": reused, "fc": fc}
        l1_qi[n1] = qi; l1_ok[n1] = ps(fc, "answer") <= EA
        l1_q.append(q); l1_urls.append(served); n1 += 1

    m = {"n_requests": n, "search_calls": search, "l1_hits": l1h,
         "l2_hits": l2h, "l3_hits": l3h, "fetches": fetches}
    return m, outcomes, pages, reqs


# ------------------------------------------------------- mixed-age workload
def pages_from_prep2(stream, rounds, rows):
    """Per-page reuse records from prep2's provenance rows. A page is REUSED
    when its recorded cache time precedes the request time."""
    pages, reqs = [], {}
    for t, r in stream:
        qid = r["query_id"]
        p = rows.get(qid)
        if not p:
            continue
        if p["tier"] == "L1":
            reqs[qid] = {"tier": "L1", "reused": [], "fc": r["freshness_class"]}
            continue
        reused = []
        for uh, ct in (p.get("ev") or []):
            if ct >= t:
                continue
            reused.append(uh)
            pages.append({"request_id": qid, "url_hash": uh,
                          "freshness_class": r["freshness_class"],
                          "artifact_version": ma.version_at(ct),
                          "reference_version": ma.version_at(t),
                          "artifact_age_seconds": round(t - ct, 1),
                          "outcome": me.outcome(uh, ct, t, rounds)})
        reqs[qid] = {"tier": p["tier"], "reused": reused,
                     "fc": r["freshness_class"]}
    return pages, reqs


def page_rate(pages, cid, label=""):
    obs = [p for p in pages if p["outcome"] in (CHANGED, UNCHANGED)]
    ch = sum(1 for p in obs if p["outcome"] == CHANGED)
    uo = sum(1 for p in pages if p["outcome"] == UNOBS)
    units = [(cid.get(p["request_id"], p["request_id"]),
              int(p["outcome"] == CHANGED), 1) for p in obs]
    lo, hi = cluster_boot(units, None, None, None)
    return {"pages": len(pages), "observable": len(obs), "changed": ch,
            "unobservable": uo,
            "rate_pct": round(100*ch/len(obs), 4) if obs else None,
            "wilson_ci": wilson(ch, len(obs)),
            "cluster_bootstrap_ci": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["fixed24", "mixed", "both"],
                    default="both")
    a = ap.parse_args()
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("EXPERIMENT 1 -- artifact-level (per-page) content freshness")
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats(); ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    RES = {}

    def analyse(tag, pon, ron, poff, roff, mon, moff, orig_on, orig_off):
        say(f"\n{'='*70}\n  WORKLOAD: {tag}\n{'='*70}")
        say(f"\n  -- 1A. ORIGINAL request-level metric, definition unchanged --")
        say(f"    gate ON  drift {orig_on[0]:.4f}%  ({orig_on[1]:,}/{orig_on[2]:,})")
        say(f"    gate OFF drift {orig_off[0]:.4f}%  ({orig_off[1]:,}/{orig_off[2]:,})")
        say(f"    Delta (OFF - ON) = {orig_off[0]-orig_on[0]:+.4f} pp")

        say(f"\n  -- 1B. PER-PAGE freshness of reused artifacts --")
        A, Bp = page_rate(pon, cid), page_rate(poff, cid)
        say(f"    {'arm':<10}{'pages':>9}{'observable':>12}{'changed':>9}"
            f"{'unobs':>8}{'per-page rate':>15}{'cluster boot 95% CI':>24}")
        for nm, e in (("gate ON", A), ("gate OFF", Bp)):
            say(f"    {nm:<10}{e['pages']:>9,}{e['observable']:>12,}"
                f"{e['changed']:>9,}{e['unobservable']:>8,}"
                f"{e['rate_pct']:>14.4f}%{str(e['cluster_bootstrap_ci']):>24}")
        say(f"    Delta (OFF - ON) per page = "
            f"{Bp['rate_pct']-A['rate_pct']:+.4f} pp")
        say(f"    mean pages reused per content-exposing request: "
            f"ON {A['pages']/max(1,sum(1 for v in ron.values() if v['reused'])):.4f}  "
            f"OFF {Bp['pages']/max(1,sum(1 for v in roff.values() if v['reused'])):.4f}")
        say(f"    by freshness class (per-page rate, observable pages)")
        say(f"      {'class':<12}{'ON pages':>10}{'ON rate':>10}"
            f"{'OFF pages':>11}{'OFF rate':>10}{'delta pp':>10}")
        bycl = {}
        for c in sorted({p["freshness_class"] for p in pon} |
                        {p["freshness_class"] for p in poff}):
            x = page_rate([p for p in pon if p["freshness_class"] == c], cid)
            y = page_rate([p for p in poff if p["freshness_class"] == c], cid)
            bycl[c] = {"on": x, "off": y}
            dd = ((y["rate_pct"] - x["rate_pct"])
                  if x["rate_pct"] is not None and y["rate_pct"] is not None else None)
            say(f"      {c:<12}{x['observable']:>10,}"
                + (f"{x['rate_pct']:>9.3f}%" if x['rate_pct'] is not None else f"{'-':>10}")
                + f"{y['observable']:>11,}"
                + (f"{y['rate_pct']:>9.3f}%" if y['rate_pct'] is not None else f"{'-':>10}")
                + (f"{dd:>+10.3f}" if dd is not None else f"{'-':>10}"))

        say(f"\n  -- 1C. MATCHED page-identity analysis (nested) --")
        lv = {}
        both = [q for q in ron if q in roff and ron[q]["reused"] and roff[q]["reused"]]
        same_n = [q for q in both if len(ron[q]["reused"]) == len(roff[q]["reused"])]
        same_id = [q for q in both
                   if sorted(ron[q]["reused"]) == sorted(roff[q]["reused"])]
        pv_on = {(p["request_id"], p["url_hash"]): p["artifact_version"] for p in pon}
        pv_off = {(p["request_id"], p["url_hash"]): p["artifact_version"] for p in poff}
        same_ver = [q for q in same_id
                    if all(pv_on.get((q, u)) == pv_off.get((q, u))
                           for u in ron[q]["reused"])]
        say(f"    {'level':<44}{'requests':>10}{'pages ON':>10}{'pages OFF':>11}"
            f"{'rate ON':>10}{'rate OFF':>10}{'delta pp':>10}")
        for nm, qs in (("1 both expose content", both),
                       ("2 + same NUMBER of pages", same_n),
                       ("3 + same URL IDENTITIES", same_id),
                       ("4 + same ARTIFACT VERSIONS", same_ver)):
            S = set(qs)
            x = page_rate([p for p in pon if p["request_id"] in S], cid)
            y = page_rate([p for p in poff if p["request_id"] in S], cid)
            dd = ((y["rate_pct"] - x["rate_pct"])
                  if x["rate_pct"] is not None and y["rate_pct"] is not None else None)
            lv[nm] = {"requests": len(qs), "on": x, "off": y,
                      "delta_pp": round(dd, 4) if dd is not None else None}
            say(f"    {nm:<44}{len(qs):>10,}{x['pages']:>10,}{y['pages']:>11,}"
                + (f"{x['rate_pct']:>9.3f}%" if x['rate_pct'] is not None else f"{'-':>10}")
                + (f"{y['rate_pct']:>9.3f}%" if y['rate_pct'] is not None else f"{'-':>10}")
                + (f"{dd:>+10.3f}" if dd is not None else f"{'-':>10}"))
        if lv["4 + same ARTIFACT VERSIONS"]["delta_pp"] == 0.0:
            say(f"    NOTE: at level 4 the two arms reuse the SAME URLs at the "
                f"SAME versions, so their per-page outcomes are IDENTICAL BY "
                f"CONSTRUCTION. A zero difference there is a consistency check, "
                f"NOT evidence that the temporal gate is effective.")
        RES[tag] = {"original_request_level": {"gate_on": orig_on,
                                               "gate_off": orig_off},
                    "per_page": {"gate_on": A, "gate_off": Bp},
                    "per_page_by_class": bycl, "matched_levels": lv,
                    "engine": {"gate_on": mon, "gate_off": moff}}
        return pon, poff

    # ---------------- fixed-24h ----------------
    if a.workload in ("fixed24", "both"):
        labels = ce.load_labels("rerun_24h")
        m1, o1, pg1, rq1 = replay_pages_fixed(records, labels, AGE)
        e1, _ = ea.replay(records, labels, AGE, "FreshCache_Full", rich=rich)
        assert all(m1[k] == e1[k] for k in ("l1_hits", "l2_hits", "l3_hits",
                                            "search_calls", "fetches")), \
            f"fixed-24h ON transcription mismatch {m1} vs {e1}"
        _p = exp.p_stale
        exp.p_stale = lambda fc, age, tier="content": 0.0
        try:
            m2, o2, pg2, rq2 = replay_pages_fixed(records, labels, AGE)
            e2, _ = ea.replay(records, labels, AGE, "FreshCache_Full", rich=rich)
        finally:
            exp.p_stale = _p
        assert all(m2[k] == e2[k] for k in ("l1_hits", "l2_hits", "l3_hits",
                                            "search_calls", "fetches")), \
            f"fixed-24h OFF transcription mismatch {m2} vs {e2}"
        say(f"\n  FIDELITY GATE PASSED (fixed-24h): both arms reproduce "
            f"engine_all tier counts exactly")
        def od(o):
            ins = [x for _, x in o if x is not None]
            ch = sum(1 for x in ins if x == CHANGED)
            de = ch + sum(1 for x in ins if x == UNCHANGED)
            return (round(100*ch/de, 4), ch, de)
        analyse("fixed-24h (uniform 24 h age)", pg1, rq1, pg2, rq2, m1, m2,
                od(o1), od(o2))

    # ---------------- mixed-age ----------------
    if a.workload in ("mixed", "both"):
        full = sc.build_stream(records, SCHEDULE, SEED)
        assert len(full) == 31201
        mA, rA = p2.replay(full, {}, "FreshCache")
        assert abs(mA["search_saved_pct"] - 62.7320) < 1e-4 and \
            mA["l1_hits"] == 1175 and mA["l2_hits"] == 18398, \
            f"mixed-age anchor not reproduced: {mA}"
        say(f"\n  ANCHOR GATE PASSED (mixed-age): saved "
            f"{mA['search_saved_pct']:.4f}%, L1 {mA['l1_hits']:,}, "
            f"L2 {mA['l2_hits']:,}")
        mB, rB = p2.replay(full, {}, "FreshCache_AlwaysPassTemporal")
        pg1, rq1 = pages_from_prep2(full, rounds, rA)
        pg2, rq2 = pages_from_prep2(full, rounds, rB)
        eA, perA = me.replay(full, rounds, "FreshCache", rich=rich)
        eB, perB = me.replay(full, rounds, "FreshCache_AlwaysPassTemporal",
                             rich=rich)
        def od2(per):
            ins = [o for _, o in per if o is not None]
            ch = sum(1 for o in ins if o == CHANGED)
            de = ch + sum(1 for o in ins if o == UNCHANGED)
            return (round(100*ch/de, 4), ch, de)
        say(f"  published mixed-age reference: gate ON drift "
            f"{od2(perA)[0]:.4f}% (manuscript 3.42%), gate OFF "
            f"{od2(perB)[0]:.4f}% (manuscript 7.45%)")
        analyse("mixed-age (published 7-day workload)", pg1, rq1, pg2, rq2,
                mA, mB, od2(perA), od2(perB))
        for tag, pgs in (("mixed_on", pg1), ("mixed_off", pg2)):
            with open(HERE / f"pages_{tag}.csv", "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(pgs[0]))
                w.writeheader(); w.writerows(pgs)

    json.dump(RES, open(HERE / "per_page_results.json", "w"), indent=2, default=str)
    (RR / "logs").mkdir(exist_ok=True)
    (RR / "logs" / "exp1_per_page.log").write_text("\n".join(log) + "\n",
                                                   encoding="utf-8")
    say(f"\n  wrote per_page_results.json")


if __name__ == "__main__":
    main()
