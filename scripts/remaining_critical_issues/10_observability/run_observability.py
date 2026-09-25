#!/usr/bin/env python3
"""
Section 10 -- observability bias and domain-filter robustness.

Three calibration variants, each refit from scratch on VALIDATION clusters only
and then frozen before the held-out TEST replay:

  A  published   the shipped calibration: manifest rows filtered by
                 freshcache.risk_model.is_noise_url, change_log rows filtered by
                 the `noise` flag.
  B  unfiltered  no domain-volatility filter on either side.
  C  body_obs    no domain filter, but restricted to the URLs whose content is
                 actually OBSERVABLE -- a substantive extracted body in run_00
                 and in the comparison round -- i.e. exactly the population on
                 which drift can be scored.

Half-lives are refit per class with calibrate.fit_half_life_multi_window over
calibrate.FIT_WINDOWS_BY_CLASS, the published estimator, unchanged.

Also reported: missingness by freshness class, domain volatility, URL type; and
conservative SENSITIVITY BOUNDS (not probabilities) for the unobservable reused
pages -- all-unchanged and all-changed.

CPU only, read-only on every existing file.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, pathlib, sys, time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(BASE / "lib"))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402
import calibrate as cal                  # noqa: E402
from e1_robustness import support_of     # noqa: E402
from freshcache.risk_model import (   # noqa: E402
    is_noise_url, _DEFAULT_DOMAIN_VOLATILITY, _DEFAULT_DOMAIN_VOL)


def get_domain_volatility(domain: str) -> float:
    """is_noise_url's own lookup, transcribed (the module exposes the tables
    but not the accessor)."""
    domain = (domain or "").lower().removeprefix("www.")
    vol = _DEFAULT_DOMAIN_VOLATILITY.get(domain, _DEFAULT_DOMAIN_VOL)
    if vol == _DEFAULT_DOMAIN_VOL:
        for known, v in _DEFAULT_DOMAIN_VOLATILITY.items():
            if domain.endswith("." + known) or known in domain:
                return v
    return vol

CHANGED, UNCHANGED, UNOBS = me.CHANGED, me.UNCHANGED, me.UNOBS
SCHEDULE, SEED = "zipf_uniform", 42
RUN_AGES = {"rerun_1h": 3600.0, "rerun_12h": 43200.0,
            "rerun_24h": 86400.0, "rerun_7d": 604800.0}
CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST"]
WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
         "v14_baselines/engine_all.py", "calibrate.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fit(manifest, changes, rounds, variant, keep_urls):
    """Return {class: half_life} refit under one observability variant."""
    tracked, changed = Counter(), Counter()
    for m in manifest:
        if not m.get("snapshot_available") or m.get("run_id", "run_00") == "run_00":
            continue
        uh, fc, dom = m.get("url_hash"), m.get("freshness_class", ""), m.get("domain", "")
        if uh not in keep_urls or not fc:
            continue
        if variant == "published" and is_noise_url(dom, fc):
            continue
        if variant == "body_obs":
            per = rounds.get(uh) or {}
            a, b = per.get("run_00"), per.get(m["run_id"])
            if not (a and b and a.get("substantive") and b.get("substantive")):
                continue
        tracked[(m["run_id"], fc)] += 1
    for c in changes:
        uh, fc = c.get("url_hash"), c.get("freshness_class", "")
        if uh not in keep_urls:
            continue
        if variant == "published" and c.get("noise", False):
            continue
        if variant == "published" and c.get("detected_at", "") <= cal.NEW_RUN_CUTOFF:
            continue
        if variant == "body_obs":
            per = rounds.get(uh) or {}
            a, b = per.get("run_00"), per.get(c.get("run_id"))
            if not (a and b and a.get("substantive") and b.get("substantive")):
                continue
        changed[(c.get("run_id"), fc)] += 1
    rates = {k: changed.get(k, 0) / v for k, v in tracked.items() if v}
    hl, detail = dict(exp.HALF_LIFE), {}
    for fc in CLASSES:
        pairs = [(rates.get((r, fc)), RUN_AGES[r])
                 for r in cal.FIT_WINDOWS_BY_CLASS[fc]]
        h, note = cal.fit_half_life_multi_window(pairs)
        detail[fc] = {"rate_age_pairs": pairs, "note": note,
                      "tracked": {r: tracked.get((r, fc), 0)
                                  for r in cal.FIT_WINDOWS_BY_CLASS[fc]},
                      "half_life": h, "published_half_life": exp.HALF_LIFE[fc]}
        if h:
            hl[fc] = float(h)
    return hl, detail, dict(tracked), rates


def score(m, per, S, rounds=None):
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == CHANGED)
    un = sum(1 for o in ins if o == UNCHANGED)
    uo = sum(1 for o in ins if o == UNOBS)
    det = ch + un
    return {"search_saved_pct": m["search_saved_pct"],
            "drift_pct": (100 * ch / det) if det else None,
            "coverage_pct": (100 * det / len(S)) if S else None,
            "determinable": det, "changed": ch, "unobservable": uo,
            "drift_lower_bound_pct": (100 * ch / (det + uo)) if det + uo else None,
            "drift_upper_bound_pct": (100 * (ch + uo) / (det + uo)) if det + uo else None,
            "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
            "l3_hits": m["l3_hits"], "fetches": m["fetches"],
            "n_requests": m["n_requests"]}


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    say("V3 SECTION 10 -- observability and domain-filter robustness")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    changes = exp.load_jsonl(ROOT / "data" / "change_log.jsonl")
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]

    full = sc.build_stream(records, SCHEDULE, SEED)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])
    val = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in val_c]
    test = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in test_c]

    # calibration universe = URLs reachable from VALIDATION clusters only
    val_urls = {u["url_hash"] for _, r in val for u in r["urls"]}
    test_urls = {u["url_hash"] for _, r in test for u in r["urls"]}
    say(f"  validation {len(val):,} requests / {len(val_urls):,} distinct URLs")
    say(f"  test       {len(test):,} requests / {len(test_urls):,} distinct URLs")
    say(f"  calibration uses VALIDATION URLs only; refit half-lives are frozen "
        f"before the test replay (no tuning on test)")

    # ---------------- missingness ----------------
    say("\n  == missingness of the change signal ==")
    miss = {"by_class": defaultdict(lambda: [0, 0]),
            "by_domain_vol": defaultdict(lambda: [0, 0]),
            "by_url_type": defaultdict(lambda: [0, 0])}
    for m in manifest:
        if m.get("run_id", "run_00") == "run_00":
            continue
        uh, fc, dom = m.get("url_hash"), m.get("freshness_class", ""), m.get("domain", "")
        per = rounds.get(uh) or {}
        a, b = per.get("run_00"), per.get(m["run_id"])
        obs = bool(a and b and a.get("substantive") and b.get("substantive"))
        vol = get_domain_volatility((dom or "").lower().removeprefix("www."))
        vb = ("low <0.2" if vol < 0.2 else "mid 0.2-0.5" if vol < 0.5 else "high >=0.5")
        ut = ("filtered_out" if fc and is_noise_url(dom, fc) else "kept")
        for k, key in (("by_class", fc), ("by_domain_vol", vb), ("by_url_type", ut)):
            miss[k][key][0] += 1
            miss[k][key][1] += int(obs)
    missing_out = {}
    for k in ("by_class", "by_domain_vol", "by_url_type"):
        say(f"    {k}")
        missing_out[k] = {}
        for key, (tot, obs) in sorted(miss[k].items()):
            pct = 100 * obs / tot if tot else 0
            missing_out[k][key] = {"rows": tot, "observable": obs,
                                   "observable_pct": round(pct, 2)}
            say(f"      {str(key):<16} rows {tot:>7,}  observable {obs:>7,}  "
                f"{pct:>6.2f}%")

    # ---------------- three calibrations ----------------
    ORIG_HL = dict(exp.HALF_LIFE)
    S_test = support_of(test, rounds)
    say(f"\n  test |S| = {len(S_test):,}")
    out = {}
    for variant in ("published", "unfiltered", "body_obs"):
        hl, detail, tracked, rates = fit(manifest, changes, rounds, variant, val_urls)
        say(f"\n  ==== variant {variant} ====")
        for fc in CLASSES:
            d = detail[fc]
            say(f"    {fc:<10} refit {str(d['half_life']):>12}s  "
                f"(published {d['published_half_life']:>10,.0f}s)  "
                f"tracked {d['tracked']}  {d['note']}")
        exp.HALF_LIFE.clear(); exp.HALF_LIFE.update(hl)
        try:
            m_on, p_on = me.replay(test, rounds, "FreshCache", rich=rich)
            m_off, p_off = me.replay(test, rounds,
                                     "FreshCache_AlwaysPassTemporal", rich=rich)
            s_on, s_off = score(m_on, p_on, S_test), score(m_off, p_off, S_test)
            # paired common support: requests where BOTH policies reused and
            # both outcomes are determinate
            d_on = {q: o for q, o in p_on if o in (CHANGED, UNCHANGED) and q in S_test}
            d_off = {q: o for q, o in p_off if o in (CHANGED, UNCHANGED) and q in S_test}
            comm = set(d_on) & set(d_off)
            c_on = 100 * sum(1 for q in comm if d_on[q] == CHANGED) / len(comm) if comm else None
            c_off = 100 * sum(1 for q in comm if d_off[q] == CHANGED) / len(comm) if comm else None
        finally:
            exp.HALF_LIFE.clear(); exp.HALF_LIFE.update(ORIG_HL)
        say(f"    gate ON   saved {s_on['search_saved_pct']:>8.4f}%  "
            f"drift {s_on['drift_pct']:>7.4f}%  cov {s_on['coverage_pct']:>7.4f}%  "
            f"det {s_on['determinable']:>6,}  L1 {s_on['l1_hits']:>6,}  "
            f"L2 {s_on['l2_hits']:>6,}")
        say(f"    gate OFF  saved {s_off['search_saved_pct']:>8.4f}%  "
            f"drift {s_off['drift_pct']:>7.4f}%  cov {s_off['coverage_pct']:>7.4f}%")
        say(f"    gate effect  drift {s_off['drift_pct']-s_on['drift_pct']:+.4f} pp   "
            f"savings {s_off['search_saved_pct']-s_on['search_saved_pct']:+.4f} pp")
        if comm:
            say(f"    paired common support n = {len(comm):,}: "
                f"gate ON {c_on:.4f}%  gate OFF {c_off:.4f}%  "
                f"delta {c_off-c_on:+.4f} pp")
        say(f"    SENSITIVITY BOUNDS on gate-ON drift (not probabilities): "
            f"[{s_on['drift_lower_bound_pct']:.4f}%, "
            f"{s_on['drift_upper_bound_pct']:.4f}%] over "
            f"{s_on['determinable']+s_on['unobservable']:,} reused requests "
            f"({s_on['unobservable']:,} unobservable)")
        out[variant] = {"half_lives": hl, "detail": detail,
                        "gate_on": s_on, "gate_off": s_off,
                        "paired_common_support": {
                            "n": len(comm), "gate_on_drift_pct": c_on,
                            "gate_off_drift_pct": c_off}}

    assert exp.HALF_LIFE == ORIG_HL
    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post
    say("\n  implementation files unchanged: True")

    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "calibration_universe": "validation clusters only",
               "estimator": "calibrate.fit_half_life_multi_window over "
                            "calibrate.FIT_WINDOWS_BY_CLASS (published)",
               "published_half_lives": ORIG_HL,
               "missingness": missing_out, "variants": out, "hashes": pre},
              open(HERE / "observability_results.json", "w"), indent=2)
    with open(HERE / "observability_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "gate", "search_saved_pct", "drift_pct",
                    "coverage_pct", "determinable", "changed", "unobservable",
                    "drift_lb_pct", "drift_ub_pct", "l1_hits", "l2_hits"])
        for v, blk in out.items():
            for g in ("gate_on", "gate_off"):
                r = blk[g]
                w.writerow([v, g, r["search_saved_pct"], r["drift_pct"],
                            r["coverage_pct"], r["determinable"], r["changed"],
                            r["unobservable"], r["drift_lower_bound_pct"],
                            r["drift_upper_bound_pct"], r["l1_hits"], r["l2_hits"]])
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote observability_results.json/.csv, run.log")


if __name__ == "__main__":
    main()
