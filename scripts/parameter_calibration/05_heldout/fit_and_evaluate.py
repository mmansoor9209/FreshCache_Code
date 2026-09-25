#!/usr/bin/env python3
"""TASK 5 + 6 -- freeze, then evaluate on cluster-disjoint held-out data.

Configurations
  A original      published multipliers, published budgets
  B mult-only     m_content re-fitted by MLE on VALIDATION L3 labels;
                  m_answer / m_url_list UNCHANGED (no label exists to fit
                  them -- they stay design parameters); published budgets
  C budget-only   published multipliers, budgets selected on validation
  D full          B's multipliers + C's budgets
  E EquivalentTTL the explicit TTL form of D's rule, h_fc * k_t

Selection used validation clusters only and was frozen and hashed before this
script ran. Nothing here tunes on held-out outcomes.
"""
from __future__ import annotations
import hashlib, json, math, os, pathlib, random, sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
ROOT = PC.parents[1]
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402

SEED, SCHEDULE, B = 42, "zipf_uniform", 10_000
ANCHOR = {"search_saved_pct": 60.5776, "l1_hits": 806, "l2_hits": 12429}
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p, d = k / n, 1 + z * z / n
    c, m = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0.0, (c - m) / d), 4), round(100 * min(1.0, (c + m) / d), 4)]


def fit_m_content():
    """MLE for the content multiplier on validation-only L3 labels."""
    import csv
    rows = []
    with open(PC / "02_targets" / "l3_observations.csv", encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            if d["freshness_class"] in exp.HALF_LIFE:
                rows.append((exp.HALF_LIFE[d["freshness_class"]],
                             float(d["age_s"]), int(d["changed_denoised"])))

    def nll(m):
        s = 0.0
        for h, a, y in rows:
            p = min(max(1.0 - math.exp(-math.log(2) * m * a / h), 1e-12), 1 - 1e-12)
            s -= math.log(p) if y else math.log(1 - p)
        return s
    lo, hi = 1e-3, 10.0
    for _ in range(200):                       # golden-section
        m1 = lo + 0.382 * (hi - lo); m2 = lo + 0.618 * (hi - lo)
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    mhat = (lo + hi) / 2
    # profile-likelihood 95% interval (chi2_1 cutoff 3.841/2)
    base = nll(mhat)
    def bound(d):
        x, step = mhat, d * mhat * 0.01
        for _ in range(10_000):
            x += step
            if x <= 1e-4 or x > 50:
                break
            if nll(x) - base > 1.920729:
                return x
        return None
    return mhat, [bound(-1), bound(+1)], len(rows)


def main():
    say("TASK 5/6 -- frozen configurations on cluster-disjoint held-out data")
    fz = json.load(open(PC / "04_budgets" / "frozen_config.json", encoding="utf-8"))
    sha = hashlib.sha256(
        (PC / "04_budgets" / "frozen_config.json").read_bytes()).hexdigest()
    say(f"  frozen_config.json sha256 {sha}")
    say(f"  selected on: {fz['selected_on']}; held-out touched during "
        f"selection: {fz['heldout_touched_during_selection']}")
    sel = fz["selected_eps"]

    mhat, mci, nlab = fit_m_content()
    say(f"\n  == m_content MLE on validation L3 labels (n={nlab:,}) ==")
    say(f"     m_content_hat = {mhat:.4f}  95% profile-likelihood interval "
        f"[{mci[0]:.4f}, {mci[1]:.4f}]   (published 1.00)")
    say(f"     m_answer and m_url_list are NOT re-fitted: no temporal label "
        f"exists for either tier (Task 2). They remain design parameters.")

    pub_m, pub_e = dict(exp.TIER_MULT), (exp.EPS_ANSWER, exp.EPS_URL_LIST,
                                         exp.EPS_CONTENT)
    CONFIGS = {
        "A_original":      (pub_m, pub_e, "FreshCache"),
        "B_mult_only":     ({**pub_m, "content": mhat}, pub_e, "FreshCache"),
        "C_budget_only":   (pub_m, (sel["answer"], sel["url_list"],
                                    sel["content"]), "FreshCache"),
        "D_full":          ({**pub_m, "content": mhat},
                            (sel["answer"], sel["url_list"], sel["content"]),
                            "FreshCache"),
        "E_equivalent_ttl": ({**pub_m, "content": mhat},
                             (sel["answer"], sel["url_list"], sel["content"]),
                             "EquivalentTTL"),
    }

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    for line in open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
                     encoding="utf-8"):
        d = json.loads(line)
        rounds[d["url_hash"]] = d["rounds"]
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c = set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    test = [(t, r) for t, r in full
            if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert len(test) == split["test"]["requests"]
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    say(f"  held-out stream {len(test):,} requests, {len(test_c):,} clusters "
        f"(the same population for every configuration)")

    out, per_cfg = {}, {}
    try:
        for name, (mult, eps, variant) in CONFIGS.items():
            exp.TIER_MULT.update(mult)
            exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = eps
            m, per = me.replay(test, rounds, variant, rich=rich)
            if name == "A_original":
                bad = [k for k, v in ANCHOR.items()
                       if abs(m[k] - v) > (1e-4 if isinstance(v, float) else 0)]
                assert not bad, f"ANCHOR GATE FAILED on {bad}: {m}"
                say(f"  ANCHOR GATE PASSED: config A reproduces the published "
                    f"held-out result exactly")
            det = [(q, o) for q, o in per if o in (me.CHANGED, me.UNCHANGED)]
            chg = sum(1 for _, o in det if o == me.CHANGED)
            byc = defaultdict(list)
            for q, o in det:
                byc[cid[q]].append(o)
            ks = list(byc); rng = random.Random(SEED); ds = []
            for _ in range(B):
                fl = [x for _ in range(len(ks))
                      for x in byc[ks[rng.randrange(len(ks))]]]
                ds.append(100 * sum(1 for x in fl if x == me.CHANGED) / len(fl))
            ds.sort()
            out[name] = {
                "variant": variant, "tier_mult": dict(exp.TIER_MULT),
                "eps": list(eps),
                "k": {t: round(-math.log(1-e)/(exp.TIER_MULT[t]*math.log(2)), 6)
                      for t, e in zip(("answer", "url_list", "content"), eps)},
                "search_saved_pct": round(m["search_saved_pct"], 4),
                "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
                "l3_hits": m["l3_hits"],
                "observable_n": len(det),
                "coverage_pct": round(100*len(det)/len(per), 4),
                "drift_pct": round(100*chg/len(det), 4) if det else None,
                "drift_ci95": [round(ds[int(.025*B)], 4), round(ds[int(.975*B)], 4)],
                "drift_wilson95": wilson(chg, len(det)),
            }
            per_cfg[name] = dict(per)
            say(f"    {name:<18} saved {m['search_saved_pct']:>8.4f}%  "
                f"L1 {m['l1_hits']:>6,}  L2 {m['l2_hits']:>7,}  "
                f"L3 {m['l3_hits']:>7,}  cov {out[name]['coverage_pct']:>7.4f}%  "
                f"drift {out[name]['drift_pct']:>7.4f}% "
                f"{out[name]['drift_ci95']}")
    finally:
        exp.TIER_MULT.update(pub_m)
        exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = pub_e

    # decision-equivalence between D and E
    say(f"\n  == decision equivalence: D (risk form) vs E (TTL form) ==")
    dq = {q: o for q, o in per_cfg["D_full"].items()}
    eq = {q: o for q, o in per_cfg["E_equivalent_ttl"].items()}
    diff = sum(1 for q in dq if dq[q] != eq.get(q))
    same_metrics = (out["D_full"]["search_saved_pct"] ==
                    out["E_equivalent_ttl"]["search_saved_pct"]
                    and out["D_full"]["l1_hits"] == out["E_equivalent_ttl"]["l1_hits"])
    say(f"     per-request outcomes differing: {diff}  |  identical headline "
        f"metrics: {same_metrics}")
    say(f"     The exponential risk bound and its equivalent TTL are the same "
        f"rule written two ways; they are not independent evidence.")

    # paired McNemar between A and D on jointly observable requests
    say(f"\n  == A vs D, paired on jointly observable held-out requests ==")
    a, d = per_cfg["A_original"], per_cfg["D_full"]
    both = [q for q in a if a[q] in (me.CHANGED, me.UNCHANGED)
            and d.get(q) in (me.CHANGED, me.UNCHANGED)]
    b01 = sum(1 for q in both if a[q] == me.UNCHANGED and d[q] == me.CHANGED)
    b10 = sum(1 for q in both if a[q] == me.CHANGED and d[q] == me.UNCHANGED)
    from fractions import Fraction
    n = b01 + b10
    p = (float(sum(Fraction(math.comb(n, i), 2 ** n)
                   for i in range(0, min(b01, b10) + 1)) * 2) if n else 1.0)
    say(f"     n = {len(both):,}; discordant A-clean/D-drift {b01}, "
        f"A-drift/D-clean {b10}; exact McNemar p = {min(1.0, p):.4g}")

    json.dump({"frozen_sha256": sha, "m_content_mle": round(mhat, 6),
               "m_content_ci95": [round(x, 6) for x in mci],
               "m_answer_fitted": False, "m_url_list_fitted": False,
               "configs": out,
               "D_vs_E_outcome_differences": diff,
               "A_vs_D_mcnemar": {"n": len(both), "b01": b01, "b10": b10,
                                  "p_exact": min(1.0, p)}},
              open(HERE / "heldout_results.json", "w"), indent=2)
    with open(HERE / "heldout_per_request.csv", "w", encoding="utf-8") as fh:
        fh.write("query_id,cluster_id," + ",".join(CONFIGS) + "\n")
        for q in sorted(per_cfg["A_original"]):
            fh.write(f"{q},{cid[q]}," +
                     ",".join(str(per_cfg[c].get(q)) for c in CONFIGS) + "\n")
    (PC / "logs" / "t5.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote heldout_results.json, heldout_per_request.csv")


if __name__ == "__main__":
    main()
