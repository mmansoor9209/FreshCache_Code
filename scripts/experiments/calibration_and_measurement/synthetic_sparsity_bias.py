"""Controlled synthetic-change measurement of snapshot-sparsity bias, where the TRUE change times are known (optional analysis).
Real ingredients (frozen, read-only): every page-reuse event (page, cache time tc, serving time tr) of FreshCache with the age gate ON and OFF
(reported traces.pkl, primary support S), per-page observability (share of the five real rounds with a substantive snapshot), and per-class
change hazards estimated from the real panel. Synthetic ingredient: each page changes as a Poisson process with its class hazard; change times
are therefore known exactly. Truth: a reuse event is CHANGED iff a change falls in (tc, tr]. Measurement: the frozen mixed_age_replay rules
(round_at = latest snapshot <= t with an available snapshot; same snapshot -> unchanged; request = CHANGED if any page CHANGED, UNCHANGED only
if all observable) applied to K snapshots. Schedules: the real 5-round schedule, and K evenly spaced snapshots over the 7-day window for
K in {2, 3, 5, 10, 20, 50, 100}. Two hazard calibrations (24 h and 7 d panel contrasts). 20 seeds each. No policy parameter is changed."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import collections, json, math, pathlib, pickle, sys
import numpy as np
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "observation_density")); CRR = (OUTPUT_ROOT / "replay")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
E = C.load("reported")
TR = pickle.load(open(CRR / "table2/reported/traces.pkl", "rb")); ROUNDS = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open((DATA_ROOT / "derived/round_table.jsonl"))}
qids, S, CLU = TR["qids"], set(TR["support"]), dict(zip(TR["qids"], TR["clusters"])); t2q = dict(zip(TR["arrival_times"], qids))
ARMS = {"on": "FreshCache", "off": "FreshCache_AlwaysPassTemporal"}
SIG = {a: dict(zip(qids, TR["methods"][m]["_outcome_sig"])) for a, m in ARMS.items()}
EV = {a: collections.defaultdict(list) for a in ARMS}
for a, m in ARMS.items():
    for ln, u, tc, tr, lab in TR["age_events"][m]: EV[a][t2q[tr]].append((u, tc, tr))
REQS = {a: [q for q in qids if q in S and SIG[a][q] is not None and EV[a].get(q)] for a in ARMS}
UCL = {}
for r in E["recs"]:
    for u in r.get("urls") or []: UCL.setdefault(u if isinstance(u, str) else u.get("url_hash"), r["freshness_class"])
URLS = sorted({u for a in ARMS for q in REQS[a] for u, _, _ in EV[a][q]})
RUNS = [("run_00", 0.0), ("rerun_1h", 3600.0), ("rerun_12h", 43200.0), ("rerun_24h", 86400.0), ("rerun_7d", 604800.0)]
def hazard(r_b, dt):
    ch, n = collections.Counter(), collections.Counter()
    for u in URLS:
        per = ROUNDS.get(u) or {}; a, b = per.get("run_00"), per.get(r_b)
        if a and b and a.get("substantive") and b.get("substantive"):
            c = UCL.get(u, "MEDIUM"); n[c] += 1; ch[c] += a["content_hash"] != b["content_hash"]
    return {c: {"n": n[c], "p_changed": ch[c] / n[c], "lambda_per_s": -math.log(max(1 - ch[c] / n[c], 1e-9)) / dt} for c in n}
CAL = {"24h": hazard("rerun_24h", 86400.0), "7d": hazard("rerun_7d", 604800.0)}
AV = {u: sum(1 for k, _ in RUNS if (ROUNDS.get(u) or {}).get(k, {}).get("substantive")) / len(RUNS) for u in URLS}
TMAX = max(tr for a in ARMS for q in REQS[a] for _, _, tr in EV[a][q]) + 1.0
SCHED = {"real_5": [t for _, t in RUNS]} | {f"uniform_{k}": list(np.linspace(0.0, 604800.0, k)) for k in (2, 3, 5, 10, 20, 50, 100)}
def run(cal, seed):
    rng = np.random.default_rng(seed); lam = CAL[cal]
    CHG = {}
    for u in URLS:
        l = lam.get(UCL.get(u, "MEDIUM"), lam.get("MEDIUM"))["lambda_per_s"]; n = rng.poisson(l * TMAX); CHG[u] = np.sort(rng.uniform(0, TMAX, n))
    nch = lambda u, t: int(np.searchsorted(CHG[u], t, side="right"))
    truth = {}
    for a in ARMS:
        ev = [any(nch(u, tr) > nch(u, tc) for u, tc, tr in EV[a][q]) for q in REQS[a]]; truth[a] = 100 * sum(ev) / len(ev)
    out = {"true_drift_pct": truth, "true_gate_effect_pp": truth["off"] - truth["on"]}
    for name, T in SCHED.items():
        T = np.array(T); avail = {u: rng.random(len(T)) < AV[u] for u in URLS}; res = {}
        for a in ARMS:
            det = ch = xdet = xch = 0; tdet = 0
            for q in REQS[a]:
                labs, cross = [], []
                for u, tc, tr in EV[a][q]:
                    ok = np.where(avail[u])[0]; ia = ok[T[ok] <= tc]; ib = ok[T[ok] <= tr]
                    if len(ia) == 0 or len(ib) == 0: labs.append("U"); cross.append(False); continue
                    ra, rb = ia[-1], ib[-1]; labs.append("C" if nch(u, T[rb]) != nch(u, T[ra]) else "N"); cross.append(ra != rb)
                o = "C" if "C" in labs else ("N" if all(x == "N" for x in labs) else "U")
                if o != "U":
                    det += 1; ch += o == "C"; tdet += any(nch(u, tr) > nch(u, tc) for u, tc, tr in EV[a][q])
                vx = [l for l, c in zip(labs, cross) if c and l in ("C", "N")]
                if any(l == "C" for l in vx): xdet += 1; xch += 1
                elif labs and all(c and l == "N" for l, c in zip(labs, cross)): xdet += 1
            res[a] = {"coverage_pct": 100 * det / len(REQS[a]), "measured_drift_pct": 100 * ch / det if det else None, "true_drift_on_determinate_pct": 100 * tdet / det if det else None,
                      "cross_round_coverage_pct": 100 * xdet / len(REQS[a]), "cross_round_drift_pct": 100 * xch / xdet if xdet else None}
            res[a]["bias_pp"] = res[a]["measured_drift_pct"] - truth[a] if det else None
        res["measured_gate_effect_pp"] = (res["off"]["measured_drift_pct"] or 0) - (res["on"]["measured_drift_pct"] or 0)
        res["cross_round_gate_effect_pp"] = (res["off"]["cross_round_drift_pct"] or 0) - (res["on"]["cross_round_drift_pct"] or 0)
        out[name] = res
    return out
RES = {}
for cal in CAL:
    runs = [run(cal, s) for s in range(20)]
    agg = {"true_drift_pct": {a: [np.mean([r["true_drift_pct"][a] for r in runs]), np.std([r["true_drift_pct"][a] for r in runs])] for a in ARMS},
           "true_gate_effect_pp": [np.mean([r["true_gate_effect_pp"] for r in runs]), np.std([r["true_gate_effect_pp"] for r in runs])]}
    for name in SCHED:
        g = lambda f: (lambda v: [float(np.nanmean(v)), float(np.nanstd(v))] if not np.all(np.isnan(v)) else [None, None])(np.array([np.nan if f(r[name]) is None else f(r[name]) for r in runs], float))
        agg[name] = {a: {k: g(lambda x, a=a, k=k: x[a][k]) for k in ("coverage_pct", "measured_drift_pct", "true_drift_on_determinate_pct", "bias_pp", "cross_round_coverage_pct", "cross_round_drift_pct")} for a in ARMS}
        agg[name]["measured_gate_effect_pp"] = g(lambda x: x["measured_gate_effect_pp"]); agg[name]["cross_round_gate_effect_pp"] = g(lambda x: x["cross_round_gate_effect_pp"])
    RES[cal] = agg
    print(cal, "true", {a: round(agg["true_drift_pct"][a][0], 2) for a in ARMS}, "gate", round(agg["true_gate_effect_pp"][0], 2), flush=True)
    for name in SCHED: print(" ", name, "meas on/off", agg[name]["on"]["measured_drift_pct"][0], agg[name]["off"]["measured_drift_pct"][0], "bias on", agg[name]["on"]["bias_pp"][0], "cov on", round(agg[name]["on"]["coverage_pct"][0], 1), "gate meas", agg[name]["measured_gate_effect_pp"][0], "xr", agg[name]["cross_round_gate_effect_pp"][0], flush=True)
json.dump({"label": "controlled synthetic-change measurement (known change times); real reuse events and observability; synthetic Poisson changes", "n_requests_with_reuse": {a: len(REQS[a]) for a in ARMS},
           "n_pages": len(URLS), "hazard_calibrations": CAL, "seeds": 20, "results": RES}, open(O / "synthetic_sparsity_results.json", "w"), indent=1, default=float)
