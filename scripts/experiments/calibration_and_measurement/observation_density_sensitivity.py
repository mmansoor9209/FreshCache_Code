"""Within-panel observation-density sensitivity (not a true-bias estimate: no higher-resolution ground truth exists).
Source: frozen producer traces (outputs/replay/table2/reported/traces.pkl) - every page comparison of FreshCache (gate on)
and FreshCache_AlwaysPassTemporal (gate off) in support S of the primary replay. Routing does not depend on drift labels, so no replay is needed.
For every sub-panel R of the 5 stored rounds {run_00, rerun_1h, rerun_12h, rerun_24h, rerun_7d} that keeps run_00 (1 + 4 + 6 + 4 = 15 panels),
each page comparison is re-labelled with mixed_age_replay.outcome restricted to the rounds in R; request outcomes use mixed_age_replay.aggregate; the
cross-round rule and the gate-on/off contrasts are the frozen transcriptions. Gate: the full panel reproduces the frozen cross-round audit exactly."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, datetime, hashlib, itertools, json, os, pathlib, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "observation_density")); CRR = (OUTPUT_ROOT / "replay"); CROSS_ROUND_DIR = (OUTPUT_ROOT / "cross_round_audit")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
C.load("reported")   # puts engine_reported first on sys.path (as in the cross-round audit)
import numpy as np, mixed_age_replay as me   # noqa: E402
assert pathlib.Path(me.__file__).resolve() == (ROOT / "freshcache/replay_engine/mixed_age_replay.py")
CH, UN, UO = me.CHANGED, me.UNCHANGED, me.UNOBS
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
INP = [CRR / "table2/reported/traces.pkl", (DATA_ROOT / "derived/round_table.jsonl"), CROSS_ROUND_DIR / "cross_round_summary.json", (ROOT / "freshcache/replay_engine/mixed_age_replay.py")]; H0 = {str(p): sha(p) for p in INP}
TR = pickle.load(open(INP[0], "rb")); ROUNDS = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open(INP[1])}
qids, S, CLU = TR["qids"], set(TR["support"]), dict(zip(TR["qids"], TR["clusters"])); t2q = dict(zip(TR["arrival_times"], qids))
ARMS = {"on": "FreshCache", "off": "FreshCache_AlwaysPassTemporal"}
SIG = {a: dict(zip(qids, TR["methods"][m]["_outcome_sig"])) for a, m in ARMS.items()}
EV = {a: collections.defaultdict(list) for a in ARMS}
for a, m in ARMS.items():
    for ln, u, tc, tr, lab in TR["age_events"][m]: EV[a][t2q[tr]].append((u, tc, tr, lab))
XR_REF = json.load(open(INP[2]))
def part_b(on, off, B=1000, seed=42):   # cross-round audit transcription (cluster bootstrap, own-determinate and strict-paired)
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    for q in qids:
        if q not in S: continue
        v = per[CLU[q]]; v[0] += on.get(q) == CH; v[1] += on.get(q) in (CH, UN); v[2] += off.get(q) == CH; v[3] += off.get(q) in (CH, UN)
    M = np.array([per[c] for c in sorted(per)], np.int64); rng = np.random.RandomState(seed); idx = rng.randint(0, len(M), size=(B, len(M)))
    with np.errstate(invalid="ignore", divide="ignore"): d = M[:, 2][idx].sum(1) / M[:, 3][idx].sum(1) - M[:, 0][idx].sum(1) / M[:, 1][idx].sum(1)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    per2 = collections.defaultdict(lambda: [0, 0, 0])
    for q in qids:
        if q not in S: continue
        a_, b_ = on.get(q), off.get(q)
        if a_ in (CH, UN) and b_ in (CH, UN): v = per2[CLU[q]]; v[0] += a_ == CH; v[1] += b_ == CH; v[2] += 1
    if not per2: return {"own_determinate_diff_pp": round(100 * (M[:, 2].sum() / M[:, 3].sum() - M[:, 0].sum() / M[:, 1].sum()), 4) if M[:, 1].sum() and M[:, 3].sum() else None, "own_ci95_pp": [round(100 * lo, 4), round(100 * hi, 4)], "strict_paired_N": 0, "strict_paired_diff_pp": None, "strict_paired_ci95_pp": [None, None]}
    M2 = np.array([per2[c] for c in sorted(per2)], np.int64); N2 = int(M2[:, 2].sum()); rng2 = np.random.RandomState(seed); idx2 = rng2.randint(0, len(M2), size=(B, len(M2)))
    n2 = M2[:, 2][idx2].sum(1).astype(float); d2 = (M2[:, 1][idx2].sum(1) - M2[:, 0][idx2].sum(1)) / n2; lo2, hi2 = np.percentile(d2, [2.5, 97.5])
    return {"own_determinate_diff_pp": round(100 * (M[:, 2].sum() / M[:, 3].sum() - M[:, 0].sum() / M[:, 1].sum()), 4), "own_ci95_pp": [round(100 * lo, 4), round(100 * hi, 4)],
            "strict_paired_N": N2, "strict_paired_diff_pp": round(100 * (M2[:, 1].sum() - M2[:, 0].sum()) / N2, 4) if N2 else None, "strict_paired_ci95_pp": [round(100 * lo2, 4), round(100 * hi2, 4)]}
def evaluate(R):
    R = set(R); RT = {u: {k: v for k, v in per.items() if k in R} for u, per in ROUNDS.items()}
    out, W, O_ = {}, {}, {}
    for a in ARMS:
        c = collections.Counter(); orig, w4 = {}, {}
        for q in qids:
            if q not in S: continue
            es = EV[a].get(q, [])
            if SIG[a][q] is None: orig[q] = None; w4[q] = None; c["no_reuse"] += 1; continue
            pages = []
            for u, tc, tr, _ in es:
                per = RT.get(u) or {}; lab = me.outcome(u, tc, tr, RT); rc, rs = me.round_at(per, tc), me.round_at(per, tr); pages.append((lab, rc is not None and rc == rs))
            o = me.aggregate([p[0] for p in pages]) if pages else None; o = UO if o is None else o; orig[q] = o
            vx = [p for p in pages if not p[1] and p[0] in (CH, UN)]
            if any(p[0] == CH for p in vx): n = CH
            elif pages and all((not p[1]) and p[0] == UN for p in pages): n = UN
            elif o == UO: n = UO
            else: n = "UNDET"
            w4[q] = n
            if o in (CH, UN):
                c["det"] += 1; c["ch"] += o == CH
                if pages and all(p[1] for p in pages): c["only_same_round"] += 1
            c["xr_det"] += n in (CH, UN); c["xr_ch"] += n == CH
        out[a] = {"determinate_N": c["det"], "coverage_pct": round(100 * c["det"] / len(S), 4), "measured_drift_pct": round(100 * c["ch"] / c["det"], 4) if c["det"] else None,
                  "same_round_only_fraction_pct": round(100 * c["only_same_round"] / c["det"], 4) if c["det"] else None, "cross_round_N": c["xr_det"],
                  "cross_round_coverage_pct": round(100 * c["xr_det"] / len(S), 4), "cross_round_drift_pct": round(100 * c["xr_ch"] / c["xr_det"], 4) if c["xr_det"] else None}
        O_[a], W[a] = orig, w4
    out["gate_contrast_measured"] = part_b(O_["on"], O_["off"]); out["gate_contrast_cross_round"] = part_b(W["on"], W["off"])
    return out
OTHER = ["rerun_1h", "rerun_12h", "rerun_24h", "rerun_7d"]; RES = {}
for k in (4, 3, 2, 1):
    for sub in itertools.combinations(OTHER, k):
        R = ("run_00",) + sub; RES["+".join(R)] = {"n_rounds": len(R), **evaluate(R)}; print(len(R), R, json.dumps({a: RES["+".join(R)][a]["measured_drift_pct"] for a in ARMS}), flush=True)
full = RES["+".join(["run_00"] + OTHER)]; ref = {k: v for k, v in XR_REF["per_policy"].items()}
G = {"full panel reproduces cross-round gate-on drift/coverage/same-round/cross-round": (full["on"]["measured_drift_pct"], full["on"]["coverage_pct"], round(full["on"]["same_round_only_fraction_pct"], 2), full["on"]["cross_round_drift_pct"]) ==
         (ref["FreshCache"]["original_drift_pct"], ref["FreshCache"]["original_coverage_pct"], ref["FreshCache"]["pct_of_determinate_only_same_round"], ref["FreshCache"]["reported_drift_pct"]),
     "full panel reproduces cross-round gate-off": (full["off"]["measured_drift_pct"], full["off"]["cross_round_drift_pct"]) == (ref["FreshCache_AlwaysPassTemporal"]["original_drift_pct"], ref["FreshCache_AlwaysPassTemporal"]["reported_drift_pct"])}
H1 = {str(p): sha(p) for p in INP}; G["inputs unchanged"] = H0 == H1
json.dump({"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "label": "within-panel observation-density sensitivity (no ground truth; not a bias estimate)", "gates": G, "panels": RES, "input_sha256": H0}, open(O / "density_results.json", "w"), indent=1)
print("GATES", G)
