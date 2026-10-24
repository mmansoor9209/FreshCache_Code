"""Cross-round drift audit. READ-ONLY on every existing file; writes only inside outputs/cross_round_audit/.
Source: the frozen producer traces (outputs/replay/table2/reported/traces.pkl), which hold every
mixed_age_replay.outcome() event for FreshCache (gate on) and FreshCache_AlwaysPassTemporal (gate off) for requests in the drift support S:
(call-site line, url_hash, t_cache, t_request, label). Snapshot rounds are assigned with the current selection logic mixed_age_replay.round_at()
on the frozen round table. No replay is run; routing, admission, timestamps, thresholds, classes and seeds are untouched.
Request rule: CHANGED if any valid cross-round reused page is CHANGED; UNCHANGED only if every evaluated reused page is a valid cross-round
comparison and UNCHANGED; otherwise (no cross-round evidence, or only partial cross-round evidence) UNDETERMINABLE_CROSS_ROUND.
Requests whose original outcome is UNOBSERVABLE and that contain no cross-round CHANGED page keep UNOBSERVABLE."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, hashlib, json, os, pathlib, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); OUT = (OUTPUT_ROOT / "cross_round_audit"); CRR = (OUTPUT_ROOT / "replay")
LOG = open(OUT / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
READ = ["outputs/replay/table2/reported/traces.pkl", "outputs/replay/table2/reported/temporal_gate_bootstrap.json",
        "outputs/replay/table2/reported/parts_abcdf.json", "scripts/experiments/ablations/mixed_age_consistency_analysis.py", "scripts/experiments/ablations/mixed_age_trace_construction.py",
        "freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/execution_semantics.py", "freshcache/replay_engine/method_replay.py",
        "freshcache/simulation/page_version_replay.py", "data/derived/round_table.jsonl", "freshcache/method/tiered_cache_model.py"]
def stat(p):
    f = inpath(p); h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    st = os.stat(f); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
BEFORE = {p: stat(p) for p in READ}
log(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] start; files read (read-only):")
for p, v in BEFORE.items(): log(f"  {p} sha256={v['sha256']} mtime_ns={v['mtime_ns']} freeze_match={FH.get(p) == v['sha256'] if p in FH else 'not in freeze'}")
if any(p in FH and FH[p] != v["sha256"] for p, v in BEFORE.items()): log("STOP: a frozen input differs from the frozen manifest"); sys.exit(1)
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); sys.path.insert(0, str(ROOT / "freshcache/simulation")); sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); os.chdir(ROOT)
import numpy as np, mixed_age_replay as me   # noqa: E402
assert pathlib.Path(me.__file__).resolve() == (ROOT / "freshcache/replay_engine/mixed_age_replay.py")
CH, UN, UO = me.CHANGED, me.UNCHANGED, me.UNOBS
TR = pickle.load(open(CRR / "table2/reported/traces.pkl", "rb")); ROUNDS = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open((DATA_ROOT / "derived/round_table.jsonl"))}
qids, S, CLU = TR["qids"], set(TR["support"]), dict(zip(TR["qids"], TR["clusters"])); t2q = dict(zip(TR["arrival_times"], qids))
ARMS = {"FreshCache (gate on)": "FreshCache", "gate off (FreshCache_AlwaysPassTemporal)": "FreshCache_AlwaysPassTemporal"}
SIG = {a: dict(zip(qids, TR["methods"][m]["_outcome_sig"])) for a, m in ARMS.items()}
ROUTE = {1: "?"}
G = {}
EV = {a: collections.defaultdict(list) for a in ARMS}
for a, m in ARMS.items():
    for ln, u, tc, tr, lab in TR["age_events"][m]: EV[a][t2q[tr]].append((ln, u, tc, tr, lab))
    G[f"{m}: every recorded page label reproduces with mixed_age_replay.outcome"] = all(me.outcome(u, tc, tr, ROUNDS) == lab for es in EV[a].values() for _, u, tc, tr, lab in es)
    agg_ok = True
    for q in S:
        es = EV[a].get(q)
        if SIG[a][q] is None: agg_ok &= not es; continue
        o = me.aggregate([e[4] for e in es]) if es else None
        agg_ok &= ((UO if o is None else o) == SIG[a][q])
    G[f"{m}: aggregating events reproduces every frozen request outcome in S"] = agg_ok
# ---------------- frozen paired results reproduced exactly (mixed_age_consistency_analysis.py part B transcription)
PB = json.load(open(CRR / "table2/reported/temporal_gate_bootstrap.json"))
def part_b(on, off, B=1000, seed=42):
    per = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for q in qids:
        if q not in S: continue
        v = per[CLU[q]]; v[0] += on.get(q) == CH; v[1] += on.get(q) in (CH, UN); v[2] += off.get(q) == CH; v[3] += off.get(q) in (CH, UN); v[4] += 1
    cl = sorted(per); M = np.array([per[c] for c in cl], np.int64); C = len(cl)
    rng = np.random.RandomState(seed); idx = rng.randint(0, C, size=(B, C))
    diff = M[:, 2][idx].sum(1) / M[:, 3][idx].sum(1) - M[:, 0][idx].sum(1) / M[:, 1][idx].sum(1); lo, hi = np.percentile(diff, [2.5, 97.5])
    per2 = collections.defaultdict(lambda: [0, 0, 0])
    for q in qids:
        if q not in S: continue
        a_, b_ = on.get(q), off.get(q)
        if a_ in (CH, UN) and b_ in (CH, UN): v = per2[CLU[q]]; v[0] += a_ == CH; v[1] += b_ == CH; v[2] += 1
    cl2 = sorted(per2); M2 = np.array([per2[c] for c in cl2], np.int64); C2, N2 = len(cl2), int(M2[:, 2].sum())
    rng2 = np.random.RandomState(seed); idx2 = rng2.randint(0, C2, size=(B, C2)); n2 = M2[:, 2][idx2].sum(1).astype(float)
    d2 = M2[:, 1][idx2].sum(1) / n2 - M2[:, 0][idx2].sum(1) / n2; lo2, hi2 = np.percentile(d2, [2.5, 97.5])
    return {"own_determinate": {"gate_on_drift_pct": 100 * M[:, 0].sum() / M[:, 1].sum(), "gate_off_drift_pct": 100 * M[:, 2].sum() / M[:, 3].sum(),
                                "difference_pp": 100 * (M[:, 2].sum() / M[:, 3].sum() - M[:, 0].sum() / M[:, 1].sum()), "ci95_pp": [100 * lo, 100 * hi]},
            "strict_paired": {"clusters": C2, "requests": N2, "gate_on_drift_pct": 100 * M2[:, 0].sum() / N2, "gate_off_drift_pct": 100 * M2[:, 1].sum() / N2,
                              "difference_pp": 100 * (M2[:, 1].sum() - M2[:, 0].sum()) / N2, "ci95_pp": [100 * lo2, 100 * hi2], "replicates_off_gt_on": int((d2 > 0).sum())}}
ORIG = part_b(SIG["FreshCache (gate on)"], SIG["gate off (FreshCache_AlwaysPassTemporal)"])
sp = PB["sensitivity_strict_paired_denominator"]
G["frozen part B point difference reproduced"] = abs(ORIG["own_determinate"]["difference_pp"] - PB["point"]["difference_pp"]) < 1e-9
G["frozen part B CI reproduced"] = np.allclose(ORIG["own_determinate"]["ci95_pp"], PB["bootstrap"]["ci95_difference_pp"], atol=1e-9)
G["frozen strict-paired N/difference/CI reproduced"] = (ORIG["strict_paired"]["requests"] == sp["requests"] and abs(ORIG["strict_paired"]["difference_pp"] - sp["difference_pp"]) < 1e-9
                                                      and np.allclose(ORIG["strict_paired"]["ci95_pp"], sp["ci95_difference_pp"], atol=1e-9))
for k, v in G.items(): log("GATE", k, v)
if not all(G.values()): log("STOP: gate failed"); sys.exit(1)
# ---------------- cross-round classification
rows, XR_LABELS = [], {a: {} for a in ARMS}; stats = {}
for a, m in ARMS.items():
    c = collections.Counter()
    for q in qids:
        if q not in S: continue
        orig = SIG[a][q]; es = EV[a].get(q, [])
        pages = []
        for ln, u, tc, tr, lab in es:
            per = ROUNDS.get(u) or {}; rc, rs = me.round_at(per, tc), me.round_at(per, tr)
            same = (rc is not None and rc == rs); pages.append((ln, u, tc, tr, rc, rs, same, lab))
            rows.append({"request_id": q, "policy": m, "route": "L1" if ln == 286 else "L2/miss (L3 reuse)", "url_hash": u, "cache_version_time": tc, "serving_time": tr,
                         "cached_snapshot_round": rc, "serving_snapshot_round": rs, "same_round": "yes" if same else "no", "page_label": lab, "original_request_label": orig})
        valid_x = [p for p in pages if not p[6] and p[7] in (CH, UN)]
        if orig is None: new = None
        elif any(p[7] == CH for p in valid_x): new = CH
        elif pages and all((not p[6]) and p[7] == UN for p in pages): new = UN
        elif orig == UO: new = UO
        else: new = "UNDETERMINABLE_CROSS_ROUND"
        XR_LABELS[a][q] = new
        det = orig in (CH, UN)
        if det:
            c["orig_det"] += 1; c["orig_ch"] += orig == CH; c["orig_un"] += orig == UN
            if pages and all(p[6] for p in pages): c["det_only_same_round"] += 1
            elif any(p[6] for p in pages): c["det_mixed_same_and_cross"] += 1
            else: c["det_only_cross_round"] += 1
        c[f"xr_{new}"] += 1
        c["orig_ch_became_undeterminable"] += orig == CH and new != CH
    ndet = c[f"xr_{CH}"] + c[f"xr_{UN}"]
    stats[m] = {"support_S": len(S), "original_determinate_N": c["orig_det"], "original_CHANGED": c["orig_ch"], "original_UNCHANGED": c["orig_un"],
                "original_drift_pct": round(100 * c["orig_ch"] / c["orig_det"], 4), "original_coverage_pct": round(100 * c["orig_det"] / len(S), 4),
                "determinate_with_only_same_round_comparisons": c["det_only_same_round"], "pct_of_determinate_only_same_round": round(100 * c["det_only_same_round"] / c["orig_det"], 2),
                "determinate_with_mixed_same_and_cross_round": c["det_mixed_same_and_cross"], "determinate_with_only_cross_round": c["det_only_cross_round"],
                "reported_cross_round_determinate_N": ndet, "reported_CHANGED": c[f"xr_{CH}"], "reported_UNCHANGED": c[f"xr_{UN}"],
                "reported_drift_pct": round(100 * c[f"xr_{CH}"] / ndet, 4) if ndet else None, "coverage_after_exclusion_pct": round(100 * ndet / len(S), 4),
                "UNDETERMINABLE_CROSS_ROUND": c["xr_UNDETERMINABLE_CROSS_ROUND"], "UNOBSERVABLE": c[f"xr_{UO}"], "no_reuse": c["xr_None"],
                "original_CHANGED_not_CHANGED_after_cross_round": c["orig_ch_became_undeterminable"]}
    log(m, json.dumps(stats[m]))
COR = part_b(XR_LABELS["FreshCache (gate on)"], XR_LABELS["gate off (FreshCache_AlwaysPassTemporal)"])
pages_total = collections.Counter((r["policy"], r["same_round"]) for r in rows)
same_round_changed = sum(1 for r in rows if r["same_round"] == "yes" and r["page_label"] == CH)
SUM = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source": "outputs/replay/table2/reported/traces.pkl (frozen producer of the primary drift)",
       "gates": G, "per_policy": stats, "page_events": {f"{k[0]}|same_round={k[1]}": v for k, v in pages_total.items()}, "same_round_pages_labelled_CHANGED": same_round_changed,
       "paired_gate_off_minus_gate_on": {"original_frozen": ORIG, "cross_round_audit": COR}}
with open(OUT / "cross_round_request_level.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["cross_round_request_label"]); w.writeheader()
    for r in rows: w.writerow({**r, "cross_round_request_label": XR_LABELS[next(a for a, mm in ARMS.items() if mm == r["policy"])][r["request_id"]]})
AFTER = {p: stat(p) for p in READ}
SUM["read_only_files"] = {p: {"sha256": BEFORE[p]["sha256"], "mtime_ns": BEFORE[p]["mtime_ns"], "unchanged_after_run": BEFORE[p] == AFTER[p],
                              "matches_frozen_manifest": (FH[p] == BEFORE[p]["sha256"]) if p in FH else "not in freeze"} for p in READ}
json.dump(SUM, open(OUT / "cross_round_summary.json", "w"), indent=1, default=str)
log("paired original", json.dumps(ORIG)); log("paired cross-round", json.dumps(COR)); log("same-round CHANGED pages", same_round_changed)
log("all read-only files unchanged:", all(v["unchanged_after_run"] for v in SUM["read_only_files"].values()))
