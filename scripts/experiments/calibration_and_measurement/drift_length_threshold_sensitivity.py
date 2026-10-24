"""Observability sensitivity of the drift estimate to the snapshot body-length threshold, computed from the frozen
producer traces behind the FreshCache primary result (drift and coverage of S). No replay is run.

Frozen producer inputs: outputs/replay/table2/reported/traces.pkl (per-request outcomes of 8 methods, plus FreshCache
and FreshCache_AlwaysPassTemporal outcome() events for requests in S, each with call-site line, url, t_cache,
t_request and label) and outputs/replay/results/replays_reported_per_request.pkl (adds FreshCache_NoL1 and per-request
routes/reused lists). Input hashes are checked against configs/integrity/FILE_HASHES.csv.

The substantive-snapshot rule of scripts/data/round_content_table.py is re-applied in memory to the stored snapshot
text at MIN_BODY in {50, 100, 200, 400} (400 is the frozen rule). Page labels at each threshold are recomputed with
outcome() from freshcache/replay_engine/mixed_age_replay.py (a pure function of url, t_cache, t_request and the round table)
on the frozen events; within-request duplicate-URL L3 events are kept exactly as recorded. Policies without recorded
events get derived drift intervals. The population is the fixed primary support S for every threshold.

Writes outputs/observability_sensitivity/frozen/prereg.json (on first run), logs/gates.json and results/
(threshold_sensitivity.csv, worst_case_bounds.csv, pairwise_common_support.csv, missingness_characterization.csv,
request_level_observability.csv.gz, verdict.json)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, glob, gzip, hashlib, importlib.util, json, os, pathlib, pickle, statistics, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); CRR = (OUTPUT_ROOT / "replay"); W = (OUTPUT_ROOT / "observability_sensitivity")
for d in ("results", "logs", "frozen"): (W / d).mkdir(exist_ok=True)
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
TH = [50, 100, 200, 400]
PRE = {"written_utc": now(), "thresholds": TH, "primary": "400 = frozen rule; FreshCache drift / coverage on S (support_of), unchanged",
       "fixed_population": "frozen primary support S for every threshold",
       "exact_at_all_thresholds": ["FreshCache", "FreshCache_AlwaysPassTemporal"],
       "derived_for_other_policies": "at T<400 drift lies in [CH/(CH+UN+UO), (CH+UO)/(CH+UN+UO)] of the frozen 400 counts (monotone rule; verified)",
       "pairwise_T400": "Sec. 5.2 definition (all requests, both determinate) and within S",
       "pairwise_T_lt_400": "within S; exact vs AlwaysPassTemporal; for other scripts/baselines the baseline side of requests UNOBSERVABLE at 400 but newly FreshCache-determinable is unresolved and is set to all-CHANGED and all-UNCHANGED extremes",
       "classification_rule": {"a": "every FreshCache-vs-baseline difference keeps its sign at every threshold under both extremes, and for every baseline with higher common-support drift at 400, FreshCache's pessimistic bound < baseline's optimistic bound on S",
                               "b": "signs stable (under both extremes) for every comparison with |diff| >= 1 pp at 400, but the bound condition fails for some baseline, or only |diff| < 1 pp comparisons can flip",
                               "c": "any comparison with |diff| >= 1 pp at 400 can change sign at some threshold under the extremes"},
       "duplicate_url_semantics": "kept exactly as recorded by the frozen producer (later occurrences of a URL within a request can be L3 hits after the first occurrence populates L3)"}
PRE["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in PRE.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (W / "frozen/prereg.json").exists(): json.dump(PRE, open(W / "frozen/prereg.json", "w"), indent=1)
PRE = json.load(open(W / "frozen/prereg.json"))
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
INP = ["outputs/replay/table2/reported/traces.pkl", "outputs/replay/results/replays_reported_per_request.pkl", "outputs/replay/results/replays_reported.json",
       "outputs/replay/results/extras_reported.json", "outputs/replay/table2/reported/primary_metrics.csv", "scripts/experiments/ablations/mixed_age_trace_construction.py",
       "freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/execution_semantics.py", "freshcache/simulation/page_version_replay.py", "scripts/data/round_content_table.py",
       "data/derived/round_table.jsonl", "freshcache/method/tiered_cache_model.py", "data/queries.jsonl", "data/url_manifest.jsonl", "data/paraphrase_clusters.jsonl"]
INT = {p: {"sha256": sha(inpath(p)), "freeze": FH.get(p)} for p in INP}
for v in INT.values(): v["match"] = (v["sha256"] == v["freeze"]) if v["freeze"] else None
G = {"no_input_differs_from_freeze": not any(v["match"] is False for v in INT.values())}; NOT_IN_FREEZE = [p for p, v in INT.items() if v["match"] is None]
if not G["no_input_differs_from_freeze"]: json.dump({"integrity": INT}, open(W / "logs/STOPPED.json", "w"), indent=1); sys.exit("STOP integrity")
# ---------------- frozen engine functions (outcome/aggregate only; no replay)
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); sys.path.insert(0, str(ROOT / "freshcache/simulation")); sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); os.chdir(ROOT)
import mixed_age_replay as me   # noqa: E402  (must resolve to freshcache/replay_engine/mixed_age_replay.py; asserted below)
assert pathlib.Path(me.__file__).resolve() == (ROOT / "freshcache/replay_engine/mixed_age_replay.py")
import tiered_cache_model as exp    # noqa: E402
CH, UN, UO = me.CHANGED, me.UNCHANGED, me.UNOBS
TR = pickle.load(open(CRR / "table2/reported/traces.pkl", "rb")); PK = pickle.load(open(CRR / "results/replays_reported_per_request.pkl", "rb"))
FR = json.load(open(CRR / "results/replays_reported.json")); EX = json.load(open(CRR / "results/extras_reported.json"))
qids, S = TR["qids"], set(TR["support"]); T_OF = dict(zip(qids, TR["arrival_times"])); t2q = dict(zip(TR["arrival_times"], qids))
POL = list(TR["methods"]); SIG = {m: dict(zip(qids, TR["methods"][m]["_outcome_sig"])) for m in POL}
SIG["FreshCache_NoL1"] = PK["primary/FreshCache_NoL1"]["per"]; POL.append("FreshCache_NoL1")
G["traces_and_per_request_pkl_agree_for_shared_methods"] = all(SIG[m].get(q) == PK[f"primary/{m}"]["per"].get(q) for m in TR["methods"] if f"primary/{m}" in PK for q in qids)
def cnt(sig, pop):
    c = collections.Counter(sig.get(q) for q in pop); return {"CHANGED": c[CH], "UNCHANGED": c[UN], "UNOBSERVABLE": c[UO], "no_reuse": c[None]}
c0 = cnt(SIG["FreshCache"], S); G["primary_drift_and_coverage_from_frozen_trace_equal_reported_replay"] = (round(100 * c0["CHANGED"] / (c0["CHANGED"] + c0["UNCHANGED"]), 4), round(100 * (c0["CHANGED"] + c0["UNCHANGED"]) / len(S), 4)) == (FR["primary/FreshCache"]["drift_pct"], FR["primary/FreshCache"]["coverage_pct"])
# ---------------- thresholds on the frozen snapshot text
spec = importlib.util.spec_from_file_location("a1b", ROOT / "scripts/data/round_content_table.py"); A1B = importlib.util.module_from_spec(spec); spec.loader.exec_module(A1B)
assert A1B.MIN_BODY == 400
FROZ = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open((DATA_ROOT / "derived/round_table.jsonl"))}
TXT = {}
for run in A1B.RUNS:
    for p in glob.glob(str((DATA_ROOT / "snapshots") / run / "*.json")):
        try: s = json.load(open(p, encoding="utf-8"))
        except Exception: continue
        t = s.get("extracted_text") or ""; TXT[(s.get("url_hash") or pathlib.Path(p).stem, run)] = (len(t.strip()), bool(A1B.BLOCK.search(t[:400])), s.get("status_code"))
sub = lambda k, T: bool(TXT[k][0] >= T and not TXT[k][1] and TXT[k][2] == 200)
G["snapshot_set_equals_round_table"] = set(TXT) == {(u, r) for u, per in FROZ.items() for r in per}
G["T400_reproduces_every_frozen_substantive_flag"] = all(sub((u, r), 400) == v["substantive"] for u, per in FROZ.items() for r, v in per.items())
G["rule_monotone_substantive400_implies_lower"] = all(sub(k, T) for k in TXT if sub(k, 400) for T in TH)
ROUNDS = {T: {u: {r: {**v, "substantive": sub((u, r), T)} for r, v in per.items()} for u, per in FROZ.items()} for T in TH}
# ---------------- frozen events -> page labels at each threshold
EV = {m: TR["age_events"][m] for m in ("FreshCache", "FreshCache_AlwaysPassTemporal")}
G["events_only_from_engine_call_sites_286_349"] = all({e[0] for e in EV[m]} <= {286, 349} for m in EV)
G["T400_event_labels_reproduce_recorded"] = all(me.outcome(u, tc, tr, ROUNDS[400]) == lab for m in EV for _, u, tc, tr, lab in EV[m])
BYQ = {m: collections.defaultdict(list) for m in EV}
for m in EV:
    for e in EV[m]: BYQ[m][t2q[e[3]]].append(e)
def agg_T(m, q, T):
    es = BYQ[m].get(q)
    if not es: return SIG[m][q] if SIG[m][q] is None else ("NOEVENTS", SIG[m][q])
    labs = [me.outcome(u, tc, tr, ROUNDS[T]) for _, u, tc, tr, _ in es]; o = me.aggregate(labs)
    return (UO if o is None else o), labs
bad = [(m, q) for m in EV for q in S if SIG[m][q] is not None and (not BYQ[m].get(q) or agg_T(m, q, 400)[0] != SIG[m][q])]
G["T400_event_aggregation_reproduces_frozen_outcome_for_every_S_request"] = not bad
G["no_events_for_no_reuse_requests"] = all(not BYQ[m].get(q) for m in EV for q in S if SIG[m][q] is None)
dup = [(q, e) for q, es in BYQ["FreshCache"].items() for e in es if e[0] == 349 and e[2] == e[3]]
json.dump({"gates": G, "integrity": INT, "prereg_sha256": PRE["sha256"], "t400_aggregation_mismatches": bad[:20]}, open(W / "logs/gates.json", "w"), indent=1, default=str)
if not all(G.values()): sys.exit(f"STOP gates {[k for k, v in G.items() if not v]}")
# ---------------- per-threshold outcomes (exact for event arms, within S)
OUT = {(m, T): {} for m in EV for T in TH}; CAT = {}
for m in EV:
    for T in TH:
        for q in S:
            if SIG[m][q] is None: OUT[(m, T)][q] = None; continue
            o, labs = agg_T(m, q, T); OUT[(m, T)][q] = o
            if m == "FreshCache": CAT[(q, T)] = "determinable" if o in (CH, UN) else ("partially_observable" if any(l in (CH, UN) for l in labs) else "undeterminable")
for q in S:
    for T in TH: CAT.setdefault((q, T), "no_reuse")
def drift(c): d = c["CHANGED"] + c["UNCHANGED"]; return round(100 * c["CHANGED"] / d, 4) if d else None
def cov(c): return round(100 * (c["CHANGED"] + c["UNCHANGED"]) / len(S), 4)
def bnd(c):
    n = c["CHANGED"] + c["UNCHANGED"] + c["UNOBSERVABLE"]; return (round(100 * c["CHANGED"] / n, 4), round(100 * (c["CHANGED"] + c["UNOBSERVABLE"]) / n, 4)) if n else (None, None)
TS, BD = [], []
for T in TH:
    for m in POL:
        if m in EV:
            c = cnt(OUT[(m, T)], S); TS.append({"threshold_chars": T, "policy": m, "method": "exact (frozen events)", **c, "drift_pct": drift(c), "coverage_pct": cov(c),
                                                "drift_range_pct": "", "coverage_range_pct": ""})
            o, p = bnd(c); BD.append({"threshold_chars": T, "policy": m, "method": "exact (frozen events)", **c, "primary_drift_pct": drift(c), "optimistic_bound_pct": o, "pessimistic_bound_pct": p})
        else:
            c = cnt(SIG[m], S); o, p = bnd(c)
            TS.append({"threshold_chars": T, "policy": m, "method": "frozen 400 outcome" if T == 400 else "derived interval (monotone rule)", **(c if T == 400 else {k: "" for k in c}),
                       "drift_pct": drift(c) if T == 400 else "", "coverage_pct": cov(c) if T == 400 else "",
                       "drift_range_pct": "" if T == 400 else f"[{o}, {p}]", "coverage_range_pct": "" if T == 400 else f"[{cov(c)}, {round(100 * (c['CHANGED'] + c['UNCHANGED'] + c['UNOBSERVABLE']) / len(S), 4)}]"})
            if T == 400: BD.append({"threshold_chars": 400, "policy": m, "method": "frozen 400 outcome", **c, "primary_drift_pct": drift(c), "optimistic_bound_pct": o, "pessimistic_bound_pct": p})
# ---------------- pairwise
PW = []
def pair_exact(a, b, pop):
    both = [q for q in pop if a.get(q) in (CH, UN) and b.get(q) in (CH, UN)]
    if not both: return {"shared_n": 0}
    ra = 100 * sum(a[q] == CH for q in both) / len(both); rb = 100 * sum(b[q] == CH for q in both) / len(both)
    return {"shared_n": len(both), "freshcache_drift_pct": round(ra, 4), "baseline_drift_pct": round(rb, 4), "diff_pp": round(rb - ra, 4)}
ALLQ = qids
for b in POL[1:]:
    x = pair_exact(SIG["FreshCache"], SIG[b], ALLQ); PW.append({"threshold_chars": 400, "baseline": b, "population": "all requests (Sec. 5.2)", "method": "exact", **x})
    if b in EX["pairwise_5_2"]: G[f"pairwise_400_reproduces_frozen/{b}"] = (x["shared_n"], x["freshcache_drift_pct"], x["baseline_drift_pct"]) == (EX["pairwise_5_2"][b]["shared_n"], EX["pairwise_5_2"][b]["a_drift_pct"], EX["pairwise_5_2"][b]["b_drift_pct"])
    for T in TH:
        fc = OUT[("FreshCache", T)]
        if b in EV:
            PW.append({"threshold_chars": T, "baseline": b, "population": "within S", "method": "exact", **pair_exact(fc, OUT[(b, T)], S)}); continue
        bs = SIG[b]; known = [q for q in S if fc.get(q) in (CH, UN) and bs.get(q) in (CH, UN)]
        pend = [q for q in S if fc.get(q) in (CH, UN) and bs.get(q) == UO] if T < 400 else []
        res = {"threshold_chars": T, "baseline": b, "population": "within S", "method": "exact" if T == 400 else "known pairs + unresolved-baseline extremes",
               "shared_n": len(known), "unresolved_pairs": len(pend)}
        ext = {}
        for tag, val in (("all_unresolved_UNCHANGED", UN), ("all_unresolved_CHANGED", CH)):
            pool = known + pend
            if not pool: ext[tag] = None; continue
            fcd = 100 * sum(fc[q] == CH for q in pool) / len(pool); bd = 100 * (sum(bs[q] == CH for q in known) + (len(pend) if val == CH else 0)) / len(pool)
            ext[tag] = round(bd - fcd, 4)
        ra = 100 * sum(fc[q] == CH for q in known) / len(known) if known else None; rb = 100 * sum(bs[q] == CH for q in known) / len(known) if known else None
        res.update({"freshcache_drift_pct": round(ra, 4) if known else None, "baseline_drift_pct": round(rb, 4) if known else None, "diff_pp": round(rb - ra, 4) if known else None,
                    "diff_pp_if_unresolved_all_UNCHANGED": ext["all_unresolved_UNCHANGED"], "diff_pp_if_unresolved_all_CHANGED": ext["all_unresolved_CHANGED"]}); PW.append(res)
if not all(G.values()): json.dump({"gates": G}, open(W / "logs/gates.json", "w"), indent=1); sys.exit(f"STOP pairwise gate {[k for k, v in G.items() if not v]}")
# ---------------- missingness characterization (FreshCache, requests in S with reused content, 400 and per threshold counts)
REC = {r["query_id"]: r for r in exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))}
DOM = {u["url_hash"]: u.get("domain") or "" for r in REC.values() for u in r["urls"]}
tier = PK["primary/FreshCache"]["tier"]
feat = {}
for q in S:
    es = BYQ["FreshCache"].get(q, []); t = T_OF[q]; r = REC[q]
    feat[q] = {"freshness_class": r["freshness_class"], "paraphrase": bool(r.get("is_paraphrase")), "language": r.get("language"), "url_count": len(r["urls"]), "tier": tier[q],
               "reused_page_events": len(es), "within_request_duplicate_events": sum(1 for e in es if e[0] == 349 and e[2] == e[3]),
               "cache_age_h": (t - min(e[2] for e in es)) / 3600 if es else None,
               "snapshot_rounds_available_mean": statistics.mean(len(FROZ.get(e[1], {})) for e in es) if es else None,
               "substantive_rounds_available_mean": statistics.mean(sum(v["substantive"] for v in FROZ.get(e[1], {}).values()) for e in es) if es else None,
               "domains": [DOM.get(e[1], "") for e in es]}
GR = {g: [q for q in S if CAT[(q, 400)] == g] for g in ("determinable", "partially_observable", "undeterminable")}
def smd(vals_a, vals_b):
    a, b = [float(v) for v in vals_a if v is not None], [float(v) for v in vals_b if v is not None]
    if len(a) < 2 or len(b) < 2: return ""
    s = ((statistics.pvariance(a) + statistics.pvariance(b)) / 2) ** .5; return round((statistics.mean(a) - statistics.mean(b)) / s, 3) if s else 0.0
MC = [{"feature": "N (FreshCache requests in S with reused content, T=400)", "level": "", **{g: len(v) for g, v in GR.items()}, "SMD_determinable_vs_undeterminable": ""}]
for f, lv in [("freshness_class", x) for x in ("TIMELESS", "SLOW", "MEDIUM", "FAST")] + [("paraphrase", True), ("language", "en"), ("language", "ko")] + [("tier", x) for x in ("L1", "L2", "miss")]:
    MC.append({"feature": f"{f} (% of group)", "level": lv, **{g: round(100 * sum(feat[q][f] == lv for q in v) / len(v), 2) if v else "" for g, v in GR.items()},
               "SMD_determinable_vs_undeterminable": smd([feat[q][f] == lv for q in GR["determinable"]], [feat[q][f] == lv for q in GR["undeterminable"]])})
for f in ("url_count", "reused_page_events", "within_request_duplicate_events", "cache_age_h", "snapshot_rounds_available_mean", "substantive_rounds_available_mean"):
    def ms(v): x = [feat[q][f] for q in v if feat[q][f] is not None]; return f"{statistics.mean(x):.3f}; {statistics.median(x):.3f}" if x else ""
    MC.append({"feature": f"{f} (mean; median)", "level": "", **{g: ms(v) for g, v in GR.items()},
               "SMD_determinable_vs_undeterminable": smd([feat[q][f] for q in GR["determinable"]], [feat[q][f] for q in GR["undeterminable"]])})
top = collections.Counter(d for q in S for d in feat[q]["domains"]).most_common(10)
for dn, _ in top:
    MC.append({"feature": "reused-page domain (% of group's reused-page events)", "level": dn,
               **{g: round(100 * sum(feat[q]["domains"].count(dn) for q in v) / max(1, sum(len(feat[q]["domains"]) for q in v)), 2) for g, v in GR.items()}, "SMD_determinable_vs_undeterminable": ""})
for T in TH:
    c = collections.Counter(CAT[(q, T)] for q in S)
    MC.append({"feature": f"category counts at T={T} (within S)", "level": "", "determinable": c["determinable"], "partially_observable": c["partially_observable"], "undeterminable": c["undeterminable"],
               "SMD_determinable_vs_undeterminable": f"no_reuse={c['no_reuse']}"})
def wcsv(n, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(W / "results" / n, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
wcsv("threshold_sensitivity.csv", TS); wcsv("worst_case_bounds.csv", BD); wcsv("pairwise_common_support.csv", PW); wcsv("missingness_characterization.csv", MC)
with gzip.open(W / "results/request_level_observability.csv.gz", "wt", newline="", encoding="utf-8") as f:
    cols = ["query_id", "t", "in_S", "freshcache_tier"] + [f"freshcache_T{T}" for T in TH] + [f"category_T{T}" for T in TH] + [f"alwayspass_T{T}" for T in TH] + [f"{m}_T400" for m in POL] + \
           ["freshness_class", "paraphrase", "language", "url_count", "reused_page_events", "within_request_duplicate_events", "cache_age_h", "snapshot_rounds_available_mean"]
    w = csv.writer(f); w.writerow(cols)
    for q in qids:
        inS = q in S; fq = feat.get(q, {}); r = REC[q]
        w.writerow([q, T_OF[q], inS, tier[q]] + [OUT[("FreshCache", T)].get(q) if inS else "" for T in TH] + [CAT[(q, T)] if inS else "outside_S" for T in TH] +
                   [OUT[("FreshCache_AlwaysPassTemporal", T)].get(q) if inS else "" for T in TH] + [SIG[m].get(q) for m in POL] +
                   [r["freshness_class"], bool(r.get("is_paraphrase")), r.get("language"), len(r["urls"]), fq.get("reused_page_events", ""), fq.get("within_request_duplicate_events", ""),
                    fq.get("cache_age_h", ""), fq.get("snapshot_rounds_available_mean", "")])
# ---------------- verdict (pre-registered)
pw400 = {r["baseline"]: r for r in PW if r["threshold_chars"] == 400 and r["population"] == "within S"}
fc_pess = {T: next(x for x in BD if x["policy"] == "FreshCache" and x["threshold_chars"] == T)["pessimistic_bound_pct"] for T in TH}
V = {}
for b in POL[1:]:
    d400 = pw400[b].get("diff_pp")
    if d400 is None: V[b] = {"no_common_determinate_pairs_within_S": True}; continue
    vals = []
    for r in PW:
        if r["baseline"] != b or r["population"] != "within S": continue
        vals += [r["diff_pp"]] + ([r["diff_pp_if_unresolved_all_UNCHANGED"], r["diff_pp_if_unresolved_all_CHANGED"]] if "diff_pp_if_unresolved_all_UNCHANGED" in r and r.get("diff_pp_if_unresolved_all_UNCHANGED") != "" else [])
    signs = {(v > 0) - (v < 0) for v in vals if v is not None}
    bopt = next(x for x in BD if x["policy"] == b and x["threshold_chars"] == 400)["optimistic_bound_pct"]
    V[b] = {"diff_pp_400_within_S": d400, "all_diffs_considered": vals, "sign_stable": len(signs) <= 1, "material": abs(d400) >= 1.0, "baseline_higher": d400 > 0,
            "fc_pessimistic_by_T": fc_pess, "baseline_optimistic_400": bopt, "bound_condition": all(fc_pess[T] < bopt for T in TH) if d400 > 0 else None}
V_ = {b: v for b, v in V.items() if "material" in v}
mat = [b for b, v in V_.items() if v["material"] and not v["sign_stable"]]; bf = [b for b, v in V_.items() if v["baseline_higher"] and not v["bound_condition"]]; sm = [b for b, v in V_.items() if not v["material"] and not v["sign_stable"]]
verdict = "c" if mat else ("a" if not bf and not sm else "b")
DUP = {"freshcache_L3_events_with_t_cache_equal_t_request_in_S": len(dup), "requests": len({q for q, _ in dup}), "labels": dict(collections.Counter(e[4] for _, e in dup))}
json.dump({"gates": G, "prereg_sha256": PRE["sha256"], "stability": V, "material_flips": mat, "bound_failures": bf, "small_flips": sm, "verdict": verdict, "duplicate_url_semantics_kept": DUP,
           "finished": now()}, open(W / "results/verdict.json", "w"), indent=1, default=str)
json.dump({"gates": G, "integrity": INT, "not_in_freeze_recorded_by_hash": NOT_IN_FREEZE, "prereg_sha256": PRE["sha256"]}, open(W / "logs/gates.json", "w"), indent=1)
print("GATES OK", all(G.values())); print("verdict", verdict, "material flips", mat, "bound failures", bf, "small flips", sm); print(DUP)
for r in TS:
    if r["policy"] == "FreshCache" or r["threshold_chars"] == 400: print(r)
