"""Re-aggregation of the held-out operating-point results (read-only inputs; writes outputs/operating_point_summaries/operating_point_reaggregation.json).
1. Cost break-even: generation cost at which FreshCache's modeled cost ties each Table-3 comparator (live and local search/fetch regimes).
2. Operation dominance of CategoryAware+FreshCache vs CategoryAware-style alone (searches, fetches, generations per 1k).
3. Holm correction across all Table-3 paired tests vs FreshCache.
4. Search savings without TemporalAlignQA: (a) FreshCache's held-out per-request routes restricted to non-TemporalAlignQA requests;
   (b) frozen engine replayed on the held-out stream with TemporalAlignQA clusters removed (read-only engine, no parameter changes);
   (c) as (b), additionally without paraphrase requests."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import collections, json, math, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); OUT = (OUTPUT_ROOT / "operating_point_summaries"); T1 = (OUTPUT_ROOT / "standalone_operating_points")
CO = json.load(open((OUTPUT_ROOT / "concurrency_cost/component_costs.json")))["coefficients"]; LK = json.load(open(T1 / "design_frozen.json"))["lookup_ms"]
DT = {r["point"]: r for r in json.load(open(T1 / "heldout_decision_table.json"))["decision_table"]}
DT.update({r["point"]: r for r in json.load(open(T1 / "heldout_decision_table_addendum.json"))["decision_table"] if r["point"] == "sttl_t0.70_k2"})
_T = json.load(open(OUTPUT_ROOT / "validation_split/split.json"))["test"]; N = _T["requests"]; NONRT = N - _T["requests_by_class"].get("REAL_TIME", 0)   # held-out requests and non-REAL_TIME requests (validation_heldout_split.py)
pts = {"FreshCache 0.75": ("FreshCache_primary", "FreshCache"), "FreshCache 0.80": ("FreshCache_thetaL2_0.80", "FreshCache"), "STTL QM (.80,1/4)": ("sttl_t0.80_k0.25", "SemanticTTL"),
       "STTL relaxed (.80,2)": ("sttl_t0.80_k2", "SemanticTTL"), "STTL CM (.60,1/4)": ("sttl_t0.60_k0.25", "SemanticTTL"), "STTL SM (.70,2)": ("sttl_t0.70_k2", "SemanticTTL"),
       "CA QM": ("ca_ca_d+0.000_m1.00", "CategoryAware"), "CA CM (+.075)": ("ca_ca_d+0.075_m1.00", "CategoryAware")}
OPS = {}
for name, (pid, fam) in pts.items():
    r = DT[pid]; OPS[name] = {"search": 1 - r["search_saved_pct"] / 100, "fetch": r["fetch_per_1k"] / 1000, "gen": r["gen_per_1k"] / 1000, "lk": LK[fam], "table_live_s": r["live_cost_s"]}
import csv; H80 = next(r for r in csv.DictReader(open(OUTPUT_ROOT / "categoryaware_freshcache/ca_composition_heldout_results.csv")) if r["method"] == "H80")   # selected CA + FreshCache L2/L3 composition
OPS["CA+FC L2/L3"] = {"search": 1 - round(float(H80["search_saved_pct"]), 2) / 100, "fetch": float(H80["fetch_per_1k"]) / 1000, "gen": float(H80["gen_per_1k"]) / 1000, "lk": LK["CategoryAware"] + LK["FreshCache"], "table_live_s": round(float(H80["live_p50_ms_per_request"]) / 1000, 3)}
emb = NONRT / N * CO["c_embed_ms"]["value"]
def cost(o, cg, reg): return o["search"] * CO["c_search_ms"][reg] + o["fetch"] * CO["c_fetch_ms"][reg] + o["gen"] * cg + emb + o["lk"]
chk = {k: round(cost(o, CO["c_gen_ms"]["value"], "live_p50") / 1000, 3) for k, o in OPS.items()}
BE = {}
for ref in ("FreshCache 0.75", "FreshCache 0.80"):
    f = OPS[ref]
    for k, o in OPS.items():
        if k.startswith("FreshCache"): continue
        row = {}
        for reg in ("live_p50", "local"):
            dg = f["gen"] - o["gen"]; rest = cost(o, 0, reg) - cost(f, 0, reg)
            row[reg] = {"gen_ms_at_tie": round(rest / dg, 1) if dg > 0 else None, "FreshCache_cheaper_when": "c_gen < tie" if dg > 0 else ("always" if rest > 0 else "never")}
        BE[f"{ref} vs {k}"] = row
# 2. dominance
dom = {k: {"search_per_1k": round(1000 * OPS[k]["search"], 1), "fetch_per_1k": round(1000 * OPS[k]["fetch"], 1), "gen_per_1k": round(1000 * OPS[k]["gen"], 1)} for k in ("CA QM", "CA+FC L2/L3")}
dom["composition_dominates"] = all(dom["CA+FC L2/L3"][x] <= dom["CA QM"][x] for x in ("search_per_1k", "fetch_per_1k", "gen_per_1k"))
# 3. Holm over Table-3 paired tests
TESTS = []
for name, (pid, _) in pts.items():
    if pid == "FreshCache_primary": continue
    r = DT[pid]; TESTS += [(f"{name}: WAI", r["vs_FreshCache_WAI_FConly_otheronly_p"][2]), (f"{name}: Suff", r["vs_FreshCache_SUFF_FConly_otheronly_p"][2])]
# gate, verifier and composition tests: read from the outputs that produce them (sufficiency p-values at the 4 decimals reported in Table 6)
G1 = json.load(open(OUTPUT_ROOT / "learned_l2_gate/gate_heldout_results.json"))["paired_gate_vs"]["theta_0.75"]; G2 = json.load(open(OUTPUT_ROOT / "learned_l2_gate/gate_heldout_results_tau2.json"))["paired_gate_vs"]["theta_0.75"]
VG = json.load(open(OUT / "verifier_gamma_paired_tests.json"))["gamma_0.55"]; CF = json.load(open(OUT / "categoryaware_composition_paired_tests.json"))["paired_vs_H80"]["FreshCache primary (theta_L2=0.75)"]
TESTS += [("NoCache: WAI", DT["NoCache"]["vs_FreshCache_WAI_FConly_otheronly_p"][2]), ("gate tau: WAI", G1["WAI_p"]), ("gate tau: Suff", G1["suff_p"]), ("gate tau2: WAI", G2["WAI_p"]), ("gate tau2: Suff", G2["suff_p"]),
          ("verifier gamma: WAI", VG["WAI_p"]), ("verifier gamma: Suff", round(VG["suff_p"], 4)), ("CA+FC vs FC: WAI", CF["WAI_p"]), ("CA+FC vs FC: Suff", round(CF["suff_p"], 4))]
m = len(TESTS); order = sorted(range(m), key=lambda i: TESTS[i][1]); adj = [None] * m; run = 0.0
for rank, i in enumerate(order): run = max(run, min(1.0, (m - rank) * TESTS[i][1])); adj[i] = run
HOLM = [{"test": t, "p": p, "p_holm": round(a, 6), "significant_0.05": a < 0.05} for (t, p), a in zip(TESTS, adj)]
# 4. without TemporalAlignQA
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
E = C.load("reported"); H = C.streams(E)["heldout"]
SRC = {json.loads(l)["query_id"]: json.loads(l)["source"] for l in open((DATA_ROOT / "queries.jsonl"))}
REC = {r["query_id"]: r for r in E["recs"]}; src = lambda q: SRC.get(REC[q].get("cluster_id") or q, SRC.get(q, "unknown"))
S = E["support_of"](H, E["rounds"]); m0, _, t0 = C.metrics(E, H, "FreshCache", S)
def saved(ids, tiers): n = len(ids); miss = sum(1 for q in ids if tiers[q]["tier"] in ("miss", "REAL_TIME_BYPASS")); return round(100 * (1 - miss / n), 2), n
qs = [r["query_id"] for _, r in H]; nt = [q for q in qs if src(q) != "temporal_align_qa"]
gate = {"route_recompute_reproduces_60.52": saved(qs, t0)[0] == round(m0["search_saved_pct"], 2)}
H2 = [(t, r) for t, r in H if src(r["query_id"]) != "temporal_align_qa"]; S2 = E["support_of"](H2, E["rounds"]); m2, _, t2 = C.metrics(E, H2, "FreshCache", S2)
H2o = [(t, r) for t, r in H2 if not r.get("is_paraphrase")]; S2o = E["support_of"](H2o, E["rounds"]); m3, _, _ = C.metrics(E, H2o, "FreshCache", S2o)
NOTAQ = {"source_counts_heldout": dict(collections.Counter(src(q) for q in qs)), "a_full_stream_routes_restricted_to_nonTAQ": dict(zip(("search_saved_pct", "n"), saved(nt, t0))),
         "b_replay_without_TAQ_clusters": {"n": len(H2), "search_saved_pct": m2["search_saved_pct"], "l2_hits": m2.get("l2_hits"), "l1_hits": m2.get("l1_hits")},
         "c_replay_without_TAQ_and_paraphrases": {"n": len(H2o), "search_saved_pct": m3["search_saved_pct"]}, "gates": gate}
R = {"cost_check_live_s": chk, "break_even": BE, "operation_dominance": dom, "holm": HOLM, "without_TemporalAlignQA": NOTAQ}
json.dump(R, open(OUT / "operating_point_reaggregation.json", "w"), indent=1, default=str); print(json.dumps(R, indent=1, default=str))
