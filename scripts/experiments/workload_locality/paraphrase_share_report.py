"""Aggregation and report for the paraphrase-share measurement, from work/runs.json. All statements computed from data."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import csv, datetime, hashlib, json, math, os, pathlib, statistics as stt; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
from scipy.stats import t as tdist
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "paraphrase_share")
R = json.load(open(O / "work/runs.json")); RUNS, G = R["runs"], R["gates"]; D = json.load(open(O / "design_frozen.json"))
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
BEF = json.load(open(O / "work/read_only_before.json")); UNCH = {p: fstat(inpath(p)) == v for p, v in BEF.items()}
M = ["requests", "base_requests", "paraphrase_requests", "paraphrase_share_pct", "unique_clusters", "search_saved_pct", "fresh_searches", "l1_hits", "l2_hits", "l3_hits", "l1_hit_rate_pct", "l2_hit_rate_pct",
     "l3_hits_per_request", "l3_page_reuse_rate_pct", "fetch_per_1k", "gen_per_1k", "avoided_searches", "avoided_within_cluster", "avoided_cross_cluster", "within_cluster_share_of_avoided_pct",
     "savings_within_cluster_pp", "savings_cross_cluster_pp", "step_drift_pct_SECONDARY", "step_coverage_pct_SECONDARY", "audit_cross_round_drift_pct", "audit_cross_round_determinate", "audit_cross_round_coverage_pct"]
COND = ["A_0pct", "B_10pct", "C_30pct", "D_full"]; by = {c: [r for r in RUNS if r["condition"] == c] for c in COND}
tq = tdist.ppf(0.975, 9)
def agg(c, m):
    v = [r[m] for r in by[c] if r[m] is not None]
    if len(by[c]) == 1: return {"value": v[0] if v else None}
    mu, sd = stt.mean(v), stt.stdev(v); hw = tq * sd / math.sqrt(len(v))
    return {"mean": round(mu, 4), "sd": round(sd, 4), "min": min(v), "max": max(v), "ci95_lo": round(mu - hw, 4), "ci95_hi": round(mu + hw, 4), "n_seeds": len(v),
            "seed42": next(r[m] for r in by[c] if r["seed"] == 42)}
SUMM = {c: {m: agg(c, m) for m in M} for c in COND}
with open(O / "paraphrase_share_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition", "metric", "value_or_mean", "sd", "min", "max", "ci95_lo", "ci95_hi", "seed42", "n_seeds"])
    for c in COND:
        for m in M:
            a = SUMM[c][m]; w.writerow([c, m, a.get("value", a.get("mean")), a.get("sd"), a.get("min"), a.get("max"), a.get("ci95_lo"), a.get("ci95_hi"), a.get("seed42"), a.get("n_seeds", 1)])
pt = lambda c, m: SUMM[c][m].get("value", SUMM[c][m].get("mean"))
TR = {}
for m in ("search_saved_pct", "l1_hit_rate_pct", "l2_hit_rate_pct", "l3_page_reuse_rate_pct", "fetch_per_1k", "gen_per_1k", "within_cluster_share_of_avoided_pct", "savings_cross_cluster_pp", "audit_cross_round_drift_pct", "step_drift_pct_SECONDARY"):
    seq = [pt(c, m) for c in COND]; inc = all(a < b for a, b in zip(seq, seq[1:])); dec = all(a > b for a, b in zip(seq, seq[1:]))
    TR[m] = {"sequence_A_B_C_D": seq, "monotone": "increasing" if inc else "decreasing" if dec else "not monotone"}
sep = max(r["search_saved_pct"] for r in by["B_10pct"]) < min(r["search_saved_pct"] for r in by["C_30pct"])
SUM = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "design_sha256": D["sha256"], "gates": G, "conditions": SUMM, "trend": TR, "every_B_seed_below_every_C_seed_savings": sep,
       "freshcache_configuration_changed": False, "configuration_selected": False, "read_only_files": {p: {**BEF[p], "unchanged": UNCH[p]} for p in BEF}, "all_read_only_unchanged": all(UNCH.values())}
json.dump(SUM, open(O / "paraphrase_share_summary.json", "w"), indent=1, default=str)
f2 = lambda v, d=2: "—" if v is None else f"{v:,.{d}f}"
def cell(c, m, d=2):
    a = SUMM[c][m]
    if "value" in a: return f2(a["value"], d)
    return f"{f2(a['mean'], d)} ± {f2(a['sd'], d)}"
L = []; P = L.append
P("# FreshCache savings under reduced paraphrase repetition"); P("")
P("This is a workload-sensitivity diagnostic.")
P("- The frozen engine and frozen settings are used; **no FreshCache parameter was tuned and no configuration was changed or selected**.")
P("- Every replay starts from an empty cache, and no request timestamp was changed.")
P("- These are subsamples of a benchmark trace; no claim is made that they represent production traffic."); P("")
P("## Did this experiment exist before?"); P("")
P("**No** at 10% and 30% under the frozen engine. The repo search found:")
P("- an earlier key-set keep-probability sweep (not part of this release) on the key-set engine. It has no 10% or 30% condition and did not use the replay engine.")
P("- `outputs/benchmark_locality/`: the replay-engine base-only replay (primary stream). It is reproduced here as condition A.")
P("- `outputs/archived_runs/.../natural_workload_l1/`: DailyQA natural repeats, L1 only, with no paraphrase-rate control.")
P("")
P("Workload construction code: `freshcache/simulation/request_schedules.py::build_stream(recs, 'zipf_uniform', 42)` via `freshcache/replay_engine/engine_loader.py::streams`. The query records come from `experiment.build_query_records` over `data/queries.jsonl`, `data/url_manifest.jsonl` and the paraphrase file."); P("")
P("## Construction of the four conditions (design sha256 `" + D["sha256"] + "`)"); P("")
NB, PB, PC, NPOOL = by["A_0pct"][0]["base_requests"], by["B_10pct"][0]["paraphrase_requests"], by["C_30pct"][0]["paraphrase_requests"], by["D_full"][0]["paraphrase_requests"]
P(f"- **A (0%)**: every base request (`is_paraphrase == False`), {NB:,} requests, one per cluster, with the original timestamps. This is exactly the base-only replay of cluster_locality_decomposition.py, reproduced (gate below).")
P(f"- **B (10%) and C (30%)**: all {NB:,} base requests plus P existing paraphrase requests, with P / ({NB:,} + P) = target share. That gives P = {PB:,} at 10% and P = {PC:,} at 30%.")
P(f"  - Paraphrase quotas per freshness class are proportional to each class's share of the {NPOOL:,}-request paraphrase pool.")
P("  - Within a class, each query cluster is a stratum and receives its proportional allocation (floor, plus a seeded systematic PPS draw on the fractional parts), so cluster popularity is preserved in expectation. Requests are drawn at random within a cluster.")
P("  - The retained requests keep their original timestamps and chronological order. Nothing is synthesised, and no metric is consulted.")
P("  - Seeds 42–51; seed 42 is the representative run.")
P(f"- **D (original)**: the frozen primary replay, {by['D_full'][0]['requests']:,} requests.")
P("- **Within- vs cross-cluster**: an avoided search (an L1 or L2 hit) is *within-cluster* when the reused entry was created by a request from the same query cluster, i.e. paraphrase reuse, and *cross-cluster* otherwise. This is the locality decomposition of cluster_locality_decomposition.py, via its gated provenance transcription."); P("")
P("**Gates:**")
for k, v in G.items(): P(f"- {k}: **{v}**")
P(""); P("## Results"); P("")
P("For B and C the entries are mean ± SD over 10 seeds; seed-level detail, min/max, CIs and seed 42 are in the next table and in `paraphrase_share_summary.csv`.")
P("")
P("| Metric | A 0% | B 10% | C 30% | D original |"); P("|---|---|---|---|---|")
LAB = {"requests": "Total requests", "base_requests": "Base requests", "paraphrase_requests": "Paraphrase requests", "paraphrase_share_pct": "Paraphrase share %", "unique_clusters": "Unique clusters",
       "search_saved_pct": "**Search savings %**", "fresh_searches": "Fresh searches", "l1_hits": "L1 hits", "l2_hits": "L2 hits", "l3_hits": "L3 hits", "l1_hit_rate_pct": "L1 hit rate %", "l2_hit_rate_pct": "L2 hit rate %",
       "l3_page_reuse_rate_pct": "L3 page-reuse rate %", "fetch_per_1k": "Fetches / 1k", "gen_per_1k": "Generations / 1k", "avoided_within_cluster": "Avoided searches: within-cluster",
       "avoided_cross_cluster": "Avoided searches: cross-cluster", "within_cluster_share_of_avoided_pct": "Within-cluster share of avoided %", "savings_within_cluster_pp": "Savings from within-cluster (pp)",
       "savings_cross_cluster_pp": "Savings from cross-cluster (pp)", "audit_cross_round_drift_pct": "Cross-round drift %", "audit_cross_round_determinate": "Cross-round determinate n",
       "audit_cross_round_coverage_pct": "Cross-round coverage %", "step_drift_pct_SECONDARY": "Step drift % (secondary)", "step_coverage_pct_SECONDARY": "Step coverage % (secondary)"}
INTM = {"requests", "base_requests", "paraphrase_requests", "unique_clusters", "fresh_searches", "l1_hits", "l2_hits", "l3_hits", "avoided_within_cluster", "avoided_cross_cluster", "audit_cross_round_determinate"}
for m, lab in LAB.items(): P(f"| {lab} | " + " | ".join(cell(c, m, 0 if m in INTM else 2) for c in COND) + " |")
P(""); P("**10-seed aggregates for the key metrics:**"); P("")
P("| Condition | Metric | Mean | SD | Min | Max | Seed-level 95% CI | Seed 42 |"); P("|---|---|---|---|---|---|---|---|")
for c in ("B_10pct", "C_30pct"):
    for m in ("search_saved_pct", "savings_cross_cluster_pp", "savings_within_cluster_pp", "within_cluster_share_of_avoided_pct", "fetch_per_1k", "gen_per_1k", "audit_cross_round_drift_pct", "step_drift_pct_SECONDARY"):
        a = SUMM[c][m]; P(f"| {c} | {LAB[m].strip('*')} | {f2(a['mean'])} | {f2(a['sd'])} | {f2(a['min'])} | {f2(a['max'])} | [{f2(a['ci95_lo'])}, {f2(a['ci95_hi'])}] | {f2(a['seed42'])} |")
P(""); P("The CIs reflect only the sampling variability of which paraphrases are retained, across seeds.")
P(""); P("**Trend across A → B → C → D (seed means):**"); P("")
P("| Metric | A | B | C | D | Pattern |"); P("|---|---|---|---|---|---|")
for m, v in TR.items(): P(f"| {LAB.get(m, m).strip('*')} | " + " | ".join(f2(x) for x in v["sequence_A_B_C_D"]) + f" | {v['monotone']} |")
P(f""); P(f"Every 10% seed has lower savings than every 30% seed: **{sep}**.")
sA, sB, sC, sD = (pt(c, "search_saved_pct") for c in COND); xB, xC, xD = (pt(c, "savings_cross_cluster_pp") for c in ("B_10pct", "C_30pct", "D_full"))
wB, wC, wD = (pt(c, "within_cluster_share_of_avoided_pct") for c in ("B_10pct", "C_30pct", "D_full"))
P(""); P("## Answers to the key questions"); P("")
P(f"1. **Savings at 10% paraphrase traffic:** {f2(sB)}% (seed-level 95% CI {f2(SUMM['B_10pct']['search_saved_pct']['ci95_lo'])}–{f2(SUMM['B_10pct']['search_saved_pct']['ci95_hi'])}; seed 42: {f2(SUMM['B_10pct']['search_saved_pct']['seed42'])}%). That is {f2(100 * sB / sD, 1)}% of the original replay's {f2(sD)}%, and {f2(sB - sA)} pp above the base-only {f2(sA)}%.")
P(f"2. **Savings at 30%:** {f2(sC)}% (CI {f2(SUMM['C_30pct']['search_saved_pct']['ci95_lo'])}–{f2(SUMM['C_30pct']['search_saved_pct']['ci95_hi'])}; seed 42: {f2(SUMM['C_30pct']['search_saved_pct']['seed42'])}%), which is {f2(100 * sC / sD, 1)}% of the original.")
P(f"3. **Is the benefit monotonic with repetition?** Savings {TR['search_saved_pct']['monotone']} across 0% → 10% → 30% → original ({f2(sA)} → {f2(sB)} → {f2(sC)} → {f2(sD)}), and the 10% and 30% seed distributions {'do not overlap' if sep else 'overlap'}.")
P(f"4. **Cross-cluster vs paraphrase reuse:** cross-cluster reuse contributes {f2(pt('A_0pct','savings_cross_cluster_pp'))} pp at 0% (all of it), {f2(xB)} pp at 10%, {f2(xC)} pp at 30% and {f2(xD)} pp in the original. Within-cluster (paraphrase) reuse accounts for {f2(wB)}%, {f2(wC)}% and {f2(wD)}% of avoided searches at 10%, 30% and original.")
cA, cB, cC, cD = (pt(c, "audit_cross_round_drift_pct") for c in COND)
P(f"5. **Does lower repetition change the operating point?**")
P(f"   - Savings: {f2(sA)}% at 0% → {f2(sB)}% at 10% → {f2(sC)}% at 30% → {f2(sD)}% original.")
P(f"   - Fetches per 1k: {cell('A_0pct','fetch_per_1k',1)} / {cell('B_10pct','fetch_per_1k',1)} / {cell('C_30pct','fetch_per_1k',1)} / {cell('D_full','fetch_per_1k',1)}.")
P(f"   - Generations per 1k: {cell('A_0pct','gen_per_1k',1)} / {cell('B_10pct','gen_per_1k',1)} / {cell('C_30pct','gen_per_1k',1)} / {cell('D_full','gen_per_1k',1)}.")
P(f"   - Cross-round drift: {f2(cA)}% / {f2(cB)}% / {f2(cC)}% / {f2(cD)}%, at coverage {cell('A_0pct','audit_cross_round_coverage_pct')} / {cell('B_10pct','audit_cross_round_coverage_pct')} / {cell('C_30pct','audit_cross_round_coverage_pct')} / {cell('D_full','audit_cross_round_coverage_pct')}%.")
P(f"   - Step drift (secondary): {cell('A_0pct','step_drift_pct_SECONDARY')} / {cell('B_10pct','step_drift_pct_SECONDARY')} / {cell('C_30pct','step_drift_pct_SECONDARY')} / {cell('D_full','step_drift_pct_SECONDARY')}%.")
P("   - The same configuration therefore sits at a different efficiency point depending on repetition. The configuration itself was not changed.")
P("")
P("## Do meaningful savings persist under reduced repetition?"); P("")
P(f"Yes, but at a lower level. With 10% paraphrase traffic FreshCache still avoids {f2(sB)}% of searches, and {f2(xB)} pp of that comes from cross-cluster semantic reuse rather than paraphrase repetition. The original replay's {f2(sD)}% depends on its {f2(pt('D_full','paraphrase_share_pct'))}% paraphrase share: most of its avoided searches ({f2(wD)}%) are within-cluster.")
P("These are properties of subsamples of this benchmark trace, not of production traffic. The subsampled replays also have lower request density than the original.")
P(""); P("## Integrity"); P("")
P(f"- All pre-existing files read are unchanged: **{SUM['all_read_only_unchanged']}** ({len(BEF)} files; sha256, size and mtime compared before and after).")
P("- The earlier step-function drift is shown only as condition D's secondary step drift and is not used as evidence.")
P("- New files exist only in `outputs/paraphrase_share/`. The retained paraphrase IDs for every seed are in `work/retained_paraphrase_ids.json`."); P("")
P("| Pre-existing file | sha256 | Unchanged |"); P("|---|---|---|")
for p_, v in SUM["read_only_files"].items(): P(f"| `{p_}` | `{v['sha256'][:16]}…` | {v['unchanged']} |")
open(O / "PARAPHRASE_SHARE_REPORT.md", "w").write("\n".join(L) + "\n")
open(O / "run.log", "a").write(f"[{SUM['generated_utc']}] paraphrase-share report written; all read-only unchanged: {SUM['all_read_only_unchanged']}\n"); print("report written", SUM["all_read_only_unchanged"])
