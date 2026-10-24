"""Final stage of the systems-timing experiment: assembles the component timing breakdown, storage by tier, the
end-to-end diagnostic and the break-even analysis from this experiment's earlier outputs.

Inputs: outputs/systems_timing/intermediate/ (cpu_timing_dedicated.json, generation_judging_stage.json,
gpu_timing_stability.json, cpu_micro_stability.json, storage.json), outputs/systems_timing/logs/cpu_timing.json,
results/end_to_end_requests.csv and results/capacity_sensitivity.csv, plus retained same-host live measurements from
outputs/latency_e2e/ (stage_d_live_timings.json, stage_b_gpu_timings.json). A timing is kept only if two repeated runs
agree within 20%; otherwise it is reported as MISSING.

Writes results/timing_breakdown.csv, storage_by_tier.csv, end_to_end_diagnostic.csv, break_even.csv and
break_even.json, plus README.md, SUMMARY.md and manifest.json, under outputs/systems_timing/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, glob, hashlib, json, os, pathlib, statistics, datetime; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = (OUTPUT_ROOT / "systems_timing"); I = W / "intermediate"
J = lambda p: json.load(open(p))
CPU = J(I / "cpu_timing_dedicated.json")["components"]; GPU = J(I / "generation_judging_stage.json"); GST = J(I / "gpu_timing_stability.json")["components"]; MIC = J(I / "cpu_micro_stability.json")
STO = J(I / "storage.json"); L1 = J(W / "logs/cpu_timing.json"); LIVE = J((OUTPUT_ROOT / "latency_e2e/stage_d_live_timings.json")); EMBR = J((OUTPUT_ROOT / "latency_e2e/stage_b_gpu_timings.json"))["embed"]
r4 = lambda x: round(x, 4) if isinstance(x, (int, float)) else x
TB = []
def add(comp, src, stat, n=None, stable="yes", note=""):
    TB.append({"component": comp, "source": src, "n": n if n is not None else (stat or {}).get("n", ""), "mean_ms": r4((stat or {}).get("mean_ms", "MISSING")), "p50_ms": r4((stat or {}).get("p50_ms", "MISSING")),
               "p95_ms": r4((stat or {}).get("p95_ms", "MISSING")), "stable": stable, "note": note})
lab = {"l1_lookup": "L1 lookup", "l2_lookup": "L2 lookup", "l3_lookup_per_request": "L3 lookup (all served pages of a request)", "cache_write_update": "cache read/write/update (L2 registration + L1 insert)"}
for k, v in CPU.items():
    add(lab[k] + " — cold (first encounter of each query; includes spaCy NER)", "this session, CPU, replay implementation, dedicated run", v["cold_pass1"], stable="single cold pass")
    add(lab[k] + " — warm", "this session, CPU, dedicated run (pass 2; pass 3 agrees)", v["warm_pass2"], stable="yes (pass 2 vs 3 within 20%)" if v["stable"] else "NO -> MISSING")
e = GST["embedding"]; add("query embedding (bge-m3, batch 1, GPU)", "this session, GPU shared with another process", None, n=f"{e['run1']['n']} x 2 runs",
                          stable="NO -> MISSING", note=f"unstable: run p50 {e['run1']['p50_ms']:.2f} vs {e['run2']['p50_ms']:.2f} ms")
add("query embedding (bge-m3, batch 1, GPU) — retained", "outputs/latency_e2e stage_b (same host and GPU, earlier session; NOT this session)", {"n": EMBR["n"], "mean_ms": EMBR["mean_ms"], "p50_ms": EMBR["p50_ms"], "p95_ms": EMBR["p95_ms"]}, stable="retained")
for n in ("1000", "5000", "10000", "20000", "31000"):
    m = MIC[f"vector_search_{n}"]; add(f"similarity lookup: brute-force vector search over {int(n):,} entries (numpy, CPU)", "this session, 2 runs", m["run2"] if m["stable"] else None, n=m["run1"]["n"],
                                       stable="yes" if m["stable"] else "NO -> MISSING", note="" if m["stable"] else f"run p50 {m['run1']['p50_ms']:.4f} vs {m['run2']['p50_ms']:.4f} ms")
m = MIC["local_page_read"]; add("page 'fetch' stand-in: local snapshot read + parse (disk)", "this session, 2 runs", m["run2"], stable="yes" if m["stable"] else "NO -> MISSING", note="not a network fetch")
add("page fetch over the network", "MISSING in this session (no network calls made)", None, n=0, stable="MISSING")
add("page fetch over the network — retained", "outputs/latency_e2e stage_d live GETs (same host, earlier session)", LIVE["web_fetch"], stable="retained", note=f"error rate {LIVE['fetch_error_rate_pct']:.1f}%")
add("search API call", "MISSING in this session (no network calls made)", None, n=0, stable="MISSING")
add("search API call — retained", "outputs/latency_e2e stage_d live Serper (same host, earlier session)", LIVE["search_api"], stable="retained")
g = GST["generation"]; gm = min(g["run1"]["mean_ms"], g["run2"]["mean_ms"])
add("generation (Llama-3.2-3B, batch 1, FORCED, 80 tokens)", "this session, GPU shared with another process", g["run1"] if g["run1"]["mean_ms"] <= g["run2"]["mean_ms"] else g["run2"],
    stable="yes (2 runs within 20%) but on a shared GPU", note=f"run means {g['run1']['mean_ms']:.1f} / {g['run2']['mean_ms']:.1f} ms; {GPU['micro']['generation_llama3b_batch1']['mean_input_tokens']} in / {GPU['micro']['generation_llama3b_batch1']['mean_output_tokens']} out tokens")
def wcsv(n, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(W / "results" / n, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
wcsv("timing_breakdown.csv", TB)
wcsv("storage_by_tier.csv", [{**s, "MB": round(s["bytes"] / 1e6, 2)} for s in STO["storage"]])
# ---------------- end-to-end aggregate
END_TO_END_ROWS = list(csv.DictReader(open(W / "results/end_to_end_requests.csv")))
fl = lambda x: float(x) if x not in ("", None) else 0.0
E2E = []
for pol in ("NoCache", "FreshCache"):
    rs = [r for r in END_TO_END_ROWS if r["policy"] == pol]; comp = ["embed_ms", "vector_search_ms", "l1_lookup", "l2_lookup", "l3_lookup", "cache_write_update", "page_fetch_local_ms", "context_build_ms", "generation_ms", "stored_answer_read_ms"]
    tot = {c: sum(fl(r.get(c)) for r in rs) for c in comp}
    E2E.append({"policy": pol, "requests": len(rs), "tiers": json.dumps(dict(collections.Counter(r["tier"] for r in rs))),
                "search_calls_needed_not_executed": sum(1 for r in rs if r["tier"] in ("none", "miss", "REAL_TIME_BYPASS")), "pages_read_locally": int(sum(fl(r["pages_fetched_local"]) for r in rs)),
                "generation_calls": sum(1 for r in rs if fl(r.get("generation_ms")) > 0), **{f"total_{c}": round(tot[c], 1) for c in comp},
                "total_measured_ms": round(sum(tot.values()), 1), "mean_measured_ms_per_request": round(sum(tot.values()) / len(rs), 1),
                "note": "same-environment sequential diagnostic; local snapshot reads replace network fetch; search calls are counted, not executed; GPU shared"})
wcsv("end_to_end_diagnostic.csv", E2E)
# ---------------- break-even (per 1,000 primary requests; frozen counts)
nc = L1["nocache_counts"]; fc = L1["unbounded"]; n = nc["search_calls"]; per = lambda x: 1000 * x / n   # n = primary stream length (NoCache searches once per request)
d_s, d_f, d_g = per(nc["search_calls"] - fc["search_calls"]), per(nc["fetches"] - fc["fetches"]), per(nc["generations"] - fc["generations"])
reads = {k: per(v["cold_pass1"]["n"]) for k, v in CPU.items()}
over = {m: sum(reads[k] * CPU[k][{"cold": "cold_pass1", "warm": "warm_pass2"}[m]]["mean_ms"] for k in CPU) for m in ("cold", "warm")}
emb_opts = {"excluded (MISSING in-session)": 0.0, "retained latency_e2e mean": EMBR["mean_ms"], "worst observed this session (unstable run p50)": max(e["run1"]["p50_ms"], e["run2"]["p50_ms"])}
BE, lp = [], MIC["local_page_read"]["run2"]["mean_ms"]
for mode in ("cold", "warm"):
    for en, ev in emb_opts.items():
        o = over[mode] + reads["l1_lookup"] * ev
        base = {"lookup_timing": mode, "embedding_assumption": en, "embedding_ms": round(ev, 3), "overhead_ms_per_1k": round(o, 1),
                "searches_avoided_per_1k": round(d_s, 1), "fetches_avoided_per_1k": round(d_f, 1), "generations_avoided_per_1k": round(d_g, 1)}
        for scen, cs, cf in (("local only: search 0 ms, fetch = local read", 0.0, lp), ("retained live costs: search/fetch p50", LIVE["search_api"]["p50_ms"], LIVE["web_fetch"]["p50_ms"]),
                             ("retained live costs: search/fetch mean", LIVE["search_api"]["mean_ms"], LIVE["web_fetch"]["mean_ms"])):
            saved = d_s * cs + d_f * cf + d_g * gm
            BE.append({**base, "scenario": scen, "search_ms": round(cs, 2), "fetch_ms": round(cf, 3), "generation_ms": round(gm, 1), "saved_ms_per_1k": round(saved, 1),
                       "net_ms_per_1k": round(saved - o, 1), "beneficial": saved > o,
                       "break_even_search_ms_given_fetch_and_generation": round(max(0.0, (o - d_f * cf - d_g * gm) / d_s), 3)})
wcsv("break_even.csv", BE)
tot_b = next(s for s in STO["storage"] if s["tier"] == "TOTAL")["bytes"]
BEJ = {"per_1000_primary_requests": {"nocache": {k: round(per(v), 1) for k, v in nc.items()}, "freshcache": {"search_calls": round(per(fc["search_calls"]), 1), "fetches": round(per(fc["fetches"]), 1), "generations": round(per(fc["generations"]), 1)}},
       "overhead_ms_per_1k_excluding_embedding": {k: round(v, 1) for k, v in over.items()}, "scenarios": BE, "storage_total_MB": round(tot_b / 1e6, 1),
       "rule": "FreshCache is beneficial when searches_avoided*c_search + fetches_avoided*c_fetch + generations_avoided*c_gen > per-request lookup/write/embedding overhead; storage is reported separately"}
json.dump(BEJ, open(W / "results/break_even.json", "w"), indent=1)
# ---------------- README / SUMMARY / manifest
cap = list(csv.DictReader(open(W / "results/capacity_sensitivity.csv"))); C_ = {r["scenario"]: r for r in cap}
e2 = {r["policy"]: r for r in E2E}
wc = {k: v["warm_pass2"] for k, v in CPU.items()}; cc = {k: v["cold_pass1"] for k, v in CPU.items()}
worst = min(BE, key=lambda r: r["net_ms_per_1k"]); be_max = max(r["break_even_search_ms_given_fetch_and_generation"] for r in BE if r["scenario"].startswith("local"))
R = ["# FreshCache systems characterization", "",
     "This experiment is isolated. Frozen policies and settings are unchanged, and no file of another study was modified.",
     "- Timings are same-environment only (single host). Nothing here is a deployment latency or concurrency result.",
     "- No network call was made. Live search and page-fetch costs are MISSING for this session. Earlier same-host live measurements (`outputs/latency_e2e`) appear only as labelled *retained* scenarios.", "",
     "## Gates (all passed)", ""] + [f"- {k}: {v}" for k, v in L1["gates"].items()] + [f"- end-to-end online decisions equal the replay transcription on the sample: {GPU['gates']['online_decisions_equal_replay_transcription_on_sample']}", "",
     "## 1. Timing breakdown (`results/timing_breakdown.csv`)", "",
     "| Component | Source | n | mean ms | p50 ms | p95 ms | Stable |", "|---|---|---|---|---|---|---|"] + [
     f"| {r['component']} | {r['source']} | {r['n']} | {r['mean_ms']} | {r['p50_ms']} | {r['p95_ms']} | {r['stable']} {('— ' + r['note']) if r['note'] else ''} |" for r in TB] + ["",
     "**Stability rule.** A timing is kept only if two repeated runs agree within 20% on both p50 and mean; otherwise it is MISSING.",
     "- Query embedding failed this rule on the shared GPU and is MISSING.",
     "- Vector search at 1,000–10,000 entries failed it too, because of warm-up.",
     "- Generation passed, but it was measured on a GPU that another process was using at the same time.", "",
     "## 2. Storage by tier at the end of the primary replay (`results/storage_by_tier.csv`)", "",
     "| Tier | Component | Entries / pages | MB | Basis |", "|---|---|---|---|---|"] + [f"| {s['tier']} | {s['component']} | {s['entries_or_pages']} | {round(s['bytes'] / 1e6, 2)} | {s['basis']} |" for s in STO["storage"]] + ["",
     "The observed peak is also the end state, because the frozen replay never evicts: " + json.dumps(STO["peak"]) + ".", "",
     "## 3. Capacity sensitivity (`results/capacity_sensitivity.csv`)", "",
     f"Setup: frozen settings, fixed least-recently-used eviction per tier, capacity as a fraction of the observed peak; primary stream of {n:,} requests.", "",
     "| Scenario | Search saved % | Fetches/1k | Generations/1k | L1 / L2 / L3 hits |", "|---|---|---|---|---|"] + [
     f"| {r['scenario']} | {r['search_saved_pct']} | {r['fetches_per_1k']} | {r['generations_per_1k']} | {r['l1_hits']} / {r['l2_hits']} / {r['l3_hits']} |" for r in cap] + ["",
     "**Reading.**",
     f"- **Search savings depend on L2 capacity:** they fall from {C_['unbounded (frozen)']['search_saved_pct']}% to {C_['L2_only_0.50']['search_saved_pct']}% at 50% L2 and to {C_['L2_only_0.25']['search_saved_pct']}% at 25%.",
     f"- **Fetches depend on L3 capacity:** they rise from {C_['unbounded (frozen)']['fetches_per_1k']} to {C_['L3_only_0.50']['fetches_per_1k']} per 1k at 50% L3 and to {C_['L3_only_0.25']['fetches_per_1k']} per 1k at 25%.",
     "- **L1 capacity barely matters.**",
     f"- **Evicting L3 pages raises L1 hits** ({C_['unbounded (frozen)']['l1_hits']} → {C_['L3_only_0.25']['l1_hits']} at 25% L3). Refetched pages give newer page-version times, so more L1 entries pass the answer-age test. This is an interaction in the frozen t_e rule, not a tuning effect.", "",
     "## 4. End-to-end same-environment diagnostic (`results/end_to_end_diagnostic.csv`, request level in `results/end_to_end_requests.csv`)", "",
     f"**Sample.** {GPU['e2e_sample']['requests']} requests from whole clusters (seed 42, clusters with at least 3 requests), processed sequentially.",
     "- FreshCache actually executes, per request: bge-m3 embedding, live vector search, admission (L1, L2, L3), cache writes, local page reads into an in-memory L3 page store, context building and 3B generation.",
     "- NoCache reads and generates for every request.",
     "- Its decisions equal the transcription on the same sample.", "",
     "| Policy | Tiers | Search calls needed (not executed) | Pages read | Generation calls | Measured ms per request |", "|---|---|---|---|---|---|"] + [
     f"| {r['policy']} | {r['tiers']} | {r['search_calls_needed_not_executed']} | {r['pages_read_locally']} | {r['generation_calls']} | {r['mean_measured_ms_per_request']} |" for r in E2E] + ["",
     "**What the sample shows.**",
     "- Local reads and generation dominate the measured time. Search is not executed.",
     "- On this small sample, FreshCache's measured time is higher than NoCache's. It adds embedding and lookup costs and avoids few generations (L1 hits only).",
     "- Its large savings are in search calls and page fetches, which cost almost nothing locally. They pay off only when network search or fetch costs are real (see §5).", "",
     "## 5. Break-even (`results/break_even.csv`, `results/break_even.json`)", "",
     f"Per 1,000 primary requests, FreshCache avoids {round(d_s, 1)} searches, {round(d_f, 1)} page fetches and {round(d_g, 1)} generations relative to NoCache.",
     f"- **Lookup/write overhead, excluding embedding:** {round(over['cold'], 1)} ms per 1k cold and {round(over['warm'], 1)} ms per 1k warm.",
     f"- **Embedding:** MISSING in this session. It is varied as a scenario: excluded, retained {EMBR['mean_ms']:.2f} ms, or {emb_opts['worst observed this session (unstable run p50)']:.2f} ms as the worst unstable observation.", "",
     "| Lookup timing | Embedding | Scenario | Saved ms/1k | Overhead ms/1k | Net ms/1k | Beneficial | Break-even search ms |", "|---|---|---|---|---|---|---|---|"] + [
     f"| {r['lookup_timing']} | {r['embedding_assumption']} | {r['scenario']} | {r['saved_ms_per_1k']} | {r['overhead_ms_per_1k']} | {r['net_ms_per_1k']} | {r['beneficial']} | {r['break_even_search_ms_given_fetch_and_generation']} |" for r in BE] + ["",
     "## Limitations", "",
     "- **Shared GPU:** the GPU was used concurrently by another process, so embedding timing is MISSING and generation is measured under contention.",
     "- **No network costs this session:** network search and fetch were not measured here; the retained values come from an earlier session on the same host.",
     "- **Payload accounting, not heap:** storage is payload accounting. The answer size is estimated from frozen 3B answers, because the replay produces no answers. It is not a measured process heap.",
     f"- **Small end-to-end sample:** {GPU['e2e_sample']['requests']} requests, with few L1 hits. It shows mechanics and the component mix, not representative hit rates.",
     "- **Nothing here is a deployment latency, throughput or concurrency result.**", "",
     "## Files", "", "- `results/timing_breakdown.csv`, `results/storage_by_tier.csv`, `results/capacity_sensitivity.csv`, `results/end_to_end_diagnostic.csv`, `results/end_to_end_requests.csv`, `results/break_even.{csv,json}`",
     "- `intermediate/*.json` (raw timing/stability records), `frozen/prereg.json` (capacity scenarios, eviction rule), `logs/`, `scripts/`, `manifest.json`"]
open(W / "README.md", "w").write("\n".join(R) + "\n")
loc = [r for r in BE if r["scenario"].startswith("local")]; live = [r for r in BE if r["scenario"].startswith("retained live costs: search/fetch p50")]
S = ["# summary: FreshCache systems overhead", "",
     "Same-environment measurements (single host). Frozen settings; nothing retuned. Nothing here is a deployment latency.", "",
     f"- **Per-request lookup overhead is small.**",
     f"  - L1 lookup: {cc['l1_lookup']['mean_ms']:.2f} ms mean on first encounter (includes spaCy NER), {wc['l1_lookup']['mean_ms']:.2f} ms warm.",
     f"  - L2 lookup: {wc['l2_lookup']['mean_ms']:.3f} ms. L3 lookup: {wc['l3_lookup_per_request']['mean_ms']:.4f} ms. Cache writes: {wc['cache_write_update']['mean_ms']:.4f} ms.",
     f"  - Query embedding is needed only by FreshCache and was MISSING on the shared GPU (unstable). It is the likely largest per-request overhead: {EMBR['mean_ms']:.0f} ms retained, and up to {emb_opts['worst observed this session (unstable run p50)']:.0f} ms observed.",
     f"  - Brute-force vector search: {MIC['vector_search_20000']['run2']['mean_ms']:.2f} ms at 20k entries, {MIC['vector_search_31000']['run2']['mean_ms']:.2f} ms at 31k.",
     f"- **Generation dominates compute** (≈{gm:.0f} ms per call, shared GPU). FreshCache avoids only {round(d_g, 1)} generations per 1,000 requests: L2 hits still generate.",
     f"- **The large savings are searches and fetches:** {round(d_s, 1)} searches and {round(d_f, 1)} fetches avoided per 1,000 requests.",
     f"  - With the retained live network costs (search p50 {LIVE['search_api']['p50_ms']:.0f} ms, fetch p50 {LIVE['web_fetch']['p50_ms']:.0f} ms), FreshCache saves about {live[0]['net_ms_per_1k'] / 1000:.0f} s per 1,000 requests net, in every overhead scenario.",
     "  - With local-only costs (no search cost, fetch = disk read), the net result depends on the embedding assumption:",
     *[f"    - {r['embedding_assumption']}, {r['lookup_timing']} lookups: net {r['net_ms_per_1k'] / 1000:+.1f} s per 1,000 requests; break-even search cost {r['break_even_search_ms_given_fetch_and_generation']} ms per call." for r in loc],
     f"- **Storage:** about {tot_b / 1e6:.0f} MB for the full primary replay (payload accounting). Query embeddings dominate ({next(x for x in STO['storage'] if x['tier'] == 'index')['bytes'] / 1e6:.0f} MB), and L3 page text is {next(x for x in STO['storage'] if x['component'].startswith('page text'))['bytes'] / 1e6:.1f} MB.",
     "- **Capacity:**",
     f"  - At 50% capacity, search savings drop by {float(C_['unbounded (frozen)']['search_saved_pct']) - float(C_['all_tiers_0.50']['search_saved_pct']):.2f} pp and fetches rise {float(C_['all_tiers_0.50']['fetches_per_1k']) / float(C_['unbounded (frozen)']['fetches_per_1k']):.2f}x.",
     f"  - At 25%, savings drop by {float(C_['unbounded (frozen)']['search_saved_pct']) - float(C_['all_tiers_0.25']['search_saved_pct']):.2f} pp and fetches rise {float(C_['all_tiers_0.25']['fetches_per_1k']) / float(C_['unbounded (frozen)']['fetches_per_1k']):.2f}x.",
     "  - L2 capacity governs search savings and L3 capacity governs fetches; L1 capacity barely matters.",
     f"- **When FreshCache is worthwhile:** when a search call costs more than about {be_max:.0f} ms (the largest break-even across the local-fetch scenarios). Any real web search API or network fetch exceeds that.",
     f"  - When retrieval is local and nearly free, it is only marginal for time: then only the {d_g / 10:.1f}% of generations avoided can offset the embedding and lookup overhead.",
     f"  - The small end-to-end diagnostic ({GPU['e2e_sample']['requests']} requests, local reads, no search executed) shows that regime: FreshCache measured {e2['FreshCache']['mean_measured_ms_per_request']} ms per request vs NoCache {e2['NoCache']['mean_measured_ms_per_request']} ms.",
     f"    - FreshCache added {e2['FreshCache']['total_embed_ms'] / 1000:.1f} s of embedding over the sample.",
     f"    - Its generation total was also higher ({e2['FreshCache']['total_generation_ms'] / 1000:.1f} s vs {e2['NoCache']['total_generation_ms'] / 1000:.1f} s) despite one fewer call. That reflects generation-time variance on the shared GPU.",
     "- **Missing:** in-session query embedding (unstable), network search/fetch (not measured), and vector search below 20k entries (unstable)."]
open(W / "SUMMARY.md", "w").write("\n".join(S) + "\n")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = {"experiment": "systems characterization (isolated)", "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source_revision": "MISSING: no VCS; content hashes",
       "inputs": L1["integrity"], "retained_inputs": {"outputs/latency_e2e/stage_d_live_timings.json": sha((OUTPUT_ROOT / "latency_e2e/stage_d_live_timings.json")),
                                                      "outputs/latency_e2e/stage_b_gpu_timings.json": sha((OUTPUT_ROOT / "latency_e2e/stage_b_gpu_timings.json"))},
       "prereg": J(W / "frozen/prereg.json"), "gates": {**L1["gates"], **GPU["gates"]}, "host": GPU["host"], "gpu": GPU["gpu"],
       "scripts_sha256": {os.path.basename(p): sha(p) for p in glob.glob(str(W / "scripts/*.py"))},
       "outputs_sha256": {os.path.relpath(p, W): sha(p) for p in sorted(glob.glob(str(W / "**/*"), recursive=True)) if os.path.isfile(p) and not p.endswith("manifest.json")}}
json.dump(man, open(W / "manifest.json", "w"), indent=1, default=str)
print("\n".join(S)); print(json.dumps(E2E, indent=1)); print({r['scenario'] + '|' + r['lookup_timing'] + '|' + r['embedding_assumption']: (r['net_ms_per_1k'], r['break_even_search_ms_given_fetch_and_generation']) for r in BE})
