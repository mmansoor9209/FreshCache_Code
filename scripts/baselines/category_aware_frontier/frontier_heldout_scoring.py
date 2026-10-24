"""CategoryAware-style comparator: our reimplementation of the per-category similarity threshold and TTL policy of
category-aware semantic caching (Wang et al., 2025), not the complete published system; it omits quotas, the storage tier,
indexing and load adaptation.

CategoryAware frontier, stage 4: held-out quality audit, evidence sufficiency, paired tests (pre-specified Holm family), decision table, outputs, report.
Decision rule (fixed before scoring): FreshCache has a defensible quality-constrained advantage over the selected CategoryAware configuration only if,
in the pre-specified family, a Holm-adjusted exact McNemar test shows FreshCache significantly better on WAI (fewer) or on evidence sufficiency (more)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, hashlib, json, os, pickle, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/category_aware_frontier"))
import frontier_helpers as X
ST = pickle.load(open(X.O / "work/stage3.pkl", "rb")); M = json.load(open(X.O / "frontier_selection_manifest.json")); OPS = ST["ops"]
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]; SQ = {d["query_id"]: d for d in SAMPLE}; IDS = sorted(SQ)
L2_REQUESTS = collections.defaultdict(dict)
for r in csv.DictReader(open(X.L2_POINTS_DIR / "l2_threshold_request_level.csv", encoding="utf-8")): L2_REQUESTS[r["point"]][r["request_id"]] = r
FRESH = {q: L2_REQUESTS["theta_0.75"][q]["fresh_label"] for q in IDS}; FCN = sum(v == "CORRECT" for v in FRESH.values())
LAB = {}
for name, pt in (("FreshCache primary (theta_L2=0.75)", "theta_0.75"), ("FreshCache theta_L2=0.80 (sensitivity)", "theta_0.80"), ("FreshCache verifier gamma=0.55 (sensitivity)", "gamma_0.55")):
    LAB[name] = {q: {"answer": L2_REQUESTS[pt][q]["answer_label"], "suff": L2_REQUESTS[pt][q]["sufficiency"], "tier": L2_REQUESTS[pt][q]["tier"], "evidence_source": "frozen L2-sweep audit (served evidence)"} for q in IDS}
hl = X.Held(); SL = {}
def addsuff(tasks, l8, q7):
    for t in tasks:
        a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
        if a is not None and b is not None: SL.setdefault((t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()), (a, b))
jl = X.jl
addsuff(jl(X.OV / "audit_suff_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(X.OV / "audit_suff_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(X.OV / "audit_suff_qwen7b.jsonl")})
RU = json.load(open(X.CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
for r in jl(X.CRR / "gpu/out/suffv_llama8b.jsonl"): cl8[(r["query_id"], r["arm"])] = r["label"]
for r in jl(X.CRR / "gpu/out/suffv_qwen7b.jsonl"): cq7[(r["query_id"], r["arm"])] = r["label"]
addsuff(jl(X.CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7)
for D in (X.L2_POINTS_DIR, X.CALIBRATION_AUDIT_DIR, X.O):
    addsuff(jl(D / "gpu/in/suffv_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(D / "gpu/out/suffv_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(D / "gpu/out/suffv_qwen7b.jsonl")})
for name, A in ST["audit"].items():
    LAB[name] = {}
    for q in IDS:
        a = A[q]; l, ans, _ = hl.labels(SQ[q]["query"], SQ[q]["gold"], a["key"])
        if a["evidence_sha"] is None: s = "EVIDENCE_UNAVAILABLE"
        else:
            x = SL.get((q, a["evidence_sha"])); s = "UNRESOLVED" if x is None else (x[0] if x[0] == x[1] else "JUDGE_DISAGREEMENT")
        LAB[name][q] = {"answer": l or "UNRESOLVED", "suff": s, "tier": a["tier"], "evidence_source": a["evidence_source"]}
_LP = LAB["FreshCache primary (theta_L2=0.75)"]; _SP = collections.Counter(v["suff"] for v in _LP.values())
G = {"fresh-correct (run-time reference)": REF.check("heldout_audit/fresh_correct", FCN), "FreshCache primary correct / WAI / sufficiency (run-time reference)": REF.check("heldout_audit/FreshCache_primary/CORRECT", sum(v["answer"] == "CORRECT" for v in _LP.values())) and REF.check("heldout_audit/FreshCache_primary/WAI", sum(1 for q in IDS if FRESH[q] == "CORRECT" and _LP[q]["answer"] == "WRONG")) and set(_SP) <= {"SUFFICIENT", "INSUFFICIENT", "JUDGE_DISAGREEMENT"} and REF.check("heldout_audit/FreshCache_primary/sufficiency", {k: _SP[k] for k in ("SUFFICIENT", "INSUFFICIENT", "JUDGE_DISAGREEMENT")}),
     "no unresolved answer or sufficiency label": all(v["answer"] != "UNRESOLVED" and v["suff"] != "UNRESOLVED" for L_ in LAB.values() for v in L_.values())}
BASELINE_POINTS = {r["arm"]: r for r in csv.DictReader(open(X.BASELINE_SEL_DIR / "baseline_selection_points.csv"))}
CAO = "CategoryAware-style (original)"
G["original CategoryAware 400 audit equals baseline-selection (CORRECT, WAI)"] = (sum(v["answer"] == "CORRECT" for v in LAB[CAO].values()), sum(1 for q in IDS if FRESH[q] == "CORRECT" and LAB[CAO][q]["answer"] == "WRONG")) == (int(BASELINE_POINTS["CategoryAware_style"]["audit_CORRECT"]), int(BASELINE_POINTS["CategoryAware_style"]["audit_WAI"]))
for k, v in G.items(): X.log("GATE", k, v)
if not all(G.values()): X.log("STOP: scoring gate failed"); sys.exit(1)
wa = lambda L_, q: FRESH[q] == "CORRECT" and L_[q]["answer"] == "WRONG"
def qual(L_):
    c = collections.Counter(v["answer"] for v in L_.values()); w = sum(1 for q in IDS if wa(L_, q)); s = collections.Counter(v["suff"] for v in L_.values())
    ev = s["SUFFICIENT"] + s["INSUFFICIENT"]
    return {"N": len(IDS), "correct": c["CORRECT"], "correct_pct": round(100 * c["CORRECT"] / len(IDS), 2), "abstain": c["ABSTAIN"], "WAI": w, "WAI_pct_of_audit": round(100 * w / len(IDS), 2), "C_WAI_pct": round(100 * w / FCN, 3),
            "sufficient": s["SUFFICIENT"], "insufficient": s["INSUFFICIENT"], "unresolved_judge_disagreement": s["JUDGE_DISAGREEMENT"] + s["UNRESOLVED"], "evidence_unavailable": s["EVIDENCE_UNAVAILABLE"],
            "evidence_evaluable_N": ev, "sufficient_of_evaluable_pct": round(100 * s["SUFFICIENT"] / ev, 2) if ev else None}
POL = ["FreshCache primary (theta_L2=0.75)", CAO] + [k for k in ST["audit"] if k != CAO] + ["FreshCache theta_L2=0.80 (sensitivity)", "FreshCache verifier gamma=0.55 (sensitivity)"]
Q = {p: qual(LAB[p]) for p in POL}
FCP = "FreshCache primary (theta_L2=0.75)"
def paired(other, base=None):
    a, b = LAB[base or FCP], LAB[other]
    w_fc = sum(1 for q in IDS if wa(a, q) and not wa(b, q)); w_ot = sum(1 for q in IDS if wa(b, q) and not wa(a, q))
    com = [q for q in IDS if a[q]["suff"] in ("SUFFICIENT", "INSUFFICIENT") and b[q]["suff"] in ("SUFFICIENT", "INSUFFICIENT")]
    s_fc = sum(1 for q in com if a[q]["suff"] == "SUFFICIENT" and b[q]["suff"] != "SUFFICIENT"); s_ot = sum(1 for q in com if b[q]["suff"] == "SUFFICIENT" and a[q]["suff"] != "SUFFICIENT")
    return {"WAI_FreshCache_only": w_fc, "WAI_other_only": w_ot, "WAI_mcnemar_p": X.mcnemar(w_fc, w_ot), "suff_common_evaluable_N": len(com), "suff_FreshCache_N": sum(a[q]["suff"] == "SUFFICIENT" for q in com),
            "suff_other_N": sum(b[q]["suff"] == "SUFFICIENT" for q in com), "suff_FreshCache_only": s_fc, "suff_other_only": s_ot, "suff_mcnemar_p": X.mcnemar(s_fc, s_ot)}
PAIR = {p: paired(p) for p in POL if p != FCP}
EXPL = {s_: paired(CAO, base=s_) for s_ in ("FreshCache theta_L2=0.80 (sensitivity)", "FreshCache verifier gamma=0.55 (sensitivity)")}   # exploratory, outside the pre-specified family
SELECTED = sorted({v for v in M["selected"].values() if v != "NONE"}); SELNAMES = [n for n, cid in ST["policies_ca"].items() if cid in SELECTED]
FAM = []
for nm in SELNAMES:
    FAM.append({"test": f"FreshCache vs {nm}: WAI", "p_raw": PAIR[nm]["WAI_mcnemar_p"], "FreshCache_better_direction": PAIR[nm]["WAI_FreshCache_only"] < PAIR[nm]["WAI_other_only"]})
    FAM.append({"test": f"FreshCache vs {nm}: evidence sufficiency", "p_raw": PAIR[nm]["suff_mcnemar_p"], "FreshCache_better_direction": PAIR[nm]["suff_FreshCache_only"] > PAIR[nm]["suff_other_only"]})
for f_, a_ in zip(FAM, X.holm([f["p_raw"] for f in FAM])): f_["p_holm"] = a_
ADV = any(f["p_holm"] < 0.05 and f["FreshCache_better_direction"] for f in FAM)
WORSE = any(f["p_holm"] < 0.05 and not f["FreshCache_better_direction"] for f in FAM)
# ---------------- outputs
with open(X.O / "frontier_heldout_results.csv", "w", newline="") as f:
    cols = ["method", "search_saved_pct", "search_per_1k", "fetch_per_1k", "gen_per_1k", "local_ms_per_request", "live_p50_ms_per_request", "lookup_ms_per_request", "note"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for p in POL: o = OPS[p]; w.writerow({"method": p, **{k: o[k] for k in cols[1:7]}, "lookup_ms_per_request": o["lookup"], "note": o["note"]})
with open(X.O / "frontier_quality_audit.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["request_id", "fresh_label"] + [x for p in POL for x in (f"{p} | tier", f"{p} | answer")])
    for q in IDS: w.writerow([q, FRESH[q]] + [x for p in POL for x in (LAB[p][q]["tier"], LAB[p][q]["answer"])])
with open(X.O / "frontier_evidence_sufficiency.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["request_id"] + [x for p in POL for x in (f"{p} | sufficiency", f"{p} | evidence_source")])
    for q in IDS: w.writerow([q] + [x for p in POL for x in (LAB[p][q]["suff"], LAB[p][q]["evidence_source"])])
BEF = json.load(open(X.O / "work/read_only_before.json")); UNCH = {p: X.fstat(inpath(p)) == v for p, v in BEF.items()}
fco, cao = OPS[FCP], OPS[CAO]; fq, cq = Q[FCP], Q[CAO]; pc = PAIR[CAO]
ANS = {"1_strict_feasible": f"Yes: {M['n_meeting_STRICT']} of 20 candidates meet C-WAI <= {round(M['constraints_pct']['STRICT'], 2)}% on validation; CA-Strict = {M['selected']['CA_Strict']}.",
       "2_relaxed_feasible": f"Yes: {M['n_meeting_RELAXED']} of 20 meet C-WAI <= {round(M['constraints_pct']['RELAXED'], 2)}%; CA-Relaxed = {M['selected']['CA_Relaxed']} ({'distinct' if M['CA_Relaxed_distinct_from_CA_Strict'] else 'the same configuration as CA-Strict'}).",
       "3_strict_cost": f"CategoryAware (CA-Strict) has the lower held-out modeled cost: {cao['local_ms_per_request']:.0f} vs {fco['local_ms_per_request']:.0f} ms/request (local) and {cao['live_p50_ms_per_request']:.0f} vs {fco['live_p50_ms_per_request']:.0f} ms/request (live medians).",
       "4_relaxed_cost": "Same as 3: CA-Relaxed is the same configuration." if not M["CA_Relaxed_distinct_from_CA_Strict"] else "see table",
       "5_evidence_at_comparable_quality": f"On the common evidence-evaluable population (N = {pc['suff_common_evaluable_N']}), FreshCache {pc['suff_FreshCache_N']} vs CategoryAware {pc['suff_other_N']} SUFFICIENT; discordant {pc['suff_FreshCache_only']} vs {pc['suff_other_only']}, exact McNemar p = {pc['suff_mcnemar_p']:.3g} (Holm {next(f['p_holm'] for f in FAM if 'sufficiency' in f['test']):.3g}). {'FreshCache higher, significantly' if (pc['suff_FreshCache_only'] > pc['suff_other_only'] and next(f['p_holm'] for f in FAM if 'sufficiency' in f['test']) < 0.05) else 'No significant difference'}.",
       "6_dominance": f"Yes on cost and savings: CategoryAware saves {cao['search_saved_pct']:.1f}% vs {fco['search_saved_pct']:.1f}% and is cheaper in both regimes. Held-out WAI is {cq['WAI']} vs {fq['WAI']} (C-WAI {cq['C_WAI_pct']:.1f}% vs {fq['C_WAI_pct']:.1f}%); paired WAI p = {pc['WAI_mcnemar_p']:.3g} (Holm {next(f['p_holm'] for f in FAM if 'WAI' in f['test']):.3g}) — {'significant' if next(f['p_holm'] for f in FAM if 'WAI' in f['test']) < 0.05 else 'not significant'}.",
       "7_operating_condition": ("Yes (confirmatory family)." if ADV else "Not confirmatorily: no test in the pre-specified family shows FreshCache primary significantly better than the quality-matched CategoryAware configuration.") + " Exploratory, outside the pre-specified family (raw p, no multiplicity correction): " + "; ".join(f"{k.split(' (')[0]} vs CategoryAware: WAI {Q[k]['WAI']} vs {Q[CAO]['WAI']} (discordant {v['WAI_FreshCache_only']}/{v['WAI_other_only']}, p = {v['WAI_mcnemar_p']:.3g}), sufficient {v['suff_FreshCache_N']} vs {v['suff_other_N']} of {v['suff_common_evaluable_N']} (discordant {v['suff_FreshCache_only']}/{v['suff_other_only']}, p = {v['suff_mcnemar_p']:.3g}), cost {OPS[k]['live_p50_ms_per_request']:.0f} vs {OPS[CAO]['live_p50_ms_per_request']:.0f} ms/request" for k, v in EXPL.items()) + ". These stricter FreshCache settings cost more than CategoryAware; any significant exploratory difference is hypothesis-generating only, because these comparisons were not pre-specified.",
       "8_statement": ("FreshCache shows a significant advantage in the pre-specified family." if ADV else "Stated plainly: CategoryAware-frontier provides no evidence of a quality-constrained cost/quality or evidence-quality advantage for FreshCache over CategoryAware-style; FreshCache's held-out WAI is numerically lower (" + f"{fq['WAI']} vs {cq['WAI']}" + ") but not significantly so, and CategoryAware is cheaper with higher savings.")}
SUM = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "sensitivity/frontier study added after the primary experiment; CategoryAware-frontier-selected configurations are not new primary scripts/baselines; selection uses validation only and was frozen before CategoryAware-frontier held-out scoring",
       "grid_sha256": json.load(open(X.O / "grid_frozen.json"))["sha256"], "selection_manifest_sha256": M["sha256"], "selection": M, "heldout_ops": OPS, "quality": Q, "paired_vs_FreshCache_primary": PAIR,
       "multiple_testing_family": FAM, "exploratory_sensitivity_vs_CategoryAware": EXPL, "freshcache_defensible_advantage": ADV, "freshcache_significantly_worse_in_family": WORSE, "answers": ANS, "gates": {**ST["gates"], **G},
       "read_only_files": {p: {**BEF[p], "unchanged": UNCH[p]} for p in BEF}, "all_read_only_unchanged": all(UNCH.values())}
json.dump(SUM, open(X.O / "frontier_summary.json", "w"), indent=1, default=str)
f2 = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"
L = []; P = L.append
P("# CategoryAware-frontier — CategoryAware-style quality-constrained frontier"); P("")
P("**This is a sensitivity/frontier study added after the primary experiment.**")
P("- CategoryAware-frontier-selected configurations are not new primary scripts/baselines, and they were not selected before the original held-out study.")
P("- CategoryAware-frontier selection uses validation only and was frozen before CategoryAware-frontier held-out scoring.")
P("- FreshCache is not retuned: the primary is θ_L2 = 0.75, with θ_L2 = 0.80 and verifier γ = 0.55 as sensitivities.")
P("- The reported CategoryAware-style point is unchanged. No Web or search calls were made.")
P(f"- Grid sha256 `{SUM['grid_sha256']}`; selection manifest sha256 `{M['sha256']}` (frozen {M['frozen_utc']}).")
P("- Gates: " + "; ".join(f"{k}: **{v}**" for k, v in SUM["gates"].items()) + ".")
D = json.load(open(X.O / "grid_frozen.json"))["design"]
P(""); P("## Step 1 — CategoryAware implementation (recovered)"); P("")
for k, v in D["categoryaware_recovered"].items(): P(f"- **{k}**: {json.dumps(v) if isinstance(v, dict) else v}")
P(f"- **Grid**: {D['existing_grid_reusable']}")
P(f"- Candidates: Δθ ∈ {D['grid']['dtheta']} × TTL multiplier ∈ {D['grid']['m_ttl']} ({D['grid']['n']} configurations), applied identically to every class, thresholds capped at 0.99. Frozen before any held-out CategoryAware-frontier run.")
P(""); P("## Step 2 — Validation scoring and selection (validation only)"); P("")
P(f"The validation quality set is the screened validation sample (fresh-correct denominator from the frozen fresh arm). Constraints: STRICT C-WAI ≤ {M['constraints_pct']['STRICT']}%, RELAXED ≤ {M['constraints_pct']['RELAXED']}%. Selection minimises validation modeled cost ({M['selection_cost_regime']}).")
P(""); P("| Config | Δθ | m_TTL | Saved % | Fetch/1k | Gen/1k | L1 hits | Cost ms/req (local / live) | Correct | WAI | C-WAI % | STRICT | RELAXED |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in csv.DictReader(open(X.O / "frontier_validation_grid.csv")):
    if r["config_id"] == "FreshCache_primary": P(f"| **FreshCache primary** | — | — | {r['search_saved_pct']} | {r['fetch_per_1k']} | {r['gen_per_1k']} | {r['l1_hits']} | {float(r['local_ms_per_request']):.0f} / {float(r['live_p50_ms_per_request']):.0f} | {r['CORRECT']} | {r['WAI']} | {r['C_WAI_pct']} | — | — |")
    else: P(f"| {r['config_id']} | {r['dtheta']} | {r['m_ttl']} | {r['search_saved_pct']} | {r['fetch_per_1k']} | {r['gen_per_1k']} | {r['l1_hits']} | {float(r['local_ms_per_request']):.0f} / {float(r['live_p50_ms_per_request']):.0f} | {r['CORRECT']} | {r['WAI']} | {r['C_WAI_pct']} | {r['meets_STRICT']} | {r['meets_RELAXED']} |")
P("")
P(f"**Selected** (frozen before held-out): CA-Strict = `{M['selected']['CA_Strict']}`, CA-Relaxed = `{M['selected']['CA_Relaxed']}` (distinct: {M['CA_Relaxed_distinct_from_CA_Strict']}). Both are the original CategoryAware-style configuration, which is the cheapest candidate and meets both constraints. Under the local regime the selection would be {json.dumps(M['selection_under_local_regime_for_reference'])}.")
P("")
P(f"Dominance check (validation cost ≤ FreshCache **and** validation C-WAI ≤ FreshCache): {len(M['dominance_check']['live_p50'])} candidates in the live regime ({', '.join(M['dominance_check']['live_p50'])}) and {len(M['dominance_check']['local'])} in the local regime.")
P(""); P("## Step 3–4 — Held-out decision table (replay; audit)"); P("")
P("| Method | Saved | Cost ms/req (local / live) | Correct | WAI | C-WAI | Sufficient/N | Evidence unavailable |"); P("|---|---|---|---|---|---|---|---|")
for p in POL:
    o, q = OPS[p], Q[p]
    P(f"| {p} | {o['search_saved_pct']:.1f}% | {o['local_ms_per_request']:.0f} / {o['live_p50_ms_per_request']:.0f}{' †' if 'gamma' in p else ''} | {q['correct']}/{len(IDS)} ({q['correct_pct']:.1f}%) | {q['WAI']} ({q['WAI_pct_of_audit']:.1f}%) | {q['C_WAI_pct']:.1f}% | {q['sufficient']}/{len(IDS)} ({q['insufficient']} insufficient, {q['unresolved_judge_disagreement']} unresolved) | {q['evidence_unavailable']} |")
P("")
P("- † The γ = 0.55 cost excludes the verifier's scoring cost, which is unmeasured.")
P("- Search/1k and fetch/1k are in `frontier_heldout_results.csv`.")
P("- The CA-Strict and CA-Relaxed rows are identical to the original CategoryAware-style row, because the selection returned that configuration.")
P("")
P("**Evidence provenance:**")
P("- CategoryAware: a cached-answer hit uses the stored evidence of the entry that created the answer (pages at the entry's creation time, from the local snapshot store); a generated answer uses the evidence given to generation.")
P("- FreshCache arms: the frozen L2-sweep served-evidence labels.")
P("- No evidence was reconstructed from current Web pages.")
P(""); P("**Paired tests vs FreshCache primary** (exact McNemar):"); P("")
P("| Comparison | WAI: FreshCache-only / other-only (p) | Common evidence-evaluable N | Sufficient: FreshCache / other | Discordant FreshCache-only / other-only (p) |"); P("|---|---|---|---|---|")
for p, x in PAIR.items(): P(f"| {p} | {x['WAI_FreshCache_only']} / {x['WAI_other_only']} ({x['WAI_mcnemar_p']:.3g}) | {x['suff_common_evaluable_N']} | {x['suff_FreshCache_N']} / {x['suff_other_N']} | {x['suff_FreshCache_only']} / {x['suff_other_only']} ({x['suff_mcnemar_p']:.3g}) |")
P(""); P("**Exploratory (outside the pre-specified family; raw p): FreshCache sensitivities vs CategoryAware-style**"); P("")
P("| Comparison | WAI: FreshCache-only / CategoryAware-only (p) | Common evaluable N | Sufficient: FreshCache / CategoryAware | Discordant (p) |"); P("|---|---|---|---|---|")
for k, x in EXPL.items(): P(f"| {k} vs CategoryAware | {x['WAI_FreshCache_only']} / {x['WAI_other_only']} ({x['WAI_mcnemar_p']:.3g}) | {x['suff_common_evaluable_N']} | {x['suff_FreshCache_N']} / {x['suff_other_N']} | {x['suff_FreshCache_only']} / {x['suff_other_only']} ({x['suff_mcnemar_p']:.3g}) |")
P(""); P("**Pre-specified CategoryAware-frontier multiple-testing family** (FreshCache vs selected CategoryAware; Holm):"); P("")
P("| Test | Raw p | Holm p | Direction favours FreshCache |"); P("|---|---|---|---|")
for f_ in FAM: P(f"| {f_['test']} | {f_['p_raw']:.3g} | {f_['p_holm']:.3g} | {f_['FreshCache_better_direction']} |")
P(""); P("## Step 5 — Answers"); P("")
for k, v in ANS.items(): P(f"{k.split('_')[0]}. {v}")
INTERP = (f"In a validation-only sensitivity study frozen before held-out scoring, the cheapest of 20 stricter-or-equal CategoryAware-style configurations was the original one, which already met the strict quality constraint (validation C-WAI {next(r for r in csv.DictReader(open(X.O / 'frontier_validation_grid.csv')) if r['config_id'] == 'ca_d+0.000_m1.00')['C_WAI_pct']}% ≤ {round(M['constraints_pct']['STRICT'], 2)}%), so CA-Strict and CA-Relaxed coincide with the reported comparator. "
          f"On the held-out replay it saves more searches than FreshCache ({cao['search_saved_pct']:.1f}% vs {fco['search_saved_pct']:.1f}%) at lower modeled cost ({cao['live_p50_ms_per_request']:.0f} vs {fco['live_p50_ms_per_request']:.0f} ms/request with retained live search/fetch medians), because its answer hits also avoid generation. "
          f"On the audit FreshCache has numerically fewer wrong-after-correct answers ({fq['WAI']} vs {cq['WAI']}; C-WAI {fq['C_WAI_pct']:.1f}% vs {cq['C_WAI_pct']:.1f}%) and {'more' if pc['suff_FreshCache_N'] > pc['suff_other_N'] else 'not more'} sufficient served evidence ({pc['suff_FreshCache_N']} vs {pc['suff_other_N']} of {pc['suff_common_evaluable_N']} commonly evaluable cases), but neither difference is significant after Holm adjustment (WAI p = {FAM[0]['p_holm']:.2f}; sufficiency p = {FAM[1]['p_holm']:.2f}). "
          f"CategoryAware-frontier therefore provides no confirmatory support for a quality-constrained cost or evidence-quality advantage of FreshCache primary over CategoryAware-style; FreshCache's only measured operational edge over this comparator is fewer page fetches ({fco['fetch_per_1k']:.0f} vs {cao['fetch_per_1k']:.0f} per 1,000), which its higher generation count outweighs in modeled cost, and the stricter FreshCache sensitivities (θ_L2 = 0.80, γ = 0.55) show significantly more sufficient served evidence than CategoryAware (unadjusted p = {EXPL['FreshCache theta_L2=0.80 (sensitivity)']['suff_mcnemar_p']:.2g} and {EXPL['FreshCache verifier gamma=0.55 (sensitivity)']['suff_mcnemar_p']:.2g}) with numerically but not significantly lower WAI (p = {EXPL['FreshCache theta_L2=0.80 (sensitivity)']['WAI_mcnemar_p']:.2f} and {EXPL['FreshCache verifier gamma=0.55 (sensitivity)']['WAI_mcnemar_p']:.2f}), at higher modeled cost; because these comparisons were not pre-specified, this evidence-support signal is hypothesis-generating only.")
P(""); P("## Manuscript-ready interpretation"); P(""); P(INTERP)
P(""); P("## Integrity"); P("")
P(f"- All pre-existing files read are unchanged: **{SUM['all_read_only_unchanged']}** ({len(BEF)} files; sha256, size and mtime compared before and after).")
P(f"- New outputs exist only in `outputs/categoryaware_frontier/`. New GPU outputs: {sum(1 for _ in open(X.O / 'gpu/out/answers_llama3b.jsonl'))} validation answers, {sum(1 for _ in open(X.O / 'gpu/out/judge_llama8b.jsonl'))} judge pairs, {sum(1 for _ in open(X.O / 'gpu/out/suffv_llama8b.jsonl'))} sufficiency pairs.")
P(""); P("| Pre-existing file | sha256 | Unchanged |"); P("|---|---|---|")
for p_, v in SUM["read_only_files"].items(): P(f"| `{p_}` | `{v['sha256'][:16]}…` | {v['unchanged']} |")
SUM["manuscript_interpretation"] = INTERP; json.dump(SUM, open(X.O / "frontier_summary.json", "w"), indent=1, default=str)
open(X.O / "FRONTIER_REPORT.md", "w").write("\n".join(L) + "\n"); X.log(f"CategoryAware-frontier complete; defensible FreshCache advantage: {ADV}; read-only unchanged: {SUM['all_read_only_unchanged']}")
print("\n".join(L[L.index("## Step 3–4 — Held-out decision table (replay; audit)"):]))
