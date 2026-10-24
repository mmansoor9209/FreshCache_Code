"""SemanticTTL frontier, held-out scoring: scores the existing held-out audit (outputs/general_audit/sample.json) for fresh,
FreshCache, the fixed SemanticTTL (0.60, 1/2) configuration and the frozen quality- and cost-matched selections, with exact McNemar tests
against FreshCache and against the fixed configuration. Reads heldout/audit_keys.json and heldout/replay_metrics.json
(semanticttl_frontier_heldout_replay.py) and the judged answers; writes outputs/semanticttl_frontier/heldout/quality_stats.json,
heldout/request_level_quality.csv and results/heldout_table.csv. Stops (logs/heldout_scoring_STOPPED_gates.json) if a reproduction gate fails."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, json, os, sys; OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_frontier"))
import semanticttl_frontier_helpers as X, semanticttl_frontier_labels as L
for _d in ("heldout", "logs", "results"): (X.STTL_FRONTIER_DIR / _d).mkdir(parents=True, exist_ok=True)
lab = L.Held(); K = json.load(open(X.STTL_FRONTIER_DIR / "heldout/audit_keys.json")); OPS = json.load(open(X.STTL_FRONTIER_DIR / "heldout/replay_metrics.json"))["metrics"]
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
ARMS = ["freshcache", "sttl_t0.60_k0.5_fixed"] + list(K["picked"])
rows = []
for d in SAMPLE:
    q = d["query_id"]; r = {"query_id": q, "query": d["query"], "gold": d["gold"], "freshness_class": d.get("fc") or d.get("freshness_class")}
    for arm in ["fresh"] + ARMS:
        lb, ans, _ = lab.labels(d["query"], d["gold"], K["keys"][q][arm])
        r[arm] = lb if lb is not None else "UNRESOLVED"; r[f"{arm}_answer"] = ans; r[f"{arm}_key"] = "|".join(K["keys"][q][arm])
        if arm != "fresh": r[f"{arm}_tier"] = K["tiers"][q][arm]
    rows.append(r)
fc = [r for r in rows if r["fresh"] == "CORRECT"]; N = len(rows)
wai = lambda r, a: r["fresh"] == "CORRECT" and r[a] == "WRONG"
def paired(a, b):
    x = sum(1 for r in rows if wai(r, a) and not wai(r, b)); y = sum(1 for r in rows if wai(r, b) and not wai(r, a)); z = sum(1 for r in rows if wai(r, a) and wai(r, b))
    cx = sum(1 for r in rows if r[a] == "CORRECT" and r[b] != "CORRECT"); cy = sum(1 for r in rows if r[b] == "CORRECT" and r[a] != "CORRECT")
    return {"wai_only_" + a: x, "wai_only_" + b: y, "wai_both": z, "wai_neither": N - x - y - z, "wai_exact_mcnemar_p": X.mcnemar(x, y),
            "correct_only_" + a: cx, "correct_only_" + b: cy, "correct_exact_mcnemar_p": X.mcnemar(cx, cy)}
ST = {"fresh": {"N": N, "CORRECT": len(fc), "counts": dict(collections.Counter(r["fresh"] for r in rows))}}
for a in ARMS:
    c = collections.Counter(r[a] for r in rows); w = sum(1 for r in fc if r[a] == "WRONG")
    ST[a] = {"N": N, "CORRECT": c["CORRECT"], "WRONG": c["WRONG"], "ABSTAIN": c["ABSTAIN"], "UNRESOLVED": c["UNRESOLVED"], "accuracy_pct": round(100 * c["CORRECT"] / N, 3),
             "fresh_correct": len(fc), "WAI": w, "WAI_pct": round(100 * w / N, 3), "C_WAI_pct": round(100 * w / len(fc), 4),
             "fresh_correct_to_abstain": sum(1 for r in fc if r[a] == "ABSTAIN")}
    if a != "freshcache": ST[a]["paired_vs_freshcache"] = paired(a, "freshcache")
    if a in K["picked"]: ST[a]["paired_vs_fixed_060"] = paired(a, "sttl_t0.60_k0.5_fixed")
G = {"fresh_correct": REF.check("heldout_audit/fresh_correct", len(fc)), "freshcache_correct_wai": REF.check("heldout_audit/FreshCache_primary/CORRECT", ST["freshcache"]["CORRECT"]) and REF.check("heldout_audit/FreshCache_primary/WAI", ST["freshcache"]["WAI"]),
     "fixed060_correct_wai": REF.check("heldout_audit/SemanticTTL_0.60_k0.5_fixed/CORRECT", ST["sttl_t0.60_k0.5_fixed"]["CORRECT"]) and REF.check("heldout_audit/SemanticTTL_0.60_k0.5_fixed/WAI", ST["sttl_t0.60_k0.5_fixed"]["WAI"]),
     "fixed060_vs_fc_paired": REF.check("heldout_audit/SemanticTTL_0.60_k0.5_fixed/paired_vs_FreshCache_WAI", [ST["sttl_t0.60_k0.5_fixed"]["paired_vs_freshcache"][k] for k in ("wai_only_sttl_t0.60_k0.5_fixed", "wai_only_freshcache", "wai_both")]),
     "no_unresolved": all(ST[a]["UNRESOLVED"] == 0 for a in ARMS) and ST["fresh"]["counts"].get("UNRESOLVED", 0) == 0}
if not all(G.values()):
    json.dump({"gates": G, "stats": ST}, open(X.STTL_FRONTIER_DIR / "logs/heldout_scoring_STOPPED_gates.json", "w"), indent=1); sys.exit(f"STOP: held-out gate failed {G}")
json.dump({"gates": G, "stats": ST}, open(X.STTL_FRONTIER_DIR / "heldout/quality_stats.json", "w"), indent=1)
with open(X.STTL_FRONTIER_DIR / "heldout/request_level_quality.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
opname = {"freshcache": "FreshCache", "sttl_t0.60_k0.5_fixed": "sttl_t0.60_k0.5_fixed"}
tab = []
for a in ARMS:
    o = OPS[opname.get(a) or next(k for k in OPS if k.startswith(a + ":"))]; s = ST[a]
    pf = s.get("paired_vs_freshcache", {}); pr = s.get("paired_vs_fixed_060", {})
    tab.append({"arm": a if a in opname else f"{a}:{K['picked'][a]['id']}", "search_saved_pct": o["search_saved_pct"], "searches_per_1k": o["searches_per_1k"],
                "fetch_per_1k": o["fetch_per_1k"], "gen_per_1k": o["gen_per_1k"], "l1_hits": o["l1_hits"], "drift_pct": o["drift_pct"], "drift_determinable": o["determinable"],
                "coverage_pct": o["coverage_pct"], **{k: s[k] for k in ("N", "fresh_correct", "CORRECT", "WRONG", "ABSTAIN", "UNRESOLVED", "accuracy_pct", "WAI", "C_WAI_pct", "fresh_correct_to_abstain")},
                "vsFC_wai_config_only": pf.get(f"wai_only_{a}"), "vsFC_wai_freshcache_only": pf.get("wai_only_freshcache"), "vsFC_wai_both": pf.get("wai_both"), "vsFC_wai_p": pf.get("wai_exact_mcnemar_p"),
                "vs060_wai_config_only": pr.get(f"wai_only_{a}"), "vs060_wai_fixed060_only": pr.get("wai_only_sttl_t0.60_k0.5_fixed"), "vs060_wai_both": pr.get("wai_both"), "vs060_wai_p": pr.get("wai_exact_mcnemar_p")})
with open(X.STTL_FRONTIER_DIR / "results/heldout_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tab[0])); w.writeheader(); w.writerows(tab)
print("GATES", G)
for t in tab: print(t)
