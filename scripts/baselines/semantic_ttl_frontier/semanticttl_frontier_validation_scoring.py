"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

SemanticTTL frontier, validation scoring (validation split only): score the 20 configurations on the screened
validation requests, gate against the fixed FreshCache and SemanticTTL (theta=0.60, kappa=1/2) validation results in
outputs/replay/results/selection_and_8b.json, apply the selection rules fixed in
outputs/semanticttl_frontier/frozen_selection/selection_rules.json, and write frozen_selection/selection.json (read-only).
No held-out file is opened by this script."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, datetime, hashlib, json, os, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_frontier"))
import semanticttl_frontier_helpers as X, semanticttl_frontier_labels as L

now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
SEL = X.STTL_FRONTIER_DIR / "frozen_selection/selection.json"
if SEL.exists():
    sys.exit("STOP: selection already frozen; refusing to overwrite")
RULES = json.load(open(X.STTL_FRONTIER_DIR / "frozen_selection/selection_rules.json"))
chk = hashlib.sha256(json.dumps({k: v for k, v in RULES.items() if k not in ("written_utc", "sha256")}, sort_keys=True).encode()).hexdigest()
assert chk == RULES["sha256"], "selection rules changed after they were written"
lab = L.Val(); K = json.load(open(X.STTL_FRONTIER_DIR / "validation/quality_keys.json")); OPS = json.load(open(X.STTL_FRONTIER_DIR / "validation/replay_metrics.json"))
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
EV = K["screened_set"]; FK = K["frozen_fresh_and_freshcache_keys"]
ARMS = ["fc", "sttl_t0.60_k1_2"] + [c["id"] for c in X.CONFIGS]
def key(arm, q): return FK[q][arm] if arm in ("fresh", "fc", "sttl_t0.60_k1_2") else K["configs"][arm][q]["key"]
rows, judge_dis = [], collections.Counter()
for q in EV:
    d = QS[q]; r = {"query_id": q, "query": d["query"], "gold": d["gold"], "freshness_class": d["fc"]}
    for arm in ["fresh"] + ARMS:
        lb, ans, jl_ = lab.labels(d["query"], d["gold"], key(arm, q))
        r[arm] = lb if lb is not None else "UNRESOLVED"; r[f"{arm}_answer"] = ans; r[f"{arm}_key"] = "|".join(key(arm, q))
        if arm in K["configs"]: r[f"{arm}_tier"] = K["configs"][arm][q]["tier"]
        if isinstance(jl_, tuple) and len(jl_) == 2 and jl_[0] != jl_[1]: judge_dis[arm] += 1
    rows.append(r)
fcn = sum(1 for r in rows if r["fresh"] == "CORRECT")
def stats(arm):
    c = collections.Counter(r[arm] for r in rows); fcr = [r for r in rows if r["fresh"] == "CORRECT"]
    wai = sum(1 for r in fcr if r[arm] == "WRONG"); ab = sum(1 for r in fcr if r[arm] == "ABSTAIN")
    s = {"N": len(rows), "CORRECT": c["CORRECT"], "WRONG": c["WRONG"], "ABSTAIN": c["ABSTAIN"], "UNRESOLVED": c["UNRESOLVED"],
         "accuracy_pct": round(100 * c["CORRECT"] / len(rows), 3), "fresh_correct": fcn, "WAI": wai, "C_WAI_pct": round(100 * wai / fcn, 4),
         "fresh_correct_to_wrong": wai, "fresh_correct_to_abstain": ab, "judge_disagreements_scored_WRONG": judge_dis[arm]}
    if arm != "fc":
        w = lambda r, a: r["fresh"] == "CORRECT" and r[a] == "WRONG"
        b = sum(1 for r in rows if w(r, arm) and not w(r, "fc")); cc = sum(1 for r in rows if w(r, "fc") and not w(r, arm)); both = sum(1 for r in rows if w(r, "fc") and w(r, arm))
        s["paired_WAI_vs_FreshCache"] = {"config_only": b, "freshcache_only": cc, "both": both, "neither": len(rows) - b - cc - both, "exact_mcnemar_p": X.mcnemar(b, cc)}
    return s
ST = {a: stats(a) for a in ARMS}; ST["fresh"] = {"N": len(rows), "CORRECT": fcn}
# ---------------- gates against the fixed validation quality selection
FROZ = json.load(open(X.CRR / "results/selection_and_8b.json"))["quality_selection_reported"]
f060 = next(c for c in FROZ["candidates"] if c["arm"] == "sttl_t0.60_k1_2")
G = {"n_equals_quality_selection": len(rows) == FROZ["evidence_valid"], "fresh_correct_equals_quality_selection": fcn == FROZ["fresh_correct"],
     "freshcache_wai_correct_equal_quality_selection": [ST["fc"]["WAI"], ST["fc"]["CORRECT"]] == [FROZ["freshcache"]["wai"], FROZ["freshcache"]["correct"]],
     "sttl_060_wai_correct_reproduced": [ST["sttl_t0.60_k1_2"]["WAI"], ST["sttl_t0.60_k1_2"]["CORRECT"]] == [f060["wai"], f060["correct"]],
     "no_unresolved_in_new_configs": all(ST[c["id"]]["UNRESOLVED"] == 0 for c in X.CONFIGS)}
if not all(G.values()):
    json.dump({"gates": G, "stats": ST}, open(X.STTL_FRONTIER_DIR / "logs/validation_selection_STOPPED_gates.json", "w"), indent=1); sys.exit(f"STOP: validation scoring gate failed {G}")
# ---------------- selection (rules fixed by semanticttl_frontier_validation_replay.py)
FC_SAVED = OPS["FreshCache"]["search_saved_pct"]; LIM_WAI = ST["fc"]["WAI"]
cand = [{"id": c["id"], "theta": c["theta"], "kappa": c["kappa"], "search_saved_pct": OPS[c["id"]]["search_saved_pct"], "WAI": ST[c["id"]]["WAI"],
         "C_WAI_pct": ST[c["id"]]["C_WAI_pct"], "CORRECT": ST[c["id"]]["CORRECT"], "order": i} for i, c in enumerate(X.CONFIGS)]
qa = [c for c in cand if c["WAI"] <= LIM_WAI]
A = sorted(qa, key=lambda c: (-c["search_saved_pct"], c["WAI"], -c["CORRECT"], c["order"]))[0] if qa else None
qa_lit = [c for c in cand if c["C_WAI_pct"] <= round(ST["fc"]["C_WAI_pct"], 2)]   # literal 2-decimal reading of the FreshCache limit
A_lit = sorted(qa_lit, key=lambda c: (-c["search_saved_pct"], c["WAI"], -c["CORRECT"], c["order"]))[0] if qa_lit else None
B = sorted(cand, key=lambda c: (abs(c["search_saved_pct"] - FC_SAVED), c["WAI"], -c["search_saved_pct"], c["order"]))[0]
selection = {"frozen_utc": now(), "heldout_consulted": False, "selection_rules_sha256": RULES["sha256"],
             "freshcache_validation": {"search_saved_pct": FC_SAVED, "WAI": ST["fc"]["WAI"], "fresh_correct": fcn, "C_WAI_pct": ST["fc"]["C_WAI_pct"]},
             "A_quality_matched": ({k: A[k] for k in ("id", "theta", "kappa", "search_saved_pct", "WAI", "C_WAI_pct", "CORRECT")} if A else "NONE"),
             "A_qualifying_configs": [c["id"] for c in qa],
             "A_literal_7.59pct_reading": ({"id": A_lit["id"]} if A_lit else "NONE"), "A_literal_reading_differs": (A_lit or {}).get("id") != (A or {}).get("id"),
             "B_cost_matched": {**{k: B[k] for k in ("id", "theta", "kappa", "search_saved_pct", "WAI", "C_WAI_pct", "CORRECT")},
                                "abs_savings_gap_pp": round(abs(B["search_saved_pct"] - FC_SAVED), 4)}}
selection["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in selection.items() if k != "frozen_utc"}, sort_keys=True).encode()).hexdigest()
# ---------------- outputs
json.dump({"gates": G, "stats": ST}, open(X.STTL_FRONTIER_DIR / "validation/quality_stats.json", "w"), indent=1)
with open(X.STTL_FRONTIER_DIR / "validation/request_level_quality.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tab = []
for name in ["FreshCache", "sttl_t0.60_k0.5_reference"] + [c["id"] for c in X.CONFIGS]:
    o = OPS[name]; s = ST["fc" if name == "FreshCache" else "sttl_t0.60_k1_2" if name.startswith("sttl_t0.60") else name]
    tab.append({"config": name, "theta": o.get("theta", 0.60 if name.startswith("sttl_t0.60") else ""), "kappa": o.get("kappa", 0.5 if name.startswith("sttl_t0.60") else ""),
                "search_saved_pct": o["search_saved_pct"], "searches_per_1k": o["searches_per_1k"], "fetch_per_1k": o["fetch_per_1k"], "gen_per_1k": o["gen_per_1k"],
                "l1_hits": o["l1_hits"], "l2_hits": o["l2_hits"], "l3_hits": o["l3_hits"], "drift_pct": o["drift_pct"], "drift_changed": o["changed"],
                "drift_determinable": o["determinable"], "support": o["support"], "coverage_pct": o["coverage_pct"],
                **{k: s[k] for k in ("N", "CORRECT", "WRONG", "ABSTAIN", "UNRESOLVED", "accuracy_pct", "fresh_correct", "WAI", "C_WAI_pct", "fresh_correct_to_wrong", "fresh_correct_to_abstain")},
                **({f"paired_{k}": v for k, v in s["paired_WAI_vs_FreshCache"].items()} if "paired_WAI_vs_FreshCache" in s else {})})
with open(X.STTL_FRONTIER_DIR / "results/validation_table.csv", "w", newline="") as f:
    cols = sorted({k for r in tab for k in r}, key=lambda k: list(tab[-1]).index(k) if k in tab[-1] else 99)
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(tab)
json.dump(selection, open(SEL, "w"), indent=1)
os.chmod(SEL, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
with open(X.STTL_FRONTIER_DIR / "logs/chronology.jsonl", "a") as f:
    f.write(json.dumps({"event": "selection_frozen", "utc": selection["frozen_utc"], "sha256": selection["sha256"], "heldout_files_opened_so_far": 0}) + "\n")
print(json.dumps(selection, indent=1))
