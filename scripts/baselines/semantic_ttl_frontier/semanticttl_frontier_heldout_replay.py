"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

Held-out stage of the SemanticTTL frontier baseline. Runs only after the validation selection has been frozen.

Requires outputs/semanticttl_frontier/frozen_selection/selection.json to be read-only and its sha256 to verify.
Replays the held-out stream for the frozen quality-matched selection (if one exists), the frozen
cost-matched selection, FreshCache, and the fixed SemanticTTL(theta=0.60, kappa=1/2) configuration, then builds answer
keys for the existing held-out audit (outputs/general_audit/sample.json; no resampling).

Gates (the script stops and writes logs/heldout_replay_STOPPED_gates.json if any fails): FreshCache held-out search savings equal
the run-time reference (reference_values.json); the fixed (0.60, 1/2) replay equals outputs/replay/results/extras_reported.json; the audit sample has 400
requests matching outputs/replay/results/audit_keys_reported.json; the fixed configuration reproduces those held-out
answer keys; engine and transcription replays agree on every tier.

Outputs under outputs/semanticttl_frontier/: heldout/replay_metrics.json, heldout/audit_keys.json,
heldout/request_level_replay.csv.gz, and gpu/in/ho_gen_tasks.jsonl (generation tasks for keys with no existing answer)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, datetime, gzip, hashlib, json, os, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_frontier"))
import semanticttl_frontier_helpers as X, semanticttl_frontier_labels as L

now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
SELP = X.STTL_FRONTIER_DIR / "frozen_selection/selection.json"
assert SELP.exists() and not (os.stat(SELP).st_mode & stat.S_IWUSR), "selection must be frozen (read-only) before any held-out work"
SEL = json.load(open(SELP))
assert hashlib.sha256(json.dumps({k: v for k, v in SEL.items() if k not in ("frozen_utc", "sha256")}, sort_keys=True).encode()).hexdigest() == SEL["sha256"]
with open(X.STTL_FRONTIER_DIR / "logs/chronology.jsonl", "a") as f:
    f.write(json.dumps({"event": "heldout_stage_started", "utc": now(), "selection_sha256": SEL["sha256"]}) + "\n")
PICK = {}
if SEL["A_quality_matched"] != "NONE": PICK["A_quality_matched"] = SEL["A_quality_matched"]
PICK["B_cost_matched"] = SEL["B_cost_matched"]

E = X.engine(); p2 = E["p2"]; Q2 = X.q2(); h, ctx_for = p2.h, p2.ctx_for
H = X.C.streams(E)["heldout"]; assert DS.check("heldout_requests", len(H)), len(H)
S = E["support_of"](H, E["rounds"])
def opm(variant, theta=None, kappa=None):
    if theta is None: row, per, tr = X.C.metrics(E, H, variant, S)
    else:
        with X.patched(theta, kappa): row, per, tr = X.C.metrics(E, H, variant, S)
    row["search_calls"] = sum(1 for q in tr if tr[q]["tier"] in ("miss", "REAL_TIME_BYPASS")); row["searches_per_1k"] = round(1000 * row["search_calls"] / len(H), 1)
    row["determinable"] = row["changed"] + row["unchanged"]; return row, per, tr
def srows(theta, kappa):
    with X.patched(theta, kappa) as pt: return Q2.sttl_replay(H, theta, kappa, pt.orig_ttl)
def key_for(p, query):
    ctx = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", query); return ["gen", h(ctx), h(gq)], ctx, gq
mp = {"L1": "L1", "miss": "miss", "REAL_TIME_BYPASS": "REAL_TIME"}
EX = json.load(open(X.CRR / "results/extras_reported.json"))
G, OPS, reqrows = {}, {}, []
fc_row, fc_per, fc_tr = opm("FreshCache"); OPS["FreshCache"] = fc_row
import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF); G["freshcache_heldout_saved_equals_reference"] = REF.check("reported_engine/heldout/FreshCache/search_saved_pct", fc_row["search_saved_pct"])
r060, p060, t060 = opm("SemanticTTL", 0.60, 0.5); OPS["sttl_t0.60_k0.5_fixed"] = r060
G["sttl_060_heldout_replay_equals_frozen_extras"] = {k: r060[k] for k in ("search_saved_pct", "fetches", "generations", "changed", "unchanged")} == \
    {k: EX["heldout_semanticttl"]["quality_selected_0.60_1/2"]["replay"][k] for k in ("search_saved_pct", "fetches", "generations", "changed", "unchanged")}
m060, rw060 = srows(0.60, 0.5)
AK = json.load(open(X.CRR / "results/audit_keys_reported.json"))["keys"]["heldout"]
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
G["audit_sample_size"] = DS.check("audit_sample_requests", len(SAMPLE)) and set(d["query_id"] for d in SAMPLE) == set(AK)
G["recorded_060_heldout_keys_reproduced"] = sum(1 for d in SAMPLE if key_for(rw060[d["query_id"]], d["query"])[0] == AK[d["query_id"]]["sttl_q060_k1_2"][0]) == len(SAMPLE)
G["sttl_060_engine_vs_transcription_tier_disagreements_0"] = sum(1 for _, r in H if mp[t060[r["query_id"]]["tier"]] != rw060[r["query_id"]]["tier"]) == 0
if not all(G.values()):
    json.dump(G, open(X.STTL_FRONTIER_DIR / "logs/heldout_replay_STOPPED_gates.json", "w"), indent=1); sys.exit(f"STOP: held-out gate failed {G}")
lab = L.Held(); keys = {d["query_id"]: {"fresh": AK[d["query_id"]]["fresh"][0], "freshcache": AK[d["query_id"]]["freshcache"][0],
                                        "sttl_t0.60_k0.5_fixed": AK[d["query_id"]]["sttl_q060_k1_2"][0]} for d in SAMPLE}
tiers = {d["query_id"]: {"freshcache": AK[d["query_id"]]["freshcache"][1], "sttl_t0.60_k0.5_fixed": AK[d["query_id"]]["sttl_q060_k1_2"][1]} for d in SAMPLE}
tasks = {}
for role, c in PICK.items():
    row, per, tr = opm("SemanticTTL", c["theta"], c["kappa"]); m, rows = srows(c["theta"], c["kappa"])
    dis = sum(1 for _, r in H if mp[tr[r["query_id"]]["tier"]] != rows[r["query_id"]]["tier"])
    if dis or m["search_saved_pct"] != row["search_saved_pct"]:
        sys.exit(f"STOP: engine/transcription inconsistency on held-out for {c['id']}: {dis} tier disagreements")
    OPS[f"{role}:{c['id']}"] = {**c, **row}
    for _, r in H:
        q = r["query_id"]; reqrows.append({"arm": f"{role}:{c['id']}", "query_id": q, "tier": rows[q]["tier"], "similarity": rows[q].get("similarity"),
                                           "age_s": rows[q].get("age"), "matched_query": rows[q].get("matched_query"), "drift_outcome": per.get(q), "in_support": q in S})
    for d in SAMPLE:
        q = d["query_id"]; k, ctx, gq = key_for(rows[q], d["query"]); keys[q][role] = k; tiers[q][role] = rows[q]["tier"]
        if lab.answer(k) is None: tasks.setdefault(tuple(k), {"key": k, "question": gq, "context": ctx})
for name, (pr, tr_) in {"FreshCache": (fc_per, fc_tr), "sttl_t0.60_k0.5_fixed": (p060, t060)}.items():
    for _, r in H:
        q = r["query_id"]; reqrows.append({"arm": name, "query_id": q, "tier": tr_[q]["tier"], "similarity": None, "age_s": None, "matched_query": None,
                                           "drift_outcome": pr.get(q), "in_support": q in S})
json.dump({"gates": G, "metrics": OPS, "selection_sha256": SEL["sha256"]}, open(X.STTL_FRONTIER_DIR / "heldout/replay_metrics.json", "w"), indent=1)
json.dump({"keys": keys, "tiers": tiers, "picked": PICK}, open(X.STTL_FRONTIER_DIR / "heldout/audit_keys.json", "w"))
with gzip.open(X.STTL_FRONTIER_DIR / "heldout/request_level_replay.csv.gz", "wt", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(reqrows[0])); w.writeheader(); w.writerows(reqrows)
X.wjl(X.STTL_FRONTIER_DIR / "gpu/in/ho_gen_tasks.jsonl", list(tasks.values()))
print("GATES", G); print({k: {x: v[x] for x in ("search_saved_pct", "fetch_per_1k", "gen_per_1k", "drift_pct")} for k, v in OPS.items()}); print("new gen tasks", len(tasks))
