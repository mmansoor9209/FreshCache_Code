"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

SemanticTTL / CategoryAware scripts/baselines, held-out stage (runs only after frozen_selection.json is read-only).
Held-out replays of A1, A2 and CategoryAware-style; the combined CategoryAware-style validation replay (report-only, no re-tuning); answer keys
for the frozen audit and the validation set."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import datetime, hashlib, json, os, stat, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/semantic_ttl_category_aware"))
import baseline_selection_helpers as X
SELP = X.O / "frozen_selection.json"; assert SELP.exists() and not (os.stat(SELP).st_mode & stat.S_IWUSR)
FZ = json.load(open(SELP)); assert hashlib.sha256(json.dumps({k: v for k, v in FZ.items() if k not in ("frozen_utc", "sha256")}, sort_keys=True).encode()).hexdigest() == FZ["sha256"]
X.log(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] held-out stage started; selection sha256 {FZ['sha256']}")
E = X.engine(); exp, p2, me = E["exp"], E["p2"], E["me"]; h, ctx_for = p2.h, p2.ctx_for; Q2 = X.q2(); ST = X.C.streams(E)
H, V = ST["heldout"], ST["validation"]; SH, SV = E["support_of"](H, E["rounds"]), E["support_of"](V, E["rounds"])
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
AK = json.load(open(X.CRR / "results/audit_keys_reported.json"))["keys"]["heldout"]; FR = json.load(open(X.CRR / "results/replays_reported.json"))
EX = json.load(open(X.CRR / "results/extras_reported.json"))
def key_for(p, query):
    ctx = ctx_for(tuple(tuple(x) for x in p["ev"])) if p.get("ev") else ""; gq = p.get("gen_query", query); return ["gen", h(ctx), h(gq)], ctx, gq
G, OPS, KEYS, TASKS = {}, {}, {}, {}
_o = me.outcome
for name in ("A1", "A2"):
    th, ka = FZ[name]["theta"], FZ[name]["kappa"]; evs = []
    def rec(uh, tc, tr, rounds, _o=_o): l = _o(uh, tc, tr, rounds); evs.append((uh, tc, tr, l)); return l
    with X.patched(th, ka):
        me.outcome = rec
        try: m, per, tr_ = X.C.metrics(E, H, "SemanticTTL", SH)
        finally: me.outcome = _o
    with X.patched(th, ka) as pt: mt, rows = Q2.sttl_replay(H, th, ka, pt.ottl)
    G[f"{name}: engine == sttl_replay savings"] = m["search_saved_pct"] == mt["search_saved_pct"]
    per2, ev2, d2 = X.drift_from_rows(E, H, rows, SH)
    G[f"{name}: row-based drift scorer reproduces mixed_age_replay per-request outcomes"] = all(per2.get(q) == per.get(q) for q in SH)
    t2q = {t: r["query_id"] for t, r in H}; byq = {}
    for uh, tc, trq, l in evs: byq.setdefault(t2q[trq], []).append((uh, tc, trq, l))
    OPS[name] = {"method": f"SemanticTTL theta={th} kappa={ka:g}", **{k: m[k] for k in ("search_saved_pct", "l1_hits", "fetches", "generations", "fetch_per_1k", "gen_per_1k", "drift_pct", "coverage_pct")},
                 "search_calls": mt["search_calls"], "audit_secondary": X.cross_round_audit(E, SH, per, byq)}
    KEYS[name] = {}
    for d in SAMPLE:
        kk, ctx, gq = key_for(rows[d["query_id"]], d["query"]); KEYS[name][d["query_id"]] = {"key": kk, "tier": rows[d["query_id"]]["tier"]}; TASKS.setdefault(tuple(kk), {"key": kk, "question": gq, "context": ctx})
CA = FZ["CategoryAware_style"]["per_category"]; TH = {c: v["theta"] for c, v in CA.items()}; KA = {c: v["kappa"] for c, v in CA.items()}; TH["REAL_TIME"] = 1.1; KA["REAL_TIME"] = 0.0
m, rows = X.ca_replay(H, TH, KA); per, evs, d = X.drift_from_rows(E, H, rows, SH)
OPS["CategoryAware_style"] = {"method": "CategoryAware-style " + json.dumps({c: (TH[c], KA[c]) for c in CA}), **m, **{k: d[k] for k in ("drift_pct", "coverage_pct")},
                              "audit_secondary": X.cross_round_audit(E, SH, per, evs)}
REQ = {r["query_id"]: r for _, r in H}
OPS["CategoryAware_style"]["cross_category_hits"] = sum(1 for q, r in rows.items() if r["tier"] == "L1" and r["entry_class"] != REQ[q]["freshness_class"])
KEYS["CategoryAware_style"] = {}
for dd in SAMPLE:
    kk, ctx, gq = key_for(rows[dd["query_id"]], dd["query"]); KEYS["CategoryAware_style"][dd["query_id"]] = {"key": kk, "tier": rows[dd["query_id"]]["tier"]}; TASKS.setdefault(tuple(kk), {"key": kk, "question": gq, "context": ctx})
OPS["FreshCache"] = {**FR["heldout/FreshCache"], "method": "FreshCache (frozen primary)"}; OPS["SemanticTTL_0.60_k0.5_fixed"] = {**EX["heldout_semanticttl"]["quality_selected_0.60_1/2"]["replay"], "method": "existing fixed SemanticTTL (0.60,1/2)"}
KEYS["FreshCache"] = {q: {"key": v["freshcache"][0], "tier": v["freshcache"][1]} for q, v in AK.items()}; KEYS["SemanticTTL_0.60_k0.5_fixed"] = {q: {"key": v["sttl_q060_k1_2"][0], "tier": v["sttl_q060_k1_2"][1]} for q, v in AK.items()}
KEYS["fresh"] = {q: {"key": v["fresh"][0], "tier": "fresh"} for q, v in AK.items()}
# combined CategoryAware-style on validation (report-only)
mv, rv = X.ca_replay(V, TH, KA); pv, ev_, dv = X.drift_from_rows(E, V, rv, SV)
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}; EVS = json.load(open(X.O / "work/stage1.json"))["EV"]
VK = {}
for q in EVS:
    kk, ctx, gq = key_for(rv[q], QS[q]["query"]); VK[q] = {"key": kk, "tier": rv[q]["tier"]}; TASKS.setdefault(tuple(kk), {"key": kk, "question": gq, "context": ctx})
REQV = {r["query_id"]: r for _, r in V}
VCOMB = {**mv, **{k: dv[k] for k in ("drift_pct", "coverage_pct")}, "cross_category_hits": sum(1 for q, r in rv.items() if r["tier"] == "L1" and r["entry_class"] != REQV[q]["freshness_class"])}
for k_, v in G.items(): X.log("GATE", k_, v)
if not all(G.values()): X.log("STOP: held-out gate failed"); sys.exit(1)
hl, vl = X.Held(), X.Val(); todo = [t for t in TASKS.values() if hl.answer(t["key"]) is None and vl.answer(t["key"]) is None]
X.wjl(X.O / "gpu/in/gen_tasks.jsonl", todo)
json.dump({"gates": G, "ops": OPS, "keys": KEYS, "ca_validation_combined": {"ops": VCOMB, "keys": VK}}, open(X.O / "work/stage3.json", "w"))
X.log(json.dumps({k: {x: v.get(x) for x in ("search_saved_pct", "l1_hits", "fetches", "drift_pct")} for k, v in OPS.items()})); X.log("CA validation combined", json.dumps(VCOMB)); X.log("new answers needed:", len(todo))
