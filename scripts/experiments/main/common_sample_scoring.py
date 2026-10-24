"""Scoring of the policy comparisons on the common sample (work/common.pkl), including the 2x2 root-clone diagnostic. Writes outputs/common_sample_construction.json:
 accounting per comparison (newly wrong, abstention losses, total losses, gains, net, conditional loss rate, expanded totals, stream change with
 Wald-FPC and root-clustered intervals, sufficiency change on the full evidence block and on the exact generator context), the additive
 decomposition D'-B = (D'-B-)+(B--B), lineage categories of every loss and gain, exposure denominators from the full traces, and a check
 of the earlier stream-level expansions."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, math, pickle, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); sys.argv = [sys.argv[0], "score"]
import output_workspace; output_workspace.use("common_sample"); import composition_study_helpers as X   # noqa: E402
from common_sample_construction import goldof, REC, Lab, POOLDIRS, POL   # noqa: E402
C = pickle.load(open(O / "work/common.pkl", "rb")); rows, S, U, Nref = C["rows"], C["S"], len(C["U"]), C["N_ref"]; n = len(S)
T = pickle.load(open(O / "work/traces.pkl", "rb")); TR = json.load(open(O / "outputs/write_control_traces.json"))
lab = Lab()
SU = {"llama8b": {}, "qwen7b": {}}
for d in [X.RCD / "gpu/out"] + POOLDIRS:
    for j in SU:
        for f in sorted(d.glob(f"suffv_{j}*.jsonl")):
            for r in X.jl(f): SU[j].setdefault((r["query_id"], r["arm"]), r["label"])
def suff(q, arm):
    if arm is None: return False, True
    s = (SU["llama8b"].get((q, arm)), SU["qwen7b"].get((q, arm)))
    if None in s: return None, False
    return s[0] == s[1] == "SUFFICIENT", s[0] == s[1]
L, RL, SF, SC = {}, {}, {}, {}
for (q, p), a in rows.items():
    L[(q, p)] = lab.label(REC[q]["query"], goldof(q), a["key"])[0] or "UNRESOLVED"
    if a["tier"] == "L1": RL[(q, p)] = (lab.label(REC[a["root"]]["query"], goldof(a["root"]), a["key"])[0] or "UNRESOLVED") if goldof(a["root"]) else "NO_REFERENCE"
    SF[(q, p)] = suff(q, a["blk_sha"])[0]; SC[(q, p)] = suff(q, ("ctx_" + a["ctx_sha"]) if a["ctx_sha"] else None)[0]
unres = sum(v == "UNRESOLVED" for v in L.values()); suff_unres = sum(v is None for v in SF.values()) + sum(v is None for v in SC.values())
cor = lambda q, p: L[(q, p)] == "CORRECT"
def wilson(k, m, z=1.96):
    if not m: return None
    p = k / m; d = 1 + z * z / m; c = p + z * z / (2 * m); h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)); return [round(100 * (c - h) / d, 2), round(100 * (c + h) / d, 2)]
def wald_total(k):   # total in the frame U from k of n (SRS without replacement)
    p = k / n; se = U * math.sqrt(p * (1 - p) / n * (1 - n / U)); return [round(U * p, 1), [round(max(0, U * p - 1.96 * se), 1), round(U * p + 1.96 * se, 1)]]
def stream(d, cl=None):
    mu = sum(d) / n; var = sum((x - mu) ** 2 for x in d) / (n - 1); se = math.sqrt(var / n * (1 - n / U)); f = 100 * U / Nref
    out = {"pp": round(f * mu, 3), "ci_pp": [round(f * (mu - 1.96 * se), 3), round(f * (mu + 1.96 * se), 3)]}
    if cl is not None:   # root-clustered (cluster = served answer-cache entry's root, else the request itself)
        g = collections.defaultdict(float)
        for x, c in zip(d, cl): g[c] += x - mu
        sec = math.sqrt(len(g) / (len(g) - 1) * sum(v * v for v in g.values()) / n ** 2 * (1 - n / U)); out["ci_pp_root_clustered"] = [round(f * (mu - 1.96 * sec), 3), round(f * (mu + 1.96 * sec), 3)]; out["clusters"] = len(g)
    return out
OWN = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in X.engine()["recs"]}
def category(q, base, p, loss):
    # pre-registered taxonomy, refined (reported as a refinement): a regeneration where the baseline served a stored answer is classed as
    # regeneration whatever its route; an L2 route is split by whether the reused list differs from the request's own list
    a, b = rows[(q, p)], rows[(q, base)]
    lst = lambda: "list_identical" if T[p][q]["served"] == OWN[q] else "list_differs"
    if b["tier"] == "L1" and a["tier"] in ("L2", "miss"):
        return ("regeneration_loss_" if loss else "regeneration_gain_") + ("via_reused_list_" + lst() if a["tier"] == "L2" else "fresh")
    if a["tier"] == "L2": return ("immediate_L2_" if loss else "immediate_L2_gain_") + lst()
    if a["tier"] == "L1":
        if not loss: return "served_entry_gain_" + ("L2_root" if a["root_route"] == "L2" else "fresh_root")
        rl = RL[(q, p)]; tag = {"WRONG": "wrong", "ABSTAIN": "wrong", "CORRECT": "correct"}.get(rl, "unlabelled")
        if a["root_route"] == "L2": return {"wrong": "propagation_of_wrong_L2_root", "correct": "misuse_of_correct_L2_root"}.get(tag, "L2_root_unlabelled")
        return {"wrong": "served_wrong_fresh_root", "correct": "misuse_of_correct_fresh_root"}.get(tag, "fresh_root_unlabelled")
    return "fresh_vs_fresh_other"
def compare(base, p):
    nw = [q for q in S if cor(q, base) and L[(q, p)] == "WRONG"]; ab = [q for q in S if cor(q, base) and L[(q, p)] == "ABSTAIN"]
    gw = [q for q in S if not cor(q, base) and cor(q, p)]; bc = sum(cor(q, base) for q in S)
    d = [int(cor(q, p)) - int(cor(q, base)) for q in S]
    cl = [rows[(q, p)]["root"] if rows[(q, p)]["tier"] == "L1" else (rows[(q, base)]["root"] if rows[(q, base)]["tier"] == "L1" else q) for q in S]
    ds = [int(bool(SF[(q, p)])) - int(bool(SF[(q, base)])) for q in S]; dc = [int(bool(SC[(q, p)])) - int(bool(SC[(q, base)])) for q in S]
    lo = nw + ab
    return {"base": base, "policy": p, "n": n, "base_correct": bc, "policy_correct": sum(cor(q, p) for q in S), "newly_wrong": len(nw), "abstention_losses": len(ab), "total_losses": len(lo),
            "gains": len(gw), "gains_from_wrong": sum(L[(q, base)] == "WRONG" for q in gw), "gains_from_abstain": sum(L[(q, base)] == "ABSTAIN" for q in gw), "net": len(gw) - len(lo),
            "mcnemar_p": round(X.mcnemar(len(lo), len(gw)), 4),
            "loss_rate_given_base_correct_pct": round(100 * len(lo) / bc, 2) if bc else None, "loss_rate_given_base_correct_ci": wilson(len(lo), bc),
            "expanded_losses_in_U": wald_total(len(lo)), "expanded_newly_wrong_in_U": wald_total(len(nw)), "expanded_abstention_in_U": wald_total(len(ab)), "expanded_gains_in_U": wald_total(len(gw)),
            "stream_accuracy": stream(d, cl), "stream_losses_pp": round(100 * U / Nref * len(lo) / n, 3), "stream_gains_pp": round(100 * U / Nref * len(gw) / n, 3),
            "suff_full_evidence": stream(ds), "suff_generator_context": stream(dc),
            "loss_categories": dict(collections.Counter(category(q, base, p, True) for q in lo)), "gain_categories": dict(collections.Counter(category(q, base, p, False) for q in gw)),
            "distinct_served_roots_among_losses": len({rows[(q, p)]["root"] for q in lo if rows[(q, p)]["tier"] == "L1"}), "losses_served_from_answer_cache": sum(rows[(q, p)]["tier"] == "L1" for q in lo)}
PAIRS = [("B", "D"), ("B", "Dp"), ("B", "Bm"), ("Bm", "Dp"), ("D", "Dp"), ("B", "Be"), ("Be", "Dp")]
R = {"design": {"frame_U": U, "N_ref": Nref, "N": json.load(open(OUTPUT_ROOT / "validation_split/split.json"))["test"]["requests"], "n": n, "seed": 20261008, "estimator": "(U/N_ref)*mean(d), d in {-1,0,1}; Wald with FPC; root-clustered variant",
                "zero_outside_U": "requests outside U have identical generation inputs under every policy (greedy, cached by input key), so d = 0"},
     "unresolved_labels": unres, "unresolved_sufficiency": suff_unres, "accuracy_in_sample_pct": {p: round(100 * sum(cor(q, p) for q in S) / n, 2) for p in POL},
     "comparisons": {f"{a}->{b}": compare(a, b) for a, b in PAIRS}}
c = R["comparisons"]
R["decomposition_quality_pp"] = {"Dp-B": c["B->Dp"]["stream_accuracy"]["pp"], "Dp-Bm": c["Bm->Dp"]["stream_accuracy"]["pp"], "Bm-B": c["B->Bm"]["stream_accuracy"]["pp"],
                                 "check_sum": round(c["Bm->Dp"]["stream_accuracy"]["pp"] + c["B->Bm"]["stream_accuracy"]["pp"], 3)}
P = TR["policies"]; R["decomposition_cost_ms"] = {"B": P["B"]["cost_ms"], "Bm": P["Bm"]["cost_ms"], "Dp": P["Dp"]["cost_ms"], "D": P["D"]["cost_ms"], "Be": P["Be"]["cost_ms"],
                                                  "Dp-B": round(P["Dp"]["cost_ms"] - P["B"]["cost_ms"], 1), "Dp-Bm": round(P["Dp"]["cost_ms"] - P["Bm"]["cost_ms"], 1), "Bm-B": round(P["Bm"]["cost_ms"] - P["B"]["cost_ms"], 1)}
# exposure denominators (full held-out traces, requests with a reference, not REAL_TIME)
ref = [q for q in T["B"] if goldof(q) and T["B"][q]["tier"] != "REAL_TIME"]
R["exposure"] = {p: {"L2_reuse_events": sum(T[p][q]["tier"] == "L2" for q in ref), "answer_hits_from_L2_root": sum(T[p][q]["tier"] == "L1" and T[p][q]["root_route"] == "L2" for q in ref),
                     "answer_hits_from_fresh_root": sum(T[p][q]["tier"] == "L1" and T[p][q]["root_route"] == "miss" for q in ref), "fresh_generations": sum(T[p][q]["tier"] == "miss" for q in ref)} for p in POL}
# reused list identical to the request's own list (the benchmark's paraphrases inherit their base query's URL list)
R["L2_served_equals_own_list"] = {}
for p in ("D", "Dp"):
    ev = [q for q in ref if T[p][q]["tier"] == "L2"]
    R["L2_served_equals_own_list"][p] = {"L2_events": len(ev), "identical_list": sum(T[p][q]["served"] == OWN[q] for q in ev), "same_url_set": sum(set(T[p][q]["served"]) == set(OWN[q]) for q in ev)}
    sv = [q for q in S if rows[(q, p)]["tier"] == "L2"]; base = "Bm" if p == "Dp" else "B"
    for kind in ("identical", "differs"):
        e = [q for q in sv if (T[p][q]["served"] == OWN[q]) == (kind == "identical")]
        R["L2_served_equals_own_list"][p][f"sample_{kind}_events"] = len(e)
        R["L2_served_equals_own_list"][p][f"sample_{kind}_losses_vs_{base}"] = sum(cor(q, base) and not cor(q, p) for q in e)
        R["L2_served_equals_own_list"][p][f"sample_{kind}_gains_vs_{base}"] = sum(not cor(q, base) and cor(q, p) for q in e)
# D' re-serving the lists behind D's propagated errors
dl = [q for q in S if cor(q, "B") and not cor(q, "Dp") and rows[(q, "Dp")]["tier"] == "L2" and T["Dp"][q]["served"] != OWN[q]]
R["Dp_list_losses_vs_D_route"] = {"Dp_losses_on_differing_reused_list": len(dl), "D_route_and_label": {f"{rows[(q, 'D')]['tier']}|{rows[(q, 'D')].get('root_route')}|{L[(q, 'D')]}": 0 for q in dl}}
for q in dl: R["Dp_list_losses_vs_D_route"]["D_route_and_label"][f"{rows[(q, 'D')]['tier']}|{rows[(q, 'D')].get('root_route')}|{L[(q, 'D')]}"] += 1
R["Dp_list_losses_vs_D_route"]["same_list_origin_as_D"] = sum(T["Dp"][q]["list_root"] == (T["D"][T["D"][q]["root"]].get("list_root") if T["D"][q]["tier"] == "L1" else T["D"][q].get("list_root")) for q in dl)
R["Dp_list_losses_vs_D_route"]["distinct_list_origins"] = len({T["Dp"][q]["list_root"] for q in dl})
json.dump(R, open(O / "outputs/common_sample_construction.json", "w"), indent=1); print(json.dumps(R, indent=1))
