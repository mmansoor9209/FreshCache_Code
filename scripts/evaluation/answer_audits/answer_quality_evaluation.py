"""Recompute answer-quality results for the keyset keys (recorded keys; reproduction gate) and the reported keys.
Labelling = each study's reported rule, transcribed: ABSTAIN on the study's abstention pattern, else CORRECT only if BOTH
llama8b and qwen7b say CORRECT, else WRONG. Each study's own answer/judgment files take precedence (last-wins, as in the
originals); the read-only pools and this folder's gpu/out fill only keys/sigs the study never had."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, csv, math, collections, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
from fractions import Fraction
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import answer_pools as P
R = P.R; W = ((OUTPUT_ROOT / "replay")); GO = W/"gpu/out"; V = OUTPUT_ROOT / "archived_runs"; PRIOR = OUTPUT_ROOT / "archived_runs"
h, jl = P.h16, P.jl
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
PRIM = ["i don't know", "i dont know", "cannot determine", "not enough information", "insufficient", "no information", "unclear from the context"]
GA = P.AnswerLookup(extra=[GO/"answers_llama3b.jsonl"]); GJ = P.judge2(extra=[GO])
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def wil(k, n, z=1.96):
    if not n: return [None, None]
    p, d = k / n, 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0, (c - m) / d), 3), round(100 * min(1, (c + m) / d), 3)]
class Labeler:
    def __init__(self, ans_files, judge_files, abst, ctxq_files=()):
        self.exact, self.content = {}, {}   # exact (role, ctx, q) keys first, as the original analyses do; content (ctx, q) only as fallback
        for f in ans_files:
            for r in jl(f): self.exact[tuple(r["key"])] = r["answer"]
        for f in ctxq_files:
            for r in jl(f): self.content.setdefault((r["ctx_sha"], r["q_sha"]), r["answer"])
        for k, a in self.exact.items(): self.content.setdefault((k[1], k[2]), a)
        self.j = {"llama8b": {}, "qwen7b": {}}
        for j in self.j:
            for f in judge_files:
                for r in jl(str(f).format(j=j)):
                    if "sig" in r: self.j[j][tuple(r["sig"])] = r["label"]
        self.abst = abst; self.filled = collections.Counter()
    def answer(self, key):
        if tuple(key) in self.exact: return self.exact[tuple(key)]
        k = (key[1], key[2])
        if k in self.content: self.filled["answer_by_content"] += 1; return self.content[k]
        a = GA.get(key)
        if a is not None: self.filled["answer_from_pool"] += 1; return a
        return None
    def __call__(self, q, gold, key):
        a = self.answer(key)
        if a is None: return None
        if any(p in a.strip().lower() for p in self.abst): return "ABSTAIN"
        s = (h(q), h(gold), h(a)); l = []
        for j in ("llama8b", "qwen7b"):
            v = self.j[j].get(s)
            if v is None: v = GJ[j].get(s); self.filled["judge_from_pool"] += v is not None
            l.append(v)
        if None in l: return None
        return "CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"
def score(rows, arms, ref="fresh", base="fc"):
    N = len(rows); fcn = sum(1 for r in rows if r[ref] == "CORRECT"); out = {"N": N, "fresh_correct": fcn}
    for a in arms:
        c = collections.Counter(r[a] for r in rows); w = sum(1 for r in rows if r[ref] == "CORRECT" and r[a] == "WRONG")
        out[a] = {"correct": c["CORRECT"], "wrong": c["WRONG"], "abstain": c["ABSTAIN"], "accuracy_pct": round(100 * c["CORRECT"] / N, 3) if N else None, "accuracy_ci95": wil(c["CORRECT"], N)}
        if a != ref:
            out[a].update({"wai": w, "wai_pct": round(100 * w / N, 3) if N else None, "wai_ci95": wil(w, N), "conditional_wai_pct": round(100 * w / fcn, 3) if fcn else None,
                           "conditional_wai_ci95": wil(w, fcn), "reverse": sum(1 for r in rows if r[ref] == "WRONG" and r[a] == "CORRECT")})
    for a in arms:
        if a in (ref, base): continue
        wi = lambda r, x: r[ref] == "CORRECT" and r[x] == "WRONG"
        bo, fo, both = sum(1 for r in rows if wi(r, a) and not wi(r, base)), sum(1 for r in rows if wi(r, base) and not wi(r, a)), sum(1 for r in rows if wi(r, a) and wi(r, base))
        cb, ca = sum(1 for r in rows if r[base] == "CORRECT" and r[a] != "CORRECT"), sum(1 for r in rows if r[a] == "CORRECT" and r[base] != "CORRECT")
        out[f"paired_{a}_vs_{base}"] = {"wai_both": both, "wai_baseline_only": bo, "wai_freshcache_only": fo, "wai_mcnemar_p": mcn(bo, fo),
                                        "correct_freshcache_only": cb, "correct_baseline_only": ca, "correct_mcnemar_p": mcn(cb, ca)}
    return out
def cluster(rows, arm, base="fc", ref="fresh"):
    cl = collections.defaultdict(lambda: [0, 0])
    for r in rows: cl[r["cluster"]][0] += r[ref] == "CORRECT" and r[base] == "WRONG"; cl[r["cluster"]][1] += r[ref] == "CORRECT" and r[arm] == "WRONG"
    ob = sum(1 for a, b in cl.values() if b and not a); oa = sum(1 for a, b in cl.values() if a and not b)
    return {"clusters": len(cl), "baseline_only": ob, "freshcache_only": oa, "both": sum(1 for a, b in cl.values() if a and b), "exact_mcnemar_p": mcn(ob, oa)}
def by_tier(rows, arm, tiers, ref="fresh"):
    out = {}
    for t in tiers:
        s = [r for r in rows if r[f"{arm}_tier"] == t]
        out[t] = {"N": len(s), "correct": sum(1 for r in s if r[arm] == "CORRECT"), "wai": sum(1 for r in s if r[ref] == "CORRECT" and r[arm] == "WRONG")}
    return out
RES, GATES = {}, {}
def gate(name, got, want=None):
    """want: the value recorded by the stage that produces it. None: no other stage computes this value, so it is
    recorded here under answer_quality/<name> and reported without a pass/fail check."""
    if want is None:
        REF.record("answer_quality/" + name, got); GATES[name] = {"got": got, "recorded": None, "pass": None}; return
    GATES[name] = {"got": got, "recorded": want, "pass": got == want}
def refs(names, got):
    """Recorded values for `names` (general_audit_analysis.py / heldout_sensitivity_analysis.py); the predicted-class arm is
    computed only here, so its values are recorded by this stage."""
    return [REF.record(n, v) if n.startswith("heldout_audit/predicted_class/") else REF.value(n) for n, v in zip(names, got)]
HA_ = "heldout_audit/"
# ============ 1. held-out audit (Table 6 blocks; supported / screened / intersection subsets; cluster test; predicted class) ============
A2, HS, PC = ((OUTPUT_ROOT / "general_audit")), ((OUTPUT_ROOT / "quality_selection/heldout_sensitivity")), ((OUTPUT_ROOT / "predicted_class_audit"))
AUDIT_SAMPLE = json.load(open(A2/"sample.json"))["requests"]; HSm = {json.loads(l)["query_id"]: json.loads(l) for l in open(HS/"h_sample.jsonl")}
XW = json.load(open(OUTPUT_ROOT / "audit_recounts/artifacts/screened_subset_crosswalk.json")); SUB = {"screened": set(XW["screened"]), "supported": set(XW["supported"]), "intersection": set(XW["intersection"])}
LH = Labeler([A2/"answers.jsonl", HS/"h_answers.jsonl"], [A2/"judge_{j}.jsonl", HS/"h_judge_{j}.jsonl", PC/"new_judge_{j}.jsonl"], HELD, ctxq_files=[PC/"new_answers.jsonl"])
ARM = {"fresh": "fresh", "fc": "freshcache", "sttl_k1_16": "sttl_k1_16", "sttl_t060_k1_2": "sttl_q060_k1_2", "predicted": "predicted_class"}
def heldout(eng):
    K = json.load(open(W/f"results/audit_keys_{eng}.json"))["keys"]["heldout"]; rows, drop = [], []
    for d in AUDIT_SAMPLE:
        q = d["query_id"]; r = {"qid": q, "cluster": HSm[q]["cluster_id"]}
        for a, src in ARM.items(): r[a] = LH(d["query"], d["gold"], K[q][src][0]); r[f"{a}_tier"] = K[q][src][1]
        (drop if any(r[a] is None for a in ARM) else rows).append(r if all(r[a] is not None for a in ARM) else q)
    out = {"dropped": drop, "all_audit": score(rows, list(ARM))}
    for n, ids in SUB.items(): out[n] = score([r for r in rows if r["qid"] in ids], list(ARM))
    out["cluster_level"] = {a: cluster(rows, a) for a in ("sttl_t060_k1_2", "sttl_k1_16")}
    out["fc_by_tier"] = by_tier(rows, "fc", ["L1", "L2", "miss"]); out["predicted_by_tier"] = by_tier(rows, "predicted", ["L1", "L2", "miss", "REAL_TIME"])
    out["wai_request_ids"] = {a: sorted(r["qid"] for r in rows if r["fresh"] == "CORRECT" and r[a] == "WRONG") for a in ARM if a != "fresh"}
    return out, rows
ROWS = {}
for eng in ("keyset", "reported"): RES[f"heldout_{eng}"], ROWS[f"heldout_{eng}"] = heldout(eng)
L = RES["heldout_keyset"]; a4 = L["all_audit"]
gate("heldout all_audit fresh/fc/k16/t060/pred correct", [a4["fresh_correct"], a4["fc"]["correct"], a4["sttl_k1_16"]["correct"], a4["sttl_t060_k1_2"]["correct"], a4["predicted"]["correct"]]
     , refs([HA_ + "fresh_correct", HA_ + "FreshCache_primary/CORRECT", HA_ + "SemanticTTL_k1_16/CORRECT", HA_ + "SemanticTTL_0.60_k0.5_fixed/CORRECT", HA_ + "predicted_class/CORRECT"],
            [a4["fresh_correct"], a4["fc"]["correct"], a4["sttl_k1_16"]["correct"], a4["sttl_t060_k1_2"]["correct"], a4["predicted"]["correct"]]))
gate("heldout all_audit WAI fc/k16/t060/pred", [a4["fc"]["wai"], a4["sttl_k1_16"]["wai"], a4["sttl_t060_k1_2"]["wai"], a4["predicted"]["wai"]],
     refs([HA_ + "FreshCache_primary/WAI", HA_ + "SemanticTTL_k1_16/WAI", HA_ + "SemanticTTL_0.60_k0.5_fixed/WAI", HA_ + "predicted_class/WAI"], [a4["fc"]["wai"], a4["sttl_k1_16"]["wai"], a4["sttl_t060_k1_2"]["wai"], a4["predicted"]["wai"]]))
for n in ("screened", "supported", "intersection"):
    x = L[n]; gate(f"heldout {n} freshC/fcC/fcW/t060C/t060W/k16C/k16W", [x["fresh_correct"], x["fc"]["correct"], x["fc"]["wai"], x["sttl_t060_k1_2"]["correct"], x["sttl_t060_k1_2"]["wai"], x["sttl_k1_16"]["correct"], x["sttl_k1_16"]["wai"]])
gate("heldout screened paired t060 vs fc (both/base/fc)", [L["screened"]["paired_sttl_t060_k1_2_vs_fc"][k] for k in ("wai_both", "wai_baseline_only", "wai_freshcache_only")])
gate("heldout screened paired k16 vs fc (both/base/fc)", [L["screened"]["paired_sttl_k1_16_vs_fc"][k] for k in ("wai_both", "wai_baseline_only", "wai_freshcache_only")])
gate("heldout cluster t060 (clusters/base/fc)", [L["cluster_level"]["sttl_t060_k1_2"][k] for k in ("clusters", "baseline_only", "freshcache_only")])
gate("heldout cluster k16 (base/fc)", [L["cluster_level"]["sttl_k1_16"][k] for k in ("baseline_only", "freshcache_only")])
# ============ 2. primary audit (gate on/off; SemanticTTL kappa 1/8; screened subset) ============
MA, BO = ((OUTPUT_ROOT / "mixed_age_audit")), ((OUTPUT_ROOT / "semanticttl_audit"))
SP = [json.loads(l) for l in open(MA/"mixed_age_sample.jsonl")]
LP = Labeler([MA/"answers2.jsonl", BO/"sttl_answers.jsonl"], [MA/"judge2_{j}.jsonl", BO/"sttl_judge_{j}.jsonl"], PRIM)
GV, RV = ((OUTPUT_ROOT / "gold_verification")), ((OUTPUT_ROOT / "gold_verification/timestamp_validity"))
fid = {r["query_id"]: r["final_label"] for r in csv.DictReader(open(GV/"paraphrase_fidelity.csv"))}; unit = {}
for r in csv.DictReader(open(RV/"r_gold_validity.csv")):
    for q in r["request_ids"].split("|"): unit[q] = r
EV = {q for q in unit if fid.get(q, "BASE_QUERY") in ("SAME", "BASE_QUERY") and unit[q]["status_majority"] == "VALID_AT_T"}
PARM = {"fresh": "fresh", "fc": "freshcache", "gate_off": "gate_off", "sttl_k1_8": "sttl_k1_8"}
def primary(eng):
    K = json.load(open(W/f"results/audit_keys_{eng}.json"))["keys"]["primary"]; rows = []
    for d in SP:
        q = d["query_id"]; r = {"qid": q}
        for a, src in PARM.items(): r[a] = LP(d["query"], d["gold"], K[q][src][0]); r[f"{a}_tier"] = K[q][src][1]
        if all(r[a] is not None for a in PARM): rows.append(r)
    ROWS[f"primary_{eng}"] = rows
    out = {"all_audit": score(rows, list(PARM)), "screened_subset": score([r for r in rows if r["qid"] in EV], list(PARM)), "fc_by_tier": by_tier(rows, "fc", ["L1", "L2", "miss"]),
           "gate_off_by_tier": by_tier(rows, "gate_off", ["L1", "L2", "miss"])}
    return out
for eng in ("keyset", "reported"): RES[f"primary_{eng}"] = primary(eng)
L = RES["primary_keyset"]; a4 = L["all_audit"]
gate("primary fresh/on/off/sttl8 correct", [a4["fresh_correct"], a4["fc"]["correct"], a4["gate_off"]["correct"], a4["sttl_k1_8"]["correct"]])
gate("primary WAI on/off/sttl8", [a4["fc"]["wai"], a4["gate_off"]["wai"], a4["sttl_k1_8"]["wai"]])
gate("primary paired off vs on (base/fc)", [a4["paired_gate_off_vs_fc"]["wai_baseline_only"], a4["paired_gate_off_vs_fc"]["wai_freshcache_only"]])
gate("primary paired sttl8 vs on (base/fc)", [a4["paired_sttl_k1_8_vs_fc"]["wai_baseline_only"], a4["paired_sttl_k1_8_vs_fc"]["wai_freshcache_only"]])
gate("primary screened subset N, sttl8-only/fc-only", [L["screened_subset"]["N"], L["screened_subset"]["paired_sttl_k1_8_vs_fc"]["wai_baseline_only"], L["screened_subset"]["paired_sttl_k1_8_vs_fc"]["wai_freshcache_only"]])
# ============ 3. budget audit A-D (same held-out audit requests) ============
AQ = ((OUTPUT_ROOT / "parameter_calibration/answer_quality"))
LB = Labeler([AQ/"cached_answers.jsonl", AQ/"answers_llama3b.jsonl"], [AQ/"judge_{j}.jsonl"], HELD)
BC = ["fresh", "A_original", "B_mult_only", "C_budget_only", "D_full"]
def budget(eng):
    B = json.load(open(W/f"results/budget_keys_{eng}.json")); rows = []
    for d in B["records"]:
        r = {"qid": d["query_id"]}
        for a in BC: r[a] = LB(d["query"], d["gold"], d[f"{a}_key" if a != "fresh" else "fresh_key"])
        for a in BC[1:]: r[f"{a}_tier"] = d[f"{a}_tier"]
        if all(r[a] is not None for a in BC): rows.append(r)
    return {"eps_used": B["eps_used"], "scores": score(rows, BC, base="A_original"), "tiers": {a: dict(collections.Counter(r[f"{a}_tier"] for r in rows)) for a in BC[1:]}}
for eng in ("keyset", "reported"): RES[f"budget_{eng}"] = budget(eng)
x = RES["budget_keyset"]["scores"]
gate("budget correct/WAI A (FreshCache as recorded)", [x["A_original"]["correct"], x["A_original"]["wai"]],
     refs([HA_ + "FreshCache_primary/CORRECT", HA_ + "FreshCache_primary/WAI"], [x["A_original"]["correct"], x["A_original"]["wai"]]))   # A = FreshCache on the same audit
gate("budget correct/WAI B,C,D", [[x[a]["correct"], x[a]["wai"]] for a in BC[2:]])
# ============ 4. stricter L1 gate: held-out audit arms, held-out census (jury), validation stage 1 + stage 2 ============
LG = ((OUTPUT_ROOT / "l1_precision")); HA = json.load(open(LG/"heldout_audit_sample.json"))["requests"]; VA = json.load(open(LG/"validation_audit_sample.json"))["requests"]
LS = Labeler([LG/"heldout_answers_llama3b.jsonl", LG/"validation_answers_llama3b.jsonl"], [LG/"heldout_judge_{j}.jsonl", LG/"validation_judge_{j}.jsonl"], HELD)
JY = P.jury(extra=[str(GO/"equiv_{j}.jsonl")])
def verdict(pid):
    v = {j: l for j, l in JY.get(pid, {}).items() if l in ("SAME", "DIFFERENT")}
    if len(v) < 2: return "UNJUDGED"
    c = collections.Counter(v.values()); return "JURY_TIE" if c["SAME"] == c["DIFFERENT"] else c.most_common(1)[0][0]
def census(hits):
    vs = [verdict(x["pair_id"]) for x in hits]; n = len(vs); d = vs.count("DIFFERENT"); t = vs.count("JURY_TIE"); det = d + vs.count("SAME")
    xc = [verdict(x["pair_id"]) for x in hits if x["cross_cluster"]]
    return {"l1_hits": n, "jury_ties": t, "jury_determinate": det, "unjudged": vs.count("UNJUDGED"), "mismatch_ties_diff_pct": round(100 * (d + t) / n, 4) if n else None,
            "mismatch_no_ties_pct": round(100 * d / det, 4) if det else None, "mismatch_ties_same_pct": round(100 * d / n, 4) if n else None,
            "cross_cluster_hits": len(xc), "cross_cluster_mismatch_ties_diff_pct": round(100 * (xc.count("DIFFERENT") + xc.count("JURY_TIE")) / len(xc), 4) if xc else None}
def stricter(eng):
    SC = json.load(open(W/f"results/stricter_cpu_{eng}.json")); VF = json.load(open(W/f"results/stricter_validation_fresh_{eng}.json"))
    HK = json.load(open(W/f"results/audit_keys_{eng}.json"))["keys"]["heldout"]; out = {"heldout": {}, "validation_cells": {}}
    for arm, v in SC["heldout"].items():
        rows = []
        for d in HA:
            q = d["query_id"]; r = {"qid": q, "fresh": LS(d["query"], d["gold"], HK[q]["fresh"][0]), "arm": LS(d["query"], d["gold"], v["audit"][q]["key"]), "arm_tier": v["audit"][q]["tier"]}
            if r["fresh"] is not None and r["arm"] is not None: rows.append(r)
        sc = score(rows, ["fresh", "arm"], base="arm")
        out["heldout"][arm] = {"replay": {k: v[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "generations")}, "census": census(v["hits"]), "audit": sc["arm"], "audit_fresh_correct": sc["fresh_correct"], "audit_N": sc["N"],
                               "audit_l1_requests": sum(1 for r in rows if r["arm_tier"] == "L1"), "wai_request_ids": sorted(r["qid"] for r in rows if r["fresh"] == "CORRECT" and r["arm"] == "WRONG")}
    for cell, v in SC["validation"].items():
        out["validation_cells"][cell] = {**{k: v[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "generations")}, **census(v["hits"])}
    # stage 1 (l1_precision_selection.stage1): C1 savings >= recorded - 0.10, C3 >= 100 L1 hits, rank by mismatch (ties->DIFFERENT), top K=5
    cells = out["validation_cells"]; pub = cells["recorded"]; C1 = pub["search_saved_pct"] - 0.10
    cand = [dict(v, cell=c) for c, v in cells.items() if c != "recorded" and v["mismatch_ties_diff_pct"] is not None]
    c1 = [r for r in cand if r["search_saved_pct"] >= C1]; c13 = [r for r in c1 if r["l1_hits"] >= 100]
    top = sorted(c13, key=lambda r: r["mismatch_ties_diff_pct"])[:5]
    st = {"C1_floor": round(C1, 4), "n_cells": len(cand), "pass_C1": len(c1), "pass_C1_C3": len(c13), "top_k": [r["cell"] for r in top]}
    # stage 2: validation audit WAI for recorded + top-K; C2 WAI <= recorded; tie-breaks mismatch, WAI, more L1, fewer extra generations, lower sim
    def vwai(cell):
        rows = []
        for d in VA:
            q = d["query_id"]; f = LS(d["query"], d["gold"], VF[q]["key"]); a = LS(d["query"], d["gold"], SC["validation"][cell]["audit"][q]["key"])
            if f is not None and a is not None: rows.append({"fresh": f, "arm": a})
        w = sum(1 for r in rows if r["fresh"] == "CORRECT" and r["arm"] == "WRONG"); fc = sum(1 for r in rows if r["fresh"] == "CORRECT")
        return {"wai": w, "n": len(rows), "correct": sum(1 for r in rows if r["arm"] == "CORRECT"), "fresh_correct": fc, "conditional_wai_pct": round(100 * w / fc, 4) if fc else None}
    wai = {c: vwai(c) for c in ["recorded"] + st["top_k"]}; st["validation_wai"] = wai
    kept = [r for r in top if wai[r["cell"]]["wai"] <= wai["recorded"]["wai"]]
    simf = lambda c: float(c.split("_")[0][1:])
    kept.sort(key=lambda r: (r["mismatch_ties_diff_pct"], wai[r["cell"]]["wai"], -r["l1_hits"], r["generations"] - pub["generations"], simf(r["cell"])))
    st["frozen"] = kept[0]["cell"] if kept else None; st["recorded_validation"] = pub
    if not c13: st["frozen"] = None; st["note"] = "no cell satisfies C1 and C3; constraints are not relaxed (pre-registered)"
    out["selection"] = st
    return out
for eng in ("keyset", "reported"): RES[f"stricter_{eng}"] = stricter(eng)
L = RES["stricter_keyset"]; SUMM = json.load(open(LG/"summary.json")); S1 = json.load(open(LG/"stage1_selection.json"))
for arm, key in (("recorded", "recorded"), ("strict090", "strict090"), ("precision_frozen", "precision")):
    c = L["heldout"][arm]["census"]; pp = SUMM["populations"][key]
    gate(f"stricter heldout census {arm} (hits/ties/det/mm_tD/mm_noT)", [c["l1_hits"], c["jury_ties"], c["jury_determinate"], c["mismatch_ties_diff_pct"], c["mismatch_no_ties_pct"]],
         [pp["l1_hits"], pp["jury_ties"], pp["jury_determinate"], pp["mismatch_ties_diff"]["pct"], pp["mismatch_no_ties"]["pct"]])
    a = L["heldout"][arm]["audit"]; pa = SUMM["audit"][key]; gate(f"stricter heldout audit {arm} (correct/wai)", [a["correct"], a["wai"]], [pa["correct"], pa["wai"]])
gate("stricter stage1 top-K", L["selection"]["top_k"], [r["cell"] for r in S1["top_k"]] if "top_k" in S1 else None)
gate("stricter frozen cell", L["selection"]["frozen"], SUMM["frozen"]["cell"])
gate("stricter validation WAI recorded", L["selection"]["validation_wai"]["recorded"]["wai"], SUMM["frozen"]["recorded_validation_wai"]["wai"])
# ============ 5. held-out L1 hit-level analysis (every realised held-out L1 hit; 2-judge; 4-juror jury) ============
def hitlevel(eng):
    T = [json.loads(l) for l in open(OUTPUT_ROOT / "tier_quality/inputs/l1_reuse_hits.jsonl")] if eng == "keyset" else json.load(open(W/"results/tier_cpu_reported.json"))["l1_reuse"]
    LHt = Labeler([A2/"answers.jsonl", ((OUTPUT_ROOT / "l1only_exactttl/l1_hit_analysis/l1_answers.jsonl"))], [((OUTPUT_ROOT / "l1only_exactttl/l1_hit_analysis/l1_judge_{j}.jsonl")), A2/"judge_{j}.jsonl"], HELD)
    rows = []
    for x in T:
        v = x.get("equivalence_4juror") if eng == "keyset" else verdict(x["pair_id"])
        rows.append({"qid": x["query_id"], "eq": v, "stored": LHt(x["incoming_query"], x["reference_answer"], x["paths"]["A_stored_answer"]["key"]) if x["reference_answer"] else None,
                     "fresh": LHt(x["incoming_query"], x["reference_answer"], x["paths"]["C_fresh"]["key"]) if x["reference_answer"] else None})
    c = collections.Counter(r["eq"] for r in rows); det = [r for r in rows if r["stored"] is not None and r["fresh"] is not None]
    w = sum(1 for r in det if r["fresh"] == "CORRECT" and r["stored"] == "WRONG")
    return {"hits": len(rows), "jury": dict(c), "mismatch_no_ties_pct": round(100 * c["DIFFERENT"] / (c["DIFFERENT"] + c["SAME"]), 4) if (c["DIFFERENT"] + c["SAME"]) else None,
            "answer_determinate": len(det), "stored_correct": sum(1 for r in det if r["stored"] == "CORRECT"), "fresh_correct": sum(1 for r in det if r["fresh"] == "CORRECT"), "wai": w,
            "wai_by_equivalence": {e: sum(1 for r in det if r["eq"] == e and r["fresh"] == "CORRECT" and r["stored"] == "WRONG") for e in ("SAME", "DIFFERENT", "JURY_TIE")}}
for eng in ("keyset", "reported"): RES[f"l1_hits_{eng}"] = hitlevel(eng)
json.dump(ROWS, open(W/"results/answer_quality_rows.json", "w"))
json.dump({"results": RES, "keyset_gates": GATES, "pool_fills": {"heldout": dict(LH.filled), "primary": dict(LP.filled), "budget": dict(LB.filled), "stricter": dict(LS.filled)}},
          open(W/"results/answer_quality.json", "w"), indent=1)
for k, v in GATES.items(): print({True: "PASS ", False: "FAIL ", None: "RECORDED "}[v["pass"]] + k, v["got"], "reference", v["recorded"])
