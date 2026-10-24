"""(a) Quality-selection check (quality_selection.py rule, validation only; configuration NOT changed) and (b) Llama-3.1-8B replication
(stronger-generator audit labelling) for keyset (recorded keys; gates) and reported keys."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, csv, collections, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import answer_pools as P
R = P.R; W = ((OUTPUT_ROOT / "replay")); h, jl = P.h16, P.jl
QB = OUTPUT_ROOT / "quality_selection"; SG = OUTPUT_ROOT / "l1only_exactttl/stronger_generator"; O3 = W/"gpu/out3"
PRIM = ("i don't know", "i dont know", "cannot determine", "not enough information", "insufficient", "no information", "unclear from the context")
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
GA = P.AnswerLookup(extra=[W/"gpu/out/answers_llama3b.jsonl", O3/"answers_llama3b.jsonl"]); GJ = P.judge2(extra=[W/"gpu/out", O3])
out = {}
# ---------------- (a) quality selection ----------------
pre = json.load(open((OUTPUT_ROOT / "quality_selection/quality_selection_prereg.json"), encoding="utf-8")); S = [json.loads(l) for l in open(QB/"q_sample.jsonl")]
ans = {tuple(r["key"]): r["answer"] for r in jl(QB/"q_answers.jsonl")}; j = {x: {tuple(r["sig"]): r["label"] for r in jl(QB/f"q_judge_{x}.jsonl")} for x in ("llama8b", "qwen7b")}
EVSET = {r["query_id"] for r in csv.DictReader(open(QB/"q_verification.csv")) if r.get("evidence_valid") == "1"}
arms = ["fc"] + [c["arm"] for c in pre["candidates"]]
def outc(d, key):
    a = ans.get(tuple(key)); a = a if a is not None else GA.get(list(key))
    if a is None: return None
    if any(p in a.strip().lower() for p in PRIM): return "ABSTAIN"
    s = (h(d["query"]), h(d["gold"]), h(a)); l = [j[x].get(s, GJ[x].get(s)) for x in ("llama8b", "qwen7b")]
    if None in l: return None
    return "CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"
def qsel(eng):
    K = {r["query_id"]: r for r in json.load(open(W/f"results/quality_selection_keys_{eng}.json"))["records"]}; rows = []
    for d in S:
        k = K[d["query_id"]]; r = {"qid": d["query_id"], "fresh": outc(d, k["fresh_key"])}
        for a in arms: r[a] = outc(d, k[f"{a}_key"])
        if all(r[x] is not None for x in ["fresh"] + arms): rows.append(r)
    EV = [r for r in rows if r["qid"] in EVSET]; fcn = sum(1 for r in EV if r["fresh"] == "CORRECT")
    st = {a: {"wai": sum(1 for r in EV if r["fresh"] == "CORRECT" and r[a] == "WRONG"), "correct": sum(1 for r in EV if r[a] == "CORRECT")} for a in arms}
    for a in arms: st[a]["conditional_wai_pct"] = round(100 * st[a]["wai"] / fcn, 3)
    TH = st["fc"]["conditional_wai_pct"]
    cands = [{"arm": c["arm"], "theta": c["theta"], "kappa": c["kappa"], "saved": c["saved"], **st[c["arm"]], "primary": st[c["arm"]]["conditional_wai_pct"] <= TH, "relaxed_5pp": st[c["arm"]]["conditional_wai_pct"] <= TH + 5.0} for c in pre["candidates"]]
    return {"scored": len(rows), "evidence_valid": len(EV), "fresh_correct": fcn, "freshcache": st["fc"], "primary_limit_pct": TH, "relaxed_limit_pct": round(TH + 5, 3),
            "candidates": cands, "pass_primary": [c["arm"] for c in cands if c["primary"]], "pass_relaxed": [c["arm"] for c in cands if c["relaxed_5pp"]]}
for e in ("keyset", "reported"): out[f"quality_selection_{e}"] = qsel(e)
L = out["quality_selection_keyset"]; pub = json.load(open(QB/"quality_selection_results.json"))
out["quality_selection_keyset_gate"] = {"fc_wai_fresh_correct_screened_n": {n: REF.record("validation_quality/" + n, v) for n, v in (("keyset_rule/FreshCache_WAI", L["freshcache"]["wai"]), ("fresh_correct", L["fresh_correct"]), ("N", L["evidence_valid"]))}, "pass_relaxed_only_060_1_2": L["pass_relaxed"] == ["sttl_t0.60_k1_2"],
                                        "wai_per_candidate": {c["arm"]: c["wai"] for c in L["candidates"]}}
# ---------------- (b) Llama-3.1-8B replication ----------------
AUDIT_SAMPLE = json.load(open(OUTPUT_ROOT / "general_audit/sample.json"))["requests"]
def is_abst8(t): t = (t or "").strip(); return t == "INSUFFICIENT_CONTEXT" or any(p in t.lower() for p in HELD)
def rep8(eng, mode):
    K = json.load(open(W/f"results/audit_keys_{eng}.json"))["keys"]["heldout"]
    A8 = {tuple(r["key"]): r["answer"] for r in jl(SG/f"answers_llama8b_{mode}.jsonl")}
    for r in jl(O3/f"answers_llama8b_{mode}.jsonl"): A8.setdefault(tuple(r["key"]), r["answer"])
    J8 = {x: {tuple(r["sig"]): r["label"] for r in jl(SG/f"judge_llama8b_{mode}__{x}.jsonl")} for x in ("qwen7b", "mistral7b")}
    for x in J8:
        for r in jl(O3/f"judge_llama8b_{mode}__{x}.jsonl"): J8[x].setdefault(tuple(r["sig"]), r["label"])
    def lab(d, key):
        a = A8.get(tuple(key))
        if a is None: return None
        if is_abst8(a): return "ABSTAIN"
        s = (h(d["query"]), h(d["gold"]), h(a)); ls = [J8[x].get(s) for x in ("qwen7b", "mistral7b")]
        if any(v is None or v == "UNPARSED" for v in ls): return "UNJUDGED"
        if ls[0] != ls[1]: return "JUDGE_DISAGREEMENT"
        return "CORRECT" if ls[0] == "CORRECT" else "WRONG"
    L = {arm: {d["query_id"]: lab(d, K[d["query_id"]][arm][0]) for d in AUDIT_SAMPLE} for arm in ("fresh", "freshcache", "sttl_k1_16")}
    fc = [q for q in L["fresh"] if L["fresh"][q] == "CORRECT"]
    return {arm: {"counts": dict(collections.Counter(L[arm].values())), **({"wai": sum(1 for q in fc if L[arm][q] == "WRONG")} if arm != "fresh" else {})} for arm in L} | {"fresh_correct": len(fc),
            "paired_wai": {"fc_only": sum(1 for q in fc if L["freshcache"][q] == "WRONG" and L["sttl_k1_16"][q] != "WRONG"), "sttl_only": sum(1 for q in fc if L["sttl_k1_16"][q] == "WRONG" and L["freshcache"][q] != "WRONG")}}
for e in ("keyset", "reported"):
    for m in ("forced", "abstain"): out[f"llama8b_{m}_{e}"] = rep8(e, m)
out["llama8b_keyset_gate"] = {"forced_fc_sttl": REF.record("validation_quality/llama8b_forced_keyset/FreshCache_SemanticTTL_k1_16_WAI", [out["llama8b_forced_keyset"]["freshcache"]["wai"], out["llama8b_forced_keyset"]["sttl_k1_16"]["wai"]]),
                              "abstain_fc_sttl": REF.record("validation_quality/llama8b_abstain_keyset/FreshCache_SemanticTTL_k1_16_WAI", [out["llama8b_abstain_keyset"]["freshcache"]["wai"], out["llama8b_abstain_keyset"]["sttl_k1_16"]["wai"]])}
json.dump(out, open(W/"results/selection_and_8b.json", "w"), indent=1)
for k, v in out.items(): print(k, json.dumps(v)[:700])
