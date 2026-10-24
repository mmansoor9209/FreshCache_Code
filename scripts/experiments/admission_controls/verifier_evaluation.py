"""Optional L2 verifier (gamma frozen at 0.55 on validation; not retuned): operational replay on the replay engine
and the answer / sufficiency audit (original audit analysis transcribed). Keyset gate: the transcription applied to the
recorded audit_sample.jsonl and recorded judgments must reproduce the recorded audit_results.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, math, collections, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import answer_pools as P
R = P.R; W = ((OUTPUT_ROOT / "replay")); OV = OUTPUT_ROOT / "l2_verifier/out"; A2 = OUTPUT_ROOT / "general_audit"; GO = W/"gpu/out"
h, jl = P.h16, P.jl
ABST = ("i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information")
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def wil(k, n, z=1.96):
    if not n: return [None, None]
    p, d = k / n, 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0, (c - m) / d), 3), round(100 * min(1, (c + m) / d), 3)]
GA = P.AnswerLookup(extra=[GO/"answers_llama3b.jsonl"]); GJ = P.judge2(extra=[GO])
def analyze(S, pool_fill, sl, sq):
    ans = {tuple(r["key"]): r["answer"] for r in jl(A2/"answers.jsonl")}
    for r in jl(OV/"audit_answers.jsonl"): ans[tuple(r["key"])] = r["answer"]
    j = {}
    for name in ("llama8b", "qwen7b"):
        j[name] = {tuple(r["sig"]): r["label"] for r in jl(A2/f"judge_{name}.jsonl")}
        for r in jl(OV/f"audit_judge_{name}.jsonl"): j[name][tuple(r["sig"])] = r["label"]
    def outc(d, key):
        v = ans.get(tuple(key))
        if v is None and pool_fill: v = GA.get(key)
        if v is None: return None
        if any(p in v.strip().lower() for p in ABST): return "ABSTAIN"
        sig = (h(d["query"]), h(d["gold"]), h(v)); a, b = j["llama8b"].get(sig), j["qwen7b"].get(sig)
        if pool_fill: a = a if a is not None else GJ["llama8b"].get(sig); b = b if b is not None else GJ["qwen7b"].get(sig)
        if a is None or b is None: return None
        return "CORRECT" if (a == "CORRECT" and b == "CORRECT") else "WRONG"
    rows = []
    for d in S:
        r = {"qid": d["query_id"], "fresh": outc(d, d["fresh_key"]), "fc": outc(d, d["fc_key"]), "lv": outc(d, d["lv_key"]), "changed": 1 - d["serving_unchanged"]}
        if any(v is None for v in (r["fresh"], r["fc"], r["lv"])): continue
        rows.append(r)
    N = len(rows); FCc = sum(1 for r in rows if r["fresh"] == "CORRECT"); res = {"N": N, "fresh_correct": FCc}
    for a in ("fc", "lv"):
        c = collections.Counter(r[a] for r in rows); w = sum(1 for r in rows if r[a] == "WRONG" and r["fresh"] == "CORRECT")
        res[a] = {"correct": c["CORRECT"], "abstain": c["ABSTAIN"], "accuracy_pct": round(100 * c["CORRECT"] / N, 3), "accuracy_ci95": wil(c["CORRECT"], N), "wai": w, "wai_pct": round(100 * w / N, 3),
                  "wai_ci95": wil(w, N), "conditional_wai_pct": round(100 * w / FCc, 3) if FCc else None, "conditional_wai_ci95": wil(w, FCc)}
    wa = lambda r, x: r[x] == "WRONG" and r["fresh"] == "CORRECT"
    res["paired_wai"] = {"both": sum(1 for r in rows if wa(r, "fc") and wa(r, "lv")), "fc_only": sum(1 for r in rows if wa(r, "fc") and not wa(r, "lv")), "lv_only": sum(1 for r in rows if wa(r, "lv") and not wa(r, "fc"))}
    res["paired_wai"]["exact_mcnemar_p"] = mcn(res["paired_wai"]["fc_only"], res["paired_wai"]["lv_only"])
    cw = sum(1 for r in rows if r["fc"] == "CORRECT" and r["lv"] != "CORRECT"); wc = sum(1 for r in rows if r["fc"] != "CORRECT" and r["lv"] == "CORRECT")
    res["paired_correctness"] = {"fc_only": cw, "lv_only": wc, "exact_mcnemar_p": mcn(cw, wc)}; res["serving_changed"] = sum(r["changed"] for r in rows)
    suff = {}
    for arm in ("fc", "lv"):
        lab = [(sl.get((r["qid"], arm)) if sl.get((r["qid"], arm)) == sq.get((r["qid"], arm)) else "JUDGE_DISAGREEMENT") for r in rows]; c = collections.Counter(lab); ag = c["SUFFICIENT"] + c["INSUFFICIENT"]
        suff[arm] = {"n": len(lab), "SUFFICIENT": c["SUFFICIENT"], "INSUFFICIENT": c["INSUFFICIENT"], "UNCLEAR": c["UNCLEAR"], "JUDGE_DISAGREEMENT": c["JUDGE_DISAGREEMENT"],
                     "sufficient_pct_of_all": round(100 * c["SUFFICIENT"] / len(lab), 3), "ci95_of_all": wil(c["SUFFICIENT"], len(lab)),
                     "sufficient_pct_of_agreed": round(100 * c["SUFFICIENT"] / ag, 3) if ag else None, "ci95_of_agreed": wil(c["SUFFICIENT"], ag)}
    pl = [(sl.get((r["qid"], "fc")) == sq.get((r["qid"], "fc")) and sl.get((r["qid"], "fc")), sl.get((r["qid"], "lv")) == sq.get((r["qid"], "lv")) and sl.get((r["qid"], "lv"))) for r in rows]
    ao = sum(1 for a, b in pl if a == "SUFFICIENT" and b != "SUFFICIENT"); bo = sum(1 for a, b in pl if b == "SUFFICIENT" and a != "SUFFICIENT")
    suff["paired"] = {"fc_sufficient_only": ao, "lv_sufficient_only": bo, "exact_mcnemar_p": mcn(ao, bo)}; res["sufficiency"] = suff
    return res
out = {}
# keyset gate
sl = {(r["query_id"], r["arm"]): r["label"] for r in jl(OV/"audit_suff_llama8b.jsonl")}; sq = {(r["query_id"], r["arm"]): r["label"] for r in jl(OV/"audit_suff_qwen7b.jsonl")}
leg = analyze(jl(OV/"audit_sample.jsonl"), False, sl, sq); pub = json.load(open(OV/"audit_results.json")); out["keyset_audit"] = leg; out["keyset_gate_pass"] = leg == pub
# reported
RU = json.load(open(W/"verifier/audit/suff_reused.json"))
csl = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; csq = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
for r in jl(GO/"suffv_llama8b.jsonl"): csl[(r["query_id"], r["arm"])] = r["label"]
for r in jl(GO/"suffv_qwen7b.jsonl"): csq[(r["query_id"], r["arm"])] = r["label"]
out["reported_audit"] = analyze(jl(W/"verifier/audit/audit_sample.jsonl"), True, csl, csq)
out["operational"] = {"recorded": json.load(open(OV/"test_operational.json"))["results"], "reported": json.load(open(W/"verifier/out/test_operational.json"))["results"]}
json.dump(out, open(W/"results/verifier_recompute.json", "w"), indent=1)
print("keyset gate", out["keyset_gate_pass"]); print(json.dumps(out["reported_audit"])[:1500])
