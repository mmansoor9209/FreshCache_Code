"""Generator-replication analysis (Llama-3.1-8B generator): labels, metrics, paired tests, per-request outputs, model/prompt hashes,
and a comparison with the earlier 8B and 3B results (results/old_vs_new.json). Reads only files listed in config/hashes.json; writes to
outputs/generator_replication/results/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, csv, re, math, hashlib, collections, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); RV = _ilu.module_from_spec(_rs); _rs.loader.exec_module(RV)
from fractions import Fraction
W = ((OUTPUT_ROOT / "generator_replication")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = W/"outputs"; RS = W/"results"
SG = (OUTPUT_ROOT / "l1only_exactttl/stronger_generator"); CR = (OUTPUT_ROOT / "replay"); MB = (OUTPUT_ROOT / "reference_validation")
def h(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def wil(k, n, z=1.96):
    if not n: return [None, None]
    p, d = k / n, 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0, (c - m) / d), 2), round(100 * min(1, (c + m) / d), 2)]
REQ = json.load(open(W/"inputs/requests.json")); ARMS = ["fresh", "freshcache", "sttl_q060_k1_2"]
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]   # held-out abstention patterns (ABSTAIN_PAT in scripts/evaluation/generator_replication/stronger_generator_audit_analysis.py)
P3 = json.load(open(MB/"config/protocol.json")); ABP = P3["answer_stage"]["abstain_patterns"]
REF = {r["query_id"]: r for r in csv.DictReader(open(MB/"results/reference_validation_results.csv"))}
VAL = {q for q, r in REF.items() if r["final_label"] == "VALID_AT_T"}
XW = json.load(open((OUTPUT_ROOT / "audit_recounts/artifacts/screened_subset_crosswalk.json")))
POPS = {"all_audit": [d["query_id"] for d in REQ], "validated": [d["query_id"] for d in REQ if d["query_id"] in VAL]}
SUPP = {"screened": set(XW["screened"]), "supported": set(XW["supported"]), "intersection": set(XW["intersection"])}
assert RV.check("reference_validation/validated_subset_size", len(POPS["validated"]))
def ans3(raw):   # same parser as scripts/evaluation/reference_validation/reference_validation_reparse.py
    u = raw.upper(); u = re.sub(r"\bINCORRECT\b", "WRONG", u); u = re.sub(r"^(WROONG|WONG)\b", "WRONG", u)
    hits = [(u.find(l), l) for l in ("CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE") if u.find(l) >= 0]; return min(hits)[1] if hits else "UNPARSED"
LAB, ANS, PR = {}, {}, {}
for mode in ("forced", "abstain"):
    A = {}
    for p in (SG/f"answers_llama8b_{mode}.jsonl", CR/f"gpu/out3/answers_llama8b_{mode}.jsonl", O/f"answers_llama8b_{mode}.jsonl"):
        for r in jl(p): A.setdefault(tuple(r["key"]), r["answer"])
    J = {}
    for j in ("qwen7b", "mistral7b"):
        J[j] = {}
        for p in (SG/f"judge_llama8b_{mode}__{j}.jsonl", CR/f"gpu/out3/judge_llama8b_{mode}__{j}.jsonl", O/f"judge_llama8b_{mode}__{j}.jsonl"):
            for r in jl(p): J[j].setdefault(tuple(r["sig"]), r["label"])
    TABLE_ROWS = {j: {r["task_id"]: ans3(r["raw"]) for r in jl(O/f"answer3_{mode}_{j}.jsonl")} for j in ("llama8b", "qwen7b", "mistral7b")}
    def prim(d, a):   # primary label, as label() in scripts/evaluation/generator_replication/stronger_generator_audit_analysis.py
        t = (a or "").strip()
        if t == "INSUFFICIENT_CONTEXT" or any(p in t.lower() for p in HELD): return "ABSTAIN"
        s = (h(d["query"]), h(d["gold"]), h(a)); ls = [J[j].get(s) for j in ("qwen7b", "mistral7b")]
        if any(x is None or x == "UNPARSED" for x in ls): return "UNJUDGED"
        if ls[0] != ls[1]: return "JUDGE_DISAGREEMENT"
        return "CORRECT" if ls[0] == "CORRECT" else "WRONG"
    def third(d, a):   # as in scripts/evaluation/reference_validation/reference_validation_analysis.py: pattern abstention first, then 2-of-3 majority else UNRESOLVABLE
        if any(p in (a or "").strip().lower() for p in ABP): return "ABSTAIN", ["-"] * 3
        tid = hashlib.sha256(f"{d['query_id']}\x00{a}".encode()).hexdigest()[:16]; ls = [TABLE_ROWS[j].get(tid, "MISSING") for j in ("llama8b", "qwen7b", "mistral7b")]
        c = collections.Counter(l for l in ls if l not in ("UNPARSED", "MISSING")).most_common(1)
        return (c[0][0] if c and c[0][1] >= 2 else "UNRESOLVABLE"), ls
    for d in REQ:
        for arm in ARMS:
            a = A[tuple(d[f"{arm}_key"])]; ANS[(mode, d["query_id"], arm)] = a
            LAB[("primary", mode, d["query_id"], arm)] = prim(d, a); t, ls = third(d, a); LAB[("frozen3", mode, d["query_id"], arm)] = t; PR[(mode, d["query_id"], arm)] = ls
def metrics(proto, mode, ids):
    L = lambda q, a: LAB[(proto, mode, q, a)]; n = len(ids); fc = [q for q in ids if L(q, "fresh") == "CORRECT"]; out = {"N": n, "fresh_correct": len(fc)}
    for a in ARMS:
        c = collections.Counter(L(q, a) for q in ids)
        row = {"counts": dict(c), "accuracy_pct": round(100 * c["CORRECT"] / n, 2), "accuracy_ci95": wil(c["CORRECT"], n), "abstention_rate_pct": round(100 * c["ABSTAIN"] / n, 2)}
        if a != "fresh":
            w = sum(1 for q in fc if L(q, a) == "WRONG")
            row.update({"wai": w, "wai_pct": round(100 * w / n, 2), "wai_ci95": wil(w, n), "conditional_wai_pct": round(100 * w / len(fc), 2) if fc else None, "conditional_wai_ci95": wil(w, len(fc)),
                        "fresh_correct_to_abstain": sum(1 for q in fc if L(q, a) == "ABSTAIN"),
                        "fresh_correct_to_undetermined": sum(1 for q in fc if L(q, a) in ("JUDGE_DISAGREEMENT", "UNJUDGED", "UNRESOLVABLE")),
                        "vs_fresh_correct_lost": sum(1 for q in ids if L(q, "fresh") == "CORRECT" and L(q, a) != "CORRECT"),
                        "vs_fresh_correct_gained": sum(1 for q in ids if L(q, "fresh") != "CORRECT" and L(q, a) == "CORRECT")})
            row["vs_fresh_mcnemar_p"] = mcn(row["vs_fresh_correct_lost"], row["vs_fresh_correct_gained"])
        out[a] = row
    f, s = "freshcache", "sttl_q060_k1_2"
    wai = lambda q, a: L(q, "fresh") == "CORRECT" and L(q, a) == "WRONG"; ab = lambda q, a: L(q, "fresh") == "CORRECT" and L(q, a) == "ABSTAIN"
    cf = sum(1 for q in ids if L(q, f) == "CORRECT" and L(q, s) != "CORRECT"); cs = sum(1 for q in ids if L(q, s) == "CORRECT" and L(q, f) != "CORRECT")
    wf = sum(1 for q in ids if wai(q, f) and not wai(q, s)); ws = sum(1 for q in ids if wai(q, s) and not wai(q, f))
    af = sum(1 for q in ids if ab(q, f) and not ab(q, s)); as_ = sum(1 for q in ids if ab(q, s) and not ab(q, f))
    out["paired_freshcache_vs_sttl"] = {"correct_freshcache_only": cf, "correct_sttl_only": cs, "correct_mcnemar_p": mcn(cf, cs),
        "wai_freshcache_only": wf, "wai_sttl_only": ws, "wai_both": sum(1 for q in ids if wai(q, f) and wai(q, s)), "wai_mcnemar_p": mcn(wf, ws),
        "fresh_correct_to_abstain_freshcache_only": af, "fresh_correct_to_abstain_sttl_only": as_, "abstain_loss_mcnemar_p": mcn(af, as_)}
    return out
RES = {proto: {mode: {pop: metrics(proto, mode, ids) for pop, ids in POPS.items()} for mode in ("forced", "abstain")} for proto in ("primary", "frozen3")}
SUP = {proto: {mode: {pop: metrics(proto, mode, [q for q in POPS["all_audit"] if q in ids]) for pop, ids in SUPP.items()} for mode in ("forced", "abstain")} for proto in ("primary", "frozen3")}
json.dump(RES, open(RS/"results.json", "w"), indent=1); json.dump(SUP, open(RS/"supplementary_subsets.json", "w"), indent=1)
pt = []
for proto in RES:
    for mode in RES[proto]:
        for pop, x in list(RES[proto][mode].items()) + [(k, v) for k, v in SUP[proto][mode].items()]:
            pt.append({"protocol": proto, "mode": mode, "population": pop, "contrast": "FreshCache vs SemanticTTL(0.60,1/2)", **x["paired_freshcache_vs_sttl"]})
            for a in ("freshcache", "sttl_q060_k1_2"):
                pt.append({"protocol": proto, "mode": mode, "population": pop, "contrast": f"{a} vs fresh", "correct_lost": x[a]["vs_fresh_correct_lost"], "correct_gained": x[a]["vs_fresh_correct_gained"], "mcnemar_p": x[a]["vs_fresh_mcnemar_p"]})
with open(RS/"paired_tests.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list({k: 1 for r in pt for k in r})); w.writeheader(); w.writerows(pt)
with open(RS/"per_request.csv", "w", newline="", encoding="utf-8") as f:
    cols = ["query_id", "in_validated", "in_screened", "in_supported", "mode", "arm", "tier", "answer_key", "answer", "primary_label", "frozen3_label", "frozen3_llama8b", "frozen3_qwen7b", "frozen3_mistral7b"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for d in REQ:
        q = d["query_id"]
        for mode in ("forced", "abstain"):
            for a in ARMS:
                ls = PR[(mode, q, a)]
                w.writerow({"query_id": q, "in_validated": int(q in VAL), "in_screened": int(q in SUPP["screened"]), "in_supported": int(q in SUPP["supported"]), "mode": mode, "arm": a,
                            "tier": d[f"{a}_tier"], "answer_key": "|".join(d[f"{a}_key"]), "answer": ANS[(mode, q, a)], "primary_label": LAB[("primary", mode, q, a)], "frozen3_label": LAB[("frozen3", mode, q, a)],
                            "frozen3_llama8b": ls[0], "frozen3_qwen7b": ls[1], "frozen3_mistral7b": ls[2]})
# fidelity of reused answers
fid = {m: (lambda r: {"n": len(r), "byte_identical": sum(x["answer"] == x["reused"] for x in r)})(jl(O/f"fidelity_llama8b_{m}.jsonl")) for m in ("forced", "abstain")}
# comparison with the earlier 8B and 3B results
old8 = json.load(open(SG/"audit_analysis.json"))["generators"]; c8 = json.load(open(CR/"results/selection_and_8b.json")); aq = json.load(open(CR/"results/answer_quality.json"))["results"]["heldout_reported"]["all_audit"]
cmpx = {"previous_8b_replication_recorded_engine_drift_selected": {m: {"freshcache_wai": old8[f"llama8b_{m}"]["fc"]["wai"]["all_audit"]["k"], "sttl_1_16_wai": old8[f"llama8b_{m}"]["sttl"]["wai"]["all_audit"]["k"]} for m in ("forced", "abstain")},
        "previous_8b_replication_reported_engine_drift_selected": {m: {"freshcache_wai": c8[f"llama8b_{m}_reported"]["freshcache"]["wai"], "sttl_1_16_wai": c8[f"llama8b_{m}_reported"]["sttl_k1_16"]["wai"]} for m in ("forced", "abstain")},
        "llama3b_reported_quality_selected_2judge": {"freshcache_wai": aq["fc"]["wai"], "sttl_060_wai": aq["sttl_t060_k1_2"]["wai"], "freshcache_acc": aq["fc"]["accuracy_pct"], "sttl_060_acc": aq["sttl_t060_k1_2"]["accuracy_pct"]},
        "this_study_llama8b_quality_selected_primary": {m: {"freshcache_wai": RES["primary"][m]["all_audit"]["freshcache"]["wai"], "sttl_060_wai": RES["primary"][m]["all_audit"]["sttl_q060_k1_2"]["wai"]} for m in ("forced", "abstain")}}
# gate: the fresh and FreshCache arms reproduce the replay-engine 8B replication in selection_and_8b.json (same keys, same answers, same judges)
gate = {m: {"freshcache_wai_equal": RES["primary"][m]["all_audit"]["freshcache"]["wai"] == c8[f"llama8b_{m}_reported"]["freshcache"]["wai"],
            "fresh_correct_equal": RES["primary"][m]["all_audit"]["fresh_correct"] == c8[f"llama8b_{m}_reported"]["fresh_correct"]} for m in ("forced", "abstain")}
json.dump({"comparison": cmpx, "reproduction_gate": gate, "fidelity_regeneration_of_reused_answers": fid}, open(RS/"old_vs_new.json", "w"), indent=1)
# hashes
hs = {"models": {"generator": "meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659", "judge_qwen7b": "Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28",
                 "judge_mistral7b": "mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71", "judge_llama8b_frozen3_only": "meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659"},
      "prompts_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in json.load(open(SG/"prompts_verbatim.json"))["prompts"].items()},
      "frozen3_protocol_sha256": sha(MB/"config/protocol.json"),
      "engine_files_sha256": {p.name: sha(p) for p in sorted(((REPO_ROOT / "freshcache/replay_engine")).glob("*.py"))},
      "scripts_sha256": {p.name: sha(p) for p in sorted((W/"scripts").glob("*.py"))},
      "inputs_sha256": {p.name: sha(p) for p in sorted((W/"inputs").glob("*"))},
      "read_only_sources_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [SG/"answers_llama8b_forced.jsonl", SG/"answers_llama8b_abstain.jsonl", MB/"results/reference_validation_results.csv", (OUTPUT_ROOT / "audit_recounts/artifacts/screened_subset_crosswalk.json"), CR/"results/audit_keys_reported.json"]},
      "decoding": {"generation": "chat template, greedy, max_new_tokens 80, truncation 2048, left padding, batch 16, clip at first '.' or newline", "primary_judging": "JUDGE prompt, chat template, greedy, 6 tokens, batch 16; primary pair qwen7b+mistral7b must agree", "frozen3": "protocol.json answer stage, 2-of-3 majority"},
      "fixed": {"SemanticTTL": {"theta": 0.60, "kappa": 0.5}, "freshcache": "final replay engine, recorded thresholds"}}
json.dump(hs, open(W/"config/hashes.json", "w"), indent=1)
for proto in RES:
    for mode in RES[proto]:
        for pop in RES[proto][mode]:
            x = RES[proto][mode][pop]
            print(proto, mode, pop, "N", x["N"], "freshC", x["fresh_correct"], {a: (x[a]["accuracy_pct"], x[a].get("wai"), x[a].get("conditional_wai_pct"), x[a]["abstention_rate_pct"], x[a].get("fresh_correct_to_abstain")) for a in ARMS}, x["paired_freshcache_vs_sttl"])
print("gate", gate, "fidelity", fid)
