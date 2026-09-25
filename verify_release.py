#!/usr/bin/env python3
"""Verify the FreshCache release: file hashes against the frozen manifest, and a set of
headline numbers re-derived from row-level records. Standard library only; no network, no GPU."""
import csv, hashlib, json, pathlib, sys, collections, statistics
ROOT = pathlib.Path(__file__).resolve().parent
ok = True
def check(name, got, want):
    global ok
    good = got == want; ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got!r}, expected {want!r}")
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def h16(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

print("== 1. frozen-manifest hashes (docs/release/MANIFEST.json) ==")
m = json.load(open(ROOT/"docs/release/MANIFEST.json"))["files"]
present = matched = missing = 0
for rel, meta in m.items():
    p = ROOT/rel
    if not p.exists(): missing += 1; continue
    present += 1
    sha = meta["sha256"] if isinstance(meta, dict) else meta
    if hashlib.sha256(p.read_bytes()).hexdigest() == sha: matched += 1
print(f"  manifest entries {len(m)}: present {present}, hash-identical {matched}, missing {missing}")
check("all present files hash-identical", matched, present)

print("== 2. held-out answer audit: WAI 7/400 (artifacts/audits/answer_audit_k1_16) ==")
A = ROOT/"artifacts/audits/answer_audit_k1_16"
S = json.load(open(A/"sample.json"))["requests"]
ans = {tuple(r["key"]): r["answer"] for r in jl(A/"answers.jsonl")}
J = {j: {tuple(r["sig"]): r["label"] for r in jl(A/f"judge_{j}.jsonl")} for j in ("llama8b","qwen7b")}
ABST = ("i don't know","i do not know","cannot be determined","not enough information","insufficient","unable to determine","no information")
def outc(d, key):
    a = ans.get(tuple(key))
    if a is None: return None
    if any(p in a.strip().lower() for p in ABST): return "ABSTAIN"
    s = (h16(d["query"]), h16(d["gold"]), h16(a)); l1, l2 = J["llama8b"].get(s), J["qwen7b"].get(s)
    if l1 is None or l2 is None: return None
    return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"
rows = [(outc(d, d["fresh_key"]), outc(d, d["fc_key"]), outc(d, d["sttl_key"])) for d in S]
check("requests", len(rows), 400)
check("fresh correct", sum(1 for f, _, _ in rows if f == "CORRECT"), 81)
check("FreshCache WAI", sum(1 for f, c, _ in rows if f == "CORRECT" and c == "WRONG"), 7)
check("SemanticTTL (k=1/16) WAI", sum(1 for f, _, s in rows if f == "CORRECT" and s == "WRONG"), 64)

print("== 3. stricter L1 gate: complete-hit mismatch (artifacts/validation_V3/l1_precision_gate) ==")
J2 = list(csv.DictReader(open(ROOT/"artifacts/validation_V3/l1_precision_gate/heldout_l1_hits_judged.csv", encoding="utf-8")))
for arm, want in (("published", (806, 43.30)), ("strict090", (450, 18.22)), ("precision", (314, 14.97))):
    P = [r for r in J2 if r["arm"] == arm]; c = collections.Counter(r["equivalence"] for r in P)
    check(f"{arm} hits", len(P), want[0]); check(f"{arm} mismatch % (ties->DIFFERENT)", round(100*(c["DIFFERENT"]+c["JURY_TIE"])/len(P), 2), want[1])

print("== 4. validation budget selection (artifacts/validation_V3/parameter_calibration/04_budgets) ==")
sw = json.load(open(ROOT/"artifacts/validation_V3/parameter_calibration/04_budgets/validation_sweep.json"))
pub = sw["published_validation"]; C1, C2, C3 = pub["drift_pct"] + 0.25, pub["l1_hits"], 0.95*pub["coverage_pct"]
feas = [v for v in sw["cells"].values() if v["drift_pct"] <= C1 and v["l1_hits"] <= C2 and v["coverage_pct"] >= C3]
sel = sorted(feas, key=lambda v: (-v["search_saved_pct"], v["drift_pct"], v["l1_hits"], v["eps_answer"], v["eps_url_list"], v["eps_content"]))[0]
check("grid cells", len(sw["cells"]), 48); check("eligible triples", len(feas), 10)
check("selected (eps_L1, eps_L2, eps_L3)", (sel["eps_answer"], sel["eps_url_list"], sel["eps_content"]), (0.1, 0.35, 0.25))
check("selected validation savings %", sel["search_saved_pct"], 67.4864)

print("== 5. optional L2 verifier (artifacts/verifier) ==")
g = json.load(open(ROOT/"artifacts/verifier/gamma_selection.json"))
check("gamma", g["selected_gamma"], 0.55); check("usable labels", g["usable_labels"], 463); check("AUC", g["auc"], 0.7787)
sc = collections.Counter(r["verdict"] for r in jl(ROOT/"artifacts/verifier/study/out/test_scores_L2Verify_g0.55.jsonl"))
check("scored L2 hits", sc["ACCEPT"] + sc["REJECT"], 13093); check("rejected", sc["REJECT"], 6261); check("accepted unscored", sc["ACCEPT_UNSCORED"], 278)
to = json.load(open(ROOT/"artifacts/verifier/test_operational.json"))["results"]
check("search savings FreshCache / L2Verify", (to["FreshCache"]["search_saved_pct"], to["L2Verify_g0.55"]["search_saved_pct"]), (60.5776, 36.2321))
sl = {(r["query_id"], r["arm"]): r["label"] for r in jl(ROOT/"artifacts/verifier/study/out/audit_suff_llama8b.jsonl")}
sq = {(r["query_id"], r["arm"]): r["label"] for r in jl(ROOT/"artifacts/verifier/study/out/audit_suff_qwen7b.jsonl")}
suff = {arm: sum(1 for (q, a), l in sl.items() if a == arm and l == "SUFFICIENT" and sq.get((q, a)) == "SUFFICIENT") for arm in ("fc", "lv")}
check("joint-sufficient FreshCache / verifier", (suff["fc"], suff["lv"]), (129, 145))

print("== 6. replay-routed timing, 40 requests (artifacts/latency/direct_40) ==")
T = list(csv.DictReader(open(ROOT/"artifacts/latency/direct_40/direct_per_request.csv", encoding="utf-8")))
def rank_q(v, p): s = sorted(v); return s[min(len(s)-1, int(round(p*(len(s)-1))))]
for pol, want in (("NoCache", (3255.3, 3014.4, 5050.3)), ("FreshCache", (1214.4, 563.8, 3107.3))):
    v = [float(r["total_ms"]) for r in T if r["policy"] == pol]
    check(f"{pol} n", len(v), 40); check(f"{pol} mean/rank-p50/rank-p95", (round(statistics.mean(v), 1), round(rank_q(v, .5), 1), round(rank_q(v, .95), 1)), want)

print("== 7. reference crosswalk (artifacts/manuscript/crosswalk_57_107.json) ==")
cw = json.load(open(ROOT/"artifacts/manuscript/crosswalk_57_107.json"))
check("57 / 107 / 41", (len(cw["supported_57"]), len(cw["screened_107"]), len(cw["intersection"])), (57, 107, 41))
check("intersection is the set intersection", sorted(set(cw["supported_57"]) & set(cw["screened_107"])), cw["intersection"])

print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
