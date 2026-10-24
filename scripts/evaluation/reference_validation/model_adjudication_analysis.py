"""Extended analysis of the reference validation under the frozen rules of scripts/evaluation/reference_validation/reference_validation_analysis.py.
Pattern abstention first, then the 2-of-3 majority (else UNRESOLVED). Adds, for the principal comparisons (FreshCache vs CategoryAware-style;
FreshCache vs SemanticTTL quality-matched 0.80/1-4 and fixed 0.60/1-2): unanimous-rule and per-judge results, Fleiss' kappa, pairwise agreement,
three reference subsets (all evaluable / majority VALID_AT_T / unanimous VALID_AT_T) and a deterministic normalized match for structured
references (dates, numbers, percentages, prices, scores, short named entities). Model majorities are NOT ground truth. Reads only this folder
(+ frozen inputs it copied)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, hashlib, json, math, pathlib, re; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from fractions import Fraction
O = ((OUTPUT_ROOT / "model_adjudication")); P = json.load(open((REPO_ROOT / "configs/protocols/reference_validity_protocol.json"))); J = list(P["judges"])
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
REQ = {r["query_id"]: r for r in jl(O / "inputs/requests.jsonl")}; EV = {r["query_id"]: r for r in jl(O / "inputs/evidence.jsonl")}; OUTS = jl(O / "inputs/policy_outputs.jsonl")
ARMS = sorted({o["arm"] for o in OUTS}); ABST = P["answer_stage"]["abstain_patterns"]
def mcn(b, c):
    n = b + c; return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def wilson(k, n, z=1.96):
    if not n: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [round(100 * (c - m) / d, 2), round(100 * (c + m) / d, 2)]
def fleiss(rows, cats):
    rows = [r for r in rows if all(x in cats for x in r)]
    if not rows: return None
    n, N = len(rows[0]), len(rows); p = {c: sum(r.count(c) for r in rows) / (N * n) for c in cats}
    Pi = [(sum(r.count(c) ** 2 for c in cats) - n) / (n * (n - 1)) for r in rows]; Pb, Pe = sum(Pi) / N, sum(v * v for v in p.values())
    return round((Pb - Pe) / (1 - Pe), 4) if Pe < 1 else None
# ---------------- reference stage (3 families, frozen)
RR = {j: {r["task_id"]: r["label"] for r in jl(O / "outputs/raw" / f"reference_{j}.jsonl")} for j in J}
REF = {}
for q in REQ:
    used = [p for p in EV[q]["pages"] if p.get("used")]
    labs = [RR[j].get(q, "MISSING") for j in J] if used else ["UNVERIFIABLE"] * 3
    c = collections.Counter(l for l in labs if l != "UNPARSED").most_common(1); maj = c[0][0] if c and c[0][1] >= 2 else "NO_MAJORITY"
    REF[q] = {"labels": labs, "majority": maj, "unanimous": len(set(labs)) == 1}
SUB = {"all_evaluable": sorted(REQ), "majority_valid": sorted(q for q in REQ if REF[q]["majority"] == "VALID_AT_T"), "unanimous_valid": sorted(q for q in REQ if REF[q]["unanimous"] and REF[q]["labels"][0] == "VALID_AT_T")}
# ---------------- answer stage
def ans3(raw):   # scripts/evaluation/answer_audits/reference_and_tier_recomputation.py ans3, verbatim, for raw rows without a parsed label
    u = raw.upper(); u = re.sub(r"\bINCORRECT\b", "WRONG", u); u = re.sub(r"^(WROONG|WONG)\b", "WRONG", u)
    hits = [(u.find(l), l) for l in ("CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE") if u.find(l) >= 0]; return min(hits)[1] if hits else "UNPARSED"
AR = {j: {r["task_id"]: (r.get("label") or ans3(r["raw"])) for r in jl(O / "outputs/raw" / f"answer_{j}.jsonl")} for j in J}
tid = lambda o: hashlib.sha256(f"{o['query_id']}\x00{o['answer']}".encode()).hexdigest()[:16]
LAB = {"majority": {}, "unanimous": {}, **{f"judge_{j}": {} for j in J}}; ANS = {}; MISSING = 0
for o in OUTS:
    t = tid(o); labs = [AR[j].get(t, "MISSING") for j in J]; MISSING += "MISSING" in labs; pat = any(p in (o["answer"] or "").strip().lower() for p in ABST); k = (o["query_id"], o["arm"]); ANS[k] = o["answer"]
    c = collections.Counter(l for l in labs if l not in ("UNPARSED", "MISSING")).most_common(1)
    LAB["majority"][k] = "ABSTAIN" if pat else (c[0][0] if c and c[0][1] >= 2 else "UNRESOLVED")
    LAB["unanimous"][k] = "ABSTAIN" if pat else (labs[0] if len(set(labs)) == 1 and labs[0] not in ("UNPARSED", "MISSING") else "UNRESOLVED")
    for j, l in zip(J, labs): LAB[f"judge_{j}"][k] = "ABSTAIN" if pat else ("UNRESOLVED" if l in ("UNPARSED", "MISSING", "UNRESOLVABLE") else l)
for k in list(LAB):
    for kk, v in LAB[k].items():
        if v == "UNRESOLVABLE": LAB[k][kk] = "UNRESOLVED"
# ---------------- deterministic structured check
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
def kind(ref):
    r = ref.strip().lower()
    if re.search(r"\b(1[89]|20)\d{2}\b", r) and (any(m in r for m in MONTHS) or re.search(r"\d{1,2}[/-]\d{1,2}", r)): return "date"
    if re.search(r"\d+(\.\d+)?\s*%", r): return "percentage"
    if re.search(r"[$€£¥₩]\s*\d|\d\s*(usd|eur|dollars?|won|yen)\b", r): return "price"
    if re.fullmatch(r"\s*\d{1,3}\s*[-–:]\s*\d{1,3}\s*", r): return "score"
    if re.fullmatch(r"[\s\d.,]+", r) and re.search(r"\d", r): return "number"
    if len(r.split()) <= 4 and not re.search(r"\d", r): return "short_entity"
    return None
def norm_nums(s): return {re.sub(r"[,\s]", "", x).rstrip(".0") or "0" for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}
def norm_text(s): return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\b(the|a|an)\b", " ", s.lower())).split()
def det(ref, cand):
    k = kind(ref)
    if k is None: return None, None
    c = (cand or "").lower()
    if any(p in c for p in ABST): return k, "ABSTAIN"
    if k in ("number", "percentage", "price", "score"):
        rn = norm_nums(ref); return k, "CORRECT" if rn and rn <= norm_nums(c) else "WRONG"
    if k == "date":
        yrs = set(re.findall(r"\b(?:1[89]|20)\d{2}\b", ref)); mons = {m for m in MONTHS if m in ref.lower()}; days = set(re.findall(r"\b([0-3]?\d)\b", re.sub(r"\b(?:1[89]|20)\d{2}\b", "", ref)))
        ok = yrs <= set(re.findall(r"\b(?:1[89]|20)\d{2}\b", c)) and mons <= {m for m in MONTHS if m in c} and (not days or days & set(re.findall(r"\b([0-3]?\d)\b", c)))
        return k, "CORRECT" if ok else "WRONG"
    rt = [w for w in norm_text(ref) if len(w) > 1]; ct = set(norm_text(c)); return k, "CORRECT" if rt and all(w in ct for w in rt) else "WRONG"
FRESH_ARM = "fresh"
PAIRS = [("freshcache", "categoryaware_style"), ("freshcache", "semanticttl_qualitymatched_0.80_1-4"), ("freshcache", "semanticttl_quality_0.60_1-2")] + \
        [("freshcache", a) for a in ARMS if a.startswith("op_")]
def summarize(rule, ids):
    L = LAB[rule]; fc = [q for q in ids if L.get((q, FRESH_ARM)) == "CORRECT"]; out = {"N": len(ids), "fresh_correct": len(fc)}
    for a in ARMS:
        c = collections.Counter(L.get((q, a)) for q in ids); w = sum(1 for q in fc if L.get((q, a)) == "WRONG")
        out[a] = {"correct": c["CORRECT"], "wrong": c["WRONG"], "abstain": c["ABSTAIN"], "unresolved": c["UNRESOLVED"], "accuracy_pct": round(100 * c["CORRECT"] / len(ids), 2) if ids else None,
                  "WAI": w, "C_WAI_pct": round(100 * w / len(fc), 2) if fc else None, "C_WAI_ci95": wilson(w, len(fc))}
    out["paired"] = {}
    for a, b in PAIRS:
        if (ids and (ids[0], b) not in L): continue
        wa = sum(1 for q in fc if L.get((q, a)) == "WRONG" and L.get((q, b)) != "WRONG"); wb = sum(1 for q in fc if L.get((q, b)) == "WRONG" and L.get((q, a)) != "WRONG")
        ca = sum(1 for q in ids if L.get((q, a)) == "CORRECT" and L.get((q, b)) != "CORRECT"); cb = sum(1 for q in ids if L.get((q, b)) == "CORRECT" and L.get((q, a)) != "CORRECT")
        out["paired"][f"{a} vs {b}"] = {"WAI_first_only": wa, "WAI_second_only": wb, "WAI_p": mcn(wa, wb), "correct_first_only": ca, "correct_second_only": cb, "correct_p": mcn(ca, cb),
                                       "WAI_direction": "first lower" if wa < wb else ("second lower" if wb < wa else "tie")}
    return out
RES = {rule: {s: summarize(rule, ids) for s, ids in SUB.items()} for rule in LAB}
uniq = {tid(o): o for o in OUTS}
AGREE = {"answer_fleiss_kappa": fleiss([[AR[j].get(t, "MISSING") for j in J] for t in uniq], P["answer_stage"]["labels"]),
         "answer_pairwise_agreement_pct": {f"{a}|{b}": round(100 * sum(1 for t in uniq if AR[a].get(t) == AR[b].get(t)) / len(uniq), 2) for i, a in enumerate(J) for b in J[i + 1:]},
         "answer_unresolved_majority": sum(1 for v in LAB["majority"].values() if v == "UNRESOLVED"), "answer_missing_judgments": MISSING,
         "reference_fleiss_kappa": fleiss([REF[q]["labels"] for q in REQ], P["reference_stage"]["labels"]), "reference_majority": dict(collections.Counter(REF[q]["majority"] for q in REQ)),
         "reference_unanimous": sum(REF[q]["unanimous"] for q in REQ), "subset_sizes": {s: len(v) for s, v in SUB.items()}}
# deterministic vs model majority
DT = collections.Counter(); DTARM = collections.defaultdict(collections.Counter); DROWS = []
for (q, a), ans in ANS.items():
    k, d = det(REQ[q]["reference_answer"], ans)
    if k is None: continue
    m = LAB["majority"][(q, a)]; DT[(d, m)] += 1; DTARM[a][d] += 1; DROWS.append({"query_id": q, "arm": a, "ref_kind": k, "deterministic": d, "model_majority": m})
fresh_det = {r["query_id"]: r["deterministic"] for r in DROWS if r["arm"] == FRESH_ARM}
DET = {"structured_requests": len(fresh_det), "kinds": dict(collections.Counter(r["ref_kind"] for r in DROWS if r["arm"] == FRESH_ARM)),
       "agreement_with_model_majority_pct": round(100 * sum(v for (d, m), v in DT.items() if d == m) / sum(DT.values()), 2) if DT else None,
       "confusion_deterministic_vs_majority": {f"{d}|{m}": v for (d, m), v in sorted(DT.items())}, "per_arm": {a: dict(c) for a, c in DTARM.items()}}
DET["paired"] = {}
fcq = [q for q, d in fresh_det.items() if d == "CORRECT"]; DL = {(r["query_id"], r["arm"]): r["deterministic"] for r in DROWS}
for a, b in PAIRS:
    wa = sum(1 for q in fcq if DL.get((q, a)) == "WRONG" and DL.get((q, b)) != "WRONG"); wb = sum(1 for q in fcq if DL.get((q, b)) == "WRONG" and DL.get((q, a)) != "WRONG")
    DET["paired"][f"{a} vs {b}"] = {"fresh_correct_structured": len(fcq), "WAI_first_only": wa, "WAI_second_only": wb, "WAI_p": mcn(wa, wb)}
ROB = {}
for pair in [f"{a} vs {b}" for a, b in PAIRS]:
    dirs = {f"{rule}/{s}": RES[rule][s]["paired"].get(pair, {}).get("WAI_direction") for rule in LAB for s in SUB if pair in RES[rule][s]["paired"]}
    ROB[pair] = {"directions": dirs, "all_same_direction": len({d for d in dirs.values() if d}) == 1, "deterministic_direction": ("first lower" if DET["paired"][pair]["WAI_first_only"] < DET["paired"][pair]["WAI_second_only"] else "second lower" if DET["paired"][pair]["WAI_second_only"] < DET["paired"][pair]["WAI_first_only"] else "tie") if pair in DET["paired"] else None}
json.dump({"agreement": AGREE, "results": RES, "deterministic": DET, "ranking_robustness": ROB}, open(O / "results/adjudication_robustness.json", "w"), indent=1)
with open(O / "results/deterministic_rows.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=list(DROWS[0])); w.writeheader(); w.writerows(DROWS)
print(json.dumps({"agreement": AGREE, "robustness": ROB, "det_agreement": DET["agreement_with_model_majority_pct"]}, indent=1)[:4000])
