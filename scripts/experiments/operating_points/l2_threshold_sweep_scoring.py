"""L2 operating points, stage 2. `items`: two-judge items for answers without a judgment in any source. `score`: labels, per-point metrics,
pairing vs. the primary configuration, Pareto sets, figures, CSV/JSON, report. Read-only on existing files."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, glob, hashlib, json, math, os, pathlib, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
from fractions import Fraction
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "l2_operating_points"); CRR = (OUTPUT_ROOT / "replay"); OV = (OUTPUT_ROOT / "l2_verifier/out")
A2 = (OUTPUT_ROOT / "general_audit")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import answer_pools as P   # noqa: E402
LOG = open(O / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
mode = sys.argv[1]; jl, h = P.jl, P.h16
ST = pickle.load(open(O / "work/stage1.pkl", "rb")); POINTS, OPS, AUD = ST["points"], ST["ops"], ST["audit"]
SAMPLE = json.load(open(A2 / "sample.json"))["requests"]; AK = json.load(open(CRR / "results/audit_keys_reported.json"))["keys"]["heldout"]
ABST = ("i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information")
# answers: A2 + OV exact (recorded-audit precedence), then pools (CRR gpu/out), then new answers of this study
ANS = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
for r in jl(OV / "audit_answers.jsonl"): ANS[tuple(r["key"])] = r["answer"]
GA = P.AnswerLookup(extra=[CRR / "gpu/out/answers_llama3b.jsonl", O / "gpu/out/answers_llama3b.jsonl"])
J = {}
for n in ("llama8b", "qwen7b"):
    J[n] = {tuple(r["sig"]): r["label"] for r in jl(A2 / f"judge_{n}.jsonl")}
    for r in jl(OV / f"audit_judge_{n}.jsonl"): J[n][tuple(r["sig"])] = r["label"]
GJ = P.judge2(extra=[CRR / "gpu/out", O / "gpu/out"])
def answer(key):
    v = ANS.get(tuple(key)); return v if v is not None else GA.get(list(key))
def label(d, key, items=None):
    v = answer(key)
    if v is None: return None, None
    if any(p in v.strip().lower() for p in ABST): return "ABSTAIN", v
    sig = (h(d["query"]), h(d["gold"]), h(v)); a, b = J["llama8b"].get(sig, GJ["llama8b"].get(sig)), J["qwen7b"].get(sig, GJ["qwen7b"].get(sig))
    if a is None or b is None:
        if items is not None: items.setdefault(sig, {"sig": list(sig), "q": d["query"], "gold": d["gold"], "cand": v})
        return None, v
    return ("CORRECT" if (a == "CORRECT" and b == "CORRECT") else "WRONG"), v
if mode == "items":
    items, miss = {}, 0
    for pid, A in AUD.items():
        for d in SAMPLE:
            for key in (AK[d["query_id"]]["fresh"][0], A[d["query_id"]]["key"]):
                lb, v = label(d, key, items); miss += v is None
    if miss: log(f"STOP: {miss} answer keys still missing after generation"); sys.exit(1)
    with open(O / "gpu/in/judge2_items.jsonl", "w", encoding="utf-8") as f:
        for t in items.values(): f.write(json.dumps(t, ensure_ascii=False) + "\n")
    log("judge2 items needed:", len(items)); sys.exit(0)
# ---------------- sufficiency labels by (qid, sha(evidence))
SL = {}
def addsuff(tasks, l8, q7):
    for t in tasks:
        a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
        if a is not None and b is not None: SL.setdefault((t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()), (a, b))
addsuff(jl(OV / "audit_suff_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(OV / "audit_suff_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(OV / "audit_suff_qwen7b.jsonl")})
RU = json.load(open(CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
for r in jl(CRR / "gpu/out/suffv_llama8b.jsonl"): cl8[(r["query_id"], r["arm"])] = r["label"]
for r in jl(CRR / "gpu/out/suffv_qwen7b.jsonl"): cq7[(r["query_id"], r["arm"])] = r["label"]
addsuff(jl(CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7)
addsuff(jl(O / "gpu/in/suffv_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(O / "gpu/out/suffv_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(O / "gpu/out/suffv_qwen7b.jsonl")})
def suff(q, sha):
    x = SL.get((q, sha))
    if x is None: return None
    return x[0] if x[0] == x[1] else "JUDGE_DISAGREEMENT"
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
# ---------------- per-request labels
RL, REQ = {}, []
for pid, A in AUD.items():
    for d in SAMPLE:
        q = d["query_id"]; a = A[q]; fr, _ = label(d, AK[q]["fresh"][0]); lb, ans = label(d, a["key"]); sf = suff(q, a["evidence_sha"])
        RL[(pid, q)] = {"fresh": fr, "label": lb if lb is not None else "UNRESOLVED", "suff": sf if sf is not None else "UNRESOLVED", "tier": a["tier"], "sha": a["evidence_sha"], "key": "|".join(a["key"])}
        REQ.append({"point": pid, "request_id": q, "tier": a["tier"], "verified": a.get("verified"), "answer_key": "|".join(a["key"]), "answer": ans, "answer_label": RL[(pid, q)]["label"],
                    "fresh_label": fr, "evidence_sha256": a["evidence_sha"], "sufficiency": RL[(pid, q)]["suff"]})
def pstats(pid):
    rows = [RL[(pid, d["query_id"])] for d in SAMPLE]; c = collections.Counter(r["label"] for r in rows); fcn = sum(1 for r in rows if r["fresh"] == "CORRECT")
    w = sum(1 for r in rows if r["fresh"] == "CORRECT" and r["label"] == "WRONG"); s = collections.Counter(r["suff"] for r in rows); ag = s["SUFFICIENT"] + s["INSUFFICIENT"]
    return {"N": len(rows), "fresh_correct": fcn, "CORRECT": c["CORRECT"], "WRONG": c["WRONG"], "ABSTAIN": c["ABSTAIN"], "UNRESOLVED": c["UNRESOLVED"], "accuracy_pct": round(100 * c["CORRECT"] / len(rows), 3),
            "WAI": w, "WAI_pct": round(100 * w / len(rows), 3), "C_WAI_pct": round(100 * w / fcn, 3), "fresh_correct_to_abstain": sum(1 for r in rows if r["fresh"] == "CORRECT" and r["label"] == "ABSTAIN"),
            "SUFFICIENT": s["SUFFICIENT"], "INSUFFICIENT": s["INSUFFICIENT"], "UNCLEAR": s["UNCLEAR"], "JUDGE_DISAGREEMENT": s["JUDGE_DISAGREEMENT"], "SUFF_UNRESOLVED": s["UNRESOLVED"],
            "sufficient_pct_of_all": round(100 * s["SUFFICIENT"] / len(rows), 3), "sufficient_pct_of_agreed": round(100 * s["SUFFICIENT"] / ag, 3) if ag else None}
Q = {pid: pstats(pid) for pid in AUD}
SUFF3 = lambda p: {k: Q[p][k] for k in ("SUFFICIENT", "INSUFFICIENT", "JUDGE_DISAGREEMENT")}
REF.record("heldout_audit/FreshCache_primary/sufficiency", SUFF3("theta_0.75")); REF.record("heldout_audit/verifier_gamma_0.55/sufficiency", SUFF3("gamma_0.55"))
REF.record("heldout_audit/verifier_gamma_0.55/CORRECT", Q["gamma_0.55"]["CORRECT"]); REF.record("heldout_audit/verifier_gamma_0.55/WAI", Q["gamma_0.55"]["WAI"])
G = {"primary fresh-correct / correct / WAI (run-time reference)": REF.check("heldout_audit/fresh_correct", Q["theta_0.75"]["fresh_correct"]) and REF.check("heldout_audit/FreshCache_primary/CORRECT", Q["theta_0.75"]["CORRECT"]) and REF.check("heldout_audit/FreshCache_primary/WAI", Q["theta_0.75"]["WAI"]),

     "no unresolved answer or sufficiency label at any point": all(Q[p]["UNRESOLVED"] == 0 and Q[p]["SUFF_UNRESOLVED"] == 0 for p in Q)}
for k, v in G.items(): log("GATE", k, v)
if not all(G.values()): json.dump({"gates": G, "quality": Q}, open(O / "work/STOPPED_stage2.json", "w"), indent=1); log("STOP: scoring gate failed"); sys.exit(1)
PRIM = "theta_0.75"
def paired(pid):
    r0 = {d["query_id"]: RL[(PRIM, d["query_id"])] for d in SAMPLE}; r1 = {d["query_id"]: RL[(pid, d["query_id"])] for d in SAMPLE}
    wa = lambda r: r["fresh"] == "CORRECT" and r["label"] == "WRONG"
    w_pt = sum(1 for q in r0 if wa(r1[q]) and not wa(r0[q])); w_pr = sum(1 for q in r0 if wa(r0[q]) and not wa(r1[q]))
    c_pt = sum(1 for q in r0 if r1[q]["label"] == "CORRECT" and r0[q]["label"] != "CORRECT"); c_pr = sum(1 for q in r0 if r0[q]["label"] == "CORRECT" and r1[q]["label"] != "CORRECT")
    s_pt = sum(1 for q in r0 if r1[q]["suff"] == "SUFFICIENT" and r0[q]["suff"] != "SUFFICIENT"); s_pr = sum(1 for q in r0 if r0[q]["suff"] == "SUFFICIENT" and r1[q]["suff"] != "SUFFICIENT")
    return {"serving_changed_vs_primary": sum(1 for q in r0 if r0[q]["sha"] != r1[q]["sha"] or r0[q]["key"] != r1[q]["key"]),
            "WAI_point_only": w_pt, "WAI_primary_only": w_pr, "WAI_mcnemar_p": mcn(w_pt, w_pr), "correct_point_only": c_pt, "correct_primary_only": c_pr, "correct_mcnemar_p": mcn(c_pt, c_pr),
            "sufficient_point_only": s_pt, "sufficient_primary_only": s_pr, "sufficiency_mcnemar_p": mcn(s_pt, s_pr)}
# ---------------- secondary: cross-round drift for A/B points (producer outcome() events)
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); sys.path.insert(0, str(ROOT / "freshcache/simulation")); sys.path.insert(0, str(ROOT / "freshcache/replay_engine"))
import page_version_replay as ma   # noqa: E402
RT = {json.loads(l)["url_hash"]: json.loads(l)["rounds"] for l in open((DATA_ROOT / "derived/round_table.jsonl"))}
def round_at(per, t):   # mixed_age_replay.round_at (same logic)
    best = None
    for name, age in ma.RUNS:
        if age > t: break
        if name in per: best = name
    return best
S = ST["support"]; tq = {t: q for q, t in ST["t_of"].items()}
def w4(pid):
    E_ = ST["events"].get(pid)
    if not E_: return None
    per, by = E_["per"], E_["events_by_t"]; ch = un = 0
    for q in S:
        o = per.get(q)
        if o is None: continue
        es = by.get(ST["t_of"][q], [])
        pages = [(round_at(RT.get(u) or {}, tc) == round_at(RT.get(u) or {}, tr) and round_at(RT.get(u) or {}, tc) is not None, lab) for u, tc, tr, lab in es]
        if any((not s) and lab == "CHANGED" for s, lab in pages): ch += 1
        elif pages and all((not s) and lab == "UNCHANGED" for s, lab in pages): un += 1
    return {"cross_round_determinate": ch + un, "cross_round_changed": ch, "cross_round_drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None, "cross_round_coverage_pct": round(100 * (ch + un) / len(S), 4)}
PTS = []
for pt in POINTS:
    pid = pt["id"]; o = OPS[pid]; pr = paired(pid) if pid != PRIM else {}; x = w4(pid) if pt["verifier"] == "off" else None
    PTS.append({"point": pid, "sweep": pt["sweep"], "theta_L2": pt["theta_L2"], "epsilon_L2": pt["epsilon_L2"], "verifier_gamma": pt["verifier"], "is_primary": pid in (PRIM, "eps_0.20", "gamma_off"),
                "search_saved_pct": o["search_saved_pct"], "fresh_searches": o["search_calls"], "l1_hits": o["l1_hits"], "l2_hits": o["l2_hits"], "fetches": o["fetches"], "fetch_per_1k": o["fetch_per_1k"], "gen_per_1k": o["gen_per_1k"],
                "drift_pct_secondary": o["drift_pct"], "coverage_pct_secondary": o["coverage_pct"], "operational_source": o["source"],
                **Q[pid], **pr, **({f"audit_{k}": v for k, v in x.items()} if x else {"audit_cross_round_drift_pct": "MISSING (verifier transcription engine)" if pt["verifier"] != "off" else None})})
def pareto(key_y, better_high):
    out = []
    for a in PTS:
        dom = any((b["search_saved_pct"] >= a["search_saved_pct"] and ((b[key_y] >= a[key_y]) if better_high else (b[key_y] <= a[key_y]))
                   and (b["search_saved_pct"] > a["search_saved_pct"] or b[key_y] != a[key_y])) for b in PTS if b["point"] != a["point"])
        if not dom: out.append(a["point"])
    return out
PS, PW = pareto("sufficient_pct_of_all", True), pareto("C_WAI_pct", False)
for r in PTS: r["pareto_savings_vs_sufficiency"] = r["point"] in PS; r["pareto_savings_vs_CWAI"] = r["point"] in PW
CR = json.load(open(ROOT / "configs/protocols/l2_threshold_sweep_criteria.json")); thr = 0.75 * OPS[PRIM]["search_saved_pct"]
REG = [r["point"] for r in PTS if r["point"] != PRIM and not r["is_primary"] and r["search_saved_pct"] >= thr and
       ((r["sufficiency_mcnemar_p"] < 0.05 and r["sufficient_point_only"] > r["sufficient_primary_only"]) or (r["WAI_mcnemar_p"] < 0.05 and r["WAI_primary_only"] > r["WAI_point_only"]))]
with open(O / "l2_threshold_points.csv", "w", newline="") as f:
    cols = list(dict.fromkeys(k for r in PTS for k in r)); w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(PTS)
with open(O / "l2_threshold_request_level.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(REQ[0])); w.writeheader(); w.writerows(REQ)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: E402
col = {"A_theta_L2": "tab:blue", "B_epsilon_L2": "tab:green", "C_verifier_gamma": "tab:orange"}
for fn, key, lab, par in (("pareto_savings_vs_sufficiency.pdf", "sufficient_pct_of_all", "jointly judged SUFFICIENT (% of 400)", PS), ("pareto_savings_vs_wai.pdf", "C_WAI_pct", "C-WAI (%, fresh-correct denominator)", PW)):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for sw in col:
        xs = [r for r in PTS if r["sweep"] == sw]; ax.plot([r["search_saved_pct"] for r in xs], [r[key] for r in xs], "o-", color=col[sw], label=sw.replace("_", " "), ms=5, lw=1)
    for r in PTS:
        ax.annotate(r["point"].split("_", 1)[1], (r["search_saved_pct"], r[key]), fontsize=6, xytext=(3, 2), textcoords="offset points")
        if r["point"] in par: ax.scatter([r["search_saved_pct"]], [r[key]], s=90, facecolors="none", edgecolors="k", lw=1)
    pr_ = next(r for r in PTS if r["point"] == PRIM); ax.scatter([pr_["search_saved_pct"]], [pr_[key]], marker="*", s=180, c="red", zorder=5, label="primary FreshCache")
    ax.set_xlabel("held-out search savings (%)"); ax.set_ylabel(lab); ax.set_title("L2 threshold sweeps (diagnostic; circled = Pareto)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(O / fn); plt.close(fig)
BEFORE = json.load(open(O / "work/read_only_before.json"))
def fstat(p):
    hh = hashlib.sha256()
    with open(p, "rb") as f_:
        for b in iter(lambda: f_.read(1 << 24), b""): hh.update(b)
    st = os.stat(p); return {"sha256": hh.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
AFTER = {p: fstat(inpath(p)) for p in BEFORE}; UNCH = {p: BEFORE[p] == AFTER[p] for p in BEFORE}
gpu = [json.loads(l) for l in open(O / "work/gpu_runs.jsonl")] if (O / "work/gpu_runs.jsonl").exists() else []
newans = len(jl(O / "gpu/out/answers_llama3b.jsonl")); newj = {n: len(jl(O / f"gpu/out/judge_{n}.jsonl")) for n in ("llama8b", "qwen7b")}; news = {n: len(jl(O / f"gpu/out/suffv_{n}.jsonl")) for n in ("llama8b", "qwen7b")}
SUM = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "design_sha256": json.load(open(O / "design_frozen.json"))["sha256"], "criteria_sha256": hashlib.sha256((ROOT / "configs/protocols/l2_threshold_sweep_criteria.json").read_bytes()).hexdigest(),
       "gates_stage1": json.load(open(O / "work/gates_stage1.json"))["gates"], "gates_stage2": G, "points": PTS, "pareto_savings_vs_sufficiency": PS, "pareto_savings_vs_CWAI": PW,
       "substantial_savings_threshold_pct": round(thr, 4), "regions_meeting_frozen_criteria": REG,
       "new_outputs": {"answers": newans, "judge2": newj, "sufficiency": news}, "gpu_runs": gpu, "read_only_files_unchanged": UNCH, "read_only_files": {p: {**BEFORE[p], "unchanged": UNCH[p]} for p in BEFORE}}
json.dump(SUM, open(O / "l2_threshold_summary.json", "w"), indent=1, default=str)
log("pareto sufficiency", PS); log("pareto C-WAI", PW); log("regions meeting criteria", REG); log("all read-only files unchanged:", all(UNCH.values()))
