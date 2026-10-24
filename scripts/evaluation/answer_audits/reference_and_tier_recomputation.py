"""Recomputes the model-based validated-subset study and tier-specific diagnostics by running unmodified copies of the original analysis
scripts (scripts/evaluation/reference_validation/reference_validation_analysis.py, scripts/evaluation/tier_quality/tier_quality_analysis.py) on shadow folders.
keyset    = the recorded inputs (gate: must reproduce the recorded results byte-for-byte in the key tables)
reported = replay-engine policy outputs / hit sets; judge outputs = original raw/parsed files + gpu/out (new tasks only).
The reference stage (VALID_AT_T subset) is copied unchanged, so reference subsets are preserved."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, shutil, pathlib, re, collections, subprocess, csv; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import answer_pools as P
R = P.R; W = ((OUTPUT_ROOT / "replay")); GO = W/"gpu/out"; PY = os.environ.get("PYTHON", sys.executable)
MB, TQ = OUTPUT_ROOT / "reference_validation", OUTPUT_ROOT / "tier_quality"
def jl(p): return P.jl(p)
def wl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def ans3(raw):
    u = raw.upper(); u = re.sub(r"\bINCORRECT\b", "WRONG", u); u = re.sub(r"^(WROONG|WONG)\b", "WRONG", u)
    hits = [(u.find(l), l) for l in ("CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE") if u.find(l) >= 0]; return min(hits)[1] if hits else "UNPARSED"
J3 = ("llama8b", "qwen7b", "mistral7b")
def raw_answer_pool(j):   # task_id -> raw record; original study files first, new GPU outputs last; first occurrence wins
    pool = {}
    for p in (MB/"outputs/raw"/f"answer_{j}.jsonl", TQ/"outputs/raw"/f"answer_{j}.jsonl", GO/f"answer3_{j}.jsonl"):
        for r in jl(p): pool.setdefault(r["task_id"], r)
    return pool
def raw_suff_pool(j):
    pool = {}
    for p in (TQ/"outputs/raw"/f"sufficiency_{j}.jsonl", GO/f"suff3_{j}.jsonl"):
        for r in jl(p): pool.setdefault(r["task_id"], r)
    return pool
def patch_copy(src, dst, W_line_old, W_line_new):
    s = open(src).read(); assert W_line_old in s, src; s = s.replace(W_line_old, W_line_new); dst.parent.mkdir(parents=True, exist_ok=True); open(dst, "w").write(s)
# ---------------- model-based study ----------------
def mb(eng):
    D = W/"shadow"/eng/"model_based"
    if D.exists(): shutil.rmtree(D)
    for sub in ("config", "inputs", "outputs/parsed", "results", "scripts"): (D/sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(MB/"config/protocol.json", D/"config/protocol.json")
    for f in ("requests.jsonl", "evidence.jsonl"): shutil.copy(MB/"inputs"/f, D/"inputs"/f)
    if eng == "keyset": shutil.copy(MB/"inputs/policy_outputs.jsonl", D/"inputs/policy_outputs.jsonl")
    else: wl(D/"inputs/policy_outputs.jsonl", [{k: r[k] for k in ("query_id", "arm", "tier", "answer_key", "answer")} for r in jl(W/"results/model_based_policy_outputs_reported.jsonl")])
    for j in J3:
        shutil.copy(MB/"outputs/parsed"/f"reference_{j}.jsonl", D/"outputs/parsed"/f"reference_{j}.jsonl")
        need = {P.tid(o["query_id"], o["answer"]) for o in jl(D/"inputs/policy_outputs.jsonl")}
        parsed = {r["task_id"]: r for r in jl(MB/"outputs/parsed"/f"answer_{j}.jsonl")}
        rawp = raw_answer_pool(j); rows = []
        for t in sorted(need):
            if t in parsed: rows.append(parsed[t])
            elif t in rawp: r = rawp[t]; rows.append({**r, "label": ans3(r["raw"])})
        wl(D/"outputs/parsed"/f"answer_{j}.jsonl", rows)
    shutil.copy((REPO_ROOT / "scripts/evaluation/reference_validation/reference_validation_analysis.py"), D/"scripts/reference_validation_analysis.py")   # unmodified copy; the shadow folder is passed as its workspace
    out = subprocess.run([PY, str(D/"scripts/reference_validation_analysis.py"), str(D)], capture_output=True, text=True); open(D/"analyze.log", "w").write(out.stdout + out.stderr); assert out.returncode == 0, out.stderr[-2000:]
# ---------------- tier study ----------------
def verdict(v):
    v = {j: l for j, l in v.items() if l in ("SAME", "DIFFERENT")}
    if len(v) < 2: return "UNJUDGED"
    c = collections.Counter(v.values()); return "JURY_TIE" if c["SAME"] == c["DIFFERENT"] else c.most_common(1)[0][0]
def tier(eng):
    D = W/"shadow"/eng/"tier"
    if D.exists(): shutil.rmtree(D)
    for sub in ("config", "inputs", "outputs/raw", "outputs/parsed", "results", "scripts"): (D/sub).mkdir(parents=True, exist_ok=True)
    for s, f in (("reference_validity_protocol.json", "answer_protocol_frozen.json"), ("tier_quality_study_config.json", "study_config.json"), ("tier_quality_study_config.sha256", "study_config.sha256")): shutil.copy(REPO_ROOT / "configs" / s, D/"config"/f)
    if eng == "keyset":
        for f in ("l1_reuse_hits.jsonl", "l2_reuse_hits.jsonl", "answer_tasks.jsonl", "sufficiency_tasks.jsonl", "path_rows.jsonl"): shutil.copy(TQ/"inputs"/f, D/"inputs"/f)
    else:
        TR = json.load(open(W/"results/tier_cpu_reported.json")); JY = P.jury(extra=[str(GO/"equiv_{j}.jsonl")])
        t1 = [{**x, "equivalence_4juror": verdict(JY.get(x["pair_id"], {}))} for x in TR["l1_reuse"]]
        assert all(x["equivalence_4juror"] != "UNJUDGED" for x in t1)
        wl(D/"inputs/l1_reuse_hits.jsonl", t1); wl(D/"inputs/l2_reuse_hits.jsonl", TR["l2_reuse"])
        PR = jl(W/"results/tier_path_rows_reported.jsonl"); wl(D/"inputs/path_rows.jsonl", PR)
        T1 = {x["query_id"]: x for x in TR["l1_reuse"]}; T2 = {x["query_id"]: x for x in TR["l2_reuse"]}
        at, st = {}, {}
        for r in PR:
            x = (T1 if r["task"] == "l1_reuse" else T2)[r["query_id"]]; qf = "incoming_query" if r["task"] == "l1_reuse" else "question"
            at.setdefault(r["answer_task_id"], {"task_id": r["answer_task_id"], "query_id": r["query_id"], "question": x[qf], "target": r["target_date"], "reference": x["reference_answer"], "candidate": r["answer"]})
            if r["sufficiency_task_id"]:
                st.setdefault(r["sufficiency_task_id"], {"task_id": r["sufficiency_task_id"], "query_id": r["query_id"], "question": x[qf], "evidence": x["paths"][r["path"]]["evidence_block"]})
        wl(D/"inputs/answer_tasks.jsonl", list(at.values())); wl(D/"inputs/sufficiency_tasks.jsonl", list(st.values()))
    at = {t["task_id"] for t in jl(D/"inputs/answer_tasks.jsonl")}; st = {t["task_id"] for t in jl(D/"inputs/sufficiency_tasks.jsonl")}
    for j in J3:
        ap, sp = raw_answer_pool(j), raw_suff_pool(j)
        wl(D/"outputs/raw"/f"answer_{j}.jsonl", [ap[t] for t in sorted(at) if t in ap]); wl(D/"outputs/raw"/f"sufficiency_{j}.jsonl", [sp[t] for t in sorted(st) if t in sp])
        miss = len(at - set(ap)) + len(st - set(sp)); assert miss == 0, (eng, j, miss)
    s = open((REPO_ROOT / "scripts/evaluation/tier_quality/tier_quality_analysis.py")).read()
    # population labels already follow the measured sizes (all_<n>)
    open(D/"scripts/tier_analysis.py", "w").write(s)
    out = subprocess.run([PY, str(D/"scripts/tier_analysis.py"), str(D)], capture_output=True, text=True); open(D/"analyze.log", "w").write(out.stdout + out.stderr); assert out.returncode == 0, out.stderr[-2000:]
for eng in ("keyset", "reported"):
    mb(eng); tier(eng); print(eng, "done", flush=True)
# ---- keyset gates: shadow results must equal the recorded results ----
def same_csv(a, b): return list(csv.DictReader(open(a))) == list(csv.DictReader(open(b)))
g = {"model_based/answer_quality_results.csv": same_csv(W/"shadow/keyset/model_based/results/answer_quality_results.csv", MB/"results/answer_quality_results.csv"),
     "model_based/paired_tests.csv": same_csv(W/"shadow/keyset/model_based/results/paired_tests.csv", MB/"results/paired_tests.csv"),
     "tier/path_results.csv": same_csv(W/"shadow/keyset/tier/results/path_results.csv", TQ/"results/path_results.csv"),
     "tier/paired_tests.csv": same_csv(W/"shadow/keyset/tier/results/paired_tests.csv", TQ/"results/paired_tests.csv")}
json.dump(g, open(W/"results/shadow_keyset_gates.json", "w"), indent=1); print("keyset gates", g)
