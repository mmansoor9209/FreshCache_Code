"""Completion and audit of the answer-verifier (AV) runs: a run that is missing answers is completed until no answer is missing.
  loop      iterate the frozen D+AV replays until no answer is missing (held-out: answer_verifier.py heldout + frozen 3B; paraphrase S:
            live_stream_admission_controls.py replay S + frozen 14B), at most 40 passes each; existing run outputs are left untouched
  label     held-out: the completed D+AV trace on the extended common sample (same seed design); paraphrase S: census vs B
  audit     missing answers and fail-open propagation (requests whose signature differs between the fail-open trace and the completed one),
            completed vs earlier losses / gains / net / cost, gross losses per 1,000 eligible requests"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, math, os, pickle, subprocess, sys, time, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
O = ((OUTPUT_ROOT / "cross_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); NW_DIR = ((OUTPUT_ROOT / "natural_workload")); MODE = sys.argv[1]
CODE = (REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls"); PY = os.environ.get("PYTHON", sys.executable); MC = os.environ.get("HF_HUB_CACHE", (REPO_ROOT / "models_cache/hub"))
LOG = open(O / "logs/av_completion.log", "a")
def log(*a): s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def env(hf, cuda=None):
    e = dict(os.environ, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if hf: e["HF_HUB_CACHE"] = MC
    else: e.pop("HF_HUB_CACHE", None)
    if cuda is not None: e["CUDA_VISIBLE_DEVICES"] = cuda
    return e
def sh(args, hf=True, cuda="1", out="av_steps"):
    with open(O / f"logs/{out}.out", "a") as f: r = subprocess.run([PY] + args, cwd=O, env=env(hf, cuda), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode: log("FAILED", args); sys.exit(1)
def nlines(p): p = O / "gpu/in" / p; return sum(1 for l in open(p) if l.strip()) if p.exists() else 0
def gen(tasks, big, tag, gpus):
    ps = [subprocess.Popen([PY, str((REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls/cross_workload_generation_judging.py")), "gen14" if big else "gen3", "--gpu", g, "--tasks", tasks, "--tag", tag, "--shard", f"{i}/{len(gpus)}"], cwd=O, env=env(big),
                           stdout=open(O / f"logs/av_gen_{tag}_{i}.out", "a"), stderr=subprocess.STDOUT) for i, g in enumerate(gpus)]
    for p in ps: p.wait()
def judge(tasks):
    ps = [subprocess.Popen([PY, str((REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls/cross_workload_generation_judging.py")), "judge2", "--judge", j, "--gpu", g, "--tasks", tasks], cwd=O, env=env(False), stdout=open(O / f"logs/av_judge_{j}.out", "a"),
                           stderr=subprocess.STDOUT) for j, g in (("llama8b", "7"), ("qwen7b", "9"))]
    for p in ps: p.wait()
if MODE == "loop":
    which = sys.argv[2]
    if which == "heldout":
        for it in range(40):
            sh([str((REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls/answer_verifier.py")), "heldout"]); m = json.load(open(O / "outputs/av_replay.json"))["missing_answers_fail_open"]; log("held-out pass", it, "missing", m)
            if not m: break
            gen("gen_tasks.jsonl", False, f"avho{it}", ["3", "4"])
    else:
        for it in range(40):
            sh([str((REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls/live_stream_admission_controls.py")), "replay", "S"]); m = nlines("cctl_tasks14.jsonl"); log("paraphrase S pass", it, "missing", m)
            if not m: break
            gen("cctl_tasks14.jsonl", True, f"avpS{it}", ["4"])
    log("LOOP DONE", which)
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
if MODE == "label":
    for it in range(4):
        sh([str((REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls/live_stream_admission_controls.py")), "items", "S"]); g, j = nlines("cctl_tasks14.jsonl"), nlines("judge2_items.jsonl"); log("paraphrase S labels pass", it, g, j)
        if not (g or j): break
        if g: gen("cctl_tasks14.jsonl", True, f"avpSL{it}", ["4"])
        if j: judge("judge2_items.jsonl")
    for it in range(4):
        sh([str((REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls/answer_verifier_completion.py")), "hoitems"]); g, j = nlines("av_ho_gen.jsonl"), nlines("av_ho_judge.jsonl"); log("held-out labels pass", it, g, j)
        if not (g or j): break
        if g: gen("av_ho_gen.jsonl", False, f"avhoL{it}", ["3", "4"])
        if j: judge("av_ho_judge.jsonl")
    log("LABEL DONE")
if MODE in ("hoitems", "audit"):
    import output_workspace; output_workspace.use("cross_workload"); import composition_study_helpers as X, live_stream_labeller as CL
    sys.argv = [sys.argv[0], "pool"]; import answer_verifier as AVL
    E = X.engine(); REC = {r["query_id"]: r for r in E["recs"]}; lab = CL.Lab(); POOL = AVL.Pool(); goldof = AVL.goldof
    AV_EVAL = pickle.load(open(NW_DIR / "work/admission_control_evaluation.pkl", "rb")); S_ = AV_EVAL["S"]
    TRACES_PREVIOUS = pickle.load(open(NW_DIR / "work/av_traces.pkl", "rb"))["DAV"]; TNEW = pickle.load(open(O / "work/av_traces.pkl", "rb"))["DAV"]; TRACES = pickle.load(open(CS_DIR / "work/traces.pkl", "rb"))
    def key(r, q): k, ctx, gq = X.key_for({"ev": r["ev"], "gen_query": r["gen_query"]}, REC[q]["query"]); return tuple(k), ctx, gq
    def L(r, q, G=None, J=None):
        k, ctx, gq = key(r, q); a = POOL.answer(k)
        if a is None:
            if G is not None: G.setdefault(k, {"key": list(k), "question": gq, "context": ctx})
            return None
        l, _, s = lab.label_answer(REC[q]["query"], goldof(q), a)
        if l is None and J is not None: J.setdefault(s, {"sig": list(s), "q": REC[q]["query"], "gold": goldof(q), "cand": a})
        return l
if MODE == "hoitems":
    G, J = {}, {}
    for q in S_:
        for r in (TRACES["B"][q], TNEW[q]): L(r, q, G, J)
    X.wjl(O / "gpu/in/av_ho_gen.jsonl", list(G.values())); X.wjl(O / "gpu/in/av_ho_judge.jsonl", list(J.values())); print("gen", len(G), "judge", len(J))
if MODE == "audit":
    C = pickle.load(open(CS_DIR / "work/common.pkl", "rb")); Nref = C["N_ref"]; Up = AV_EVAL["U_prime"]; n = len(S_); f = 100 * Up / Nref
    def est(T):
        d = [int(L(T[q], q) == "CORRECT") - int(L(TRACES["B"][q], q) == "CORRECT") for q in S_]; mu = sum(d) / n; var = sum((x - mu) ** 2 for x in d) / (n - 1); se = math.sqrt(var / n * (1 - n / Up))
        lo, ga = sum(x < 0 for x in d), sum(x > 0 for x in d)
        return {"losses": lo, "gains": ga, "stream_pp": round(f * mu, 3), "ci_pp": [round(f * (mu - 1.96 * se), 3), round(f * (mu + 1.96 * se), 3)],
                "losses_per_1k_sample": round(1000 * lo / n, 2), "gains_per_1k_sample": round(1000 * ga / n, 2), "unresolved": sum(L(T[q], q) is None for q in S_)}
    held = {"n_sample": n, "frame": Up, "previous_fail_open_answers": json.load(open(NW_DIR / "outputs/av_replay.json")).get("missing_answers_fail_open"),
            "completed_missing": json.load(open(O / "outputs/av_replay.json"))["missing_answers_fail_open"],
            "requests_differing_previous_vs_completed_all": sum(sig(TRACES_PREVIOUS[q]) != sig(TNEW[q]) for q in TNEW),
            "requests_differing_in_sample": sum(sig(TRACES_PREVIOUS[q]) != sig(TNEW[q]) for q in S_),
            "previous": est(TRACES_PREVIOUS), "completed": est(TNEW),
            "cost_ms_previous": json.load(open(NW_DIR / "outputs/av_replay.json"))["cost_ms"], "cost_ms_completed": json.load(open(O / "outputs/av_replay.json"))["cost_ms"]}
    # paraphrase S: earlier fail-open run vs completed census
    import types
    c8 = json.load(open(NW_DIR / "outputs/live_stream_admission_controls.json"))["S"]["DAV"]
    CCTL_S_NEW = pickle.load(open(O / "work/cctl_S.pkl", "rb")); CCTL_S_PREVIOUS = pickle.load(open(NW_DIR / "work/cctl_S.pkl", "rb"))
    para = {"previous": c8, "requests_differing_previous_vs_completed": sum(sig(CCTL_S_PREVIOUS["rows"]["DAV"][q]) != sig(CCTL_S_NEW["rows"]["DAV"][q]) for q in CCTL_S_NEW["rows"]["DAV"]),
            "pages_absent_no_fetch": len(CCTL_S_NEW.get("fetched_now", []))}
    R = {"heldout": held, "paraphrase_S": para}
    json.dump(R, open(O / "outputs/answer_verifier_completion.json", "w"), indent=1); print(json.dumps(R, indent=1))
