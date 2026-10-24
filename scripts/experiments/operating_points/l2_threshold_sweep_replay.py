"""L2 operating points, stage 1 (CPU): design freeze, held-out replays of every operating point, served evidence / answer keys for the frozen
audit, reuse lookup, and task lists for anything missing. Read-only on all existing files; writes only inside outputs/l2_operating_points/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, csv, datetime, hashlib, json, os, pathlib, pickle, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "l2_operating_points"); CRR = (OUTPUT_ROOT / "replay"); OV = (OUTPUT_ROOT / "l2_verifier/out")
A2 = (OUTPUT_ROOT / "general_audit")
for d in ("work", "gpu/in", "gpu/out"): (O / d).mkdir(parents=True, exist_ok=True)
LOG = open(O / "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
# ---------------- design, frozen before any execution
DESIGN = {"written_utc": now(), "status": "sensitivity diagnostics; no primary policy is selected from held-out results",
          "primary": {"theta_L2": 0.75, "epsilon_L2": 0.20, "verifier": "off"},
          "A_theta_L2_sweep": {"grid": [0.35, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], "grid_source": "recovered: l2_equivalence_fit.py:199-200 (documented L2 threshold grid)",
                               "epsilon_L2": 0.20, "verifier": "off", "implementation": "experiment._L2_EQ_SIM_FLOOR set in memory (L2 floor = max(L2_SIM_THRESHOLD=0.35, floor))"},
          "B_epsilon_L2_sweep": {"grid": [0.10, 0.20, 0.30, 0.35], "theta_L2": 0.75, "verifier": "off", "implementation": "experiment.EPS_URL_LIST set in memory"},
          "C_verifier_gamma_sweep": {"grid": ["off", 0.50, 0.55, 0.60], "theta_L2": 0.75, "epsilon_L2": 0.20,
                                     "source": "existing validation-selected gamma 0.55 (outputs/archived_runs/.../gamma_selection.json) and its held-out +-0.05 sensitivity arms; replay-engine replays reused from outputs/replay/verifier/out"},
          "quality": "frozen held-out audit; answers Llama-3.2-3B FORCED; 2-judge correctness (both CORRECT); sufficiency on served evidence, 2 judges, both agree else JUDGE_DISAGREEMENT (verifier_evaluation)",
          "reuse_rule": "answers by exact key then (context, question) content; judgments by (question, gold, answer) signature; sufficiency labels by (query_id, sha256(evidence block))",
          "secondary": "cross-round cross-round drift (producer outcome() events captured by observation wrapper) for A/B points; MISSING for verifier arms (not computable in producer semantics read-only)"}
DESIGN["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in DESIGN.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (O / "design_frozen.json").exists(): json.dump(DESIGN, open(O / "design_frozen.json", "w"), indent=1)
DESIGN = json.load(open(O / "design_frozen.json")); log(f"[{now()}] design sha256 {DESIGN['sha256']}")
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/method_replay.py",
        "freshcache/replay_engine/execution_semantics.py", "freshcache/replay_engine/engine_loader.py", "freshcache/replay_engine/answer_pools.py",
        "scripts/experiments/admission_controls/verifier_audit_keys.py", "scripts/experiments/admission_controls/verifier_evaluation.py", "scripts/evaluation/generation_judging_stage.py",
        "outputs/replay/verifier/out/test_rows.pkl", "outputs/replay/verifier/out/test_operational.json", "outputs/replay/verifier/audit/audit_sample.jsonl",
        "outputs/replay/verifier/audit/suff_tasks.jsonl", "outputs/replay/verifier/audit/suff_reused.json", "outputs/replay/gpu/in/suffv_tasks.jsonl",
        "outputs/replay/gpu/out/suffv_llama8b.jsonl", "outputs/replay/gpu/out/suffv_qwen7b.jsonl", "outputs/replay/results/verifier_recompute.json",
        "outputs/replay/results/audit_keys_reported.json", "outputs/replay/results/replays_reported.json", "outputs/replay/results/replays_reported_per_request.pkl",
        "outputs/l2_verifier/out/audit_suff_tasks.jsonl", "outputs/l2_verifier/out/audit_suff_llama8b.jsonl", "outputs/l2_verifier/out/audit_suff_qwen7b.jsonl",
        "outputs/l2_verifier/out/audit_answers.jsonl", "outputs/l2_verifier/out/audit_judge_llama8b.jsonl", "outputs/l2_verifier/out/audit_judge_qwen7b.jsonl",
        "outputs/l2_verifier/out/gamma_selection.json", "outputs/general_audit/sample.json", "outputs/general_audit/answers.jsonl",
        "outputs/general_audit/judge_llama8b.jsonl", "outputs/general_audit/judge_qwen7b.jsonl", "freshcache/method/l2_equivalence_fit.py",
        "freshcache/method/tiered_cache_model.py", "freshcache/simulation/page_version_replay.py", "scripts/evaluation/answer_grading.py", "data/derived/round_table.jsonl", "data/url_manifest.jsonl", "data/queries.jsonl", "data/sim_matrix_bgem3.npy"]
BEFORE = {p: fstat(inpath(p)) for p in READ}
json.dump(BEFORE, open(O / "work/read_only_before.json", "w"), indent=1)
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
bad = [p for p, v in BEFORE.items() if p in FH and FH[p] != v["sha256"]]
if bad: log("STOP: frozen inputs differ", bad); sys.exit(1)
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C, answer_pools as P   # noqa: E402
E = C.load("reported"); exp, p2, me = E["exp"], E["p2"], E["me"]; h, ctx_for = p2.h, p2.ctx_for
import answer_grading as e2e   # noqa: E402
H = C.streams(E)["heldout"]; assert DS.check("heldout_requests", len(H)); S = E["support_of"](H, E["rounds"]); T = {r["query_id"]: t for t, r in H}
SAMPLE = json.load(open(A2 / "sample.json"))["requests"]; AK = json.load(open(CRR / "results/audit_keys_reported.json"))["keys"]["heldout"]
FR = json.load(open(CRR / "results/replays_reported.json"))["heldout/FreshCache"]
man = {}
for m in exp.load_jsonl(exp.MANIFEST_FILE):
    if m.get("run_id") == "run_00" and m["url_hash"] not in man: man[m["url_hash"]] = {"title": m.get("title", "") or "", "snippet": m.get("snippet", "") or ""}
def evblock(ev):   # verifier_audit_keys.evblock, verbatim logic
    lines = []
    for uh, tv in ev:
        mt = man.get(uh, {}); b = re.sub(r"\s+", " ", (e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv)) or "")).strip()[:2000]
        blk = f"{len(lines)+1}. {mt.get('title','').strip()}"
        if mt.get("snippet"): blk += f"\n   {mt['snippet'].strip()}"
        if len(b) >= 200: blk += f"\n   PAGE TEXT: {b}"
        lines.append(blk)
    return "\n".join(lines) or "(no stored evidence)"
class patch:
    def __init__(self, theta=None, eps=None): self.t, self.e = theta, eps
    def __enter__(self):
        self.o = (exp._L2_EQ_SIM_FLOOR, exp.EPS_URL_LIST)
        if self.t is not None: exp._L2_EQ_SIM_FLOOR = self.t
        if self.e is not None: exp.EPS_URL_LIST = self.e
    def __exit__(self, *a):
        exp._L2_EQ_SIM_FLOOR, exp.EPS_URL_LIST = self.o; assert (exp._L2_EQ_SIM_FLOOR, exp.EPS_URL_LIST) == (0.75, 0.20)
POINTS = [{"id": f"theta_{t:.2f}", "sweep": "A_theta_L2", "theta_L2": t, "epsilon_L2": 0.20, "verifier": "off"} for t in DESIGN["A_theta_L2_sweep"]["grid"]] + \
         [{"id": f"eps_{e:.2f}", "sweep": "B_epsilon_L2", "theta_L2": 0.75, "epsilon_L2": e, "verifier": "off"} for e in DESIGN["B_epsilon_L2_sweep"]["grid"]] + \
         [{"id": f"gamma_{g}", "sweep": "C_verifier_gamma", "theta_L2": 0.75, "epsilon_L2": 0.20, "verifier": g} for g in DESIGN["C_verifier_gamma_sweep"]["grid"]]
G, OPS, AUD, EVT = {}, {}, {}, {}
_orig_outcome = me.outcome
for pt in POINTS:
    if pt["verifier"] != "off": continue
    ev_rec = []
    def rec(uh, tc, tr, rounds, _o=_orig_outcome): lab = _o(uh, tc, tr, rounds); ev_rec.append((uh, tc, tr, lab)); return lab
    with patch(pt["theta_L2"], pt["epsilon_L2"]):
        me.outcome = rec
        try: row, per, tr_ = C.metrics(E, H, "FreshCache", S)
        finally: me.outcome = _orig_outcome
        _, rows = p2.replay(H, {}, "FreshCache")
    mp = {"L1": "L1", "L2": "L2", "miss": "miss", "REAL_TIME_BYPASS": "REAL_TIME"}
    G[f"{pt['id']}: audit_replay and mixed_age_replay route every request identically"] = all(mp[tr_[q]["tier"]] == rows[q]["tier"] for q in tr_)
    row["search_calls"] = sum(1 for q in tr_ if tr_[q]["tier"] in ("miss", "REAL_TIME_BYPASS")); OPS[pt["id"]] = {**row, "source": "mixed_age_replay.replay (frozen engine), this run"}
    by = collections.defaultdict(list)
    for uh, tc, trq, lab in ev_rec: by[trq].append((uh, tc, trq, lab))
    EVT[pt["id"]] = {"per": dict(per), "events_by_t": by}
    AUD[pt["id"]] = {}
    for d in SAMPLE:
        q = d["query_id"]; p = rows[q]
        if p["tier"] == "L1": ev = [tuple(x) for x in p["stored_answer_ev"]]; sctx = ctx_for(tuple(ev)); key = ["stored", h(sctx), h(p["stored_answer_query"])]; gq = p["stored_answer_query"]; ctx = sctx
        else: ev = [tuple(x) for x in (p.get("ev") or [])]; ctx = ctx_for(tuple(ev)) if ev else ""; key = ["gen", h(ctx), h(d["query"])]; gq = d["query"]
        AUD[pt["id"]][q] = {"tier": p["tier"], "ev": ev, "key": key, "gen_question": gq, "context": ctx}
    log(pt["id"], {k: row[k] for k in ("search_saved_pct", "l1_hits", "l2_hits", "fetches", "search_calls", "drift_pct", "coverage_pct")})
prim = OPS["theta_0.75"]
G["primary point equals frozen held-out FreshCache metrics"] = {k: prim[k] for k in FR} == FR
G["theta_0.75 and eps_0.20 identical"] = OPS["theta_0.75"] == OPS["eps_0.20"]
G["primary audit keys equal frozen FreshCache keys (all audit requests)"] = all(AUD["theta_0.75"][d["query_id"]]["key"][1:] == AK[d["query_id"]]["freshcache"][0][1:] and AUD["theta_0.75"][d["query_id"]]["tier"] == AK[d["query_id"]]["freshcache"][1] for d in SAMPLE)
# ---------------- verifier arms (reused reported replays)
VR = pickle.load(open(CRR / "verifier/out/test_rows.pkl", "rb")); VO = json.load(open(CRR / "verifier/out/test_operational.json"))["results"]
VA = {json.loads(l)["query_id"]: json.loads(l) for l in open(CRR / "verifier/audit/audit_sample.jsonl")}
G["verifier FreshCache arm equals frozen held-out metrics"] = VO["FreshCache"]["search_saved_pct"] == FR["search_saved_pct"] and VO["FreshCache"]["fetches"] == FR["fetches"]
for g in (0.50, 0.55, 0.60):
    name = f"L2Verify_g{g:g}" if g != 0.50 else "L2Verify_g0.5"; name = {0.50: "L2Verify_g0.5", 0.55: "L2Verify_g0.55", 0.60: "L2Verify_g0.6"}[g]
    rw = VR[name]; o = VO[name]; pid = f"gamma_{g}"
    OPS[pid] = {"n": o["n_requests"], "search_saved_pct": o["search_saved_pct"], "search_calls": o["search_calls"], "l1_hits": o["l1_hits"], "l2_hits": o["l2_hits"], "l3_hits": o["l3_hits"],
                "fetches": o["fetches"], "fetch_per_1k": o["fetches_per_1k"], "gen_per_1k": o["generations_per_1k"], "generations": o["generations"], "l2_accepted": o["l2_accepted"], "l2_rejected": o["l2_rejected"],
                "l2_unscored_fail_open": o["l2_unscored"], "drift_pct": o["drift_pct"], "coverage_pct": o["coverage_pct"], "changed": o["changed"], "unchanged": o["determinable"] - o["changed"],
                "source": "reused: outputs/replay/verifier/out/test_operational.json (verifier transcription engine; its drift scorer differs from mixed_age_replay)"}
    AUD[pid] = {}
    for d in SAMPLE:
        q = d["query_id"]; b = rw[q]; ev = [tuple(x) for x in ((b.get("stored_answer_ev") if b["tier"] == "L1" else b.get("ev")) or [])]
        ctx = ctx_for(tuple(ev)); gq = b.get("gen_query", d["query"]); AUD[pid][q] = {"tier": b["tier"], "ev": ev, "key": ["gen", h(ctx), h(gq)], "gen_question": gq, "context": ctx, "verified": b.get("verified")}
    if g == 0.55: G["gamma_0.55 audit keys equal frozen L2Verify keys (all audit requests)"] = all(AUD[pid][q]["key"] == VA[q]["lv_key"] for q in VA)
OPS["gamma_off"] = {**OPS["theta_0.75"], "source": OPS["theta_0.75"]["source"] + " (= primary FreshCache)"}; AUD["gamma_off"] = AUD["theta_0.75"]
# ---------------- reuse lookups
EXACT = P.answers_exact(extra=[CRR / "gpu/out/answers_llama3b.jsonl", OV / "audit_answers.jsonl", O / "gpu/out/answers_llama3b.jsonl"])
CONTENT = P.answers(extra=[CRR / "gpu/out/answers_llama3b.jsonl", OV / "audit_answers.jsonl", O / "gpu/out/answers_llama3b.jsonl"])
def answer(key):
    a = EXACT.get(tuple(key)); return a if a is not None else CONTENT.get((key[1], key[2]))
SL = {}   # (qid, sha(evidence)) -> {"llama8b":..,"qwen7b":..}
def addsuff(tasks, l8, q7, src):
    for t in tasks:
        k = (t["query_id"], hashlib.sha256(t["evidence"].encode()).hexdigest()); a, b = l8.get((t["query_id"], t["arm"])), q7.get((t["query_id"], t["arm"]))
        if a is None or b is None: continue
        if k in SL and SL[k]["labels"] != (a, b): G.setdefault("sufficiency_label_conflicts", []).append(k)
        SL.setdefault(k, {"labels": (a, b), "source": src})
jl = P.jl
addsuff(jl(OV / "audit_suff_tasks.jsonl"), {(r["query_id"], r["arm"]): r["label"] for r in jl(OV / "audit_suff_llama8b.jsonl")}, {(r["query_id"], r["arm"]): r["label"] for r in jl(OV / "audit_suff_qwen7b.jsonl")}, "earlier_validation verifier study")
RU = json.load(open(CRR / "verifier/audit/suff_reused.json")); cl8 = {tuple(k.split("|")): v["llama8b"] for k, v in RU.items()}; cq7 = {tuple(k.split("|")): v["qwen7b"] for k, v in RU.items()}
for r in jl(CRR / "gpu/out/suffv_llama8b.jsonl"): cl8[(r["query_id"], r["arm"])] = r["label"]
for r in jl(CRR / "gpu/out/suffv_qwen7b.jsonl"): cq7[(r["query_id"], r["arm"])] = r["label"]
addsuff(jl(CRR / "verifier/audit/suff_tasks.jsonl"), cl8, cq7, "reported verifier recompute")
new8 = {(r["query_id"], r["arm"]): r["label"] for r in jl(O / "gpu/out/suffv_llama8b.jsonl")}; new7 = {(r["query_id"], r["arm"]): r["label"] for r in jl(O / "gpu/out/suffv_qwen7b.jsonl")}
addsuff(jl(O / "gpu/in/suffv_tasks.jsonl"), new8, new7, "L2-sweep new judgments")
G["no sufficiency label conflicts between reused sources"] = not G.get("sufficiency_label_conflicts")
gen_tasks, suff_tasks = {}, {}
for pid, A in AUD.items():
    for q, a in A.items():
        a["evidence_block"] = evblock(a["ev"]); a["evidence_sha"] = hashlib.sha256(a["evidence_block"].encode()).hexdigest()
        if answer(a["key"]) is None: gen_tasks.setdefault(tuple(a["key"]), {"key": a["key"], "question": a["gen_question"], "context": a["context"]})
        if (q, a["evidence_sha"]) not in SL: suff_tasks.setdefault((q, a["evidence_sha"]), {"query_id": q, "arm": a["evidence_sha"][:16], "query": next(d["query"] for d in SAMPLE if d["query_id"] == q), "evidence": a["evidence_block"]})
json.dump({"gates": {k: v for k, v in G.items() if k != "sufficiency_label_conflicts"}, "conflicts": G.get("sufficiency_label_conflicts", [])}, open(O / "work/gates_stage1.json", "w"), indent=1, default=str)
for k, v in G.items(): log("GATE", k, v if not isinstance(v, list) else len(v))
if not all(v is True for k, v in G.items() if k != "sufficiency_label_conflicts") or G.get("sufficiency_label_conflicts"): log("STOP: gate failed"); sys.exit(1)
P.__dict__  # noqa
with open(O / "gpu/in/gen_tasks.jsonl", "w", encoding="utf-8") as f:
    for t in gen_tasks.values(): f.write(json.dumps(t, ensure_ascii=False) + "\n")
old = {(t["query_id"], t["arm"]) for t in jl(O / "gpu/in/suffv_tasks.jsonl")}
with open(O / "gpu/in/suffv_tasks.jsonl", "a", encoding="utf-8") as f:
    for t in suff_tasks.values():
        if (t["query_id"], t["arm"]) not in old: f.write(json.dumps(t, ensure_ascii=False) + "\n")
pickle.dump({"points": POINTS, "ops": OPS, "audit": {pid: {q: {k: v for k, v in a.items() if k != "context"} for q, a in A.items()} for pid, A in AUD.items()},
             "events": EVT, "support": S, "t_of": T}, open(O / "work/stage1.pkl", "wb"))
log(f"new answers needed: {len(gen_tasks)}; new sufficiency tasks needed: {len(suff_tasks)}")
