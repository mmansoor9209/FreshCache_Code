"""Same-environment concurrency study: NoCache vs. frozen primary FreshCache (reported decisions), local replayed backend.
Usage: concurrency_benchmark.py GPU   (physical GPU index). Design and run order frozen before any timing. Writes only outputs/concurrency_cost/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, copy, csv, datetime, hashlib, importlib.util, json, os, pathlib, platform, re, statistics, sys, threading, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p; import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
from concurrent.futures import Future
sys.dont_write_bytecode = True
GPU = sys.argv[1] if len(sys.argv) > 1 else "4"; os.environ["CUDA_VISIBLE_DEVICES"] = GPU
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "concurrency_cost"); CRR = (OUTPUT_ROOT / "replay"); SYS_DIR = (OUTPUT_ROOT / "systems_timing")
(O / "work").mkdir(parents=True, exist_ok=True); LOGF = open(O / "run.log", "a")
_LTP = OUTPUT_ROOT / "latency_e2e/stage_d_live_timings.json"   # live medians (latency_network_measurement.py), used only by the sleep sensitivity
SLEEP_SEARCH_S, SLEEP_FETCH_S = ((round(json.load(open(_LTP))["search_api"]["p50_ms"], 2) / 1000, round(json.load(open(_LTP))["web_fetch"]["p50_ms"], 2) / 1000) if _LTP.exists() else (None, None))
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def fstat(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    st = os.stat(p); return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
sha_obj = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(); now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
LEVELS = [1, 2, 4, 8, 16, 32]; REPS = 3; WINDOW = 512
ORDER = [(rep, c, pol) for rep in range(1, REPS + 1) for c in LEVELS for pol in (("NoCache", "FreshCache") if rep % 2 else ("FreshCache", "NoCache"))]
SENS = [(1, c, pol) for c in (8, 32) for pol in ("NoCache", "FreshCache")]
DESIGN = {"status": "SAME-ENVIRONMENT CONCURRENCY on a local replayed backend; not production latency; no FreshCache parameter, capacity, batching or model changed; no configuration selected",
  "policies": {"NoCache": "every request: search (replayed URL list, local artifact, no network), fetch every own page (local snapshot read), build context, generate",
               "FreshCache": "frozen primary FreshCache decisions (stepwise_replay.Online, the component-timing study's step-wise transcription of the replay path, unbounded capacity) + query embedding (BGE-M3) + brute-force vector search over cached-entry embeddings (cost charged; decisions use the frozen similarity matrix exactly as the frozen engine) + local fetch of L3 misses + generation unless L1 hit"},
  "shared_code": "identical page read (e2e.load_snapshot_text on audit_replay.snapshot_round_on_disk), identical context assembly (audit_replay.ctx_for logic without its memo, gated byte-identical to audit_replay.ctx_for), identical generator (Llama-3.2-3B-Instruct, FORCED prompt, bf16, greedy, 80 new tokens, max_length 1024, batch 1 per request), same GPU, same process",
  "request_set": f"fixed contiguous window of {WINDOW} requests of the frozen PRIMARY stream (zipf_uniform, seed 42) starting at the first request with t >= median(t); original content, timestamps and logical order; chosen by rule before any timing",
  "initial_state": "FreshCache cache state = exact logical state after all requests before the window (Online decisions over the prefix, gated to frozen engine tiers); content needed by the window from pre-window entries (page texts, stored answers) materialised untimed before measurement; identical deep copy for every measured run; NoCache has no state",
  "concurrency_semantics": "a sequencer executes each request's cache lookup + admission in logical order (request i after i-1); fetch, embedding and generation overlap across workers; a request reusing a page version or a stored answer waits until that content exists (futures); no request can read state from a logically later request",
  "levels": LEVELS, "repetitions": REPS, "run_order": "rep 1..3 x levels ascending; policies interleaved, NoCache first on odd reps, FreshCache first on even reps",
  "measurement": "closed loop: all requests queued at start; c worker threads released by a barrier; latency = worker pickup -> completion (includes sequencer and dependency waits); throughput = N / (last completion - first pickup); per-stage timers; counts; process CPU (psutil), GPU util / memory (pynvml, 100 ms sampling), torch peak allocated, process RSS",
  "gates_every_run": "tiers and operation counts equal the logical sequential run; answers identical to the c=1 rep-1 answers of the same policy; contexts identical to audit_replay.ctx_for; any GPU-4 process other than this one is recorded (contamination flag)",
  "sensitivity": "SEPARATE from measured throughput: c in {8, 32}, 1 rep, both policies, sleeps equal to the live medians of outputs/latency_e2e/stage_d_live_timings.json (search per search call, fetch per page fetched, sequential within a request)",
  "saturation_rules": "saturation point = lowest level whose throughput gain over the previous level is < 10%; stable-p95 level = highest level with p95 <= 1.5 x its c=1 p95 (per policy, rep means)",
  "tier_diagnostic": "FreshCache-L1Only is not a configuration of the frozen engine and L3Only (method_replay baseline, no REAL_TIME bypass) has no frozen mixed-engine replay to gate against: the secondary configuration runs are NOT performed; per-tier lookup timings from the FreshCache runs are reported instead"}
DESIGN["sha256"] = sha_obj(DESIGN)
if not (O / "design_frozen.json").exists(): json.dump({**DESIGN, "written_utc": now()}, open(O / "design_frozen.json", "w"), indent=1)
assert json.load(open(O / "design_frozen.json"))["sha256"] == DESIGN["sha256"], "design changed after freezing"
log(f"[{now()}] cost-model start; design sha256 {DESIGN['sha256']}")
READ = ["freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py", "freshcache/replay_engine/method_replay.py",
        "freshcache/replay_engine/engine_loader.py", "scripts/evaluation/generation_judging_stage.py", "outputs/replay/results/replays_reported.json", "freshcache/method/tiered_cache_model.py", "scripts/evaluation/answer_grading.py",
        "data/derived/round_table.jsonl", "freshcache/simulation/request_schedules.py", "freshcache/simulation/page_version_replay.py", "data/queries.jsonl", "data/url_manifest.jsonl", "data/sim_matrix_bgem3.npy", "data/query_embeddings_bgem3.npy",
        "freshcache/simulation/stepwise_replay.py", "freshcache/simulation/reference_replay.py",
        "scripts/experiments/cost_and_latency/component_timing_gpu.py", "outputs/systems_timing/results/break_even.json",
        "outputs/systems_timing/intermediate/generation_judging_stage.json", "configs/integrity/FILE_HASHES.csv"]
BEF = {p: fstat(inpath(p)) for p in READ}
if not (O / "work/read_only_before.json").exists(): json.dump(BEF, open(O / "work/read_only_before.json", "w"), indent=1)
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
if any(p in FH and FH[p] != v["sha256"] for p, v in BEF.items()): log("STOP: frozen input differs"); sys.exit(1)
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
import numpy as np, psutil, pynvml, torch   # noqa: E402
E = C.load("reported"); exp, p2, me = E["exp"], E["p2"], E["me"]
import answer_grading as e2e   # noqa: E402
spec = importlib.util.spec_from_file_location("stepwise_replay", (ROOT / "freshcache/simulation/stepwise_replay.py")); ON = importlib.util.module_from_spec(spec); spec.loader.exec_module(ON)
PRI = C.streams(E)["primary"]; assert DS.check("primary_requests", len(PRI))
ts = [t for t, _ in PRI]; med = statistics.median(ts); i0 = next(i for i, t in enumerate(ts) if t >= med); WIN = PRI[i0:i0 + WINDOW]; PREFIX = PRI[:i0]
G = {}
_, _, TRC = C.metrics(E, PRI, "FreshCache"); mp = {"L1": "L1", "L2": "L2", "miss": "miss", "REAL_TIME_BYPASS": "REAL_TIME_BYPASS"}
ONL = ON.Online(E); pre_t = time.time()
for t, r in PREFIX:
    d = ONL.decide(t, r); assert d["tier"] == TRC[r["query_id"]]["tier"], ("prefix tier mismatch", r["query_id"])
def clone(o):   # copy cache state only; module references (exp, p2, SIM) are shared, read-only
    n = ON.Online.__new__(ON.Online); n.__dict__.update({k: v for k, v in o.__dict__.items() if k not in ('L1', 'L2', 'L3')}); n.L1 = copy.deepcopy(o.L1); n.L2 = copy.deepcopy(o.L2); n.L3 = dict(o.L3); return n
STATE0 = clone(ONL); log(f"prefix {len(PREFIX)} requests replayed logically in {time.time()-pre_t:.1f}s; window starts at index {i0}, t={WIN[0][0]:.1f}s")
# logical sequential reference for the window
REF = {}; dry = clone(STATE0); n1_0 = len(STATE0.L1["qi"]); PRE_PAGES, PRE_ANS = set(), set()
for t, r in WIN:
    d = dry.decide(t, r); REF[r["query_id"]] = d
    if d["tier"] == "L1" and d["l1_entry"] < n1_0: PRE_ANS.add(d["l1_entry"])
    for x, tv in d["ev"]:
        if tv < WIN[0][0]: PRE_PAGES.add((x, tv))
G["window decisions equal frozen engine tiers (every window request)"] = all(REF[r["query_id"]]["tier"] == TRC[r["query_id"]]["tier"] for _, r in WIN)
TIERS = collections.Counter(REF[r["query_id"]]["tier"] for _, r in WIN)
def read_page(u, tv):
    return re.sub(r"\s+", " ", e2e.load_snapshot_text(u, p2.snapshot_round_on_disk(u, tv)) or "").strip()
def assemble(texts):   # audit_replay.ctx_for body on already-read, whitespace-normalised texts (no memo)
    parts, total = [], 0
    for txt in texts:
        if len(txt) < p2.MINCHARS: continue
        room = p2.MAXCHARS - total
        if room <= 0: break
        parts.append(txt[:room]); total += min(len(txt), room)
    return "\n\n".join(parts)
REFCTX = {}
for t, r in WIN:
    d = REF[r["query_id"]]; own = [u["url_hash"] for u in r["urls"]]
    REFCTX[("NoCache", r["query_id"])] = p2.ctx_for(tuple((u, t) for u in own))
    if d["tier"] != "L1": REFCTX[("FreshCache", r["query_id"])] = p2.ctx_for(tuple(tuple(x) for x in d["ev"]))
G["context assembly (no memo) byte-identical to audit_replay.ctx_for"] = all(assemble([read_page(u, t) for u in [x["url_hash"] for x in r["urls"]]]) == REFCTX[("NoCache", r["query_id"])] for t, r in WIN)
# ---------------- models (warm once)
RUN = {n.targets[0].id: __import__("ast").literal_eval(n.value) for n in __import__("ast").parse(open((ROOT / "scripts/evaluation/generation_judging_stage.py")).read()).body if isinstance(n, __import__("ast").Assign) and isinstance(n.targets[0], __import__("ast").Name) and n.targets[0].id in ("FORCED", "MODELS")}
from sentence_transformers import SentenceTransformer   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
emb = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
tok = AutoTokenizer.from_pretrained(RUN["MODELS"]["llama3b"]); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
gm = AutoModelForCausalLM.from_pretrained(RUN["MODELS"]["llama3b"], dtype=torch.bfloat16, device_map="cuda:0"); gm.eval()
TOKL, EMBL = threading.Lock(), threading.Lock()
def generate(question, context):
    with TOKL: enc = tok([RUN["FORCED"].format(question=question, context=context)], return_tensors="pt", truncation=True, max_length=1024)
    enc = enc.to(gm.device)
    with torch.no_grad(): g = gm.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    out = g[0, enc["input_ids"].shape[1]:].cpu()
    with TOKL: return tok.decode(out, skip_special_tokens=True)
def embed(q):
    with EMBL: return emb.encode([q], show_progress_bar=False, normalize_embeddings=True)[0]
EMB = np.asarray(np.load((DATA_ROOT / "query_embeddings_bgem3.npy"), mmap_mode="r"), dtype=np.float32)
for t, r in WIN[:4]: embed(r["query"]); generate(r["query"], REFCTX[("NoCache", r["query_id"])])
torch.cuda.synchronize()
# initial content (untimed): pre-window page versions and stored answers used by the window
PRE_TEXT = {(x, tv): read_page(x, tv) for x, tv in PRE_PAGES}
PRE_ANSWER = {}
for j in sorted(PRE_ANS):
    ev = STATE0.L1["ev"][j]; PRE_ANSWER[j] = generate(STATE0.L1["q"][j], p2.ctx_for(tuple(tuple(x) for x in ev)))
live_cos = [float(np.dot(embed(r["query"]), EMB[exp._QUERY_TO_IDX[r["query"]]])) for _, r in WIN[:64]]
G["live BGE-M3 embedding equals stored embedding (min cosine >= 0.999, 64 queries)"] = min(live_cos) >= 0.999
ENV = {"host": os.environ.get("FRESHCACHE_HOST_LABEL", "local"), "cpu_count": os.cpu_count(), "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
       "load_avg_at_start": os.getloadavg(), "ram_GB": round(psutil.virtual_memory().total / 2**30, 1), "gpu_physical_index": GPU, "gpu": torch.cuda.get_device_name(0),
       "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": __import__("transformers").__version__,
       "sentence_transformers": __import__("sentence_transformers").__version__, "numpy": np.__version__, "generator": RUN["MODELS"]["llama3b"], "embedder": "BAAI/bge-m3",
       "window": {"start_index": i0, "size": len(WIN), "t_first_s": WIN[0][0], "t_last_s": WIN[-1][0], "logical_tiers": dict(TIERS), "classes": dict(collections.Counter(r["freshness_class"] for _, r in WIN)),
                  "pre_window_pages_needed": len(PRE_PAGES), "pre_window_answers_needed": len(PRE_ANS)}}
for k, v in G.items(): log("GATE", k, v)
log("ENV", json.dumps(ENV))
if not all(G.values()): json.dump({"gates": G, "env": ENV}, open(O / "work/STOPPED_setup.json", "w"), indent=1); log("STOP: setup gate failed"); sys.exit(1)
pynvml.nvmlInit(); NVH = pynvml.nvmlDeviceGetHandleByIndex(int(GPU)); ME = psutil.Process(); MYPID = os.getpid()
FILES = sorted({(u["url_hash"], t) for t, r in WIN for u in r["urls"]} | {(x, tv) for d in REF.values() for x, tv in d["ev"]})
def warm_files():
    for u, tv in FILES: read_page(u, tv)
# ---------------- one measured run
def run_once(policy, c, sens=False):
    warm_files(); state = clone(STATE0) if policy == "FreshCache" else None
    pages = {k: Future() for k in PRE_TEXT}; answers = {j: Future() for j in PRE_ANSWER}
    for k, v in PRE_TEXT.items(): pages[k].set_result(v)
    for j, v in PRE_ANSWER.items(): answers[j].set_result(v)
    FL = threading.Lock(); SEQ = threading.Condition(); turn = [0]; nxt = [0]; NL = threading.Lock(); recs = [None] * len(WIN); errs = []
    def fut(dct, k):
        with FL:
            if k not in dct: dct[k] = Future()
            return dct[k]
    def process(i):
        t, r = WIN[i]; q, qid = r["query"], r["query_id"]; own = [u["url_hash"] for u in r["urls"]]; T = collections.Counter(); st = time.perf_counter(); pc = time.perf_counter
        rec = {"i": i, "query_id": qid, "freshness_class": r["freshness_class"]}
        if policy == "NoCache":
            rec.update(tier="none", searches=1, fetches=len(own), generations=1, cache_reads=0, cache_writes=0)
            if sens: time.sleep(SLEEP_SEARCH_S)
            a0 = pc(); texts = []
            for u in own:
                texts.append(read_page(u, t))
                if sens: time.sleep(SLEEP_FETCH_S)
            T["fetch_ms"] = (pc() - a0) * 1e3; a0 = pc(); ctx = assemble(texts); T["context_ms"] = (pc() - a0) * 1e3
            a0 = pc(); ans = generate(q, ctx); T["generation_ms"] = (pc() - a0) * 1e3; rec["ctx_sha"] = hashlib.sha256(ctx.encode()).hexdigest()[:16]
        else:
            if r["freshness_class"] != "REAL_TIME":
                a0 = pc(); v = embed(q); T["embed_ms"] = (pc() - a0) * 1e3
            a0 = pc()
            with SEQ:
                while turn[0] != i: SEQ.wait()
                T["sequencer_wait_ms"] = (pc() - a0) * 1e3
                if r["freshness_class"] != "REAL_TIME":
                    b0 = pc(); n1 = len(state.L1["qi"])
                    if n1: s = EMB[np.asarray(state.L1["qi"])] @ v; np.argpartition(-s, min(5, n1 - 1))
                    T["vector_search_ms"] = (pc() - b0) * 1e3
                d = state.decide(t, r)
                for k_, v_ in d["timers_ms"].items(): T[k_ + "_ms"] = v_
                for x in d["fetch"]: fut(pages, (x, t))
                if d["tier"] != "L1" and d.get("new_l1_entry") is not None: fut(answers, d["new_l1_entry"])
                turn[0] += 1; SEQ.notify_all()
            tier = d["tier"]; nre = sum(1 for x, tv in d["ev"] if not (tv == t and x in d["fetch"])) if tier != "L1" else 0
            rec.update(tier=tier, searches=int(tier in ("miss", "REAL_TIME_BYPASS")), fetches=len(d["fetch"]), generations=int(tier != "L1"),
                       cache_reads=int(tier in ("L1", "L2")) + (nre if tier in ("L2", "miss") else 0), cache_writes=(0 if tier in ("L1", "REAL_TIME_BYPASS") else 2 + len(d["fetch"])), l3_reuses=nre)
            if sens and rec["searches"]: time.sleep(SLEEP_SEARCH_S)
            a0 = pc(); got = {}
            for x in d["fetch"]:
                got[x] = read_page(x, t)
                if sens: time.sleep(SLEEP_FETCH_S)
                if tier != "REAL_TIME_BYPASS": fut(pages, (x, t)).set_result(got[x])
            T["fetch_ms"] = (pc() - a0) * 1e3
            if tier == "L1":
                a0 = pc(); ans = fut(answers, d["l1_entry"]).result(timeout=900); T["dependency_wait_ms"] = (pc() - a0) * 1e3; rec["ctx_sha"] = None
            else:
                a0 = pc(); texts = [got[x] if (tv == t and x in got) else fut(pages, (x, tv)).result(timeout=900) for x, tv in d["ev"]]; T["dependency_wait_ms"] = (pc() - a0) * 1e3
                a0 = pc(); ctx = assemble(texts); T["context_ms"] = (pc() - a0) * 1e3
                a0 = pc(); ans = generate(q, ctx); T["generation_ms"] = (pc() - a0) * 1e3; rec["ctx_sha"] = hashlib.sha256(ctx.encode()).hexdigest()[:16]
                if d.get("new_l1_entry") is not None: fut(answers, d["new_l1_entry"]).set_result(ans)
        rec.update({k: round(v, 4) for k, v in T.items()}); rec["start"] = st; rec["end"] = time.perf_counter(); rec["latency_ms"] = (rec["end"] - st) * 1e3; rec["answer"] = ans
        return rec
    bar = threading.Barrier(c + 1)
    def worker():
        bar.wait()
        while True:
            with NL:
                i = nxt[0]; nxt[0] += 1
            if i >= len(WIN): return
            try: recs[i] = process(i)
            except Exception as ex:
                errs.append(f"{i}: {type(ex).__name__}: {ex}")
                with SEQ:
                    if turn[0] == i: turn[0] += 1; SEQ.notify_all()
    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(NVH); m = pynvml.nvmlDeviceGetMemoryInfo(NVH); pr = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(NVH)]
                samples.append((u.gpu, m.used / 2**20, ME.memory_info().rss / 2**20, psutil.cpu_percent(None), [p for p in pr if p != MYPID]))
            except Exception: pass
            time.sleep(0.1)
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(c)]
    for th in ths: th.start()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); smp = threading.Thread(target=sampler, daemon=True); psutil.cpu_percent(None)
    cpu0 = ME.cpu_times(); smp.start(); bar.wait(); w0 = time.perf_counter()
    for th in ths: th.join()
    torch.cuda.synchronize(); wall = time.perf_counter() - w0; cpu1 = ME.cpu_times(); stop.set(); smp.join()
    return recs, errs, {"wall_s": wall, "proc_cpu_s": (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system), "samples": samples, "torch_peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2**20}
def pct(v, p): v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]
BASEANS = {}; RUNS = []
def record(rep, c, pol, sens=False):
    recs, errs, M = run_once(pol, c, sens); ok = not errs and all(x is not None for x in recs)
    lat = [x["latency_ms"] for x in recs if x]; first = min(x["start"] for x in recs if x); last = max(x["end"] for x in recs if x)
    sem = {}
    if ok:
        sem["tiers_equal_logical"] = all(x["tier"] == (REF[x["query_id"]]["tier"] if pol == "FreshCache" else "none") for x in recs)
        sem["contexts_equal_ctx_for"] = all(x["ctx_sha"] == (hashlib.sha256(REFCTX[(pol, x["query_id"])].encode()).hexdigest()[:16] if (pol, x["query_id"]) in REFCTX else None) for x in recs)
        key = (pol, sens)
        if key not in BASEANS and c == 1: BASEANS[key] = {x["query_id"]: x["answer"] for x in recs}
        base = BASEANS.get(key) or BASEANS.get((pol, False))
        sem["answers_equal_c1_rep1"] = all(base[x["query_id"]] == x["answer"] for x in recs) if base else None
        sem["n_answers_differing"] = sum(base[x["query_id"]] != x["answer"] for x in recs) if base else None
    S_ = M["samples"]; others = sorted({p for s in S_ for p in s[4]})
    row = {"rep": rep, "concurrency": c, "policy": pol, "sensitivity_io_sleep": sens, "n_requests": len(WIN), "errors": len(errs), "error_detail": "; ".join(errs[:3]),
           "wall_s": round(M["wall_s"], 3), "throughput_rps": round(len(WIN) / (last - first), 4), "latency_mean_ms": round(statistics.mean(lat), 2), "latency_median_ms": round(statistics.median(lat), 2),
           "latency_p95_ms": round(pct(lat, 95), 2), "latency_p99_ms": round(pct(lat, 99), 2),
           **{f"{k}_total_ms": round(sum(x.get(k, 0) for x in recs if x), 2) for k in ("embed_ms", "vector_search_ms", "sequencer_wait_ms", "l1_lookup_ms", "l2_lookup_ms", "l3_lookup_ms", "cache_write_update_ms", "fetch_ms", "dependency_wait_ms", "context_ms", "generation_ms")},
           **{k: sum(x.get(k, 0) for x in recs if x) for k in ("searches", "fetches", "generations", "cache_reads", "cache_writes", "l3_reuses")},
           "tiers": json.dumps(dict(collections.Counter(x["tier"] for x in recs if x))), "proc_cpu_s": round(M["proc_cpu_s"], 2), "proc_cpu_util_pct_of_one_core": round(100 * M["proc_cpu_s"] / M["wall_s"], 1),
           "system_cpu_util_pct_mean": round(statistics.mean(s[3] for s in S_), 1) if S_ else None, "gpu_util_pct_mean": round(statistics.mean(s[0] for s in S_), 1) if S_ else None,
           "gpu_mem_used_MiB_peak": round(max(s[1] for s in S_), 1) if S_ else None, "torch_peak_alloc_MiB": round(M["torch_peak_alloc_MiB"], 1), "rss_MiB_peak": round(max(s[2] for s in S_), 1) if S_ else None,
           "other_gpu_processes": json.dumps(others), "load_avg_1m_end": os.getloadavg()[0], **{f"gate_{k}": v for k, v in sem.items()}}
    RUNS.append(row); log(json.dumps({k: row[k] for k in ("rep", "concurrency", "policy", "sensitivity_io_sleep", "errors", "throughput_rps", "latency_median_ms", "latency_p95_ms", "searches", "fetches", "generations", "gpu_util_pct_mean", "other_gpu_processes")} | {k: v for k, v in row.items() if k.startswith("gate_")}))
    json.dump({"env": ENV, "setup_gates": G, "runs": RUNS}, open(O / "work/runs.json", "w"), indent=1)
    return row
LOGICAL = {"FreshCache": {"searches": sum(1 for d in REF.values() if d["tier"] in ("miss", "REAL_TIME_BYPASS")), "fetches": sum(len(d["fetch"]) for d in REF.values()), "generations": sum(1 for d in REF.values() if d["tier"] != "L1")},
           "NoCache": {"searches": len(WIN), "fetches": sum(len(r["urls"]) for _, r in WIN), "generations": len(WIN)}}
ENV["logical_counts"] = LOGICAL; log("logical counts", json.dumps(LOGICAL))
for rep, c, pol in ORDER:
    try: record(rep, c, pol)
    except torch.cuda.OutOfMemoryError as ex: RUNS.append({"rep": rep, "concurrency": c, "policy": pol, "errors": 1, "error_detail": f"INFEASIBLE: CUDA OOM {ex}"[:300]}); log("INFEASIBLE", rep, c, pol); torch.cuda.empty_cache()
if SLEEP_SEARCH_S is None: log(f"sleep sensitivity skipped: {_LTP} not found (run latency_network_measurement.py with LIVE=1)")
for rep, c, pol in (SENS if SLEEP_SEARCH_S is not None else []):
    try: record(rep, c, pol, sens=True)
    except torch.cuda.OutOfMemoryError as ex: RUNS.append({"rep": rep, "concurrency": c, "policy": pol, "sensitivity_io_sleep": True, "errors": 1, "error_detail": f"INFEASIBLE: CUDA OOM {ex}"[:300]}); torch.cuda.empty_cache()
json.dump({"env": ENV, "setup_gates": G, "runs": RUNS, "finished": now()}, open(O / "work/runs.json", "w"), indent=1); log(f"[{now()}] all runs done: {len(RUNS)}")
