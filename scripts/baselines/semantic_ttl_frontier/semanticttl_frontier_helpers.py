"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

Shared helpers for the SemanticTTL frontier experiment. Every pre-existing input is imported or read READ-ONLY; nothing outside outputs/semanticttl_frontier/ is written.

Frozen components reused (not copied, not modified):
  freshcache/replay_engine/engine_loader.py   -> engine loader (replay engine first on sys.path), streams, metrics()
  freshcache/replay_engine/answer_pools.py    -> read-only answer/judgment pools
  scripts/evaluation/answer_audits/quality_selection_sample.py    -> sttl_replay (the SemanticTTL transcription used for every reported SemanticTTL answer key)
Only the in-memory globals exp.L1_SIM_THRESHOLD and exp.FIXED_TTL are set per configuration, exactly as
scripts/experiments/main/supplementary_replays.py does, and they are restored (and asserted restored) afterwards.
Every reused input is checked against the frozen hashes in configs/integrity/FILE_HASHES.csv.
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import copy, csv, hashlib, importlib.util, json, math, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); inpath = lambda p: DATA_ROOT / str(p)[5:] if str(p).startswith("data/") else OUTPUT_ROOT / str(p)[8:] if str(p).startswith("outputs/") else pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve() / p   # repository-relative path -> DATA_DIR / OUTPUT_DIR / repository
from fractions import Fraction

sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
STTL_FRONTIER_DIR = (OUTPUT_ROOT / "semanticttl_frontier")
CRR = (OUTPUT_ROOT / "replay")
sys.path.insert(0, str(ROOT / "freshcache/replay_engine"))
import engine_loader as C   # noqa: E402
import answer_pools as P    # noqa: E402

THETAS = [0.70, 0.75, 0.80, 0.85, 0.90]
KAPPAS = [0.5, 1.0, 2.0, 4.0]
def cfg_id(theta, kappa): return f"sttl_t{theta:.2f}_k{kappa:g}"
CONFIGS = [{"id": cfg_id(t, k), "theta": t, "kappa": k} for t in THETAS for k in KAPPAS]
REF_STTL = {"id": "sttl_t0.60_k0.5 (existing fixed comparator)", "theta": 0.60, "kappa": 0.5}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
FH = {r["source_path"]: r["sha256"] for r in csv.DictReader(open(ROOT / "configs/integrity/FILE_HASHES.csv")) if r["mode"] == "copied"}
INPUTS = [  # every pre-existing file this experiment reads, checked against configs/integrity/FILE_HASHES.csv
    "freshcache/replay_engine/mixed_age_replay.py", "freshcache/replay_engine/audit_replay.py",
    "freshcache/replay_engine/method_replay.py", "freshcache/replay_engine/execution_semantics.py",
    "freshcache/replay_engine/engine_loader.py", "freshcache/replay_engine/answer_pools.py", "scripts/evaluation/generation_judging_stage.py",
    "freshcache/method/tiered_cache_model.py", "freshcache/simulation/page_version_replay.py", "scripts/evaluation/answer_grading.py", "scripts/evaluation/answer_audits/quality_selection_sample.py",
    "outputs/quality_selection/q_sample.jsonl", "outputs/quality_selection/q_verification.csv",
    "outputs/quality_selection/q_answers.jsonl", "outputs/quality_selection/q_judge_llama8b.jsonl",
    "outputs/quality_selection/q_judge_qwen7b.jsonl", "outputs/quality_selection/quality_selection_prereg.json",
    "outputs/general_audit/sample.json", "outputs/general_audit/answers.jsonl",
    "outputs/general_audit/judge_llama8b.jsonl", "outputs/general_audit/judge_qwen7b.jsonl",
    "outputs/validation_split/split.json", "data/derived/round_table.jsonl", "data/queries.jsonl",
    "data/url_manifest.jsonl", "data/paraphrase_clusters.jsonl", "data/sim_matrix_bgem3.npy",
    "outputs/replay/results/quality_selection_keys_reported.json", "outputs/replay/results/audit_keys_reported.json",
    "outputs/replay/results/selection_and_8b.json", "outputs/replay/results/answer_quality.json",
    "outputs/replay/results/extras_reported.json", "outputs/replay/results/stricter_cpu_reported.json",
]
def integrity(paths=INPUTS):
    out = {}
    for p in paths:
        f = inpath(p)
        out[p] = {"sha256": sha(f), "frozen_sha256": FH.get(p), "match": FH.get(p) is not None and sha(f) == FH.get(p)}
    return out

_E = None
def engine():
    global _E
    if _E is None:
        _E = C.load("reported")
    return _E
def q2():
    E = engine()
    spec = importlib.util.spec_from_file_location("quality_selection_sample", ROOT / "scripts/evaluation/answer_audits/quality_selection_sample.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.P2 is E["p2"], (m.P2.__file__, E["p2"].__file__)
    return m

class patched:
    """Set SemanticTTL (theta, kappa) in memory exactly as scripts/experiments/main/supplementary_replays.py does; always restore."""
    def __init__(self, theta, kappa):
        self.t, self.k = theta, kappa; E = engine(); self.exp = E["exp"]
        self.orig_ttl = copy.deepcopy(self.exp.FIXED_TTL); self.orig_l1 = self.exp.L1_SIM_THRESHOLD
    def __enter__(self):
        self.exp.L1_SIM_THRESHOLD = self.t
        self.exp.FIXED_TTL.clear(); self.exp.FIXED_TTL.update({c: v * self.k for c, v in self.orig_ttl.items()})
        return self
    def __exit__(self, *a):
        self.exp.FIXED_TTL.clear(); self.exp.FIXED_TTL.update(self.orig_ttl); self.exp.L1_SIM_THRESHOLD = self.orig_l1
        assert self.exp.FIXED_TTL == self.orig_ttl and self.exp.L1_SIM_THRESHOLD == self.orig_l1

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(Fraction(1), Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 2, 2 ** n)))
def jl(p): return P.jl(p)
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
h16 = P.h16
