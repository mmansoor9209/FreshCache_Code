"""Builds the blinded 3-family model adjudication (no human annotation) for the 400 audit requests.
Copies the frozen protocol, requests and evidence of model_based_reference_validation (reference stage already run by 3 families; reused unchanged)
and extends the reported policy outputs (outputs/replay/results/model_based_policy_outputs_reported.jsonl: fresh, freshcache, semanticttl 0.60
and 0.40) with the principal comparators' answers: CategoryAware-style and quality-matched SemanticTTL (A1, theta 0.80, kappa 1/4).
Answer text is resolved from the frozen pools via the answer keys in outputs/baseline_selection/baseline_selection_request_level.csv; existing raw judge outputs
are seeded so only new (request, answer) pairs are judged. Selected operating points can be appended with --extra. Writes only into this folder."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import csv, hashlib, json, os, pathlib, shutil, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "model_adjudication")); MB = (OUTPUT_ROOT / "reference_validation"); CRR = (OUTPUT_ROOT / "replay")
sys.path.insert(0, str(ROOT / "scripts/experiments/operating_points")); import operating_point_helpers as X   # read-only answer pools (Held rule)
for d in ("config", "inputs", "outputs/raw", "outputs/parsed", "results", "logs"): (O / d).mkdir(parents=True, exist_ok=True)
for f in ("protocol.json", "protocol.sha256"): shutil.copy(MB / "config" / f, O / "config" / f)
for f in ("requests.jsonl", "evidence.jsonl"): shutil.copy(MB / "inputs" / f, O / "inputs" / f)
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
REQ = {r["query_id"]: r for r in jl(O / "inputs/requests.jsonl")}; assert DS.check("audit_sample_requests", len(REQ))
POL = [{k: r[k] for k in ("query_id", "arm", "tier", "answer_key", "answer")} for r in jl(CRR / "results/model_based_policy_outputs_reported.jsonl")]
BASE_ROWS = {r["request_id"]: r for r in csv.DictReader(open((OUTPUT_ROOT / "baseline_selection/baseline_selection_request_level.csv"), encoding="utf-8"))}
hl = X.Held(); miss = []
for arm, (tcol, kcol) in {"categoryaware_style": ("CategoryAware_style_tier", "CategoryAware_style_answer_key"), "semanticttl_qualitymatched_0.80_1-4": ("A1_tier", "A1_answer_key")}.items():
    for q in REQ:
        key = BASE_ROWS[q][kcol].split("|"); a = hl.answer(key)
        if a is None: miss.append((arm, q)); continue
        POL.append({"query_id": q, "arm": arm, "tier": BASE_ROWS[q][tcol], "answer_key": key, "answer": a})
extra = sys.argv[sys.argv.index("--extra") + 1] if "--extra" in sys.argv else None
if extra:
    for r in jl(extra): POL.append({k: r[k] for k in ("query_id", "arm", "tier", "answer_key", "answer")})
assert not miss, miss[:5]
with open(O / "inputs/policy_outputs.jsonl", "w", encoding="utf-8") as f:
    for r in POL: f.write(json.dumps(r, ensure_ascii=False) + "\n")
# seed raw outputs: original study files, then the reported-recompute GPU outputs (first occurrence wins, as reference_and_tier_recomputation.raw_answer_pool)
for j in ("llama8b", "qwen7b", "mistral7b"):
    shutil.copy(MB / "outputs/raw" / f"reference_{j}.jsonl", O / "outputs/raw" / f"reference_{j}.jsonl")
    seen, rows = set(), []
    for p in (MB / "outputs/raw" / f"answer_{j}.jsonl", (OUTPUT_ROOT / "tier_quality/outputs/raw") / f"answer_{j}.jsonl", CRR / "gpu/out" / f"answer3_{j}.jsonl"):
        for r in jl(p):
            if r["task_id"] not in seen: seen.add(r["task_id"]); rows.append(r)
    with open(O / "outputs/raw" / f"answer_{j}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
# judging: scripts/evaluation/reference_validation/reference_validation_judging.py --study model_adjudication
need = {hashlib.sha256(f"{r['query_id']}\x00{r['answer']}".encode()).hexdigest()[:16] for r in POL}
have = {j: {r["task_id"] for r in jl(O / "outputs/raw" / f"answer_{j}.jsonl")} for j in ("llama8b", "qwen7b", "mistral7b")}
print(json.dumps({"arms": dict(__import__("collections").Counter(r["arm"] for r in POL)), "distinct_answer_tasks": len(need), "to_judge": {j: len(need - h) for j, h in have.items()}}))
