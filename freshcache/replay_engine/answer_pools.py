"""Read-only answer_pools of existing outputs, indexed by content. Earlier files win on conflicts; the recorded audits come first."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os
import json, pathlib, hashlib
R = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); at = lambda f: pathlib.Path(f) if str(f).startswith("/") else OUTPUT_ROOT / str(f)[8:] if str(f).startswith("outputs/") else R / f   # pool path -> OUTPUT_DIR
def jl(p):
    p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
ANSWER_FILES = [  # all Llama-3.2-3B-Instruct, FORCED prompt, greedy, 80 tokens, clipped at the first period
    "outputs/general_audit/answers.jsonl", "outputs/mixed_age_audit/answers2.jsonl",
    "outputs/quality_selection/heldout_sensitivity/h_answers.jsonl", "outputs/semanticttl_audit/sttl_answers.jsonl",
    "outputs/predicted_class_audit/new_answers.jsonl", "outputs/quality_selection/q_answers.jsonl", "outputs/semanticttl_joint_grid/new_answers.jsonl",
    "outputs/l1_precision/heldout_answers_llama3b.jsonl", "outputs/l1_precision/validation_answers_llama3b.jsonl",
    "outputs/l1_strict_gate/strict_gate_audit/answers_llama3b.jsonl", "outputs/l1only_exactttl/l1_hit_analysis/l1_answers.jsonl",
    "outputs/l1only_exactttl/stronger_generator/answers_llama3b.jsonl", "outputs/parameter_calibration/answer_quality/answers_llama3b.jsonl",
    "outputs/l2_verifier/out/audit_answers.jsonl", "outputs/tier_quality/outputs/raw/generated_answers.jsonl",
    "outputs/l1_validation/outputs/gpu/answers_llama3b.jsonl"]
JUDGE2_DIRS = ["outputs/general_audit", "outputs/mixed_age_audit", "outputs/quality_selection/heldout_sensitivity",
    "outputs/semanticttl_audit", "outputs/predicted_class_audit", "outputs/quality_selection", "outputs/semanticttl_joint_grid",
    "outputs/l1_precision", "outputs/l1_strict_gate/strict_gate_audit", "outputs/l1only_exactttl/l1_hit_analysis",
    "outputs/l1only_exactttl/stronger_generator", "outputs/parameter_calibration/answer_quality", "outputs/l2_verifier/out",
    "outputs/l1_validation/outputs/gpu"]
JURY_FILES = ["outputs/l1_precision/heldout_equiv_{j}.jsonl", "outputs/l1_precision/validation_equiv_{j}.jsonl",
    "outputs/l1_strict_gate/strict_gate_audit/equiv_{j}.jsonl", "outputs/l1only_exactttl/l1_hit_analysis/l1_equiv_{j}.jsonl",
    "outputs/l1_validation/outputs/gpu/equiv_{j}.jsonl"]
J3_DIRS = ["outputs/reference_validation/outputs/parsed", "outputs/tier_quality/outputs/parsed"]
def answers(extra=()):
    pool = {}
    for f in ANSWER_FILES + list(extra):
        for r in jl(at(f)):
            k = (r["key"][1], r["key"][2]) if "key" in r else (r.get("ctx_sha"), r.get("q_sha"))
            if k[0] and k not in pool: pool[k] = r["answer"]
    return pool
def judge2(extra=()):
    pool = {"llama8b": {}, "qwen7b": {}}
    for d in JUDGE2_DIRS + list(extra):
        for p in sorted(at(d).glob("*judge*_*.jsonl")):
            for j in pool:
                if p.stem.endswith(j):
                    for r in jl(p):
                        if "sig" in r and r.get("label") in ("CORRECT", "INCORRECT"): pool[j].setdefault(tuple(r["sig"]), r["label"])
    return pool
def jury(extra=()):
    pool = {}
    for j in ("llama3b", "llama8b", "qwen7b", "mistral7b"):
        for f in JURY_FILES + list(extra):
            for r in jl(at(f.format(j=j))):
                if r.get("label") in ("SAME", "DIFFERENT"):
                    k = r.get("pair_id") or ("qid", r.get("qid"))   # l1_hit_analysis is keyed by qid of the keyset hits
                    pool.setdefault(k, {}).setdefault(j, r["label"])
    return pool
def judge3(extra=()):
    pool = {}
    for d in J3_DIRS + list(extra):
        for j in ("llama8b", "qwen7b", "mistral7b"):
            for r in jl(at(d)/f"answer_{j}.jsonl"): pool.setdefault(r["task_id"], {}).setdefault(j, r["label"])
    return pool
def suff3(extra=()):
    pool = {}
    for d in ["outputs/tier_quality/outputs/parsed"] + list(extra):
        for j in ("llama8b", "qwen7b", "mistral7b"):
            for r in jl(at(d)/f"sufficiency_{j}.jsonl"): pool.setdefault(r["task_id"], {}).setdefault(j, r["label"])
    return pool
def h16(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
def tid(qid, text): return hashlib.sha256(f"{qid}\x00{text}".encode()).hexdigest()[:16]
def answers_exact(extra=()):
    """Exact (role, ctx, q) pool, first file wins (ANSWER_FILES order), plus the content pool as fallback."""
    ex = {}
    for f in ANSWER_FILES + list(extra):
        for r in jl(at(f)):
            if "key" in r: ex.setdefault(tuple(r["key"]), r["answer"])
    return ex
class AnswerLookup:
    def __init__(self, extra=()): self.ex = answers_exact(extra); self.ct = answers(extra)
    def get(self, key):
        return self.ex.get(tuple(key)) if tuple(key) in self.ex else self.ct.get((key[1], key[2]))
