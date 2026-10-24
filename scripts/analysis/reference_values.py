"""Run-time reference values shared between stages.

Some values are computed by one stage and re-derived by later stages, which must agree with it (for example
FreshCache's held-out search savings, or the fresh-correct count of the held-out answer audit). Each such value has a
name and exactly one producing script, listed in PRODUCERS below.

  record(name, value)   called only by the producing script, after it computed the value; writes it to
                        OUTPUT_DIR/reference_values.json (a re-run of the producer overwrites its entry)
  value(name)           the recorded value; stops with a message naming the producing script if it is missing
  check(name, measured) True if `measured` equals the recorded value; never records anything

A consumer stage therefore cannot run before its producer: check() and value() stop with
"run <producing script> first" instead of accepting the first value they see."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, os, pathlib, sys
OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
PATH = OUTPUT_ROOT / "reference_values.json"

_FR = "scripts/experiments/main/freshcache_replay.py"
_HS = "scripts/baselines/semantic_ttl_category_aware/heldout_split_evaluation.py"
_GA = "scripts/evaluation/answer_audits/general_audit_analysis.py"
_HSA = "scripts/evaluation/answer_audits/heldout_sensitivity_analysis.py"
_L2S = "scripts/experiments/operating_points/l2_threshold_sweep_scoring.py"
_QS8 = "scripts/evaluation/answer_audits/quality_selection_and_8b_replication.py"
_BVS = "scripts/baselines/semantic_ttl_category_aware/baseline_validation_selection.py"
_SCS = "scripts/experiments/cost_and_latency/serving_cost_summary.py"
PRODUCERS = {
    # reported-engine replays (outputs/replay/results/replays_reported.json and extras_reported.json)
    **{f"reported_engine/{s}/{v}/{m}": _FR for s in ("primary", "heldout") for v in ("FreshCache", "FreshCache_AlwaysPassTemporal")
       for m in ("search_saved_pct", "l1_hits", "l2_hits")},
    "reported_engine/heldout/SemanticTTL_0.60_k0.5/search_saved_pct": "scripts/experiments/main/supplementary_replays.py",
    "reported_engine/validation/FreshCache/search_saved_pct": "scripts/experiments/admission_controls/l2_gate_candidate_tracing.py",
    # selection-engine held-out replay (outputs/validation_split/test_results.json)
    **{f"selection_engine/heldout/FreshCache/{m}": _HS for m in ("search_saved_pct", "l1_hits", "l2_hits")},
    # held-out answer audit (outputs/general_audit/answer_audit_results.json)
    "heldout_audit/fresh_correct": _GA, "heldout_audit/FreshCache_primary/CORRECT": _GA, "heldout_audit/FreshCache_primary/WAI": _GA,
    "heldout_audit/SemanticTTL_k1_16/CORRECT": _GA, "heldout_audit/SemanticTTL_k1_16/WAI": _GA,
    # held-out sensitivity audit (outputs/quality_selection/heldout_sensitivity/heldout_sensitivity_results.json)
    "heldout_audit/SemanticTTL_0.60_k0.5_fixed/CORRECT": _HSA, "heldout_audit/SemanticTTL_0.60_k0.5_fixed/WAI": _HSA,
    "heldout_audit/SemanticTTL_0.60_k0.5_fixed/paired_vs_FreshCache_WAI": _HSA,
    # L2 operating-point audit (outputs/l2_operating_points)
    "heldout_audit/FreshCache_primary/sufficiency": _L2S, "heldout_audit/verifier_gamma_0.55/CORRECT": _L2S,
    "heldout_audit/verifier_gamma_0.55/WAI": _L2S, "heldout_audit/verifier_gamma_0.55/sufficiency": _L2S,
    # validation quality set (outputs/quality_selection, outputs/baseline_selection)
    "validation_quality/N": _QS8, "validation_quality/fresh_correct": _QS8, "validation_quality/keyset_rule/FreshCache_WAI": _QS8,
    "validation_quality/llama8b_forced_keyset/FreshCache_SemanticTTL_k1_16_WAI": _QS8,
    "validation_quality/llama8b_abstain_keyset/FreshCache_SemanticTTL_k1_16_WAI": _QS8,
    "validation_quality/FreshCache_WAI": _BVS, "validation_quality/FreshCache_correct": _BVS,
    "validation_quality/CategoryAware_original_WAI": "scripts/baselines/semantic_ttl_category_aware/baseline_heldout_scoring.py",
    # reference validation (outputs/reference_validation)
    "reference_validation/validated_subset_size": "scripts/evaluation/reference_validation/reference_validation_analysis.py",
    # serving cost (outputs/concurrency_cost/baseline_serving_cost.csv)
    "cost/FreshCache_heldout/local_ms_per_request_2dp": _SCS, "cost/FreshCache_heldout/live_p50_ms_per_request_2dp": _SCS,
    # recorded for reporting; no later stage re-derives it
    "semanticttl_audit/primary/SemanticTTL_k0.125/search_saved_pct": "scripts/evaluation/answer_audits/semanticttl_audit_sample.py",
}
_PREFIX_PRODUCERS = {"answer_quality/": "scripts/evaluation/answer_audits/answer_quality_evaluation.py",
                     "heldout_audit/predicted_class/": "scripts/evaluation/answer_audits/answer_quality_evaluation.py"}


def producer(name):
    """Repository path of the script that records `name`."""
    if name in PRODUCERS: return PRODUCERS[name]
    for p, s in _PREFIX_PRODUCERS.items():
        if name.startswith(p): return s
    raise KeyError(f"reference value '{name}' has no producing script in scripts/analysis/reference_values.py (PRODUCERS)")


def _load():
    try:
        return json.load(open(PATH, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _plain(v):
    return json.loads(json.dumps(v))   # tuples -> lists; numpy scalars must already be cast by the caller


def record(name, value):
    """Write `value` under `name`; only the producing script of `name` may call this. Returns `value`."""
    prod = producer(name); caller = pathlib.Path(sys.argv[0]).resolve().as_posix()
    if not caller.endswith(prod):
        raise SystemExit(f"reference value '{name}' is recorded by {prod}; {sys.argv[0]} may only check it")
    reg = _load(); reg[name] = {"value": _plain(value), "producer": prod}; PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp"); json.dump(reg, open(tmp, "w", encoding="utf-8"), indent=1, sort_keys=True); os.replace(tmp, PATH)
    print(f"  reference recorded: {name} = {reg[name]['value']}", flush=True)
    return value


def value(name):
    """The value recorded for `name`; stops with a message naming the producing script if it is missing."""
    reg = _load()
    if name not in reg:
        raise SystemExit(f"reference value '{name}' is missing from {PATH}: run {producer(name)} first (it records this value)")
    return reg[name]["value"]


def check(name, measured):
    """True if `measured` equals the value recorded by the producing script of `name`; records nothing."""
    return _plain(measured) == value(name)
