#!/usr/bin/env python3
"""
00_audit/audit.py -- repository inventory and integrity baseline for the
validation_V3 remaining-critical-issues programme.

Read-only. Records, for every artifact any later stage will consume:
  * SHA-256,
  * size and row count where applicable,
so section 15 can prove no original file was modified.

Also enumerates: the cluster-disjoint split, the replay engines and their
policy configurations, the existing answer-audit samples, the embedding
matrices, the snapshot round table, the external datasets and the existing
baseline implementations.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys, time

try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
os.chdir(ROOT)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    try:
        with open(p, "rb") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return None


# ---------------------------------------------------------------- inventory
ENGINES = [
    "experiment.py", "collect.py", "calibrate.py", "schedules.py",
    "v16_exp12/mixed_engine.py", "v16_exp12/schedules.py",
    "v14_baselines/engine_all.py", "v13_corrected/corrected_engine.py",
    "v9/e1_robustness.py", "e2e_answer_grading.py",
    "evolvingqa_adapter.py", "dailyqa_adapter.py", "dataset_adapter.py",
    "vcache_implementation.py",
]
DATA = [
    "data/queries.jsonl", "data/url_manifest.jsonl", "data/paraphrases.jsonl",
    "data/sim_matrix_bgem3.npy", "data/query_rich_features.json",
    "data/change_log.jsonl", "data/risk_model.pt",
    "data/evolvingqa_features.jsonl", "data/calibration_report.json",
    "v13_corrected/corrected_round_table.jsonl",
]
SPLIT = "validation/heldout_baseline_tuning/split.json"
AUDITS = [
    "validation/heldout_baseline_tuning/answer_audit_k1_16/sample.json",
    "validation/heldout_baseline_tuning/answer_audit_k1_16/answers.jsonl",
    "validation/heldout_baseline_tuning/answer_audit_k1_16/judge_llama8b.jsonl",
    "validation/heldout_baseline_tuning/answer_audit_k1_16/judge_qwen7b.jsonl",
    "validation/heldout_baseline_tuning/answer_audit_k1_16/answer_audit_results.json",
    "validation/heldout_baseline_tuning/test_results.json",
    "validation/heldout_baseline_tuning/validation_selection.json",
    "validation/heldout_baseline_tuning/validation_frontier.csv",
    "validation/mixed_age_full_policy_audit/prep2.py",
    "validation/latency_e2e_direct/run_direct.py",
    "validation/latency_e2e_direct/direct_results.json",
    "validation_V2/l2_evidence_verification/out/prereg_gamma.json",
]


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("VALIDATION_V3 -- 00_AUDIT  repository inventory and integrity baseline")
    say(f"  root {ROOT}")
    say(f"  utc  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    inv = {}
    for group, files in (("engines", ENGINES), ("data", DATA),
                         ("split", [SPLIT]), ("audits", AUDITS)):
        say(f"\n  == {group} ==")
        inv[group] = {}
        for f in files:
            p = ROOT / f
            if not p.exists():
                say(f"    MISSING  {f}")
                inv[group][f] = None
                continue
            d = {"sha256": sha(p), "bytes": p.stat().st_size}
            if p.suffix in (".jsonl", ".csv"):
                d["rows"] = rows(p)
            inv[group][f] = d
            extra = f"  rows {d['rows']:,}" if "rows" in d and d["rows"] else ""
            say(f"    {d['sha256'][:16]}  {d['bytes']:>12,}B  {f}{extra}")

    # ---- split ----
    sp = json.load(open(ROOT / SPLIT, encoding="utf-8"))
    vc, tc = set(sp["validation_clusters"]), set(sp["test_clusters"])
    say(f"\n  == cluster-disjoint split ==")
    say(f"    validation {len(vc):,} clusters / {sp['validation']['requests']:,} requests"
        f" / support {sp['validation']['support']:,}")
    say(f"    test       {len(tc):,} clusters / {sp['test']['requests']:,} requests"
        f" / support {sp['test']['support']:,}")
    say(f"    overlap    {len(vc & tc)}   (must be 0)")
    assert not (vc & tc)

    # ---- engine policy configs ----
    for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
        sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
    import engine_all as ea, mixed_engine as me, experiment as exp
    say(f"\n  == existing policy implementations ==")
    say(f"    engine_all.CONFIG  ({len(ea.CONFIG)}): {', '.join(sorted(ea.CONFIG))}")
    say(f"    mixed_engine.VARIANTS ({len(me.VARIANTS)}): {', '.join(sorted(me.VARIANTS))}")
    missing = []
    for want in ("FreshCache_L1Only", "ExactTTL", "ExactNoTTL"):
        if want not in ea.CONFIG:
            missing.append(want)
    say(f"    NOT PRESENT (to be implemented in V3): {missing}")
    say(f"\n  == frozen FreshCache parameters (experiment.py) ==")
    params = {
        "L1_SIM_THRESHOLD": exp.L1_SIM_THRESHOLD,
        "L2_SIM_THRESHOLD": exp.L2_SIM_THRESHOLD,
        "_L2_EQ_SIM_FLOOR": exp._L2_EQ_SIM_FLOOR,
        "_EQ_SIM_FLOOR": exp._EQ_SIM_FLOOR,
        "_EQ_JACCARD_MIN": exp._EQ_JACCARD_MIN,
        "EPS_ANSWER": exp.EPS_ANSWER, "EPS_URL_LIST": exp.EPS_URL_LIST,
        "EPS_CONTENT": exp.EPS_CONTENT,
        "TIER_MULT": dict(exp.TIER_MULT), "HALF_LIFE": dict(exp.HALF_LIFE),
        "FIXED_TTL": dict(exp.FIXED_TTL),
    }
    for k, v in params.items():
        say(f"    {k:<20} {v}")

    # ---- external datasets ----
    say(f"\n  == external datasets ==")
    dq = sorted((ROOT / "external" / "DailyQA" / "data" / "qa").glob("qa_*.jsonl"))
    say(f"    DailyQA qa files: {len(dq)}  ({dq[0].name} .. {dq[-1].name})" if dq
        else "    DailyQA: MISSING")
    eq = ROOT / "data" / "evolvingqa_features.jsonl"
    say(f"    EvolvingQA feature rows: {rows(eq):,}" if eq.exists()
        else "    EvolvingQA: MISSING")
    hf = pathlib.Path("~/.cache/huggingface/hub").expanduser()
    say(f"    HF cache datasets: "
        f"{[d.name for d in hf.glob('datasets--*')] if hf.exists() else 'none'}")

    # ---- snapshots ----
    say(f"\n  == snapshot rounds ==")
    import mixed_age_v2 as ma  # noqa
    say(f"    ROUNDS {getattr(ma, 'ROUNDS', 'n/a')}")
    seen = {}
    with open(ROOT / "data" / "url_manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            seen[d.get("round", "?")] = seen.get(d.get("round", "?"), 0) + 1
    for k in sorted(seen):
        say(f"    {k:<12} {seen[k]:,} manifest rows")
    say("    NO post-freeze (prospective) snapshot round is present.")

    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "root": str(ROOT), "inventory": inv,
           "split": {"validation_clusters": len(vc), "test_clusters": len(tc),
                     "overlap": len(vc & tc),
                     "validation_requests": sp["validation"]["requests"],
                     "test_requests": sp["test"]["requests"]},
           "engine_all_CONFIG": {k: v for k, v in ea.CONFIG.items()},
           "mixed_engine_VARIANTS": {k: list(v) for k, v in me.VARIANTS.items()},
           "missing_policies": missing,
           "frozen_params": params,
           "manifest_rounds": seen,
           "dailyqa_files": len(dq),
           "prospective_snapshot_present": False}
    json.dump(out, open(HERE / "audit_inventory.json", "w"), indent=2, default=str)
    (HERE / "audit.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote audit_inventory.json, audit.log")


if __name__ == "__main__":
    main()
