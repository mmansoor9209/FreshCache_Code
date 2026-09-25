#!/usr/bin/env python3
"""P1 -- F08 transition matrices, F10 failure cross-tab, F26 extended
row-level reproducibility harness.

All three read row-level artifacts already on disk. No replay, no generation,
no judging, no network. Every published aggregate is RE-DERIVED from its row
file and compared against the stored summary; a mismatch is reported, never
silently reconciled.
"""
from __future__ import annotations
import csv, hashlib, json, pathlib, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
LAB3 = ["CORRECT", "WRONG", "ABSTAIN"]
log, checks = [], []


def say(s=""):
    print(s, flush=True); log.append(s)


def rows_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def check(name, got, want, detail=""):
    ok = got == want
    checks.append({"check": name, "status": "PASS" if ok else "FAIL",
                   "recomputed": got, "published": want, "detail": detail})
    say(f"    {'PASS' if ok else 'FAIL'}  {name:<62} "
        f"row-level {got!s:>12}  published {want!s:>12}")
    return ok


def matrix(rs, fkey, ckey):
    m = Counter((r[fkey], r[ckey]) for r in rs)
    off = {k: v for k, v in m.items() if k[0] not in LAB3 or k[1] not in LAB3}
    return m, off


def show(title, m, n):
    say(f"\n  {title} (n={n})   rows = fresh path, cols = cached path")
    say("    " + "fresh\\cached".rjust(14)
        + "".join(c.rjust(10) for c in LAB3) + "total".rjust(8))
    for f in LAB3:
        v = [m.get((f, c), 0) for c in LAB3]
        say("    " + f.rjust(14) + "".join(str(x).rjust(10) for x in v)
            + str(sum(v)).rjust(8))
    tot = [sum(m.get((f, c), 0) for f in LAB3) for c in LAB3]
    say("    " + "total".rjust(14) + "".join(str(x).rjust(10) for x in tot)
        + str(sum(tot)).rjust(8))


# ===================================================================== F08
def f08():
    say("=" * 78)
    say("F08 -- fresh-vs-cached 3x3 transition matrices")
    say("=" * 78)
    src = ROOT / "v16_validation_wai_depth" / "wai_by_tier.json"
    d = json.load(open(src, encoding="utf-8"))
    say(f"  source {src.relative_to(ROOT)}  sha256 "
        f"{hashlib.sha256(src.read_bytes()).hexdigest()[:16]}")
    say(f"  judge rule: {d['judge_rule']}")
    rows = d["rows"]
    out = {}
    say(f"\n  -- verification of every margin against the stored tier summary --")
    for tier in ("L1", "L2", "ALL"):
        rs = [r for r in rows if tier == "ALL" or r["tier"] == tier]
        m, off = matrix(rs, "fresh", "cached")
        t = d["tiers"][tier]
        show(tier, m, len(rs))
        if off:
            say(f"    OFF-GRID LABELS: {off}")
        ok = all([
            check(f"{tier}: n", len(rs), t["graded_n"]),
            check(f"{tier}: cached CORRECT", sum(m.get((f, 'CORRECT'), 0) for f in LAB3), t["cached_correct"]),
            check(f"{tier}: cached WRONG", sum(m.get((f, 'WRONG'), 0) for f in LAB3), t["cached_wrong"]),
            check(f"{tier}: cached ABSTAIN", sum(m.get((f, 'ABSTAIN'), 0) for f in LAB3), t["cached_abstain"]),
            check(f"{tier}: fresh CORRECT", sum(m.get(('CORRECT', c), 0) for c in LAB3), t["fresh_correct"]),
            check(f"{tier}: fresh WRONG", sum(m.get(('WRONG', c), 0) for c in LAB3), t["fresh_wrong"]),
            check(f"{tier}: fresh ABSTAIN", sum(m.get(('ABSTAIN', c), 0) for c in LAB3), t["fresh_abstain"]),
            check(f"{tier}: WAI cell (cached WRONG & fresh CORRECT)", m.get(("CORRECT", "WRONG"), 0), t["induced_wrong"]),
            check(f"{tier}: cache-helped cell", m.get(("WRONG", "CORRECT"), 0), t["cache_helped"]),
        ])
        out[tier] = {"n": len(rs), "all_margins_verified": ok,
                     "matrix": {f: {c: m.get((f, c), 0) for c in LAB3} for f in LAB3},
                     "off_grid": {str(k): v for k, v in off.items()}}
    # secondary population, for completeness
    aq = HERE.parent / "parameter_calibration" / "07_answer_quality" / "answer_quality_rows.csv"
    if aq.exists():
        rs = rows_csv(aq)
        m, off = matrix(rs, "fresh", "A_original")
        show("held-out 400-request audit (FreshCache published arm)", m, len(rs))
        out["heldout_audit_400"] = {
            "n": len(rs),
            "matrix": {f: {c: m.get((f, c), 0) for c in LAB3} for f in LAB3},
            "off_grid": {str(k): v for k, v in off.items()},
            "note": "different population from the 325 graded reuse events; "
                    "reported separately, never merged"}
    return out


# ===================================================================== F10
def f10():
    say("\n" + "=" * 78)
    say("F10 -- reconciled fresh-path failure decomposition")
    say("=" * 78)
    a = rows_csv(ROOT / "validation" / "fresh_path_failure_audit"
                 / "fresh_path_failure_per_request.csv")
    b = rows_csv(V3 / "strong_reference_evaluation" / "02_reference_verification"
                 / "reference_and_diagnosis.csv")
    say(f"  taxonomy A (prior 4-way) and B (audit 5-way): "
        f"{len(a):,} rows over {len(set(x['audit'] for x in a))} audits")
    say(f"  taxonomy C (evidence-based): {len(b):,} rows")
    out = {}
    for aud in ("mixed_age_400", "heldout_400"):
        rs = [x for x in a if x["audit"] == aud]
        pa, pb = Counter(x["prior_category"] for x in rs), Counter(x["category"] for x in rs)
        check(f"{aud}: taxonomy A sums to 400", sum(pa.values()), 400)
        check(f"{aud}: taxonomy B sums to 400", sum(pb.values()), 400)
        ct = Counter((x["prior_category"], x["category"]) for x in rs)
        say(f"\n  {aud}: A x B cross-tab")
        for (x, y), v in sorted(ct.items(), key=lambda kv: -kv[1]):
            say(f"    {x:<32} -> {y:<34} {v:>4}")
        out[aud] = {"A": dict(pa), "B": dict(pb),
                    "A_x_B": {f"{x} -> {y}": v for (x, y), v in ct.items()}}
    # three-way on the held-out 400
    ah = {x["query_id"]: x for x in a if x["audit"] == "heldout_400"}
    bh = {x["query_id"]: x for x in b}
    common = set(ah) & set(bh)
    check("held-out 400 joins A/B rows to taxonomy C on query_id",
          len(common), 400)
    pc = Counter(bh[q]["primary_category"] for q in common)
    check("taxonomy C sums to 400", sum(pc.values()), 400)
    say(f"\n  taxonomy C: {dict(pc)}")
    tri = Counter((ah[q]["category"], bh[q]["primary_category"]) for q in common)
    say(f"\n  held-out: taxonomy B x taxonomy C cross-tab")
    say("    " + "B (audit 5-way)".ljust(36) + "C (evidence-based)".ljust(34) + "n")
    for (x, y), v in sorted(tri.items(), key=lambda kv: -kv[1]):
        say(f"    {x:<36}{y:<34}{v:>4}")
    out["heldout_B_x_C"] = {f"{x} -> {y}": v for (x, y), v in tri.items()}
    out["taxonomy_C"] = dict(pc)
    return out


# ===================================================================== F26
def f26():
    say("\n" + "=" * 78)
    say("F26 -- row-level reproducibility, extended to the four later workstreams")
    say("=" * 78)

    # ---- strong_reference_evaluation ----
    say(f"\n  strong_reference_evaluation")
    b = rows_csv(V3 / "strong_reference_evaluation" / "02_reference_verification"
                 / "reference_and_diagnosis.csv")
    js = json.load(open(V3 / "strong_reference_evaluation" / "02_reference_verification"
                        / "reference_and_diagnosis.json", encoding="utf-8"))
    st = Counter(x["reference_status_final"] for x in b)
    for k, v in js["reference_status"].items():
        check(f"sre: reference_status {k}", st.get(k, 0), v)
    fcm = Counter(x["primary_category"] for x in b)
    for k, v in js["failure_categories"].items():
        check(f"sre: failure_category {k}", fcm.get(k, 0), v)

    # ---- observability_resolution ----
    say(f"\n  observability_resolution")
    orr = json.load(open(V3 / "observability_resolution" / "02_recovery"
                         / "recovery_results.json", encoding="utf-8"))
    pr = rows_csv(V3 / "observability_resolution" / "02_recovery"
                  / "per_request_recovered.csv")
    OBS = ("CHANGED", "UNCHANGED")
    for thr, kon, koff in (("400", "outcome_on_thr400", "outcome_off_thr400"),
                           ("50", "outcome_on_thr50", "outcome_off_thr50")):
        # the published denominator is the PAIRED set: both arms reused. 132
        # requests reuse under gate-off only and are excluded by design.
        pair = [x for x in pr if x[kon] and x[koff]]
        jo = [x for x in pair if x[kon] in OBS and x[koff] in OBS]
        cell = orr["by_threshold"][thr]
        check(f"obs: paired N @MIN_BODY={thr}", len(pair), cell["N_paired"])
        check(f"obs: jointly observable @MIN_BODY={thr}", len(jo),
              cell["jointly_observable"])
        check(f"obs: changed_on @MIN_BODY={thr}",
              sum(1 for x in pair if x[kon] == "CHANGED"), cell["changed_on"])
        check(f"obs: changed_off @MIN_BODY={thr}",
              sum(1 for x in pair if x[koff] == "CHANGED"), cell["changed_off"])
        d = round(100*sum(1 for x in jo if x[koff]=="CHANGED")/len(jo)
                  - 100*sum(1 for x in jo if x[kon]=="CHANGED")/len(jo), 4)
        check(f"obs: observed delta pp @MIN_BODY={thr}", d,
              cell["observed_delta_pp"])

    # ---- parameter_calibration ----
    say(f"\n  parameter_calibration")
    hr = json.load(open(V3 / "parameter_calibration" / "05_heldout"
                        / "heldout_results.json", encoding="utf-8"))
    hp = rows_csv(V3 / "parameter_calibration" / "05_heldout"
                  / "heldout_per_request.csv")
    for cfg, v in hr["configs"].items():
        det = [x for x in hp if x[cfg] in OBS]
        check(f"pc: {cfg} observable n", len(det), v["observable_n"])
        check(f"pc: {cfg} drift pct",
              round(100*sum(1 for x in det if x[cfg]=="CHANGED")/len(det), 4),
              v["drift_pct"])
    aq = json.load(open(V3 / "parameter_calibration" / "07_answer_quality"
                        / "answer_quality.json", encoding="utf-8"))
    ar = rows_csv(V3 / "parameter_calibration" / "07_answer_quality"
                  / "answer_quality_rows.csv")
    for cfg, v in aq["per_config"].items():
        check(f"pc: {cfg} CORRECT", sum(1 for x in ar if x[cfg]=="CORRECT"),
              v["counts"].get("CORRECT", 0))
        check(f"pc: {cfg} WAI",
              sum(1 for x in ar if x[cfg]=="WRONG" and x["fresh"]=="CORRECT"),
              v["wai"])

    # ---- baseline_fidelity_resolution ----
    say(f"\n  baseline_fidelity_resolution")
    ej = json.load(open(V3 / "baseline_fidelity_resolution" / "03_exactttl_mixed"
                        / "exactttl_mixed_results.json", encoding="utf-8"))
    er = rows_csv(V3 / "baseline_fidelity_resolution" / "03_exactttl_mixed"
                  / "exactttl_mixed_per_request.csv")
    row = ej["exactttl_row"]
    check("bfr: ExactTTL n_requests", len(er), row["n_requests"])
    check("bfr: ExactTTL L1 hits", sum(1 for x in er if x["tier"] == "L1"),
          row["l1_hits"])
    ins = [x for x in er if x["in_support"] == "1" and x["outcome"] in OBS]
    check("bfr: ExactTTL determinable", len(ins), row["determinable"])
    check("bfr: ExactTTL changed",
          sum(1 for x in ins if x["outcome"] == "CHANGED"), row["changed"])
    return {"n_checks": len(checks),
            "n_fail": sum(1 for c in checks if c["status"] == "FAIL")}


if __name__ == "__main__":
    r08 = f08(); r10 = f10(); r26 = f26()
    nf = sum(1 for c in checks if c["status"] == "FAIL")
    say("\n" + "=" * 78)
    say(f"  TOTAL row-level checks: {len(checks)}   PASS "
        f"{len(checks)-nf}   FAIL {nf}")
    if nf:
        say("  FAILING CHECKS (numerical contradictions):")
        for c in checks:
            if c["status"] == "FAIL":
                say(f"    - {c['check']}: row-level {c['recomputed']} vs "
                    f"published {c['published']}")
    json.dump({"f08": r08, "f10": r10, "f26": r26, "checks": checks,
               "n_checks": len(checks), "n_fail": nf},
              open(HERE / "out" / "p1_results.json", "w"), indent=2)
    (HERE / "out" / "p1.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote out/p1_results.json, out/p1.log")
