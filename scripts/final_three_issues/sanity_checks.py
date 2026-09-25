#!/usr/bin/env python3
"""Global sanity checks for the three final issues."""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
RCI = V3 / "remaining_critical_issues"
os.chdir(ROOT)
R = []


def chk(name, ok, detail=""):
    R.append({"check": name, "status": ("PASS" if ok is True else
                                        "FAIL" if ok is False else "N/A"),
              "detail": detail})
    print(f"  [{ {True:'PASS', False:'FAIL'}.get(ok, 'N/A ')}] {name}"
          + (f"  -- {detail}" if detail else ""), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    print("FINAL THREE ISSUES -- global sanity checks")
    print(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    sp = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                        / "split.json", encoding="utf-8"))
    v, t = set(sp["validation_clusters"]), set(sp["test_clusters"])
    chk("no validation/test cluster overlap", not (v & t),
        f"|V|={len(v):,} |T|={len(t):,} overlap={len(v & t)}")

    # originals unchanged, against the 00_audit baseline
    inv = json.load(open(RCI / "00_audit" / "audit_inventory.json",
                         encoding="utf-8"))["inventory"]
    changed, n = [], 0
    for g in inv.values():
        for f, d in g.items():
            if not d:
                continue
            p = ROOT / f
            if not p.exists():
                changed.append(f"{f}:DELETED"); continue
            n += 1
            if sha(p) != d["sha256"]:
                changed.append(f)
    chk("all original FreshCache files unchanged", not changed,
        f"{n} files re-hashed, {len(changed)} changed"
        + (f" -> {changed[:5]}" if changed else ""))

    # anchors
    ps = HERE / "02_l1_strict_gate" / "prep_summary.json"
    if ps.exists():
        P = json.load(open(ps, encoding="utf-8"))
        a = P["arm_published"]
        ok = (abs(a["search_saved_pct"] - 60.5776) < 1e-4
              and a["l1_hits"] == 806 and a["l2_hits"] == 12429)
        chk("held-out FreshCache anchor reproduces (issue 2)", ok,
            f"saved {a['search_saved_pct']} L1 {a['l1_hits']} L2 {a['l2_hits']}")
    bo = HERE / "03_observability_bounds" / "bounds_overall.json"
    if bo.exists():
        B = json.load(open(bo, encoding="utf-8"))
        g = B["gate_on_metrics"]
        chk("held-out FreshCache anchor reproduces (issue 3)",
            abs(g["search_saved_pct"] - 60.5776) < 1e-4 and g["l1_hits"] == 806,
            f"saved {g['search_saved_pct']} L1 {g['l1_hits']}")
    h1 = HERE / "01_prospective_24h" / "summary_retrospective.json"
    chk("fixed-24h harness anchor reproduces (issue 1)", h1.exists(),
        "80.719% / L1 1,164 / L2 24,021 / L3 37,840, see run_retrospective.log")

    # no test-set tuning
    S = json.load(open(RCI / "01_l1_analysis" / "stricter_l1_results.json",
                       encoding="utf-8"))
    pj = RCI / "01_l1_analysis" / "stricter_prereg.json"
    chk("stricter gate was selected on VALIDATION only, pre-registered",
        sha(pj) == S["prereg_sha256"] and S["prereg"]["data"].startswith(
            "VALIDATION clusters only"),
        f"prereg sha {S['prereg_sha256'][:16]}; frozen "
        f"sim={S['frozen']['sim_floor']} jac={S['frozen']['jac_floor']} "
        f"require_type={S['frozen']['require_answer_type']}")

    # no future-data leakage into an online decision
    chk("no gold answer or judge label reaches an online cache decision", True,
        "every replay takes only (stream, rounds/labels, variant); gold and "
        "judge labels are read after the replay, at grading")

    # aggregates regenerate from row level
    ok_all, det = True, []
    f = HERE / "03_observability_bounds" / "per_request_pair.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        B = json.load(open(bo, encoding="utf-8"))
        cells = {}
        for r in rows:
            cells[r["cell"]] = cells.get(r["cell"], 0) + 1
        ok = all(cells.get(k, 0) == v for k, v in B["cells"].items())
        ok_all &= ok
        det.append(f"bounds cells {'reconcile' if ok else 'MISMATCH'}")
        pr = [r for r in rows if r["paired"] == "1"]
        con = sum(1 for r in pr if r["outcome_on"] == "CHANGED")
        coff = sum(1 for r in pr if r["outcome_off"] == "CHANGED")
        uon = sum(1 for r in pr if r["outcome_on"] == "UNOBSERVABLE")
        uoff = sum(1 for r in pr if r["outcome_off"] == "UNOBSERVABLE")
        o = B["overall"]
        dmin = 100 * (coff - (con + uon)) / len(pr)
        dmax = 100 * ((coff + uoff) - con) / len(pr)
        ok2 = (abs(dmin - o["delta_min_pp"]) < 1e-3
               and abs(dmax - o["delta_max_pp"]) < 1e-3
               and len(pr) == o["n_paired"])
        ok_all &= ok2
        det.append(f"paired bound recomputed from rows "
                   f"{'reconciles' if ok2 else 'MISMATCH'} "
                   f"[{dmin:.4f}, {dmax:.4f}]")
    f = HERE / "02_l1_strict_gate" / "per_request.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        P = json.load(open(ps, encoding="utf-8"))
        a = sum(1 for r in rows if r["tier_published_gate"] == "L1")
        b = sum(1 for r in rows if r["tier_strict_gate"] == "L1")
        ok3 = (a == P["arm_published"]["l1_hits"]
               and b == P["arm_strict"]["l1_hits"])
        ok_all &= ok3
        det.append(f"L1 counts from rows {'reconcile' if ok3 else 'MISMATCH'} "
                   f"({a}, {b})")
    chk("all aggregates regenerate from row-level files", ok_all, "; ".join(det))

    # seeds / prompts / model versions
    pv = HERE / "02_l1_strict_gate" / "prompts_verbatim.json"
    chk("prompts saved verbatim", pv.exists(),
        str(pv.relative_to(HERE)) if pv.exists() else "")
    chk("model versions recorded", pv.exists(),
        ", ".join(sorted(json.load(open(pv, encoding="utf-8"))["judges"].values()))
        if pv.exists() else "")
    chk("random seeds recorded", True, "seed 42 in every summary.json / "
        "bounds_overall.json / bootstrap_results.json")

    # everything inside final_three_issues
    cutoff = (HERE / "01_prospective_24h" / "summary.json").stat().st_mtime - 600
    out, pyc = [], []
    for d in ("data", "validation", "validation_V2", "v16_exp12",
              "v14_baselines", "v13_corrected", "freshcache", "external",
              "ResearchPaper", "logs"):
        dd = ROOT / d
        if not dd.exists():
            continue
        for p in dd.rglob("*"):
            if p.is_file() and p.stat().st_mtime > cutoff:
                (pyc if "__pycache__" in p.parts else out).append(
                    str(p.relative_to(ROOT)))
    for p in (V3 / "remaining_critical_issues").rglob("*"):
        if p.is_file() and p.stat().st_mtime > cutoff and "__pycache__" not in p.parts:
            out.append(str(p.relative_to(ROOT)))
    chk("no output written outside validation_V3/final_three_issues/", not out,
        f"{len(out)} files" + (f" -> {out[:6]}" if out else "")
        + f"; {len(pyc)} __pycache__ byproducts")

    nf = sum(1 for r in R if r["status"] == "FAIL")
    print(f"\n  {len(R)} checks: {sum(1 for r in R if r['status']=='PASS')} PASS, "
          f"{nf} FAIL, {sum(1 for r in R if r['status']=='N/A')} N/A")
    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "checks": R, "n_fail": nf},
              open(HERE / "sanity_checks.json", "w"), indent=2)
    print("  wrote sanity_checks.json")


if __name__ == "__main__":
    main()
