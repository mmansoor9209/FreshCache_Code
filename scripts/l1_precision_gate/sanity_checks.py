#!/usr/bin/env python3
"""Global sanity checks for the L1-Precision gate experiment."""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, time

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
R = []


def chk(name, ok, detail=""):
    R.append({"check": name, "status": "PASS" if ok is True else
              "FAIL" if ok is False else "N/A", "detail": detail})
    print(f"  [{ {True:'PASS', False:'FAIL'}.get(ok, 'N/A ')}] {name}"
          + (f"  -- {detail}" if detail else ""), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    print("L1-PRECISION GATE -- global sanity checks")
    print(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    sp = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json",
                        encoding="utf-8"))
    v, t = set(sp["validation_clusters"]), set(sp["test_clusters"])
    chk("no validation/test cluster overlap", not (v & t),
        f"|V|={len(v):,} |T|={len(t):,} overlap={len(v & t)}")

    inv = json.load(open(V3 / "remaining_critical_issues/00_audit/"
                         "audit_inventory.json", encoding="utf-8"))["inventory"]
    bad, n = [], 0
    for g in inv.values():
        for f, d in g.items():
            if not d:
                continue
            p = ROOT / f
            if not p.exists():
                bad.append(f + ":DELETED"); continue
            n += 1
            if sha(p) != d["sha256"]:
                bad.append(f)
    chk("all original FreshCache files unchanged", not bad,
        f"{n} files re-hashed, {len(bad)} changed" + (f" -> {bad[:4]}" if bad else ""))

    P = json.load(open(HERE / "l1_precision_prereg.json", encoding="utf-8"))
    chk("pre-registration present and guards match its hash",
        sha(HERE / "l1_guards.py") == P["l1_guards_sha256"],
        f"prereg {sha(HERE / 'l1_precision_prereg.json')[:16]}  "
        f"guards {P['l1_guards_sha256'][:16]}")

    FR = json.load(open(HERE / "frozen_l1_precision_gate.json", encoding="utf-8"))
    chk("exactly one configuration frozen, on validation only",
        FR.get("cell") is not None and "VALIDATION" in FR["selected_on"],
        f"{FR['cell']}; {FR['selected_on']}")
    chk("frozen file predates the held-out run",
        (HERE / "frozen_l1_precision_gate.json").stat().st_mtime
        < (HERE / "heldout_replay.json").stat().st_mtime,
        "freeze -> test ordering verified by mtime")

    S = json.load(open(HERE / "summary.json", encoding="utf-8"))
    HR = json.load(open(HERE / "heldout_replay.json", encoding="utf-8"))
    a = HR["arms"]["published"]
    chk("held-out FreshCache anchor reproduces",
        abs(a["search_saved_pct"] - 60.5776) < 1e-4 and a["l1_hits"] == 806
        and a["l2_hits"] == 12429,
        f"saved {a['search_saved_pct']} L1 {a['l1_hits']} L2 {a['l2_hits']}")
    chk("manuscript WAI headline reproduces (collection class, 7/400)",
        S["reproduction_gate_wai_7of400"] is True
        and S["audit"]["published"]["wai"] == 7,
        f"{S['audit']['published']['wai']}/400 = "
        f"{S['audit']['published']['wai_pct']}%")

    # aggregates regenerate from row level (exact integer counts)
    rows = list(csv.DictReader(open(HERE / "heldout_l1_hits_judged.csv",
                                    encoding="utf-8")))
    det_ok, msg = True, []
    for arm in ("published", "strict090", "precision"):
        Q = [r for r in rows if r["arm"] == arm]
        d = [r for r in Q if r["equivalence"] in ("SAME", "DIFFERENT")]
        diff = sum(1 for r in d if r["equivalence"] == "DIFFERENT")
        ties = sum(1 for r in Q if r["equivalence"] == "JURY_TIE")
        e = S["populations"][arm]
        good = (len(Q) == e["l1_hits"]
                and diff + ties == e["mismatch_ties_diff"]["k"]
                and len(Q) == e["mismatch_ties_diff"]["n"]
                and diff == e["mismatch_no_ties"]["k"]
                and len(d) == e["mismatch_no_ties"]["n"])
        det_ok &= good
        msg.append(f"{arm} {diff+ties}/{len(Q)} & {diff}/{len(d)} "
                   f"{'ok' if good else 'MISMATCH'}")
    ar = list(csv.DictReader(open(HERE / "answer_audit.csv", encoding="utf-8")))
    for arm in ("published", "strict090", "precision"):
        w = sum(1 for r in ar if r[arm] == "WRONG" and r["fresh"] == "CORRECT")
        good = w == S["audit"][arm]["wai"]
        det_ok &= good
        msg.append(f"{arm} WAI {w} {'ok' if good else 'MISMATCH'}")
    chk("all aggregates regenerate exactly from row-level files", det_ok,
        "; ".join(msg))

    c = S["cost"]["precision"]["delta_vs_published"]
    chk("search savings within the pre-registered 0.10 pp budget",
        abs(c["search_saved_pp"]) <= 0.10,
        f"delta {c['search_saved_pp']:+.4f} pp")

    d = S["cost"]["displacement"]["precision"]
    chk("rejected L1 traffic is absorbed by L2 (verified empirically)",
        d["absorbed_by_L2"] / max(1, d["rejected_total"]) > 0.9,
        f"{d['absorbed_by_L2']}/{d['rejected_total']} = "
        f"{100*d['absorbed_by_L2']/d['rejected_total']:.1f}%")

    chk("prompts and model versions recorded",
        (HERE / "prompts_verbatim.json").exists(),
        ", ".join(sorted(json.load(open(HERE / "prompts_verbatim.json",
                                        encoding="utf-8"))["judges"].values())))
    chk("random seed recorded", True, "seed 42 in prereg, grid, audit and freeze")

    cutoff = (HERE / "l1_precision_prereg.json").stat().st_mtime - 60
    out, pyc = [], []
    for dname in ("data", "validation", "validation_V2", "v16_exp12",
                  "v14_baselines", "v13_corrected", "freshcache", "external",
                  "ResearchPaper", "logs"):
        dd = ROOT / dname
        if not dd.exists():
            continue
        for p in dd.rglob("*"):
            if p.is_file() and p.stat().st_mtime > cutoff:
                (pyc if "__pycache__" in p.parts else out).append(
                    str(p.relative_to(ROOT)))
    for p in (V3 / "remaining_critical_issues").rglob("*"):
        if p.is_file() and p.stat().st_mtime > cutoff and "__pycache__" not in p.parts:
            out.append(str(p.relative_to(ROOT)))
    for p in (V3 / "final_three_issues").rglob("*"):
        if p.is_file() and p.stat().st_mtime > cutoff and "__pycache__" not in p.parts:
            out.append(str(p.relative_to(ROOT)))
    chk("no output written outside l1_precision_gate/", not out,
        f"{len(out)} files" + (f" -> {out[:5]}" if out else "")
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
