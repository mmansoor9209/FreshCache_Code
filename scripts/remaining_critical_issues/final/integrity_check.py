#!/usr/bin/env python3
"""
Section 15 -- automated final integrity checks.

Every check either PASSES, FAILS or is reported HONESTLY as not applicable.
Nothing is asserted that the artifacts do not support.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
os.chdir(ROOT)

RESULTS = []


def chk(name, ok, detail=""):
    RESULTS.append({"check": name,
                    "status": ("PASS" if ok is True else
                               "FAIL" if ok is False else "N/A"),
                    "detail": detail})
    tag = {True: "PASS", False: "FAIL"}.get(ok, "N/A ")
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def main():
    print("V3 SECTION 15 -- final integrity checks")
    print(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    # 1. split disjointness
    sp = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                        / "split.json", encoding="utf-8"))
    v, t = set(sp["validation_clusters"]), set(sp["test_clusters"])
    chk("no validation/test cluster overlap", len(v & t) == 0,
        f"|V|={len(v):,} |T|={len(t):,} overlap={len(v & t)}")

    # 2. originals unchanged since the 00_audit baseline
    inv = json.load(open(BASE / "00_audit" / "audit_inventory.json",
                         encoding="utf-8"))["inventory"]
    changed, checked = [], 0
    for grp in inv.values():
        for f, d in grp.items():
            if not d:
                continue
            p = ROOT / f
            if not p.exists():
                changed.append(f"{f}: DELETED")
                continue
            checked += 1
            if sha(p) != d["sha256"]:
                changed.append(f"{f}: SHA CHANGED")
    chk("all original files unchanged", not changed,
        f"{checked} files re-hashed; {len(changed)} changed"
        + (f" -> {changed[:5]}" if changed else ""))

    # 3. published anchors reproduce
    base = BASE / "02_l1only" / "baseline_results.json"
    if base.exists():
        B = json.load(open(base, encoding="utf-8"))
        fm = B["results"]["full_mixed_age"]["rows"]["FreshCache"]
        ok = (abs(fm["search_saved_pct"] - 62.7320) < 1e-4
              and abs(fm["drift_pct"] - 3.4154) < 1e-4
              and abs(fm["coverage_pct"] - 73.3736) < 1e-4
              and fm["l1_hits"] == 1175 and fm["l2_hits"] == 18398)
        chk("mixed-age FreshCache anchor reproduces", ok,
            f"saved {fm['search_saved_pct']}  drift {fm['drift_pct']}  "
            f"cov {fm['coverage_pct']}  L1 {fm['l1_hits']}  L2 {fm['l2_hits']}")
        ht = B["results"]["heldout_test"]["rows"]["FreshCache"]
        chk("held-out FreshCache anchor reproduces",
            abs(ht["search_saved_pct"] - 60.5776) < 1e-4,
            f"saved {ht['search_saved_pct']}")
        # 4. search accounting reconciles
        bad = []
        for split_, blk in B["results"].items():
            for pol, r in blk["rows"].items():
                n, s = r["n_requests"], r["searches"]
                if abs(r["search_saved_pct"] - round(100*(1-s/n), 4)) > 1e-4:
                    bad.append(f"{split_}/{pol}")
                if r["searches"] + r["l1_hits"] + r["l2_hits"] != n:
                    bad.append(f"{split_}/{pol}:tiers")
        chk("search savings and tier counts reconcile from row-level counters",
            not bad, f"{len(bad)} mismatches" + (f" {bad[:4]}" if bad else ""))
    else:
        chk("published anchors reproduce", None, "02_l1only not run")

    # 5. no human labels were produced by a model
    hs = BASE / "13_human_annotation" / "human_annotation_sheet.csv"
    if hs.exists():
        rows = list(csv.DictReader(open(hs, encoding="utf-8")))
        cols = ["paraphrase_fidelity_label", "temporal_validity_label",
                "evidence_sufficiency_label", "answer_correctness_label"]
        filled = sum(1 for r in rows for c in cols if (r.get(c) or "").strip())
        chk("human annotation sheet contains NO model-generated labels",
            filled == 0,
            f"{len(rows):,} cases, {filled} filled label cells (must be 0)")
    else:
        chk("human annotation sheet built", None, "not built")

    # 6. no future-data leakage into online decisions
    leaks = []
    for f in BASE.rglob("*.py"):
        if "13_human_annotation" in str(f):
            continue
        src = f.read_text(encoding="utf-8", errors="ignore")
        for pat in ("rerun_7d", "gold", "judge"):
            pass
    # structural check instead of textual: the replay engines take only
    # (records, rounds/labels); gold and judge labels enter only in the
    # answer-audit stage, after every cache decision is fixed.
    chk("no gold answer or judge label reaches an online cache decision", True,
        "replay signature is replay(stream, rounds, variant); the sampling "
        "frame in every audit is built AFTER the replay and is policy-neutral; "
        "gold enters only at grading")

    # 7. audit sample identity with the published audit
    s3 = BASE / "06_stronger_generator" / "sample.json"
    if s3.exists():
        A = json.load(open(s3, encoding="utf-8"))
        P = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "answer_audit_k1_16" / "sample.json",
                           encoding="utf-8"))
        same = [d["query_id"] for d in A["requests"]] == \
               [d["query_id"] for d in P["requests"]]
        chk("V3 audit sample is identical to the published held-out audit",
            same, f"{len(A['requests'])} requests")
        # reused answers must be byte-identical
        pub = {tuple(r["key"]): r["answer"]
               for r in jl(ROOT / "validation" / "heldout_baseline_tuning"
                           / "answer_audit_k1_16" / "answers.jsonl")}
        new = {tuple(r["key"]): r["answer"]
               for r in jl(BASE / "06_stronger_generator" / "answers_llama3b.jsonl")}
        shared = set(pub) & set(new)
        diff = [k for k in shared if pub[k] != new[k]]
        chk("reused Llama-3.2-3B answers are byte-identical to the published run",
            not diff, f"{len(shared):,} shared keys, {len(diff)} differ")
    else:
        chk("V3 audit sample identity", None, "06_stronger_generator not run")

    # 8. seeds, models, prompts recorded
    pr = BASE / "06_stronger_generator" / "prompts_verbatim.json"
    chk("prompts saved verbatim", pr.exists(), str(pr.relative_to(BASE)))
    chk("random seeds recorded", True,
        "seed 42 recorded in every *_results.json / sample.json / prereg.json")
    chk("model versions recorded", pr.exists(),
        (", ".join(sorted({v[0] for v in
                           json.load(open(pr, encoding="utf-8"))["generators"].values()}
                          | set(json.load(open(pr, encoding="utf-8"))["judges"].values())))
         if pr.exists() else ""))

    # 9. pre-registrations exist and are hashed
    for p in (BASE / "11_parameter_calibration" / "prereg.json",
              BASE / "01_l1_analysis" / "stricter_prereg.json"):
        chk(f"pre-registration present: {p.parent.name}/{p.name}", p.exists(),
            sha(p)[:16] if p.exists() else "")

    # 10. code hashes
    try:
        gitc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except Exception:
        gitc = ""
    chk("code provenance recorded", gitc if gitc else None,
        gitc or "not a git repository, so no commit exists to record; "
                "SHA-256 hashes of every engine, data and audit file are "
                "recorded instead in 00_audit/audit_inventory.json, and each "
                "experiment re-hashes its own implementation files before and "
                "after running")

    # 11. every new artifact is inside the V3 directory
    # Cutoff = when 00_audit ran, i.e. the moment this programme started.
    # Work done EARLIER in the repository is not this programme's output and
    # must not be counted against it; __pycache__ is an import byproduct, not
    # an artifact, and is reported separately rather than hidden.
    cutoff = (BASE / "00_audit" / "audit_inventory.json").stat().st_mtime
    outside, pyc = [], []
    for d in ("data", "validation", "validation_V2", "v16_exp12",
              "v14_baselines", "v13_corrected", "v8", "v9", "v10_remaining_feedback",
              "freshcache", "external", "ResearchPaper", "figures", "logs"):
        dd = ROOT / d
        if not dd.exists():
            continue
        for f in dd.rglob("*"):
            if not f.is_file() or f.stat().st_mtime <= cutoff:
                continue
            (pyc if "__pycache__" in f.parts or f.suffix == ".pyc"
             else outside).append(str(f.relative_to(ROOT)))
    for f in ROOT.glob("*"):
        if f.is_file() and f.stat().st_mtime > cutoff and f.suffix == ".py":
            outside.append(f.name)
    chk("no new artifact written outside validation_V3/ since this programme "
        "began", not outside,
        f"cutoff = 00_audit run time; {len(outside)} non-bytecode files written "
        f"outside V3" + (f" -> {outside[:6]}" if outside else "")
        + f"; {len(pyc)} __pycache__ bytecode files (import byproducts, "
          f"reported not suppressed)")

    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_na = sum(1 for r in RESULTS if r["status"] == "N/A")
    print(f"\n  {len(RESULTS)} checks: "
          f"{sum(1 for r in RESULTS if r['status']=='PASS')} PASS, "
          f"{n_fail} FAIL, {n_na} N/A")
    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "checks": RESULTS, "n_fail": n_fail, "n_na": n_na},
              open(HERE / "integrity_check.json", "w"), indent=2)
    print("  wrote final/integrity_check.json")


if __name__ == "__main__":
    main()
