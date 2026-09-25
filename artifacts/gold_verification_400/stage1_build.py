#!/usr/bin/env python3
"""
validation/gold_verification_400/stage1_build.py

STAGE 1 of the Feedback 1 gold-verification experiment — BUILD THE VERIFICATION
DATASET. Revision 4 of the plan (no human stage; dual-LLM verification).

Builds, from existing artifacts only:
  * audit_rows.csv          all 800 original audit rows, preserved
  * request_map.csv         786 unique requests, fully mapped
  * verification_units.csv  715 (base query, snapshot round) units
  * paraphrase_pairs.jsonl  paraphrase requests, marked reusable vs needs-new
  * stage1_issues.csv       missing metadata, failed mappings, duplicates,
                            unavailable snapshots
  * stage1_summary.json     counts and integrity record

READ-ONLY on every pre-existing artifact. No LLM, no GPU, no web/API call, no
answer generation, no judging. Everything written lands in this directory.

Stage 0 must have passed; this script asserts that before doing anything.

Reproduce:
    <HOME>/miniconda3/envs/graphrag/bin/python \
        validation/gold_verification_400/stage1_build.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v9", "v16_exp12", "v14_baselines"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import experiment as exp          # noqa: E402  read-only
import mixed_age_v2 as ma         # noqa: E402  read-only

A1 = ROOT / "validation" / "mixed_age_full_policy_audit"
A1B = ROOT / "validation" / "baseline_operating_points"
A2 = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"
SNAP = ROOT / "data" / "snapshots"

norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())   # noqa: E731
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]  # noqa: E731

LOG = []
ISSUES = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def issue(kind, scope, ident, detail):
    ISSUES.append({"kind": kind, "scope": scope, "id": ident, "detail": detail})


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def main():
    say("STAGE 1 — build the verification dataset (Revision 4)")
    say("  read-only on existing artifacts; no LLM, no GPU, no web calls")

    # ---- Stage 0 must have passed ----
    gate_p = HERE / "reproduction_gate.json"
    if not gate_p.exists():
        say("  STOP: reproduction_gate.json not found — Stage 0 has not run.")
        sys.exit(2)
    gate = json.load(open(gate_p, encoding="utf-8"))
    if not gate.get("gate_passed"):
        say("  STOP: Stage 0 gate did not pass.")
        sys.exit(2)
    say(f"  Stage 0 gate: PASSED ({gate['checks_passed']}/{gate['checks_total']})")

    reads = [
        A1 / "sample400.jsonl", A1B / "sttl_sample400.jsonl", A2 / "sample.json",
        ROOT / "data" / "paraphrase_clusters.jsonl",
        ROOT / "data" / "paraphrase_fidelity_judgments.jsonl",
        ROOT / "data" / "queries.jsonl",
        ROOT / "v10_remaining_feedback" / "c2_gold_answers.json",
        ROOT / "v10_remaining_feedback" / "c2_gold_provenance.csv",
        ROOT / "data" / "url_manifest.jsonl",
    ]
    for p in reads:
        if not p.exists():
            say(f"  MISSING INPUT: {p}")
            sys.exit(2)
    pre = {str(p.relative_to(ROOT)): sha(p) for p in reads}
    say(f"  hashed {len(pre)} input artifacts")

    # ---- audit rows (all 800 preserved) ----
    s1 = jl(A1 / "sample400.jsonl")
    s1b = {d["query_id"]: d for d in jl(A1B / "sttl_sample400.jsonl")}
    s2 = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    assert len(s1) == 400 and len(s2) == 400 and len(s1b) == 400

    rows = []
    for d in s1:
        sd = s1b.get(d["query_id"])
        if sd is None:
            issue("failed_mapping", "audit1b", d["query_id"],
                  "no SemanticTTL k=1/8 row for this primary request")
        rows.append({"audit": "mixed_age_400", "query_id": d["query_id"],
                     "query": d["query"], "gold": d["gold"], "fc": d["fc"],
                     "t": d["t"], "in_S": d.get("in_S"),
                     "arms": "fresh|on|off" + ("|sttl_k1_8" if sd else ""),
                     "sttl_k1_8_present": int(sd is not None)})
    for d in s2:
        rows.append({"audit": "heldout_400", "query_id": d["query_id"],
                     "query": d["query"], "gold": d["gold"], "fc": d["fc"],
                     "t": d["t"], "in_S": d.get("in_S"),
                     "arms": "fresh|fc|sttl_k1_16", "sttl_k1_8_present": 0})
    say(f"\n  audit rows preserved: {len(rows)} "
        f"({sum(1 for r in rows if r['audit']=='mixed_age_400')} primary + "
        f"{sum(1 for r in rows if r['audit']=='heldout_400')} held-out)")
    assert len(rows) == 800

    # ---- unique requests ----
    uniq = {}
    dup_across = []
    for r in rows:
        q = r["query_id"]
        if q in uniq:
            dup_across.append(q)
            prev = uniq[q]
            for f in ("query", "gold", "t", "fc"):
                if prev[f] != r[f]:
                    issue("inconsistent_duplicate", "cross_audit", q,
                          f"{f} differs between audits: {prev[f]!r} vs {r[f]!r}")
            uniq[q]["audits"] = prev["audits"] + "|" + r["audit"]
        else:
            uniq[q] = dict(r, audits=r["audit"])
    say(f"  unique requests: {len(uniq)}  "
        f"(appearing in both audits: {len(dup_across)})")

    # ---- base-query mapping ----
    para = jl(ROOT / "data" / "paraphrase_clusters.jsonl")
    base_of = {p["query_id"]: p.get("base_query_id") for p in para}
    para_text = {p["query_id"]: p["query"] for p in para}
    queries = {q["query_id"]: q for q in jl(ROOT / "data" / "queries.jsonl")}

    # ---- gold metadata ----
    gold = json.load(open(ROOT / "v10_remaining_feedback" / "c2_gold_answers.json",
                          encoding="utf-8"))
    prov = {r["query_norm"]: r for r in csv.DictReader(
        open(ROOT / "v10_remaining_feedback" / "c2_gold_provenance.csv",
             encoding="utf-8"))}

    # ---- URL lists per request (for snapshot availability) ----
    L = lambda p: exp.load_jsonl(pathlib.Path(p))          # noqa: E731
    recs = exp.build_query_records(
        L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
        L(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else [])
    urls_of = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in recs}

    # ---- per-request map ----
    rmap = []
    for q, r in uniq.items():
        bq = base_of.get(q) or q
        is_para = q in base_of
        base_row = queries.get(bq)
        base_text = base_row["query"] if base_row else None
        if base_text is None:
            issue("missing_metadata", "base_query", q,
                  f"base_query_id {bq} not in queries.jsonl")
        g = gold.get(norm(r["query"]))
        p = prov.get(norm(r["query"]))
        if g is None:
            issue("missing_metadata", "gold_entry", q,
                  "query not present in c2_gold_answers.json")
        if p is None:
            issue("missing_metadata", "gold_provenance", q,
                  "query not present in c2_gold_provenance.csv")
        rd = ma.version_at(r["t"])
        uh = urls_of.get(q)
        if uh is None:
            issue("failed_mapping", "url_record", q,
                  "no record in build_query_records")
            uh = []
        dist = list(dict.fromkeys(uh))
        have = [u for u in dist if (SNAP / rd / f"{u}.json").exists()]
        if dist and not have:
            issue("unavailable_snapshot", "request", q,
                  f"no snapshot for any of {len(dist)} URLs at round {rd}")
        elif len(have) < len(dist):
            issue("unavailable_snapshot", "request", q,
                  f"{len(dist)-len(have)} of {len(dist)} URLs missing at round {rd}")
        rmap.append({
            "query_id": q, "audits": r["audits"], "audit_rows": r["audits"].count("|") + 1,
            "is_paraphrase": int(is_para),
            "base_query_id": bq, "base_query": base_text,
            "paraphrase_text": para_text.get(q) if is_para else "",
            "incoming_query": r["query"], "gold": r["gold"],
            "gold_answer_c2": (g or {}).get("answer", ""),
            "gold_matches_c2": int(bool(g) and norm(g.get("answer", "")) == norm(r["gold"])),
            "n_aliases": len((g or {}).get("aliases", []) or []),
            "gold_source": (g or {}).get("gold_source", ""),
            "gold_year": (g or {}).get("gold_year", "") or "",
            "gold_note": (g or {}).get("note", ""),
            "provenance_note": (p or {}).get("note", ""),
            "freshness_class": r["fc"], "t": r["t"], "snapshot_round": rd,
            "n_url_slots": len(uh), "n_distinct_urls": len(dist),
            "n_snapshots_available": len(have),
            "snapshot_complete": int(bool(dist) and len(have) == len(dist)),
        })
    say(f"  successfully mapped requests: "
        f"{sum(1 for x in rmap if x['base_query'] and x['gold_source'])}/{len(rmap)}")

    # ---- verification units: (base query, snapshot round) ----
    units = defaultdict(list)
    for x in rmap:
        units[(x["base_query_id"], x["snapshot_round"])].append(x)
    urows = []
    for (bq, rd), members in sorted(units.items()):
        allu, havu = set(), set()
        for m in members:
            for u in dict.fromkeys(urls_of.get(m["query_id"], [])):
                allu.add(u)
                if (SNAP / rd / f"{u}.json").exists():
                    havu.add(u)
        srcs = {m["gold_source"] for m in members if m["gold_source"]}
        yrs = {m["gold_year"] for m in members if m["gold_year"]}
        golds = {norm(m["gold"]) for m in members}
        if len(golds) > 1:
            issue("inconsistent_duplicate", "unit", f"{bq}@{rd}",
                  f"{len(golds)} distinct gold strings among {len(members)} requests")
        if allu and not havu:
            issue("unavailable_snapshot", "unit", f"{bq}@{rd}",
                  f"no snapshot for any of {len(allu)} URLs")
        base_row = queries.get(bq)
        urows.append({
            "unit_id": f"{bq}@{rd}", "base_query_id": bq, "snapshot_round": rd,
            "base_query": base_row["query"] if base_row else "",
            "n_requests": len(members),
            "request_ids": "|".join(m["query_id"] for m in members),
            "gold": sorted(golds)[0] if golds else "",
            "n_distinct_golds": len(golds),
            "gold_source": "|".join(sorted(srcs)), "gold_year": "|".join(sorted(yrs)),
            "n_distinct_urls": len(allu), "n_snapshots_available": len(havu),
            "snapshot_complete": int(bool(allu) and havu == allu),
            "any_snapshot": int(bool(havu)),
        })
    say(f"  verification units (base query, snapshot round): {len(urows)}")
    say(f"    base queries: {len({x['base_query_id'] for x in rmap})}   "
        f"units with >1 round: "
        f"{len([b for b, c in Counter(u['base_query_id'] for u in urows).items() if c > 1])}")
    say(f"    round spread: {dict(Counter(u['snapshot_round'] for u in urows))}")

    # ---- paraphrase fidelity reuse ----
    fid = jl(ROOT / "data" / "paraphrase_fidelity_judgments.jsonl")
    by_pid = {r["para_id"]: r for r in fid}
    by_txt = {norm(r["paraphrase"]): r for r in fid}
    ppairs, reuse, need = [], 0, 0
    for x in rmap:
        if not x["is_paraphrase"]:
            continue
        j = by_pid.get(x["query_id"]) or by_txt.get(norm(x["incoming_query"]))
        rec = {"query_id": x["query_id"], "base_query_id": x["base_query_id"],
               "base_query": x["base_query"], "paraphrase": x["incoming_query"],
               "freshness_class": x["freshness_class"],
               "existing_verdict": (j or {}).get("verdict"),
               "existing_source": ("para_id" if x["query_id"] in by_pid
                                   else ("text" if j else None)),
               "needs_new_judgment": int(j is None)}
        if j is None:
            need += 1
        else:
            reuse += 1
        ppairs.append(rec)
    say(f"  paraphrase requests: {len(ppairs)}  reusable fidelity judgments: "
        f"{reuse}  need new: {need}")
    say(f"    reused verdicts: "
        f"{dict(Counter(p['existing_verdict'] for p in ppairs if p['existing_verdict']))}")
    say(f"  base-query requests (no paraphrase step): "
        f"{sum(1 for x in rmap if not x['is_paraphrase'])}")

    # ---- snapshot availability summary ----
    tot_d = sum(x["n_distinct_urls"] for x in rmap)
    tot_h = sum(x["n_snapshots_available"] for x in rmap)
    say(f"\n  snapshot availability (per request, at its own round): "
        f"{tot_h}/{tot_d} distinct URLs present "
        f"({100*tot_h/tot_d:.2f}%)")
    say(f"    requests with complete snapshots: "
        f"{sum(x['snapshot_complete'] for x in rmap)}/{len(rmap)}")
    say(f"    requests with no snapshot at all: "
        f"{sum(1 for x in rmap if x['n_snapshots_available'] == 0)}")
    say(f"    units with at least one snapshot: "
        f"{sum(u['any_snapshot'] for u in urows)}/{len(urows)}")
    say(f"    units with complete snapshots: "
        f"{sum(u['snapshot_complete'] for u in urows)}/{len(urows)}")

    # ---- issues ----
    say(f"\n  issues recorded: {len(ISSUES)}")
    for k, c in sorted(Counter(i["kind"] for i in ISSUES).items()):
        say(f"    {k}: {c}")
    say(f"  gold string matches c2_gold_answers: "
        f"{sum(x['gold_matches_c2'] for x in rmap)}/{len(rmap)}")
    say(f"  gold_source spread: "
        f"{dict(Counter(x['gold_source'] or '(none)' for x in rmap))}")
    say(f"  gold_year spread: "
        f"{dict(sorted(Counter(x['gold_year'] or '(none)' for x in rmap).items()))}")
    say(f"  gold provenance note spread: "
        f"{dict(Counter(x['provenance_note'] or '(own gold)' for x in rmap))}")

    # ---- write ----
    def wcsv(path, data):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    wcsv(HERE / "audit_rows.csv", rows)
    wcsv(HERE / "request_map.csv", rmap)
    wcsv(HERE / "verification_units.csv", urows)
    with open(HERE / "paraphrase_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in ppairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if ISSUES:
        wcsv(HERE / "stage1_issues.csv", ISSUES)
    else:
        with open(HERE / "stage1_issues.csv", "w", encoding="utf-8") as f:
            f.write("kind,scope,id,detail\n")

    post = {str(p.relative_to(ROOT)): sha(p) for p in reads}
    unchanged = (pre == post)
    say(f"\n  input artifacts unchanged: {unchanged}")

    json.dump({
        "stage": "1 — build verification dataset",
        "stage0_gate_passed": True,
        "audit_rows_preserved": len(rows),
        "unique_requests": len(rmap),
        "requests_in_both_audits": len(dup_across),
        "fully_mapped_requests": sum(1 for x in rmap
                                     if x["base_query"] and x["gold_source"]),
        "base_queries": len({x["base_query_id"] for x in rmap}),
        "verification_units": len(urows),
        "unit_round_spread": dict(Counter(u["snapshot_round"] for u in urows)),
        "paraphrase_requests": len(ppairs),
        "fidelity_reusable": reuse, "fidelity_needs_new": need,
        "base_query_requests": sum(1 for x in rmap if not x["is_paraphrase"]),
        "snapshots": {
            "distinct_urls_needed": tot_d, "distinct_urls_present": tot_h,
            "pct_present": round(100 * tot_h / tot_d, 4) if tot_d else None,
            "requests_complete": sum(x["snapshot_complete"] for x in rmap),
            "requests_with_none": sum(1 for x in rmap
                                      if x["n_snapshots_available"] == 0),
            "units_with_any": sum(u["any_snapshot"] for u in urows),
            "units_complete": sum(u["snapshot_complete"] for u in urows)},
        "gold": {
            "matches_c2_gold_answers": sum(x["gold_matches_c2"] for x in rmap),
            "by_source": dict(Counter(x["gold_source"] or "(none)" for x in rmap)),
            "by_year": dict(Counter(x["gold_year"] or "(none)" for x in rmap)),
            "by_provenance_note": dict(Counter(x["provenance_note"] or "(own gold)"
                                               for x in rmap))},
        "issues_total": len(ISSUES),
        "issues_by_kind": dict(Counter(i["kind"] for i in ISSUES)),
        "llm_calls": 0, "gpu_used": False, "web_calls": 0,
        "answer_generation": 0, "judging": 0,
        "input_artifacts_unchanged": unchanged,
        "hashes_pre": pre, "hashes_post": post,
    }, open(HERE / "stage1_summary.json", "w"), indent=2, ensure_ascii=False)
    (HERE / "stage1.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say("\n  wrote audit_rows.csv, request_map.csv, verification_units.csv, "
        "paraphrase_pairs.jsonl, stage1_issues.csv, stage1_summary.json, stage1.log")
    say("  STOP — Stage 2 and all LLM judging not started, as instructed.")
    (HERE / "stage1.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
