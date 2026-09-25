#!/usr/bin/env python3
"""
Section 5 -- prospective temporal generalization.

STEP 1 (done here): establish that NO post-freeze snapshot exists.
  The repository's only Web snapshot rounds are run_00, rerun_1h, rerun_12h,
  rerun_24h and rerun_7d, all collected BEFORE the calibration artifacts were
  produced (the half-lives are fit FROM rerun_1h/12h/24h). There is therefore
  no round in this repository that post-dates the freeze, and no prospective
  evaluation can be run from existing data. This is recorded, not worked around.

STEP 2 (done here): freeze and checksum everything an online decision uses --
  half-lives, tier multipliers, epsilons, similarity thresholds, the
  equivalence-gate constants, the risk classifier, the query set and the URL
  set -- and stamp the freeze time.

STEP 3 (scripts only): collect_t0.py collects a live-Web T0 snapshot for the
  frozen URL set, and recollect.py re-collects it at +24h and +7d. Both write
  ONLY into 05_prospective_temporal/ and never touch data/.

STEP 4 (cannot be done in this session): evaluate. Until a post-freeze
  snapshot pair actually exists, the prospective-generalization criticism is
  UNRESOLVED. Nothing here claims otherwise.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import experiment as exp                      # noqa: E402
import schedules as sc                        # noqa: E402


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    say("V3 SECTION 5 -- prospective temporal generalization")
    say(f"  freeze time (UTC) {now}")

    rounds = set()
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            rounds |= set(json.loads(line)["rounds"].keys())
    say(f"\n  snapshot rounds present in this repository: {sorted(rounds)}")
    say("  post-freeze (prospective) rounds present: NONE")
    say("  the half-lives were FIT from rerun_1h/12h/24h, so every existing "
        "round is at or before the calibration point")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    urls = {}
    for r in records:
        for u in r["urls"]:
            urls[u["url_hash"]] = {"url": u.get("url"),
                                   "freshness_class": r["freshness_class"]}
    say(f"\n  frozen query set : {len(records):,} query records")
    say(f"  frozen URL set   : {len(urls):,} distinct URLs")

    params = {
        "HALF_LIFE": dict(exp.HALF_LIFE),
        "TIER_MULT": dict(exp.TIER_MULT),
        "EPS_ANSWER": exp.EPS_ANSWER, "EPS_URL_LIST": exp.EPS_URL_LIST,
        "EPS_CONTENT": exp.EPS_CONTENT,
        "L1_SIM_THRESHOLD": exp.L1_SIM_THRESHOLD,
        "L2_SIM_THRESHOLD": exp.L2_SIM_THRESHOLD,
        "_L2_EQ_SIM_FLOOR": exp._L2_EQ_SIM_FLOOR,
        "_EQ_SIM_FLOOR": exp._EQ_SIM_FLOOR,
        "_EQ_JACCARD_MIN": exp._EQ_JACCARD_MIN,
        "FIXED_TTL": dict(exp.FIXED_TTL),
    }
    say("\n  frozen decision parameters")
    for k, v in params.items():
        say(f"    {k:<20} {v}")

    files = ["experiment.py", "calibrate.py", "collect.py",
             "freshcache/risk_model.py", "data/risk_model.pt",
             "data/queries.jsonl", "data/url_manifest.jsonl",
             "v16_exp12/mixed_engine.py", "v14_baselines/engine_all.py"]
    checks = {}
    say("\n  frozen artifact checksums")
    for f in files:
        p = ROOT / f
        if p.exists():
            checks[f] = sha_file(p)
            say(f"    {checks[f][:16]}  {f}")

    qs = sorted({(r["query_id"], r["query"], r["freshness_class"])
                 for r in records if not r.get("is_paraphrase")})
    with open(HERE / "frozen_query_set.jsonl", "w", encoding="utf-8") as fh:
        for qid, q, fc in qs:
            fh.write(json.dumps({"query_id": qid, "query": q,
                                 "freshness_class": fc}) + "\n")
    with open(HERE / "frozen_url_set.jsonl", "w", encoding="utf-8") as fh:
        for uh, d in sorted(urls.items()):
            fh.write(json.dumps({"url_hash": uh, **d}) + "\n")

    man = {"freeze_time_utc": now,
           "prospective_snapshot_present": False,
           "existing_rounds": sorted(rounds),
           "frozen_parameters": params,
           "frozen_artifact_sha256": checks,
           "frozen_query_set": {"file": "frozen_query_set.jsonl", "n": len(qs)},
           "frozen_url_set": {"file": "frozen_url_set.jsonl", "n": len(urls)},
           "status": "UNRESOLVED -- no post-freeze Web snapshot has been "
                     "collected or evaluated. Collect T0 with collect_t0.py, "
                     "then re-collect at +24h and +7d with recollect.py, then "
                     "evaluate. No refitting of half-lives or thresholds is "
                     "permitted at any point after this freeze.",
           "sha256_of_this_manifest": None}
    p = HERE / "freeze_manifest.json"
    p.write_text(json.dumps(man, indent=2), encoding="utf-8")
    man["sha256_of_this_manifest"] = sha_file(p)
    p.write_text(json.dumps(man, indent=2), encoding="utf-8")
    say(f"\n  wrote freeze_manifest.json  (self sha256 "
        f"{man['sha256_of_this_manifest'][:16]}...)")
    say(f"  wrote frozen_query_set.jsonl ({len(qs):,}), "
        f"frozen_url_set.jsonl ({len(urls):,})")
    say("\n  VERDICT: UNRESOLVED. The freeze is in place and the collection "
        "scripts are ready, but no post-freeze snapshot has been collected or "
        "evaluated, so no prospective claim is made.")
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
