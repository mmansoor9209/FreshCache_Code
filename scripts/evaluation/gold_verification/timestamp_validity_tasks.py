#!/usr/bin/env python3
"""
scripts/evaluation/gold_verification/timestamp_validity_tasks.py -- timestamp-validity judge tasks
with expanded evidence and question-type classification.

Two differences from the generator-budget tasks built by
scripts/evaluation/gold_verification/reference_validity_screen.py:

  1. EXPANDED CONTEXT. The generator's budget (2,000 chars per page, 200-char
     minimum, 2,800-char total) hides part of the text already stored in the
     snapshots and discards short pages outright. This build uses the FULL stored
     extracted_text per page (up to 5,000 chars as captured), no minimum-length
     floor, and a 12,000-char assembly budget.

  2. QUESTION TYPE. Questions are classified deterministically by regex into
     CURRENT_SEEKING / RECURRING_UNMARKED (ask "is it still current?") versus
     DATED / HISTORICAL (ask "is it correct as stated?"). This prevents a
     historically correct, explicitly dated answer from being called superseded
     merely because newer information exists.

No new Web retrieval: every character comes from DATA_DIR/snapshots/ already on disk.
Existing artifacts in outputs/gold_verification/ (including gold_validity_at_t.csv)
are read but never modified; the tasks (r_validity_tasks.jsonl) and a summary
(timestamp_tasks_summary.json) are written to a subdirectory of outputs/gold_verification/.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, json, os, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter

HERE = ((OUTPUT_ROOT / "gold_verification/timestamp_validity"))
GV = ((OUTPUT_ROOT / "gold_verification"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "freshcache/simulation", "scripts/evaluation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)
import tiered_cache_model as exp                                   # noqa: E402

SNAP = (DATA_ROOT / "snapshots")
MAXCHARS_TOTAL = 12000        # generator budget: 2800
MAXCHARS_PAGE = 5000          # generator budget: 2000 (scripts/data/web_collection.py stores at most 5000)
MINCHARS = 0                  # generator budget: 200

RECENCY = re.compile(r"\b(current|currently|now|latest|most recent|recent|recently|"
                     r"today|this year|as of now|newest|present|ongoing|upcoming|"
                     r"next|still)\b", re.I)
YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")
HISTORICAL = re.compile(r"\b(was|were|did|first|originally|founded|born|died|"
                        r"invented|discovered|premiered|established|used to)\b", re.I)


def qtype(q):
    r, y = bool(RECENCY.search(q)), bool(YEAR.search(q))
    if y and not r:
        return "DATED"
    if y and r:
        return "DATED_CURRENT"
    if r:
        return "CURRENT_SEEKING"
    if HISTORICAL.search(q):
        return "HISTORICAL"
    return "RECURRING_UNMARKED"


# DATED and HISTORICAL are asked "correct as stated"; the rest "still current".
AS_STATED = {"DATED", "HISTORICAL"}


def main():
    units = list(csv.DictReader(open(GV / "verification_units.csv", encoding="utf-8")))
    rmap = {r["query_id"]: r for r in csv.DictReader(
        open(GV / "request_map.csv", encoding="utf-8"))}
    vol = {r["base_query_id"]: r for r in csv.DictReader(
        open(GV / "gold_temporal_volatility.csv", encoding="utf-8"))}
    old = {r["unit_id"]: r for r in csv.DictReader(
        open(GV / "gold_validity_at_t.csv", encoding="utf-8"))}
    gold = json.load(open((OUTPUT_ROOT / "gold_answers/reference_answers.json"),
                          encoding="utf-8"))
    nq = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())   # noqa: E731
    L = lambda p: exp.load_jsonl(pathlib.Path(p))                   # noqa: E731
    recs = exp.build_query_records(
        L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
        L(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else [])
    urls_of = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in recs}

    out, st = [], Counter()
    old_chars = new_chars = 0
    for u in units:
        rd = u["snapshot_round"]
        uh = []
        for m in u["request_ids"].split("|"):
            for x in urls_of.get(m, []):
                if x not in uh:
                    uh.append(x)
        parts, kept = [], 0
        for x in uh:
            p = SNAP / rd / f"{x}.json"
            if not p.exists():
                continue
            t = (json.load(open(p, encoding="utf-8")).get("extracted_text") or "")
            t = re.sub(r"\s+", " ", t).strip()[:MAXCHARS_PAGE]
            if len(t) <= MINCHARS:
                continue
            parts.append(t)
            kept += 1
        keep, tot = [], 0
        for t in parts:
            room = MAXCHARS_TOTAL - tot
            if room <= 0:
                break
            keep.append(t[:room])
            tot += min(len(t), room)
        ev = "\n\n".join(keep)
        g = gold.get(nq(u["base_query"])) or {}
        v = vol.get(u["base_query_id"], {})
        o = old.get(u["unit_id"], {})
        qt = qtype(u["base_query"])
        st[qt] += 1
        st["with_evidence" if ev else "no_evidence"] += 1
        new_chars += len(ev)
        old_chars += int(o.get("evidence_chars") or 0)
        fcs = sorted({rmap[m]["freshness_class"] for m in u["request_ids"].split("|")
                      if m in rmap})
        out.append({
            "unit_id": u["unit_id"], "base_query_id": u["base_query_id"],
            "base_query": u["base_query"], "snapshot_round": rd,
            "gold": u["gold"], "gold_display": g.get("answer") or u["gold"],
            "aliases": [a for a in (g.get("aliases") or []) if a][:8],
            "gold_source": u["gold_source"], "gold_year": u["gold_year"],
            "volatility": v.get("volatility", ""),
            "freshness_classes": "|".join(fcs),
            "n_requests": int(u["n_requests"]), "request_ids": u["request_ids"],
            "question_type": qt, "mode": "AS_STATED" if qt in AS_STATED else "CURRENT",
            "n_urls": len(uh), "n_pages_kept": len(keep),
            "evidence_chars": len(ev),
            "old_evidence_chars": int(o.get("evidence_chars") or 0),
            "old_pages_kept": int(o.get("n_pages_kept") or 0),
            "old_status": o.get("final_status", ""),
            "evidence": ev, "has_evidence": int(bool(ev))})

    with open(HERE / "r_validity_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  units: {len(out)}")
    print(f"  question types: {dict(st)}")
    print(f"  mode split: {dict(Counter(t['mode'] for t in out))}")
    print(f"  evidence chars: old total {old_chars:,} -> new total {new_chars:,} "
          f"({new_chars / max(old_chars,1):.2f}x)")
    print(f"  pages kept: old {sum(t['old_pages_kept'] for t in out)} -> "
          f"new {sum(t['n_pages_kept'] for t in out)}")
    print(f"  units with no evidence: {st['no_evidence']}")
    json.dump({"units": len(out), "question_types": dict(st),
               "old_evidence_chars": old_chars, "new_evidence_chars": new_chars,
               "expansion_factor": round(new_chars / max(old_chars, 1), 4),
               "caps": {"per_page": MAXCHARS_PAGE, "total": MAXCHARS_TOTAL,
                        "min_page": MINCHARS},
               "previous_caps": {"per_page": 2000, "total": 2800, "min_page": 200},
               "web_retrieval": 0},
              open(HERE / "timestamp_tasks_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
