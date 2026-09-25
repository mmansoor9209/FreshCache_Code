#### New for 1000 ##
"""
build_queries.py — Build unified queries.jsonl

Sources:
  1. SealQA          (vtllms/sealqa)                  — English, all non-REAL_TIME classes
  2. FreshQA         (natyou/freshqa_10_06)            — English, TIMELESS / SLOW / MEDIUM / FAST
  3. TemporalAlignQA (ROIM/temporal-alignment-qa)      — English, SLOW / MEDIUM / FAST
  4. collect.py SEED_QUERIES                           — Korean (all classes) + English REAL_TIME

Freshness mapping:
  SealQA:
    never-changing                              → TIMELESS
    slow-changing                               → SLOW
    fast-changing + Science/Tech/Politics       → MEDIUM
    fast-changing + Sports/Entertainment/Others → FAST

  FreshQA (false_premise=True excluded):
    never-changing                              → TIMELESS
    slow-changing                               → SLOW
    fast-changing + tech/science keywords       → MEDIUM
    fast-changing + other                       → FAST

  TemporalAlignQA (derived from answer-change rate):
    unique_answer_ratio >= 0.75                 → FAST
    unique_answer_ratio 0.40–0.74               → MEDIUM
    unique_answer_ratio 0.10–0.39               → SLOW
    unique_answer_ratio < 0.10                  → skip (too stable, SealQA covers TIMELESS better)

Target: 100 queries per class (500 total)
"""

from __future__ import annotations

import json
import random
import uuid
import sys
from collections import Counter
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
# from collect import SEED_QUERIES

OUTPUT_FILE    = Path("data/queries.jsonl")
TARGET          = 2000  # per non-REAL_TIME freshness class
TARGET_REALTIME = 200  # REAL_TIME is never cached; 200 is sufficient
RANDOM_SEED    = 42

OUTPUT_FILE.parent.mkdir(exist_ok=True)
random.seed(RANDOM_SEED)

# ── Freshness mapping helpers ──────────────────────────────────────────────

SEALQA_MEDIUM_TOPICS = {"Science & Technology", "Politics", "History & Geography"}

# Keywords that push a FreshQA fast-changing question into MEDIUM rather than FAST
FRESHQA_MEDIUM_KW = frozenset([
    "android", "macos", "ios", "operating system", "software version",
    "h-index", "citations", "google scholar", "exoplanets", "papers accepted",
    "neurips", "emnlp", "acl ", "ijcai", "conference",
    "vaccination", "covid", "legalize", "cannabis",
    "same-sex marriage", "countries have recognized",
    "tomatometer", "rotten tomatoes",
    "netflix", "base price", "tesla model", "homepod",
])


def sealqa_map(freshness: str, topic: str) -> str:
    if freshness == "never-changing":
        return "TIMELESS"
    if freshness == "slow-changing":
        return "SLOW"
    if freshness == "fast-changing":
        return "MEDIUM" if topic in SEALQA_MEDIUM_TOPICS else "FAST"
    return "MEDIUM"


def freshqa_map(fact_type: str, question: str) -> str:
    if fact_type == "never-changing":
        return "TIMELESS"
    if fact_type == "slow-changing":
        return "SLOW"
    if fact_type == "fast-changing":
        q = question.lower()
        return "MEDIUM" if any(kw in q for kw in FRESHQA_MEDIUM_KW) else "FAST"
    return "MEDIUM"


def temporal_align_map(row: dict) -> str | None:
    """
    Derive freshness class from the year-keyed answer dict.
    Returns None if the question should be skipped.
    """
    ans = row.get("answer")
    if not isinstance(ans, dict) or len(ans) < 4:
        return None
    values = []
    for v in ans.values():
        if isinstance(v, list) and v:
            values.append(str(v[0]).lower().strip())
        else:
            values.append(str(v).lower().strip())
    unique_ratio = len(set(values)) / len(values)
    if unique_ratio >= 0.75:
        return "FAST"
    if unique_ratio >= 0.40:
        return "MEDIUM"
    if unique_ratio >= 0.10:
        return "SLOW"
    return None   # nearly static — skip, SealQA covers TIMELESS better


# ── Record builder ─────────────────────────────────────────────────────────

def make_record(query, freshness_class, language, source,
                topic="", original_freshness="") -> dict:
    return {
        "query_id":           f"q_{uuid.uuid4().hex[:8]}",
        "query":              query,
        "freshness_class":    freshness_class,
        "language":           language,
        "source":             source,
        "topic":              topic,
        "original_freshness": original_freshness,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # ── Load existing queries to preserve their query_ids ─────────────────
    records: list[dict]    = []
    seen:    set[str]      = set()
    counts:  Counter       = Counter()

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = r["query"].strip().lower()
                if key not in seen:
                    seen.add(key)
                    records.append(r)
                    counts[r["freshness_class"]] += 1
        print(f"  Loaded {len(records)} existing queries from {OUTPUT_FILE}")
        print(f"  Existing counts: {dict(counts)}")

    def try_add(query: str, fc: str, lang: str, source: str,
                topic: str = "", orig_fresh: str = "") -> bool:
        key = query.strip().lower()
        if key in seen:
            return False
        limit = TARGET_REALTIME if fc == "REAL_TIME" else TARGET
        if counts[fc] >= limit:
            return False
        seen.add(key)
        records.append(make_record(query.strip(), fc, lang, source, topic, orig_fresh))
        counts[fc] += 1
        return True

    # ── 1. SealQA ─────────────────────────────────────────────────────────
    print("Loading SealQA (vtllms/sealqa)...")
    for config in ["seal_0", "seal_hard", "longseal"]:
        ds = load_dataset("vtllms/sealqa", config, split="test")
        for row in ds:
            q  = row["question"].strip()
            fc = sealqa_map(row["freshness"], row.get("topic", ""))
            try_add(q, fc, "en", f"sealqa/{config}",
                    row.get("topic", ""), row["freshness"])
    print(f"  After SealQA  : {sum(counts.values())} total  {dict(counts)}")

    # ── 2. FreshQA ────────────────────────────────────────────────────────
    print("Loading FreshQA (natyou/freshqa_10_06)...")
    ds = load_dataset("natyou/freshqa_10_06", split="test")
    ds = ds.filter(lambda r: not r["false_premise"])
    rows = list(ds)
    random.shuffle(rows)
    for row in rows:
        q  = row["question"].strip()
        fc = freshqa_map(row["fact_type"], q)
        try_add(q, fc, "en", "freshqa/natyou_10_06", "", row["fact_type"])
    print(f"  After FreshQA : {sum(counts.values())} total  {dict(counts)}")

    # ── 3. TemporalAlignQA ────────────────────────────────────────────────
    print("Loading TemporalAlignQA (ROIM/temporal-alignment-qa)...")
    ds    = load_dataset("ROIM/temporal-alignment-qa", split="test")
    rows  = random.sample(list(ds), min(9000, len(ds)))
    added = 0
    for row in rows:
        fc = temporal_align_map(row)
        if fc is None:
            continue
        q = row["question"].strip()
        if try_add(q, fc, "en", "temporal_align_qa"):
            added += 1
    print(f"  After TAlignQA: {sum(counts.values())} total  {dict(counts)}  (+{added} added)")


    # ── 4. Existing Korean + REAL_TIME queries from backup ─────────────────
    print("Loading existing Korean + REAL_TIME queries from backup...")
    backup_path = Path("data/queries.jsonl.bak_prededup")
    fallback    = Path("data/queries.jsonl")
    source_path = fallback if fallback.exists() else backup_path

    existing = []
    if source_path.exists():
        with open(source_path, encoding="utf-8") as f:
            existing = [json.loads(l) for l in f]

    seed_added = 0
    for q in existing:
        if q.get("language") == "ko" or q.get("freshness_class") == "REAL_TIME":
            query_text = q["query"].strip()
            fc         = q["freshness_class"]
            if try_add(query_text, fc, q["language"], q.get("source", "existing")):
                seed_added += 1
    print(f"  Korean + REAL_TIME added : {seed_added}")

    # ── 5. TriviaQA (TIMELESS top-up) ────────────────────────────────────
    if counts["TIMELESS"] < TARGET:
        needed = TARGET - counts["TIMELESS"]
        print(f"Loading TriviaQA to fill {needed} TIMELESS slots...")
        ds   = load_dataset("mandarjoshi/trivia_qa", "rc", split="train",
                            trust_remote_code=True)
        rows = random.sample(list(ds), min(5000, len(ds)))
        added = 0
        for row in rows:
            q = row["question"].strip()
            if try_add(q, "TIMELESS", "en", "triviaqa/rc"):
                added += 1
            if counts["TIMELESS"] >= TARGET:
                break
        print(f"  After TriviaQA: {sum(counts.values())} total  {dict(counts)}  (+{added} added)")

    # ── 6. Write output ───────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fc_dist   = Counter(r["freshness_class"] for r in records)
    lang_dist = Counter(r["language"]         for r in records)
    src_dist  = Counter(r["source"]           for r in records)

    print(f"\n{'='*58}")
    print(f"  Total queries    : {len(records)}")
    print(f"  Per class        : {dict(fc_dist)}")
    print(f"  Language split   : {dict(lang_dist)}")
    print(f"  Sources          : {dict(src_dist)}")
    print(f"  Output           : {OUTPUT_FILE}")
    print(f"{'='*58}\n")

    short = {fc: (TARGET_REALTIME if fc == "REAL_TIME" else TARGET) - cnt
             for fc, cnt in fc_dist.items()
             if cnt < (TARGET_REALTIME if fc == "REAL_TIME" else TARGET)}
    if short:
        print("Classes below target — add more queries to SEED_QUERIES in collect.py:")
        for fc, gap in short.items():
            print(f"  {fc:<12} : needs {gap} more")
    else:
        print("All classes reached target of 100 queries.")


if __name__ == "__main__":
    main()