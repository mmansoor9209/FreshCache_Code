"""Benchmark composition (Table 2 and Section 4.1): base questions per freshness class, Web-record coverage and language split.

A base question is retained when the baseline collection round (run_00) holds at least one manifest row with an available snapshot,
the same rule as tiered_cache_model.build_query_records. Writes outputs/benchmark/benchmark_statistics.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, json, os, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
CLASSES = ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]
queries = [json.loads(l) for l in open((DATA_ROOT / "queries.jsonl"), encoding="utf-8") if l.strip()]
with_record = set()
for l in open((DATA_ROOT / "url_manifest.jsonl"), encoding="utf-8"):
    if not l.strip(): continue
    m = json.loads(l)
    if m.get("run_id") == "run_00" and m.get("snapshot_available"): with_record.add(m["query_id"])
base = collections.Counter(q["freshness_class"] for q in queries)
kept = collections.Counter(q["freshness_class"] for q in queries if q["query_id"] in with_record)
lang = collections.Counter(q.get("language") for q in queries if q["query_id"] in with_record)
out = {"by_class": {c: {"base": base[c], "with_web_record": kept[c], "coverage_pct": round(100 * kept[c] / base[c], 2) if base[c] else None} for c in CLASSES},
       "total": {"base": len(queries), "with_web_record": sum(kept.values()), "coverage_pct": round(100 * sum(kept.values()) / len(queries), 2)},
       "retained_by_language": dict(lang)}
((OUTPUT_ROOT / "benchmark")).mkdir(parents=True, exist_ok=True)
json.dump(out, open((OUTPUT_ROOT / "benchmark/benchmark_statistics.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
