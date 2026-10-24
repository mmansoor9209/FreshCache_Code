"""Sufficiency tasks for the NoCache (fresh-pipeline) evidence on the 400 held-out audit requests
(outputs/general_audit/sample.json), in the frozen SUFFV evidence format (operating_point_helpers.evblock of the
request's own fresh page versions).

Writes outputs/nocache_sufficiency/gpu/in/suffv_tasks.jsonl. The tasks are judged by the same two judges with
scripts/experiments/operating_points/nocache_sufficiency_judging.py, which uses the same protocol as
scripts/experiments/operating_points/operating_point_generation_judging.py."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import json, pathlib, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = OUTPUT_ROOT
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/operating_points")); import operating_point_helpers as X   # noqa: E402
SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
T = [{"query_id": d["query_id"], "arm": "nocache_fresh", "query": d["query"], "evidence": X.evblock([tuple(x) for x in (d.get("fresh_versions") or [])])} for d in SAMPLE]
(((OUTPUT_ROOT / "nocache_sufficiency/gpu/in/suffv_tasks.jsonl"))).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in T))
print(len(T), "empty evidence:", sum(1 for t in T if t["evidence"] == "(no stored evidence)"))
