"""Reference-validity inputs for the 150 live paired cases: one request per case and the A_full (live top-10) evidence at the search time."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, os, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "unseen_queries"))
C = [json.loads(l) for l in open(O / "work/cases_built.jsonl")]
EV = {}
for t in (json.loads(l) for l in open(O / "gpu/in/suffv_tasks.jsonl")): EV[(t["query_id"], t["arm"])] = t["evidence"]
req = []; ev = []; miss = 0
for c in C:
    if not (c["both_evaluable"] and c["reference_available"]): continue
    e = EV.get((c["query_id"], c["paths"]["A_full"]["evidence_sha"]))
    if e is None: miss += 1; continue
    q = c["case_id"]
    req.append({"query_id": q, "question": c["query"], "reference_answer": c["reference"]})
    ev.append({"query_id": q, "evidence_time_max": c["searched_utc"], "pages": [{"used": True, "fetched_at": c["searched_utc"], "text": e}]})
(O / "refval/inputs").mkdir(parents=True, exist_ok=True)
open(O / "refval/inputs/requests.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in req))
open(O / "refval/inputs/evidence.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ev))
open(O / "refval/inputs/policy_outputs.jsonl", "w").write("")
print(len(req), miss)
