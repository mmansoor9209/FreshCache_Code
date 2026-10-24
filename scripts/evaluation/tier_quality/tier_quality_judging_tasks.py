"""Build blinded judging tasks after generation.
Answer grading: one task per distinct (request, answer text). Sufficiency: one task per distinct (request, evidence text).
Both shuffled with seed 42; prompts carry no path label."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, hashlib, pathlib, random, sys, os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "tier_quality"))
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); sys.path.insert(0, str(ROOT / "freshcache/simulation")); os.chdir(ROOT)
import tiered_cache_model as exp, page_version_replay as ma
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
own = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))}
gen = {tuple(t["key"]): t["preserved"]["answer"] for t in jl(W/"inputs/generation_tasks.jsonl") if t["preserved"]}
for r in jl(W/"outputs/raw/generated_answers.jsonl"): gen[tuple(r["key"])] = r["answer"]
def target_date(qid, t):   # latest capture date of the request's own substantive pages at its replay round (rule of model_based_reference_validation)
    ds = []
    for uh in dict.fromkeys(own[qid]):
        p = (DATA_ROOT / "snapshots")/ma.version_at(float(t))/f"{uh}.json"
        if p.exists():
            s = json.load(open(p, encoding="utf-8"))
            if len(" ".join((s.get("extracted_text") or "").split())) >= 200: ds.append((s.get("fetched_at") or "")[:10])
    return max(ds) if ds else "unknown"
rows, ans_tasks, suf_tasks = [], {}, {}
for task, fn, qf in (("l1_reuse", "l1_reuse_hits.jsonl", "incoming_query"), ("l2_reuse", "l2_reuse_hits.jsonl", "question")):
    for x in jl(W/"inputs"/fn):
        td = target_date(x["query_id"], x["t"])
        for path, v in x["paths"].items():
            a = gen.get(tuple(v["key"])); assert a is not None, (task, x["query_id"], path)
            aid = hashlib.sha256(f"{x['query_id']}\x00{a}".encode()).hexdigest()[:16]
            ans_tasks.setdefault(aid, {"task_id": aid, "query_id": x["query_id"], "question": x[qf], "target": td, "reference": x["reference_answer"], "candidate": a})
            sid = None
            if "evidence_block" in v:
                sid = hashlib.sha256(f"{x['query_id']}\x00{v['evidence_block']}".encode()).hexdigest()[:16]
                suf_tasks.setdefault(sid, {"task_id": sid, "query_id": x["query_id"], "question": x[qf], "evidence": v["evidence_block"]})
            rows.append({"task": task, "query_id": x["query_id"], "path": path, "answer": a, "answer_task_id": aid, "sufficiency_task_id": sid,
                         "target_date": td, "context_empty": not v["context"]})
for name, d in (("answer_tasks.jsonl", ans_tasks), ("sufficiency_tasks.jsonl", suf_tasks)):
    L = list(d.values()); random.Random(42).shuffle(L)
    with open(W/"inputs"/name, "w", encoding="utf-8") as f:
        for r in L: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(W/"inputs/path_rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"path rows {len(rows)}; answer tasks {len(ans_tasks)}; sufficiency tasks {len(suf_tasks)}")
