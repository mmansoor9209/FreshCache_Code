"""After generation: build correctness items (question, gold, candidate) for hit-level harm and the validation audit, written to outputs/l1_validation/inputs/judge_items.jsonl."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import json, pathlib, hashlib, glob; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "l1_validation")); OLD = (OUTPUT_ROOT / "l1_precision")
def h(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ans = {tuple(r["key"]): r["answer"] for r in jl(OLD/"validation_answers_llama3b.jsonl")}
for r in jl(W/"outputs/gpu/answers_llama3b.jsonl"): ans[tuple(r["key"])] = r["answer"]
AUD = {d["query_id"]: d for d in json.load(open(OLD/"validation_audit_sample.json"))["requests"]}
items, miss = {}, 0
def add(q, gold, key):
    global miss
    a = ans.get(tuple(key))
    if a is None: miss += 1; return
    if not gold: return
    sig = (h(q), h(gold), h(a)); items.setdefault(sig, {"sig": list(sig), "q": q, "gold": gold, "cand": a})
for f in sorted(glob.glob(str(W/"outputs/replay/s*.json"))):
    d = json.load(open(f))
    for x in d["hits"]: add(x["question"], x["gold"], x["stored_key"]); add(x["question"], x["gold"], x["fresh_key"])
    for q, v in d["audit"].items(): add(AUD[q]["query"], AUD[q]["gold"], v["key"])
for d in AUD.values(): add(d["query"], d["gold"], d["fresh_key"])
with open(W/"inputs/judge_items.jsonl", "w", encoding="utf-8") as f:
    for v in items.values(): f.write(json.dumps(v, ensure_ascii=False) + "\n")
print(f"judge items {len(items)}; missing answers {miss}")
