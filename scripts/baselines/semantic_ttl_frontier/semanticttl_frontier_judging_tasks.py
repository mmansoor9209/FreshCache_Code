"""SemanticTTL and CategoryAware-style comparators (validation grid and held-out evaluation). CategoryAware-style is our
reimplementation of the per-category threshold and TTL policy of category-aware semantic caching (Wang et al., 2025), not the
complete published system; it omits quotas, the storage tier, indexing and load adaptation.

SemanticTTL frontier, judging-task stage (run after generation): list the two-judge items whose (query, gold, answer)
signature has no judgment in any existing judgment source, and write them to
outputs/semanticttl_frontier/gpu/in/<tag>_judge2_items.jsonl. Stops if any answer key still lacks an answer.
Usage: semanticttl_frontier_judging_tasks.py val|ho"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, __import__("os").path.dirname(__file__))
import semanticttl_frontier_helpers as X, semanticttl_frontier_labels as L
tag = sys.argv[1]
if tag == "val":
    lab = L.Val(); K = json.load(open(X.STTL_FRONTIER_DIR / "validation/quality_keys.json"))
    QS = {json.loads(l)["query_id"]: json.loads(l) for l in open((OUTPUT_ROOT / "quality_selection/q_sample.jsonl"))}
    items = [(QS[q]["query"], QS[q]["gold"], v["key"]) for c in K["configs"].values() for q, v in c.items()]
    items += [(QS[q]["query"], QS[q]["gold"], k) for q, d in K["frozen_fresh_and_freshcache_keys"].items() for k in d.values()]
else:
    lab = L.Held(); K = json.load(open(X.STTL_FRONTIER_DIR / "heldout/audit_keys.json"))
    S = {d["query_id"]: d for d in json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]}
    items = [(S[q]["query"], S[q]["gold"], k) for q, arms in K["keys"].items() for k in arms.values()]
todo, miss = L.judge_items(lab, items)
if miss:
    sys.exit(f"STOP: {len(miss)} answer keys still have no answer after generation, e.g. {miss[:3]}")
X.wjl(X.STTL_FRONTIER_DIR / f"gpu/in/{tag}_judge2_items.jsonl", todo)
print(tag, "items needing judgment:", len(todo))
