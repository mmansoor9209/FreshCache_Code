"""Fresh keys of the stricter study's validation audit (l1_precision_selection.stage1: ctx of own pages at t). Keyset must equal recorded."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); p2 = E["p2"]; h, ctx_for = p2.h, p2.ctx_for
VA = json.load(open((OUTPUT_ROOT / "l1_precision/validation_audit_sample.json")))["requests"]
own = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in E["recs"]}
out = {}
for d in VA:
    c = ctx_for(tuple((u, d["t"]) for u in own[d["query_id"]])); out[d["query_id"]] = {"key": ["fresh", h(c), h(d["query"])], "question": d["query"], "context": c}
ch = [d["query_id"] for d in VA if out[d["query_id"]]["key"] != d["fresh_key"]]
json.dump(out, open(C.W/"results"/f"stricter_validation_fresh_{ENG}.json", "w")); print(ENG, "validation-audit fresh keys changed vs recorded:", len(ch), ch[:10])
