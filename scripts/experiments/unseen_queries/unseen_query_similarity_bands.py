"""Operating-region breakdown of the PRIMARY matched-depth comparison (A_matched vs B) by L2 admission similarity band, list length k, and class."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, sys, importlib.util, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "unseen_queries")); sys.argv = ["unseen_query_scoring.py", "score"]
import io, contextlib
spec = importlib.util.spec_from_file_location("s", (REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py")); s = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(s)
FZ = {c["case_id"]: c for c in json.load(open(O / "sample_frozen.json"))["cases"]}
def band(x): return "0.75-0.80" if x < 0.80 else ("0.80-0.85" if x < 0.85 else ("0.85-0.90" if x < 0.90 else ">=0.90"))
out = {}
for name, key in (("similarity", lambda i: band(FZ[i]["similarity"])), ("k", lambda i: str(s.byid[i]["k"])), ("class", lambda i: FZ[i]["freshness_class"])):
    g = collections.defaultdict(list)
    for i in s.ids: g[key(i)].append(i)
    out[name] = {k: {"N": len(v), "A_suff": sum(s.L["A_matched"][i][1] == "SUFFICIENT" for i in v), "B_suff": sum(s.L["B_cached_live"][i][1] == "SUFFICIENT" for i in v),
                     "A_correct": sum(s.L["A_matched"][i][0] == "CORRECT" for i in v), "B_correct": sum(s.L["B_cached_live"][i][0] == "CORRECT" for i in v)} for k, v in sorted(g.items())}
json.dump(out, open(O / "similarity_band_region.json", "w"), indent=1); print(json.dumps(out, indent=1))
