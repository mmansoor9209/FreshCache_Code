"""Post hoc, exploratory: live-stream (C) comparisons split by request type (base question / paraphrase judged SAME by both judges /
paraphrase judged DIFFERENT by at least one judge; paraphrase_equivalence_judging.py), and the source of reused lists. Reads the frozen C labels; changes no
primary result. Writes outputs/paraphrase_stream_strata.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import collections, importlib.util, json, sys, pathlib
O = ((OUTPUT_ROOT / "common_sample")); sys.argv = ["x", "score"]
spec = importlib.util.spec_from_file_location("cs", (REPO_ROOT / "scripts/experiments/live_streams/paraphrase_stream_scoring.py")); cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
PE = {}
for j in ("llama8b", "qwen7b"):
    for l in open(O / f"gpu/out/c_paraeq_{j}.jsonl"): r = json.loads(l); PE.setdefault(r["query_id"], {})[j] = r["label"]
def stratum(q):
    if not cs.REQ[q]["is_paraphrase"]: return "base"
    v = PE.get(q, {}); return "para_same" if v.get("llama8b") == v.get("qwen7b") == "SAME" else "para_different"
n_by = collections.Counter(stratum(q) for q in cs.REQ)
R = {"note": "post hoc exploratory; equivalence = both judges SAME", "requests_by_stratum": dict(n_by), "configs": {}}
for cfg in cs.CFG:
    R["configs"][cfg] = {}
    for a, b in (("B", "D"), ("Be", "Dp"), ("B", "Be"), ("D", "Dp")):
        out = {}
        for st in ("base", "para_same", "para_different"):
            ids = [q for q in cs.REQ if stratum(q) == st]; lo = ga = un = 0
            for q in ids:
                if cs.sig(cs.row(cfg, a, q)) == cs.sig(cs.row(cfg, b, q)): continue
                la, lb = cs.label(cfg, q, a)[0], cs.label(cfg, q, b)[0]
                if None in (la, lb): un += 1; continue
                lo += la == "CORRECT" and lb != "CORRECT"; ga += la != "CORRECT" and lb == "CORRECT"
            out[st] = {"requests": len(ids), "losses": lo, "gains": ga, "net_pp_of_stratum": round(100 * (ga - lo) / len(ids), 2), "unresolved": un}
        R["configs"][cfg][f"{a}->{b}"] = out
json.dump(R, open(O / "outputs/paraphrase_stream_strata.json", "w"), indent=1); print(json.dumps(R, indent=1))
