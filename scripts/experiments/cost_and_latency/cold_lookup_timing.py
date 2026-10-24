"""Fresh-process COLD timing of one method's decision-only replay over the frozen held-out stream (same protocol for every method:
new process, engine load, then ONE timed pass with mixed_age_replay.outcome stubbed; then two warm passes). Usage: cold_lookup_timing.py METHOD"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import ast, copy, importlib.util, json, pathlib, sys, time
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "concurrency_cost"); CRR = (OUTPUT_ROOT / "replay"); BASELINE_SEL_DIR = (OUTPUT_ROOT / "baseline_selection")
M = sys.argv[1]; sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
import numpy as np   # noqa: E402
E = C.load("reported"); me, RT = E["me"], E["rounds"]; H = C.streams(E)["heldout"]; n = len(H)
NS = {"np": np, "copy": copy, "engine": lambda: E}
for node in ast.parse(open((ROOT / "scripts/baselines/semantic_ttl_category_aware/baseline_selection_helpers.py")).read()).body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("patched", "ca_replay"): exec(compile(ast.Module([node], []), "baseline_selection_helpers.py", "exec"), NS)
FZ3 = json.load(open(BASELINE_SEL_DIR / "frozen_selection.json")); CA = FZ3["CategoryAware_style"]["per_category"]
TH = {c: v["theta"] for c, v in CA.items()}; KA = {c: v["kappa"] for c, v in CA.items()}; TH["REAL_TIME"] = 1.1; KA["REAL_TIME"] = 0.0
P = {"SemanticTTL (.60,1/2)": (0.60, 0.5), "SemanticTTL (.80,1/4)": (0.80, 0.25), "SemanticTTL (.80,2)": (0.80, 2.0)}
def run():
    if M == "FreshCache": me.replay(H, RT, "FreshCache")
    elif M in P:
        with NS["patched"](*P[M]): me.replay(H, RT, "SemanticTTL")
    elif M == "CategoryAware-style": NS["ca_replay"](H, TH, KA)
_o = me.outcome; v = []
for _ in range(3):
    me.outcome = lambda uh, tc, tr, rounds: me.UNOBS
    try: t0 = time.perf_counter(); run(); v.append((time.perf_counter() - t0) * 1e3)
    finally: me.outcome = _o
json.dump({"method": M, "total_ms": v, "cold_ms_per_request": round(v[0] / n, 4), "warm_ms_per_request_mean": round(sum(v[1:]) / 2 / n, 4), "n": n,
           "protocol": "fresh process; engine load; pass 1 timed cold; passes 2-3 warm; mixed_age_replay.outcome stubbed"}, open(O / f"work/local_cold_{M.replace(' ', '_').replace('/', '-')}.json", "w"), indent=1)
print(M, round(v[0] / n, 4), round(sum(v[1:]) / 2 / n, 4))
