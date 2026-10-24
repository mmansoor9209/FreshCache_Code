"""GPTCache baseline. Uses the official GPTCache package (gptcache==0.1.44 from PyPI, not vendored) with BGE-M3 embeddings and
FreshCache-matched answer admission; only the adapter is part of this repository.

Bound for the matched variant's GPTCache top_k (performance only; must not change any decision).
Re-plays the matched decision rule with numpy (FreshCache admission over ALL stored entries, highest frozen-SIM passing entry wins) and records,
at every request, how many stored entries have normalised-embedding cosine >= 0.79 (a safe margin below 0.80 minus the maximum |emb cos - SIM|).
If that count is always < K, GPTCache's faiss top_K contains every entry that could pass admission, so top_k=K is decision-identical."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, os, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
HERE = ((OUTPUT_ROOT / "baselines/gptcache")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/gptcache"))
import gptcache_helpers as X, gptcache_adapter as A
import numpy as np
E = X.engine(); exp = E["exp"]; SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX; emb = A.BGEM3Embedding((OUTPUT_ROOT / "l2_verifier/out/emb_queries.npz"))
out = {}
for sname in ("primary", "heldout"):
    st = X.C.streams(E)[sname]; qi, qs = [], []; V = np.empty((len(st), 1024), np.float32); nv = 0; hits = mx = 0; maxsim_gap = 0.0
    for t, r in st:
        q, fc = r["query"], r["freshness_class"]
        if fc == "REAL_TIME": continue
        h = None
        if qi:
            ec = V[:nv] @ emb.emb[emb.idx[q]]; mx = max(mx, int((ec >= 0.79).sum()))
            sims = np.asarray(SIM[IDX[q]])[np.asarray(qi)]; maxsim_gap = max(maxsim_gap, float(np.abs(sims - ec).max()))
            cand = np.nonzero(sims >= exp._EQ_SIM_FLOOR)[0]
            ok = [int(j) for j in cand if exp.semantic_equivalent(q, qs[int(j)], float(sims[j])) and exp._entity_match(q, qs[int(j)])]
            if ok: h = max(ok, key=lambda j: sims[j])
        if h is not None: hits += 1
        else: qi.append(IDX[q]); qs.append(q); V[nv] = emb.emb[emb.idx[q]]; nv += 1
    out[sname] = {"max_entries_emb_cos_ge_0.79": mx, "max_abs_emb_cos_minus_SIM": maxsim_gap, "emulated_hits": hits}
    X.log("TOPK-BOUND", sname, json.dumps(out[sname]))
json.dump(out, open(X.O / "work/topk_bound.json", "w"), indent=1)
