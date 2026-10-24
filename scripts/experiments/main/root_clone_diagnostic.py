"""2x2 root diagnostic for policy D's URL-list reuse. Roots: 60 requests that D serves from a reused URL list (reference present, not REAL_TIME),
seed 20261009: 45 among roots whose answer-cache entry D serves at least once later (exposure from the trace) and 15 among unexposed roots.
For each root, D's state just before the root is cloned into four arms (fresh|reuse) x (write|no write); each arm continues the remaining stream
under D's rules in its own state. (reuse, write) is D itself (gate: identical to the D trace). Stored: root rows of every arm and every later
request whose signature differs from D. Modes: replay | items | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, pickle, random, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("common_sample"); import composition_study_helpers as X, composition_engine_resumable as RV   # noqa: E402
MODE = sys.argv[1]; SET = sys.argv[2] if len(sys.argv) > 2 else "pre"; PK = O / ("work/root2x2.pkl" if SET == "pre" else "work/root2x2_diff.pkl"); E = X.engine(); REC = {r["query_id"]: r for r in E["recs"]}
from common_sample_construction import goldof, sig, Lab   # noqa: E402
ARMS = [("fresh", True), ("fresh", False), ("reuse", True), ("reuse", False)]
an = lambda a: f"{a[0]}_{'write' if a[1] else 'nowrite'}"
if MODE == "replay":
    T = pickle.load(open(O / "work/traces.pkl", "rb"))["D"]; TH, KA = X.ca_params(0.0, 1.0)
    H = X.C.streams(E)["heldout"]
    expo = collections.Counter(r["root"] for r in T.values() if r["tier"] == "L1")
    cand = sorted(q for q, r in T.items() if r["tier"] == "L2" and goldof(q))
    ex = [q for q in cand if expo[q] >= 1]; ux = [q for q in cand if expo[q] == 0]
    own = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in E["recs"]}
    if SET == "pre": rng = random.Random(20261009); roots = sorted(rng.sample(ex, 45)) + sorted(rng.sample(ux, 15))
    else: roots = sorted(random.Random(20261013).sample([q for q in cand if T[q]["served"] != own[q] and REC[q]["freshness_class"] != "REAL_TIME"], 40))   # protocol amendment
    RS = set(roots)
    st = RV.State(E, len(H), floor=0.80, theta_by_class=TH, kappa_by_class=KA); out = {}
    for i, (t, r) in enumerate(H):
        q = r["query_id"]
        if q in RS:
            res = {"stratum": "exposed" if expo[q] else "unexposed", "exposure_D": expo[q], "arms": {}}
            for a in ARMS:
                s = st.clone(); root_row = s.step(t, r, force_evidence=a[0], force_write=a[1]); diff = {}
                for t2, r2 in H[i + 1:]:
                    x = s.step(t2, r2)
                    if sig(x) != sig(T[r2["query_id"]]): diff[r2["query_id"]] = {k: x[k] for k in ("tier", "ev", "gen_query", "root", "root_route") if k in x}
                if a == ("reuse", True): assert not diff and sig(root_row) == sig(T[q]), "gate: (reuse, write) must equal D"
                res["arms"][an(a)] = {"root_row": {k: root_row[k] for k in ("tier", "ev", "gen_query")}, "diff": diff, "metrics": s.metrics()}
            out[q] = res; print(q, res["stratum"], {k: len(v["diff"]) for k, v in res["arms"].items()}, flush=True)
        st.step(t, r)
    pickle.dump({"roots": roots, "res": out}, open(PK, "wb"))
    print("diff sizes", {an(a): sum(len(out[q]["arms"][an(a)]["diff"]) for q in roots) for a in ARMS}); sys.exit(0)
R = pickle.load(open(PK, "rb")); T = pickle.load(open(O / "work/traces.pkl", "rb"))["D"]
def items_of():   # (request id, arm, row) for every root row and every differing later row with a reference; D rows for the same requests
    for q in R["roots"]:
        for a, v in R["res"][q]["arms"].items():
            yield q, a, "root", v["root_row"]
            for q2, x in v["diff"].items():
                if goldof(q2) and REC[q2]["freshness_class"] != "REAL_TIME": yield q2, a, q, x; yield q2, "reuse_write", q, T[q2]
def key(row, q): return X.key_for({"ev": row["ev"], "gen_query": row["gen_query"]}, REC[q]["query"])
if MODE == "items":
    lab = Lab(); gen, J = {}, {}
    for q, a, _, row in items_of():
        k, ctx, gq = key(row, q)
        if lab.answer(k) is None: gen.setdefault(tuple(k), {"key": k, "question": gq, "context": ctx})
        else:
            l, ans, s = lab.label(REC[q]["query"], goldof(q), k)
            if l is None: J.setdefault(s, {"sig": list(s), "q": REC[q]["query"], "gold": goldof(q), "cand": ans})
    X.wjl(O / f"gpu/in/gen_tasks_2x2_{SET}.jsonl", list(gen.values())); X.wjl(O / f"gpu/in/judge2_items_2x2_{SET}.jsonl", list(J.values())); print("gen", len(gen), "judge2", len(J))
if MODE == "score":
    import statistics
    lab = Lab(); A = [an(a) for a in ARMS]
    def lb(q, row): return lab.label(REC[q]["query"], goldof(q), key(row, q)[0])[0] or "UNRESOLVED"
    per = []
    for q in R["roots"]:
        res = R["res"][q]; d = {"root": q, "stratum": res["stratum"], "exposure_D": res["exposure_D"]}
        for a in A: d[f"root_{a}"] = lb(q, res["arms"][a]["root_row"])
        down = sorted({q2 for a in A for q2 in res["arms"][a]["diff"] if goldof(q2) and REC[q2]["freshness_class"] != "REAL_TIME"})
        d["downstream_requests_differing"] = len(down)
        for a in A: d[f"down_correct_{a}"] = sum(lb(q2, res["arms"][a]["diff"].get(q2, T[q2])) == "CORRECT" for q2 in down)
        d["unresolved"] = sum(lb(q2, res["arms"][a]["diff"].get(q2, T[q2])) == "UNRESOLVED" for q2 in down for a in A)
        per.append(d)
    cor = lambda d, a: int(d[f"root_{a}"] == "CORRECT") + d[f"down_correct_{a}"]
    C = {"root_evidence_effect_reuse_minus_fresh": lambda d: int(d["root_reuse_write"] == "CORRECT") - int(d["root_fresh_write"] == "CORRECT"),
         "downstream_write_effect_given_reuse (write - no write)": lambda d: d["down_correct_reuse_write"] - d["down_correct_reuse_nowrite"],
         "downstream_write_effect_given_fresh (write - no write)": lambda d: d["down_correct_fresh_write"] - d["down_correct_fresh_nowrite"],
         "downstream_evidence_effect_given_write (reuse - fresh)": lambda d: d["down_correct_reuse_write"] - d["down_correct_fresh_write"],
         "total_D_minus_fresh_write (root + downstream)": lambda d: cor(d, "reuse_write") - cor(d, "fresh_write"),
         "total_Dprime_root_minus_fresh_nowrite": lambda d: cor(d, "reuse_nowrite") - cor(d, "fresh_nowrite"),
         "total_reuse_nowrite_minus_reuse_write": lambda d: cor(d, "reuse_nowrite") - cor(d, "reuse_write")}
    rng = random.Random(20261009); out = {"set": SET, "roots": len(per), "unresolved": sum(d["unresolved"] for d in per), "per_root": per, "contrasts": {}}
    for nm, f in C.items():
        v = [f(d) for d in per]; bs = sorted(sum(rng.choice(v) for _ in v) for _ in range(2000))
        out["contrasts"][nm] = {"sum": sum(v), "per_root_mean": round(sum(v) / len(v), 3), "roots_neg": sum(x < 0 for x in v), "roots_pos": sum(x > 0 for x in v), "sum_ci_root_bootstrap": [bs[49], bs[1949]]}
    out["root_correct"] = {a: sum(d[f"root_{a}"] == "CORRECT" for d in per) for a in A}
    out["downstream_requests_differing_total"] = sum(d["downstream_requests_differing"] for d in per)
    json.dump(out, open(O / f"outputs/root_clone_2x2_{SET}.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "per_root"}, indent=1))
