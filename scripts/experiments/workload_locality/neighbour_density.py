"""Equal-length neighbour density of the benchmark stream versus a public query log.
Statistic: share of requests with an EARLIER request at cosine >= 0.75 (frozen BGE-M3, normalized; empty start; exposure = all prior requests).
Sensitivities: neighbour must not be an exact normalized duplicate; bounded exposure (previous 5,000 requests); near-duplicate share (cosine >= 0.95).
Benchmark: held-out stream (schedule order, frozen similarity matrix); also originals only, and within- vs cross-cluster neighbours.
Public log (MS MARCO train queries; random order, no timestamps; NOT a production traffic log): 9 disjoint consecutive windows of the held-out length in the
seeded 200k order (seed 20261006) and 20 independent random samples of that length from the full log (seed 20261010). Writes outputs/neighbour_density.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, random, re, statistics, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
import numpy as np, torch
O = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); DATA = ((OUTPUT_ROOT / "write_controls/data/queries.train.tsv"))
TH, W = 0.75, 5000   # similarity threshold and the sensitivity window; the length N is the held-out stream's (set below)
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
def stats(S, txt, clus=None):
    """S: (n, n) similarity (torch, cuda, float32) in arrival order."""
    n = S.shape[0]; i = torch.arange(n, device=S.device)
    prior = i[None, :] < i[:, None]; Sp = S.masked_fill(~prior, -2.0)
    out = {"n": n, "all_prior_pct": 100 * float(((Sp >= TH).any(1)).float().mean())}
    win = prior & (i[:, None] - i[None, :] <= W); out["window_5000_pct"] = 100 * float(((S.masked_fill(~win, -2.0) >= TH).any(1)).float().mean())
    codes = {t: k for k, t in enumerate(sorted(set(txt)))}; c = torch.tensor([codes[t] for t in txt], device=S.device)
    nd = prior & (c[None, :] != c[:, None]); out["all_prior_excl_exact_duplicates_pct"] = 100 * float(((S.masked_fill(~nd, -2.0) >= TH).any(1)).float().mean())
    out["exact_duplicate_of_earlier_pct"] = 100 * float(((prior & (c[None, :] == c[:, None])).any(1)).float().mean())
    out["near_duplicate_095_pct"] = 100 * float(((Sp >= 0.95).any(1)).float().mean())
    if clus is not None:
        k = {t: j for j, t in enumerate(sorted(set(clus)))}; cl = torch.tensor([k[t] for t in clus], device=S.device)
        same = prior & (cl[None, :] == cl[:, None]); cross = prior & (cl[None, :] != cl[:, None])
        out["neighbour_within_cluster_pct"] = 100 * float(((S.masked_fill(~same, -2.0) >= TH).any(1)).float().mean())
        out["neighbour_cross_cluster_pct"] = 100 * float(((S.masked_fill(~cross, -2.0) >= TH).any(1)).float().mean())
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}
dev = "cuda:0"; R = {"threshold": TH, "exposure": "all prior requests (primary); previous 5,000 (sensitivity)", "init": "empty"}
import output_workspace; output_workspace.use("common_sample"); import composition_study_helpers as X   # noqa: E402
E = X.engine(); exp = E["exp"]; H = X.C.streams(E)["heldout"]; N = len(H); assert DS.check("heldout_requests", N); R["length"] = N
idx = [exp._QUERY_TO_IDX[r["query"]] for _, r in H]; SM = np.asarray(exp._SIM_MATRIX[np.ix_(idx, idx)], dtype=np.float32)
S = torch.from_numpy(SM).to(dev); R["benchmark_heldout"] = stats(S, [norm(r["query"]) for _, r in H], [r.get("cluster_id") or r["query_id"] for _, r in H])
orig = [k for k, (_, r) in enumerate(H) if not r.get("is_paraphrase")]; So = S[orig][:, orig]
R["benchmark_heldout_originals_only"] = stats(So, [norm(H[k][1]["query"]) for k in orig]); del S, So; torch.cuda.empty_cache()
from sentence_transformers import SentenceTransformer   # noqa: E402
m = SentenceTransformer("BAAI/bge-m3", device=dev)
# gate: the frozen similarity matrix is reproduced by this model on 200 benchmark queries
gi = idx[:200]; gq = [H[k][1]["query"] for k in range(200)]; ge = m.encode(gq, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
R["gate_max_abs_diff_vs_frozen_matrix"] = round(float(np.abs(ge @ ge.T - np.asarray(exp._SIM_MATRIX[np.ix_(gi, gi)])).max()), 4)
Q = [l.rstrip("\n").split("\t", 1)[1] for l in open(DATA, encoding="utf-8")]
def run(qs):
    V = torch.from_numpy(m.encode(qs, batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)).to(dev); return stats(V @ V.T, [norm(q) for q in qs])
order = random.Random(20261006).sample(range(len(Q)), 200000)
R["public_disjoint_windows"] = [run([Q[i] for i in order[k * N:(k + 1) * N]]) for k in range(9)]; print("windows done", flush=True)
rng = random.Random(20261010); R["public_random_samples"] = [run([Q[i] for i in rng.sample(range(len(Q)), N)]) for _ in range(20)]
for nm in ("public_disjoint_windows", "public_random_samples"):
    v = [x["all_prior_pct"] for x in R[nm]]; R[nm + "_summary"] = {k: {"mean": round(statistics.mean([x[k] for x in R[nm]]), 2), "sd": round(statistics.stdev([x[k] for x in R[nm]]), 2),
                                                                   "min": min(x[k] for x in R[nm]), "max": max(x[k] for x in R[nm])} for k in R[nm][0] if k != "n"}
R["first_window_all_prior_pct"] = R["public_disjoint_windows"][0]["all_prior_pct"]
R["density_ratio_benchmark_over_public_mean"] = round(R["benchmark_heldout"]["all_prior_pct"] / R["public_random_samples_summary"]["all_prior_pct"]["mean"], 2)
json.dump(R, open(O / "outputs/neighbour_density.json", "w"), indent=1); print(json.dumps({k: v for k, v in R.items() if "summary" in k or k.startswith("bench") or k.startswith("gate") or k.startswith("first_window") or k.startswith("density")}, indent=1))
