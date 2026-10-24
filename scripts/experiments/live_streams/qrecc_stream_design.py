"""Natural QReCC workload: stream construction, design only (no Web call). Source: QReCC test split (Anantha et al.; conversations from
QuAC, Natural Questions and TREC CAsT with human-written answers; data/qrecc_test.json, downloaded into this folder). Query = the human
self-contained rewrite of each turn; reference = the human answer; turns without an answer are excluded. Conversations are drawn in a seeded
random order (seed 20261014) and added while the stream stays at or below 2,300 requests. Arrival: conversation start uniform over 7 virtual
days (same seed), turns 60 virtual seconds apart in their original order. Every request gets class SLOW (pre-registered; stable-knowledge
questions; no class predictor applies to new queries). Neighbour share from queries only (frozen BGE-M3, cosine >= 0.75, all prior requests).
Writes work/qr_block.json (sha256 recorded) and work/qr_sim.npy."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, hashlib, json, random, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import numpy as np
O = ((OUTPUT_ROOT / "natural_workload")); SEED, CAP, H = 20261014, 2300, 7 * 86400
D = json.load(open(O / "data/qrecc_test.json", encoding="utf-8"))
conv = collections.defaultdict(list)
for r in D:
    if r["Answer"].strip() and r["Rewrite"].strip(): conv[(r["Conversation_source"], r["Conversation_no"])].append(r)
keys = sorted(conv); rng = random.Random(SEED); rng.shuffle(keys)
REQ, n = [], 0
for k in keys:
    turns = sorted(conv[k], key=lambda r: r["Turn_no"])
    if n + len(turns) > CAP: continue
    t0 = rng.uniform(0, H)
    for j, r in enumerate(turns):
        REQ.append({"query_id": f"qr_{k[0]}_{k[1]}_{r['Turn_no']}", "cluster_id": f"qr_{k[0]}_{k[1]}", "query": r["Rewrite"].strip(), "freshness_class": "SLOW",
                    "reference": r["Answer"].strip(), "aliases": [], "source": k[0], "turn": r["Turn_no"], "is_paraphrase": j > 0, "t_virtual": round(t0 + 60.0 * j, 3)})
    n += len(turns)
    if n >= CAP: break
REQ.sort(key=lambda r: (r["t_virtual"], r["query_id"]))
for i, r in enumerate(REQ): r["order"] = i
from sentence_transformers import SentenceTransformer   # noqa: E402
m = SentenceTransformer("BAAI/bge-m3", device="cuda:0"); V = m.encode([r["query"] for r in REQ], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
S = (V @ V.T).astype(np.float32); np.save(O / "work/qr_sim.npy", S)
N = len(REQ); cl = [r["cluster_id"] for r in REQ]
has = [i > 0 and float(S[i, :i].max()) >= 0.75 for i in range(N)]
within = [i > 0 and any(S[i, j] >= 0.75 and cl[j] == cl[i] for j in range(i)) for i in range(N)]
F = {"seed": SEED, "source": "QReCC test split", "n_requests": N, "n_conversations": len(set(cl)), "by_source": dict(collections.Counter(r["source"] for r in REQ)),
     "neighbour_share_pct": round(100 * sum(has) / N, 2), "within_conversation_neighbour_pct": round(100 * sum(within) / N, 2), "requests": REQ}
F["sha256"] = hashlib.sha256(json.dumps(REQ, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
json.dump(F, open(O / "work/qr_block.json", "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in F.items() if k != "requests"})
