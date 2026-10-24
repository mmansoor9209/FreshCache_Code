"""Online inspection cost of the L2 verifier (gamma) and the learned gate on HELD-OUT candidates.
A recording scorer is plugged into the frozen verifier replay (composition_engine, l1='freshcache', gamma path) to capture the first 500 scored candidates'
tentative evidence (page versions). Per candidate we time what each method must compute online: BGE-M3 embeddings of the candidate's page texts
(GPU, batch = candidate) [verifier and gate]; plus the gate's text features (l2_gate_features.features_core) and model inference [gate only].
The query embedding is already counted for every policy. Writes outputs/inspection_cost.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pickle, re, sys, time, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import numpy as np
O = (OUTPUT_ROOT / "composition"); GATE = (OUTPUT_ROOT / "learned_l2_gate")
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); sys.path.insert(1, str(REPO_ROOT / "scripts/experiments/admission_controls")); sys.dont_write_bytecode = True
import composition_helpers as X, composition_engine as RE   # noqa: E402
import l2_gate_features as Lb   # noqa: E402  (read-only)
E = X.engine(); H = X.C.streams(E)["heldout"]; ma = sys.modules["page_version_replay"]
class Rec:
    def __init__(self): self.c = []
    def score(self, q, ev):
        if len(self.c) < 500: self.c.append((q, [tuple(x) for x in ev]))
        return 1.0
rec = Rec(); m, _ = RE.replay(E, H, l1="freshcache", floor=0.75, gamma=0.0, support=rec)
GZ = pickle.load(open(GATE / "gate_frozen.pkl", "rb")); FEATS, MODEL = GZ["features"], GZ["model"]
from sentence_transformers import SentenceTransformer   # noqa: E402
enc = SentenceTransformer("BAAI/bge-m3", device="cuda:0"); enc.encode(["warm up"], normalize_embeddings=True)
QZ = np.load(X.OV / "emb_queries.npz", allow_pickle=True); QI = {k: i for i, k in enumerate(QZ["keys"])}
tv, tg = [], []
for q, ev in rec.c:
    texts = [Lb.page_text(uh, t) for uh, t in ev]; qv = QZ["emb"][QI[q]]
    t0 = time.perf_counter(); pe = enc.encode([t for t in texts if t], normalize_embeddings=True) if any(texts) else []; sims = [float(qv @ e) for e in pe]; t1 = time.perf_counter()
    f = Lb.features_core(q, sims, texts, len(texts), q, 0.8, "MEDIUM", 0.0, "MEDIUM"); p = float(MODEL.predict_proba(np.array([[f[k] for k in FEATS]], float))[:, 1][0]); t2 = time.perf_counter()
    tv.append((t1 - t0) * 1e3); tg.append((t2 - t0) * 1e3)
VER = json.load(open(X.CRR / "verifier/out/test_operational.json"))["results"]["L2Verify_g0.55"]; ver_scored = VER["l2_accepted"] + VER["l2_rejected"]
G1 = json.load(open(GATE / "gate_heldout_results.json"))["gate_ops"]; G2 = json.load(open(GATE / "gate_heldout_results_tau2.json"))["gate_ops"]; N = json.load(open(OUTPUT_ROOT / "validation_split/split.json"))["test"]["requests"]   # held-out requests
R = {"candidates_timed": len(rec.c), "verifier_ms_per_candidate_mean": round(float(np.mean(tv)), 2), "verifier_ms_per_candidate_p50": round(float(np.median(tv)), 2),
     "gate_ms_per_candidate_mean": round(float(np.mean(tg)), 2), "gate_ms_per_candidate_p50": round(float(np.median(tg)), 2),
     "scored_candidates": {"verifier_gamma_0.55": ver_scored, "gate_tau": G1["l2_scored"], "gate_tau2": G2["l2_scored"]}}
R["added_ms_per_request"] = {"verifier_gamma_0.55": round(ver_scored / N * R["verifier_ms_per_candidate_mean"], 2), "gate_tau": round(G1["l2_scored"] / N * R["gate_ms_per_candidate_mean"], 2), "gate_tau2": round(G2["l2_scored"] / N * R["gate_ms_per_candidate_mean"], 2)}
import csv; VER_LIVE_S = round(float(next(r for r in csv.DictReader(open(OUTPUT_ROOT / "categoryaware_frontier/frontier_heldout_results.csv")) if r["method"] == "FreshCache verifier gamma=0.55 (sensitivity)")["live_p50_ms_per_request"]) / 1000, 4)
R["reported_live_cost_s"] = {"verifier_gamma_0.55": round(VER_LIVE_S + R["added_ms_per_request"]["verifier_gamma_0.55"] / 1000, 3), "gate_tau": round(G1["live_p50_ms_per_request"] / 1000 + R["added_ms_per_request"]["gate_tau"] / 1000, 3),
                              "gate_tau2": round(G2["live_p50_ms_per_request"] / 1000 + R["added_ms_per_request"]["gate_tau2"] / 1000, 3)}
json.dump(R, open(O / "outputs/inspection_cost.json", "w"), indent=1); print(json.dumps(R, indent=1))
