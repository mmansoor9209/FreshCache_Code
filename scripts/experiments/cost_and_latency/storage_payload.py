"""Storage payload and measured page-store write cost per policy (held-out replay).
Payload: bytes of page text written to L3 (each fetched page version stored once), URL-list entries (L2), and answer-cache entries (answer text
+ evidence manifest). Write cost: the actual page texts of one policy's L3 writes are written to a temporary directory inside this folder
(one file per page version, buffered, no fsync) and timed; the directory is removed afterwards. Embeddings are not re-computed: every policy
embeds each non-REAL_TIME query once (c_embed, component-cost measurement), and L2 reuses that embedding; no page embeddings are needed for A-D."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pickle, shutil, sys, time, pathlib
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import composition_helpers as X, composition_engine as RE
import answer_grading as e2e
O = X.O; E = X.engine(); p2 = E["p2"]; H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0)
POL = {"A": dict(l2l3=False), "B": dict(l2_on=False), "C": dict(l3_on=False), "D": dict()}
hl = X.Held(); ST = pickle.load(open(O / "work/composition_ablation_replay.pkl", "rb"))["policies"]
R = {}
for p, kw in POL.items():
    m, rows = RE.replay(E, H, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **kw)
    writes = []
    if kw.get("l2l3", True) and kw.get("l3_on", True):
        for q, r in rows.items():
            if r["tier"] in ("L2", "miss", "miss_rejected"):
                for uh, tc in r["ev"]:
                    if tc == r["t"]: writes.append((uh, tc))
    texts = [(e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tc)) or "") for uh, tc in writes]
    l3_bytes = sum(len(t.encode("utf-8")) for t in texts)
    l2_entries = [r.get("served") or [] for r in rows.values() if r["tier"] in ("L2", "miss", "miss_rejected")] if kw.get("l2l3", True) and kw.get("l2_on", True) else []
    l2_bytes = sum(24 * len(s) + 8 + 1024 * 4 for s in l2_entries)   # url hashes (24 B) + timestamp + stored query embedding (1,024 float32)
    l1_bytes = sum(1024 * 4 + 200 + 32 * len(r.get("ev") or []) for r in rows.values() if r["tier"] not in ("L1", "REAL_TIME"))   # query embedding + ~200 B answer + manifest
    tmp = O / "work/tmp_store"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    t0 = time.perf_counter()
    for i, t in enumerate(texts): (tmp / f"{i}.txt").write_text(t, encoding="utf-8")
    wms = (time.perf_counter() - t0) * 1e3; shutil.rmtree(tmp, ignore_errors=True)
    R[p] = {"L3_page_writes": len(writes), "L3_payload_MB": round(l3_bytes / 1e6, 2), "L2_entries": len(l2_entries), "L2_payload_MB_est": round(l2_bytes / 1e6, 2), "answer_cache_payload_MB_est": round(l1_bytes / 1e6, 2),
            "page_write_total_ms": round(wms, 1), "page_write_ms_per_request": round(wms / len(H), 4), "page_write_ms_per_page": round(wms / max(len(writes), 1), 4)}
    print(p, R[p], flush=True)
json.dump({"n": len(H), "policies": R, "note": "payload = retained cache contents; process memory and monetary cost are not measured"}, open(O / "outputs/payload_writes.json", "w"), indent=1)
