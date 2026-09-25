#!/usr/bin/env python3
"""
s1_embed.py — precompute BGE-M3 dense embeddings for the L2Verify support score.

V(q, C) = max_c cosine(E(q), E(c))

  E(q)  the incoming query text
  E(c)  the text of each page in the L3 evidence C, at the version L3 actually
        served, loaded with the audits' own loader (2,000-char cap) so the
        verifier sees exactly what the generator would see.

Everything is precomputed so the sequential replay is pure numpy and no model is
in the online verification path. No LLM is used here — BGE-M3 is an embedding
model, matching the existing L1/L2 similarity machinery.

Outputs (this folder only): out/emb_queries.npz, out/emb_pages.npz
"""
from __future__ import annotations
import json, os, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v9", "v16_exp12", "v14_baselines", "v13_corrected"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)
import experiment as exp                      # noqa: E402
import schedules as sc                        # noqa: E402
import e2e_answer_grading as e2e              # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
SNAP = ROOT / "data" / "snapshots"
ROUNDS = ("run_00", "rerun_1h", "rerun_12h", "rerun_24h", "rerun_7d")
MODEL = "BAAI/bge-m3"
BATCH = 64


def encode(texts, tok, model, dev, bs=BATCH, tag=""):
    import torch
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    for s in range(0, len(texts), bs):
        ch = texts[s:s + bs]
        enc = tok(ch, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, 0]       # BGE-M3 dense = CLS
            h = torch.nn.functional.normalize(h, p=2, dim=1)
        out[s:s + len(ch)] = h.float().cpu().numpy()
        if (s // bs) % 50 == 0:
            print(f"    {tag} {min(s + bs, len(texts))}/{len(texts)}", flush=True)
    return out


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    L = lambda p: exp.load_jsonl(pathlib.Path(p))                 # noqa
    recs = exp.build_query_records(L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
                                   L(exp.PARAPHRASE_FILE))
    full = sc.build_stream(recs, "zipf_uniform", 42)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning" /
                           "split.json", encoding="utf-8"))
    keep = set(split["validation_clusters"]) | set(split["test_clusters"])
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in keep]
    queries = sorted({r["query"] for _, r in stream})
    urls = sorted({x["url_hash"] for _, r in stream for x in r["urls"]})
    pairs = [(u, rd) for u in urls for rd in ROUNDS
             if (SNAP / rd / f"{u}.json").exists()]
    print(f"  queries {len(queries):,} | (url,round) page texts {len(pairs):,}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to("cuda:0")
    model.eval()

    qv = encode(queries, tok, model, "cuda:0", tag="queries")
    np.savez_compressed(OUT / "emb_queries.npz",
                        keys=np.array(queries, dtype=object), emb=qv)
    print(f"  wrote emb_queries.npz {qv.shape}")

    texts, keys = [], []
    for u, rd in pairs:
        t = (e2e.load_snapshot_text(u, rd) or "").strip()
        if not t:
            continue
        texts.append(t)
        keys.append(f"{u}|{rd}")
    print(f"  non-empty page texts {len(texts):,}")
    pv = encode(texts, tok, model, "cuda:0", tag="pages")
    np.savez_compressed(OUT / "emb_pages.npz",
                        keys=np.array(keys, dtype=object), emb=pv)
    print(f"  wrote emb_pages.npz {pv.shape}")
    json.dump({"model": MODEL, "pooling": "CLS, L2-normalised",
               "max_length": 512, "page_text_loader": "e2e.load_snapshot_text "
               "(2000-char cap), same text the generator sees",
               "n_queries": len(queries), "n_pages": len(texts),
               "rounds": list(ROUNDS)},
              open(OUT / "s1_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
