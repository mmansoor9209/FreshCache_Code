#!/usr/bin/env python3
"""
DIRECT end-to-end latency validation.

For each policy and each sampled request, the complete request path is executed
in one process and the WALL CLOCK from query input to final answer is recorded.
No Monte Carlo, no component reassembly: one timer around the whole path, with
per-stage timers nested inside it.

Path executed per request (the real operations, in order):
  embed        BAAI/bge-m3, batch 1
  L1 lookup    FAISS search over a real index sized to that request's live L1
               index, then the entity + lexical gate (C18 policies only)
  L1 hit       -> return the stored answer; STOP (no search, fetch or generation)
  L2 lookup    FAISS search over a real index sized to the live L2 index
  L2 hit       -> serve the recorded URL list
  miss         -> LIVE Serper search
  L3           per served URL: local snapshot read if the policy serves it from
               cache, otherwise a LIVE cold GET
  generate     Llama-3.2-3B-Instruct, greedy, the audit's FORCED prompt

Live responses are used for latency only. Nothing is written to data/, no
snapshot or cache is updated, and no existing result is touched.
"""
from __future__ import annotations

import argparse, json, os, pathlib, random, re, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# pin the GPU before anything imports torch/experiment
_gpu = None
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _gpu = sys.argv[_i + 1]
if _gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("OMP_NUM_THREADS", "4")

for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import requests                       # noqa: E402
import urllib3                        # noqa: E402
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import experiment as exp              # noqa: E402
import collect                        # noqa: E402
import e2e_answer_grading as e2e      # noqa: E402
import mixed_age_v2 as ma             # noqa: E402
import b6_matched_answers as b6       # noqa: E402

POLICIES = ["NoCache", "ExactTTL", "SemanticTTL_k1_16", "FreshCache_L1Only",
            "FreshCache_NoL1", "FreshCache"]
C18 = {"FreshCache", "FreshCache_L1Only"}
USE_L1 = {"FreshCache", "FreshCache_L1Only", "SemanticTTL_k1_16"}
USE_L2 = {"FreshCache", "FreshCache_NoL1"}
NO_EMBED = {"NoCache", "ExactTTL"}   # exact match is a hash lookup
SERPER = "https://google.serper.dev/search"
MAXCTX, SEED = b6.MAXCTX, 42
STAGES = ["embedding", "l1_lookup", "l2_lookup", "l3_processing",
          "search_api", "page_fetch", "llm_generation"]


def key():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.strip().startswith("SERPER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("SERPER_API_KEY missing")


def q(v, p):
    s = sorted(v)
    return float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])


def summ(v):
    return {"n": len(v), "mean_ms": float(np.mean(v)), "p50_ms": q(v, .50),
            "p95_ms": q(v, .95), "min_ms": float(min(v)), "max_ms": float(max(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec = json.load(open(HERE / "direct_sample_spec.json", encoding="utf-8"))
    url_of = spec["url_of"]
    rng = random.Random(SEED)

    n_search = {p: sum(1 for r in spec["specs"][p]
                       if r["path"] in ("miss", "nocache", "realtime_bypass", "exact_miss"))
                for p in POLICIES}
    n_fetch = {p: sum(1 for r in spec["specs"][p]
                      for s in r["served"] if not s["from_cache"])
               for p in POLICIES}
    print("  planned live calls:", flush=True)
    for p in POLICIES:
        print(f"    {p:<20} searches {n_search[p]:>3}   fetches {n_fetch[p]:>3}",
              flush=True)
    print(f"    TOTAL                searches {sum(n_search.values()):>3}   "
          f"fetches {sum(n_fetch.values()):>3}", flush=True)
    if a.dry_run:
        print("  --dry-run: nothing executed.", flush=True)
        return

    import faiss                       # noqa: E402
    import torch                       # noqa: E402
    from sentence_transformers import SentenceTransformer  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print("\n  loading models ...", flush=True)
    emb = SentenceTransformer("BAAI/bge-m3")
    tok = AutoTokenizer.from_pretrained(b6.GEN)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    gm = AutoModelForCausalLM.from_pretrained(b6.GEN, dtype=torch.bfloat16,
                                              device_map="cuda:0")
    gm.eval()

    vecs = np.ascontiguousarray(
        np.load(str(ROOT / "data" / "query_embeddings_bgem3.npy")), dtype="float32")
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    all_q = [r["query"] for r in records]

    idx_cache = {}

    def index_of(n):
        n = max(1, min(int(n), len(vecs)))
        b = 1 << (max(0, int(n) - 1)).bit_length()      # bucket to a power of 2
        b = max(1024, min(b, len(vecs)))
        if b not in idx_cache:
            ix = faiss.IndexFlatIP(dim)
            ix.add(np.ascontiguousarray(vecs[:b]))
            idx_cache[b] = ix
        return idx_cache[b], b

    hdr = {"X-API-KEY": key(), "Content-Type": "application/json"}
    sess = requests.Session()

    # warm up both models and faiss so the first request is not an outlier
    for w in all_q[:3]:
        v = emb.encode([w], show_progress_bar=False, normalize_embeddings=True)
        index_of(4096)[0].search(np.ascontiguousarray(v, dtype="float32"), 5)
        e = tok([b6.FORCED.format(question=w, context="warm up context")],
                return_tensors="pt", truncation=True, max_length=1024).to(gm.device)
        with torch.no_grad():
            gm.generate(**e, max_new_tokens=8, do_sample=False,
                        pad_token_id=tok.pad_token_id)
    print("  warm-up done; starting timed runs\n", flush=True)

    rows = []
    for pol in POLICIES:
        use_l1 = pol in USE_L1
        use_l2 = pol in USE_L2
        use_l3 = use_l2
        for r in spec["specs"][pol]:
            st = {k: 0.0 for k in STAGES}
            nf = nc = 0
            t_all = time.perf_counter()

            if pol not in NO_EMBED and r["path"] != "realtime_bypass":
                t0 = time.perf_counter()
                qv = emb.encode([r["query"]], show_progress_bar=False,
                                normalize_embeddings=True).astype("float32")
                st["embedding"] = (time.perf_counter() - t0) * 1000
            else:
                qv = None

            hit = False
            if use_l1 and r["path"] != "realtime_bypass":
                t0 = time.perf_counter()
                ix, _ = index_of(r["l1_index"])
                ix.search(np.ascontiguousarray(qv), 5)
                if pol in C18:
                    other = r["matched_query"] or all_q[rng.randrange(len(all_q))]
                    exp._entity_match(r["query"], other)
                    exp.semantic_equivalent(r["query"], other, 0.85)
                st["l1_lookup"] = (time.perf_counter() - t0) * 1000
                if r["path"] == "L1":
                    hit = True

            answer = None
            if hit:
                answer = "[stored answer served from L1]"
            else:
                if use_l2 and r["path"] != "realtime_bypass":
                    t0 = time.perf_counter()
                    ix, _ = index_of(r["l2_index"])
                    ix.search(np.ascontiguousarray(qv), 5)
                    st["l2_lookup"] = (time.perf_counter() - t0) * 1000

                if r["path"] in ("miss", "nocache", "realtime_bypass",
                                 "exact_miss"):
                    t0 = time.perf_counter()
                    try:
                        sess.post(SERPER, headers=hdr,
                                  json={"q": r["query"],
                                        "num": collect.MAX_URLS_PER_QUERY},
                                  timeout=10)
                    except Exception:
                        pass
                    st["search_api"] = (time.perf_counter() - t0) * 1000

                parts = []
                for s in r["served"]:
                    uh = s["url_hash"]
                    if s["from_cache"]:
                        t0 = time.perf_counter()
                        txt = e2e.load_snapshot_text(uh, ma.version_at(0.0)) or ""
                        exp.p_stale(r["fc"], 43200.0, "content")
                        st["l3_processing"] += (time.perf_counter() - t0) * 1000
                        nc += 1
                    else:
                        u = url_of.get(uh)
                        txt = ""
                        t0 = time.perf_counter()
                        if u:
                            try:
                                resp = sess.get(u, headers=collect.HEADERS,
                                                timeout=collect.FETCH_TIMEOUT,
                                                verify=False)
                                txt = resp.text if resp.status_code == 200 else ""
                            except Exception:
                                txt = ""
                        st["page_fetch"] += (time.perf_counter() - t0) * 1000
                        nf += 1
                        txt = re.sub(r"<[^>]+>", " ", txt)
                    parts.append(re.sub(r"\s+", " ", txt).strip())
                ctx = " ".join(parts)[:MAXCTX]
                if not ctx:
                    ctx = " ".join(
                        (e2e.load_snapshot_text(s["url_hash"],
                                                ma.version_at(0.0)) or "")
                        for s in r["served"])[:MAXCTX]

                t0 = time.perf_counter()
                enc = tok([b6.FORCED.format(question=r["query"], context=ctx)],
                          return_tensors="pt", truncation=True,
                          max_length=1024).to(gm.device)
                with torch.no_grad():
                    o = gm.generate(**enc, max_new_tokens=80, do_sample=False,
                                    pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                st["llm_generation"] = (time.perf_counter() - t0) * 1000
                answer = tok.decode(o[0, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()

            total = (time.perf_counter() - t_all) * 1000
            rows.append({"policy": pol, "query_id": r["query_id"],
                         "fc": r["fc"], "path": r["path"],
                         "n_fetched": nf, "n_from_cache": nc,
                         **{k: round(st[k], 4) for k in STAGES},
                         "total_ms": round(total, 4),
                         "answer_chars": len(answer or "")})
        v = [x["total_ms"] for x in rows if x["policy"] == pol]
        print(f"  {pol:<20} n={len(v)}  mean {np.mean(v):>9.1f}  "
              f"p50 {q(v,.50):>9.1f}  p95 {q(v,.95):>9.1f} ms", flush=True)

    import csv
    with open(HERE / "direct_per_request.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for x in rows:
            w.writerow(x)
    res = {}
    for pol in POLICIES:
        sub = [x for x in rows if x["policy"] == pol]
        res[pol] = {"total": summ([x["total_ms"] for x in sub]),
                    "stages": {k: summ([x[k] for x in sub]) for k in STAGES},
                    "paths": {p: sum(1 for x in sub if x["path"] == p)
                              for p in {x["path"] for x in sub}},
                    "live_searches": sum(1 for x in sub if x["search_api"] > 0),
                    "live_fetches": sum(x["n_fetched"] for x in sub),
                    "cache_reads": sum(x["n_from_cache"] for x in sub)}
    json.dump({"seed": SEED, "n_sample": spec["n_sample"],
               "measurement": "direct wall-clock, one process, per request",
               "policies": res},
              open(HERE / "direct_results.json", "w"), indent=2)
    print("\n  wrote direct_per_request.csv, direct_results.json", flush=True)


if __name__ == "__main__":
    main()
