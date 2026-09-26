# Timing code excerpts (verbatim, line-numbered)

## validation/latency_e2e_direct/run_direct.py lines 121-123

```python
 121      vecs = np.ascontiguousarray(
 122          np.load(str(ROOT / "data" / "query_embeddings_bgem3.npy")), dtype="float32")
 123      faiss.normalize_L2(vecs)
```

## validation/latency_e2e_direct/run_direct.py lines 134-142

```python
 134      def index_of(n):
 135          n = max(1, min(int(n), len(vecs)))
 136          b = 1 << (max(0, int(n) - 1)).bit_length()      # bucket to a power of 2
 137          b = max(1024, min(b, len(vecs)))
 138          if b not in idx_cache:
 139              ix = faiss.IndexFlatIP(dim)
 140              ix.add(np.ascontiguousarray(vecs[:b]))
 141              idx_cache[b] = ix
 142          return idx_cache[b], b
```

## validation/latency_e2e_direct/run_direct.py lines 147-156

```python
 147      # warm up both models and faiss so the first request is not an outlier
 148      for w in all_q[:3]:
 149          v = emb.encode([w], show_progress_bar=False, normalize_embeddings=True)
 150          index_of(4096)[0].search(np.ascontiguousarray(v, dtype="float32"), 5)
 151          e = tok([b6.FORCED.format(question=w, context="warm up context")],
 152                  return_tensors="pt", truncation=True, max_length=1024).to(gm.device)
 153          with torch.no_grad():
 154              gm.generate(**e, max_new_tokens=8, do_sample=False,
 155                          pad_token_id=tok.pad_token_id)
 156      print("  warm-up done; starting timed runs\n", flush=True)
```

## validation/latency_e2e_direct/run_direct.py lines 159-163

```python
 159      for pol in POLICIES:
 160          use_l1 = pol in ("FreshCache", "SemanticTTL_k1_16")
 161          use_l2 = pol in ("FreshCache", "FreshCache_NoL1")
 162          use_l3 = use_l2
 163          for r in spec["specs"][pol]:
```

## validation/latency_e2e_direct/run_direct.py lines 168-208

```python
 168              if pol != "NoCache" and r["path"] != "realtime_bypass":
 169                  t0 = time.perf_counter()
 170                  qv = emb.encode([r["query"]], show_progress_bar=False,
 171                                  normalize_embeddings=True).astype("float32")
 172                  st["embedding"] = (time.perf_counter() - t0) * 1000
 173              else:
 174                  qv = None
 175  
 176              hit = False
 177              if use_l1 and r["path"] != "realtime_bypass":
 178                  t0 = time.perf_counter()
 179                  ix, _ = index_of(r["l1_index"])
 180                  ix.search(np.ascontiguousarray(qv), 5)
 181                  if pol in C18:
 182                      other = r["matched_query"] or all_q[rng.randrange(len(all_q))]
 183                      exp._entity_match(r["query"], other)
 184                      exp.semantic_equivalent(r["query"], other, 0.85)
 185                  st["l1_lookup"] = (time.perf_counter() - t0) * 1000
 186                  if r["path"] == "L1":
 187                      hit = True
 188  
 189              answer = None
 190              if hit:
 191                  answer = "[stored answer served from L1]"
 192              else:
 193                  if use_l2 and r["path"] != "realtime_bypass":
 194                      t0 = time.perf_counter()
 195                      ix, _ = index_of(r["l2_index"])
 196                      ix.search(np.ascontiguousarray(qv), 5)
 197                      st["l2_lookup"] = (time.perf_counter() - t0) * 1000
 198  
 199                  if r["path"] in ("miss", "nocache", "realtime_bypass"):
 200                      t0 = time.perf_counter()
 201                      try:
 202                          sess.post(SERPER, headers=hdr,
 203                                    json={"q": r["query"],
 204                                          "num": collect.MAX_URLS_PER_QUERY},
 205                                    timeout=10)
 206                      except Exception:
 207                          pass
 208                      st["search_api"] = (time.perf_counter() - t0) * 1000
```

## validation/latency_e2e_direct/run_direct.py lines 210-253

```python
 210                  parts = []
 211                  for s in r["served"]:
 212                      uh = s["url_hash"]
 213                      if s["from_cache"]:
 214                          t0 = time.perf_counter()
 215                          txt = e2e.load_snapshot_text(uh, ma.version_at(0.0)) or ""
 216                          exp.p_stale(r["fc"], 43200.0, "content")
 217                          st["l3_processing"] += (time.perf_counter() - t0) * 1000
 218                          nc += 1
 219                      else:
 220                          u = url_of.get(uh)
 221                          txt = ""
 222                          t0 = time.perf_counter()
 223                          if u:
 224                              try:
 225                                  resp = sess.get(u, headers=collect.HEADERS,
 226                                                  timeout=collect.FETCH_TIMEOUT,
 227                                                  verify=False)
 228                                  txt = resp.text if resp.status_code == 200 else ""
 229                              except Exception:
 230                                  txt = ""
 231                          st["page_fetch"] += (time.perf_counter() - t0) * 1000
 232                          nf += 1
 233                          txt = re.sub(r"<[^>]+>", " ", txt)
 234                      parts.append(re.sub(r"\s+", " ", txt).strip())
 235                  ctx = " ".join(parts)[:MAXCTX]
 236                  if not ctx:
 237                      ctx = " ".join(
 238                          (e2e.load_snapshot_text(s["url_hash"],
 239                                                  ma.version_at(0.0)) or "")
 240                          for s in r["served"])[:MAXCTX]
 241  
 242                  t0 = time.perf_counter()
 243                  enc = tok([b6.FORCED.format(question=r["query"], context=ctx)],
 244                            return_tensors="pt", truncation=True,
 245                            max_length=1024).to(gm.device)
 246                  with torch.no_grad():
 247                      o = gm.generate(**enc, max_new_tokens=80, do_sample=False,
 248                                      pad_token_id=tok.pad_token_id)
 249                  torch.cuda.synchronize()
 250                  st["llm_generation"] = (time.perf_counter() - t0) * 1000
 251                  answer = tok.decode(o[0, enc["input_ids"].shape[1]:],
 252                                      skip_special_tokens=True).strip()
 253  
```

## validation/latency_e2e_direct/run_direct.py lines 70-77

```python
  70  def q(v, p):
  71      s = sorted(v)
  72      return float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])
  73  
  74  
  75  def summ(v):
  76      return {"n": len(v), "mean_ms": float(np.mean(v)), "p50_ms": q(v, .50),
  77              "p95_ms": q(v, .95), "min_ms": float(min(v)), "max_ms": float(max(v))}
```

## validation/latency_e2e_direct/prep_sample.py lines 181-200

```python
 181      # sample from the warm second half, stratified by freshness class,
 182      # and require a real URL so every policy has something to serve/fetch
 183      rng = random.Random(SEED)
 184      half = len(stream) // 2
 185      by = defaultdict(list)
 186      url_of = {}
 187      for m in manifest:
 188          if m.get("run_id") == "run_00" and m.get("url"):
 189              url_of.setdefault(m["url_hash"], m["url"])
 190      seen = set()
 191      for pos in range(half, len(stream)):
 192          t, r = stream[pos]
 193          if r["query_id"] in seen or not r["urls"]:
 194              continue
 195          if not all(u["url_hash"] in url_of for u in r["urls"]):
 196              continue
 197          seen.add(r["query_id"])
 198          by[r["freshness_class"]].append(r["query_id"])
 199      per = N_SAMPLE // len(CLASSES)
 200      want = []
```

## validation/latency_e2e_direct/compare.py lines 16-30

```python
  16  WORKLOAD = {"TIMELESS": 6929, "SLOW": 8137, "MEDIUM": 7626,
  17              "FAST": 8155, "REAL_TIME": 354}
  18  TOT = sum(WORKLOAD.values())
  19  STAGES = ["embedding", "l1_lookup", "l2_lookup", "l3_processing",
  20            "search_api", "page_fetch", "llm_generation"]
  21  
  22  
  23  def q(v, p, w=None):
  24      v = np.asarray(v, float)
  25      if w is None:
  26          s = np.sort(v)
  27          return float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])
  28      o = np.argsort(v); v, w = v[o], np.asarray(w, float)[o]
  29      c = np.cumsum(w) / w.sum()
  30      return float(v[np.searchsorted(c, p, side="left").clip(0, len(v) - 1)])
```

## validation/latency_e2e_direct/compare.py lines 38-47

```python
  38      for p in POL:
  39          sub = [r for r in rows if r["policy"] == p]
  40          v = np.array([float(r["total_ms"]) for r in sub])
  41          cls = [r["fc"] for r in sub]
  42          cnt = {c: sum(1 for x in cls if x == c) for c in WORKLOAD}
  43          w = np.array([(WORKLOAD[c] / TOT) / (cnt[c] / len(sub)) for c in cls])
  44          direct = {"mean_ms": float(v.mean()), "p50_ms": q(v, .5),
  45                    "p95_ms": q(v, .95)}
  46          rew = {"mean_ms": float(np.average(v, weights=w)),
  47                 "p50_ms": q(v, .5, w), "p95_ms": q(v, .95, w)}
```

## validation/latency_e2e/stage_e_compose.py lines 51-72

```python
  51  def main():
  52      rng = np.random.default_rng(SEED)
  53      b = json.load(open(HERE / "stage_b_gpu_timings.json", encoding="utf-8"))
  54      c1 = json.load(open(HERE / "stage_c1_ann_timings.json", encoding="utf-8"))
  55      c2 = json.load(open(HERE / "stage_c2_gate_timings.json", encoding="utf-8"))
  56      d = json.load(open(HERE / "stage_d_live_timings.json", encoding="utf-8"))
  57      raw = [json.loads(l) for l in
  58             open(HERE / "stage_d_live_raw.jsonl", encoding="utf-8") if l.strip()]
  59  
  60      S_embed = np.array(b["embed"]["samples_ms"])
  61      S_gen = np.array(b["generate"]["samples_ms"])
  62      S_ent = np.array(c2["entity"]["samples_ms"])
  63      S_lex = np.array(c2["lexical"]["samples_ms"])
  64      S_l3 = np.array(c2["l3_lookup"]["samples_ms"])
  65      S_search = np.array([r["ms"] for r in raw if r["kind"] == "search" and r["ok"]])
  66      S_fetch = np.array([r["ms"] for r in raw if r["kind"] == "fetch" and r["ok"]])
  67      ann_sizes = np.array(sorted(int(k) for k in c1["ann"]))
  68      ann_samples = {int(k): np.array(v["samples_ms"]) for k, v in c1["ann"].items()}
  69      ann_p50 = np.array([c1["ann"][str(n)]["p50_ms"] for n in ann_sizes])
  70      print(f"  measured pools: embed {len(S_embed)}, gen {len(S_gen)}, "
  71            f"entity {len(S_ent)}, l3 {len(S_l3)}, search {len(S_search)}, "
  72            f"fetch {len(S_fetch)}", flush=True)
```

## validation/latency_e2e/stage_e_compose.py lines 107-136

```python
 107          c = {k: np.zeros(n) for k in COMPONENTS}
 108          c["embedding"] = embed * S_embed[rng.integers(0, len(S_embed), n)]
 109          if l1l.any():
 110              a1 = ann_draw(np.maximum(n1, 1))
 111              c["l1_lookup"] = l1l * a1
 112              if pol in C18:
 113                  c["l1_lookup"] += l1l * (S_ent[rng.integers(0, len(S_ent), n)]
 114                                           + S_lex[rng.integers(0, len(S_lex), n)])
 115          if l2l.any():
 116              c["l2_lookup"] = l2l * ann_draw(np.maximum(n2, 1))
 117          tot_l3 = int(l3l.sum())
 118          if tot_l3:
 119              draws = S_l3[rng.integers(0, len(S_l3), tot_l3)]
 120              acc, k = np.zeros(n), 0
 121              for i, cnt in enumerate(l3l):
 122                  if cnt:
 123                      acc[i] = draws[k:k + cnt].sum(); k += cnt
 124              c["l3_processing"] = acc
 125          tot_f = int(fetches.sum())
 126          if tot_f:
 127              draws = S_fetch[rng.integers(0, len(S_fetch), tot_f)]
 128              acc, k = np.zeros(n), 0
 129              for i, cnt in enumerate(fetches):
 130                  if cnt:
 131                      acc[i] = draws[k:k + cnt].sum(); k += cnt
 132              c["page_fetch"] = acc
 133          c["search_api"] = search * S_search[rng.integers(0, len(S_search), n)]
 134          c["llm_generation"] = gen * S_gen[rng.integers(0, len(S_gen), n)]
 135  
 136          total = sum(c.values())
```

## validation/latency_e2e/stage_e_compose.py lines 170-186

```python
 170      json.dump({"seed": SEED, "policies": results,
 171                 "measured_inputs": {
 172                     "embed": {k: b["embed"][k] for k in ("n", "mean_ms", "p50_ms", "p95_ms")},
 173                     "generate": {k: b["generate"][k] for k in
 174                                  ("n", "mean_ms", "p50_ms", "p95_ms", "mean_new_tokens")},
 175                     "entity": {k: c2["entity"][k] for k in ("n", "p50_ms", "p95_ms")},
 176                     "lexical": {k: c2["lexical"][k] for k in ("n", "p50_ms", "p95_ms")},
 177                     "l3_lookup": {k: c2["l3_lookup"][k] for k in ("n", "p50_ms", "p95_ms")},
 178                     "ann": {k: {kk: v[kk] for kk in ("n", "p50_ms", "p95_ms")}
 179                             for k, v in c1["ann"].items()},
 180                     "search_api_live": d["search_api"],
 181                     "web_fetch_live": d["web_fetch"]},
 182                 "simulator_constants_for_reference": {
 183                     "search_api": 500, "web_fetch": 800, "llm_generate": 2000,
 184                     "l1_lookup": 5, "l2_lookup": 5, "l3_lookup": 5,
 185                     "query_embedding": "absent from the simulator's model"}},
 186                open(HERE / "latency_results.json", "w"), indent=2)
```

## validation/latency_e2e/stage_c1_ann.py lines 20-24

```python
  20  import faiss
  21  
  22  HERE = pathlib.Path(__file__).resolve().parent
  23  ROOT = HERE.parent.parent
  24  EMB = ROOT / "data" / "query_embeddings_bgem3.npy"
```

## validation/latency_e2e/stage_d_live.py lines 175-182

```python
 175             "fetch_error_rate_pct": 100 * (len(frows) - len(fok)) / max(len(frows), 1),
 176             "fetch_by_class": {c: stats([r["ms"] for r in frows
 177                                          if r["ok"] and r["fc"] == c])
 178                                for c in CLASSES},
 179             "assumed_search_ms": exp.LATENCY["search_api"],
 180             "assumed_fetch_ms": exp.LATENCY["web_fetch"],
 181             "note": ("live timings for the latency benchmark only; not used to "
 182                      "recompute any accuracy/drift/freshness result and not "
```

## experiment.py lines 79-88

```python
  79  # ---------------------------------------------------------------------------
  80  LATENCY = {
  81      "search_api":       500,
  82      "web_fetch":        800,
  83      "conditional_get":  150,
  84      "llm_generate":    2000,
  85      "l1_lookup":          5,
  86      "l2_lookup":          5,
  87      "l3_lookup":          5,
  88  }
```

## experiment.py lines 592-597

```python
 592      # Build latencies in record order (preserves alignment with records)
 593      latency_list = [
 594          LATENCY["search_api"] +
 595          len(r["urls"]) * LATENCY["web_fetch"] +
 596          LATENCY["llm_generate"]
 597          for r in records
```

