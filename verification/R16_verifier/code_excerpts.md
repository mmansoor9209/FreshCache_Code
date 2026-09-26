# R16 code excerpts (verbatim, line-numbered)

## validation_V2/l2_evidence_verification/s3a_prereg.py lines 30-31

```python
  30  P["prereg_sha256"] = hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest()
  31  json.dump(P, open(HERE / "out" / "prereg_gamma.json", "w"), indent=2)
```

## validation_V2/l2_evidence_verification/s3b_label.py lines 71-82

```python
  71      split = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json"))
  72      VAL = set(split["validation_clusters"])
  73      vs = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in VAL]
  74      _, rows, _ = E.replay(vs, gamma=None, support=None)
  75      byq = {r["query_id"]: r for _, r in vs}
  76      import csv as _csv
  77      audited = {c["incoming_query_id"] for c in _csv.DictReader(
  78          open(ROOT / "validation/l2_independent_audit_300/selected_cases.csv"))}
  79      hits = [q for q, d in rows.items() if d["tier"] == "L2"]
  80      elig = [q for q in hits if q not in audited]
  81      print(f"  validation L2 hits {len(hits):,} | excluded (in the 300 audit) "
  82            f"{len(hits) - len(elig)} | eligible {len(elig):,}")
```

## validation_V2/l2_evidence_verification/s3b_label.py lines 178-203

```python
 178  def combine(_a):
 179      import s2_engine as E
 180      T = {t["query_id"]: t for t in jl(OUT / "val_label_tasks.jsonl")}
 181      L = {r["query_id"]: r["label"] for r in jl(OUT / "val_support_llama8b.jsonl")}
 182      Q = {r["query_id"]: r["label"] for r in jl(OUT / "val_support_qwen7b.jsonl")}
 183      sup = E.Support(OUT)
 184      import experiment as exp, schedules as sc, engine_all as ea
 185      L_ = lambda p: exp.load_jsonl(pathlib.Path(p))                     # noqa
 186      recs = exp.build_query_records(L_(exp.QUERIES_FILE), L_(exp.MANIFEST_FILE),
 187                                     L_(exp.PARAPHRASE_FILE))
 188      exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}
 189      exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
 190      ea.set_cluster_base(recs)
 191      full = sc.build_stream(recs, "zipf_uniform", 42)
 192      split = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json"))
 193      VAL = set(split["validation_clusters"])
 194      vs = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in VAL]
 195      _, rows, _ = E.replay(vs, gamma=None, support=None)
 196      byq = {r["query_id"]: r for _, r in vs}
 197      out = []
 198      for q, t in T.items():
 199          l, w = L.get(q), Q.get(q)
 200          lab = l if (l == w and l in ("SUFFICIENT", "INSUFFICIENT")) else \
 201              ("UNCLEAR_AGREED" if l == w else "JUDGE_DISAGREEMENT")
 202          out.append({"query_id": q, "label": lab, "llama8b": l, "qwen7b": w,
 203                      "support_score": sup.score(byq[q]["query"], rows[q]["ev"]),
```

## validation_V2/l2_evidence_verification/s4_select_gamma.py lines 8-29

```python
   8  assert P["preregistered"]
   9  rows = [json.loads(l) for l in open(OUT / "val_labels.jsonl") if l.strip()]
  10  use = [r for r in rows if r["label"] in ("SUFFICIENT", "INSUFFICIENT")
  11         and r["support_score"] is not None]
  12  pos = [r for r in use if r["label"] == "INSUFFICIENT"]      # should be rejected
  13  neg = [r for r in use if r["label"] == "SUFFICIENT"]        # should be kept
  14  print(f"  prereg sha {P['prereg_sha256'][:24]}...")
  15  print(f"  usable labels {len(use)}  (INSUFFICIENT {len(pos)}, SUFFICIENT {len(neg)})")
  16  print(f"  support score: INSUF mean {np.mean([r['support_score'] for r in pos]):.4f} "
  17        f"| SUF mean {np.mean([r['support_score'] for r in neg]):.4f}")
  18  best, tab = None, []
  19  for g in [round(x / 100, 2) for x in range(0, 91)]:
  20      tpr = sum(1 for r in pos if r["support_score"] < g) / len(pos)
  21      fpr = sum(1 for r in neg if r["support_score"] < g) / len(neg)
  22      j = tpr - fpr
  23      tab.append({"gamma": g, "tpr_reject_insufficient": round(tpr, 4),
  24                  "fpr_reject_sufficient": round(fpr, 4), "youden_j": round(j, 4),
  25                  "reject_rate": round(sum(1 for r in use
  26                                           if r["support_score"] < g) / len(use), 4)})
  27      if best is None or (j, -g, -fpr) > (best["youden_j"], -best["gamma"],
  28                                          -best["fpr_reject_sufficient"]):
  29          best = tab[-1]
```

## validation_V2/l2_evidence_verification/s4_select_gamma.py lines 40-43

```python
  40  auc = sum((1 if a["support_score"] > b["support_score"] else
  41             0.5 if a["support_score"] == b["support_score"] else 0)
  42            for a in neg for b in pos) / (len(neg) * len(pos))
  43  print(f"\n  discrimination AUC (SUFFICIENT scores higher) = {auc:.4f}")
```

## validation_V2/l2_evidence_verification/s2_engine.py lines 52-64

```python
  52      def score(self, query, ev):
  53          qi = self.qi.get(query)
  54          if qi is None:
  55              return None
  56          best = None
  57          for uh, tv in ev:
  58              j = self.pi.get(f"{uh}|{ma.version_at(tv)}")
  59              if j is None:
  60                  continue
  61              s = float(self.qe[qi] @ self.pe[j])
  62              if best is None or s > best:
  63                  best = s
  64          return best
```

## validation_V2/l2_evidence_verification/s2_engine.py lines 90-99

```python
  90      def l3_pure(served, t, fc):
  91          """prep2's L3 loop, simulated without mutating l3_cache."""
  92          local, ev, reused, tofetch = {}, [], 0, []
  93          for x in served:
  94              ct = local.get(x, l3_cache.get(x))
  95              if ct is not None and ok(fc, t - ct, "content"):
  96                  reused += 1; ev.append((x, ct))
  97              else:
  98                  tofetch.append(x); local[x] = t; ev.append((x, t))
  99          return ev, reused, tofetch
```

## validation_V2/l2_evidence_verification/s2_engine.py lines 146-200

```python
 146          # ---- L2 (unchanged selection) ----
 147          hit2 = -1; sim2 = None
 148          if n2:
 149              s2 = row[l2_qi[:n2]]
 150              cand = np.nonzero(s2 >= L2FLOOR)[0]
 151              if cand.size:
 152                  okm = np.fromiter((ok(l2_fc[int(j)], t - l2_t[j], "url_list")
 153                                     for j in cand), bool, cand.size)
 154                  keep = cand[okm]
 155                  if keep.size:
 156                      hit2 = int(keep[np.argmax(s2[keep])]); sim2 = float(s2[hit2])
 157  
 158          verdict = None; vscore = None
 159          if hit2 >= 0 and gamma is not None and support is not None:
 160              tent_served = list(l2_urls[hit2])
 161              tent_ev, _, _ = l3_pure(tent_served, t, fc)
 162              vscore = support.score(q, tent_ev)
 163              if vscore is None:
 164                  unscored += 1; verdict = "ACCEPT_UNSCORED"
 165              elif vscore >= gamma:
 166                  accepted += 1; verdict = "ACCEPT"
 167              else:
 168                  rejected += 1; verdict = "REJECT"
 169              scores.append({"query_id": qid, "support_score": vscore,
 170                             "verdict": verdict, "l2_similarity": sim2,
 171                             "freshness_class": fc, "t": t})
 172              if verdict == "REJECT":
 173                  hit2 = -1; sim2 = None
 174  
 175          if hit2 >= 0:
 176              l2h += 1
 177              served = list(l2_urls[hit2]); disc = float(l2_t[hit2])
 178              l2_qi[n2] = qi; l2_t[n2] = disc
 179              l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
 180          else:
 181              search += 1
 182              served = list(own); disc = t
 183              l2_qi[n2] = qi; l2_t[n2] = t
 184              l2_fc.append(fc); l2_q.append(q); l2_urls.append(served); n2 += 1
 185  
 186          ev, n_reused, tofetch = l3_pure(served, t, fc)
 187          l3h += n_reused; fetches += len(tofetch)
 188          for x in tofetch:
 189              l3_cache[x] = t
 190  
 191          rows[qid] = {"tier": ("L2" if hit2 >= 0 else
 192                                ("miss_rejected" if verdict == "REJECT" else "miss")),
 193                       "t": t, "similarity": sim2, "served": served,
 194                       "t_cached": disc, "age": (t - disc) if hit2 >= 0 else 0.0,
 195                       "ev": ev, "l3_reused_pages": n_reused, "own": own,
 196                       "gen_query": q, "verified": verdict, "support_score": vscore}
 197  
 198          l1_qi[n1] = qi; l1_t[n1] = t
 199          l1_q.append(q); l1_fc.append(fc); l1_urls.append(served)
 200          l1_src.append({"query": q, "ev": ev, "t": t}); n1 += 1
```

## validation_V2/l2_evidence_verification/s2_engine.py lines 202-211

```python
 202      n = len(stream)
 203      m = {"n_requests": n, "search_calls": search,
 204           "search_saved_pct": round(100 * (1 - search / n), 4),
 205           "l1_hits": l1h, "l2_hits": l2h, "l3_hits": l3h, "fetches": fetches,
 206           "generations": search + l2h,
 207           "fetches_per_1k": round(1000 * fetches / n, 2),
 208           "generations_per_1k": round(1000 * (search + l2h) / n, 2),
 209           "l2_accepted": accepted, "l2_rejected": rejected,
 210           "l2_unscored": unscored}
 211      return m, rows, scores
```

## validation_V2/l2_evidence_verification/s5_test_replay.py lines 30-36

```python
  30  full = sc.build_stream(recs, "zipf_uniform", 42)
  31  split = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json"))
  32  TEST = set(split["test_clusters"]); VAL = set(split["validation_clusters"])
  33  ts = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in TEST]
  34  assert not any((r.get("cluster_id") or r["query_id"]) in VAL for _, r in ts)
  35  S = support_of(ts, rounds)
  36  print(f"  test stream {len(ts):,}  |S| {len(S):,}  validation clusters present 0")
```

## validation_V2/l2_evidence_verification/s5_test_replay.py lines 63-73

```python
  63  res, allrows = {}, {}
  64  for tag, g in (("FreshCache", None), (f"L2Verify_g{G}", G),
  65                 (f"L2Verify_g{round(G-0.05,2)}", round(G - 0.05, 2)),
  66                 (f"L2Verify_g{round(G+0.05,2)}", round(G + 0.05, 2))):
  67      m, rows, sc_ = E.replay(ts, gamma=g, support=(sup if g is not None else None))
  68      res[tag] = score(m, rows)
  69      allrows[tag] = rows
  70      if g is not None:
  71          with open(OUT / f"test_scores_{tag}.jsonl", "w") as f:
  72              for r in sc_:
  73                  f.write(json.dumps(r) + "\n")
```

## validation_V2/l2_evidence_verification/s5_test_replay.py lines 80-82

```python
  80  base = res["FreshCache"]
  81  assert base["search_saved_pct"] == 60.5776, base["search_saved_pct"]
  82  print(f"  anchor: FreshCache reproduces the published held-out 60.5776% savings: True")
```

## validation_V2/l2_evidence_verification/s6_answer_audit.py lines 94-94

```python
  94      sample = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
```

## validation_V2/l2_evidence_verification/s6_answer_audit.py lines 116-139

```python
 116      for d in sample:
 117          q = d["query_id"]
 118          a, b = FC[q], LV[q]
 119          ev_a = [tuple(x) for x in (a.get("stored_answer_ev") if a["tier"] == "L1"
 120                                     else a.get("ev")) or []]
 121          ev_b = [tuple(x) for x in (b.get("stored_answer_ev") if b["tier"] == "L1"
 122                                     else b.get("ev")) or []]
 123          ctx_b = P2.ctx_for(tuple(ev_b))
 124          gq_b = b.get("gen_query", d["query"])
 125          key_b = ["gen", h(ctx_b), h(gq_b)]
 126          tasks[tuple(key_b)] = {"key": key_b, "question": gq_b, "context": ctx_b}
 127          same = (a["tier"] == b["tier"] and ev_a == ev_b)
 128          out.append({"query_id": q, "query": d["query"], "gold": d["gold"],
 129                      "fc_tier": a["tier"], "lv_tier": b["tier"],
 130                      "lv_verdict": b.get("verified"),
 131                      "lv_support_score": b.get("support_score"),
 132                      "serving_unchanged": int(same),
 133                      "fc_key": d["fc_key"], "fresh_key": d["fresh_key"],
 134                      "lv_key": key_b, "lv_ctx_sha": h(ctx_b),
 135                      "fc_ctx_sha": d["fc_ctx_sha"]})
 136          suff[(q, "fc")] = {"query_id": q, "arm": "fc", "query": d["query"],
 137                             "evidence": evblock(ev_a)}
 138          suff[(q, "lv")] = {"query_id": q, "arm": "lv", "query": d["query"],
 139                             "evidence": evblock(ev_b)}
```

## validation_V2/l2_evidence_verification/s6_answer_audit.py lines 317-327

```python
 317      def outc(d, key):
 318          v = ans.get(tuple(key))
 319          if v is None:
 320              return None
 321          if any(p in v.strip().lower() for p in ABST):
 322              return "ABSTAIN"
 323          sig = (h(d["query"]), h(d["gold"]), h(v))
 324          a, b = j["llama8b"].get(sig), j["qwen7b"].get(sig)
 325          if a is None or b is None:
 326              return None
 327          return "CORRECT" if (a == "CORRECT" and b == "CORRECT") else "WRONG"
```

## validation_V2/l2_evidence_verification/s6_answer_audit.py lines 341-359

```python
 341      for a in ("fc", "lv"):
 342          c = collections.Counter(r[a] for r in rows)
 343          w = sum(1 for r in rows if r[a] == "WRONG" and r["fresh"] == "CORRECT")
 344          res[a] = {"correct": c["CORRECT"], "abstain": c["ABSTAIN"],
 345                    "accuracy_pct": round(100 * c["CORRECT"] / N, 3),
 346                    "accuracy_ci95": wil(c["CORRECT"], N),
 347                    "wai": w, "wai_pct": round(100 * w / N, 3), "wai_ci95": wil(w, N),
 348                    "conditional_wai_pct": round(100 * w / FCc, 3) if FCc else None,
 349                    "conditional_wai_ci95": wil(w, FCc)}
 350      wa = lambda r, x: r[x] == "WRONG" and r["fresh"] == "CORRECT"          # noqa
 351      res["paired_wai"] = {"both": sum(1 for r in rows if wa(r, "fc") and wa(r, "lv")),
 352                           "fc_only": sum(1 for r in rows if wa(r, "fc") and not wa(r, "lv")),
 353                           "lv_only": sum(1 for r in rows if wa(r, "lv") and not wa(r, "fc"))}
 354      res["paired_wai"]["exact_mcnemar_p"] = mcn(res["paired_wai"]["fc_only"],
 355                                                 res["paired_wai"]["lv_only"])
 356      cw = sum(1 for r in rows if r["fc"] == "CORRECT" and r["lv"] != "CORRECT")
 357      wc = sum(1 for r in rows if r["fc"] != "CORRECT" and r["lv"] == "CORRECT")
 358      res["paired_correctness"] = {"fc_only": cw, "lv_only": wc,
 359                                   "exact_mcnemar_p": mcn(cw, wc)}
```

## validation_V2/l2_evidence_verification/s6_answer_audit.py lines 362-386

```python
 362      sl = {(r["query_id"], r["arm"]): r["label"] for r in jl(OUT / "audit_suff_llama8b.jsonl")}
 363      sq = {(r["query_id"], r["arm"]): r["label"] for r in jl(OUT / "audit_suff_qwen7b.jsonl")}
 364      suffres = {}
 365      for arm in ("fc", "lv"):
 366          lab = []
 367          for r in rows:
 368              a, b = sl.get((r["qid"], arm)), sq.get((r["qid"], arm))
 369              lab.append(a if a == b else "JUDGE_DISAGREEMENT")
 370          c = collections.Counter(lab)
 371          agreed = c["SUFFICIENT"] + c["INSUFFICIENT"]
 372          suffres[arm] = {"n": len(lab), "SUFFICIENT": c["SUFFICIENT"],
 373                          "INSUFFICIENT": c["INSUFFICIENT"], "UNCLEAR": c["UNCLEAR"],
 374                          "JUDGE_DISAGREEMENT": c["JUDGE_DISAGREEMENT"],
 375                          "sufficient_pct_of_all": round(100 * c["SUFFICIENT"] / len(lab), 3),
 376                          "ci95_of_all": wil(c["SUFFICIENT"], len(lab)),
 377                          "sufficient_pct_of_agreed": round(100 * c["SUFFICIENT"] / agreed, 3) if agreed else None,
 378                          "ci95_of_agreed": wil(c["SUFFICIENT"], agreed)}
 379      # paired sufficiency
 380      pl = [(sl.get((r["qid"], "fc")) == sq.get((r["qid"], "fc")) and sl.get((r["qid"], "fc")),
 381             sl.get((r["qid"], "lv")) == sq.get((r["qid"], "lv")) and sl.get((r["qid"], "lv")))
 382            for r in rows]
 383      a_only = sum(1 for a, b in pl if a == "SUFFICIENT" and b != "SUFFICIENT")
 384      b_only = sum(1 for a, b in pl if b == "SUFFICIENT" and a != "SUFFICIENT")
 385      suffres["paired"] = {"fc_sufficient_only": a_only, "lv_sufficient_only": b_only,
 386                           "exact_mcnemar_p": mcn(a_only, b_only)}
```

