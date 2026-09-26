# Addendum code excerpts (verbatim, line-numbered)

## v9/build_version_timeline.py lines 46-49

```python
  46  def substantive(t: str) -> bool:
  47      if not t or len(t.strip()) < MIN_BODY:
  48          return False
  49      return not BLOCK.search(t[:400])
```

## experiment.py lines 503-518

```python
 503      baseline = [m for m in manifest
 504                  if m.get("run_id") == "run_00"
 505                  and m.get("snapshot_available")]
 506  
 507      # Build url index keyed by query_id
 508      query_urls: dict = defaultdict(list)
 509      for m in baseline:
 510          qid = m["query_id"]
 511          query_urls[qid].append({
 512              "url_hash": m["url_hash"],
 513              "url":      m["url"],
 514              "domain":   m.get("domain", ""),
 515              "rank":     m.get("rank", 99),
 516          })
 517      for qid in query_urls:
 518          query_urls[qid].sort(key=lambda x: x["rank"])
```

## experiment.py lines 541-556

```python
 541      # Add paraphrase records (borrow URLs from base query)
 542      if paraphrases:
 543          for p in paraphrases:
 544              base_id = p.get("base_query_id", "")
 545              base_urls = base_url_by_qid.get(base_id)
 546              if not base_urls:
 547                  continue   # base query had no URLs — skip
 548              records.append({
 549                  "query_id":        p["query_id"],
 550                  "query":           p["query"],
 551                  "freshness_class": p["freshness_class"],
 552                  "language":        p.get("language", "en"),
 553                  "cluster_id":      p.get("cluster_id") or base_id,
 554                  "is_paraphrase":   True,
 555                  "urls":            base_urls,
 556              })
```

## v16_exp12/e1_robustness.py lines 50-61

```python
  50  def support_of(stream, rounds):
  51      S = set()
  52      for t, r in stream:
  53          if r["freshness_class"] == "REAL_TIME":
  54              continue
  55          v = ma.version_at(t)
  56          for u in r["urls"]:
  57              per = rounds.get(u["url_hash"])
  58              if per and per.get("run_00", {}).get("substantive") \
  59                      and per.get(v, {}).get("substantive"):
  60                  S.add(r["query_id"]); break
  61      return S
```

## v16_exp12/mixed_engine.py lines 86-94

```python
  86  def outcome(uh, tc, tr, rounds):
  87      per = rounds.get(uh)
  88      if not per:
  89          return UNOBS
  90      a, b = per.get(ma.version_at(tc)), per.get(ma.version_at(tr))
  91      if not a or not b or not a["substantive"] or not b["substantive"]:
  92          return UNOBS
  93      return CHANGED if a["content_hash"] != b["content_hash"] else UNCHANGED
  94  
```

## v16_exp12/mixed_engine.py lines 176-183

```python
 176          if fc == "REAL_TIME" and rt_bypass:
 177              rt_requests += 1
 178              req["realtime_bypass"] += 1
 179              search += 1
 180              for h in own:
 181                  fetches += 1
 182                  l3_cache[h] = t
 183              per_req.append((qid, None))
```

## v16_exp12/mixed_engine.py lines 223-228

```python
 223              labs = [outcome(x, float(l1_t[hit1]), t, rounds) for x in l1_urls[hit1]]
 224              o = (CHANGED if CHANGED in labs else
 225                   (UNCHANGED if UNCHANGED in labs else UNOBS))
 226              if l1m == "scalm":
 227                  l1_freq[hit1] += 1
 228              per_req.append((qid, o))
```

## v16_exp12/mixed_engine.py lines 281-290

```python
 281                          labs.append(outcome(x, l3_cache[x], t, rounds))
 282                          continue
 283                      ent["l3_temporal_rejections"] += 1
 284                  fetches += 1
 285                  l3_cache[x] = t
 286          else:
 287              fetches += len(served)
 288          per_req.append((qid, (CHANGED if CHANGED in labs else
 289                                (UNCHANGED if UNCHANGED in labs else UNOBS))
 290                          if labs else None))
```

## validation_V3/observability_resolution/02_recovery/recover_and_recompute.py lines 152-161

```python
 152      def scoreset(ON, OFF):
 153          pr = [q for q in ON if ON[q] is not None and OFF.get(q) is not None]
 154          N = len(pr)
 155          c_on = sum(1 for q in pr if ON[q] == CHANGED)
 156          c_off = sum(1 for q in pr if OFF[q] == CHANGED)
 157          u_on = sum(1 for q in pr if ON[q] == UNOBS)
 158          u_off = sum(1 for q in pr if OFF[q] == UNOBS)
 159          jo = [q for q in pr if ON[q] != UNOBS and OFF[q] != UNOBS]
 160          con = sum(1 for q in jo if ON[q] == CHANGED)
 161          coff = sum(1 for q in jo if OFF[q] == CHANGED)
```

