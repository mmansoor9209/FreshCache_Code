# Observability audit: verbatim code excerpts (line-numbered, read-only copies)

## v9/build_version_timeline.py lines 37-49

```python
  37  RUNS = ["run_00", "rerun_1h", "rerun_12h", "rerun_24h", "rerun_7d"]
  38  RUN_AGE = {"run_00": 0.0, "rerun_1h": 3_600.0, "rerun_12h": 43_200.0,
  39             "rerun_24h": 86_400.0, "rerun_7d": 604_800.0}
  40  MIN_BODY = 400            # chars of extracted text for a body change to be visible
  41  BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|"
  42                     r"access denied|are you a robot|captcha|cloudflare|"
  43                     r"403 forbidden|404 not found|page not found", re.I)
  44  
  45  
  46  def substantive(t: str) -> bool:
  47      if not t or len(t.strip()) < MIN_BODY:
  48          return False
  49      return not BLOCK.search(t[:400])
```

## v9/build_version_timeline.py lines 70-75

```python
  70      rows = []
  71      for uh, per in tl.items():
  72          base = per.get("run_00")
  73          reruns = [r for r in RUNS[1:] if r in per]
  74          observable = bool(base and base["sub"] and
  75                            any(per[r]["sub"] for r in reruns))
```

## v13_corrected/a1b_round_table.py lines 18-28

```python
  18  MIN_BODY = 400
  19  BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|"
  20                     r"access denied|are you a robot|captcha|cloudflare|"
  21                     r"403 forbidden|404 not found|page not found", re.I)
  22  TABLE = OUT/"corrected_round_table.jsonl"
  23  
  24  
  25  def substantive(t):
  26      if not t or len(t.strip()) < MIN_BODY:
  27          return False
  28      return not BLOCK.search(t[:400])
```

## v13_corrected/a1b_round_table.py lines 49-54

```python
  49                  "content_hash": s.get("content_hash"),
  50                  "text_hash": hashlib.md5(t.encode("utf-8")).hexdigest(),
  51                  "n_chars": len(t), "status_code": s.get("status_code"),
  52                  "fetched_at": s.get("fetched_at"),
  53                  "substantive": bool(substantive(t) and s.get("status_code") == 200)}
  54      with open(TABLE, "w", encoding="utf-8") as f:
```

## v10_remaining_feedback/e1_c1_observable_refit.py lines 84-88

```python
  84      # ---- Definition 2: common observable set for t=24h -----------------
  85      common_24h = {u for u, r in tl.items()
  86                    if r["runs"].get("run_00", {}).get("sub")
  87                    and r["runs"].get("rerun_24h", {}).get("sub")}
  88  
```

## v10_remaining_feedback/e1_c1_observable_refit.py lines 191-197

```python
 191      obs_records = []
 192      dropped = Counter()
 193      for r in records:
 194          keep = [u for u in r["urls"] if u["url_hash"] in common_24h]
 195          if keep:
 196              obs_records.append({**r, "urls": keep})
 197          else:
```

## validation/observability_by_class/analyze.py lines 118-139

```python
 118      tl = {r["url_hash"]: r for r in (json.loads(l) for l in open(TL))}
 119      manifest = exp.load_jsonl(exp.MANIFEST_FILE)
 120      changes = exp.load_jsonl(exp.CHANGE_FILE)
 121  
 122      # ---- 1. reproduce the published totals ----
 123      def sub(u, run):
 124          return bool(tl[u]["runs"].get(run, {}).get("sub"))
 125      common_24h = {u for u in tl if sub(u, "run_00") and sub(u, WIN)}
 126      n_total, n_obs = len(tl), len(common_24h)
 127      say(f"\n  DEFINITION 2 (manuscript): run_00 substantive AND {WIN} substantive")
 128      say(f"    body-observable {n_obs:,} of {n_total:,} unique URLs "
 129          f"({100*n_obs/n_total:.4f}%)")
 130      ok_total = (n_total == 8635)
 131      ok_obs = (n_obs == 4382)
 132      assert ok_total and ok_obs, (
 133          f"cannot reproduce published totals: got {n_obs}/{n_total}, "
 134          f"expected 4,382/8,635")
 135      say("    reproduces the published 4,382 / 8,635 exactly")
 136      say("    DENOMINATOR UNIT = one unique URL (url_hash) in "
 137          "v9/version_timeline.jsonl,")
 138      say("    i.e. a run_00 -> rerun_24h transition of that URL. Not a URL-query")
 139      say("    pair and not a per-request observation.")
```

## validation/observability_by_class/analyze.py lines 141-163

```python
 141      # ---- 3. class attribution safety (do this BEFORE any class split) ----
 142      url_classes = defaultdict(set)
 143      url_domain = {}
 144      for m in manifest:
 145          if m.get("snapshot_available"):
 146              url_classes[m["url_hash"]].add(m.get("freshness_class", ""))
 147              url_domain.setdefault(m["url_hash"], m.get("domain", ""))
 148      multi = {u: c for u, c in url_classes.items() if len(c) > 1}
 149      in_tl_multi = {u for u in multi if u in tl}
 150      say(f"\n  CLASS ATTRIBUTION SAFETY")
 151      say(f"    URLs with a manifest class: {len(url_classes):,}")
 152      say(f"    URLs appearing under MORE THAN ONE freshness class: "
 153          f"{len(multi):,} ({100*len(multi)/max(len(url_classes),1):.2f}%)")
 154      say(f"    of those, present in the 8,635 timeline: {len(in_tl_multi):,}")
 155      say(f"    class counts per URL: "
 156          f"{dict(Counter(len(c) for c in url_classes.values()))}")
 157      ambiguous = len(in_tl_multi) > 0
 158      if ambiguous:
 159          say("    -> unique URLs are NOT uniquely class-attributable. Each unique")
 160          say("       URL is therefore NOT forced into one class. The per-class unit")
 161          say("       below is the URL-CLASS-WINDOW tracked observation, exactly the")
 162          say("       unit c1_observability_by_class.csv already uses.")
 163  
```

## validation/observability_by_class/analyze.py lines 165-178

```python
 165      chg24 = {c["url_hash"] for c in changes
 166               if not c.get("noise") and c.get("run_id") == WIN
 167               and c.get("detected_at", "") > exp.NEW_RUN_CUTOFF}
 168      rows, miss_rows, byclass = [], [], {}
 169      for fc in CLASSES:
 170          tracked = [m for m in manifest
 171                     if m.get("snapshot_available") and m.get("run_id") == WIN
 172                     and m.get("freshness_class") == fc
 173                     and not is_noise_url(m.get("domain", ""), fc)]
 174          n = len(tracked)
 175          obs = [m for m in tracked
 176                 if m["url_hash"] in tl and sub(m["url_hash"], "run_00")
 177                 and sub(m["url_hash"], WIN)]
 178          no = n - len(obs)
```

## validation/observability_by_class/analyze.py lines 224-247

```python
 224      tot_tr = sum(v["tracked"] for v in byclass.values())
 225      tot_ob = sum(v["observable"] for v in byclass.values())
 226      say(f"    {'TOTAL':<11}{tot_tr:>9,}{tot_ob:>12,}"
 227          f"{100*tot_ob/tot_tr:>8.2f}%{tot_tr-tot_ob:>9,}"
 228          f"{100*(tot_tr-tot_ob)/tot_tr:>8.2f}%")
 229      say(f"    NOTE: {tot_tr:,} observations > {n_total:,} unique URLs because a URL")
 230      say(f"    tracked under k classes contributes k observations. The two units are")
 231      say(f"    reported separately and never mixed.")
 232  
 233      # ---- 7. chi-square on observability x class ----
 234      obs_tab = [[byclass[fc]["observable"], byclass[fc]["not_observable"]]
 235                 for fc in CLASSES]
 236      N = sum(sum(r) for r in obs_tab)
 237      rs = [sum(r) for r in obs_tab]
 238      cs = [sum(obs_tab[i][j] for i in range(len(CLASSES))) for j in (0, 1)]
 239      exp_tab = [[rs[i] * cs[j] / N for j in (0, 1)] for i in range(len(CLASSES))]
 240      minexp = min(min(r) for r in exp_tab)
 241      chi2 = sum((obs_tab[i][j] - exp_tab[i][j]) ** 2 / exp_tab[i][j]
 242                 for i in range(len(CLASSES)) for j in (0, 1))
 243      df = (len(CLASSES) - 1) * 1
 244      p = chi2_p(chi2, df)
 245      V = math.sqrt(chi2 / (N * min(len(CLASSES) - 1, 1)))
 246      say(f"\n  CHI-SQUARE  observability x class: chi2 = {chi2:.4f}, df = {df}, "
 247          f"p = {p:.4g}, Cramer's V = {V:.4f}  (min expected cell {minexp:,.1f})")
```

## validation/observability_by_class/analyze.py lines 249-268

```python
 249      # ---- 4. query-level observability ----
 250      queries = exp.load_jsonl(exp.QUERIES_FILE)
 251      paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
 252               if exp.PARAPHRASE_FILE.exists() else [])
 253      records = exp.build_query_records(queries, manifest, paras)
 254      qrows, qtot = [], Counter()
 255      keptc, dropc = Counter(), Counter()
 256      for r in records:
 257          fc = r["freshness_class"]
 258          qtot[fc] += 1
 259          if any(u["url_hash"] in common_24h for u in r["urls"]):
 260              keptc[fc] += 1
 261          else:
 262              dropc[fc] += 1
 263      kept, total = sum(keptc.values()), sum(qtot.values())
 264      say(f"\n  QUERY-LEVEL: retained {kept:,} of {total:,} "
 265          f"({100*kept/total:.4f}%)")
 266      assert total == 31201, f"records total {total}, expected 31,201"
 267      assert kept == 18216, f"retained {kept}, expected 18,216"
 268      say("    reproduces the published 18,216 of 31,201 exactly")
```

## validation_V3/observability_resolution/01_reproduce/reproduce_and_taxonomy.py lines 40-52

```python
  40      rows = list(csv.DictReader(open(OB / "per_request_pair.csv", encoding="utf-8")))
  41      pr = [r for r in rows if r["paired"] == "1"]
  42      N = len(pr)
  43      c_on = sum(1 for r in pr if r["outcome_on"] == CHANGED)
  44      c_off = sum(1 for r in pr if r["outcome_off"] == CHANGED)
  45      u_on = sum(1 for r in pr if r["outcome_on"] == UNOBS)
  46      u_off = sum(1 for r in pr if r["outcome_off"] == UNOBS)
  47      jo = [r for r in pr if r["outcome_on"] != UNOBS and r["outcome_off"] != UNOBS]
  48      con = sum(1 for r in jo if r["outcome_on"] == CHANGED)
  49      coff = sum(1 for r in jo if r["outcome_off"] == CHANGED)
  50      d_obs = 100*coff/len(jo) - 100*con/len(jo)
  51      dmin = 100*(c_off - (c_on + u_on))/N
  52      dmax = 100*((c_off + u_off) - c_on)/N
```

## validation_V3/observability_resolution/01_reproduce/reproduce_and_taxonomy.py lines 84-99

```python
  84      for uh, d in rounds.items():
  85          for run, e in d["rounds"].items():
  86              if e.get("substantive"):
  87                  cat["substantive"] += 1
  88                  continue
  89              sc, nc = e.get("status_code"), e.get("n_chars", 0)
  90              if sc != 200:
  91                  k = f"non_200_status ({sc})"
  92              elif nc == 0:
  93                  k = "empty_extraction"
  94              elif nc < MIN_BODY:
  95                  k = f"short_text_under_{MIN_BODY}"
  96                  nchar_short.append(nc)
  97              else:
  98                  k = "block_page_pattern"
  99              cat[k] += 1
```

## validation_V3/observability_resolution/02_recovery/recover_and_recompute.py lines 68-78

```python
  68      base = {}
  69      with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
  70                encoding="utf-8") as fh:
  71          for line in fh:
  72              d = json.loads(line)
  73              base[d["url_hash"]] = d["rounds"]
  74      short = [(u, r) for u, rr in base.items() for r, e in rr.items()
  75               if not e.get("substantive") and e.get("status_code") == 200
  76               and 0 < e.get("n_chars", 0) < 400]
  77      say(f"  short-text observations to re-examine: {len(short):,}")
  78      blocked = 0
```

## validation_V3/observability_resolution/02_recovery/recover_and_recompute.py lines 97-107

```python
  97      def rounds_at(thr):
  98          """A round table whose `substantive` flag uses threshold `thr`.
  99          The original table is never modified."""
 100          out = {}
 101          for u, rr in base.items():
 102              out[u] = {}
 103              for r, e in rr.items():
 104                  s = bool(e.get("substantive"))
 105                  if not s and (u, r) in texts and e.get("n_chars", 0) >= thr:
 106                      s = True
 107                  out[u][r] = {**e, "substantive": s}
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

## validation_V3/observability_resolution/02_recovery/recover_and_recompute.py lines 176-178

```python
 176          return {"N_paired": N, "jointly_observable": len(jo),
 177                  "coverage_pct": round(100*len(jo)/N, 4) if N else None,
 178                  "changed_on": c_on, "changed_off": c_off,
```

