# Line-numbered code excerpts (read-only copies, current repository state)

### calibrate.py:109-122
```python
  109: FIT_WINDOWS_BY_CLASS = {
  110:     "MEDIUM":    ["rerun_1h", "rerun_12h"],
  111:     "FAST":      ["rerun_1h", "rerun_12h"],
  112:     "TIMELESS":  ["rerun_1h", "rerun_12h", "rerun_24h"],
  113:     "SLOW":      ["rerun_1h", "rerun_12h", "rerun_24h"],
  114:     "REAL_TIME": [],   # kept as prior, not fit
  115: }
  116: HOLDOUT_WINDOWS_BY_CLASS = {
  117:     "MEDIUM":    ["24h", "7d"],
  118:     "FAST":      ["24h", "7d"],
  119:     "TIMELESS":  ["7d"],
  120:     "SLOW":      ["7d"],
  121:     "REAL_TIME": [],
  122: }
```

### calibrate.py:248-290
```python
  248: def fit_half_life_multi_window(rate_age_pairs: list) -> tuple:
  249:     """
  250:     W2 per-class mixed-window calibration strategy.
  251: 
  252:     Fits half-life via time-weighted MLE across whichever (rate, age)
  253:     observations fall inside this class's FIT_WINDOWS_BY_CLASS, matching
  254:     the time-weighted averaging already used in
  255:     RuleBasedRiskModel.calibrate() in risk_model.py.
  256: 
  257:     rate_age_pairs: list of (change_rate, age_seconds) tuples, one per
  258:     run_id in the class's fit window. Pairs with rate <= 0 or rate >= 1
  259:     are skipped (no usable signal at that window).
  260: 
  261:     Returns (half_life_seconds, note_string). Returns (None, note) if no
  262:     pair in the fit window has usable signal.
  263:     """
  264:     usable = [(r, a) for r, a in rate_age_pairs if r is not None and 0 < r < 1]
  265:     if not usable:
  266:         return None, "no usable signal in fit window"
  267: 
  268:     lambdas = [_fit_lambda_from_rate(r, a) for r, a in usable]
  269:     ages    = [a for _, a in usable]
  270:     avg_lam = sum(l * a for l, a in zip(lambdas, ages)) / sum(ages)
  271:     hl      = math.log(2) / avg_lam
  272:     windows_used = [a for _, a in usable]
  273:     note = f"time-weighted MLE over {len(usable)} window(s), ages={windows_used}"
  274:     return round(hl, 1), note
  275: 
  276: 
  277: def _fit_lambda_from_rate(rate: float, age: float) -> float:
  278:     """
  279:     Fit λ from a single observed change rate at a given age.
  280:     P(stale) = 1 - exp(-λ × age) → λ = -ln(1 - P) / age
  281:     Clamp rate to avoid log(0).
  282:     """
  283:     p = max(0.01, min(0.99, rate))
  284:     return -math.log(1.0 - p) / age
  285: 
  286: 
  287: def compute_run_class_rates(manifest: list, changes: list) -> dict:
  288:     """
  289:     Direct (run_id, freshness_class) -> change_rate counter, computed from
  290:     raw manifest/change_log rather than the limited workload_analysis.json
```

### calibrate.py:296-336
```python
  296:     INCLUDING the is_noise_url filter — omitting this filter was the root
  297:     cause of the first broken run, where noisy/mismatched URLs diluted the
  298:     change rate and produced inflated half-lives for MEDIUM/FAST.
  299:     """
  300:     try:
  301:         from freshcache.risk_model import is_noise_url
  302:     except Exception as e:
  303:         print(f"  [!] Could not import is_noise_url from freshcache.risk_model "
  304:               f"({e}) — proceeding WITHOUT noise filtering, rates will not "
  305:               f"match the original calibration logic.")
  306:         is_noise_url = lambda domain, fc: False
  307: 
  308:     tracked: Dict[tuple, int] = defaultdict(int)
  309:     for m in manifest:
  310:         if (m.get("snapshot_available")
  311:                 and m.get("run_id", "run_00") != "run_00"):
  312:             domain = m.get("domain", "")
  313:             fc     = m.get("freshness_class", "")
  314:             if fc and is_noise_url(domain, fc):
  315:                 continue
  316:             key = (m["run_id"], fc)
  317:             tracked[key] += 1
  318: 
  319:     changed: Dict[tuple, int] = defaultdict(int)
  320:     for c in changes:
  321:         if c.get("noise", False):
  322:             continue
  323:         if c.get("detected_at", "") <= NEW_RUN_CUTOFF:
  324:             continue
  325:         key = (c.get("run_id"), c.get("freshness_class", ""))
  326:         changed[key] += 1
  327: 
  328:     rates = {}
  329:     for key, total in tracked.items():
  330:         if total == 0:
  331:             continue
  332:         rates[key] = changed.get(key, 0) / total
  333:     return rates
  334: 
  335: 
  336: # ---------------------------------------------------------------------------
```

### calibrate.py:493-517
```python
  493: def compute_observed_domain_volatility(ground_truth: dict) -> dict:
  494:     """
  495:     Compute per-domain change rate from both reruns combined.
  496:     Only include domains with >= 2 observations for reliability.
  497: 
  498:     Note: domain volatility is a per-domain prior used as a content-level
  499:     adjustment, not the per-class half-life that W2 concerns. It is left
  500:     using all available windows (1h, 24h) since it does not feed the
  501:     half-life fit being held out for W2 and is not the quantity the
  502:     reviewer's t=24h staleness evaluation is computed against.
  503:     """
  504:     domain_obs     = defaultdict(list)
  505: 
  506:     for info in ground_truth.values():
  507:         domain = info["domain"]
  508:         for changed in [info["changed_1h"], info["changed_24h"]]:
  509:             if changed is not None:
  510:                 domain_obs[domain].append(int(changed))
  511: 
  512:     volatility = {}
  513:     for domain, obs in domain_obs.items():
  514:         if len(obs) >= 2:
  515:             volatility[domain] = round(sum(obs) / len(obs), 3)
  516: 
  517:     return dict(sorted(volatility.items(),
```

### calibrate.py:730-748
```python
  730: 
  731:     # ── Domain volatility ─────────────────────────────────────────────
  732:     print("\nComputing observed domain volatility...")
  733:     domain_vol = compute_observed_domain_volatility(ground_truth)
  734: 
  735:     print(f"\n  Top volatile domains (from real observations):")
  736:     print(f"  {'Domain':<38} {'Obs. Volatility':>16}")
  737:     print(f"  {'-'*38} {'-'*16}")
  738:     for domain, vol in list(domain_vol.items())[:15]:
  739:         if vol > 0:
  740:             print(f"  {domain:<38} {vol:>16.3f}")
  741: 
  742:     # ── Patch risk_model.py ───────────────────────────────────────────
  743:     print("\nPatching freshcache/risk_model.py...")
  744:     patched = patch_risk_model(fitted, domain_vol)
  745:     if patched:
  746:         print("  risk_model.py updated with W2 temporally-held-out half-lives.")
  747:     else:
  748:         print("  Patch skipped — update _DEFAULT_HALF_LIFE manually using values above.")
```

### freshcache/risk_model.py:93-100
```python
   93: # ---------------------------------------------------------------------------
   94: # Default domain volatility priors (overridden by calibrate())
   95: # ---------------------------------------------------------------------------
   96: _DEFAULT_DOMAIN_VOLATILITY: Dict[str, float] = {
   97:     "twitter.com":           0.95,
   98:     "x.com":                 0.95,
   99:     "reddit.com":            0.85,
  100:     "finance.yahoo.com":     0.95,
```

### freshcache/risk_model.py:128-153
```python
  128:     "tradingeconomics.com":  0.85,
  129:     "worldbank.org":         0.20,
  130:     "statista.com":          0.25,
  131:     "ign.com":               0.30,
  132:     "socialblade.com":       0.50,
  133:     "boxofficemojo.com":     0.45,
  134: 
  135:     # These domains have dynamic page elements that cause false-positive
  136:     # change detection in TIMELESS content. Volatility set just above
  137:     # the TIMELESS ceiling (0.45) so they are excluded from TIMELESS
  138:     # calibration but remain valid for SLOW calibration.
  139:     # "namu.wiki":             0.46,   # was 0.40 — wiki with dynamic edit counters
  140:     "m.blog.naver.com":      0.46,   # Naver blog with view counters in body
  141:     "blog.naver.com":        0.46,   # same
  142:     "brainly.com":           0.46,   # Q&A site with ad content in body
  143:     "2news.com": 1.0,   # local TV news with rotating video widget
  144:     "newsbytesapp.com": 1.0,   # news app with "In the news" trending sidebar
  145: 
  146:     # Sports and entertainment — dynamic nav sections
  147:     "sports.yahoo.com":      0.65,   # Yahoo Sports with trending nav
  148: 
  149:     # Genuine misclassifications — these are not TIMELESS content
  150:     "vietcombank.com.vn":    0.95,   # live exchange rate table
  151:     "en.vietnamplus.vn": 1.0,   # Vietnamese news agency
  152: }
  153: 
```

### freshcache/risk_model.py:154-186
```python
  154: # ── Domain volatility ceiling per freshness class ──────────────────────────
  155: # If a URL's domain volatility exceeds this ceiling for its assigned query
  156: # class, the URL is a search quality mismatch and is excluded from
  157: # class-level calibration.
  158: CLASS_VOL_CEILING: Dict[str, float] = {
  159:     "TIMELESS":  0.45,   # was 0.25 — raised so unknown domains (default 0.40) pass through
  160:     "SLOW":      0.50,
  161:     "MEDIUM":    0.70,
  162:     "FAST":      0.90,
  163:     "REAL_TIME": 1.00,
  164: }
  165: 
  166: _DEFAULT_DOMAIN_VOL = 0.40   # fallback for unknown domains
  167: 
  168: 
  169: def is_noise_url(domain: str, freshness_class: str) -> bool:
  170:     """
  171:     Return True if this URL's domain volatility is too high for its
  172:     assigned query class. Scales automatically to any dataset size.
  173:     """
  174:     domain = (domain or "").lower().removeprefix("www.")
  175:     vol    = _DEFAULT_DOMAIN_VOLATILITY.get(domain, _DEFAULT_DOMAIN_VOL)
  176:     # Partial match for subdomains
  177:     if vol == _DEFAULT_DOMAIN_VOL:
  178:         for known, v in _DEFAULT_DOMAIN_VOLATILITY.items():
  179:             if domain.endswith("." + known):
  180:                 vol = v
  181:                 break
  182:     ceiling = CLASS_VOL_CEILING.get(freshness_class, 0.50)
  183:     return vol > ceiling
  184: 
  185: # ---------------------------------------------------------------------------
  186: # Answer-type keyword patterns
```

### freshcache/risk_model.py:411-452
```python
  411:         # Fit domain volatility from per-domain change rates
  412:         domain_changed: Dict[str, int] = defaultdict(int)
  413:         domain_total:   Dict[str, int] = defaultdict(int)
  414:         for c in changes:
  415:             domain_changed[c.get("domain", "")] += 1
  416:         for line in mf_path.open(encoding="utf-8"):
  417:             try:
  418:                 r = json.loads(line)
  419:                 if r.get("snapshot_available") and r.get("run_id") != "run_00":
  420:                     domain_total[r.get("domain", "")] += 1
  421:             except Exception:
  422:                 continue
  423: 
  424:         for domain, total in domain_total.items():
  425:             if not domain or total < 2:
  426:                 continue
  427:             rate = domain_changed.get(domain, 0) / total
  428:             self._domain_vol[domain] = round(rate, 3)
  429:         # Enforce monotonic minimum half-lives per class
  430:         MIN_HALF_LIVES = {
  431:             FreshnessClass.REAL_TIME: 60.0,         # min 1 minute
  432:             FreshnessClass.FAST:      1_800.0,      # min 30 min
  433:             FreshnessClass.MEDIUM:    43_200.0,     # min 12h
  434:             FreshnessClass.SLOW:      172_800.0,    # min 48h
  435:             FreshnessClass.TIMELESS:  2_592_000.0,  # min 30 days
  436:         }
  437:         for fc, min_hl in MIN_HALF_LIVES.items():
  438:             if self._half_life[fc] < min_hl:
  439:                 self._half_life[fc] = min_hl
  440:                 self._calibration_log.append(
  441:                     f"  → {fc.value}: half_life clamped to minimum {min_hl/3600:.1f}h"
  442:                 )
  443:         self._calibrated = True
  444:         self._calibration_log.append(
  445:             f"\nCalibration complete. "
  446:             f"{len(class_lambdas)} classes fitted, "
  447:             f"{len(domain_total)} domains fitted."
  448:         )
  449: 
  450:     def print_calibration_report(self) -> None:
  451:         print("\n=== Risk Model Calibration Report ===")
  452:         for line in self._calibration_log:
```

### experiment.py:113-134
```python
  113: # Calibrated half-lives (from risk_model.calibrate())
  114: # W2 UPDATE: these values are now fit using a per-class temporal holdout
  115: # (calibrate.py, FIT_WINDOWS_BY_CLASS). MEDIUM/FAST are fit on rerun_1h+
  116: # rerun_12h only; TIMELESS/SLOW are fit on rerun_1h+rerun_12h+rerun_24h.
  117: # The holdout is ASYMMETRIC, not uniform. t=24h is held out for MEDIUM and
  118: # FAST only; TIMELESS and SLOW use rerun_24h in their own fit, so for those
  119: # two classes the t=24h evaluation is in-sample. t=7d is later than the
  120: # fitting window for all four. An earlier version of this comment claimed
  121: # 24h was held out for every class; that was wrong and is corrected here.
  122: # This still replaces the pre-W2 values, which were fit on every window
  123: # including the evaluation windows for all four classes.
  124: # Synced manually from data/calibration_report.json — calibrate.py only
  125: # patches freshcache/risk_model.py, not this file.
  126: # ---------------------------------------------------------------------------
  127: HALF_LIFE = {
  128:     "REAL_TIME":         30.0,        # 30s   (kept as prior, unchanged)
  129:     "FAST":          449_280.0,       # 5.2d  (W2 LOCKED: 1h+12h fit, 24h+7d held out)
  130:     "MEDIUM":        457_920.0,       # 5.3d  (W2 LOCKED: 1h+12h fit, 24h+7d held out)
  131:     "SLOW":          993_600.0,       # 11.5d (W2 LOCKED: 1h+12h+24h fit, 7d held out)
  132:     "TIMELESS":    2_592_000.0,       # 30d   (W2 LOCKED: 1h+12h+24h fit, 7d held out)
  133: }
  134: # This is the final W2 calibration. With the C18 semantic-equivalence gate
```

### experiment.py:422-437
```python
  422: def p_stale(fc: str, age: float, tier: str = "content") -> float:
  423:     """P(stale) at full floating-point precision.
  424: 
  425:     PRECISION FIX: this previously returned round(..., 4). The value is used
  426:     directly as a decision boundary (p_stale <= EPS_*), so rounding it moved
  427:     the boundary: a true risk anywhere in (eps, eps + 5e-5) was pulled back
  428:     onto eps and the entry was reused. That made the exponential gate slightly
  429:     more permissive than its own equivalent TTL,
  430:     -h*ln(1-eps)/(m*ln2), which is the exact inverse of the UNROUNDED rule.
  431:     Rounding here is now removed; round only a copy used for display, via
  432:     p_stale_display(). Half-lives, tier multipliers and epsilons are unchanged.
  433:     """
  434:     hl  = HALF_LIFE.get(fc, 86_400.0)
  435:     mul = TIER_MULT.get(tier, 1.0)
  436:     lam = math.log(2) / hl * mul
  437:     return 1.0 - math.exp(-lam * age)
```

### experiment.py:566-578
```python
  566: 
  567: 
  568: def build_stale_sets(changes: list) -> dict:
  569:     stale = {"rerun_1h": set(), "rerun_24h": set()}
  570:     for c in changes:
  571:         if c.get("noise", False):
  572:             continue
  573:         rid = c.get("run_id")
  574:         if rid == "rerun_1h" and c.get("detected_at", "") > NEW_RUN_CUTOFF:
  575:             stale["rerun_1h"].add(c["url_hash"])
  576:         elif rid == "rerun_24h" and c.get("detected_at", "") > NEW_RUN_CUTOFF:
  577:             stale["rerun_24h"].add(c["url_hash"])
  578:         elif rid == "rerun_12h" and c.get("detected_at", "") > NEW_RUN_CUTOFF:
```

### recompute_changes.py:25-25
```python
   25: from freshcache.risk_model import _DEFAULT_DOMAIN_VOLATILITY, is_noise_url, CLASS_VOL_CEILING
```

### recompute_changes.py:44-66
```python
   44: #     "SLOW":      0.50,
   45: #     "MEDIUM":    0.70,
   46: #     "FAST":      0.90,
   47: #     "REAL_TIME": 1.00,
   48: # }
   49: 
   50: # DEFAULT_DOMAIN_VOL = 0.40   # fallback for unknown domains
   51: 
   52: 
   53: # def get_domain_volatility(domain: str) -> float:
   54: #     """Look up domain volatility from the risk model's prior table."""
   55: #     if not domain:
   56: #         return DEFAULT_DOMAIN_VOL
   57: #     domain = domain.lower().removeprefix("www.")
   58: #     if domain in _DEFAULT_DOMAIN_VOLATILITY:
   59: #         return _DEFAULT_DOMAIN_VOLATILITY[domain]
   60: #     # Partial match (e.g. subdomain of a known domain)
   61: #     for known, vol in _DEFAULT_DOMAIN_VOLATILITY.items():
   62: #         if domain.endswith("." + known):
   63: #             return vol
   64: #     return DEFAULT_DOMAIN_VOL
   65: 
   66: def get_domain_volatility(domain: str) -> float:
```

### v16_exp12/mixed_engine.py:87-93
```python
   87:     per = rounds.get(uh)
   88:     if not per:
   89:         return UNOBS
   90:     a, b = per.get(ma.version_at(tc)), per.get(ma.version_at(tr))
   91:     if not a or not b or not a["substantive"] or not b["substantive"]:
   92:         return UNOBS
   93:     return CHANGED if a["content_hash"] != b["content_hash"] else UNCHANGED
```

### v13_corrected/a1_dataset_audit.py:145-172
```python
  145:     prev_noise = defaultdict(set)
  146:     for c in changes:
  147:         (prev_noise if c.get("noise") else prev_changed)[c["run_id"]].add(c["url_hash"])
  148:     print(f"  change_log.jsonl rows {len(changes):,}")
  149:     for w in WINDOWS:
  150:         print(f"    {w:<12} changed {len(prev_changed[w]):>6,}   "
  151:               f"noise-flagged {len(prev_noise[w]):>6,}")
  152:     print(f"\n  build_stale_sets() keeps only non-noise rows and folds rerun_12h")
  153:     print(f"  into the 24h stale set as a proxy. Everything absent from that set")
  154:     print(f"  is treated as UNCHANGED by every simulator.")
  155: 
  156:     # ---------- corrected transitions ----------
  157:     print(f"\n{'='*104}\n  STEP 3 — CORRECTED TRANSITION LABELS\n{'='*104}")
  158:     rows, summary = [], {}
  159:     for w in WINDOWS:
  160:         cnt = Counter(); why = Counter(); mism = Counter()
  161:         for u, per in snaps.items():
  162:             a, b = per.get("run_00"), per.get(w)
  163:             if a is None or b is None:
  164:                 lab, reason = "UNOBSERVABLE", ("missing_run_00" if a is None
  165:                                                else f"missing_{w}")
  166:             elif not a["substantive"] or not b["substantive"]:
  167:                 lab = "UNOBSERVABLE"
  168:                 reason = (f"run_00_{a['unobservable_reason']}" if not a["substantive"]
  169:                           else f"{w}_{b['unobservable_reason']}")
  170:             else:
  171:                 reason = None
  172:                 lab = "CHANGED" if a["content_hash"] != b["content_hash"] else "UNCHANGED"
```


## Addendum excerpts

### calibrate.py:784-795
```python
  784: def _fmt_hl(seconds: float) -> str:
  785:     """Human-readable half-life label."""
  786:     if seconds < 120:
  787:         return f"{seconds:.0f}s"
  788:     if seconds < 7_200:
  789:         return f"{seconds/3600:.1f}h"
  790:     if seconds < 172_800:
  791:         return f"{seconds/3600:.0f}h"
  792:     if seconds < 1_209_600:
  793:         return f"{seconds/86400:.1f}d"
  794:     return f"{seconds/86400:.0f}d"
  795: 
```

### calibrate.py:553-560
```python
  553:     }
  554:     for fc in order:
  555:         hl  = new_half_lives.get(fc, 30.0)
  556:         lbl = labels.get(fc, "")
  557:         lines.append(f"    FreshnessClass.{fc:<10}: {hl:>15.1f},   # {lbl}")
  558: 
  559:     new_hl_block = "_DEFAULT_HALF_LIFE: Dict[FreshnessClass, float] = {\n" + \
  560:                    "\n".join(lines) + "\n}"
```

### calibrate.py:575-585
```python
  575: 
  576:     # Only update the domains we have observed data for
  577:     for domain, vol in top_domains:
  578:         safe_domain = domain.replace(".", r"\.")
  579:         pattern_d = rf'("{safe_domain}":\s*)[\d.]+,'
  580:         replacement = f'"{domain}": {vol},'
  581:         src = re.sub(pattern_d, replacement, src)
  582: 
  583:     with open(RISK_MODEL, "w", encoding="utf-8") as f:
  584:         f.write(src)
  585: 
```

### validation_V3/parameter_calibration/02_targets/build_targets.py:113-128
```python
  113:     for fc in ["REAL_TIME", "FAST", "MEDIUM", "SLOW", "TIMELESS"]:
  114:         for rnd, age in AGES.items():
  115:             g = [r for r in l3 if r["freshness_class"] == fc and r["round"] == rnd]
  116:             if len(g) < 30:
  117:                 continue
  118:             n = len(g)
  119:             raw = sum(r["changed_raw"] for r in g) / n
  120:             den = sum(r["changed_denoised"] for r in g) / n
  121:             pred = exp.p_stale(fc, age, "content")
  122:             cal.append({"freshness_class": fc, "age_s": age, "n": n,
  123:                         "observed_raw": round(100 * raw, 4),
  124:                         "observed_denoised": round(100 * den, 4),
  125:                         "ci95_denoised": wilson(sum(r['changed_denoised'] for r in g), n),
  126:                         "model_pct": round(100 * pred, 4),
  127:                         "abs_err_pp": round(100 * abs(pred - den), 4)})
  128:             say(f"    {fc:<11}{age/3600:>7.0f}h{n:>8,}{100*raw:>8.2f}%"
```

### freshcache/risk_model.py:309-316
```python
  309:     def calibrate(
  310:     self,
  311:     change_log_path: str = "data/change_log.jsonl",
  312:     manifest_path:   str = "data/url_manifest.jsonl",
  313:     excluded_runs:   list = None,
  314:     ) -> None:
  315:         excluded = set(excluded_runs or [])
  316:         """
```

### freshcache/risk_model.py:609-622
```python
  609:     def _mlp_features(self, freshness_class, delta_t, domain, query=None,
  610:                   sim_to_cached=None, n_cached_urls=None,
  611:                   evidence_domains=None, cached_query=None,
  612:                   rich_features_map=None):
  613:         # ── Existing 17 features ─────────────────────────────────────────────
  614:         CLASSES = ["TIMELESS","SLOW","MEDIUM","FAST","REAL_TIME"]
  615:         class_oh = np.zeros(5)
  616:         if freshness_class in CLASSES:
  617:             class_oh[CLASSES.index(freshness_class)] = 1.0
  618: 
  619:         log_age  = min(math.log(1 + delta_t / 3600 / 72), 1.0)
  620:         dom_vol  = self._domain_vol.get(domain, 0.1)
  621: 
  622:         TYPES = ["price","score","weather","exchange_rate","stock",
```

### experiment.py:167-178
```python
  167:     from freshcache.risk_model import LearnedRiskModel as _LearnedRiskModel
  168:     from freshcache.models   import (CacheTier      as _CacheTier,
  169:                                      FreshnessClass  as _FreshnessClass,
  170:                                      QueryFeatures   as _QueryFeatures)
  171:     _MLP_RISK = _LearnedRiskModel()
  172:     _MLP_RISK.calibrate(str(CHANGE_FILE), str(MANIFEST_FILE))
  173:     _FC_ENUM   = {fc: _FreshnessClass(fc)
  174:                   for fc in ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]}
  175:     _TIER_ENUM = {"answer":   _CacheTier.ANSWER,
  176:                   "url_list": _CacheTier.URL_LIST,
  177:                   "content":  _CacheTier.CONTENT}
  178:     _MLP_AVAILABLE = getattr(_MLP_RISK, "_mlp", None) is not None
```

