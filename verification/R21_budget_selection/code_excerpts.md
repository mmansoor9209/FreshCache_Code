# Code excerpts (verbatim, line-numbered)

## validation_V3/parameter_calibration/04_budgets/sweep_validation.py lines 24-24

```python
  24  SEED, SCHEDULE = 42, "zipf_uniform"
```

## validation_V3/parameter_calibration/04_budgets/sweep_validation.py lines 33-44

```python
  33  def main():
  34      pre = json.load(open(HERE / "prereg.json", encoding="utf-8"))
  35      sha = hashlib.sha256((HERE / "prereg.json").read_bytes()).hexdigest()
  36      say("TASK 4 -- validation-only budget selection")
  37      say(f"  prereg sha256 {sha}")
  38      assert sha == open(HERE / "prereg.sha256").read().split()[0], \
  39          "prereg.json changed after hashing"
  40      sp = pre["search_space"]
  41      grid = list(itertools.product(sp["eps_answer"], sp["eps_url_list"],
  42                                    sp["eps_content"]))
  43      say(f"  search space {len(grid)} cells (prereg says {sp['cells']})")
  44      assert len(grid) == sp["cells"]
```

## validation_V3/parameter_calibration/04_budgets/sweep_validation.py lines 60-69

```python
  60      split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
  61                             / "split.json", encoding="utf-8"))
  62      val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])
  63      full = sc.build_stream(records, SCHEDULE, SEED)
  64      val = [(t, r) for t, r in full
  65             if (r.get("cluster_id") or r["query_id"]) in val_c]
  66      assert len(val) == split["validation"]["requests"]
  67      assert not [1 for _, r in val
  68                  if (r.get("cluster_id") or r["query_id"]) in test_c], "leakage"
  69      say(f"  validation stream {len(val):,} requests, held-out leakage 0")
```

## validation_V3/parameter_calibration/04_budgets/sweep_validation.py lines 71-87

```python
  71      base = (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT)
  72  
  73      def run(ea_, eu_, ec_):
  74          exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = ea_, eu_, ec_
  75          m, per = me.replay(val, rounds, "FreshCache", rich=rich)
  76          det = sum(1 for _, o in per if o in (me.CHANGED, me.UNCHANGED))
  77          chg = sum(1 for _, o in per if o == me.CHANGED)
  78          n = len(per)
  79          return {"eps_answer": ea_, "eps_url_list": eu_, "eps_content": ec_,
  80                  "k_answer": round(-math.log(1-ea_)/(exp.TIER_MULT["answer"]*math.log(2)), 6),
  81                  "k_url_list": round(-math.log(1-eu_)/(exp.TIER_MULT["url_list"]*math.log(2)), 6),
  82                  "k_content": round(-math.log(1-ec_)/(exp.TIER_MULT["content"]*math.log(2)), 6),
  83                  "search_saved_pct": round(m["search_saved_pct"], 4),
  84                  "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
  85                  "l3_hits": m["l3_hits"],
  86                  "coverage_pct": round(100*det/n, 4) if n else None,
  87                  "drift_pct": round(100*chg/det, 4) if det else None}
```

## validation_V3/parameter_calibration/04_budgets/sweep_validation.py lines 111-124

```python
 111      # ---- apply the pre-registered objective ----
 112      C1 = pub["drift_pct"] + 0.25
 113      C2 = pub["l1_hits"]
 114      C3 = 0.95 * pub["coverage_pct"]
 115      say(f"\n  pre-registered constraints from the published validation run:")
 116      say(f"    C1 drift <= {C1:.4f}%   C2 L1 hits <= {C2:,}   "
 117          f"C3 coverage >= {C3:.4f}%")
 118      feas = [v for v in cells.values()
 119              if v["drift_pct"] is not None and v["drift_pct"] <= C1
 120              and v["l1_hits"] <= C2 and v["coverage_pct"] >= C3]
 121      say(f"    feasible cells: {len(feas)}/{len(cells)}")
 122      sel = sorted(feas, key=lambda v: (-v["search_saved_pct"], v["drift_pct"],
 123                                        v["l1_hits"], v["eps_answer"],
 124                                        v["eps_url_list"], v["eps_content"]))[0]
```

## validation_V3/parameter_calibration/04_budgets/sweep_validation.py lines 153-171

```python
 153      json.dump({"prereg_sha256": sha, "published_validation": pub,
 154                 "constraints": {"C1_drift_max": C1, "C2_l1_max": C2,
 155                                 "C3_coverage_min": C3},
 156                 "n_feasible": len(feas), "selected": sel,
 157                 "cells": cells,
 158                 "decision_equivalent_groups": dups},
 159                open(HERE / "validation_sweep.json", "w"), indent=2)
 160      json.dump({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 161                 "selected_eps": {"answer": sel["eps_answer"],
 162                                  "url_list": sel["eps_url_list"],
 163                                  "content": sel["eps_content"]},
 164                 "tier_mult_published": dict(exp.TIER_MULT),
 165                 "selected_k": {"answer": sel["k_answer"],
 166                                "url_list": sel["k_url_list"],
 167                                "content": sel["k_content"]},
 168                 "prereg_sha256": sha,
 169                 "selected_on": "validation clusters only",
 170                 "heldout_touched_during_selection": False},
 171                open(HERE / "frozen_config.json", "w"), indent=2)
```

## validation_V3/parameter_calibration/05_heldout/fit_and_evaluate.py lines 33-34

```python
  33  SEED, SCHEDULE, B = 42, "zipf_uniform", 10_000
  34  ANCHOR = {"search_saved_pct": 60.5776, "l1_hits": 806, "l2_hits": 12429}
```

## validation_V3/parameter_calibration/05_heldout/fit_and_evaluate.py lines 150-156

```python
 150              m, per = me.replay(test, rounds, variant, rich=rich)
 151              if name == "A_original":
 152                  bad = [k for k, v in ANCHOR.items()
 153                         if abs(m[k] - v) > (1e-4 if isinstance(v, float) else 0)]
 154                  assert not bad, f"ANCHOR GATE FAILED on {bad}: {m}"
 155                  say(f"  ANCHOR GATE PASSED: config A reproduces the published "
 156                      f"held-out result exactly")
```

## validation_V3/parameter_calibration/05_heldout/fit_and_evaluate.py lines 170-181

```python
 170                  "eps": list(eps),
 171                  "k": {t: round(-math.log(1-e)/(exp.TIER_MULT[t]*math.log(2)), 6)
 172                        for t, e in zip(("answer", "url_list", "content"), eps)},
 173                  "search_saved_pct": round(m["search_saved_pct"], 4),
 174                  "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
 175                  "l3_hits": m["l3_hits"],
 176                  "observable_n": len(det),
 177                  "coverage_pct": round(100*len(det)/len(per), 4),
 178                  "drift_pct": round(100*chg/len(det), 4) if det else None,
 179                  "drift_ci95": [round(ds[int(.025*B)], 4), round(ds[int(.975*B)], 4)],
 180                  "drift_wilson95": wilson(chg, len(det)),
 181              }
```

## validation_V3/parameter_calibration/04_budgets/prereg.json (verbatim)

```json
{
  "title": "FreshCache temporal-parameter selection -- pre-registration",
  "created_before_any_heldout_run": true,
  "honesty_note": "This pre-registration was written during this reviewer-resolution workstream, AFTER the published configuration already existed. It is NOT a pre-registration of the original paper's choices, and must never be described as one. Its only claim is that the search space, objective, constraints and tie-break below were fixed and hashed BEFORE any held-out number was computed.",
  "identified_parameterisation": {
    "statement": "The gate admits iff age <= h_fc * k_t with k_t = -ln(1-eps_t)/(m_t*ln2). Decisions depend on (m_t, eps_t) only through k_t, so the multiplier/budget split is not identifiable. Selection is therefore over k_t, reported in the manuscript's (m, eps) coordinates with m held at its published value.",
    "published_k": {"answer": 0.1013353, "url_list": 0.2682843, "content": 0.6214883}
  },
  "search_space": {
    "eps_answer": [0.05, 0.10, 0.15, 0.20],
    "eps_url_list": [0.10, 0.20, 0.30, 0.35],
    "eps_content": [0.25, 0.35, 0.45],
    "tier_multipliers": "held at published (1.5, 1.2, 1.0); see identified_parameterisation -- varying them is redundant with varying eps",
    "half_lives": "held at published calibrated values; this workstream does not re-fit them",
    "cells": 48
  },
  "data": {
    "selection_population": "validation clusters only (1877 clusters / 9353 requests) from validation/heldout_baseline_tuning/split.json",
    "evaluation_population": "held-out test clusters only (4381 clusters / 21848 requests)",
    "leakage_rule": "no held-out request is replayed until the configuration is frozen and hashed"
  },
  "objective": {
    "primary": "maximise search_saved_pct on validation",
    "constraints": [
      "C1: observable content drift must not exceed the published configuration's validation drift by more than 0.25 pp (absolute)",
      "C2: L1 hits must not exceed the published configuration's validation L1 hits (each L1 hit is an answer served without regeneration, so this caps answer-risk exposure)",
      "C3: observable coverage must be at least 95% of the published configuration's validation coverage, so drift is not lowered by measuring less"
    ],
    "tie_break": [
      "T1: lower observable content drift",
      "T2: fewer L1 hits",
      "T3: lexicographically smallest (eps_answer, eps_url_list, eps_content)"
    ]
  },
  "answer_quality_note": "Answer quality (WAI) is NOT part of the selection objective: scoring it per cell would require generation and judging for all 48 cells, which was not run. C2 caps answer-risk exposure by construction instead. WAI is measured only on the frozen configurations at held-out evaluation time, and is therefore an out-of-sample check, not a selection criterion.",
  "configurations_to_evaluate_on_heldout": [
    "Original FreshCache (published)",
    "Calibrated multipliers only",
    "Calibrated budgets only",
    "Fully calibrated FreshCache",
    "EquivalentTTL derived from the calibrated rule"
  ],
  "verdict_rule": "FULLY RESOLVED only if the procedure is reproducible and leakage-free AND the frozen configuration is fairly evaluated. If parameters remain non-identifiable or labels are insufficient, state precisely what is resolved and what stays a design choice."
}
```

## validation_V3/parameter_calibration/04_budgets/frozen_config.json (verbatim)

```json
{
  "frozen_utc": "2026-09-22T05:16:41Z",
  "selected_eps": {
    "answer": 0.1,
    "url_list": 0.35,
    "content": 0.25
  },
  "tier_mult_published": {
    "answer": 1.5,
    "url_list": 1.2,
    "content": 1.0
  },
  "selected_k": {
    "answer": 0.101335,
    "url_list": 0.517907,
    "content": 0.415037
  },
  "prereg_sha256": "85be2c4d6fa70a0da3299348140ed20d4535c4c84d15527a9ff71efe9a931758",
  "selected_on": "validation clusters only",
  "heldout_touched_during_selection": false
}```
