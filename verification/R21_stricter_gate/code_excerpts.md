# R21 code excerpts (verbatim, line-numbered; read-only copies)

## <PROJECT_ROOT>/experiment.py lines 298-299

```python
 298  _EQ_SIM_FLOOR   = 0.80   # answer-tier similarity floor
 299  _EQ_JACCARD_MIN = 0.30   # content-word Jaccard floor
```

## <PROJECT_ROOT>/experiment.py lines 350-387

```python
 350  def semantic_equivalent(q1: str, q2: str, sim: float) -> bool:
 351      """
 352      Plan §6.1 semantic_equivalent(). True if serving the stored answer for
 353      q2 in response to q1 is expected to be correct.
 354  
 355      Three conjunctive conditions, ordered cheapest-first so the common
 356      rejection path costs one float comparison:
 357        1. similarity at or above the answer-tier floor
 358        2. content-word Jaccard at or above the floor
 359        3. answer types agree when both are detectable
 360      """
 361      if sim < _EQ_SIM_FLOOR:
 362          return False
 363  
 364      t1 = _eq_content_tokens(q1)
 365      t2 = _eq_content_tokens(q2)
 366      union = t1 | t2
 367      if not union:
 368          return False
 369      if len(t1 & t2) / len(union) < _EQ_JACCARD_MIN:
 370          return False
 371  
 372      a1 = _eq_answer_type(q1)
 373      a2 = _eq_answer_type(q2)
 374      if a1 is not None and a2 is not None and a1 != a2:
 375          return False
 376  
 377      # Strict-subset guard. If one query's content tokens are wholly
 378      # contained in the other's, the extra tokens are a specifier that
 379      # narrows the question ("iPhone 16" vs "iPhone 16 Pro", "final" vs
 380      # "semi-final"). Overlap is high precisely because one question is a
 381      # more specific version of the other, so Jaccard cannot catch it.
 382      # A genuine paraphrase rewords rather than appends, so it seldom
 383      # produces an exact subset.
 384      if t1 != t2 and (t1 <= t2 or t2 <= t1):
 385          return False
 386  
 387      return True
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/l1_gate.py lines 1-54

```python
   1  #!/usr/bin/env python3
   2  """
   3  The configurable L1 semantic admission condition.
   4  
   5  make_gate(...) returns a drop-in replacement for experiment.semantic_equivalent.
   6  It mirrors the published function exactly and adds only what the
   7  pre-registration declares:
   8  
   9    published rule, preserved verbatim:
  10        1. sim >= sim_floor
  11        2. content-word Jaccard >= jaccard_floor
  12        3. answer-type agreement
  13        4. strict content-token subset rejection
  14    entity agreement is NOT here: prep2/mixed_engine call experiment._entity_match
  15    separately and that call is left untouched.
  16  
  17  Only the L1 path is affected. L2 and L3 never call semantic_equivalent.
  18  """
  19  from __future__ import annotations
  20  import pathlib, sys
  21  
  22  HERE = pathlib.Path(__file__).resolve().parent
  23  sys.path.insert(0, str(HERE))
  24  import l1_guards as G   # noqa: E402
  25  
  26  
  27  def make_gate(exp, sim_floor, jac_floor, answer_type_mode, guards):
  28      """answer_type_mode: 'as_implemented' | 'required'; guards: bool."""
  29      assert answer_type_mode in ("as_implemented", "required")
  30  
  31      def semantic_equivalent(q1: str, q2: str, sim: float) -> bool:
  32          if sim < sim_floor:
  33              return False
  34          t1 = exp._eq_content_tokens(q1)
  35          t2 = exp._eq_content_tokens(q2)
  36          union = t1 | t2
  37          if not union:
  38              return False
  39          if len(t1 & t2) / len(union) < jac_floor:
  40              return False
  41          a1 = exp._eq_answer_type(q1)
  42          a2 = exp._eq_answer_type(q2)
  43          if answer_type_mode == "required":
  44              if a1 is None or a2 is None or a1 != a2:
  45                  return False
  46          else:
  47              if a1 is not None and a2 is not None and a1 != a2:
  48                  return False
  49          # strict-subset guard, verbatim from the published gate
  50          if t1 != t2 and (t1 <= t2 or t2 <= t1):
  51              return False
  52          if guards and not G.structural_guards(q1, q2):
  53              return False
  54          return True
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/l1_guards.py lines 117-124

```python
 117  def numeric_guard(q1: str, q2: str) -> bool:
 118      """Admit only if the two queries carry the SAME numeric/date information.
 119  
 120      If neither query carries any, the guard is silent (admits)."""
 121      a, b = numeric_tokens(q1), numeric_tokens(q2)
 122      if not a and not b:
 123          return True
 124      return a == b
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/l1_guards.py lines 137-155

```python
 137  def has_negation(text: str) -> bool:
 138      """Conservative: an explicit negation cue that changes the proposition.
 139  
 140      'no' is counted only when it is a standalone word and not the abbreviation
 141      for 'number' ('no. 1', 'no 5'), which is the common false positive."""
 142      t = unicodedata.normalize("NFKC", text or "").lower()
 143      if _NEG_CONTRACTION.search(t):
 144          return True
 145      if _BARE_NO.search(t):
 146          return True
 147      toks = set(re.findall(r"[a-z’']+", t))
 148      return bool(toks & {"not", "never", "without", "except", "excluding",
 149                          "cannot", "neither", "nor"})
 150  
 151  
 152  def negation_guard(q1: str, q2: str) -> bool:
 153      """Admit only if both queries agree on the presence of explicit negation."""
 154      return has_negation(q1) == has_negation(q2)
 155  
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/l1_guards.py lines 188-216

```python
 188  def relation_signature(text: str) -> frozenset:
 189      t = unicodedata.normalize("NFKC", text or "").lower()
 190      for ph in _EXCLUDE:
 191          t = t.replace(ph, " ")
 192      toks = set(re.findall(r"[a-z]+", t))
 193      sig = set()
 194      for fam, (pos, neg) in _FAMILIES.items():
 195          for pole, words in (("+", pos), ("-", neg)):
 196              for w in words:
 197                  if (" " in w and w in t) or (" " not in w and w in toks):
 198                      sig.add(f"{fam}{pole}")
 199                      break
 200      return frozenset(sig)
 201  
 202  
 203  def comparative_guard(q1: str, q2: str) -> bool:
 204      """Admit only if the two queries carry the SAME comparative/superlative
 205      relation signature. A differing pole (largest vs smallest) and a
 206      present-vs-absent relation both count as a difference."""
 207      return relation_signature(q1) == relation_signature(q2)
 208  
 209  
 210  # ───────────────────────────── combined ────────────────────────────────────
 211  def structural_guards(q1: str, q2: str) -> bool:
 212      """All three guards, conjunctive. True = admit."""
 213      return (numeric_guard(q1, q2)
 214              and negation_guard(q1, q2)
 215              and comparative_guard(q1, q2))
 216  
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/run_heldout.py lines 41-46

```python
  41  
  42  SCHEDULE, SEED, TARGET = "zipf_uniform", 42, 400
  43  CHANGED, UNCHANGED = me.CHANGED, me.UNCHANGED
  44  band, ctx_for, h = p2.band, p2.ctx_for, p2.h
  45  WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
  46           "v14_baselines/engine_all.py", "v16_exp12/schedules.py",
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/run_heldout.py lines 97-126

```python
  97              rounds[d["url_hash"]] = d["rounds"]
  98      split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
  99                             / "split.json", encoding="utf-8"))
 100      test_c, val_c = set(split["test_clusters"]), set(split["validation_clusters"])
 101      assert not (test_c & val_c)
 102      full = sc.build_stream(records, SCHEDULE, SEED)
 103      stream = [(t, r) for t, r in full
 104                if (r.get("cluster_id") or r["query_id"]) in test_c]
 105      assert len(stream) == split["test"]["requests"]
 106      assert not [1 for _, r in stream
 107                  if (r.get("cluster_id") or r["query_id"]) in val_c]
 108      S = support_of(stream, rounds)
 109      say(f"\n  held-out stream {len(stream):,} requests, {len(test_c):,} "
 110          f"clusters, |S| = {len(S):,}; validation leakage 0")
 111  
 112      ORIG = exp.semantic_equivalent
 113      ARMS = {
 114          "published": (exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN, "as_implemented", False),
 115          "strict090": (0.90, 0.30, "as_implemented", False),
 116          "precision": (FR["sim_floor"], FR["jaccard_floor"],
 117                        FR["answer_type"], bool(FR["guards"])),
 118      }
 119      mets, rowsets = {}, {}
 120      for name, (sf, jf, at, gd) in ARMS.items():
 121          exp.semantic_equivalent = l1_gate.make_gate(exp, sf, jf, at, gd)
 122          try:
 123              m_, rows_ = p2.replay(stream, {}, "FreshCache")
 124          finally:
 125              exp.semantic_equivalent = ORIG
 126          m_["generations"] = m_["search_calls"] + m_["l2_hits"]
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/run_heldout.py lines 326-334

```python
 326          sample.extend(rest[:TARGET - len(sample)])
 327      sample = sample[:TARGET]
 328      pub = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
 329                           / "answer_audit_k1_16" / "sample.json",
 330                           encoding="utf-8"))
 331      assert [d["qid"] for d in sample] == [r["query_id"] for r in pub["requests"]], \
 332          "SAMPLE GATE FAILED: not the published 400-request held-out audit"
 333      say(f"  SAMPLE GATE PASSED: the 400 audit request IDs are identical to the "
 334          f"published held-out audit (answer_audit_k1_16), in the same order")
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/analyze_heldout.py lines 38-46

```python
  38  def wilson(k, n, z=1.96):
  39      if not n:
  40          return (0.0, 0.0)
  41      p, d = k / n, 1 + z * z / n
  42      c = p + z * z / (2 * n)
  43      m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
  44      return (round(100 * max(0.0, (c - m) / d), 2),
  45              round(100 * min(1.0, (c + m) / d), 2))
  46  
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/analyze_heldout.py lines 99-108

```python
  99      say(f"\n  equivalence jury: {present}; pairs judged {len(eq):,}")
 100  
 101      def verdict(pid):
 102          v = {j: l for j, l in eq.get(pid, {}).items()
 103               if l in ("SAME", "DIFFERENT")}
 104          if len(v) < 2:
 105              return "UNJUDGED"
 106          c = Counter(v.values())
 107          return "JURY_TIE" if c["SAME"] == c["DIFFERENT"] else c.most_common(1)[0][0]
 108  
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/analyze_heldout.py lines 120-160

```python
 120      def lab_manuscript(q, gold, key):
 121          """The manuscript rule, verbatim: ABSTAIN on the abstention pattern,
 122          else CORRECT only if BOTH primary judges say CORRECT, else WRONG."""
 123          if not key or not gold:
 124              return None
 125          a = ans.get(tuple(key.split("|")) if isinstance(key, str) else tuple(key))
 126          if a is None:
 127              return None
 128          if is_abstain(a):
 129              return "ABSTAIN"
 130          sig = (h16(q), h16(gold), h16(a))
 131          l1 = jud.get(PRIM[0], {}).get(sig)
 132          l2 = jud.get(PRIM[1], {}).get(sig)
 133          if l1 is None or l2 is None:
 134              return None
 135          return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"
 136  
 137      def lab_threeway(q, gold, key):
 138          if not key or not gold:
 139              return None
 140          a = ans.get(tuple(key.split("|")) if isinstance(key, str) else tuple(key))
 141          if a is None:
 142              return None
 143          if is_abstain(a):
 144              return "ABSTAIN"
 145          sig = (h16(q), h16(gold), h16(a))
 146          ls = [jud.get(j, {}).get(sig) for j in PRIM]
 147          if any(v is None or v == "UNPARSED" for v in ls):
 148              return "UNJUDGED"
 149          if ls[0] != ls[1]:
 150              return "JUDGE_DISAGREEMENT"
 151          return "CORRECT" if ls[0] == "CORRECT" else "WRONG"
 152  
 153      for x in hits:
 154          x["stored_label"] = lab_manuscript(x["incoming_query"],
 155                                             x["reference_answer"], x["stored_key"])
 156          x["fresh_label"] = lab_manuscript(x["incoming_query"],
 157                                            x["reference_answer"], x["fresh_key"])
 158          x["wai"] = int(x["stored_label"] == "WRONG"
 159                         and x["fresh_label"] == "CORRECT")
 160  
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/analyze_heldout.py lines 165-204

```python
 165          P = [x for x in hits if x["arm"] == arm]
 166          n = len(P)
 167          det = [x for x in P if x["equivalence"] in ("SAME", "DIFFERENT")]
 168          diff = sum(1 for x in det if x["equivalence"] == "DIFFERENT")
 169          ties = sum(1 for x in P if x["equivalence"] == "JURY_TIE")
 170          xc = [x for x in P if x["cross_cluster"] == "1"]
 171          xdet = [x for x in xc if x["equivalence"] in ("SAME", "DIFFERENT")]
 172          xdiff = sum(1 for x in xdet if x["equivalence"] == "DIFFERENT")
 173          xties = sum(1 for x in xc if x["equivalence"] == "JURY_TIE")
 174          ad = [x for x in P if x["stored_label"] in ("CORRECT", "WRONG")
 175                and x["fresh_label"] in ("CORRECT", "WRONG")]
 176          sc = sum(1 for x in ad if x["stored_label"] == "CORRECT")
 177          fc = sum(1 for x in ad if x["fresh_label"] == "CORRECT")
 178          w = sum(x["wai"] for x in ad)
 179          pops[arm] = {
 180              "l1_hits": n, "jury_ties": ties, "jury_determinate": len(det),
 181              "mismatch_ties_diff": {"k": diff + ties, "n": n,
 182                                     "pct": (round(100*(diff+ties)/n, 4) if n else None),
 183                                     "ci": wilson(diff + ties, n)},
 184              "mismatch_no_ties": {"k": diff, "n": len(det),
 185                                   "pct": (round(100*diff/len(det), 4) if det else None),
 186                                   "ci": wilson(diff, len(det))},
 187              "mismatch_ties_same": {"k": diff, "n": n,
 188                                     "pct": (round(100*diff/n, 4) if n else None),
 189                                     "ci": wilson(diff, n)},
 190              "cross_cluster_hits": len(xc),
 191              "cross_cluster_mismatch_ties_diff": {
 192                  "k": xdiff + xties, "n": len(xc),
 193                  "pct": (round(100*(xdiff+xties)/len(xc), 4) if xc else None)},
 194              "cross_cluster_mismatch_no_ties": {
 195                  "k": xdiff, "n": len(xdet),
 196                  "pct": (round(100*xdiff/len(xdet), 4) if xdet else None)},
 197              "answer_determinate": len(ad),
 198              "stored_accuracy": {"k": sc, "n": len(ad),
 199                                  "pct": (round(100*sc/len(ad), 4) if ad else None)},
 200              "hit_level_wai": {"k": w, "n": len(ad),
 201                                "pct": (round(100*w/len(ad), 4) if ad else None)},
 202              "hit_level_conditional_wai": {"k": w, "n": fc,
 203                                            "pct": (round(100*w/fc, 4) if fc else None)},
 204          }
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/gen_judge_l1p.py lines 70-80

```python
  70                  [{"role": "user", "content": EQUIV.format(a=c["incoming"],
  71                                                            b=c["cached"])}],
  72                  tokenize=False, add_generation_prompt=True) for c in ch]
  73              enc = tok(texts, return_tensors="pt", padding=True,
  74                        truncation=True, max_length=1024).to(model.device)
  75              with torch.no_grad():
  76                  g = model.generate(**enc, max_new_tokens=6, do_sample=False,
  77                                     pad_token_id=tok.pad_token_id)
  78              for c, t in zip(ch, tok.batch_decode(
  79                      g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
  80                  up = t.upper()
```

## <PROJECT_ROOT>/validation_V3/l1_precision_gate/gen_judge_l1p.py lines 160-170

```python
 160                  [{"role": "user", "content": G.JUDGE.format(
 161                      q=c["q"], gold=c["gold"], cand=c["cand"])}],
 162                  tokenize=False, add_generation_prompt=True) for c in ch]
 163              enc = tok(texts, return_tensors="pt", padding=True,
 164                        truncation=True, max_length=2048).to(model.device)
 165              with torch.no_grad():
 166                  g = model.generate(**enc, max_new_tokens=6, do_sample=False,
 167                                     pad_token_id=tok.pad_token_id)
 168              for c, t in zip(ch, tok.batch_decode(
 169                      g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
 170                  up = t.upper()
```

