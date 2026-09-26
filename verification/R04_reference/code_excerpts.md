# R04 code excerpts (verbatim, line-numbered)

## validation_V3/strong_reference_evaluation/02_reference_verification/verify_and_diagnose.py lines 39-43

```python
  39  VALIDITY_MAP = {"VALID_AT_T": "SUPPORTED",
  40                  "SUPERSEDED_AT_T": "CONTRADICTED",
  41                  "INDETERMINATE_AT_T": "UNVERIFIABLE",
  42                  "JUDGE_DISAGREEMENT": "UNVERIFIABLE"}
  43  YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
```

## validation_V3/strong_reference_evaluation/02_reference_verification/verify_and_diagnose.py lines 125-129

```python
 125          g = gv.get(qid, {})
 126          val = VALIDITY_MAP.get(g.get("validity", ""), "UNVERIFIABLE")
 127          vol = g.get("volatility", "")
 128          q_has_year = bool(YEAR.search(d["query"]))
 129          temporally_ambiguous = (vol == "VOLATILE" and not q_has_year)
```

## validation_V3/strong_reference_evaluation/02_reference_verification/verify_and_diagnose.py lines 152-156

```python
 152      for r in rows:
 153          r["reference_status_final"] = ("TEMPORALLY_AMBIGUOUS"
 154                                         if r["reference_status"] == "UNVERIFIABLE"
 155                                         and r["temporally_ambiguous"]
 156                                         else r["reference_status"])
```

## validation/gold_verification_400/stage3b_validity.py lines 20-31

```python
  20  Rules enforced in code, not merely in the prompt:
  21    * MISSING EVIDENCE IS NEVER SUPERSEDED. A unit with no retrievable snapshot
  22      text is assigned INDETERMINATE_AT_T with reason NO_EVIDENCE and is not sent
  23      to a judge at all.
  24    * A 2023 annotation is never assumed to hold in 2026. The judge is told the
  25      annotation year and the evidence date and is asked about the evidence date.
  26    * A status is assigned only when BOTH judges agree; otherwise the unit is
  27      JUDGE_DISAGREEMENT and stays unresolved.
  28  
  29  Evidence assembly reuses the audits' own loader and budget: per-page cap 2000
  30  chars (e2e_answer_grading.MAX_CONTEXT_CHARS), pages under 200 chars dropped,
  31  joined under a 2800-char budget — identical to prep2.ctx_for.
```

## validation/gold_verification_400/stage3b_validity.py lines 72-92

```python
  72  SYSTEM_PROMPT = (
  73      "You are checking whether a benchmark answer was still up to date at a "
  74      "specific point in time, using web page text captured at that time.\n\n"
  75      "You are given a QUESTION, a BENCHMARK ANSWER with the year it was "
  76      "annotated, and EVIDENCE consisting of web page text captured later, in "
  77      f"{CORPUS_DATE}. Decide what the evidence says about the benchmark answer "
  78      "AT THE TIME THE EVIDENCE WAS CAPTURED.\n\n"
  79      "Reply with exactly one word:\n"
  80      "VALID        the evidence indicates the benchmark answer was still the "
  81      "current answer when the evidence was captured.\n"
  82      "SUPERSEDED   the evidence indicates a DIFFERENT current answer, so the "
  83      "benchmark answer was out of date.\n"
  84      "INDETERMINATE the evidence does not settle the question either way.\n\n"
  85      "Important:\n"
  86      "- Absence of the benchmark answer from the evidence is NOT enough to say "
  87      "SUPERSEDED. Say SUPERSEDED only if the evidence states a different "
  88      "current answer.\n"
  89      "- Do not assume the benchmark answer is still correct merely because it "
  90      "was annotated in an earlier year. Judge only from the evidence.\n"
  91      "- If the evidence is off-topic, incomplete, or silent on the point, say "
  92      "INDETERMINATE."
```

## validation/gold_verification_400/stage3b_validity.py lines 236-252

```python
 236              if t["aliases"] and any(a.strip().lower() != gold.strip().lower()
 237                                      for a in t["aliases"]):
 238                  gold = gold + "  (also accepted: " + "; ".join(
 239                      a for a in t["aliases"] if a.strip().lower()
 240                      != t["gold_display"].strip().lower()) + ")"
 241              msgs = [{"role": "system", "content": SYSTEM_PROMPT},
 242                      {"role": "user", "content": USER_TEMPLATE.format(
 243                          q=t["base_query"], gold=gold,
 244                          year=t["gold_year"] or "unknown",
 245                          when=ROUND_LABEL.get(t["snapshot_round"],
 246                                               t["snapshot_round"]) +
 247                               f", in {CORPUS_DATE}",
 248                          ev=t["evidence"])}]
 249              text = tok.apply_chat_template(msgs, tokenize=False,
 250                                             add_generation_prompt=True)
 251              enc = tok([text], return_tensors="pt", truncation=True,
 252                        max_length=4096).to(model.device)
```

## validation/gold_verification_400/stage3b_validity.py lines 366-379

```python
 366              say(f"  STOP: unit {uid} missing a judge label")
 367              sys.exit(3)
 368          agree = (l == w)
 369          rows.append({**{k: t[k] for k in
 370                          ("unit_id", "base_query_id", "base_query",
 371                           "snapshot_round", "gold", "gold_source", "gold_year",
 372                           "volatility", "freshness_classes", "n_requests",
 373                           "request_ids", "n_urls", "n_pages_kept",
 374                           "evidence_chars")},
 375                       "status_llama8b": l, "status_qwen7b": w,
 376                       "judges_agree": int(agree),
 377                       "final_status": l if agree else "JUDGE_DISAGREEMENT",
 378                       "reason": "dual_judge_agreement" if agree
 379                                 else "judges_disagree"})
```

## validation/gold_verification_400/revised_3b/r1_build.py lines 33-36

```python
  33  
  34  SNAP = ROOT / "data" / "snapshots"
  35  MAXCHARS_TOTAL = 12000        # was 2800
  36  MAXCHARS_PAGE = 5000          # was 2000 (collect.py stores at most 5000)
```

## validation/gold_verification_400/revised_3b/r1_build.py lines 60-61

```python
  60  # DATED and HISTORICAL are asked "correct as stated"; the rest "still current".
  61  AS_STATED = {"DATED", "HISTORICAL"}
```

## validation/gold_verification_400/revised_3b/r1_build.py lines 126-128

```python
 126              "freshness_classes": "|".join(fcs),
 127              "n_requests": int(u["n_requests"]), "request_ids": u["request_ids"],
 128              "question_type": qt, "mode": "AS_STATED" if qt in AS_STATED else "CURRENT",
```

## validation/gold_verification_400/revised_3b/r4_analyze.py lines 104-123

```python
 104      M = {r["unit_id"]: r["status"] for r in jl(HERE / "r_validity_mistral7b.jsonl")}
 105      miss = [u for u in tasks if u not in L or u not in Q]
 106      if miss:
 107          say(f"  STOP: {len(miss)} units missing a primary judge label"); sys.exit(3)
 108      say(f"  units judged by the primary pair: {len(tasks)}")
 109      say(f"  units judged by the third model : {len(M)}")
 110  
 111      VALIDish = {"VALID_AT_T"}
 112      rows = []
 113      for u, t in tasks.items():
 114          l, q = L[u], Q[u]
 115          pair = l if l == q else "JUDGE_DISAGREEMENT"
 116          if l == q:
 117              maj, src = l, "pair_agreement"
 118          elif u in M:
 119              m = M[u]
 120              maj = (m if (m == l or m == q) else "NO_MAJORITY")
 121              src = "third_judge_majority" if maj != "NO_MAJORITY" else "three_way_split"
 122          else:
 123              maj, src = "JUDGE_DISAGREEMENT", "no_third_judge"
```

## validation/gold_verification_400/revised_3b/r4_analyze.py lines 172-178

```python
 172      for res in ("pair", "majority"):
 173          ks = set()
 174          for qid in rmap:
 175              f = fid.get(qid, "BASE_QUERY")
 176              u = unit_of.get(qid)
 177              if f in ("SAME", "BASE_QUERY") and u and u[f"status_{res}"] in VALIDish:
 178                  ks.add(qid)
```

## validation/gold_verification_400/stage3a_paraphrase.py lines 12-16

```python
  12    judges   meta-llama/Llama-3.1-8B-Instruct  (primary)
  13             Qwen/Qwen2.5-7B-Instruct          (independent second judge)
  14    decoding chat template, greedy, do_sample=False, max_new_tokens=6, bfloat16
  15    labels   SAME / DIFFERENT / UNCLEAR; a label is assigned only when BOTH
  16             judges agree, otherwise the request is JUDGE_DISAGREEMENT
```

## Reframe_Paper_Pro/scripts/crosswalk_57_107.py lines 19-52

```python
  19  fid={r["query_id"]:r["final_label"] for r in csv.DictReader(open(GV/"paraphrase_fidelity.csv"))}
  20  unit_of={}
  21  for r in csv.DictReader(open(RV/"r_gold_validity.csv")):
  22      for q in r["request_ids"].split("|"): unit_of[q]=r
  23  EV={q for q in unit_of if fid.get(q,"BASE_QUERY") in ("SAME","BASE_QUERY") and unit_of[q]["status_majority"]=="VALID_AT_T"}
  24  def outc(d,key):
  25      a=ans.get(tuple(key))
  26      if a is None: return None
  27      if any(p in a.strip().lower() for p in ABST): return "ABSTAIN"
  28      s=(h(d["query"]),h(d["gold"]),h(a)); l1,l2=j1.get(s),j2.get(s)
  29      if l1 is None or l2 is None: return None
  30      return "CORRECT" if (l1=="CORRECT" and l2=="CORRECT") else "WRONG"
  31  rows={}
  32  for d in sample:
  33      r={"fresh":outc(d,d["fresh_key"]),"fc":outc(d,d["fc_key"]),
  34         "sttl_k1_16":outc(d,d["sttl_key"]),"sttl_t060_k1_2":outc(d,new[d["query_id"]]["new_sttl_key"])}
  35      if any(v is None for v in r.values()): continue
  36      rows[d["query_id"]]=r
  37  held=set(rows); screened107=EV&held
  38  sup=csv.DictReader(open(SRC/"validation_V3/strong_reference_evaluation/02_reference_verification/reference_and_diagnosis.csv"))
  39  supported57={r["query_id"] for r in sup if r["reference_status"]=="SUPPORTED"}
  40  print(f"held-out scored: {len(rows)}   screened (fidelity+temporal-gold majority): {len(screened107)}   supported (two-judge VALID_AT_T): {len(supported57)}")
  41  g={a:sum(1 for r in rows.values() if r['fresh']=='CORRECT' and r[a]=='WRONG') for a in ('fc','sttl_k1_16','sttl_t060_k1_2')}
  42  print("gate request-level WAI fc/k1_16/t060_k1_2 =",g, "(published 7/64/25)")
  43  assert g=={'fc':7,'sttl_k1_16':64,'sttl_t060_k1_2':25}, "gate failed"
  44  I=screened107&supported57
  45  print(f"\ncrosswalk: |57 ∩ 107| = {len(I)}   57-only = {len(supported57-screened107)}   107-only = {len(screened107-supported57)}")
  46  def score(ids,label):
  47      sub=[rows[q] for q in ids]; n=len(sub); fcn=sum(1 for r in sub if r["fresh"]=="CORRECT")
  48      out={"n":n,"fresh_correct":fcn}
  49      for a in ("fc","sttl_t060_k1_2","sttl_k1_16"):
  50          c=sum(1 for r in sub if r[a]=="CORRECT"); w=sum(1 for r in sub if r["fresh"]=="CORRECT" and r[a]=="WRONG")
  51          out[a]={"correct":c,"acc_pct":round(100*c/n,2),"wai":w,"wai_pct":round(100*w/n,2),
  52                  "cwai_pct":round(100*w/fcn,2) if fcn else None}
```

## validation/heldout_baseline_tuning/answer_audit_k1_16/prep.py lines 293-296

```python
 293                 "simbase": d["simbase"], "simbase_band": band(d["simbase"]),
 294                 "fresh_ctx_sha": h(fctx), "fresh_pages": len(fcrows[qid]["own"]),
 295                 "fresh_versions": [[u, t] for u in fcrows[qid]["own"]]}
 296          tasks[("fresh", h(fctx), h(d["query"]))] = (d["query"], fctx)
```

