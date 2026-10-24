"""Stage 1: freeze the protocol (models, prompts, decoding, aggregation, metrics) BEFORE any judge runs."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, hashlib, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
W = ((OUTPUT_ROOT / "reference_validation"))
P = {
 "judges": {"llama8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct", "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"},
 "judge_families": {"llama8b": "Meta Llama", "qwen7b": "Alibaba Qwen", "mistral7b": "Mistral AI"},
 "decoding": {"do_sample": False, "max_new_tokens": 12, "dtype": "bfloat16", "chat_template": True, "max_input_tokens": 6144},
 "reference_stage": {
   "inputs_shown": ["question", "target time (capture date of the evidence)", "reference answer", "contemporaneous snapshot text"],
   "never_shown": ["policy identity", "any policy output", "any earlier validity label", "annotation year or dataset name"],
   "labels": ["VALID_AT_T", "INVALID_AT_T", "TEMPORALLY_AMBIGUOUS", "UNVERIFIABLE"],
   "system": ("You verify reference answers for a question-answering benchmark. You are given a QUESTION, the TARGET TIME at which the question "
              "is asked, a REFERENCE ANSWER, and EVIDENCE: web page text captured at the target time. Decide, using ONLY the evidence, whether the "
              "reference answer was a correct answer to the question at the target time.\n\n"
              "Reply with exactly one label:\n"
              "VALID_AT_T  the evidence supports the reference answer as correct at the target time.\n"
              "INVALID_AT_T  the evidence states a different answer that applies at the target time, so the reference answer was not correct then.\n"
              "TEMPORALLY_AMBIGUOUS  the correct answer depends on a time period the question does not fix, or the evidence supports the reference only "
              "for a different period, so correctness at the target time cannot be settled.\n"
              "UNVERIFIABLE  the evidence is off-topic, incomplete, or silent on the point.\n\n"
              "Rules: absence of the reference answer from the evidence is not enough for INVALID_AT_T. Do not use your own background knowledge "
              "about later events. Do not assume the reference is correct."),
   "user": "QUESTION: {question}\nTARGET TIME: {target}\nREFERENCE ANSWER: {reference}\n\nEVIDENCE (captured {captured}):\n{evidence}\n\nLabel:",
   "no_evidence_rule": "a request with no usable snapshot text is labelled UNVERIFIABLE without calling a judge",
   "aggregation": "final label = a label chosen by at least 2 of 3 judges; otherwise NO_MAJORITY; unparsable output counts as UNPARSED and cannot contribute to a majority",
   "validated_subset": "final label VALID_AT_T (majority of three)"},
 "answer_stage": {
   "inputs_shown": ["question", "target time", "reference answer", "candidate answer"],
   "never_shown": ["policy identity", "which arm produced the answer", "evidence", "other arms' answers"],
   "blinding": "candidate answers are deduplicated on (query_id, answer text) across arms and presented in a seeded random order (seed 42)",
   "labels": ["CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE"],
   "system": ("You grade a candidate answer against a reference answer for a question asked at a given time. Reply with exactly one label:\n"
              "CORRECT  the candidate conveys the same factual answer as the reference. Wording, extra detail or extra context do not matter.\n"
              "WRONG  the candidate gives a different or contradictory answer.\n"
              "ABSTAIN  the candidate declines to answer, says it does not know, or says the information is insufficient.\n"
              "UNRESOLVABLE  it is impossible to tell whether the candidate matches the reference (for example the reference or candidate is "
              "incoherent)."),
   "user": "QUESTION: {question}\nTIME ASKED: {target}\nREFERENCE ANSWER: {reference}\nCANDIDATE ANSWER: {candidate}\n\nLabel:",
   "abstention_rule": "the preserved pattern rule is applied first and unchanged: an answer containing any of the listed phrases is ABSTAIN",
   "abstain_patterns": ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"],
   "aggregation": "if the pattern rule fires: ABSTAIN; else the label chosen by at least 2 of 3 judges; otherwise UNRESOLVABLE"},
 "metrics": {
   "accuracy": "CORRECT / N", "abstention_rate": "ABSTAIN / N", "WAI": "fresh CORRECT and arm WRONG, divided by N",
   "conditional_WAI": "WAI count / fresh-CORRECT count", "fresh_correct_to_abstain": "fresh CORRECT and arm ABSTAIN (reported separately, not WAI)",
   "fresh_correct_to_unresolvable": "reported separately, not WAI",
   "paired_tests": "exact two-sided McNemar (binomial) on discordant pairs: correctness vs fresh; FreshCache vs each SemanticTTL arm for correctness and for WAI"},
 "arms": {"fresh": "fresh pipeline", "freshcache": "FreshCache", "semanticttl_quality_0.60_1-2": "SemanticTTL theta=0.60 kappa=1/2 (quality-selected)",
          "semanticttl_drift_0.40_1-16": "SemanticTTL theta=0.40 kappa=1/16 (drift-selected); kept strictly separate",
          "freshcache_without_l1": "NOT AVAILABLE: no preserved held-out outputs"},
 "seed": 42,
 "no_retuning": "no policy, threshold, gate or configuration is changed; policy outputs are the preserved generations"}
json.dump(P, open(W/"config/protocol.json", "w"), indent=1)
h = hashlib.sha256((W/"config/protocol.json").read_bytes()).hexdigest()
open(W/"config/protocol.sha256", "w").write(h + "  config/protocol.json\n"); print("protocol frozen", h)
