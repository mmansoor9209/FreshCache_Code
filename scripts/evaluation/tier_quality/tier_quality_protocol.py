"""Freeze this study's configuration before any generation or judging."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, hashlib, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
W = ((OUTPUT_ROOT / "tier_quality"))
C = {
 "generator": {"model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "FORCED prompt verbatim from scripts/evaluation/answer_audits/general_audit_generation_judging.py",
               "decoding": "greedy, max_new_tokens 80, no chat template, 1024-token truncation, left padding, batch 16, answer cut at first '.' or newline (scripts/evaluation/answer_audits/general_audit_generation_judging.py)",
               "reuse_rule": "a preserved answer is reused only when its (role, sha16(context), sha16(question)) key matches exactly; fidelity gate re-generates a sample of preserved answers"},
 "answer_grading": {"protocol_file": "config/answer_protocol_frozen.json", "protocol_sha256_expected": "d601fac400ec42197b8d9e8f07bf7821f9b23a2eec40dd1aa0c5cc38c5abf591",
                    "stage_used": "answer_stage only (3 judges, blinded, pattern abstention first, then 2-of-3 majority, else UNRESOLVABLE)",
                    "parser": "parser of scripts/evaluation/reference_validation/reference_validation_reparse.py (INCORRECT->WRONG; Mistral misspellings)"},
 "sufficiency_grading": {"judges": ["llama8b", "qwen7b", "mistral7b"],
   "prompt": ("You are evaluating search-result evidence for retrieval-augmented QA.\n\nIncoming query:\n{q}\n\nSearch results provided for this request:\n{ev}\n\n"
              "Determine whether these search results contain sufficient information or\ncredible evidence that could support answering the incoming query.\n\n"
              "Return exactly one label:\n\nSUFFICIENT\nINSUFFICIENT\nUNCLEAR\n\nSUFFICIENT means at least one or more results provide evidence directly\n"
              "relevant enough to support answering the query.\nINSUFFICIENT means the results are largely unrelated or do not provide\nuseful evidence for answering the query.\n"
              "UNCLEAR means relevance cannot be determined reliably from the available\nevidence."),
   "prompt_source": "scripts/experiments/admission_controls/verifier_answer_audit.py, with one change for blinding: 'served from the cache' -> 'provided', because path A is not a cache path",
   "decoding": "chat template (user turn), greedy, max_new_tokens 8", "aggregation": "2-of-3 majority, else NO_MAJORITY",
   "blinding": "one task per distinct (request, evidence text); seeded shuffle; no path label"},
 "references": {"primary": "dataset reference answer for every request",
   "validated": "previously validated target-time references (model_based_reference_validation, final VALID_AT_T) where the request is in that study; no new reference validation is run here"},
 "tests": "exact two-sided McNemar on discordant pairs", "seed": 42,
 "no_retuning": "the held-out FreshCache replay uses the recorded configuration; nothing is retuned"}
json.dump(C, open((REPO_ROOT / "configs/protocols/tier_quality_study_config.json"), "w"), indent=1)
open((REPO_ROOT / "configs/protocols/tier_quality_study_config.sha256"), "w").write(hashlib.sha256(((REPO_ROOT / "configs/protocols/tier_quality_study_config.json")).read_bytes()).hexdigest() + "\n"); print("frozen")
