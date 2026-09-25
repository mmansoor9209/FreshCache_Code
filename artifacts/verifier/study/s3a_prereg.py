#!/usr/bin/env python3
"""s3a_prereg.py — freeze the gamma-selection rule BEFORE any sufficiency label
is produced. Run first; s4 refuses to select without this file."""
import hashlib, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
P = {
 "preregistered": True,
 "written_before_any_sufficiency_label": True,
 "ablation": "FreshCache-L2Verify (optional gate after L2 hit + normal L3)",
 "support_score": "V(q,C) = max_c cosine(E(q), E(c)); BGE-M3 dense (CLS, "
                  "L2-normalised); c = each served page at the version L3 "
                  "actually served, text via the audits' own 2000-char loader",
 "online_path": "no LLM; embeddings precomputed offline",
 "labels_for_selection": "judge-agreed SUFFICIENT/INSUFFICIENT on the ACTUAL "
                         "served L2/L3 evidence, Llama-3.1-8B + Qwen2.5-7B, "
                         "both must agree; disagreements excluded",
 "label_population": "L2 hits on the VALIDATION cluster split only; every "
                     "request appearing in the existing 300-case audit is "
                     "EXCLUDED so nothing is fitted on audited examples",
 "candidate_gamma": "0.00 to 0.90 inclusive, step 0.01",
 "objective": "maximise Youden's J = TPR - FPR, where POSITIVE = INSUFFICIENT; "
              "TPR = share of INSUFFICIENT L2 hits rejected, FPR = share of "
              "SUFFICIENT L2 hits rejected",
 "tie_breaks": ["smaller gamma (retains more savings)", "lower FPR"],
 "frozen_before_test": True,
 "test_use": "the frozen gamma is replayed once on the held-out test split; "
             "no tuning on test; sensitivity at gamma+-0.05 reported separately",
 "sensitivity": [-0.05, 0.05],
}
P["prereg_sha256"] = hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest()
json.dump(P, open(HERE / "out" / "prereg_gamma.json", "w"), indent=2)
print("  gamma-selection rule frozen")
print(f"    candidates: {P['candidate_gamma']}")
print(f"    objective : {P['objective']}")
print(f"    sha256    : {P['prereg_sha256'][:32]}...")
