"""L-11: enumerated stress set for the L1 admission gate.

The manuscript reported "blocks 15 of 20" with no artifact behind it. This
rebuilds the test as something a reader can inspect and rerun: 20 explicitly
listed pairs, each labelled with the confusion it encodes, scored by the
SHIPPED gate (experiment._entity_match) rather than a reimplementation.

Every pair is a pair of DIFFERENT questions, so the gate SHOULD block all 20.
Pass-through is a real miss and is reported as one.
"""
import os, sys, json, pathlib
sys.path.insert(0, "<PROJECT_ROOT>")
os.chdir("<PROJECT_ROOT>")
import numpy as np
from experiment import _entity_match, _entities, semantic_equivalent

CASES = [
 # --- modifier confusion: same head entity, different modifier -------------
 ("modifier", "What is the battery life of the iPhone 16?",
              "What is the battery life of the iPhone 16 Pro?"),
 ("modifier", "When does the Galaxy S24 go on sale?",
              "When does the Galaxy S24 Ultra go on sale?"),
 ("modifier", "How fast is the RTX 4080?", "How fast is the RTX 4080 Super?"),
 ("modifier", "What is the price of a Tesla Model 3?",
              "What is the price of a Tesla Model 3 Performance?"),
 ("modifier", "Who won the 2024 Australian Open?",
              "Who won the 2024 Australian Open doubles?"),
 ("modifier", "What is the population of Kansas City?",
              "What is the population of Kansas City, Kansas?"),
 ("modifier", "When is the next SpaceX Starship launch?",
              "When is the next SpaceX Starship orbital launch?"),
 # --- entity confusion: different named entities altogether ----------------
 ("entity",   "What is the current CEO of Microsoft?",
              "What is the current CEO of Meta?"),
 ("entity",   "What is the inflation rate in Germany?",
              "What is the inflation rate in France?"),
 ("entity",   "Who is the head coach of the Lakers?",
              "Who is the head coach of the Celtics?"),
 ("entity",   "What is the stock price of Nvidia?",
              "What is the stock price of Intel?"),
 ("entity",   "When did Argentina last win the World Cup?",
              "When did Brazil last win the World Cup?"),
 ("entity",   "What is the capital of Slovenia?",
              "What is the capital of Slovakia?"),
 ("entity",   "How many employees does Amazon have?",
              "How many employees does Apple have?"),
 # --- no-span phrasings: the documented pass-through gap -------------------
 ("no_span",  "how much is it going up by this month",
              "how much did it go up last month"),
 ("no_span",  "what did they announce yesterday",
              "what did they announce today"),
 ("no_span",  "is it still down", "is it back up"),
 ("no_span",  "who won last night", "who won this afternoon"),
 ("no_span",  "what is the latest number", "what was the previous number"),
 ("no_span",  "has it been updated yet", "was it updated earlier"),
]
assert len(CASES) == 20

# ---- BGE-M3 cosine for each pair, so the FULL gate can be scored ---------
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
sents = [q for _, a, b in CASES for q in (a, b)]
E = model.encode(sents, batch_size=8, normalize_embeddings=True,
                 show_progress_bar=False)
sims = [float(np.dot(E[2*i], E[2*i+1])) for i in range(len(CASES))]

rows = []
n_ent = n_eq = n_full = 0
for (kind, q1, q2), sim in zip(CASES, sims):
    ent_pass = _entity_match(q1, q2)          # entity guard alone
    eq_pass  = semantic_equivalent(q1, q2, sim)  # similarity + Jaccard + type
    admitted = ent_pass and eq_pass           # the real L1 admission rule
    n_ent  += (not ent_pass)
    n_eq   += (not eq_pass)
    n_full += (not admitted)
    why = []
    if not ent_pass: why.append("entity guard")
    if sim < 0.80:   why.append("cosine < 0.80")
    elif not eq_pass: why.append("Jaccard or answer-type")
    rows.append({"category": kind, "query_a": q1, "query_b": q2,
                 "bge_m3_cosine": round(sim, 4),
                 "entities_a": sorted(_entities(q1)), "entities_b": sorted(_entities(q2)),
                 "entity_guard_blocks": not ent_pass,
                 "semantic_equivalent_blocks": not eq_pass,
                 "full_gate_blocks": not admitted,
                 "blocked_by": why or None})

by = {}
for r in rows:
    d = by.setdefault(r["category"], {"ent": 0, "full": 0, "n": 0})
    d["ent"] += r["entity_guard_blocks"]; d["full"] += r["full_gate_blocks"]; d["n"] += 1

out = {"n_cases": len(CASES),
       "entity_guard_alone_blocks": n_ent,
       "semantic_equivalent_alone_blocks": n_eq,
       "full_L1_gate_blocks": n_full,
       "by_category": by,
       "gate": ("full L1 admission = experiment._entity_match AND "
                "experiment.semantic_equivalent(q1,q2,cosine); cosine is BGE-M3 "
                "dense CLS, L2-normalised, the same encoder the replay uses"),
       "cases": rows}
OUT = pathlib.Path(__file__).resolve().parent.parent / "artifacts/l11_entity_stress"
p = OUT / "entity_stress_20.json"   # absolute: os.chdir above must not move the output
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1))

print("entity guard alone blocks   %d/%d" % (n_ent, len(CASES)))
print("semantic_equivalent blocks  %d/%d" % (n_eq, len(CASES)))
print("FULL L1 gate blocks         %d/%d" % (n_full, len(CASES)))
for k, d in sorted(by.items()):
    print("  %-9s entity %d/%d   full %d/%d" % (k, d["ent"], d["n"], d["full"], d["n"]))
print("\nfull-gate misses:")
for r in rows:
    if not r["full_gate_blocks"]:
        print("  [%s] cos=%.3f  %s || %s" % (r["category"], r["bge_m3_cosine"],
                                             r["query_a"][:40], r["query_b"][:40]))
