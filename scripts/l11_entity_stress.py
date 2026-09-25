"""L-11: enumerated stress set for the L1 entity-match gate.

The manuscript reported "blocks 15 of 20" with no artifact behind it. This
rebuilds the test as something a reader can inspect and rerun: 20 explicitly
listed pairs, each labelled with the confusion it encodes, scored by the
SHIPPED gate (experiment._entity_match) rather than a reimplementation.

Every pair is a pair of DIFFERENT questions, so the gate SHOULD block all 20.
Pass-through is a real miss and is reported as one.
"""
import sys, json, pathlib
sys.path.insert(0, "<PROJECT_ROOT>")
from experiment import _entity_match, _entities

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

rows, blocked = [], 0
for kind, q1, q2 in CASES:
    passes = _entity_match(q1, q2)       # True = gate lets the reuse through
    is_blocked = not passes
    blocked += is_blocked
    rows.append({"category": kind, "query_a": q1, "query_b": q2,
                 "entities_a": sorted(_entities(q1)), "entities_b": sorted(_entities(q2)),
                 "blocked": is_blocked,
                 "miss_reason": None if is_blocked else
                   ("no entity span in one or both queries"
                    if not _entities(q1) or not _entities(q2)
                    else "entity sets compared equal")})

by = {}
for r in rows: by.setdefault(r["category"], [0,0]); by[r["category"]][0]+=r["blocked"]; by[r["category"]][1]+=1
out = {"n_cases": 20, "n_blocked": blocked, "n_passed_through": 20-blocked,
       "by_category": {k: {"blocked": v[0], "total": v[1]} for k,v in by.items()},
       "gate": "experiment._entity_match (shipped implementation, not a reimplementation)",
       "cases": rows}
OUT = pathlib.Path(__file__).resolve().parent.parent / "artifacts/l11_entity_stress"
p = OUT / "entity_stress_20.json"   # absolute: os.chdir above must not move the output
p.write_text(json.dumps(out, indent=1))
print("blocked %d/20" % blocked)
for k,v in sorted(by.items()): print("  %-9s %d/%d" % (k, v[0], v[1]))
print("\nmisses:")
for r in rows:
    if not r["blocked"]: print("  [%s] %s || %s  -> %s" % (r["category"], r["query_a"][:44], r["query_b"][:44], r["miss_reason"]))
