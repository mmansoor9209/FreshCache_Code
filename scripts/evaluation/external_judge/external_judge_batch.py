"""Builds a blinded ChatGPT judging batch (150 items) from the common 1,500-request sample (work/common.pkl, read-only).
Strata (seed 20261010): 60 answers from requests where D and B disagree under the two-judge rule (both answers of 30 such requests;
the decisive items for the D vs. B comparison) + 90 random other (request, answer) pairs, 45 currently CORRECT and 45 WRONG.
Rule-based abstentions are excluded (no judge involved). Each (question, answer) pair appears once. The batch files contain only
question, reference, candidate answer and a random item ID; policy names and current labels are in answer_key_hidden/ only."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, random, sys, pathlib, secrets; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
V = ((OUTPUT_ROOT / "external_judge")); COMMON_SAMPLE_DIR = ((OUTPUT_ROOT / "common_sample"))
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); sys.argv = [sys.argv[0], "score"]
import pickle
from common_sample_construction import goldof, REC, Lab   # read-only import (score mode builds nothing)
C = pickle.load(open(COMMON_SAMPLE_DIR / "work/common.pkl", "rb")); rows, S = C["rows"], C["S"]; lab = Lab()
L, A = {}, {}
for (q, p), a in rows.items():
    l, ans, _ = lab.label(REC[q]["query"], goldof(q), a["key"]); L[(q, p)] = l; A[(q, p)] = ans
rng = random.Random(20261010)
disc = sorted(q for q in S if (L[(q, "D")] == "CORRECT") != (L[(q, "B")] == "CORRECT") and "ABSTAIN" not in (L[(q, "D")], L[(q, "B")]))
items, seen = [], set()
def add(q, p, stratum):
    k = (q, A[(q, p)].strip())
    if k in seen: return False
    seen.add(k); items.append({"q": q, "policy": p, "label_two_judge": L[(q, p)], "stratum": stratum}); return True
for q in rng.sample(disc, min(30, len(disc))):
    add(q, "D", "D_vs_B_discordant"); add(q, "B", "D_vs_B_discordant")
pool = [(q, p) for (q, p) in rows if q in set(S) and q not in set(disc) and L[(q, p)] in ("CORRECT", "WRONG")]
rng.shuffle(pool); need = {"CORRECT": 45, "WRONG": 45}
for q, p in pool:
    if need[L[(q, p)]] and add(q, p, "random_" + L[(q, p)].lower()): need[L[(q, p)]] -= 1
    if not any(need.values()): break
rng.shuffle(items)
for it in items: it["item_id"] = "I" + secrets.token_hex(3).upper()
PROMPT = """You are grading short answers against reference answers. For each item below, decide whether the CANDIDATE conveys the same
factual answer as the REFERENCE for the QUESTION. Minor wording, extra context, or added detail do not matter. Judge only against the
reference; do not use outside knowledge to decide which is true.

Reply with one line per item and nothing else, in exactly this format:
<item_id>,<CORRECT or INCORRECT>

"""
B = 25
for b in range(0, len(items), B):
    chunk = items[b:b + B]; txt = PROMPT
    for it in chunk:
        q = it["q"]; txt += f"ITEM {it['item_id']}\nQUESTION: {REC[q]['query']}\nREFERENCE: {goldof(q)}\nCANDIDATE: {A[(q, it['policy'])]}\n\n"
    (V / f"chatgpt_batch/batch_{b // B + 1:02d}.txt").write_text(txt.rstrip() + "\n", encoding="utf-8")
json.dump([{**it, "question": REC[it["q"]]["query"], "reference": goldof(it["q"]), "answer": A[(it["q"], it["policy"])]} for it in items],
          open(V / "answer_key_hidden/answer_key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(items), "items;", collections.Counter(i["stratum"] for i in items), "; discordant requests available:", len(disc))
