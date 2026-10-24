"""Judge items for the full-state live policies: one item per distinct (incoming query, reference, stream answer), abstentions excluded."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import hashlib, json, os, pathlib, pickle; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "composition"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
SIM = pickle.load(open(O / "work/fullstate_sim.pkl", "rb"))
FZ = {c["case_id"]: c for c in json.load(open((OUTPUT_ROOT / "unseen_queries/sample_frozen.json")))["cases"]}
ANS = {tuple(json.loads(l)["key"]): json.loads(l)["answer"] for l in open(O / "gpu/out/answers_llama3b.jsonl")}
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
items = {}
for rows in SIM.values():
    for r in rows:
        if r["path"] != "stream": continue
        z = FZ[r["case_id"]]; a = ANS[tuple(r["key"])]
        if any(x in a.strip().lower() for x in HELD) or not z["reference_answer"]: continue
        sig = [h16(z["incoming_query"]), h16(z["reference_answer"]), h16(a)]; items.setdefault(tuple(sig), {"sig": sig, "q": z["incoming_query"], "gold": z["reference_answer"], "cand": a})
open(O / "gpu/in/judge2_items.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items.values())); print("judge2 items", len(items))
