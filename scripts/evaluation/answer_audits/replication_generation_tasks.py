"""GPU round 3 inputs: (a) quality-selection check (3B answers + 2-judge labels for the reported validation keys),
(b) Llama-3.1-8B replication (forced / abstain answers for the reported held-out fresh, FreshCache and SemanticTTL-1/16 keys + qwen7b/mistral7b labels).
Everything already present in the read-only pools / original study files is removed."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import answer_pools as P
R = P.R; W = ((OUTPUT_ROOT / "replay")); I, O = W/"gpu/in3", W/"gpu/out3"; STAGE = sys.argv[1]
SG = OUTPUT_ROOT / "l1only_exactttl/stronger_generator"; QB = OUTPUT_ROOT / "quality_selection"; h = P.h16
def wl(name, rows):
    with open(I/name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), flush=True)
QT = {tuple(t["key"]): t for t in json.load(open(W/"results/quality_selection_gen_tasks_reported.json"))}
AT = {tuple(t["key"][1:]): t for t in json.load(open(W/"results/audit_gen_tasks_reported.json"))}   # r4 indexes tasks by (ctx, q)
AK = json.load(open(W/"results/audit_keys_reported.json"))["keys"]["heldout"]
AUDIT_SAMPLE = json.load(open(OUTPUT_ROOT / "general_audit/sample.json"))["requests"]
if STAGE == "gen":
    A = P.AnswerLookup(extra=[W/"gpu/out/answers_llama3b.jsonl"])
    wl("gen_tasks.jsonl", [t for k, t in QT.items() if A.get(list(k)) is None])
    for mode in ("forced", "abstain"):
        have = {tuple(r["key"]) for r in P.jl(SG/f"answers_llama8b_{mode}.jsonl")}
        need = {}
        for d in AUDIT_SAMPLE:
            for arm in ("fresh", "freshcache", "sttl_k1_16"):
                k = tuple(AK[d["query_id"]][arm][0])
                if k not in have: need[k] = {**AT[k[1:]], "key": list(k)}
        wl(f"gen8_{mode}_tasks.jsonl", list(need.values()))
    sys.exit()
# judge stage
A = P.AnswerLookup(extra=[W/"gpu/out/answers_llama3b.jsonl", ((OUTPUT_ROOT / "replay/gpu/out3/answers_llama3b.jsonl"))]); J2 = P.judge2(extra=[W/"gpu/out", ((OUTPUT_ROOT / "replay/gpu/out3"))])
QS = {json.loads(l)["query_id"]: json.loads(l) for l in open(QB/"q_sample.jsonl")}; QK = json.load(open(W/"results/quality_selection_keys_reported.json"))["records"]
items = {}
for r in QK:
    d = QS[r["query_id"]]
    for k in [r["fresh_key"]] + [r[x] for x in r if x.endswith("_key") and x != "fresh_key"]:
        a = A.get(k); assert a is not None, k
        sig = (h(d["query"]), h(d["gold"]), h(a))
        if sig not in J2["llama8b"] or sig not in J2["qwen7b"]: items.setdefault(sig, {"sig": list(sig), "q": d["query"], "gold": d["gold"], "cand": a})
wl("judge2_items.jsonl", list(items.values()))
ABST8 = "INSUFFICIENT_CONTEXT"
for mode in ("forced", "abstain"):
    ans = {tuple(r["key"]): r["answer"] for r in P.jl(SG/f"answers_llama8b_{mode}.jsonl")}
    for r in P.jl(I.parent/"out3"/f"answers_llama8b_{mode}.jsonl"): ans.setdefault(tuple(r["key"]), r["answer"])
    have = {j: {tuple(x["sig"]) for x in P.jl(SG/f"judge_llama8b_{mode}__{j}.jsonl")} for j in ("qwen7b", "mistral7b")}
    it = {}
    for d in AUDIT_SAMPLE:
        for arm in ("fresh", "freshcache", "sttl_k1_16"):
            a = ans.get(tuple(AK[d["query_id"]][arm][0])); assert a is not None
            if a == ABST8: continue
            sig = (h(d["query"]), h(d["gold"]), h(a))
            if any(sig not in have[j] for j in have): it.setdefault(sig, {"sig": list(sig), "q": d["query"], "gold": d["gold"], "cand": a})
    wl(f"judge8_{mode}_items.jsonl", list(it.values()))
