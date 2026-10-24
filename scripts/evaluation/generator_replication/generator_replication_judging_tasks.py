"""Build judge items for the generator replication (forced and abstain modes).
Primary (8B replication protocol): one blinded item per distinct (question, reference, candidate); the INSUFFICIENT_CONTEXT
abstention token is not judged. Writes inputs/judge_<mode>_items.jsonl.
Secondary (frozen three-judge protocol, outputs/reference_validation/config/protocol.json): one blinded task per distinct
(request, answer), shuffled with the protocol seed; abstention patterns are resolved before judging and not sent.
Writes inputs/judge3_<mode>_tasks.jsonl. All outputs go under outputs/generator_replication/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, csv, random, hashlib, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
W = ((OUTPUT_ROOT / "generator_replication")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); pass
def h(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
SG = (OUTPUT_ROOT / "l1only_exactttl/stronger_generator"); CR = (OUTPUT_ROOT / "replay/gpu/out3"); O = W/"outputs"
REQ = json.load(open(W/"inputs/requests.json")); ARMS = ["fresh", "freshcache", "sttl_q060_k1_2"]
MB = (OUTPUT_ROOT / "reference_validation"); P = json.load(open(MB/"config/protocol.json")); ABP = P["answer_stage"]["abstain_patterns"]
RQ = {json.loads(l)["query_id"]: json.loads(l) for l in open(MB/"inputs/requests.jsonl")}
TG = {r["query_id"]: (r["target_time"] or "unknown") for r in csv.DictReader(open(MB/"results/reference_validation_results.csv"))}
for mode in ("forced", "abstain"):
    A = {}
    for p in (SG/f"answers_llama8b_{mode}.jsonl", CR/f"answers_llama8b_{mode}.jsonl", O/f"answers_llama8b_{mode}.jsonl"):
        for r in jl(p): A.setdefault(tuple(r["key"]), r["answer"])
    items, t3 = {}, {}
    for d in REQ:
        for arm in ARMS:
            a = A[tuple(d[f"{arm}_key"])]
            if a != "INSUFFICIENT_CONTEXT":
                s = (h(d["query"]), h(d["gold"]), h(a)); items.setdefault(s, {"sig": list(s), "q": d["query"], "gold": d["gold"], "cand": a})
            if not any(p in (a or "").strip().lower() for p in ABP):
                q = d["query_id"]; tid = hashlib.sha256(f"{q}\x00{a}".encode()).hexdigest()[:16]
                t3.setdefault(tid, {"task_id": tid, "query_id": q, "question": RQ[q]["question"], "target": TG[q], "reference": RQ[q]["reference_answer"], "candidate": a})
    L = list(t3.values()); random.Random(P["seed"]).shuffle(L)
    for name, rows in ((f"judge_{mode}_items.jsonl", list(items.values())), (f"judge3_{mode}_tasks.jsonl", L)):
        with open(W/"inputs"/name, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))
