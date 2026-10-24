"""The five policies of the common 1,500-request sample re-scored with Qwen2.5-14B and an 8,000-token context built
from each policy's served snapshot pages (pages under 200 characters dropped, rank order, truncation by Qwen tokens; frozen FORCED prompt as
user turn, greedy, 80 tokens, first-sentence clip). Frozen two-judge rule; common-sample estimator. Modes: build | gen --gpu G --shard i/n | items | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, collections, hashlib, json, math, os, pickle, re, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "natural_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); MODE = sys.argv[1]
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]; POL = ("B", "Bm", "Be", "D", "Dp")
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
if MODE == "gen":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gpu", required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu; si, sn = map(int, a.shard.split("/"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FORCED"}
    mid = "Qwen/Qwen2.5-14B-Instruct"; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    out = O / f"gpu/out/larger_context_answers14_shard{si}_of{sn}.jsonl"; done = {tuple(r["key"]) for f in (O / "gpu/out").glob("larger_context_answers14_*.jsonl") for r in jl(f)}
    todo = sorted([t for i, t in enumerate(jl(O / "gpu/in/larger_context_tasks.jsonl")) if i % sn == si and tuple(t["key"]) not in done], key=lambda t: len(t["context"]))
    def clip(t):
        t = (t or "").strip()
        if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
        for sep in (".", "\n"):
            if sep in t: return t[:t.index(sep) + 1].strip()
        return t
    BS = int(os.environ.get("GEN_BS", "2")); fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), BS):
        ch = todo[s:s + BS]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": K["FORCED"].format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                  return_tensors="pt", padding=True, truncation=True, max_length=9000).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            fh.write(json.dumps({"key": c["key"], "raw": t, "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n")
        fh.flush()
    sys.exit(0)
sys.path.insert(0, str(ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("natural_workload"); import composition_study_helpers as X   # noqa: E402
E = X.engine(); p2 = E["p2"]; REC = {r["query_id"]: r for r in E["recs"]}; GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid):
    qt = REC[qid]["query"]; g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
C = pickle.load(open(CS_DIR / "work/common.pkl", "rb")); T = pickle.load(open(CS_DIR / "work/traces.pkl", "rb")); S = C["S"]; U, Nref, n = len(C["U"]), C["N_ref"], len(C["S"])
if MODE == "build":
    import answer_grading as e2e
    from transformers import AutoTokenizer
    QT = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    def ctx(ev):
        parts, total = [], 0
        for uh, tv in ev:
            t = re.sub(r"\s+", " ", e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv)) or "").strip()
            if len(t) < 200: continue
            room = 8000 - total
            if room <= 0: break
            ids = QT(t, add_special_tokens=False)["input_ids"][:room]; parts.append(QT.decode(ids)); total += len(ids)
        return "\n\n".join(parts)
    A, TK = {}, {}
    for q in S:
        for p in POL:
            r = T[p][q]; cx = ctx([tuple(x) for x in r["ev"]]); gq = r["gen_query"]; k = ["ctx8k", h16(cx), h16(gq)]
            A[(q, p)] = k; TK.setdefault(tuple(k), {"key": k, "question": gq, "context": cx})
    pickle.dump(A, open(O / "work/larger_context_keys.pkl", "wb")); wjl(O / "gpu/in/larger_context_tasks.jsonl", list(TK.values())); print("tasks", len(TK)); sys.exit(0)
A = pickle.load(open(O / "work/larger_context_keys.pkl", "rb")); ANS = {}
for f in sorted((O / "gpu/out").glob("larger_context_answers14_*.jsonl")):
    for r in jl(f): ANS[tuple(r["key"])] = r["answer"]
import live_stream_labeller as CL   # noqa: E402
lab = CL.Lab()
def L(q, p): return lab.label_answer(REC[q]["query"], goldof(q), ANS.get(tuple(A[(q, p)])))
if MODE == "items":
    it, miss = {}, 0
    for (q, p) in A:
        l, a, s = L(q, p); miss += a is None
        if a is not None and l is None: it.setdefault(s, {"sig": list(s), "q": REC[q]["query"], "gold": goldof(q), "cand": a})
    X.wjl(O / "gpu/in/judge2_items.jsonl", list(it.values())); print("judge2", len(it), "missing", miss); sys.exit(0)
if MODE == "score":
    Lb = {(q, p): L(q, p)[0] for (q, p) in A}; R = {"n": n, "U": U, "N_ref": Nref, "unresolved": sum(v is None for v in Lb.values()), "accuracy_pct": {p: round(100 * sum(Lb[(q, p)] == "CORRECT" for q in S) / n, 2) for p in POL}, "comparisons": {}}
    for a, b in (("B", "D"), ("B", "Dp"), ("B", "Bm"), ("Bm", "Dp"), ("D", "Dp")):
        d = [int(Lb[(q, b)] == "CORRECT") - int(Lb[(q, a)] == "CORRECT") for q in S]; mu = sum(d) / n; var = sum((x - mu) ** 2 for x in d) / (n - 1); se = math.sqrt(var / n * (1 - n / U)); f = 100 * U / Nref
        R["comparisons"][f"{a}->{b}"] = {"losses": sum(x < 0 for x in d), "gains": sum(x > 0 for x in d), "stream_pp": round(f * mu, 3), "ci_pp": [round(f * (mu - 1.96 * se), 3), round(f * (mu + 1.96 * se), 3)]}
    json.dump(R, open(O / "outputs/common_sample_larger_context.json", "w"), indent=1); print(json.dumps(R, indent=1))
