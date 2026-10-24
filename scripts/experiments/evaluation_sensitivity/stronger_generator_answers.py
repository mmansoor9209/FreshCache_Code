"""Stronger-generator rerun with Qwen2.5-14B-Instruct (downloaded into models_cache).
Identical contexts and generation questions as the 3B runs; only the generator changes (frozen FORCED instruction, given as the user turn of the
chat template, greedy, 80 new tokens, clipped at the first sentence end as in the 3B pipeline). Sets: (a) the held-out audit for the
fresh pipeline and policies A, B, D (outputs/composition/work/composition_ablation_replay.pkl); (b) the live candidates for the own-search path A_matched and
FreshCache's service path C (contexts of the unseen-query live comparison). Modes: tasks | gen (--shard i/n --gpu g) | merge."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import ast, json, os, pickle, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "composition_checks")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); MODE = sys.argv[1]
T2 = (OUTPUT_ROOT / "unseen_queries")
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
if MODE == "tasks":
    sys.path.insert(0, str(ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(ROOT / "freshcache/composition")); import output_workspace; output_workspace.use("composition_checks"); import composition_study_helpers as X
    E = X.engine(); p2 = E["p2"]
    SAMPLE = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
    ST = pickle.load(open((OUTPUT_ROOT / "composition/work/composition_ablation_replay.pkl"), "rb"))["policies"]; T = {}
    for d in SAMPLE:
        ctx = p2.ctx_for(tuple(tuple(x) for x in d["fresh_versions"])) if d["fresh_versions"] else ""
        T[("audit", d["query_id"], "fresh")] = {"set": "audit", "query_id": d["query_id"], "arm": "fresh", "question": d["query"], "context": ctx}
        for p in ("A", "B", "D"):
            a = ST[p]["audit"][d["query_id"]]; T[("audit", d["query_id"], p)] = {"set": "audit", "query_id": d["query_id"], "arm": p, "question": a["gen_question"], "context": a["context"]}
    G = {tuple(t["key"]): t for t in jl(T2 / "gpu/in/gen_tasks.jsonl")}
    for c in jl(T2 / "work/cases_built.jsonl"):
        for path in ("A_matched", "C_freshcache"):
            t = G[tuple(c["paths"][path]["key"])]; T[("live", c["case_id"], path)] = {"set": "live", "query_id": c["case_id"], "arm": path, "question": t["question"], "context": t["context"]}
    rows = list(T.values())
    for i, r in enumerate(rows): r["tid"] = i
    with open(O / "gpu/in/gen14b_tasks.jsonl", "w", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("tasks", len(rows)); sys.exit(0)
if MODE == "gen":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gpu", required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu; si, sn = map(int, a.shard.split("/"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FORCED"}
    mid = "Qwen/Qwen2.5-14B-Instruct"; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    out = O / f"gpu/out/answers14b_shard{si}.jsonl"; done = {r["tid"] for r in jl(out)}
    todo = [t for t in jl(O / "gpu/in/gen14b_tasks.jsonl") if t["tid"] % sn == si and t["tid"] not in done]
    def clip(t):
        t = (t or "").strip()
        if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
        for sep in (".", "\n"):
            if sep in t: return t[:t.index(sep) + 1].strip()
        return t
    fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), 8):
        ch = todo[s:s + 8]
        prompts = [tok.apply_chat_template([{"role": "user", "content": K["FORCED"].format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True) for c in ch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            fh.write(json.dumps({"tid": c["tid"], "set": c["set"], "query_id": c["query_id"], "arm": c["arm"], "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n")
        fh.flush()
    print("done", si, len(todo)); sys.exit(0)
