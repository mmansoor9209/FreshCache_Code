"""Generator-input audit of the live depth settings on 150 paired cases (recorded pages, frozen page() from the read-only prefix of unseen_query_evidence_paths.py).
Arms: own search at matched depth (k = cached-list length), depth 3, depth 5, top-10, reused list R.
Budgets: 2,800 characters (Llama-3.2-3B, frozen; input limit 1,024 tokens) and 8,000 characters (Qwen2.5-14B, as in the larger-context generation runs).
Per arm: input tokens (prompt included, generator tokenizer), contributing URLs, context hash, characters removed by page and total truncation,
whether any page beyond the matched depth reaches the generator, and whether the context is byte-identical to the matched-depth context."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, collections, hashlib, json, pathlib, statistics, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); T2 = (OUTPUT_ROOT / "unseen_queries")
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
src = open((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_evidence_paths.py")).read(); cut = src.index("CASES, FLOW, GEN, SUF = [], collections.Counter(), {}, {}")
g = {"__name__": "t2b", "__file__": str((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_evidence_paths.py"))}; exec(compile(src[:cut], "unseen_query_evidence_paths (read-only prefix)", "exec"), g); page, SR = g["page"], g["SR"]
from transformers import AutoTokenizer   # noqa: E402
LT = AutoTokenizer.from_pretrained(str(pathlib.Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"))
QT = AutoTokenizer.from_pretrained(str(next(((REPO_ROOT / "models_cache/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots")).iterdir())))
FORCED = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FORCED"}["FORCED"]
def assemble(texts, maxc, minc=200):
    parts, total, used, cut_page = [], 0, [], 0
    for i, t in enumerate(texts):
        if len(t) < minc: continue
        room = maxc - total
        if room <= 0: break
        parts.append(t[:room]); total += min(len(t), room); used.append(i); cut_page += max(0, len(t) - room)
    removed = sum(len(t) for t in texts if len(t) >= minc) - total
    return "\n\n".join(parts), used, removed
CASES = [c for c in jl(T2 / "work/cases_built.jsonl") if c["both_evaluable"] and c["reference_available"]]
A = collections.defaultdict(lambda: collections.defaultdict(list)); R = {"n_cases": len(CASES)}
for c in CASES:
    s = json.load(open(SR / f"{c['case_id']}.json", encoding="utf-8")); org = [o["link"] for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")][:10]
    arms = {"matched": org[:c["k"]], "d3": org[:3], "d5": org[:5], "top10": org, "R_reused": c["cached_urls"]}
    for maxc, tokz, nm in ((2800, "llama", "chars2800_llama3b"), (8000, "qwen", "chars8000_qwen14b")):
        base = None
        for arm, links in arms.items():
            texts = [page(u)["text"] for u in links]; cx, used, removed = assemble(texts, maxc)
            pr = FORCED.format(question=c["query"], context=cx)
            nt = len(LT(pr)["input_ids"]) if tokz == "llama" else len(QT(QT.apply_chat_template([{"role": "user", "content": pr}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
            h = hashlib.sha256(cx.encode()).hexdigest()[:16]
            if arm == "matched": base = (h, set(links))
            A[nm][arm].append({"tokens": nt, "pages": len(used), "removed": removed, "hash": h, "same_as_matched": h == base[0],
                               "beyond_matched_reaches": any(links[i] not in base[1] for i in used), "model_truncated": tokz == "llama" and nt > 1024, "ctx_chars": len(cx)})
for nm, arms in A.items():
    R[nm] = {}
    for arm, v in arms.items():
        R[nm][arm] = {"mean_input_tokens": round(statistics.mean(x["tokens"] for x in v), 1), "max_input_tokens": max(x["tokens"] for x in v),
                      "mean_contributing_pages": round(statistics.mean(x["pages"] for x in v), 2), "mean_chars_removed_by_truncation": round(statistics.mean(x["removed"] for x in v)),
                      "context_identical_to_matched": sum(x["same_as_matched"] for x in v), "page_beyond_matched_reaches_generator": sum(x["beyond_matched_reaches"] for x in v),
                      "model_side_truncated": sum(x["model_truncated"] for x in v), "budget_full": sum(x["ctx_chars"] >= (2800 if "2800" in nm else 8000) - 10 for x in v)}
json.dump(R, open(O / "outputs/generator_input_audit.json", "w"), indent=1); print(json.dumps(R, indent=1))
