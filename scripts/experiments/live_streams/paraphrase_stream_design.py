"""C block (design only; no Web call): pool = never-used benchmark base questions (outputs/unseen_queries/
untouched_queries.jsonl) not in the live experiment, not previously sent to a search API in any earlier experiment, with a reference,
not REAL_TIME. Draw 470 for C (seed 20261011); 20 of the remaining bases form the technical pilot (seed 20261012) and are never confirmation data.
Each C base gets up to 4 paraphrases with the benchmark's own procedure (paraphrase_generation.py: humarin/chatgpt_paraphraser_on_T5_base,
'paraphrase: ' prefix, sampling temperature 1.5, top_k 120, top_p 0.98, no_repeat_ngram_size 2, repetition_penalty 2.5, 4 sequences,
exact-copy and duplicate removal), seeded (torch seed 20261011). Arrival: the benchmark's schedule builder (outputs/robustness_schedules/schedules.build_stream,
zipf_uniform, 7 virtual days, seed 20261011). Writes work/c_block.json (sha256 recorded) and work/pilot_queries.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
import hashlib, json, random, sys, pathlib
O = ((OUTPUT_ROOT / "common_sample")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); T2 = (OUTPUT_ROOT / "unseen_queries")
un = [json.loads(l) for l in open(T2 / "untouched_queries.jsonl", encoding="utf-8")]
live = {c["query_id"] for c in json.load(open(T2 / "sample_frozen.json"))["cases"]}
searched = {r["qid"] for r in json.load(open(((OUTPUT_ROOT / "live_query_block/work/block_frozen.json"))))["requests"]}   # queries already sent to a search API
gold = json.load(open((OUTPUT_ROOT / "gold_answers/reference_answers.json"), encoding="utf-8"))
g = lambda x: gold.get(x["query_id"]) or gold.get(x["query"]) or gold.get(x["query"].strip().lower()) or {}
pool = sorted([x for x in un if x["query_id"] not in live and x["query_id"] not in searched and x["freshness_class"] != "REAL_TIME" and g(x).get("answer")], key=lambda x: x["query_id"])
assert DS.check("paraphrase_stream_pool", len(pool)), len(pool)
draw = sorted(random.Random(20261011).sample(pool, 470), key=lambda x: x["query_id"]); dq = {x["query_id"] for x in draw}
rest = [x for x in pool if x["query_id"] not in dq]; pilot = random.Random(20261012).sample(rest, 20)
json.dump([{"query_id": x["query_id"], "query": x["query"]} for x in pilot], open(O / "work/pilot_queries.json", "w"), indent=1, ensure_ascii=False)
import torch   # noqa: E402
from transformers import T5ForConditionalGeneration, T5Tokenizer   # noqa: E402
mn = "humarin/chatgpt_paraphraser_on_T5_base"; tok = T5Tokenizer.from_pretrained(mn); mod = T5ForConditionalGeneration.from_pretrained(mn).to("cuda").eval(); torch.manual_seed(20261011)
def para(text, n=4):
    ids = tok(f"paraphrase: {text}", return_tensors="pt", padding="longest", max_length=128, truncation=True).input_ids.to("cuda")
    with torch.no_grad():
        out = mod.generate(ids, max_length=128, num_return_sequences=n, do_sample=True, temperature=1.5, top_k=120, top_p=0.98, no_repeat_ngram_size=2, repetition_penalty=2.5)
    res = []
    for o in out:
        d = tok.decode(o, skip_special_tokens=True).strip()
        if d and d.lower() != text.lower() and d not in res: res.append(d)
    return res[:n]
recs = []
for x in draw:
    ref = g(x); base = {"query_id": x["query_id"], "cluster_id": x["query_id"], "query": x["query"], "freshness_class": x["freshness_class"], "language": x.get("language", "en"),
                        "is_paraphrase": False, "reference": ref["answer"], "aliases": ref.get("aliases", []), "source": x.get("source")}
    recs.append(base)
    if base["language"] == "en":
        for i, v in enumerate(para(x["query"])): recs.append(dict(base, query_id=f"p_{x['query_id']}_{i}", query=v, is_paraphrase=True))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "freshcache/simulation")); import request_schedules as sc   # noqa: E402
stream = sc.build_stream(recs, "zipf_uniform", 20261011)
B = [dict(r, t_virtual=round(t, 3), order=i) for i, (t, r) in enumerate(stream)]
F = {"seed": 20261011, "n_bases": len(draw), "n_requests": len(B), "pool": len(pool), "pilot_bases": len(pilot), "schedule": "zipf_uniform (benchmark builder), 7 virtual days", "requests": B}
F["sha256"] = hashlib.sha256(json.dumps(B, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
json.dump(F, open(O / "work/c_block.json", "w"), indent=1, ensure_ascii=False); print({k: F[k] for k in ("n_bases", "n_requests", "pool")}, F["sha256"][:16])
