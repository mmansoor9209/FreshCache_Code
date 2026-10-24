"""Live confirmation block, design step (no Web calls): build and fix the confirmation block before any search.
Block N: FreshQA release 2026-04-21 (public Google Sheet, outputs/live_query_block/data/freshqa_2026-04-21.csv) questions that are
NOT in FreshCache-Bench (normalized text match against data/queries.jsonl), false-premise items excluded and classes assigned with
the benchmark's own freshqa_map rule.
Block U: benchmark base questions from never-used clusters (outputs/unseen_queries/untouched_queries.jsonl) that were NOT in the
live experiment, are not REAL_TIME, and have a reference answer; a seeded random sample fills the block to 1,000 queries.
Arrival: seeded random order and seeded uniform times over 7 days (no chronology exists for either source).
Writes outputs/live_query_block/work/block_frozen.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import csv, hashlib, json, random, re, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "live_query_block")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); T2 = (OUTPUT_ROOT / "unseen_queries")
SEED, N_TOTAL, H = 20261007, 1000, 7 * 86400
# class rule, copied verbatim from scripts/data/query_construction.py
FRESHQA_MEDIUM_KW = frozenset(["android", "macos", "ios", "operating system", "software version", "h-index", "citations", "google scholar", "exoplanets", "papers accepted",
    "neurips", "emnlp", "acl ", "ijcai", "conference", "vaccination", "covid", "legalize", "cannabis", "same-sex marriage", "countries have recognized",
    "tomatometer", "rotten tomatoes", "netflix", "base price", "tesla model", "homepod"])
def freshqa_map(fact_type, question):
    if fact_type == "never-changing": return "TIMELESS"
    if fact_type == "slow-changing": return "SLOW"
    if fact_type == "fast-changing": return "MEDIUM" if any(kw in question.lower() for kw in FRESHQA_MEDIUM_KW) else "FAST"
    return "MEDIUM"
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
bench = {norm(json.loads(l).get("query") or json.loads(l).get("question")) for l in open((DATA_ROOT / "queries.jsonl"), encoding="utf-8")}
rows = list(csv.reader(open(O / "data/freshqa_2026-04-21.csv", encoding="utf-8"))); h = [i for i, r in enumerate(rows) if "question" in [c.strip().lower() for c in r]][0]
HD = [c.strip().lower() for c in rows[h]]; FQ = [dict(zip(HD, r)) for r in rows[h + 1:] if any(r)]
N = []
for d in FQ:
    if norm(d["question"]) in bench or d.get("false_premise", "").strip().upper() == "TRUE": continue
    ans = [d[k].strip() for k in HD if k.startswith("answer_") and d.get(k, "").strip()]
    if not ans: continue
    N.append({"qid": f"fq26_{d['id']}", "query": d["question"].strip(), "freshness_class": freshqa_map(d.get("fact_type", "").strip(), d["question"]), "block": "N_freshqa_2026",
              "reference": ans[0], "aliases": ans[1:], "fact_type": d.get("fact_type"), "effective_year": d.get("effective_year")})
live = {c["query_id"] for c in json.load(open(T2 / "sample_frozen.json"))["cases"]}
gold = json.load(open((OUTPUT_ROOT / "gold_answers/reference_answers.json"), encoding="utf-8"))
U = []
for l in open(T2 / "untouched_queries.jsonl", encoding="utf-8"):
    x = json.loads(l)
    if x["query_id"] in live or x["freshness_class"] == "REAL_TIME": continue
    g = gold.get(x["query_id"]) or gold.get(x["query"]) or gold.get(x["query"].strip().lower())
    if not (g or {}).get("answer"): continue
    U.append({"qid": x["query_id"], "query": x["query"], "freshness_class": x["freshness_class"], "block": "U_untouched_benchmark", "reference": g["answer"], "aliases": g.get("aliases", []), "source": x.get("source")})
rng = random.Random(SEED); Us = rng.sample(sorted(U, key=lambda r: r["qid"]), N_TOTAL - len(N))
B = N + Us; rng.shuffle(B); ts = sorted(rng.uniform(0, H) for _ in B)
for r, t in zip(B, ts): r["t"] = round(t, 3)
F = {"seed": SEED, "n": len(B), "n_block_N": len(N), "n_block_U": len(Us), "pool_U": len(U), "horizon_s": H, "requests": B}
F["sha256"] = hashlib.sha256(json.dumps(B, sort_keys=True).encode()).hexdigest()
json.dump(F, open(O / "work/block_frozen.json", "w"), indent=1, ensure_ascii=False)
import collections
print({k: F[k] for k in ("n", "n_block_N", "n_block_U", "pool_U")}, "sha", F["sha256"][:16], dict(collections.Counter(r["freshness_class"] for r in B)))
