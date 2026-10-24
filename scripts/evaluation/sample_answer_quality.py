"""Answer quality on a small subset of the bundled sample: generation, two-judge correctness and evidence sufficiency.

Runs the paper's answer pipeline end to end on --limit held-out requests (default 50) of DATA_DIR, with the paper's models and
prompts (scripts/evaluation/generation_judging_stage.py: Llama-3.2-3B-Instruct generator, FORCED prompt; Llama-3.1-8B-Instruct and
Qwen2.5-7B-Instruct as the two judges for correctness and for evidence sufficiency).

Requests: reference-bearing, non-REAL_TIME held-out requests of the primary stream, in a seeded random order (seed 42). Arms:
  fresh       answer generated from the request's own search results at request time
  FreshCache  the reported engine's FreshCache path (freshcache/replay_engine/audit_replay.py): an L1 hit returns the stored answer of
              the request that created the entry; an L2 hit or a miss generates from the evidence FreshCache serves
Labels: ABSTAIN if the answer contains an abstention pattern; otherwise CORRECT iff both judges say CORRECT, else WRONG.
WAI = fresh CORRECT and FreshCache WRONG. Sufficiency of the served evidence (L2 hits and misses): SUFFICIENT / INSUFFICIENT when
both judges agree, else JUDGE_DISAGREEMENT. Page text comes from DATA_DIR/snapshots (re-fetched pages, see
scripts/data/page_refetch.py); the reference answers come from DATA_DIR/requests.jsonl.

Modes (GPU stages run in between; see scripts/run_sample_quickstart.sh):
  tasks [--limit N]  write generation tasks and sufficiency tasks to OUTPUT_DIR/replay/gpu/sample_in/
  items              write the two-judge items for the generated answers
  score              write OUTPUT_DIR/sample_answer_quality/results.json
Usage: python scripts/evaluation/sample_answer_quality.py tasks|items|score [--limit N]"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, collections, json, os, pathlib, random, re, sys
REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve()
OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("tasks", "items", "score")); ap.add_argument("--limit", type=int, default=50); a = ap.parse_args()
GPU_IN, GPU_OUT = OUTPUT_ROOT / "replay/gpu/sample_in", OUTPUT_ROOT / "replay/gpu/sample_out"   # generation_judging_stage.py --indir sample_in --outdir sample_out
O = OUTPUT_ROOT / "sample_answer_quality"; GPU_IN.mkdir(parents=True, exist_ok=True); O.mkdir(parents=True, exist_ok=True)
JUDGES = ("llama8b", "qwen7b")
ABST = ("i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information")
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

if a.mode == "tasks":
    sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
    E = C.load("reported"); p2 = E["p2"]; H = C.streams(E)["heldout"]
    if not any((DATA_ROOT / "snapshots").glob("*/*.json")):
        sys.exit(f"no page text in {DATA_ROOT}/snapshots: run scripts/data/page_refetch.py --build-data DIR and set DATA_DIR=DIR (REFETCH=1 in the quickstart)")
    GOLD = {r["request_id"]: r["reference_answer"] for r in jl(DATA_ROOT / "requests.jsonl") if r.get("reference_answer")}
    m, rows = p2.replay(H, E["rounds"], "FreshCache")
    cand = [(t, r) for t, r in H if r["freshness_class"] != "REAL_TIME" and GOLD.get(r["query_id"])]
    random.Random(42).shuffle(cand); pick = cand[:a.limit]
    man = {r["url_hash"]: r for r in jl(DATA_ROOT / "url_manifest.jsonl")}
    def evblock(ev):   # served-evidence block of the sufficiency judge (as scripts/experiments/admission_controls/verifier_audit_keys.py)
        lines = []
        for uh, tv in ev:
            mt = man.get(uh, {}); b = re.sub(r"\s+", " ", (p2.e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv)) or "")).strip()[:2000]
            blk = f"{len(lines) + 1}. {(mt.get('title') or '').strip()}"
            if mt.get("snippet"): blk += f"\n   {mt['snippet'].strip()}"
            if len(b) >= 200: blk += f"\n   PAGE TEXT: {b}"
            lines.append(blk)
        return "\n".join(lines) or "(no stored evidence)"
    tasks, suff, recs = {}, [], []
    for t, r in pick:
        q, row = r["query_id"], rows[r["query_id"]]
        fctx = p2.ctx_for(tuple((u["url_hash"], t) for u in r["urls"])); fk = ("fresh", p2.h(fctx), p2.h(r["query"])); tasks[fk] = (r["query"], fctx)
        if row["tier"] == "L1":
            sctx = p2.ctx_for(tuple(row["stored_answer_ev"])); sq = row["stored_answer_query"]; ck = ("stored", p2.h(sctx), p2.h(sq)); tasks[ck] = (sq, sctx)
        else:
            pctx = p2.ctx_for(tuple(row["ev"])); ck = ("gen", p2.h(pctx), p2.h(r["query"])); tasks[ck] = (r["query"], pctx)
            suff.append({"query_id": q, "arm": "FreshCache", "query": r["query"], "evidence": evblock(row["ev"])})
        recs.append({"query_id": q, "query": r["query"], "gold": GOLD[q], "freshness_class": r["freshness_class"], "tier": row["tier"], "fresh_key": list(fk), "fc_key": list(ck)})
    wjl(GPU_IN / "gen_tasks.jsonl", [{"key": list(k), "question": qq, "context": cc} for k, (qq, cc) in tasks.items()])
    wjl(GPU_IN / "suffv_tasks.jsonl", suff); wjl(O / "requests.jsonl", recs)
    print(f"{len(recs)} requests (of {len(cand)} eligible held-out requests), {len(tasks)} generation tasks, {len(suff)} sufficiency tasks; "
          f"tiers {dict(collections.Counter(x['tier'] for x in recs))}", flush=True)
    sys.exit(0)

sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import answer_pools as P   # noqa: E402
REQ = jl(O / "requests.jsonl"); ANS = {tuple(r["key"]): r["answer"] for r in jl(GPU_OUT / "answers_llama3b.jsonl")}
J = {j: {tuple(r["sig"]): r["label"] for r in jl(GPU_OUT / f"judge_{j}.jsonl")} for j in JUDGES}
def label(d, key, items=None):
    v = ANS.get(tuple(key))
    if v is None: return None
    if any(p in v.strip().lower() for p in ABST): return "ABSTAIN"
    sig = (P.h16(d["query"]), P.h16(d["gold"]), P.h16(v)); lab = [J[j].get(sig) for j in JUDGES]
    if None in lab:
        if items is not None: items.setdefault(sig, {"sig": list(sig), "q": d["query"], "gold": d["gold"], "cand": v})
        return None
    return "CORRECT" if lab == ["CORRECT", "CORRECT"] else "WRONG"

if a.mode == "items":
    items = {}; missing = sum(1 for d in REQ for k in (d["fresh_key"], d["fc_key"]) if ANS.get(tuple(k)) is None)
    if missing: sys.exit(f"{missing} answers missing: run generation_judging_stage.py gen --indir sample_in --outdir sample_out first")
    for d in REQ:
        for k in (d["fresh_key"], d["fc_key"]): label(d, k, items)
    wjl(GPU_IN / "judge2_items.jsonl", list(items.values())); print(f"{len(items)} two-judge items", flush=True); sys.exit(0)

# score
SV = collections.defaultdict(dict)
for j in JUDGES:
    for r in jl(GPU_OUT / f"suffv_{j}.jsonl"): SV[r["query_id"]][j] = r["label"]
rows = []
for d in REQ:
    fr, fc = label(d, d["fresh_key"]), label(d, d["fc_key"])
    s = SV.get(d["query_id"], {}); su = None if d["tier"] == "L1" else ((s["llama8b"] if s.get("llama8b") == s.get("qwen7b") else "JUDGE_DISAGREEMENT") if len(s) == 2 else None)
    rows.append({**{k: d[k] for k in ("query_id", "freshness_class", "tier")}, "fresh": fr, "FreshCache": fc, "sufficiency": su})
unres = sum(1 for r in rows if r["fresh"] is None or r["FreshCache"] is None)
if unres: sys.exit(f"{unres} requests without both judgments: run generation_judging_stage.py judge2 --judge llama8b|qwen7b --indir sample_in --outdir sample_out")
cnt = lambda arm: dict(collections.Counter(r[arm] for r in rows))
fcn = sum(1 for r in rows if r["fresh"] == "CORRECT"); wai = sum(1 for r in rows if r["fresh"] == "CORRECT" and r["FreshCache"] == "WRONG")
R = {"n": len(rows), "models": {"generator": "meta-llama/Llama-3.2-3B-Instruct", "judges": ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]},
     "tiers": dict(collections.Counter(r["tier"] for r in rows)), "fresh": cnt("fresh"), "FreshCache": cnt("FreshCache"), "fresh_correct": fcn, "WAI": wai,
     "C_WAI_pct": round(100 * wai / fcn, 2) if fcn else None, "sufficiency_served_evidence": dict(collections.Counter(r["sufficiency"] for r in rows if r["tier"] != "L1")),
     "note": "sample-sized numbers from the bundled sample with re-fetched pages; not comparable to the paper's"}
json.dump(R, open(O / "results.json", "w"), indent=1); wjl(O / "request_labels.jsonl", rows); print(json.dumps(R, indent=1), flush=True)
