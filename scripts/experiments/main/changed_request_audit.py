"""Quality price of URL-list reuse, estimated on a random sample of the held-out requests where D (CategoryAware + L2/L3) differs
from B (CategoryAware + L3) in route, evidence, or generation question (outputs/composition/work/composition_ablation_replay.pkl). Frame excludes
REAL_TIME (never differs) and requests without a gold reference. Sample: simple random, seed 20261005, frozen to work/changed_sample.json
before any answer is generated. Both policies' answers are generated from their own evidence (Llama-3.2-3B FORCED), judged with the frozen
two-judge rule, and their evidence judged with frozen SUFFV. Modes: freeze | items | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import collections, hashlib, json, math, pickle, random, sys, pathlib
O = ((OUTPUT_ROOT / "composition_checks")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("composition_checks"); import composition_study_helpers as X   # noqa: E402
MODE = sys.argv[1]; E = X.engine(); p2 = E["p2"]
GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid, qt):
    g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
REC = {r["query_id"]: r for r in E["recs"]}
if MODE == "freeze":
    ST = pickle.load(open((OUTPUT_ROOT / "composition/work/composition_ablation_replay.pkl"), "rb"))["policies"]
    ch = [q for q in ST["B"]["sig"] if ST["B"]["sig"][q] != ST["D"]["sig"][q]]
    frame = [q for q in ch if REC[q]["freshness_class"] != "REAL_TIME" and goldof(q, REC[q]["query"])]
    samp = sorted(random.Random(20261005).sample(sorted(frame), 400))
    rows = {q: {p: {"tier": ST[p]["sig"][q][0], "ev": [list(x) for x in ST[p]["sig"][q][1]], "gen_query": ST[p]["sig"][q][2]} for p in ("B", "D")} for q in samp}
    json.dump({"seed": 20261005, "changed_requests": len(ch), "frame": len(frame), "excluded_no_gold_or_realtime": len(ch) - len(frame), "n": len(samp), "stream": len(ST["B"]["sig"]), "requests": samp, "rows": rows},
              open(O / "work/changed_sample.json", "w"))
    print(len(ch), len(frame)); sys.exit(0)
F = json.load(open(O / "work/changed_sample.json")); hl = X.Held()
A = {}
for q in F["requests"]:
    for p in ("B", "D"):
        r = F["rows"][q][p]; k, ctx, gq = X.key_for({"ev": r["ev"], "gen_query": r["gen_query"]}, REC[q]["query"]); ev = [tuple(x) for x in r["ev"]]; blk = X.evblock(ev) if ev else None
        A[(q, p)] = {"key": k, "ctx": ctx, "gq": gq, "blk": blk, "sha": hashlib.sha256(blk.encode()).hexdigest() if blk else None, "tier": r["tier"]}
if MODE == "items":
    gen = {tuple(a["key"]): {"key": a["key"], "question": a["gq"], "context": a["ctx"]} for a in A.values() if hl.answer(a["key"]) is None}
    X.wjl(O / "gpu/in/gen_tasks.jsonl", list(gen.values())); print("gen", len(gen))
    if not gen:
        items, miss = X.judge_items(hl, [(REC[q]["query"], goldof(q, REC[q]["query"]), A[(q, p)]["key"]) for q, p in A]); assert not miss
        X.wjl(O / "gpu/in/judge2_items.jsonl", items)
        X.wjl(O / "gpu/in/suffv_tasks.jsonl", [{"query_id": q, "arm": a["sha"][:16], "query": REC[q]["query"], "evidence": a["blk"]} for (q, p), a in A.items() if a["blk"]])
        print("judge2", len(items))
    sys.exit(0)
if MODE == "score":
    jl = X.jl; S8 = {(r["query_id"], r["arm"]): r["label"] for r in jl(O / "gpu/out/suffv_llama8b.jsonl")}; S7 = {(r["query_id"], r["arm"]): r["label"] for r in jl(O / "gpu/out/suffv_qwen7b.jsonl")}
    L = {}
    for (q, p), a in A.items():
        lab = hl.labels(REC[q]["query"], goldof(q, REC[q]["query"]), a["key"])[0] or "UNRESOLVED"
        s = (S8.get((q, a["sha"][:16])), S7.get((q, a["sha"][:16]))) if a["sha"] else (None, None)
        L[(q, p)] = (lab, "UNRESOLVED" if None in s else (s[0] if s[0] == s[1] else "JUDGE_DISAGREEMENT"))
    ids = F["requests"]; n = len(ids)
    def wil(k, n, z=1.96): p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [round(100 * (c - m) / d, 2), round(100 * (c + m) / d, 2)]
    T = collections.Counter()
    for q in ids:
        b, d = L[(q, "B")], L[(q, "D")]
        if b[0] == "CORRECT" and d[0] == "WRONG": T["CORRECT->WRONG"] += 1
        if b[0] == "CORRECT" and d[0] == "ABSTAIN": T["CORRECT->ABSTAIN"] += 1
        if b[0] in ("WRONG", "ABSTAIN") and d[0] == "CORRECT": T["WRONG/ABSTAIN->CORRECT"] += 1
        if b[1] == "SUFFICIENT" and d[1] != "SUFFICIENT": T["SUFF->not"] += 1
        if b[1] != "SUFFICIENT" and d[1] == "SUFFICIENT": T["not->SUFF"] += 1
        T["unresolved_answer"] += (b[0] == "UNRESOLVED" or d[0] == "UNRESOLVED")
    share = F["changed_requests"] / F["stream"]; loss = T["CORRECT->WRONG"] + T["CORRECT->ABSTAIN"]; gain = T["WRONG/ABSTAIN->CORRECT"]; net_s = T["not->SUFF"] - T["SUFF->not"]
    B_c = sum(1 for q in ids if L[(q, "B")][0] == "CORRECT")
    R = {"frame": F["frame"], "changed_requests": F["changed_requests"], "changed_share_of_stream_pct": round(100 * share, 2), "n": n, "transitions": dict(T),
         "B_correct": B_c, "D_correct": sum(1 for q in ids if L[(q, "D")][0] == "CORRECT"), "B_sufficient": sum(1 for q in ids if L[(q, "B")][1] == "SUFFICIENT"), "D_sufficient": sum(1 for q in ids if L[(q, "D")][1] == "SUFFICIENT"),
         "new_error_rate_among_changed_pct": round(100 * loss / n, 2), "new_error_rate_ci95": wil(loss, n), "correction_rate_among_changed_pct": round(100 * gain / n, 2), "correction_rate_ci95": wil(gain, n),
         "net_correct_change_among_changed_pp": round(100 * (gain - loss) / n, 2), "net_sufficiency_change_among_changed_pp": round(100 * net_s / n, 2),
         "loss_rate_given_B_correct_pct": round(100 * loss / max(B_c, 1), 2), "loss_rate_given_B_correct_ci95": wil(loss, max(B_c, 1)),
         "stream_level_new_error_rate_pct": round(100 * share * loss / n, 3), "stream_level_new_error_ci95": [round(share * x, 3) for x in wil(loss, n)],
         "stream_level_net_correct_change_pp": round(100 * share * (gain - loss) / n, 3), "stream_level_net_sufficiency_change_pp": round(100 * share * net_s / n, 3)}
    json.dump(R, open(O / "outputs/changed_request_audit.json", "w"), indent=1); print(json.dumps(R, indent=1))
