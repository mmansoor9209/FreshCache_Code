"""Deeper cached lists vs deeper fresh retrieval on the 150 paired live cases (prospective, pre-registered).
Paths: Fresh_k = the incoming query's top-k (t1 search); Cached_k = the SOURCE query's own top-k from its earlier search (t0), never expanded
with incoming results; k in {2,5,10}. Page versions: fetched at t1 for both paths; variant Cached_k@t0 uses the t0 page versions.
Cells: a3b  = Llama-3.2-3B FORCED, 2,800-character context (frozen pipeline: pages < 200 chars dropped, rank order, budget truncation; input
              limit 1,024 tokens as frozen)
       q14  = Qwen2.5-14B, same 2,800-character context
       q14L = Qwen2.5-14B, 8,000-token context (same assembly, no character cap; truncation by Qwen tokens), input limit 9,000 tokens
Identical prompt (frozen FORCED, user turn for 14B), greedy, 80 new tokens, first-sentence clip (raw output retained).
Judged: SUFFV on the full top-k evidence (frozen evblock format) and on the exact supplied context; two-judge rule on the answer.
Modes: build | gen14 --gpu G --shard i/n | items | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, collections, hashlib, json, math, os, re, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); T2 = (OUTPUT_ROOT / "unseen_queries"); RB = O / "raw/deep_lists"
MODE = sys.argv[1]; MINC, MAXC, TOKL = 200, 2800, 8000; KS = (2, 5, 10)
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]; uid = lambda u: hashlib.sha1(u.encode()).hexdigest(); qid = lambda q: hashlib.sha1(q.encode()).hexdigest()[:16]
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
if MODE == "build":
    from transformers import AutoTokenizer
    QT = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct"); LT = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", revision="0cb88a4f764b7a12671c53f0838cd831a0843b95")
    K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FORCED"}
    CASES = [c for c in jl(T2 / "work/cases_built.jsonl") if c["both_evaluable"] and c["reference_available"]]; FZ = {c["case_id"]: c for c in json.load(open(T2 / "sample_frozen.json"))["cases"]}
    def org(p): s = json.load(open(p, encoding="utf-8")); return [o for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")][:10], s["searched_utc"]
    def page(u, ph):
        p = RB / f"pages_{ph}" / f"{uid(u)}.json"
        if not p.exists(): return {"status": "not_fetched", "text": ""}
        d = json.load(open(p, encoding="utf-8")); t = re.sub(r"\s+", " ", d.get("extracted_text") or "").strip()
        if not d.get("ok"): return {"status": "fetch_failure", "text": ""}
        return {"status": "ok" if len(t) >= MINC else "insufficient_text", "text": t}
    def ctx_chars(texts):
        parts, total, used = [], 0, []
        for i, t in enumerate(texts):
            if len(t) < MINC: continue
            room = MAXC - total
            if room <= 0: break
            parts.append(t[:room]); total += min(len(t), room); used.append(i)
        return "\n\n".join(parts), used
    def ctx_tokens(texts):
        parts, total, used = [], 0, []
        for i, t in enumerate(texts):
            if len(t) < MINC: continue
            room = TOKL - total
            if room <= 0: break
            ids = QT(t, add_special_tokens=False)["input_ids"]; take = ids[:room]; parts.append(QT.decode(take)); total += len(take); used.append(i)
        return "\n\n".join(parts), used
    def evblock(items):
        lines = []
        for title, snip, t in items:
            blk = f"{len(lines) + 1}. {(title or '').strip()}"
            if snip: blk += f"\n   {snip.strip()}"
            if len(t) >= 200: blk += f"\n   PAGE TEXT: {t[:2000]}"
            lines.append(blk)
        return "\n".join(lines) or "(no stored evidence)"
    P, TASKS, SUF, AUD = {}, {}, {}, collections.Counter()
    for c in CASES:
        src = FZ[c["case_id"]]["list_origin_query"]; inc, t_in = org(RB / "search_t1" / f"{qid(c['query'])}.json"); sl, t_src = org(RB / "search_t0" / f"{qid(src)}.json")
        assert t_src < t_in; P[c["case_id"]] = {"query": c["query"], "source_query": src, "reference": c["reference"], "t_source": t_src, "t_incoming": t_in, "n_incoming": len(inc), "n_source": len(sl), "paths": {}}
        for k in KS:
            for path, items, ph in (("Fresh", inc[:k], "t1"), ("Cached", sl[:k], "t1"), ("Cached_t0pages", sl[:k], "t0")):
                pg = [page(o["link"], ph) for o in items]; texts = [x["text"] for x in pg]; eb = evblock([(o.get("title"), o.get("snippet"), x["text"]) for o, x in zip(items, pg)])
                arm = f"{path}_k{k}"; rec = {"links": [o["link"] for o in items], "usable_pages": sum(x["status"] == "ok" for x in pg), "full_sha": h16(eb)}
                SUF.setdefault((c["case_id"], "full_" + h16(eb)), {"query_id": c["case_id"], "arm": "full_" + h16(eb), "query": c["query"], "evidence": eb})
                for cell in ("a3b", "q14", "q14L"):
                    cx, used = ctx_tokens(texts) if cell == "q14L" else ctx_chars(texts)
                    prompt = K["FORCED"].format(question=c["query"], context=cx)
                    ntok = len(LT(prompt)["input_ids"]) if cell == "a3b" else len(QT(QT.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
                    rec[cell] = {"ctx_sha": h16(cx), "input_tokens": ntok, "model_truncated": cell == "a3b" and ntok > 1024, "ctx_chars": len(cx), "pages_contributing": len(used),
                                 "contributing_links": [items[i]["link"] for i in used], "chars_available": sum(len(t) for t in texts if len(t) >= MINC)}
                    TASKS.setdefault((cell, h16(cx), h16(c["query"])), {"cell": cell, "key": [cell, h16(cx), h16(c["query"])], "question": c["query"], "context": cx})
                    SUF.setdefault((c["case_id"], "ctx_" + h16(cx)), {"query_id": c["case_id"], "arm": "ctx_" + h16(cx), "query": c["query"], "evidence": cx or "(no stored evidence)"})
                P[c["case_id"]]["paths"][arm] = rec
    rows = list(TASKS.values())
    for i, r in enumerate(rows): r["tid"] = i
    wjl(O / "gpu/in/deep_list_tasks.jsonl", rows); wjl(O / "gpu/in/gen_tasks.jsonl", [{"key": r["key"], "question": r["question"], "context": r["context"]} for r in rows if r["cell"] == "a3b"])
    wjl(O / "gpu/in/suffv_tasks.jsonl", list(SUF.values())); json.dump(P, open(O / "work/deep_list_paths.json", "w"), indent=1, ensure_ascii=False)
    print("cases", len(P), "tasks", collections.Counter(r["cell"] for r in rows), "suffv", len(SUF)); sys.exit(0)
if MODE == "gen14":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gpu", required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu; si, sn = map(int, a.shard.split("/"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FORCED"}
    mid = "Qwen/Qwen2.5-14B-Instruct"; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    out = O / f"gpu/out/deep_list_answers14_shard{si}.jsonl"; done = {r["tid"] for r in jl(out)}
    todo = sorted([t for t in jl(O / "gpu/in/deep_list_tasks.jsonl") if t["cell"] != "a3b" and t["tid"] % sn == si and t["tid"] not in done], key=lambda t: len(t["context"]))
    def clip(t):
        t = (t or "").strip()
        if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
        for sep in (".", "\n"):
            if sep in t: return t[:t.index(sep) + 1].strip()
        return t
    fh = open(out, "a", encoding="utf-8")
    for s in range(0, len(todo), 4):
        ch = todo[s:s + 4]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": K["FORCED"].format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                  return_tensors="pt", padding=True, truncation=True, max_length=9000).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            fh.write(json.dumps({"tid": c["tid"], "key": c["key"], "raw": t, "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n")
        fh.flush()
    sys.exit(0)
P = json.load(open(O / "work/deep_list_paths.json")); TK = jl(O / "gpu/in/deep_list_tasks.jsonl")
ANS = {}
for f in sorted((O / "gpu/out").glob("answers_llama3b*.jsonl")):
    for r in jl(f):
        if r["key"][0] == "a3b": ANS[tuple(r["key"])] = r["answer"]
for f in sorted((O / "gpu/out").glob("deep_list_answers14_shard*.jsonl")):
    for r in jl(f): ANS[tuple(r["key"])] = r["answer"]
JD = {"llama8b": {}, "qwen7b": {}}
for j in JD:
    for f in sorted((O / "gpu/out").glob(f"judge_{j}*.jsonl")):
        for r in jl(f): JD[j][tuple(r["sig"])] = r["label"]
def label(cid, cell, arm):
    c = P[cid]; x = c["paths"][arm][cell]; a = ANS.get((cell, x["ctx_sha"], h16(c["query"])))
    if a is None: return None, None
    if any(h in a.strip().lower() for h in HELD): return "ABSTAIN", None
    s = (h16(c["query"]), h16(c["reference"]), h16(a)); l = [JD[j].get(s) for j in JD]
    if None in l: return None, {"sig": list(s), "q": c["query"], "gold": c["reference"], "cand": a}
    return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), None
if MODE == "items":
    it, miss = {}, 0
    for cid, c in P.items():
        for arm in c["paths"]:
            for cell in ("a3b", "q14", "q14L"):
                l, item = label(cid, cell, arm); miss += l is None and item is None
                if item: it[tuple(item["sig"])] = item
    wjl(O / "gpu/in/judge2_items.jsonl", list(it.values())); print("judge2", len(it), "answers missing", miss); sys.exit(0)
if MODE == "score":
    SU = {"llama8b": {}, "qwen7b": {}}
    for j in SU:
        for f in sorted((O / "gpu/out").glob(f"suffv_{j}*.jsonl")):
            for r in jl(f): SU[j][(r["query_id"], r["arm"])] = r["label"]
    def suff(cid, arm):
        s = (SU["llama8b"].get((cid, arm)), SU["qwen7b"].get((cid, arm))); return None if None in s else s[0] == s[1] == "SUFFICIENT"
    ids = sorted(P); n = len(ids); R = {"n_cases": n, "cells": {}, "collection": {"source_before_incoming_all": all(P[c]["t_source"] < P[c]["t_incoming"] for c in ids),
         "incoming_lists_with_10": sum(P[c]["n_incoming"] == 10 for c in ids), "source_lists_with_10": sum(P[c]["n_source"] == 10 for c in ids)}}
    def mc(b, c_):
        m = b + c_; return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c_) + 1)) / 2 ** m)
    unres = 0
    for cell in ("a3b", "q14", "q14L"):
        R["cells"][cell] = {}
        for k in KS:
            row = {}
            for path in ("Fresh", "Cached", "Cached_t0pages"):
                arm = f"{path}_k{k}"; L = {c: label(c, cell, arm)[0] for c in ids}; unres += sum(v is None for v in L.values())
                x = [P[c]["paths"][arm] for c in ids]
                row[path] = {"correct": sum(v == "CORRECT" for v in L.values()), "abstain": sum(v == "ABSTAIN" for v in L.values()), "wrong": sum(v == "WRONG" for v in L.values()),
                             "suff_full_evidence": sum(bool(suff(c, "full_" + P[c]["paths"][arm]["full_sha"])) for c in ids),
                             "suff_supplied_context": sum(bool(suff(c, "ctx_" + P[c]["paths"][arm][cell]["ctx_sha"])) for c in ids),
                             "mean_input_tokens": round(sum(y[cell]["input_tokens"] for y in x) / n, 1), "mean_pages_contributing": round(sum(y[cell]["pages_contributing"] for y in x) / n, 2),
                             "mean_usable_pages": round(sum(y["usable_pages"] for y in x) / n, 2), "model_truncated": sum(y[cell]["model_truncated"] for y in x),
                             "search_calls_at_serving": n if path == "Fresh" else 0, "page_fetches": sum(len(set(y["links"])) for y in x)}
                row[path]["_L"] = L
            for a, b in (("Fresh", "Cached"), ("Fresh", "Cached_t0pages")):
                La, Lb = row[a]["_L"], row[b]["_L"]; lo = sum(La[c] == "CORRECT" and Lb[c] != "CORRECT" for c in ids); ga = sum(La[c] != "CORRECT" and Lb[c] == "CORRECT" for c in ids)
                row[f"{b}_vs_{a}"] = {"losses": lo, "gains": ga, "net_pp": round(100 * (ga - lo) / n, 1), "mcnemar_p": round(mc(lo, ga), 4),
                                      "suff_full_gap_pp": round(100 * (row[b]["suff_full_evidence"] - row[a]["suff_full_evidence"]) / n, 1),
                                      "suff_context_gap_pp": round(100 * (row[b]["suff_supplied_context"] - row[a]["suff_supplied_context"]) / n, 1)}
            R["cells"][cell][f"k{k}"] = row
        for path in ("Fresh", "Cached"):   # depth effect within a path
            L2, L10 = R["cells"][cell]["k2"][path]["_L"], R["cells"][cell]["k10"][path]["_L"]
            R["cells"][cell][f"{path}_k10_vs_k2"] = {"losses": sum(L2[c] == "CORRECT" and L10[c] != "CORRECT" for c in ids), "gains": sum(L2[c] != "CORRECT" and L10[c] == "CORRECT" for c in ids)}
    for cell in R["cells"].values():
        for k in KS:
            for path in ("Fresh", "Cached", "Cached_t0pages"): cell[f"k{k}"][path].pop("_L")
    R["unresolved"] = unres
    json.dump(R, open(O / "outputs/deep_list_evaluation.json", "w"), indent=1); print(json.dumps(R, indent=1))
