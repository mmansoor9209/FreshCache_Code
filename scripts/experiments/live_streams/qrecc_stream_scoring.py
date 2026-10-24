"""Scoring of the natural QReCC workload: the live-stream confirmation scorer (paraphrase_stream_scoring.py) applied unchanged to the QReCC stream,
plus measured cost (logged search/fetch durations) and the pre-specified prediction test (mode: predict).
Configurations: P = own-list depth 2, Llama-3.2-3B FORCED, 2,800-character context (frozen assembly); S = depth 5, Qwen2.5-14B,
8,000-token context. Census: every request whose signature differs between compared policies is labelled (frozen two-judge rule,
held-out abstention patterns first); answer-cache roots of losses are labelled on their own question. Modes: build | gen14 --gpu G
--shard i/n | items | score | predict. Pages: the evaluator store (raw/qr/pages, version = (url, request order))."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, collections, hashlib, json, math, os, pickle, re, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "natural_workload")); ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); RC = O / "raw/qr"; MODE = sys.argv[1]
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
HELD = ["i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information"]
PAIRS = [("B", "D"), ("Be", "Dp"), ("Bm", "Dp"), ("B", "Dp"), ("B", "Be"), ("B", "Bm"), ("D", "Dp")]; POLS = ["B", "Be", "Bm", "D", "Dp"]
CFG = {"P": {"k": 2, "cell": "a3b"}, "S": {"k": 5, "cell": "q14L"}}
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
F = json.load(open(O / "work/qr_block.json")); REQ = {r["query_id"]: r for r in F["requests"]}; ORD = {r["query_id"]: i for i, r in enumerate(F["requests"])}
TORD = {r["t_virtual"]: i for i, r in enumerate(F["requests"])}
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
if MODE == "gen14":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gpu", required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu; si, sn = map(int, a.shard.split("/"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FORCED"}
    mid = "Qwen/Qwen2.5-14B-Instruct"; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
    out = O / f"gpu/out/qr_answers14_shard{si}_of{sn}.jsonl"; done = {tuple(r["key"]) for f in (O / "gpu/out").glob("qr_answers14_shard*.jsonl") for r in jl(f)}
    todo = sorted([t for i, t in enumerate(jl(O / "gpu/in/qr_tasks14.jsonl")) if i % sn == si and tuple(t["key"]) not in done], key=lambda t: len(t["context"]))
    def clip(t):
        t = (t or "").strip()
        if t.startswith("INSUFFICIENT_CONTEXT"): return "INSUFFICIENT_CONTEXT"
        for sep in (".", "\n"):
            if sep in t: return t[:t.index(sep) + 1].strip()
        return t
    fh = open(out, "a", encoding="utf-8")
    BS = int(os.environ.get("GEN_BS", "4"))
    for s in range(0, len(todo), BS):
        ch = todo[s:s + BS]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": K["FORCED"].format(question=c["question"], context=c["context"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                  return_tensors="pt", padding=True, truncation=True, max_length=9000).to(m.device)
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            fh.write(json.dumps({"key": c["key"], "raw": t, "answer": clip(t) or "I don't know."}, ensure_ascii=False) + "\n")
        fh.flush()
    sys.exit(0)
CR = pickle.load(open(O / "work/qr_rows.pkl", "rb")); ROWS = CR["rows"]
def row(cfg, p, q): return ROWS[(CFG[cfg]["k"], p, q)]
def need(cfg):   # (request, policy) pairs to label: census of every compared pair, plus answer-cache roots
    S = set()
    for a, b in PAIRS:
        for q in REQ:
            if sig(row(cfg, a, q)) != sig(row(cfg, b, q)): S.add((q, a)); S.add((q, b))
    return S
if MODE == "build":
    from transformers import AutoTokenizer
    QT = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    META = {}
    for p in sorted((RC / "search").glob("*.json")):
        s = json.load(open(p, encoding="utf-8"))
        for o in ((s.get("response") or {}).get("organic") or []):
            if o.get("link"): META.setdefault(o["link"], (o.get("title") or "", o.get("snippet") or ""))
    _pg = {}
    def page(u, tv):
        f = RC / "pages" / f"{hashlib.sha1(u.encode()).hexdigest()}_{TORD[tv]:05d}.json"
        if f not in _pg:
            d = json.load(open(f, encoding="utf-8")) if f.exists() else {"ok": False}
            _pg[f] = re.sub(r"\s+", " ", d.get("extracted_text") or "").strip() if d.get("ok") else ""
        return _pg[f]
    def ctx(texts, cell):
        parts, total = [], 0
        for t in texts:
            if len(t) < 200: continue
            if cell == "a3b":
                room = 2800 - total
                if room <= 0: break
                parts.append(t[:room]); total += min(len(t), room)
            else:
                room = 8000 - total
                if room <= 0: break
                ids = QT(t, add_special_tokens=False)["input_ids"][:room]; parts.append(QT.decode(ids)); total += len(ids)
        return "\n\n".join(parts)
    def evblock(ev):
        lines = []
        for u, tv in ev:
            title, snip = META.get(u, ("", "")); t = page(u, tv); blk = f"{len(lines) + 1}. {title.strip()}"
            if snip: blk += f"\n   {snip.strip()}"
            if len(t) >= 200: blk += f"\n   PAGE TEXT: {t[:2000]}"
            lines.append(blk)
        return "\n".join(lines) or "(no stored evidence)"
    A, G3, G14, SU = {}, {}, {}, {}
    for cfg in CFG:
        cell = CFG[cfg]["cell"]; NS = need(cfg)
        roots = {(row(cfg, p, q)["root"], p) for q, p in NS if row(cfg, p, q)["tier"] == "L1"}
        for q, p in sorted(NS | {(r, p) for r, p in roots}):
            rw = row(cfg, p, q) if (q, p) in NS else None
            if rw is None:   # root: its own generation (the stored answer has the same input)
                rw = row(cfg, p, q)
            ev = [tuple(x) for x in rw["ev"]]; cx = ctx([page(u, tv) for u, tv in ev], cell); gq = rw["gen_query"]; key = [f"q{cfg}", h16(cx), h16(gq)]
            eb = evblock(ev); A[(cfg, q, p)] = {"key": key, "tier": rw["tier"], "root": rw.get("root"), "root_route": rw.get("root_route"), "served": rw.get("served"), "full_sha": h16(eb), "ctx_sha": h16(cx),
                                               "in_census": (q, p) in NS}
            (G3 if cell == "a3b" else G14).setdefault(tuple(key), {"key": key, "question": gq, "context": cx})
            if (q, p) in NS:
                SU.setdefault((q, "cfull_" + h16(eb)), {"query_id": q, "arm": "cfull_" + h16(eb), "query": REQ[q]["query"], "evidence": eb})
                SU.setdefault((q, "cctx_" + h16(cx)), {"query_id": q, "arm": "cctx_" + h16(cx), "query": REQ[q]["query"], "evidence": cx or "(no stored evidence)"})
    pickle.dump(A, open(O / "work/qr_items.pkl", "wb")); wjl(O / "gpu/in/gen_tasks_qr.jsonl", list(G3.values())); wjl(O / "gpu/in/qr_tasks14.jsonl", list(G14.values())); wjl(O / "gpu/in/suffv_tasks.jsonl", list(SU.values()))
    print("labelled pairs", collections.Counter(k[0] for k in A), "gen3", len(G3), "gen14", len(G14), "suffv", len(SU)); sys.exit(0)
A = pickle.load(open(O / "work/qr_items.pkl", "rb")); ANS = {}
for f in sorted((O / "gpu/out").glob("answers_llama3b*.jsonl")):
    for r in jl(f):
        if r["key"][0] in ("qP", "qS"): ANS[tuple(r["key"])] = r["answer"]
for f in sorted((O / "gpu/out").glob("qr_answers14_shard*.jsonl")):
    for r in jl(f): ANS[tuple(r["key"])] = r["answer"]
JD = {"llama8b": {}, "qwen7b": {}}
for j in JD:
    for f in sorted((O / "gpu/out").glob(f"judge_{j}*.jsonl")):
        for r in jl(f): JD[j][tuple(r["sig"])] = r["label"]
def label(cfg, q, p):
    x = A[(cfg, q, p)]; a = ANS.get(tuple(x["key"]))
    if a is None: return None, None
    if any(h in a.strip().lower() for h in HELD): return "ABSTAIN", None
    s = (h16(REQ[q]["query"]), h16(REQ[q]["reference"]), h16(a)); l = [JD[j].get(s) for j in JD]
    if None in l: return None, {"sig": list(s), "q": REQ[q]["query"], "gold": REQ[q]["reference"], "cand": a}
    return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), None
if MODE == "items":
    it, miss = {}, 0
    for (cfg, q, p) in A:
        l, item = label(cfg, q, p); miss += l is None and item is None
        if item: it[tuple(item["sig"])] = item
    wjl(O / "gpu/in/judge2_items.jsonl", list(it.values())); print("judge2", len(it), "answers missing", miss); sys.exit(0)
if MODE == "score":
    SUF = {"llama8b": {}, "qwen7b": {}}
    for j in SUF:
        for f in sorted((O / "gpu/out").glob(f"suffv_{j}*.jsonl")):
            for r in jl(f): SUF[j][(r["query_id"], r["arm"])] = r["label"]
    def suff(q, arm):
        s = (SUF["llama8b"].get((q, arm)), SUF["qwen7b"].get((q, arm))); return None if None in s else s[0] == s[1] == "SUFFICIENT"
    COST = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]; n = len(REQ); ids = list(REQ)   # cost coefficients from component_costs.json (serving_cost_summary.py)
    R = {"n_requests": n, "summary": json.load(open(O / "work/qr_collect_summary.json")), "configs": {}}
    for cfg in CFG:
        k = CFG[cfg]["k"]; out = {"policies": {}, "comparisons": {}}
        for p in POLS:
            m = CR["metrics"][f"{k}|{p}"]; out["policies"][p] = {"cost_ms": round((m["search_calls"] * COST[0] + m["fetches"] * COST[1] + m["generations"] * COST[2]) / n + COST[3], 1),
                "search_per_1k": round(1000 * m["search_calls"] / n, 1), "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1),
                "answer_hits": m["l1_hits"], "list_reuses": m["l2_hits"]}
        unres = 0
        for a, b in PAIRS:
            ch = [q for q in ids if sig(row(cfg, a, q)) != sig(row(cfg, b, q))]; L = {}
            for q in ch:
                for p in (a, b): L[(q, p)] = label(cfg, q, p)[0]
            un = [q for q in ch if None in (L[(q, a)], L[(q, b)])]; unres += len(un); ok = [q for q in ch if q not in un]
            cor = lambda q, p: L[(q, p)] == "CORRECT"
            nw = [q for q in ok if cor(q, a) and L[(q, b)] == "WRONG"]; ab = [q for q in ok if cor(q, a) and L[(q, b)] == "ABSTAIN"]; ga = [q for q in ok if not cor(q, a) and cor(q, b)]
            d = {q: 0 for q in ids}
            for q in ga: d[q] = 1
            for q in nw + ab: d[q] = -1
            mu = sum(d.values()) / n
            def clus(key):
                g = collections.defaultdict(float)
                for q in ids: g[key(q)] += d[q] - mu
                G = len(g); return math.sqrt(G / (G - 1) * sum(v * v for v in g.values())) / n
            se_c = clus(lambda q: REQ[q]["cluster_id"]); se_r = clus(lambda q: (A[(cfg, q, b)]["root"] if (cfg, q, b) in A and A[(cfg, q, b)]["tier"] == "L1" else q))
            def cat(q, loss):
                x, y = A[(cfg, q, b)], A[(cfg, q, a)]
                if y["tier"] == "L1" and x["tier"] in ("L2", "miss"): return ("regeneration_" + ("loss" if loss else "gain")) + ("_via_reused_list" if x["tier"] == "L2" else "_fresh")
                if x["tier"] == "L2": return "immediate_L2" + ("" if loss else "_gain")
                if x["tier"] == "L1":
                    if not loss: return "served_entry_gain"
                    rl = label(cfg, x["root"], b)[0] if (cfg, x["root"], b) in A else None
                    kind = "L2_root" if x["root_route"] == "L2" else "fresh_root"
                    return {"WRONG": "propagation_of_wrong_", "ABSTAIN": "propagation_of_wrong_", "CORRECT": "misuse_of_correct_"}.get(rl, "unlabelled_") + kind
                return "other"
            bc = sum(1 for q in ids if (q, a) in L and cor(q, a))
            out["comparisons"][f"{a}->{b}"] = {"changed": len(ch), "unresolved": len(un), "newly_wrong": len(nw), "abstention_losses": len(ab), "total_losses": len(nw) + len(ab), "gains": len(ga),
                "net": len(ga) - len(nw) - len(ab), "base_correct_among_changed": bc, "stream_pp": round(100 * mu, 3),
                "ci_pp_cluster": [round(100 * (mu - 1.96 * se_c), 3), round(100 * (mu + 1.96 * se_c), 3)], "ci_pp_root": [round(100 * (mu - 1.96 * se_r), 3), round(100 * (mu + 1.96 * se_r), 3)],
                "distinct_clusters_with_loss": len({REQ[q]["cluster_id"] for q in nw + ab}),
                "suff_full_change_pp": round(100 * sum((suff(q, "cfull_" + A[(cfg, q, b)]["full_sha"]) or 0) - (suff(q, "cfull_" + A[(cfg, q, a)]["full_sha"]) or 0) for q in ch) / n, 3),
                "suff_ctx_change_pp": round(100 * sum((suff(q, "cctx_" + A[(cfg, q, b)]["ctx_sha"]) or 0) - (suff(q, "cctx_" + A[(cfg, q, a)]["ctx_sha"]) or 0) for q in ch) / n, 3),
                "loss_categories": dict(collections.Counter(cat(q, True) for q in nw + ab)), "gain_categories": dict(collections.Counter(cat(q, False) for q in ga))}
        out["unresolved"] = unres
        LINKS = {}
        for f in (RC / "search").glob("*.json"):
            s_ = json.load(open(f, encoding="utf-8")); LINKS[s_["query_id"]] = [o["link"] for o in ((s_.get("response") or {}).get("organic") or []) if o.get("link")][:10]
        out["list_identity"] = {p: {"reuses": sum(row(cfg, p, q)["tier"] == "L2" for q in ids), "identical_to_own_list": sum(row(cfg, p, q)["tier"] == "L2" and row(cfg, p, q)["served"] == LINKS[q][:k] for q in ids)} for p in ("D", "Dp")}
        R["configs"][cfg] = out
    json.dump(R, open(O / "outputs/qrecc_results.json", "w"), indent=1); print(json.dumps(R, indent=1))

if MODE == "predict":   # pre-registered prediction (configuration P, D vs B, immediate list-reuse events that change correctness)
    from scipy.stats import fisher_exact
    SUF = {"llama8b": {}, "qwen7b": {}}
    for j in SUF:
        for f in sorted((O / "gpu/out").glob(f"suffv_{j}*.jsonl")):
            for r in jl(f): SUF[j][(r["query_id"], r["arm"])] = r["label"]
    def suff(q, arm):
        s = (SUF["llama8b"].get((q, arm)), SUF["qwen7b"].get((q, arm))); return None if None in s else s[0] == s[1] == "SUFFICIENT"
    cnt = collections.Counter(); unres = 0
    for q in REQ:
        x, y = A.get(("P", q, "D")), A.get(("P", q, "B"))
        if x is None or y is None or x["tier"] != "L2": continue
        lb, ld = label("P", q, "B")[0], label("P", q, "D")[0]
        if lb is None or ld is None or (lb == "CORRECT") == (ld == "CORRECT"): continue
        sr, so = suff(q, "cctx_" + x["ctx_sha"]), suff(q, "cctx_" + y["ctx_sha"])
        if sr is None or so is None: unres += 1; continue
        stratum = "reused_worse" if (so and not sr) else "reused_better" if (sr and not so) else "same_sufficiency"
        cnt[(stratum, "gain" if ld == "CORRECT" else "loss")] += 1
    tab = [[cnt[("reused_worse", "loss")], cnt[("reused_worse", "gain")]], [cnt[("reused_better", "loss")], cnt[("reused_better", "gain")]]]
    p = fisher_exact(tab)[1] if sum(map(sum, tab)) else None
    R = {"counts": {f"{a}|{b}": v for (a, b), v in cnt.items()}, "unresolved": unres, "fisher_2x2_worse_better_by_loss_gain": tab, "fisher_p": p,
         "prediction_worse_losses_gt_gains": tab[0][0] > tab[0][1], "prediction_better_gains_gt_losses": tab[1][1] > tab[1][0]}
    json.dump(R, open(O / "outputs/qrecc_prediction.json", "w"), indent=1); print(json.dumps(R, indent=1))
if MODE == "measured":   # measured network cost per policy from the logged durations (generation: modeled 866 ms)
    SD = {}; FD = {}
    for f in (RC / "search").glob("*.json"):
        s = json.load(open(f, encoding="utf-8")); SD[s["query_id"]] = s.get("duration_s")
    for f in (RC / "pages").glob("*.json"):
        d = json.load(open(f, encoding="utf-8")); FD[(d["url"], d["order"])] = d.get("duration_s")
    out = {}
    for cfg in CFG:
        for p in POLS:
            srch = fetch = 0.0; nf = 0
            for q in REQ:
                r = row(cfg, p, q)
                if r.get("searched"): srch += SD.get(q) or 0.0
                for u, tv in r["ev"]:
                    if tv == REQ[q]["t_virtual"] and r["tier"] != "L1":
                        dd = FD.get((u, TORD[tv])); fetch += dd or 0.0; nf += 1
            m = CR["metrics"][f"{CFG[cfg]['k']}|{p}"]; n = len(REQ)
            out[f"{cfg}|{p}"] = {"measured_search_s_per_req": round(srch / n, 3), "measured_fetch_s_per_req": round(fetch / n, 3), "modeled_gen_s_per_req": round(0.866 * m["generations"] / n, 3),
                                 "measured_total_s_per_req": round((srch + fetch) / n + 0.866 * m["generations"] / n, 3)}
    med_search = sorted(v for v in SD.values() if v)[len(SD) // 2]
    json.dump({"per_policy": out, "median_search_s": med_search}, open(O / "outputs/qrecc_measured.json", "w"), indent=1); print(json.dumps(out, indent=1))
