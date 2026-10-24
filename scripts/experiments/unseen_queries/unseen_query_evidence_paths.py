"""Unseen-query experiment, evidence stage (no web search calls). Builds, for every frozen case, the evidence of four paths on the SAME incoming query, with the
identical evidence budget of the frozen generator context (audit_replay.ctx_for: pages under 200 characters dropped, 2,800-character total budget, rank order):
  A_matched  fresh Serper results of the incoming query, top-k (k = cached-list length), pages fetched in this experiment    [PRIMARY]
  A_full     fresh Serper top-10, pages fetched in this experiment                                                        [secondary only]
  B          the cached L2 URL list, pages fetched in this experiment (same window)
  C          full FreshCache: cached L2 list; pages FreshCache holds in L3 at the scripts/evaluation point are served from the stored snapshot version
             (audit_replay.snapshot_round_on_disk at the L3 cache time); the rest are the pages fetched in this experiment
Also: exclusion flow, URL/domain overlap, generation tasks (Llama-3.2-3B FORCED), sufficiency tasks (frozen SUFFV evidence format)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, hashlib, json, os, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from urllib.parse import urlparse
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "unseen_queries")); SR, PG = O / "raw/search", O / "raw/pages"
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C   # noqa: E402
E = C.load("reported"); p2 = E["p2"]
import answer_grading as e2e   # noqa: E402
MAXC, MINC = p2.MAXCHARS, p2.MINCHARS
FZ = json.load(open(O / "sample_frozen.json")); MV = {json.loads(l)["query_id"]: json.loads(l) for l in open(O / "cache_state_l2_matching.jsonl")}
G = {"matcher v2 reproduces the frozen L2 routing and cached lists for all live candidates": all(MV[c["query_id"]]["route"] == "L2" and MV[c["query_id"]]["cached_urls"] == c["cached_url_hashes"] for c in FZ["cases"])}
assert all(G.values()), G
man = {}
for l in open((DATA_ROOT / "url_manifest.jsonl"), encoding="utf-8"):
    x = json.loads(l)
    if x["run_id"] == "run_00": man.setdefault(x["url_hash"], {"title": x.get("title") or "", "snippet": x.get("snippet") or ""})
uid = lambda u: hashlib.sha1(u.encode()).hexdigest(); h16 = lambda s: hashlib.sha256((s or "").encode()).hexdigest()[:16]
def page(u):
    p = PG / f"{uid(u)}.json"
    if not u or not p.exists(): return {"status": "missing_cached_artifact" if not u else "not_fetched", "text": ""}
    d = json.load(open(p, encoding="utf-8"))
    p2_ = O / "raw/pages_pass2" / f"{uid(u)}.json"
    if not d.get("ok") and not (d.get("error") or "").strip() and p2_.exists(): d = json.load(open(p2_, encoding="utf-8"))   # pass 2: silent extraction failures only, identical rule for every path
    t = re.sub(r"\s+", " ", d.get("extracted_text") or "").strip()
    if not d.get("ok"): return {"status": "blocked_page" if str(d.get("status_code") or "").startswith(("401", "403", "429")) or "403" in str(d.get("error")) else "fetch_failure", "text": ""}
    return {"status": "ok" if len(t) >= MINC else "insufficient_text", "text": t}
def ctx(texts):
    parts, total = [], 0
    for t in texts:
        if len(t) < MINC: continue
        room = MAXC - total
        if room <= 0: break
        parts.append(t[:room]); total += min(len(t), room)
    return "\n\n".join(parts)
def evblock(items):   # frozen SUFFV evidence formatting (verifier_audit_keys.evblock): title, snippet, PAGE TEXT (>= 200 chars, first 2,000)
    lines = []
    for title, snip, t in items:
        blk = f"{len(lines) + 1}. {(title or '').strip()}"
        if snip: blk += f"\n   {snip.strip()}"
        if len(t) >= 200: blk += f"\n   PAGE TEXT: {t[:2000]}"
        lines.append(blk)
    return "\n".join(lines) or "(no stored evidence)"
norm = lambda u: re.sub(r"^www\.", "", urlparse(u).netloc.lower()) + urlparse(u).path.rstrip("/").lower() if u else ""
dom = lambda u: re.sub(r"^www\.", "", urlparse(u).netloc.lower()) if u else ""
CASES, FLOW, GEN, SUF = [], collections.Counter(), {}, {}
for c in FZ["cases"]:
    s = json.load(open(SR / f"{c['case_id']}.json", encoding="utf-8")); org = [o for o in ((s.get("response") or {}).get("organic") or []) if o.get("link")][:10]
    k = len(c["cached_url_hashes"]); r = {"case_id": c["case_id"], "query_id": c["query_id"], "query": c["incoming_query"], "k": k, "reference": c["reference_answer"], "searched_utc": s["searched_utc"], "excl": []}
    FLOW["sampled"] += 1
    search_ok = s.get("http_status") == 200 and len(org) > 0
    if not search_ok: r["excl"].append("search_api_failure" if s.get("http_status") != 200 else "search_no_results")
    cached_ok = all(c["cached_urls"]) and k > 0
    if not cached_ok: r["excl"].append("missing_cached_artifact")
    fresh_k = [(o.get("title"), o.get("snippet"), page(o["link"])) for o in org[:k]]; fresh_10 = [(o.get("title"), o.get("snippet"), page(o["link"])) for o in org]
    cached_live = [(man.get(uh, {}).get("title"), man.get(uh, {}).get("snippet"), page(u)) for uh, u in zip(c["cached_url_hashes"], c["cached_urls"])]
    l3t = MV[c["query_id"]].get("l3_cache_time", {}); cachedC = []
    for (title, snip, pg), uh in zip(cached_live, c["cached_url_hashes"]):
        if uh in l3t:
            t = re.sub(r"\s+", " ", e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, l3t[uh])) or "").strip(); cachedC.append((title, snip, {"status": "ok" if len(t) >= MINC else "insufficient_text", "text": t, "l3": True}))
        else: cachedC.append((title, snip, pg))
    fresh_avail = search_ok and any(x[2]["status"] == "ok" for x in fresh_k); cached_avail = cached_ok and any(x[2]["status"] == "ok" for x in cached_live)
    if search_ok and not fresh_avail: r["excl"].append("fresh_pages_unusable:" + ",".join(sorted({x[2]["status"] for x in fresh_k})))
    if cached_ok and not cached_avail: r["excl"].append("cached_pages_unusable:" + ",".join(sorted({x[2]["status"] for x in cached_live})))
    FLOW["search_succeeded"] += search_ok; FLOW["cached_list_available"] += cached_ok; FLOW["fresh_pages_available"] += fresh_avail; FLOW["cached_pages_available"] += cached_avail
    both = fresh_avail and cached_avail; FLOW["both_paths_evaluable"] += both
    ref_ok = bool(c["reference_answer"]); FLOW["both_evaluable_and_reference_available"] += both and ref_ok
    if not ref_ok: r["excl"].append("unavailable_reference")
    r.update({"search_ok": search_ok, "cached_ok": cached_ok, "fresh_avail": fresh_avail, "cached_avail": cached_avail, "both_evaluable": both, "reference_available": ref_ok,
              "url_overlap_k": len({norm(o["link"]) for o in org[:k]} & {norm(u) for u in c["cached_urls"] if u}), "url_overlap_top10": len({norm(o["link"]) for o in org} & {norm(u) for u in c["cached_urls"] if u}),
              "domain_overlap_k": len({dom(o["link"]) for o in org[:k]} & {dom(u) for u in c["cached_urls"] if u}), "domain_overlap_top10": len({dom(o["link"]) for o in org} & {dom(u) for u in c["cached_urls"] if u}),
              "fresh_links_top10": [o["link"] for o in org], "cached_urls": c["cached_urls"], "l3_served_pages_C": sum(1 for x in cachedC if x[2].get("l3")),
              "page_status": {"A_matched": [x[2]["status"] for x in fresh_k], "B": [x[2]["status"] for x in cached_live], "C": [x[2]["status"] for x in cachedC]}})
    r["paths"] = {}
    for name, items in (("A_matched", fresh_k), ("A_full", fresh_10), ("B_cached_live", cached_live), ("C_freshcache", cachedC)):
        cx = ctx([x[2]["text"] for x in items]); eb = evblock([(x[0], x[1], x[2]["text"]) for x in items]); key = ["t2", h16(cx), h16(c["incoming_query"])]
        r["paths"][name] = {"key": key, "ctx_chars": len(cx), "evidence_sha": hashlib.sha256(eb.encode()).hexdigest()[:16]}
        GEN.setdefault(tuple(key), {"key": key, "question": c["incoming_query"], "context": cx})
        SUF.setdefault((c["query_id"], r["paths"][name]["evidence_sha"]), {"query_id": c["query_id"], "arm": r["paths"][name]["evidence_sha"], "query": c["incoming_query"], "evidence": eb})
    CASES.append(r)
(O / "gpu/in").mkdir(parents=True, exist_ok=True); (O / "gpu/out").mkdir(parents=True, exist_ok=True)
for fn, rows in (("gen_tasks.jsonl", GEN.values()), ("suffv_tasks.jsonl", SUF.values())):
    with open(O / "gpu/in" / fn, "w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open(O / "work/cases_built.jsonl", "w", encoding="utf-8") as f:
    for r in CASES: f.write(json.dumps(r, ensure_ascii=False) + "\n")
EX = collections.Counter(e.split(":")[0] for r in CASES for e in r["excl"])
json.dump({"flow": dict(FLOW), "exclusion_reasons": dict(EX), "gen_tasks": len(GEN), "suffv_tasks": len(SUF), "gates": G}, open(O / "work/build_summary.json", "w"), indent=1)
print(json.dumps({"flow": dict(FLOW), "exclusions": dict(EX), "gen": len(GEN), "suffv": len(SUF)}, indent=1))
