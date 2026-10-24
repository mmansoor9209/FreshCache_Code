"""Reference validation stage 0: copy the minimum read-only inputs into this folder.
Reads existing project files; writes only under outputs/reference_validation/inputs/."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json, hashlib, pathlib, sys, collections
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); W = ((OUTPUT_ROOT / "reference_validation")); IN = W/"inputs"
A2 = (OUTPUT_ROOT / "general_audit"); HS = (OUTPUT_ROOT / "quality_selection/heldout_sensitivity")
sys.path.insert(0, str(ROOT / "freshcache/simulation")); import page_version_replay as ma
PAGE_CAP, TOTAL_CAP, MIN_PAGE = 4000, 10000, 200
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
src = {str(p.relative_to(ROOT)): sha(p) for p in [A2/"sample.json", A2/"answers.jsonl", HS/"h_sample.jsonl", HS/"h_answers.jsonl", (DATA_ROOT / "url_manifest.jsonl")]}
S = json.load(open(A2/"sample.json"))["requests"]; H = {d["query_id"]: d for d in jl(HS/"h_sample.jsonl")}
ans = {tuple(r["key"]): r["answer"] for r in jl(A2/"answers.jsonl")}
for r in jl(HS/"h_answers.jsonl"): ans.setdefault(tuple(r["key"]), r["answer"])
url_of = {}
for m in jl((DATA_ROOT / "url_manifest.jsonl")):
    if m.get("url_hash"): url_of.setdefault(m["url_hash"], m.get("url", ""))
req, out, ev = [], [], []; stats = collections.Counter()
for d in S:
    q = d["query_id"]; rnd = ma.version_at(float(d["t"]))
    req.append({"query_id": q, "cluster_id": d["cluster_id"], "freshness_class": d["fc"], "question": d["query"], "reference_answer": d["gold"],
                "replay_time_s": d["t"], "snapshot_round": rnd})
    arms = {"fresh": (d["fresh_key"], "fresh"), "freshcache": (d["fc_key"], d["fc_tier"]),
            "semanticttl_quality_0.60_1-2": (H[q]["new_sttl_key"], H[q]["new_sttl_tier"]),
            "semanticttl_drift_0.40_1-16": (d["sttl_key"], d["sttl_tier"])}
    for arm, (k, tier) in arms.items():
        a = ans.get(tuple(k)); stats[f"{arm}_{'ok' if a is not None else 'missing'}"] += 1
        out.append({"query_id": q, "arm": arm, "tier": tier, "answer_key": k, "answer": a})
    pages, tot, seen = [], 0, set()
    for uh, tv in d["fresh_versions"]:
        if uh in seen: continue
        seen.add(uh); r = ma.version_at(float(tv)); p = (DATA_ROOT / "snapshots")/r/f"{uh}.json"
        if not p.exists(): pages.append({"url_hash": uh, "round": r, "available": False}); stats["page_missing"] += 1; continue
        s = json.load(open(p, encoding="utf-8")); txt = " ".join((s.get("extracted_text") or "").split())
        rec = {"url_hash": uh, "url": url_of.get(uh, s.get("url", "")), "round": r, "available": True, "fetched_at": s.get("fetched_at"),
               "status_code": s.get("status_code"), "content_hash": s.get("content_hash"), "chars": len(txt)}
        if len(txt) >= MIN_PAGE and tot < TOTAL_CAP:
            rec["text"] = txt[:min(PAGE_CAP, TOTAL_CAP - tot)]; tot += len(rec["text"]); rec["used"] = True
        else: rec["used"] = False
        pages.append(rec)
    used = [x for x in pages if x.get("used")]
    ev.append({"query_id": q, "snapshot_round": rnd, "pages": pages, "n_pages_used": len(used), "evidence_chars": tot,
               "evidence_time_min": min((x["fetched_at"] for x in used), default=None), "evidence_time_max": max((x["fetched_at"] for x in used), default=None)})
    stats["no_usable_evidence" if not used else "has_evidence"] += 1
for name, rows in (("requests.jsonl", req), ("policy_outputs.jsonl", out), ("evidence.jsonl", ev)):
    with open(IN/name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"source_sha256": src, "stats": dict(stats), "n_requests": len(req), "unique_ids": len({r['query_id'] for r in req}),
           "evidence_caps": {"per_page": PAGE_CAP, "total": TOTAL_CAP, "min_page": MIN_PAGE},
           "missing_arms": {"freshcache_without_l1": "no preserved held-out outputs exist for this arm; not generated (would be a new policy run)"}},
          open(IN/"input_manifest.json", "w"), indent=1)
print(json.dumps(dict(stats), indent=1))
