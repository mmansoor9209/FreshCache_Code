"""Build step of the L2 verifier held-out answer audit under the replay engine: FreshCache vs L2Verify (gamma frozen at 0.55) on the 400
held-out audit requests, using the reported replay rows (verifier/out/test_rows.pkl), the reported fresh / FreshCache keys
(results/audit_keys_reported.json), and evidence blocks built with the snapshot rule."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, re, pickle, collections, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
E = C.load("reported"); exp, p2 = E["exp"], E["p2"]; h, ctx_for = p2.h, p2.ctx_for
import answer_grading as e2e
G = json.load(open((OUTPUT_ROOT / "l2_verifier/out/gamma_selection.json")))["selected_gamma"]
rows = pickle.load(open(C.W/"verifier/out/test_rows.pkl", "rb")); FC, LV = rows["FreshCache"], rows[f"L2Verify_g{G}"]
K = json.load(open(C.W/"results/audit_keys_reported.json"))["keys"]["heldout"]
S = json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]
man = {}
for m in exp.load_jsonl(exp.MANIFEST_FILE):
    if m.get("run_id") == "run_00" and m["url_hash"] not in man: man[m["url_hash"]] = {"title": m.get("title", "") or "", "snippet": m.get("snippet", "") or ""}
def evblock(ev):
    lines = []
    for uh, tv in ev:
        mt = man.get(uh, {}); b = re.sub(r"\s+", " ", (e2e.load_snapshot_text(uh, p2.snapshot_round_on_disk(uh, tv)) or "")).strip()[:2000]
        blk = f"{len(lines)+1}. {mt.get('title','').strip()}"
        if mt.get("snippet"): blk += f"\n   {mt['snippet'].strip()}"
        if len(b) >= 200: blk += f"\n   PAGE TEXT: {b}"
        lines.append(blk)
    return "\n".join(lines) or "(no stored evidence)"
out, tasks, suff = [], {}, {}
for d in S:
    q = d["query_id"]; a, b = FC[q], LV[q]
    assert a["tier"] == K[q]["freshcache"][1], (q, a["tier"], K[q]["freshcache"])   # the verifier's FreshCache arm equals the reported audit arm
    ev_a = [tuple(x) for x in (a.get("stored_answer_ev") if a["tier"] == "L1" else a.get("ev")) or []]
    ev_b = [tuple(x) for x in (b.get("stored_answer_ev") if b["tier"] == "L1" else b.get("ev")) or []]
    ctx_b = ctx_for(tuple(ev_b)); gq_b = b.get("gen_query", d["query"]); key_b = ["gen", h(ctx_b), h(gq_b)]
    tasks[tuple(key_b)] = {"key": key_b, "question": gq_b, "context": ctx_b}
    out.append({"query_id": q, "query": d["query"], "gold": d["gold"], "fc_tier": a["tier"], "lv_tier": b["tier"], "lv_verdict": b.get("verified"),
                "serving_unchanged": int(a["tier"] == b["tier"] and ev_a == ev_b), "fc_key": K[q]["freshcache"][0], "fresh_key": K[q]["fresh"][0], "lv_key": key_b})
    suff[(q, "fc")] = {"query_id": q, "arm": "fc", "query": d["query"], "evidence": evblock(ev_a)}
    suff[(q, "lv")] = {"query_id": q, "arm": "lv", "query": d["query"], "evidence": evblock(ev_b)}
V = C.W/"verifier/audit"; V.mkdir(exist_ok=True)
with open(V/"audit_sample.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(list(tasks.values()), open(V/"gen_tasks.json", "w"))
with open(V/"suff_tasks.jsonl", "w", encoding="utf-8") as f:
    for t in suff.values(): f.write(json.dumps(t, ensure_ascii=False) + "\n")
print("changed serving", sum(1 - r["serving_unchanged"] for r in out), "FC tiers", dict(collections.Counter(r["fc_tier"] for r in out)),
      "LV tiers", dict(collections.Counter(r["lv_tier"] for r in out)), "lv contexts", len(tasks), flush=True)
