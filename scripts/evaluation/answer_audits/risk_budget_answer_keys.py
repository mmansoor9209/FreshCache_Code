"""Answer keys of budget configurations A-D on the 400 held-out audit requests.
Transcribed from outputs/parameter_calibration/answer_quality/risk_budget_audit_records.py; the keyset configuration must reproduce audit_records.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); exp, p2 = E["exp"], E["p2"]; ctx_for, h = p2.ctx_for, p2.h
PC = (OUTPUT_ROOT / "parameter_calibration")
sel = json.load(open(PC/"budgets/frozen_config.json"))["selected_eps"]; mhat = json.load(open(PC/"heldout/heldout_results.json"))["m_content_mle"]
if ENG == "reported":   # the budget selection is re-run under the replay engine; use its selected eps if it differs (r5 output)
    b = json.load(open(C.W/"results/budget_reported.json")); s = b["selected"]; sel = {"answer": s["eps_answer"], "url_list": s["eps_url_list"], "content": s["eps_content"]}
S = json.load(open(PC/"answer_quality/audit_sample.json"))["requests"]; test = C.streams(E)["heldout"]
pub_m = dict(exp.TIER_MULT); pub_e = (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT)
CFG = {"A_original": (pub_m, pub_e), "B_mult_only": ({**pub_m, "content": mhat}, pub_e),
       "C_budget_only": (pub_m, (sel["answer"], sel["url_list"], sel["content"])), "D_full": ({**pub_m, "content": mhat}, (sel["answer"], sel["url_list"], sel["content"]))}
rs = {}
try:
    for n, (m, e) in CFG.items():
        exp.TIER_MULT.update(m); exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = e; rs[n] = p2.replay(test, {}, "FreshCache")[1]
finally:
    exp.TIER_MULT.update(pub_m); exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = pub_e
rows, tasks = [], {}
for d in S:
    q, t = d["query_id"], d["t"]; fctx = ctx_for(tuple((u, t) for u in rs["A_original"][q]["own"])); fk = ["fresh", h(fctx), h(d["query"])]; tasks[tuple(fk)] = {"key": fk, "question": d["query"], "context": fctx}
    rec = {"query_id": q, "gold": d["gold"], "query": d["query"], "fresh_key": fk}
    for n in CFG:
        p = rs[n][q]; rec[f"{n}_tier"] = p["tier"]
        if p["tier"] == "L1": c = ctx_for(tuple(p["stored_answer_ev"])); k = ["stored", h(c), h(p["stored_answer_query"])]; qq = p["stored_answer_query"]
        else: c = ctx_for(tuple(p["ev"])); k = ["gen", h(c), h(d["query"])]; qq = d["query"]
        rec[f"{n}_key"] = k; tasks[tuple(k)] = {"key": k, "question": qq, "context": c}
    rows.append(rec)
pub = {r["query_id"]: r for r in json.load(open(PC/"answer_quality/audit_records.json"))}
cmp = {n: [r["query_id"] for r in rows if r[f"{n}_key"] != pub[r["query_id"]][f"{n}_key"]] for n in ["fresh"] + list(CFG)}
json.dump({"engine": ENG, "eps_used": sel, "records": rows, "changed_vs_recorded": cmp}, open(C.W/"results"/f"budget_keys_{ENG}.json", "w"), indent=1)
json.dump(list(tasks.values()), open(C.W/"results"/f"budget_gen_tasks_{ENG}.json", "w"))
for n, v in cmp.items(): print(ENG, n, "changed", len(v), v[:10], flush=True)
