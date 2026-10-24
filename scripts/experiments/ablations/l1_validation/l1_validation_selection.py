"""Score every cell and apply the pre-registered selection rule (App. A, stricter L1 on validation). Validation outputs only. Freezes the selected rule
(outputs/l1_validation/config/frozen_l1_rule.json); writes outputs/l1_validation/results/validation_grid_results.csv and selection.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, csv, glob, hashlib, pathlib, collections, datetime, math, os; OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
W = ((OUTPUT_ROOT / "l1_validation")); OLD = (OUTPUT_ROOT / "l1_precision"); O = W/"outputs/gpu"; RES = W/"results"; RES.mkdir(parents=True, exist_ok=True)
P = json.load(open(W/"config/prereg.json")); assert hashlib.sha256((W/"config/prereg.json").read_bytes()).hexdigest() == open(W/"config/prereg.sha256").read().strip()
def jl(p): p = pathlib.Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
def h(s): return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
def wilson(k, n, z=1.96):
    if not n: return [None, None]
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n); m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return [round(100*(c-m)/d, 2), round(100*(c+m)/d, 2)]
# jury
eq = collections.defaultdict(dict); src = collections.Counter()
for j in ("llama3b", "llama8b", "qwen7b", "mistral7b"):
    for f, tag in ((OLD/f"validation_equiv_{j}.jsonl", "reused"), (O/f"equiv_{j}.jsonl", "new")):
        for r in jl(f): eq[r["pair_id"]][j] = r["label"]; src[tag] += 1
def verdict(pid):
    v = [l for l in eq.get(pid, {}).values() if l in ("SAME", "DIFFERENT")]
    if len(v) < 2: return "UNJUDGED"
    c = collections.Counter(v); return "JURY_TIE" if c["SAME"] == c["DIFFERENT"] else c.most_common(1)[0][0]
# answers and correctness (manuscript rule)
ans = {tuple(r["key"]): r["answer"] for r in jl(OLD/"validation_answers_llama3b.jsonl")}
for r in jl(O/"answers_llama3b.jsonl"): ans[tuple(r["key"])] = r["answer"]
J = {}
for j in ("llama8b", "qwen7b"):
    J[j] = {tuple(r["sig"]): r["label"] for r in jl(OLD/f"validation_judge_{j}.jsonl")}
    for r in jl(O/f"judge_{j}.jsonl"): J[j][tuple(r["sig"])] = r["label"]
ABST = ("i don't know", "i do not know", "cannot be determined", "not enough information", "insufficient", "unable to determine", "no information")
def lab(q, gold, key):
    a = ans.get(tuple(key))
    if a is None or not gold: return None
    if any(p in a.strip().lower() for p in ABST): return "ABSTAIN"
    s = (h(q), h(gold), h(a)); l1, l2 = J["llama8b"].get(s), J["qwen7b"].get(s)
    if l1 is None or l2 is None: return None
    return "CORRECT" if l1 == "CORRECT" and l2 == "CORRECT" else "WRONG"
AUD = {d["query_id"]: d for d in json.load(open(OLD/"validation_audit_sample.json"))["requests"]}
fresh_aud = {q: lab(d["query"], d["gold"], d["fresh_key"]) for q, d in AUD.items()}
rows = []
for f in sorted(glob.glob(str(W/"outputs/replay/s*.json"))):
    d = json.load(open(f)); r = dict(d["row"]); hits = d["hits"]; n = len(hits)
    vs = collections.Counter(verdict(x["pair_id"]) for x in hits)
    k = vs["DIFFERENT"] + vs["JURY_TIE"]
    r.update({"jury_same": vs["SAME"], "jury_different": vs["DIFFERENT"], "jury_ties": vs["JURY_TIE"], "jury_unjudged": vs["UNJUDGED"],
              "mismatch_pct": round(100*k/n, 4) if n else None, "mismatch_wilson95": wilson(k, n),
              "mismatch_ties_excluded_pct": round(100*vs["DIFFERENT"]/(vs["DIFFERENT"]+vs["SAME"]), 4) if vs["DIFFERENT"]+vs["SAME"] else None})
    ev = harm = st_c = fr_c = 0
    for x in hits:
        s, fr = lab(x["question"], x["gold"], x["stored_key"]), lab(x["question"], x["gold"], x["fresh_key"])
        if s is None or fr is None: continue
        ev += 1; harm += (fr == "CORRECT" and s == "WRONG"); st_c += s == "CORRECT"; fr_c += fr == "CORRECT"
    r.update({"hits_evaluable": ev, "hit_harm": harm, "hit_harm_rate_pct": round(100*harm/n, 4) if n else None, "stored_correct": st_c, "fresh_correct_on_hits": fr_c})
    aw = sum(1 for q, v in d["audit"].items() if fresh_aud[q] == "CORRECT" and lab(AUD[q]["query"], AUD[q]["gold"], v["key"]) == "WRONG")
    un = sum(1 for q, v in d["audit"].items() if lab(AUD[q]["query"], AUD[q]["gold"], v["key"]) is None)
    r.update({"audit_wai": aw, "audit_unlabelled": un, "audit_l1": sum(1 for v in d["audit"].values() if v["tier"] == "L1")})
    rows.append(r)
pub = next(r for r in rows if r["cell"] == "s0.80_noguard")
C1, C2, C3, C4 = 0.5*pub["l1_hits"], pub["search_saved_pct"] - 0.10, pub["hit_harm"], pub["audit_wai"]
gorder = {tuple(g): i for i, g in enumerate(P["grid"]["guard_subsets"])}
for r in rows:
    r["C1"] = r["l1_hits"] >= C1; r["C2"] = r["search_saved_pct"] >= C2 - 1e-9; r["C3"] = r["hit_harm"] <= C3; r["C4"] = r["audit_wai"] <= C4; r["eligible"] = r["C1"] and r["C2"] and r["C3"] and r["C4"]
el = sorted([r for r in rows if r["eligible"]], key=lambda r: (r["mismatch_pct"], r["hit_harm_rate_pct"], -r["l1_hits"], r["cosine_floor"], len(r["guards"]), gorder[tuple(r["guards"])]))
for i, r in enumerate(el): r["rank"] = i + 1
cols = ["cell", "cosine_floor", "guards", "l1_hits", "l2_hits", "search_calls", "search_saved_pct", "generations", "generation_saved_pct", "jury_same", "jury_different", "jury_ties", "jury_unjudged",
        "mismatch_pct", "mismatch_wilson95", "mismatch_ties_excluded_pct", "cross_cluster_hits", "hits_evaluable", "stored_correct", "fresh_correct_on_hits", "hit_harm", "hit_harm_rate_pct",
        "audit_l1", "audit_wai", "audit_unlabelled", "drift_pct", "coverage_pct", "C1", "C2", "C3", "C4", "eligible", "rank"]
with open(RES/"validation_grid_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r.get(c, "") for c in cols}) for r in rows]
out = {"prereg_sha256": open(W/"config/prereg.sha256").read().strip(), "jury_label_sources": dict(src), "recorded_reference": pub,
       "thresholds": {"C1_min_l1_hits": C1, "C2_min_search_saved_pct": round(C2, 4), "C3_max_hit_harm": C3, "C4_max_audit_wai": C4},
       "n_cells": len(rows), "n_eligible": len(el), "ranking": [{k: r[k] for k in ("rank", "cell", "l1_hits", "mismatch_pct", "hit_harm", "hit_harm_rate_pct", "search_saved_pct", "generation_saved_pct")} for r in el]}
if el:
    s = el[0]
    frozen = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "prereg_sha256": out["prereg_sha256"], "selected_on": "validation clusters only; held-out never read",
              "rule": {"cosine_floor": s["cosine_floor"], "jaccard_floor": 0.30, "answer_type": "as_implemented", "strict_subset_rule": "on", "entity_match": "unchanged",
                       "guards": s["guards"], "guards_sha256": "952e8c9cb59a796328481bc8cc23d7dc9042db09cd96980d6be9fecc972c3873"},
              "validation": {k: s[k] for k in ("l1_hits", "mismatch_pct", "mismatch_wilson95", "hit_harm", "hit_harm_rate_pct", "search_saved_pct", "generation_saved_pct", "audit_wai", "drift_pct", "coverage_pct")},
              "recorded_validation": {k: pub[k] for k in ("l1_hits", "mismatch_pct", "hit_harm", "hit_harm_rate_pct", "search_saved_pct", "generation_saved_pct", "audit_wai")},
              "status": "FROZEN; not evaluated on held-out/test data"}
    json.dump(frozen, open(W/"config/frozen_l1_rule.json", "w"), indent=1)
    open(W/"config/frozen_l1_rule.sha256", "w").write(hashlib.sha256((W/"config/frozen_l1_rule.json").read_bytes()).hexdigest() + "\n")
    out["selected"] = frozen
json.dump(out, open(RES/"selection.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "recorded_reference"}, indent=1, default=str)[:4000])
