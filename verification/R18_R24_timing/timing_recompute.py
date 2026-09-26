"""R18/R24 recompute from saved timing records. Read-only; writes only into this folder."""
import csv, json, pathlib, sys, os, collections, datetime, numpy as np
ROOT = pathlib.Path("<PROJECT_ROOT>"); D = ROOT/"validation/latency_e2e_direct"; E = ROOT/"validation/latency_e2e"; OUT = pathlib.Path(__file__).parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
R = {"utc": datetime.datetime.utcnow().isoformat()+"Z"}
def q(v, p):  # run_direct.py:70-72 / compare.py:23-27 convention
    s = sorted(v); return float(s[min(len(s)-1, int(round(p*(len(s)-1))))])
def qw(v, p, w):
    v = np.asarray(v, float); o = np.argsort(v); v, w = v[o], np.asarray(w, float)[o]; c = np.cumsum(w)/w.sum()
    return float(v[np.searchsorted(c, p, side="left").clip(0, len(v)-1)])
POL = ["NoCache","SemanticTTL_k1_16","FreshCache_NoL1","FreshCache"]
rows = list(csv.DictReader(open(D/"direct_per_request.csv")))
spec = json.load(open(D/"direct_sample_spec.json"))
ids = spec["sample_query_ids"]
per_pol_ids = {p: [r["query_id"] for r in rows if r["policy"] == p] for p in POL}
R["sample"] = {"n_ids": len(ids), "ids": ids, "rows": len(rows), "same_ids_every_policy": all(sorted(per_pol_ids[p]) == sorted(ids) for p in POL), "same_order_every_policy": all(per_pol_ids[p] == ids for p in POL),
    "class_counts": dict(collections.Counter(r["fc"] for r in rows if r["policy"] == "NoCache")), "spec_keys": list(spec), "specs_policies": list(spec.get("specs", {})),
    "spec_fields_per_request": list(spec["specs"]["FreshCache"][0].keys())}
import experiment as exp
recs = exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE), exp.load_jsonl(exp.PARAPHRASE_FILE))
cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in recs}
split = json.load(open(ROOT/"validation/heldout_baseline_tuning/split.json")); TEST = set(split["test_clusters"]); VAL = set(split["validation_clusters"])
R["sample"]["in_test_clusters"] = sum(1 for i in ids if cid[i] in TEST); R["sample"]["in_validation_clusters"] = sum(1 for i in ids if cid[i] in VAL)
R["sample"]["sampling_rule"] = "prep_sample.py:181-200: stratified 8 per class from the second half of the seed-42 zipf_uniform stream, requests with a live URL for every own URL; NOT filtered by the cluster split"
# direct summaries
W = {"TIMELESS": 6929, "SLOW": 8137, "MEDIUM": 7626, "FAST": 8155, "REAL_TIME": 354}; TOT = sum(W.values())
direct = {}; rew = []
for p in POL:
    sub = [r for r in rows if r["policy"] == p]; v = [float(r["total_ms"]) for r in sub]
    cls = [r["fc"] for r in sub]; cnt = collections.Counter(cls); w = [(W[c]/TOT)/(cnt[c]/len(sub)) for c in cls]
    direct[p] = {"N": len(v), "mean_ms": round(float(np.mean(v)),4), "p50_ms": round(q(v,.5),4), "p95_ms": round(q(v,.95),4), "median_numpy": round(float(np.median(v)),4),
        "p95_numpy_linear": round(float(np.percentile(v,95)),4), "class_reweighted_mean_ms": round(float(np.average(v, weights=w)),4), "class_reweighted_p50": round(qw(v,.5,w),4), "class_reweighted_p95": round(qw(v,.95,w),4),
        "paths": dict(collections.Counter(r["path"] for r in sub)), "live_searches_timed": sum(1 for r in sub if float(r["search_api"]) > 0), "fetches": sum(int(r["n_fetched"]) for r in sub), "cache_reads": sum(int(r["n_from_cache"]) for r in sub),
        "rows_with_zero_answer_chars": sum(1 for r in sub if int(r["answer_chars"]) == 0), "rows_with_fetch_but_zero_fetch_time": sum(1 for r in sub if int(r["n_fetched"]) > 0 and float(r["page_fetch"]) == 0),
        "class_means_ms": {c: round(float(np.mean([float(r["total_ms"]) for r in sub if r["fc"] == c])),4) for c in W}}
    for c in W: rew.append({"policy": p, "class": c, "n_sample": cnt[c], "workload_count": W[c], "workload_share": round(W[c]/TOT,6), "sample_share": round(cnt[c]/len(sub),4), "weight_per_request": round((W[c]/TOT)/(cnt[c]/len(sub)),6), "class_mean_ms": direct[p]["class_means_ms"][c]})
R["direct"] = direct
R["direct"]["percentile_convention"] = "sorted values s; index = round(p*(n-1)) clipped to n-1; p50 = s[round(0.5*39)] = s[20] (21st of 40, upper middle, not the average of s[19], s[20]); p95 = s[round(0.95*39)] = s[37] (38th of 40). run_direct.py:70-72; compare.py:23-27"
R["direct"]["reweighting_formula"] = "mean_w = sum_i w_i x_i / sum_i w_i with w_i = (W_c/31201)/(n_c/40) for request i of class c; equivalently sum_c (W_c/31201) * classmean_c; W from compare.py:16-17 = replay request counts per class"
R["direct"]["reweighted_check_as_class_mean_sum"] = {p: round(sum(W[c]/TOT*direct[p]["class_means_ms"][c] for c in W),4) for p in POL}
with open(OUT/"direct_timing_recompute.csv","w",newline="") as fh:
    w = csv.writer(fh); w.writerow(["policy","N","mean_ms","p50_ms","p95_ms","manuscript_mean","manuscript_p50","manuscript_p95","class_reweighted_mean_ms","numpy_median","numpy_p95_linear"])
    M = {"NoCache": (3255.3,3014.4,5050.3), "FreshCache": (1214.4,563.8,3107.3), "FreshCache_NoL1": (1238.2,743.0,3558.3), "SemanticTTL_k1_16": (703.3,13.5,3832.5)}
    for p in POL: d = direct[p]; w.writerow([p, d["N"], d["mean_ms"], d["p50_ms"], d["p95_ms"], *M[p], d["class_reweighted_mean_ms"], d["median_numpy"], d["p95_numpy_linear"]])
with open(OUT/"class_reweighting.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rew[0])); w.writeheader(); w.writerows(rew)
    w2 = csv.writer(fh); w2.writerow([]); w2.writerow(["policy","reweighted_mean_ms","expected"]);
    for p, e in (("NoCache",3271.4),("FreshCache",885.9),("FreshCache_NoL1",""),("SemanticTTL_k1_16","")): w2.writerow([p, direct[p]["class_reweighted_mean_ms"], e])
# stage pools
b = json.load(open(E/"stage_b_gpu_timings.json")); c1 = json.load(open(E/"stage_c1_ann_timings.json")); c2 = json.load(open(E/"stage_c2_gate_timings.json")); dl = json.load(open(E/"stage_d_live_timings.json"))
raw = [json.loads(l) for l in open(E/"stage_d_live_raw.jsonl") if l.strip()]
pools = [
 {"stage":"embedding","file":"validation/latency_e2e/stage_b_gpu_timings.json[embed].samples_ms","n":len(b["embed"]["samples_ms"]),"units":"ms per query, BGE-M3 batch 1 GPU","sampling":"with replacement, one draw per request that embeds (stage_e_compose.py:108)","source":"separate microbenchmark (stage_b_gpu.py), 200 timed calls, seed 42, GPU index 4 on host node1"},
 {"stage":"llm_generation","file":"stage_b_gpu_timings.json[generate].samples_ms","n":len(b["generate"]["samples_ms"]),"units":"ms per generation, Llama-3.2-3B greedy max_new_tokens 80","sampling":"with replacement, one draw per generating request (:134)","source":"separate microbenchmark (stage_b_gpu.py) on real stored contexts"},
 {"stage":"l1/l2 lookup (ANN)","file":"stage_c1_ann_timings.json[ann][size].samples_ms","n":{k:len(v["samples_ms"]) for k,v in c1["ann"].items()},"units":"ms per FAISS IndexFlatIP search, k=5, sizes 1000/5000/10000/20000/31000","sampling":"draw from nearest measured size, rescaled by log-linear interpolated p50 at the request's live index size (:74-88)","source":"separate microbenchmark (stage_c1_ann.py), 200 trials per size after 20 warm-ups"},
 {"stage":"entity gate","file":"stage_c2_gate_timings.json[entity].samples_ms","n":len(c2["entity"]["samples_ms"]),"units":"ms","sampling":"with replacement, added to every L1 lookup of C18 policies (:112-114)","source":"separate microbenchmark (stage_c2_gates.py), CPU"},
 {"stage":"lexical gate","file":"stage_c2_gate_timings.json[lexical].samples_ms","n":len(c2["lexical"]["samples_ms"]),"units":"ms","sampling":"as entity","source":"same"},
 {"stage":"l3_processing","file":"stage_c2_gate_timings.json[l3_lookup].samples_ms","n":len(c2["l3_lookup"]["samples_ms"]),"units":"ms per reused page","sampling":"one draw per reused page, summed per request (:117-124)","source":"same"},
 {"stage":"search_api","file":"stage_d_live_raw.jsonl kind=search ok=true","n":sum(1 for r in raw if r["kind"]=="search" and r["ok"]),"units":"ms per live Serper call","sampling":"with replacement, one draw per search (:133)","source":"separate live microbenchmark (stage_d_live.py), 40 attempted / 40 ok; NOT the 40-request direct run"},
 {"stage":"page_fetch","file":"stage_d_live_raw.jsonl kind=fetch ok=true","n":sum(1 for r in raw if r["kind"]=="fetch" and r["ok"]),"units":"ms per live cold GET","sampling":"one draw per fetched page, summed per request (:125-132)","source":"separate live microbenchmark, 60 attempted / 53 ok (7 failures excluded from the pool)"}]
R["pools"] = pools
with open(OUT/"stage_pool_inventory.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["stage","file","n","units","sampling","source"]); w.writeheader(); w.writerows(pools)
# Monte-Carlo rerun (aggregate only), transcribing stage_e_compose.py with the same seed
rng = np.random.default_rng(42)
S_embed = np.array(b["embed"]["samples_ms"]); S_gen = np.array(b["generate"]["samples_ms"]); S_ent = np.array(c2["entity"]["samples_ms"]); S_lex = np.array(c2["lexical"]["samples_ms"]); S_l3 = np.array(c2["l3_lookup"]["samples_ms"])
S_search = np.array([r["ms"] for r in raw if r["kind"]=="search" and r["ok"]]); S_fetch = np.array([r["ms"] for r in raw if r["kind"]=="fetch" and r["ok"]])
ann_sizes = np.array(sorted(int(k) for k in c1["ann"])); ann_samples = {int(k): np.array(v["samples_ms"]) for k,v in c1["ann"].items()}; ann_p50 = np.array([c1["ann"][str(n)]["p50_ms"] for n in ann_sizes])
def ann_draw(sizes):
    sizes = np.clip(sizes, 1, ann_sizes.max()); mu = np.interp(np.log(sizes), np.log(ann_sizes), ann_p50)
    near = ann_sizes[np.abs(np.log(ann_sizes)[None,:]-np.log(sizes)[:,None]).argmin(axis=1)]; out = np.empty(len(sizes))
    for n in np.unique(near):
        m = near == n; pool = ann_samples[int(n)]; draws = pool[rng.integers(0, len(pool), m.sum())]; out[m] = draws*(mu[m]/c1["ann"][str(int(n))]["p50_ms"])
    return out
mc = {}; rec = json.load(open(E/"latency_results.json"))["policies"]
for pol in POL:
    ops = list(csv.DictReader(open(E/f"opcounts_{pol}.csv"))); n = len(ops); iv = lambda k: np.array([int(o[k]) for o in ops])
    embed, l1l, l2l, l1h, l3l, fetches, search, gen = iv("embed"), iv("l1_lookup"), iv("l2_lookup"), iv("l1_hit"), iv("l3_lookups"), iv("fetches"), iv("search"), iv("generate")
    writes_l1 = ((l1l==1)&(l1h==0)).astype(int); n1 = np.concatenate([[0], np.cumsum(writes_l1)[:-1]]); writes_l2 = (l2l==1).astype(int); n2 = np.concatenate([[0], np.cumsum(writes_l2)[:-1]])
    c = {k: np.zeros(n) for k in ("embedding","l1_lookup","l2_lookup","l3_processing","search_api","page_fetch","llm_generation")}
    c["embedding"] = embed*S_embed[rng.integers(0, len(S_embed), n)]
    if l1l.any():
        c["l1_lookup"] = l1l*ann_draw(np.maximum(n1,1))
        if pol == "FreshCache": c["l1_lookup"] += l1l*(S_ent[rng.integers(0,len(S_ent),n)] + S_lex[rng.integers(0,len(S_lex),n)])
    if l2l.any(): c["l2_lookup"] = l2l*ann_draw(np.maximum(n2,1))
    tot_l3 = int(l3l.sum())
    if tot_l3:
        draws = S_l3[rng.integers(0,len(S_l3),tot_l3)]; acc, k = np.zeros(n), 0
        for i, cnt in enumerate(l3l):
            if cnt: acc[i] = draws[k:k+cnt].sum(); k += cnt
        c["l3_processing"] = acc
    tot_f = int(fetches.sum())
    if tot_f:
        draws = S_fetch[rng.integers(0,len(S_fetch),tot_f)]; acc, k = np.zeros(n), 0
        for i, cnt in enumerate(fetches):
            if cnt: acc[i] = draws[k:k+cnt].sum(); k += cnt
        c["page_fetch"] = acc
    c["search_api"] = search*S_search[rng.integers(0,len(S_search),n)]; c["llm_generation"] = gen*S_gen[rng.integers(0,len(S_gen),n)]
    total = sum(c.values())
    mc[pol] = {"N": n, "mean_ms": round(float(total.mean()),4), "p50_ms": round(q(total,.5),4), "p95_ms": round(q(total,.95),4), "recorded_mean": rec[pol]["total"]["mean_ms"], "recorded_p50": rec[pol]["total"]["p50_ms"], "recorded_p95": rec[pol]["total"]["p95_ms"],
        "op_counts": {"search_calls": int(search.sum()), "l1_hits": int(l1h.sum()), "l2_hits": int(iv("l2_hit").sum()), "fetches": int(fetches.sum()), "generations": int(gen.sum())}, "mean_matches_record_to_1e-6": abs(float(total.mean())-rec[pol]["total"]["mean_ms"]) < 1e-6}
R["monte_carlo"] = {"seed": 42, "generator": "numpy.random.default_rng(42), one draw per operation, single pass (no replicates)", "population": "31,201 replay requests per policy from opcounts_<policy>.csv (stage_a_opcounts.py replay)", "results": mc,
    "constants_role": "stage_e_compose.py:182-185 writes 500/800/2000/5/5/5 into latency_results.json under 'simulator_constants_for_reference' only; they are never read into any draw"}
json.dump(R["monte_carlo"], open(OUT/"monte_carlo_recompute.json","w"), indent=1)
json.dump(R, open(OUT/"timing_results.json","w"), indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k != "pools"}, indent=1, default=str)[:9000])
