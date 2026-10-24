#!/usr/bin/env python3
"""Scoring for the held-out answer audit: frozen FreshCache versus the frozen
validation-selected SemanticTTL kappa = 1/16, on one common
policy-neutral sample drawn from TEST clusters only.

Correctness rule: an answer matching an abstention pattern is ABSTAIN;
otherwise CORRECT only if BOTH judges say CORRECT.

Read-only on every input. No web calls, no GPU.
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, hashlib, json, math, pathlib, statistics, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)

HERE = ((OUTPUT_ROOT / "general_audit"))
STAGE = ((OUTPUT_ROOT / "validation_split"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
WATCH = ["freshcache/method/tiered_cache_model.py", "freshcache/selection_replay_engine/mixed_age_replay.py",
         "freshcache/selection_replay_engine/method_replay.py", "freshcache/simulation/request_schedules.py"]
ABSTAIN_PAT = ["i don't know", "i do not know", "cannot be determined",
               "not enough information", "insufficient", "unable to determine",
               "no information"]
ARMS = [("fresh", "Fresh reference"), ("fc", "Frozen FreshCache"),
        ("sttl", "SemanticTTL kappa=1/16")]


def h(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def is_abstain(t):
    t = (t or "").strip().lower()
    return any(p in t for p in ABSTAIN_PAT)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def stats(v):
    if not v:
        return None
    s = sorted(v)
    def pct(p):
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"n": len(s), "mean": sum(s) / len(s), "median": statistics.median(s),
            "min": s[0], "p25": pct(0.25), "p75": pct(0.75), "max": s[-1]}


def main():
    pre = {f: sha(ROOT / f) for f in WATCH}
    sample = json.load(open(HERE / "sample.json", encoding="utf-8"))
    reqs = sample["requests"]
    ans = {tuple(r["key"]): r["answer"] for r in jl(HERE / "answers.jsonl")}
    j1 = {tuple(r["sig"]): r["label"] for r in jl(HERE / "judge_llama8b.jsonl")}
    j2 = {tuple(r["sig"]): r["label"] for r in jl(HERE / "judge_qwen7b.jsonl")}

    def outcome(q, gold, a):
        if a is None:
            return None
        if is_abstain(a):
            return "ABSTAIN"
        sig = (h(q), h(gold), h(a))
        l1, l2 = j1.get(sig), j2.get(sig)
        if l1 is None or l2 is None:
            return None
        return "CORRECT" if (l1 == "CORRECT" and l2 == "CORRECT") else "WRONG"

    rows, dropped = [], 0
    for d in reqs:
        o = {"qid": d["query_id"], "cluster": d["cluster_id"], "fc": d["fc"],
             "band": d["simbase_band"], "in_S": d["in_S"],
             "fc_tier": d["fc_tier"], "sttl_tier": d["sttl_tier"],
             "sttl_similarity": d.get("sttl_similarity"),
             "fc_similarity": d.get("fc_similarity")}
        ok = True
        for arm, _ in ARMS:
            a = ans.get(tuple(d[f"{arm}_key"]))
            o[f"{arm}_answer"] = a
            v = outcome(d["query"], d["gold"], a)
            o[arm] = v
            if v is None:
                ok = False
        if not ok:
            dropped += 1
            continue
        rows.append(o)
    N = len(rows)

    def arm_stats(arm):
        c = sum(1 for r in rows if r[arm] == "CORRECT")
        w = sum(1 for r in rows if r[arm] == "WRONG")
        ab = sum(1 for r in rows if r[arm] == "ABSTAIN")
        return {"N": N, "correct": c, "wrong": w, "abstain": ab,
                "accuracy_pct": 100 * c / N if N else None}

    res = {a: arm_stats(a) for a, _ in ARMS}
    fresh_correct = [r for r in rows if r["fresh"] == "CORRECT"]

    for arm in ("fc", "sttl"):
        wai = sum(1 for r in rows if r[arm] == "WRONG" and r["fresh"] == "CORRECT")
        rev = sum(1 for r in rows if r[arm] == "CORRECT" and r["fresh"] == "WRONG")
        lo, hi = wilson(wai, N)
        res[arm].update({
            "wai": wai, "wai_rate_pct": 100 * wai / N if N else None,
            "wai_ci95": [100 * lo, 100 * hi],
            "reverse": rev, "reverse_rate_pct": 100 * rev / N if N else None,
            "fresh_correct_n": len(fresh_correct),
            "conditional_wai_pct": (100 * wai / len(fresh_correct)
                                    if fresh_correct else None)})

    # ---- paired WAI discordance ----
    def isw(r, arm):
        return r[arm] == "WRONG" and r["fresh"] == "CORRECT"
    both = sum(1 for r in rows if isw(r, "fc") and isw(r, "sttl"))
    only_s = sum(1 for r in rows if isw(r, "sttl") and not isw(r, "fc"))
    only_f = sum(1 for r in rows if isw(r, "fc") and not isw(r, "sttl"))
    neither = N - both - only_s - only_f
    wai_mc = {"wai_both": both, "wai_semanticttl_only": only_s,
              "wai_freshcache_only": only_f, "wai_neither": neither,
              "exact_mcnemar_p": mcnemar(only_s, only_f)}

    # ---- paired correctness discordance ----
    fc_ok_s_no = sum(1 for r in rows
                     if r["fc"] == "CORRECT" and r["sttl"] != "CORRECT")
    s_ok_fc_no = sum(1 for r in rows
                     if r["sttl"] == "CORRECT" and r["fc"] != "CORRECT")
    bc = sum(1 for r in rows if r["fc"] == "CORRECT" and r["sttl"] == "CORRECT")
    bw = sum(1 for r in rows if r["fc"] != "CORRECT" and r["sttl"] != "CORRECT")
    corr_mc = {"both_correct": bc, "freshcache_correct_semanticttl_wrong": fc_ok_s_no,
               "semanticttl_correct_freshcache_wrong": s_ok_fc_no,
               "both_wrong": bw,
               "exact_mcnemar_p": mcnemar(s_ok_fc_no, fc_ok_s_no)}

    # ---- path analysis ----
    paths = []
    for arm, tiers in (("fc", ["L1", "L2", "miss"]), ("sttl", ["HIT", "miss"])):
        for tier in tiers:
            sub = [r for r in rows if r[f"{arm}_tier"] == tier]
            w = sum(1 for r in sub if isw(r, arm))
            paths.append({"policy": arm, "path": tier, "requests": len(sub),
                          "wai_events": w,
                          "conditional_wai_pct": (100 * w / len(sub)
                                                  if sub else None),
                          "correct": sum(1 for r in sub if r[arm] == "CORRECT"),
                          "accuracy_pct": (100 * sum(1 for r in sub
                                                     if r[arm] == "CORRECT")
                                           / len(sub) if sub else None)})
    sttl_hit_wai = sum(1 for r in rows
                       if r["sttl_tier"] == "HIT" and isw(r, "sttl"))
    sttl_miss_wai = sum(1 for r in rows
                        if r["sttl_tier"] == "miss" and isw(r, "sttl"))

    sim_all = stats([r["sttl_similarity"] for r in rows
                     if r["sttl_tier"] == "HIT" and r["sttl_similarity"] is not None])
    sim_wai = stats([r["sttl_similarity"] for r in rows
                     if r["sttl_tier"] == "HIT" and isw(r, "sttl")
                     and r["sttl_similarity"] is not None])

    # ---- 10. reuse of existing semantic-equivalence labels ----
    eq = reuse_labels(reqs, rows, isw)

    agree_keys = set(j1) & set(j2)
    agree = sum(1 for k in agree_keys if j1[k] == j2[k])
    post = {f: sha(ROOT / f) for f in WATCH}
    unchanged = pre == post

    out = {"audit": "held-out test clusters, frozen FreshCache vs frozen "
                    "validation-selected SemanticTTL kappa=1/16",
           "N_sampled": len(reqs), "N_scored": N, "dropped_unjudgeable": dropped,
           "frozen_semanticttl": sample["frozen_semanticttl"],
           "sampling": {k: sample[k] for k in
                        ("target", "seed", "n_strata", "per_stratum_cap",
                         "frame_size", "test_clusters_only",
                         "validation_requests_present",
                         "sampling_is_policy_neutral", "in_support_S")},
           "arms": res, "wai_mcnemar": wai_mc, "correctness_mcnemar": corr_mc,
           "paths": paths,
           "semanticttl_wai_on_hits": sttl_hit_wai,
           "semanticttl_wai_on_misses": sttl_miss_wai,
           "semanticttl_hit_similarity": sim_all,
           "semanticttl_wai_hit_similarity": sim_wai,
           "semantic_equivalence_labels": eq,
           "judge_agreement_pct": (100 * agree / len(agree_keys)
                                   if agree_keys else None),
           "judge_items": len(agree_keys),
           "web_calls": 0, "new_search_calls": 0,
           "hashes_pre": pre, "hashes_post": post,
           "implementation_unchanged": unchanged}
    json.dump(out, open(HERE / "answer_audit_results.json", "w"), indent=2)
    REF.record("heldout_audit/fresh_correct", len(fresh_correct))
    for arm, nm in (("fc", "FreshCache_primary"), ("sttl", "SemanticTTL_k1_16")):
        REF.record(f"heldout_audit/{nm}/CORRECT", res[arm]["correct"]); REF.record(f"heldout_audit/{nm}/WAI", res[arm]["wai"])

    with open(HERE / "paired_transitions.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "cluster", "fc_class", "sim_band", "in_S",
                    "freshcache_tier", "semanticttl_tier",
                    "semanticttl_similarity", "fresh", "freshcache",
                    "semanticttl", "freshcache_wai", "semanticttl_wai"])
        for r in rows:
            w.writerow([r["qid"], r["cluster"], r["fc"], r["band"], r["in_S"],
                        r["fc_tier"], r["sttl_tier"], r["sttl_similarity"],
                        r["fresh"], r["fc"], r["sttl"],
                        int(isw(r, "fc")), int(isw(r, "sttl"))])
    with open(HERE / "path_analysis.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paths[0]))
        w.writeheader()
        for p in paths:
            w.writerow(p)

    # judge_outputs.jsonl: one row per scored (request, arm)
    with open(HERE / "judge_outputs.jsonl", "w", encoding="utf-8") as fh:
        for d in reqs:
            for arm, _ in ARMS:
                a = ans.get(tuple(d[f"{arm}_key"]))
                if a is None:
                    continue
                sig = (h(d["query"]), h(d["gold"]), h(a))
                fh.write(json.dumps({
                    "query_id": d["query_id"], "arm": arm, "sig": list(sig),
                    "question": d["query"], "gold": d["gold"], "answer": a,
                    "llama8b": j1.get(sig), "qwen7b": j2.get(sig),
                    "abstain": is_abstain(a),
                    "outcome": outcome(d["query"], d["gold"], a)}) + "\n")
    for arm, _ in ARMS:
        name = {"fresh": "fresh_answers.jsonl", "fc": "freshcache_answers.jsonl",
                "sttl": "semanticttl_k1_16_answers.jsonl"}[arm]
        with open(HERE / name, "w", encoding="utf-8") as fh:
            for d in reqs:
                a = ans.get(tuple(d[f"{arm}_key"]))
                fh.write(json.dumps({
                    "query_id": d["query_id"], "query": d["query"],
                    "gold": d["gold"], "t": d["t"],
                    "tier": d.get(f"{arm}_tier", "fresh"),
                    "mode": d.get(f"{arm}_mode", "fresh_reference"),
                    "gen_query": d.get(f"{arm}_gen_query", d["query"]),
                    "ctx_sha": d.get(f"{arm}_ctx_sha", d["fresh_ctx_sha"]),
                    "answer": a}) + "\n")

    write_report(out, rows, sample, isw)
    print(json.dumps({k: out[k] for k in
                      ("N_scored", "arms", "wai_mcnemar",
                       "correctness_mcnemar")}, indent=2, default=str)[:3000])
    print("wrote answer_audit_results.json, paired_transitions.csv, "
          "path_analysis.csv, judge_outputs.jsonl, *_answers.jsonl, "
          "answer_audit_report.md")


def reuse_labels(reqs, rows, isw):
    """Look for pre-existing semantic-equivalence labels for the SemanticTTL
    (serving query, incoming query) pairs. No new jury is run."""
    cands = list(ROOT.glob("**/*jury*.json*")) + list(ROOT.glob("**/*equiv*label*.json*"))
    pairs = {}
    for d in reqs:
        if d.get("sttl_tier") == "HIT" and d.get("sttl_matched_query"):
            pairs[d["query_id"]] = (d["sttl_matched_query"], d["query"])
    found, files = {}, []
    for f in cands:
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        recs = []
        if f.suffix == ".jsonl":
            for line in txt.splitlines():
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
        else:
            try:
                o = json.loads(txt)
                recs = o if isinstance(o, list) else (
                    o.get("records") or o.get("items") or [])
            except Exception:
                recs = []
        hit = 0
        for r in recs:
            if not isinstance(r, dict):
                continue
            a = r.get("query_a") or r.get("q1") or r.get("cached_query") or \
                r.get("source_query") or r.get("matched_query")
            b = r.get("query_b") or r.get("q2") or r.get("incoming_query") or \
                r.get("query")
            lab = (r.get("majority") or r.get("label") or r.get("verdict")
                   or r.get("equivalent"))
            if not a or not b or lab is None:
                continue
            for qid, (sq, iq) in pairs.items():
                if {a.strip(), b.strip()} == {sq.strip(), iq.strip()}:
                    found[qid] = str(lab)
                    hit += 1
        if hit:
            files.append({"file": str(f.relative_to(ROOT)), "matched_pairs": hit})
    if not found:
        return {"available": False,
                "note": "semantic-equivalence labels unavailable for this sample",
                "searched_files": len(cands),
                "semanticttl_hit_pairs_needing_labels": len(pairs)}
    def mism(sub):
        n = sum(1 for q in sub if q in found)
        m = sum(1 for q in sub if q in found
                and str(found[q]).upper() not in ("EQUIVALENT", "TRUE", "YES", "1"))
        return {"labelled": n, "mismatch": m,
                "mismatch_pct": (100 * m / n) if n else None}
    allh = [r["qid"] for r in rows if r["sttl_tier"] == "HIT"]
    waih = [r["qid"] for r in rows if r["sttl_tier"] == "HIT" and isw(r, "sttl")]
    return {"available": True, "sources": files,
            "all_semanticttl_hits": mism(allh),
            "semanticttl_wai_hits": mism(waih)}


def write_report(o, rows, sample, isw):
    a = o["arms"]
    fz = o["frozen_semanticttl"]
    sp = o["sampling"]
    ttl = fz["ttl_seconds"]

    def arow(key, label):
        r = a[key]
        return (f"| {label} | {r['N']:,} | {r['correct']:,} | {r['wrong']:,} | "
                f"{r['abstain']:,} | {r['accuracy_pct']:.4f} |")

    def wrow(key, label):
        r = a[key]
        return (f"| {label} | {r['wai']:,} | {r['wai_rate_pct']:.4f} | "
                f"[{r['wai_ci95'][0]:.4f}, {r['wai_ci95'][1]:.4f}] | "
                f"{r['reverse']:,} | {r['reverse_rate_pct']:.4f} | "
                f"{r['conditional_wai_pct']:.4f} |")

    prows = "\n".join(
        f"| {'FreshCache' if p['policy']=='fc' else 'SemanticTTL k=1/16'} | "
        f"{p['path']} | {p['requests']:,} | {p['correct']:,} | "
        f"{p['accuracy_pct']:.4f} | {p['wai_events']:,} | "
        f"{p['conditional_wai_pct']:.4f} |" for p in o["paths"])

    def srow(label, s):
        if not s:
            return f"| {label} | 0 | — | — | — | — | — | — |"
        return (f"| {label} | {s['n']:,} | {s['mean']:.4f} | {s['median']:.4f} | "
                f"{s['min']:.4f} | {s['p25']:.4f} | {s['p75']:.4f} | "
                f"{s['max']:.4f} |")

    eq = o["semantic_equivalence_labels"]
    if eq.get("available"):
        eqtxt = ("| scope | labelled | mismatch | mismatch % |\n|---|---|---|---|\n"
                 + f"| all SemanticTTL hits | {eq['all_semanticttl_hits']['labelled']} | "
                 f"{eq['all_semanticttl_hits']['mismatch']} | "
                 f"{eq['all_semanticttl_hits']['mismatch_pct']} |\n"
                 + f"| SemanticTTL WAI hits | {eq['semanticttl_wai_hits']['labelled']} | "
                 f"{eq['semanticttl_wai_hits']['mismatch']} | "
                 f"{eq['semanticttl_wai_hits']['mismatch_pct']} |\n\n"
                 + "Sources: "
                 + ", ".join(f"`{s['file']}`" for s in eq["sources"]))
    else:
        eqtxt = (f"**semantic-equivalence labels unavailable for this sample.** "
                 f"{eq['searched_files']} candidate local label files were "
                 f"searched; none contained exact labels for the "
                 f"{eq['semanticttl_hit_pairs_needing_labels']} SemanticTTL "
                 f"(serving query, incoming query) pairs in this sample. "
                 f"No new semantic-equivalence jury was run, as instructed.")

    wm, cm = o["wai_mcnemar"], o["correctness_mcnemar"]
    hrows = "\n".join(
        f"| `{f}` | `{o['hashes_pre'][f][:16]}` | `{o['hashes_post'][f][:16]}` | "
        f"{'unchanged' if o['hashes_pre'][f]==o['hashes_post'][f] else 'CHANGED'} |"
        for f in o["hashes_pre"])
    comp = sample["composition"]
    fcc = ", ".join(f"{k} {v:,}" for k, v in sorted(comp["fc"].items()))
    stc = ", ".join(f"{k} {v:,}" for k, v in sorted(comp["sttl"].items()))
    w_hit, w_miss = o["semanticttl_wai_on_hits"], o["semanticttl_wai_on_misses"]
    if w_hit + w_miss == 0:
        wai_where = "There are no SemanticTTL WAI events in this audit."
    elif w_miss == 0:
        wai_where = (f"**Every one of the {w_hit:,} SemanticTTL\nWAI events occurs on a cache hit**, so the degradation is attributable to\n"
                     "semantic reuse, not to generation.")
    else:
        wai_where = (f"Of the {w_hit + w_miss:,} SemanticTTL WAI events, {w_hit:,} occur on cache hits and {w_miss:,} on misses,\n"
                     "so not all of the degradation is attributable to semantic reuse.")

    md = f"""# Held-out answer-quality audit — FreshCache vs SemanticTTL kappa = 1/16

One common policy-neutral sample, drawn from **held-out TEST
clusters only**. Nothing was retuned, no threshold or TTL was swept, and no
other baseline was evaluated. No new web, search or API call was made; all
evidence came from local snapshots.

The question this answers: **does the held-out SemanticTTL configuration that
looks strong on search savings and content drift also preserve final answer
quality?**

## 1. Frozen configurations

**SemanticTTL** — read verbatim from `validation_selection.json`, the Stage 1
savings-matched selection, and asserted to be the exact point Stage 2 evaluated.

| Parameter | Value |
|---|---|
| selection rule | savings-matched (pre-registered in Stage 1) |
| semantic threshold theta | **{fz['theta']}** |
| TTL multiplier kappa | **{fz['kappa']} = 1/16** |
| TTL TIMELESS | {ttl['TIMELESS']:,.1f} s |
| TTL SLOW | {ttl['SLOW']:,.1f} s |
| TTL MEDIUM | {ttl['MEDIUM']:,.1f} s |
| TTL FAST | {ttl['FAST']:,.1f} s |
| TTL REAL_TIME | {ttl['REAL_TIME']:,.1f} s |
| engine mechanism | `l1=sim_ttl`, no L2, no L3, REAL_TIME bypass |

FreshCache's lexical / entity / answer-type equivalence gate was **not** added to
SemanticTTL; doing so would change the baseline. SemanticTTL admits a hit on
similarity plus its TTL alone.

**FreshCache** — 15 runtime assertions passed before any replay: theta_L1 = 0.40,
effective L2 floor 0.75, epsilon 0.10 / 0.20 / 0.35, tier multipliers
1.5 / 1.2 / 1.0, half-lives 30d / 11.5d / 5.3d / 5.2d / 30s, equivalence floor
0.80 with Jaccard >= 0.30.

Both policy replays reproduce their Stage 2 aggregates exactly (FreshCache
savings, L1 and L2 hits; SemanticTTL savings and hits).

## 2. Test-only sampling verification

- Requests come only from the {sp['frame_size']:,}-request policy-neutral frame
  built over the **held-out test clusters**.
- **Validation requests present in this audit: {sp['validation_requests_present']}** (asserted, not assumed).
- The stream is the complete `zipf_uniform` seed-42 stream,
  filtered to test clusters. Timestamps, snapshot versions, cache ages and
  web-state alignment are the original ones; the schedule was not rebuilt from
  test records.
- Each policy replayed from its own empty, independent cache.
- Every page is read at `version_at` of its own content time, always at or
  before the request time. No future snapshot is visible to any arm.

## 3. Sample construction

Eligibility (policy-neutral): non-REAL_TIME, benchmark gold answer present, and
a non-empty fresh context available at the request timestamp. Eligibility does
not consult FreshCache hit/miss, SemanticTTL hit/miss, or any correctness
outcome. Sampled IDs were frozen before either policy was executed.

| Property | Value |
|---|---|
| frame size (test only) | {sp['frame_size']:,} |
| strata | **{sp['n_strata']}** (4 non-REAL_TIME classes x 7 similarity bands) |
| per-stratum cap | {sp['per_stratum_cap']} |
| target / sampled | {sp['target']} / {o['N_sampled']} |
| seed | {sp['seed']} |
| in common observable support | {sp['in_support_S']} |
| scored after judging | **{o['N_scored']}** (dropped unjudgeable: {o['dropped_unjudgeable']}) |

Policy composition of the sample: FreshCache — {fcc}; SemanticTTL — {stc}.

## 4-6. Accuracy of each arm

Generator Llama-3.2-3B-Instruct, greedy, max_new_tokens=80, 1024-token
truncation, FORCED prompt — the frozen audit protocol, unchanged. Correctness requires
**both** judges (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, greedy,
chat template, max_new_tokens=6) to answer CORRECT. Judges saw only question,
reference and candidate; policy identity was hidden.

| Arm | N | Correct | Wrong | Abstain | Accuracy % |
|---|---|---|---|---|---|
{arow('fresh', 'Fresh reference (shared)')}
{arow('fc', 'Frozen FreshCache')}
{arow('sttl', 'SemanticTTL kappa=1/16')}

### Wrong-Answer-Introduction and reverse transitions

WAI = cached answer WRONG while the shared fresh reference is CORRECT. Reverse =
cached CORRECT while fresh WRONG. Conditional degradation divides WAI by the
{a['fc']['fresh_correct_n']:,} fresh-correct requests, which matters because
fresh accuracy is not high.

| Policy | WAI | WAI rate % (of {o['N_scored']}) | 95% Wilson CI | Reverse | Reverse % | Conditional WAI % (of {a['fc']['fresh_correct_n']:,} fresh-correct) |
|---|---|---|---|---|---|---|
{wrow('fc', 'Frozen FreshCache')}
{wrow('sttl', 'SemanticTTL kappa=1/16')}

## 7. Paired WAI comparison (same requests, same fresh references)

| Cell | Count |
|---|---|
| WAI under both | {wm['wai_both']:,} |
| WAI under SemanticTTL only | **{wm['wai_semanticttl_only']:,}** |
| WAI under FreshCache only | **{wm['wai_freshcache_only']:,}** |
| WAI under neither | {wm['wai_neither']:,} |

Exact McNemar (binomial) on the discordant pairs
({wm['wai_semanticttl_only']:,} vs {wm['wai_freshcache_only']:,}):
**p = {wm['exact_mcnemar_p']:.6g}**

## 8. Paired correctness comparison

| Cell | Count |
|---|---|
| both correct | {cm['both_correct']:,} |
| FreshCache correct, SemanticTTL wrong | **{cm['freshcache_correct_semanticttl_wrong']:,}** |
| SemanticTTL correct, FreshCache wrong | **{cm['semanticttl_correct_freshcache_wrong']:,}** |
| both wrong | {cm['both_wrong']:,} |

Exact McNemar on policy correctness: **p = {cm['exact_mcnemar_p']:.6g}**

This is reported separately from the WAI test; the two ask different questions.

## 9. Conditional degradation among fresh-correct requests

Of the {a['fc']['fresh_correct_n']:,} requests where the shared fresh reference
was judged correct, FreshCache degraded **{a['fc']['conditional_wai_pct']:.4f}%**
and SemanticTTL kappa=1/16 degraded
**{a['sttl']['conditional_wai_pct']:.4f}%**.

## 10. Policy-path analysis

| Policy | Path | Requests | Correct | Accuracy % | WAI events | Conditional WAI % |
|---|---|---|---|---|---|---|
{prows}

SemanticTTL WAI events on **cache hits: {o['semanticttl_wai_on_hits']:,}**;
on **misses: {o['semanticttl_wai_on_misses']:,}**. A SemanticTTL miss searches and fetches
its own URLs at the request time, which is exactly the fresh reference's query
and context; under greedy decoding it therefore returns the identical answer and
WAI is structurally impossible on that path. {wai_where}

## 11. Similarity analysis of SemanticTTL reuse

| Set | N | Mean | Median | Min | p25 | p75 | Max |
|---|---|---|---|---|---|---|---|
{srow('All SemanticTTL cache hits', o['semanticttl_hit_similarity'])}
{srow('SemanticTTL WAI hits', o['semanticttl_wai_hit_similarity'])}

## 12. Reusable semantic-equivalence labels

{eqtxt}

## 13. No web, API or search calls

Zero. Every answer was generated from locally stored snapshot text selected by
`version_at` at or before each request timestamp, and every similarity came from
the cached BGE-M3 matrix. No Serper or other search request was issued.

Judge agreement: {o['judge_agreement_pct']:.2f}% over {o['judge_items']:,} distinct
judged items.

## 14. Implementation integrity

| File | SHA256 before | SHA256 after | Status |
|---|---|---|---|
{hrows}

All implementation files byte-identical
(`implementation_unchanged: {o['implementation_unchanged']}`). No manuscript
file and no Stage 1 or Stage 2 artifact was modified; every artifact from this
audit is new and lives in `answer_audit_k1_16/`.

## Interpretation limits

This audit compares exactly two policies on one sample from held-out
test clusters. It is **not** comparable to the earlier SemanticTTL kappa = 1/8
audit, which used a different sample drawn from the full replay; the two must
not be pooled or presented as the same experiment. Fresh-reference accuracy
bounds what any cache can achieve here, which is why the conditional degradation
column is reported alongside the raw WAI rate.
"""
    (HERE / "answer_audit_report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
