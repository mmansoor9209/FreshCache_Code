#!/usr/bin/env python3
"""TASK 2 -- separate temporal calibration targets for L1, L2 and L3.

The three tiers reuse different objects, so they have different staleness
events and must not share a label:

  L3 reuses a PAGE      -> stale iff that page's content changed materially.
  L2 reuses a URL LIST  -> the pages are re-fetched, so page change does NOT
                           make L2 stale. L2 is stale iff a fresh search at
                           serve time would return a materially different or
                           less suitable list.
  L1 reuses an ANSWER   -> stale iff the correct answer to the query changed.

Semantic mismatch (the cached entry answers a DIFFERENT question) is a
separate failure mode and is kept in its own column throughout; it is never
folded into a temporal label.

This script only measures what the existing artifacts can support. It does not
fabricate labels for the tiers where they are absent.
"""
from __future__ import annotations
import json, math, os, pathlib, sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
ROOT = PC.parents[1]
for p in ("", "v14_baselines", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)
import experiment as exp                 # noqa: E402

AGES = {"rerun_1h": 3_600.0, "rerun_12h": 43_200.0,
        "rerun_24h": 86_400.0, "rerun_7d": 604_800.0}
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p, d = k / n, 1 + z * z / n
    c, m = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 3), round(100 * min(1.0, (c + m) / d), 3))


def main():
    say("TASK 2 -- per-tier temporal calibration targets")

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])

    # URL -> the splits whose queries cite it. A URL is usable for validation
    # calibration only if NO held-out query cites it (no leakage either way).
    url_split = defaultdict(set)
    for r in records:
        s = ("val" if (r.get("cluster_id") or r["query_id"]) in val_c else
             "test" if (r.get("cluster_id") or r["query_id"]) in test_c else "other")
        for u in r["urls"]:
            url_split[u["url_hash"]].add(s)
    val_urls = {u for u, s in url_split.items() if "val" in s and "test" not in s}
    say(f"  URLs cited only by validation clusters: {len(val_urls):,} "
        f"(of {len(url_split):,} cited URLs)")

    fc_of = {}
    for m in manifest:
        if m.get("url_hash"):
            fc_of.setdefault(m["url_hash"], m.get("freshness_class"))

    # ---------------- L3: page content change ----------------
    rounds = {}
    for line in open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
                     encoding="utf-8"):
        d = json.loads(line)
        rounds[d["url_hash"]] = d["rounds"]
    noise = {(d["url_hash"], d["run_id"])
             for d in map(json.loads,
                          open(ROOT / "data" / "change_log.jsonl", encoding="utf-8"))
             if str(d.get("noise")).lower() == "true"}
    say(f"  change_log noise-flagged (url, round) pairs: {len(noise):,}")

    l3 = []
    for u, per in rounds.items():
        if u not in val_urls:
            continue
        a = per.get("run_00")
        if not a or not a.get("substantive"):
            continue
        for rnd, age in AGES.items():
            b = per.get(rnd)
            if not b or not b.get("substantive"):
                continue
            raw = a["content_hash"] != b["content_hash"]
            l3.append({"url_hash": u, "freshness_class": fc_of.get(u),
                       "age_s": age, "round": rnd, "changed_raw": int(raw),
                       "noise_flagged": int((u, rnd) in noise),
                       "changed_denoised": int(raw and (u, rnd) not in noise)})
    say(f"\n  == L3 target: page content change (validation-only URLs) ==")
    say(f"    observations {len(l3):,} over {len({r['url_hash'] for r in l3}):,} URLs")
    say(f"    {'class':<11}{'age':>8}{'n':>8}{'raw':>9}{'denoised':>10}"
        f"{'model P(stale)':>16}{'abs err':>10}")
    cal = []
    for fc in ["REAL_TIME", "FAST", "MEDIUM", "SLOW", "TIMELESS"]:
        for rnd, age in AGES.items():
            g = [r for r in l3 if r["freshness_class"] == fc and r["round"] == rnd]
            if len(g) < 30:
                continue
            n = len(g)
            raw = sum(r["changed_raw"] for r in g) / n
            den = sum(r["changed_denoised"] for r in g) / n
            pred = exp.p_stale(fc, age, "content")
            cal.append({"freshness_class": fc, "age_s": age, "n": n,
                        "observed_raw": round(100 * raw, 4),
                        "observed_denoised": round(100 * den, 4),
                        "ci95_denoised": wilson(sum(r['changed_denoised'] for r in g), n),
                        "model_pct": round(100 * pred, 4),
                        "abs_err_pp": round(100 * abs(pred - den), 4)})
            say(f"    {fc:<11}{age/3600:>7.0f}h{n:>8,}{100*raw:>8.2f}%"
                f"{100*den:>9.2f}%{100*pred:>15.2f}%"
                f"{100*abs(pred-den):>9.2f}")
    mae = sum(c["abs_err_pp"] for c in cal) / max(1, len(cal))
    wmae = (sum(c["abs_err_pp"] * c["n"] for c in cal)
            / max(1, sum(c["n"] for c in cal)))
    say(f"    calibration error of the published content model (denoised): "
        f"MAE {mae:.2f} pp, n-weighted MAE {wmae:.2f} pp over {len(cal)} cells")

    # ---------------- L2: URL-list staleness ----------------
    say(f"\n  == L2 target: URL-list outdated or unsuitable ==")
    snapdirs = sorted(p.name for p in (ROOT / "data" / "snapshots").iterdir()
                      if p.is_dir())
    say(f"    snapshot rounds present: {snapdirs}")
    say(f"    These are PAGE re-fetches of an already-fixed URL list. The search")
    say(f"    engine was never re-queried at a later round, so no artifact")
    say(f"    records what a fresh search would have returned at serve time.")
    h3 = [json.loads(l) for l in
          open(ROOT / "v10_remaining_feedback" / "h3_freshcache_l2_judgements.jsonl",
               encoding="utf-8")]
    say(f"    nearest artifact: h3_freshcache_l2_judgements.jsonl, {len(h3):,} rows, "
        f"labels {dict(Counter(r['label'] for r in h3))}")
    say(f"    -> these grade SEMANTIC suitability of a matched query's URL list")
    say(f"       across similarity bands at a SINGLE time. They carry no age")
    say(f"       variation, so they identify the semantic gate, NOT a temporal")
    say(f"       decay rate. Using them as an L2 temporal target would be")
    say(f"       exactly the conflation this task forbids.")
    say(f"    VERDICT: no L2 temporal label exists in this dataset.")

    # ---------------- L1: answer staleness ----------------
    say(f"\n  == L1 target: correct answer changed ==")
    al = [json.loads(l) for l in
          open(ROOT / "data" / "answer_labels.jsonl", encoding="utf-8")]
    ages = sorted({int(r["delta_seconds"]) for r in al})
    say(f"    answer_labels.jsonl: {len(al):,} rows, "
        f"{len({r['query_id'] for r in al}):,} distinct queries")
    say(f"    delta_seconds range: {min(ages):,} .. {max(ages):,} s "
        f"({min(ages)/86400:.1f} .. {max(ages)/86400:.1f} days)")
    l1rows = []
    for fc in ["REAL_TIME", "FAST", "MEDIUM", "SLOW", "TIMELESS"]:
        g = [r for r in al if r["freshness_class"] == fc]
        if not g:
            continue
        ch = sum(1 for r in g if r["gold_answer_t0"] != r["gold_answer_teval"])
        mn = min(int(r["delta_seconds"]) for r in g)
        ttl = -exp.HALF_LIFE[fc] * math.log(1 - exp.EPS_ANSWER) / (
            exp.TIER_MULT["answer"] * math.log(2))
        l1rows.append({"freshness_class": fc, "n": len(g), "changed": ch,
                       "rate_pct": round(100 * ch / len(g), 3),
                       "ci95": wilson(ch, len(g)),
                       "min_delta_s": mn, "l1_equiv_ttl_s": round(ttl, 1),
                       "min_delta_over_ttl": round(mn / ttl, 2)})
        say(f"    {fc:<11} n={len(g):>5,} changed {100*ch/len(g):>6.2f}% "
            f"CI {wilson(ch,len(g))} | min age {mn/86400:>5.1f} d "
            f"= {mn/ttl:>6.1f}x the L1 equivalent TTL ({ttl/3600:.1f} h)")
    say(f"    -> every answer-level label sits at least "
        f"{min(r['min_delta_over_ttl'] for r in l1rows):.0f}x beyond the L1")
    say(f"       admission boundary, where the observed change rate is already")
    say(f"       saturated near 100%. Labels at ages far past the boundary")
    say(f"       cannot locate the boundary.")
    say(f"    VERDICT: no L1 temporal label exists in the gate's operating range.")

    json.dump({"l3_calibration_cells": cal,
               "l3_mae_pp": round(mae, 4), "l3_weighted_mae_pp": round(wmae, 4),
               "l3_observations": len(l3),
               "l2_temporal_label_available": False,
               "l2_reason": "search results were never re-collected; "
                            "h3 judgements are single-time semantic suitability",
               "l1_temporal_label_in_range": False,
               "l1_summary": l1rows,
               "validation_only_urls": len(val_urls)},
              open(HERE / "targets_summary.json", "w"), indent=2)
    with open(HERE / "l3_observations.csv", "w", encoding="utf-8") as fh:
        fh.write("url_hash,freshness_class,age_s,round,changed_raw,"
                 "noise_flagged,changed_denoised\n")
        for r in l3:
            fh.write(f"{r['url_hash']},{r['freshness_class']},{r['age_s']:.0f},"
                     f"{r['round']},{r['changed_raw']},{r['noise_flagged']},"
                     f"{r['changed_denoised']}\n")
    (PC / "logs" / "t2.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"\n  wrote targets_summary.json, l3_observations.csv")


if __name__ == "__main__":
    main()
