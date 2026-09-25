#!/usr/bin/env python3
"""TASK 4 -- select risk budgets on VALIDATION CLUSTERS ONLY.

Search space, objective, constraints and tie-break are fixed in prereg.json,
hashed before this script was run and before any held-out replay. The held-out
split is never touched here.
"""
from __future__ import annotations
import hashlib, itertools, json, math, os, pathlib, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
ROOT = PC.parents[1]
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)
import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import mixed_engine as me                # noqa: E402
import schedules as sc                   # noqa: E402

SEED, SCHEDULE = 42, "zipf_uniform"
CKPT = HERE / "sweep_checkpoint.json"
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def main():
    pre = json.load(open(HERE / "prereg.json", encoding="utf-8"))
    sha = hashlib.sha256((HERE / "prereg.json").read_bytes()).hexdigest()
    say("TASK 4 -- validation-only budget selection")
    say(f"  prereg sha256 {sha}")
    assert sha == open(HERE / "prereg.sha256").read().split()[0], \
        "prereg.json changed after hashing"
    sp = pre["search_space"]
    grid = list(itertools.product(sp["eps_answer"], sp["eps_url_list"],
                                  sp["eps_content"]))
    say(f"  search space {len(grid)} cells (prereg says {sp['cells']})")
    assert len(grid) == sp["cells"]

    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    for line in open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
                     encoding="utf-8"):
        d = json.loads(line)
        rounds[d["url_hash"]] = d["rounds"]
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    val_c, test_c = set(split["validation_clusters"]), set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    val = [(t, r) for t, r in full
           if (r.get("cluster_id") or r["query_id"]) in val_c]
    assert len(val) == split["validation"]["requests"]
    assert not [1 for _, r in val
                if (r.get("cluster_id") or r["query_id"]) in test_c], "leakage"
    say(f"  validation stream {len(val):,} requests, held-out leakage 0")

    base = (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT)

    def run(ea_, eu_, ec_):
        exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = ea_, eu_, ec_
        m, per = me.replay(val, rounds, "FreshCache", rich=rich)
        det = sum(1 for _, o in per if o in (me.CHANGED, me.UNCHANGED))
        chg = sum(1 for _, o in per if o == me.CHANGED)
        n = len(per)
        return {"eps_answer": ea_, "eps_url_list": eu_, "eps_content": ec_,
                "k_answer": round(-math.log(1-ea_)/(exp.TIER_MULT["answer"]*math.log(2)), 6),
                "k_url_list": round(-math.log(1-eu_)/(exp.TIER_MULT["url_list"]*math.log(2)), 6),
                "k_content": round(-math.log(1-ec_)/(exp.TIER_MULT["content"]*math.log(2)), 6),
                "search_saved_pct": round(m["search_saved_pct"], 4),
                "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
                "l3_hits": m["l3_hits"],
                "coverage_pct": round(100*det/n, 4) if n else None,
                "drift_pct": round(100*chg/det, 4) if det else None}

    ck = json.load(open(CKPT, encoding="utf-8")) if CKPT.exists() else {}
    try:
        pub = ck.get("published") or run(*base)
        ck["published"] = pub
        say(f"\n  published config on validation: saved {pub['search_saved_pct']:.4f}%"
            f"  L1 {pub['l1_hits']:,}  drift {pub['drift_pct']:.4f}%"
            f"  coverage {pub['coverage_pct']:.4f}%")
        cells = ck.get("cells", {})
        t0 = time.time()
        for i, (a, u, c) in enumerate(grid, 1):
            key = f"{a}_{u}_{c}"
            if key in cells:
                continue
            cells[key] = run(a, u, c)
            ck["cells"] = cells
            json.dump(ck, open(CKPT, "w"), indent=1)
            if i % 8 == 0:
                say(f"    ... {len(cells)}/{len(grid)} cells, {time.time()-t0:.0f}s")
    finally:
        exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = base
    cells = ck["cells"]

    # ---- apply the pre-registered objective ----
    C1 = pub["drift_pct"] + 0.25
    C2 = pub["l1_hits"]
    C3 = 0.95 * pub["coverage_pct"]
    say(f"\n  pre-registered constraints from the published validation run:")
    say(f"    C1 drift <= {C1:.4f}%   C2 L1 hits <= {C2:,}   "
        f"C3 coverage >= {C3:.4f}%")
    feas = [v for v in cells.values()
            if v["drift_pct"] is not None and v["drift_pct"] <= C1
            and v["l1_hits"] <= C2 and v["coverage_pct"] >= C3]
    say(f"    feasible cells: {len(feas)}/{len(cells)}")
    sel = sorted(feas, key=lambda v: (-v["search_saved_pct"], v["drift_pct"],
                                      v["l1_hits"], v["eps_answer"],
                                      v["eps_url_list"], v["eps_content"]))[0]
    say(f"\n  {'eps_a':>7}{'eps_u':>7}{'eps_c':>7}{'saved%':>10}{'drift%':>9}"
        f"{'cov%':>9}{'L1':>7}{'L2':>8}{'feasible':>10}")
    for v in sorted(cells.values(), key=lambda v: -v["search_saved_pct"])[:12]:
        ok = v in feas
        say(f"  {v['eps_answer']:>7.2f}{v['eps_url_list']:>7.2f}"
            f"{v['eps_content']:>7.2f}{v['search_saved_pct']:>10.4f}"
            f"{v['drift_pct']:>8.4f}%{v['coverage_pct']:>8.4f}%"
            f"{v['l1_hits']:>7,}{v['l2_hits']:>8,}{str(ok):>10}")
    say(f"\n  SELECTED (validation only): eps = ({sel['eps_answer']}, "
        f"{sel['eps_url_list']}, {sel['eps_content']})")
    say(f"    saved {sel['search_saved_pct']:.4f}%  drift {sel['drift_pct']:.4f}%"
        f"  coverage {sel['coverage_pct']:.4f}%  L1 {sel['l1_hits']:,}")
    say(f"    vs published {pub['search_saved_pct']:.4f}% / "
        f"{pub['drift_pct']:.4f}% / L1 {pub['l1_hits']:,}")

    # decision-equivalence classes among the swept cells
    eqv = {}
    for v in cells.values():
        sig = (v["search_saved_pct"], v["l1_hits"], v["l2_hits"], v["l3_hits"])
        eqv.setdefault(sig, []).append((v["eps_answer"], v["eps_url_list"],
                                        v["eps_content"]))
    dups = {str(k): v for k, v in eqv.items() if len(v) > 1}
    say(f"\n  == decision-equivalent budget combinations ==")
    say(f"    {len(eqv)} distinct policies among {len(cells)} budget triples; "
        f"{len(dups)} signatures are shared by more than one triple.")
    for k, v in list(dups.items())[:6]:
        say(f"      {v}  -> identical tier hits and savings")

    json.dump({"prereg_sha256": sha, "published_validation": pub,
               "constraints": {"C1_drift_max": C1, "C2_l1_max": C2,
                               "C3_coverage_min": C3},
               "n_feasible": len(feas), "selected": sel,
               "cells": cells,
               "decision_equivalent_groups": dups},
              open(HERE / "validation_sweep.json", "w"), indent=2)
    json.dump({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "selected_eps": {"answer": sel["eps_answer"],
                                "url_list": sel["eps_url_list"],
                                "content": sel["eps_content"]},
               "tier_mult_published": dict(exp.TIER_MULT),
               "selected_k": {"answer": sel["k_answer"],
                              "url_list": sel["k_url_list"],
                              "content": sel["k_content"]},
               "prereg_sha256": sha,
               "selected_on": "validation clusters only",
               "heldout_touched_during_selection": False},
              open(HERE / "frozen_config.json", "w"), indent=2)
    say(f"\n  FROZEN -> frozen_config.json  sha256 "
        f"{hashlib.sha256((HERE/'frozen_config.json').read_bytes()).hexdigest()[:16]}")
    (PC / "logs" / "t4.log").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
