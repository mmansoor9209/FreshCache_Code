#!/usr/bin/env python3
"""
Validation-only grid for the L1-Precision gate.

  python run_validation_grid.py replay   -> 96 sequential validation replays,
                                            emits validation_jury_tasks.jsonl
  python run_validation_grid.py score    -> reads the jury labels and writes
                                            validation_grid.csv

Every cell is a TRUE SEQUENTIAL replay of the whole validation stream: an
altered L1 decision changes the cache later requests see, so no cell is derived
by filtering another cell's hits.

Held-out test clusters are never loaded here.
"""
from __future__ import annotations
import csv, hashlib, itertools, json, os, pathlib, sys, time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-l1precision")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
sys.path.insert(0, str(HERE))
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import experiment as exp              # noqa: E402
import engine_all as ea               # noqa: E402
import mixed_engine as me             # noqa: E402
import schedules as sc                # noqa: E402
import prep2 as p2                    # noqa: E402
from e1_robustness import support_of  # noqa: E402
import l1_gate                        # noqa: E402

SCHEDULE, SEED = "zipf_uniform", 42
CHANGED, UNCHANGED = me.CHANGED, me.UNCHANGED
# Files whose behaviour must not change. Checked before and after every run.
WATCH = ["experiment.py", "v16_exp12/mixed_engine.py",
         "v14_baselines/engine_all.py", "v16_exp12/schedules.py",
         "validation/mixed_age_full_policy_audit/prep2.py"]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def h16(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def load_common():
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(records)
    rounds = {}
    with open(ROOT / "v13_corrected" / "corrected_round_table.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    return records, rounds, split


def score_run(m, rows, S, records, rounds, stream):
    """Drift / coverage from the per-request reuse outcomes, plus op counts."""
    ch = un = 0
    for t, r in stream:
        q = r["query_id"]
        p = rows.get(q)
        if not p:
            continue
        if p["tier"] == "L1":
            labs = [me.outcome(x, p["t_cached"], t, rounds)
                    for x in p.get("served", [])]
        elif p["tier"] in ("L2", "miss"):
            labs = [me.outcome(u, tc, t, rounds) for u, tc in p.get("ev", [])
                    if tc < t]
        else:
            labs = []
        if not labs or q not in S:
            continue
        o = (CHANGED if CHANGED in labs else
             (UNCHANGED if UNCHANGED in labs else None))
        if o == CHANGED:
            ch += 1
        elif o == UNCHANGED:
            un += 1
    det = ch + un
    n = len(stream)
    gens = m["search_calls"] + m["l2_hits"]
    return {"search_saved_pct": m["search_saved_pct"],
            "searches": m["search_calls"], "fetches": m["fetches"],
            "generations": gens,
            "gen_per_1k": round(1000 * gens / n, 3),
            "fetch_per_1k": round(1000 * m["fetches"] / n, 3),
            "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
            "l3_hits": m["l3_hits"],
            "drift_pct": (round(100 * ch / det, 4) if det else None),
            "determinable": det, "changed": ch,
            "coverage_pct": (round(100 * det / len(S), 4) if S else None),
            "n_requests": n}


def cells():
    P = json.load(open(HERE / "l1_precision_prereg.json", encoding="utf-8"))
    g = P["grid"]
    return P, list(itertools.product(g["sim_floor"], g["jaccard_floor"],
                                     ["as_implemented", "required"],
                                     [False, True]))


def replay(args):
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    pre = {f: sha(ROOT / f) for f in WATCH}
    P, CELLS = cells()
    say("L1-PRECISION GATE -- validation grid (VALIDATION CLUSTERS ONLY)")
    say(f"  prereg sha256 {sha(HERE / 'l1_precision_prereg.json')}")
    say(f"  guards sha256 {sha(HERE / 'l1_guards.py')}  "
        f"(matches prereg: {sha(HERE/'l1_guards.py') == P['l1_guards_sha256']})")
    assert sha(HERE / "l1_guards.py") == P["l1_guards_sha256"], \
        "l1_guards.py changed after pre-registration"
    say(f"  cells {len(CELLS)}")

    records, rounds, split = load_common()
    val_c = set(split["validation_clusters"])
    test_c = set(split["test_clusters"])
    full = sc.build_stream(records, SCHEDULE, SEED)
    assert len(full) == 31201
    stream = [(t, r) for t, r in full
              if (r.get("cluster_id") or r["query_id"]) in val_c]
    assert len(stream) == split["validation"]["requests"]
    assert not [1 for _, r in stream
                if (r.get("cluster_id") or r["query_id"]) in test_c], \
        "test clusters leaked into the validation grid"
    S = support_of(stream, rounds)
    say(f"  validation stream {len(stream):,} requests, {len(val_c):,} "
        f"clusters, |S| = {len(S):,}; test leakage 0")

    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in records}
    cl_of_q = {}
    for r in records:
        cl_of_q.setdefault(r["query"], r.get("cluster_id") or r["query_id"])

    ORIG = exp.semantic_equivalent
    OS, OJ = exp._EQ_SIM_FLOOR, exp._EQ_JACCARD_MIN

    # ---- gate-reimplementation check: at the published floors and modes the
    # ---- configurable gate must reproduce the published replay exactly.
    m_pub, rows_pub = p2.replay(stream, {}, "FreshCache")
    exp.semantic_equivalent = l1_gate.published_equivalent(exp, OS, OJ)
    try:
        m_chk, rows_chk = p2.replay(stream, {}, "FreshCache")
    finally:
        exp.semantic_equivalent = ORIG
    same = (m_chk == m_pub
            and all(rows_chk[q]["tier"] == rows_pub[q]["tier"] for q in rows_pub))
    say(f"  GATE REIMPLEMENTATION CHECK: reproduces the published gate "
        f"exactly: {same}")
    if not same:
        say(f"    {m_chk} vs {m_pub}")
        sys.exit(2)
    say(f"  published validation reference: saved {m_pub['search_saved_pct']:.4f}%"
        f"  L1 {m_pub['l1_hits']:,}  L2 {m_pub['l2_hits']:,}")
    pub_sc = score_run(m_pub, rows_pub, S, records, rounds, stream)

    out, pairs = [], {}
    # per-cell checkpoint so an interrupted run resumes instead of restarting
    CKPT = HERE / "validation_cells_ckpt.jsonl"
    donec = {}
    if CKPT.exists():
        for line in open(CKPT, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                donec[d["row"]["cell"]] = d
        say(f"  resuming: {len(donec)} cells already checkpointed")

    def capture(tag, rows_, m_, sim_f, jac_f, at, gd):
        hits = []
        for t, r in stream:
            q = r["query_id"]
            p = rows_.get(q)
            if not p or p["tier"] != "L1":
                continue
            inc, mq = r["query"], p["matched_query"]
            pid = f"{h16(inc)}_{h16(mq)}"
            pairs[pid] = (inc, mq)
            hits.append({"query_id": q, "pair_id": pid,
                         "cross_cluster": int(cl_of_q.get(mq) != cid[q])})
        sc_ = score_run(m_, rows_, S, records, rounds, stream)
        sc_.update({"cell": tag, "sim_floor": sim_f, "jaccard_floor": jac_f,
                    "answer_type": at, "guards": gd,
                    "cross_cluster_hits": sum(x["cross_cluster"] for x in hits)})
        return sc_, hits

    pub_row, pub_hits = capture("published", rows_pub, m_pub, OS, OJ,
                                "as_implemented", False)
    percell = {"published": pub_hits}
    out.append(pub_row)
    for d in donec.values():
        for x in d["hits"]:
            pairs.setdefault(x["pair_id"], None)

    t0 = time.time()
    say(f"\n  {'#':>3} {'sim':>5}{'jac':>6}{'atype':>16}{'guards':>8}"
        f"{'L1':>7}{'saved%':>10}{'gen/1k':>9}{'drift%':>9}")
    for i, (sf, jf, at, gd) in enumerate(CELLS, 1):
        tag = f"s{sf}_j{jf}_{at}_{'G' if gd else 'g'}"
        if tag in donec:
            out.append(donec[tag]["row"])
            percell[tag] = donec[tag]["hits"]
            continue
        exp.semantic_equivalent = l1_gate.make_gate(exp, sf, jf, at, gd)
        try:
            m_, rows_ = p2.replay(stream, {}, "FreshCache")
        finally:
            exp.semantic_equivalent = ORIG
        row, hits = capture(tag, rows_, m_, sf, jf, at, gd)
        out.append(row)
        percell[tag] = hits
        with open(CKPT, "a", encoding="utf-8") as _fh:
            _fh.write(json.dumps({"row": row, "hits": hits}) + "\n")
        say(f"  {i:>3} {sf:>5.2f}{jf:>6.2f}{at:>16}{str(gd):>8}"
            f"{row['l1_hits']:>7,}{row['search_saved_pct']:>9.4f}%"
            f"{row['gen_per_1k']:>9.2f}"
            + (f"{row['drift_pct']:>8.4f}%" if row['drift_pct'] is not None
               else f"{'-':>9}"))
        if i % 12 == 0:
            say(f"      ... {i}/{len(CELLS)} cells, {time.time()-t0:.0f}s")

    assert exp.semantic_equivalent is ORIG
    assert exp._EQ_SIM_FLOOR == OS and exp._EQ_JACCARD_MIN == OJ
    post = {f: sha(ROOT / f) for f in WATCH}
    assert pre == post, "implementation files changed"
    say(f"\n  implementation files unchanged: True")

    json.dump({"cells": out, "published_validation": pub_row,
               "prereg_sha256": sha(HERE / "l1_precision_prereg.json"),
               "hashes": pre},
              open(HERE / "validation_replays.json", "w"), indent=2)
    json.dump({k: v for k, v in percell.items()},
              open(HERE / "validation_cell_hits.json", "w"))
    # resumed cells contribute pair_ids without their text; recover the text
    if any(v is None for v in pairs.values()):
        qtext = {}
        for t, r in stream:
            qtext[h16(r["query"])] = r["query"]
        for rr in records:
            qtext.setdefault(h16(rr["query"]), rr["query"])
        for pid, v in list(pairs.items()):
            if v is None:
                a, b = pid.split("_")
                if a in qtext and b in qtext:
                    pairs[pid] = (qtext[a], qtext[b])
                else:
                    del pairs[pid]
    with open(HERE / "validation_jury_tasks.jsonl", "w", encoding="utf-8") as fh:
        for pid, (a, b) in sorted(pairs.items()):
            fh.write(json.dumps({"pair_id": pid, "incoming": a,
                                 "cached": b}) + "\n")
    say(f"  distinct (incoming, cached) pairs across ALL cells: {len(pairs):,}")
    say(f"  wrote validation_replays.json, validation_cell_hits.json, "
        f"validation_jury_tasks.jsonl")
    (HERE / "logs" / "validation_replay.log").write_text("\n".join(log) + "\n",
                                                         encoding="utf-8")


def score(args):
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    JUDGES = ["llama3b", "llama8b", "qwen7b", "mistral7b"]
    eq = defaultdict(dict)
    for j in JUDGES:
        p = HERE / f"validation_equiv_{j}.jsonl"
        if not p.exists():
            continue
        for l in open(p, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                eq[d["pair_id"]][j] = d["label"]
    present = sorted({j for v in eq.values() for j in v})
    say(f"L1-PRECISION GATE -- scoring the validation grid")
    say(f"  jury models present: {present}; pairs judged {len(eq):,}")
    if len(present) < 4:
        say(f"  WARNING: only {len(present)} of 4 jury models present")

    def verdict(pid):
        v = {j: l for j, l in eq.get(pid, {}).items()
             if l in ("SAME", "DIFFERENT")}
        if len(v) < 2:
            return "UNJUDGED"
        c = Counter(v.values())
        if c["SAME"] == c["DIFFERENT"]:
            return "JURY_TIE"
        return c.most_common(1)[0][0]

    V = {pid: verdict(pid) for pid in eq}
    cellhits = json.load(open(HERE / "validation_cell_hits.json",
                              encoding="utf-8"))
    R = json.load(open(HERE / "validation_replays.json", encoding="utf-8"))
    rows = []
    for c in R["cells"]:
        hits = cellhits[c["cell"]]
        n = len(hits)
        vs = [V.get(x["pair_id"], "UNJUDGED") for x in hits]
        diff = sum(1 for v in vs if v == "DIFFERENT")
        ties = sum(1 for v in vs if v == "JURY_TIE")
        det = sum(1 for v in vs if v in ("SAME", "DIFFERENT"))
        xc = [x for x in hits if x["cross_cluster"]]
        xvs = [V.get(x["pair_id"], "UNJUDGED") for x in xc]
        xdiff = sum(1 for v in xvs if v == "DIFFERENT")
        xties = sum(1 for v in xvs if v == "JURY_TIE")
        xdet = sum(1 for v in xvs if v in ("SAME", "DIFFERENT"))
        rows.append({**c,
                     "jury_ties": ties, "jury_determinate": det,
                     "mismatch_ties_diff_pct": (round(100*(diff+ties)/n, 4) if n else None),
                     "mismatch_no_ties_pct": (round(100*diff/det, 4) if det else None),
                     "mismatch_ties_same_pct": (round(100*diff/n, 4) if n else None),
                     "cross_cluster_mismatch_ties_diff_pct":
                         (round(100*(xdiff+xties)/len(xc), 4) if xc else None),
                     "cross_cluster_mismatch_no_ties_pct":
                         (round(100*xdiff/xdet, 4) if xdet else None),
                     "wai": None, "wai_note": "stage 2 only"})
    cols = ["cell", "sim_floor", "jaccard_floor", "answer_type", "guards",
            "l1_hits", "cross_cluster_hits", "jury_ties", "jury_determinate",
            "mismatch_ties_diff_pct", "mismatch_no_ties_pct",
            "mismatch_ties_same_pct", "cross_cluster_mismatch_ties_diff_pct",
            "cross_cluster_mismatch_no_ties_pct", "search_saved_pct",
            "searches", "fetches", "fetch_per_1k", "generations", "gen_per_1k",
            "drift_pct", "coverage_pct", "determinable", "l2_hits", "l3_hits",
            "n_requests", "wai", "wai_note"]
    with open(HERE / "validation_grid.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["mismatch_ties_diff_pct"] is None,
                                             x["mismatch_ties_diff_pct"])):
            w.writerow(r)
    pub = [r for r in rows if r["cell"] == "published"][0]
    say(f"  published validation: L1 {pub['l1_hits']:,}  "
        f"mismatch(ties->DIFF) {pub['mismatch_ties_diff_pct']}%  "
        f"saved {pub['search_saved_pct']:.4f}%")
    say(f"\n  best 12 cells by the pre-registered objective "
        f"(mismatch, ties->DIFFERENT):")
    say(f"    {'cell':<34}{'L1':>6}{'mm_tD%':>9}{'mm_noT%':>9}"
        f"{'saved%':>10}{'gen/1k':>9}")
    for r in sorted([r for r in rows if r["mismatch_ties_diff_pct"] is not None],
                    key=lambda x: x["mismatch_ties_diff_pct"])[:12]:
        say(f"    {r['cell']:<34}{r['l1_hits']:>6,}"
            f"{r['mismatch_ties_diff_pct']:>8.2f}%{r['mismatch_no_ties_pct']:>8.2f}%"
            f"{r['search_saved_pct']:>9.4f}%{r['gen_per_1k']:>9.2f}")
    json.dump(rows, open(HERE / "validation_grid_scored.json", "w"), indent=2)
    (HERE / "logs" / "validation_score.log").write_text("\n".join(log) + "\n",
                                                        encoding="utf-8")
    say(f"\n  wrote validation_grid.csv, validation_grid_scored.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replay").set_defaults(func=replay)
    sub.add_parser("score").set_defaults(func=score)
    a = ap.parse_args()
    a.func(a)
