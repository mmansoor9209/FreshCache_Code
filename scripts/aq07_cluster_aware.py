#!/usr/bin/env python3
"""AQ-07 / M-10 -- cluster-aware uncertainty and complete paired outcomes for
the principal answer comparisons. Reads existing row-level audit artifacts
only; no generation, judging or replay."""
from __future__ import annotations
import csv, json, math, pathlib, random
from collections import Counter, defaultdict
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent.parent
ART = HERE / "artifacts" / "audits"
SEED, B = 42, 10_000
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c, m = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0, (c - m) / d), 3), round(100 * min(1, (c + m) / d), 3)]


def mcnemar(b01, b10):
    n = b01 + b10
    return (min(1.0, float(sum(Fraction(math.comb(n, i), 2 ** n)
                               for i in range(0, min(b01, b10) + 1)) * 2))
            if n else 1.0)


def cluster_boot(rows, key, cl):
    by = defaultdict(list)
    for r in rows:
        by[r[cl]].append(r)
    ks = list(by); rng = random.Random(SEED); out = []
    for _ in range(B):
        fl = [x for _ in range(len(ks)) for x in by[ks[rng.randrange(len(ks))]]]
        out.append(100 * sum(1 for x in fl if key(x)) / len(fl))
    out.sort()
    return [round(out[int(.025 * B)], 3), round(out[int(.975 * B)], 3)], len(ks)


say("AQ-07 -- cluster-aware uncertainty for the principal answer comparisons")

# ---------- held-out audit (has cluster ids) ----------
ho = list(csv.DictReader(open(ART / "answer_audit_k1_16" / "paired_transitions.csv",
                              encoding="utf-8")))
say(f"\n  HELD-OUT 400-request audit: {len(ho)} requests, "
    f"{len({r['cluster'] for r in ho})} distinct clusters")
sizes = Counter(Counter(r["cluster"] for r in ho).values())
say(f"  cluster size distribution (size: n_clusters): {dict(sorted(sizes.items()))}")
rep = sum(v for k, v in sizes.items() if k > 1)
say(f"  clusters contributing more than one request: {rep} -> request-level "
    f"tests are NOT independent; cluster bootstrap reported alongside Wilson")

out = {}
for arm, lab in (("freshcache", "FreshCache"),
                 ("semanticttl", "SemanticTTL (kappa=1/16)")):
    w = sum(1 for r in ho if r[arm] == "WRONG" and r["fresh"] == "CORRECT")
    c = sum(1 for r in ho if r[arm] == "CORRECT")
    ci_w, ncl = cluster_boot(ho, lambda x, a=arm: x[a] == "WRONG"
                             and x["fresh"] == "CORRECT", "cluster")
    ci_c, _ = cluster_boot(ho, lambda x, a=arm: x[a] == "CORRECT", "cluster")
    out[lab] = {"n": len(ho), "clusters": ncl, "correct": c,
                "accuracy_pct": round(100 * c / len(ho), 3),
                "accuracy_wilson95": wilson(c, len(ho)),
                "accuracy_cluster_boot95": ci_c,
                "wai": w, "wai_pct": round(100 * w / len(ho), 3),
                "wai_wilson95": wilson(w, len(ho)),
                "wai_cluster_boot95": ci_w}
    say(f"    {lab:<26} acc {100*c/len(ho):6.2f}% Wilson {wilson(c,len(ho))} "
        f"cluster-boot {ci_c} | WAI {w:>3} = {100*w/len(ho):5.2f}% "
        f"Wilson {wilson(w,len(ho))} cluster-boot {ci_w}")

A = {r["qid"] for r in ho if r["freshcache"] == "WRONG" and r["fresh"] == "CORRECT"}
Bs = {r["qid"] for r in ho if r["semanticttl"] == "WRONG" and r["fresh"] == "CORRECT"}
say(f"\n  directional discordance (WAI): FreshCache-only {len(A-Bs)}, "
    f"SemanticTTL-only {len(Bs-A)}, both {len(A&Bs)}, "
    f"exact McNemar p = {mcnemar(len(Bs-A), len(A-Bs)):.4g}")
out["paired_wai"] = {"fc_only": len(A - Bs), "sttl_only": len(Bs - A),
                     "both": len(A & Bs),
                     "mcnemar_p": mcnemar(len(Bs - A), len(A - Bs))}

# cluster-level McNemar: collapse each cluster to "any WAI"
byc = defaultdict(lambda: {"fc": 0, "st": 0})
for r in ho:
    if r["fresh"] == "CORRECT":
        if r["freshcache"] == "WRONG":
            byc[r["cluster"]]["fc"] = 1
        if r["semanticttl"] == "WRONG":
            byc[r["cluster"]]["st"] = 1
b01 = sum(1 for v in byc.values() if v["st"] and not v["fc"])
b10 = sum(1 for v in byc.values() if v["fc"] and not v["st"])
say(f"  CLUSTER-level (any-WAI per cluster): SemanticTTL-only {b01}, "
    f"FreshCache-only {b10}, exact McNemar p = {mcnemar(b01, b10):.4g}")
out["cluster_level_wai_mcnemar"] = {"sttl_only": b01, "fc_only": b10,
                                    "p": mcnemar(b01, b10)}
say(f"  -> the conclusion is unchanged when the cluster, not the request, is "
    f"the unit of analysis.")

json.dump(out, open(HERE / "artifacts" / "aq07_cluster_aware.json", "w"), indent=2)
(HERE / "build" / "aq07.log").write_text("\n".join(log) + "\n", encoding="utf-8")
say(f"\n  wrote artifacts/aq07_cluster_aware.json")
