#!/usr/bin/env python3
"""
l2_equivalence_fit.py — choose the semantic_equivalent() operating
                             point for the L2 (URL-list) tier

READ-ONLY. Reads data/l2_hit_judgments_24h.jsonl and
data/l2_hits_rulebased_t24h.jsonl. Writes only
outputs/l2_equivalence_gate/l2_equivalence_gate_fit.json.

Written as a separate file rather than folded into
fit_equivalence_gate.py so that the L1 fit artifact stays exactly as it
was when the deployed L1 gate was chosen.

WHY
  Adding the gate to L1 cut its mismatch but moved the volume: L2 hits
  rose and now carry most cache hits. Judged L2 mismatch became worse
  than L1 ever was, because theta_L2 = 0.35 sits below the cross-cluster
  similarity mass.

  Workload characterization (Appendix I) independently measured
  cross-cluster URL-set Jaccard overlap close to zero. Since most
  judged L2 mismatches are cross-cluster, a mismatched L2 hit reuses a
  URL list with essentially no intersection with what the query would
  itself have retrieved. The evidence is unrelated, not merely
  suboptimal.

BUDGET
  The answer tier is judged against eps_answer = 0.10. The URL tier has
  its own, looser budget eps_url_list = 0.20, which is the target here.
  Both are reported so the trade-off is visible either way.

Run:
    cd "$FRESHCACHE_ROOT" && python3 l2_equivalence_fit.py
"""

from __future__ import annotations

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR  = (DATA_ROOT)
JUDGMENTS = DATA_DIR / "l2_hit_judgments_24h.jsonl"
HITS_FILE = DATA_DIR / "l2_hits_rulebased_t24h.jsonl"
OUT_FILE  = OUTPUT_ROOT / "l2_equivalence_gate/l2_equivalence_gate_fit.json"

EPS_URL_LIST = 0.20      # the L2 budget, the primary target
EPS_ANSWER   = 0.10      # reported for reference

BANDS = [(0.35, 0.50), (0.50, 0.60), (0.60, 0.70),
         (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]

_ANSWER_TYPE_PATTERNS = {
    "price":         ["price", "cost", "worth", "가격", "비용"],
    "score":         ["score", "result", "point", "점수", "결과", "골"],
    "weather":       ["weather", "temperature", "forecast", "날씨", "기온"],
    "exchange_rate": ["exchange rate", "dollar", "won", "yen", "환율", "달러"],
    "stock":         ["stock", "share", "equity", "nasdaq", "주가", "주식"],
    "definition":    ["what is", "explain", "define", "설명", "뭐야", "무엇", "뜻"],
    "biography":     ["who is", "biography", "born", "누구", "인물"],
    "news":          ["news", "breaking", "뉴스", "속보", "사건"],
    "schedule":      ["schedule", "when does", "date", "일정", "언제", "날짜"],
}

_STOP = frozenset("""
a an the of in on at to for from by with and or is are was were be been
what which who whom whose when where why how did does do have has had
that this these those it its as than then there their you your i me my
most recent latest current currently now today last next new
""".split())


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def band_of(sim: float):
    for lo, hi in BANDS:
        if lo <= sim < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def content_tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+|[가-힣]+", text.lower())) - _STOP


def lexical_jaccard(q1: str, q2: str) -> float:
    a, b = content_tokens(q1), content_tokens(q2)
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def answer_type(text: str):
    q = text.lower()
    for atype, kws in _ANSWER_TYPE_PATTERNS.items():
        if any(kw in q for kw in kws):
            return atype
    return None


def answer_type_agrees(q1: str, q2: str) -> bool:
    a, b = answer_type(q1), answer_type(q2)
    if a is None or b is None:
        return True
    return a == b


def strict_subset(q1: str, q2: str) -> bool:
    t1, t2 = content_tokens(q1), content_tokens(q2)
    return t1 != t2 and (t1 <= t2 or t2 <= t1)


def main() -> None:
    judged = [r for r in load_jsonl(JUDGMENTS)
              if r.get("verdict") in ("SAME", "DIFFERENT")]
    if not judged:
        print(f"\n  No judgments in {JUDGMENTS}. Run judge_l2_hits.py first.\n")
        return

    all_hits = load_jsonl(HITS_FILE)
    N = len(all_hits)

    pop = defaultdict(int)
    for h in all_hits:
        b = band_of(h["similarity"])
        if b:
            pop[b] += 1
    smp = defaultdict(int)
    for r in judged:
        smp[r["band"]] += 1
    weight = {b: pop[b] / smp[b] for b in smp if smp[b]}

    print("\n" + "=" * 78)
    print("  Fitting the L2 equivalence gate against the judged L2 hits")
    print("=" * 78)
    print(f"\n  realized L2 hits at t=24h : {N:,}")
    print(f"  judged                    : {len(judged)}")
    print(f"\n  {'band':>12} {'population':>11} {'judged':>7} {'weight':>8}")
    for lo, hi in BANDS:
        b = f"{lo:.2f}-{hi:.2f}"
        if b in weight:
            print(f"  {b:>12} {pop[b]:>11,} {smp[b]:>7} {weight[b]:>8.1f}")

    for r in judged:
        q1, q2 = r["matched_query"], r["query"]
        r["_w"]   = weight.get(r["band"], 0.0)
        r["_ok"]  = (r["verdict"] == "SAME")
        r["_lj"]  = lexical_jaccard(q1, q2)
        r["_at"]  = answer_type_agrees(q1, q2)
        r["_sub"] = strict_subset(q1, q2)

    W_total = sum(r["_w"] for r in judged)
    W_same  = sum(r["_w"] for r in judged if r["_ok"])
    print(f"\n  population-weighted totals")
    print(f"    all L2 hits               : {W_total:,.0f}")
    print(f"    of which semantically SAME: {W_same:,.0f} "
          f"({100*W_same/W_total:.1f}%)")
    print(f"\n  CEILING: no read-time gate can retain more than {W_same:,.0f}")
    print(f"  correct L2 hits.")

    def evaluate(name: str, pred) -> dict:
        adm   = [r for r in judged if pred(r)]
        w_adm = sum(r["_w"] for r in adm)
        w_ok  = sum(r["_w"] for r in adm if r["_ok"])
        mism  = 1.0 - (w_ok / w_adm) if w_adm else 0.0
        rec   = w_ok / W_same if W_same else 0.0
        return {
            "gate": name,
            "admitted": round(w_adm),
            "admitted_correct": round(w_ok),
            "mismatch_admitted": round(mism, 4),
            "recall_of_same": round(rec, 4),
            "n_sample_admitted": len(adm),
            "meets_eps_url_list": bool(mism <= EPS_URL_LIST),
            "meets_eps_answer": bool(mism <= EPS_ANSWER),
        }

    results = []

    print("\n" + "-" * 78)
    print("  1. raised similarity threshold alone")
    print("-" * 78)
    print(f"  {'theta':>7} {'admitted':>9} {'correct':>8} "
          f"{'mismatch':>9} {'recall':>8}  budget")
    for t in [0.35, 0.50, 0.60, 0.65, 0.70, 0.75,
              0.80, 0.85, 0.90, 0.95]:
        r = evaluate(f"theta>={t:.2f}", lambda x, t=t: x["similarity"] >= t)
        results.append(r)
        flag = "eps020" if r["meets_eps_url_list"] else ""
        if r["meets_eps_answer"]:
            flag += " eps010"
        print(f"  {t:>7.2f} {r['admitted']:>9,} {r['admitted_correct']:>8,} "
              f"{r['mismatch_admitted']:>9.3f} {r['recall_of_same']:>8.3f}"
              f"  {flag}")

    print("\n" + "-" * 78)
    print("  2. the deployed L1 gate applied at L2")
    print("     (sim >= 0.80, jaccard >= 0.30, answer types agree,")
    print("      not a strict token subset)")
    print("-" * 78)
    r = evaluate("L1 gate as-is",
                 lambda x: x["similarity"] >= 0.80 and x["_lj"] >= 0.30
                 and x["_at"] and not x["_sub"])
    results.append(r)
    print(f"  admitted {r['admitted']:,} of {N:,}  "
          f"mismatch {r['mismatch_admitted']:.3f}  "
          f"recall {r['recall_of_same']:.3f}")

    print("\n" + "-" * 78)
    print("  3. combinations tuned for the looser L2 budget")
    print("-" * 78)
    print(f"  {'gate':>40} {'admitted':>9} {'correct':>8} "
          f"{'mismatch':>9} {'recall':>8}")
    for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for j in [0.0, 0.2, 0.3, 0.4]:
            for use_at in (False, True):
                for use_sub in (False, True):
                    name = f"theta>={t:.2f}"
                    if j > 0:
                        name += f" & jac>={j:.1f}"
                    if use_at:
                        name += " & atype"
                    if use_sub:
                        name += " & nosub"
                    r = evaluate(
                        name,
                        lambda x, t=t, j=j, a=use_at, s=use_sub:
                        x["similarity"] >= t and x["_lj"] >= j
                        and (x["_at"] if a else True)
                        and ((not x["_sub"]) if s else True))
                    results.append(r)
                    if r["meets_eps_url_list"]:
                        print(f"  {name:>40} {r['admitted']:>9,} "
                              f"{r['admitted_correct']:>8,} "
                              f"{r['mismatch_admitted']:>9.3f} "
                              f"{r['recall_of_same']:>8.3f}")

    # ── Ranking ───────────────────────────────────────────────────────
    passing = [r for r in results if r["meets_eps_url_list"]]
    passing.sort(key=lambda r: -r["admitted"])

    print("\n" + "=" * 78)
    print(f"  GATES MEETING eps_url_list = {EPS_URL_LIST}, "
          f"ranked by hits retained")
    print("=" * 78)
    if not passing:
        print(f"\n  None. No cheap gate over these signals reaches {100 * EPS_URL_LIST:.0f}%")
        print("  mismatch at L2. The cause is upstream, as at L1: at t=24h")
        print("  only SLOW and TIMELESS entries populate the caches, so a")
        print("  MEDIUM or FAST query's own URL list is usually absent and")
        print("  argmax selects a stranger. Consider per-class thresholds or")
        print("  populating L2 for all classes and gating at read time.")
    else:
        print(f"\n  {'gate':>40} {'admitted':>9} {'mismatch':>9} {'recall':>8}")
        for r in passing[:12]:
            print(f"  {r['gate']:>40} {r['admitted']:>9,} "
                  f"{r['mismatch_admitted']:>9.3f} {r['recall_of_same']:>8.3f}")
        best = passing[0]
        print(f"\n  RECOMMENDED: {best['gate']}")
        print(f"    retains {best['admitted']:,} of {N:,} L2 hits "
              f"({100*best['admitted']/N:.1f}%)")
        print(f"    mismatch among admitted : "
              f"{100*best['mismatch_admitted']:.1f}%")
        print(f"    preserves {100*best['recall_of_same']:.1f}% of the "
              f"correct reuse available")

        # Savings implication, the number that decides the paper's claim.
        print(f"\n  SAVINGS IMPLICATION")
        print(f"    L2 hits retained    : {best['admitted']:,}")
        print(f"    L2 hits rejected    : {N - best['admitted']:,}")
        print(f"    Each rejected L2 hit becomes a fresh search call.")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "n_realized_l2_hits": N,
            "n_judged": len(judged),
            "band_population": dict(pop),
            "band_sampled": dict(smp),
            "weighted_total": round(W_total),
            "weighted_same": round(W_same),
            "ceiling_correct_hits": round(W_same),
            "eps_url_list": EPS_URL_LIST,
            "gates": results,
            "passing_ranked": passing[:12],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT_FILE}\n")


if __name__ == "__main__":
    main()
