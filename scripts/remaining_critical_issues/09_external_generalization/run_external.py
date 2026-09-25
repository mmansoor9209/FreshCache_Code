#!/usr/bin/env python3
"""
Section 9 -- SYMMETRIC external generalization on EvolvingQA and DailyQA.

Both datasets are loaded through ONE interface into the same row schema and
scored by ONE evaluator with ONE metric set. Every policy runs on both
datasets; nothing is retuned per dataset.

  question            the incoming query
  cached_answer       the answer a cache created one snapshot earlier
  fresh_answer        the answer a fresh retrieval gives now  (the reference)
  changed             cached_answer != fresh_answer

Freshness classes are assigned at INFERENCE TIME from the question text by the
production classifier (freshcache.risk_model._classify_freshness) for BOTH
datasets -- no dataset-specific category field, no change label, no leakage.

Policies (published FreshCache parameters, frozen, no retuning):
  NoCache        never reuse
  ExactNoTTL     always reuse
  ExactTTL       reuse iff age <= experiment.FIXED_TTL[class]
  SemanticTTL    reuse iff sim(q, q) >= theta AND age <= FIXED_TTL[class]
  FreshCache-L1Only / FreshCache
                 reuse iff p_stale(class, age, "answer") <= EPS_ANSWER,
                 the published exponential risk gate

STRUCTURAL LIMIT, stated rather than worked around: neither dataset carries the
retrieved URL list or the page bodies behind an answer, so L2 (URL-list reuse)
and L3 (content reuse) are UNDEFINED on both. FreshCache therefore reduces to
FreshCache-L1Only here, and no L2/L3 activity, fetch count or content-drift
figure can be reported for either dataset. The same limitation applies to both,
so the comparison stays symmetric.

Every request in these datasets is the SAME question re-asked one snapshot
later, so an exact-match cache hits whenever the TTL allows: ExactTTL and
SemanticTTL are therefore numerically identical here, and both are reported.

Read-only. CPU only.
"""
from __future__ import annotations
import csv, json, math, os, pathlib, re, sys, time
from collections import Counter, defaultdict
from fractions import Fraction

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(BASE / "lib"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import experiment as exp                 # noqa: E402
import v3_engine as v3                   # noqa: E402
from freshcache.risk_model import (      # noqa: E402
    _detect_answer_type, _detect_temporal_keywords, _classify_freshness)

AGE = 24 * 3600.0
NATURAL_AGE = {"EvolvingQA": 30 * 86400.0, "DailyQA": 86400.0}
DQ = ROOT / "external" / "DailyQA" / "data" / "qa"


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 2), round(100 * min(1.0, (c + m) / d), 2))


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar, computed in exact rationals."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tot = sum(Fraction(math.comb(n, i)) for i in range(0, k + 1))
    p = 2 * tot / Fraction(2) ** n
    return float(min(Fraction(1), p))


_WS = re.compile(r"\s+")


def normalize(t):
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def answers_match(a, b):
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    if len(nb) <= 40 and nb in na:
        return True
    if len(na) <= 40 and na in nb:
        return True
    return False


def fclass(q):
    return _classify_freshness(q, _detect_temporal_keywords(q),
                               _detect_answer_type(q)).value


# ------------------------------------------------------------------ loaders
def load_evolvingqa(cap_edited=10_000, cap_unchanged=33_456, seed=42):
    """EvolvingQA's real schema:
         config "edited"    question / answer1 / answer2, one row per Wikipedia
                            edit between two CONSECUTIVE MONTHLY snapshots
                            (splits edited_0203 .. edited_0708) --
                            cached_answer = answer1, fresh_answer = answer2.
         config "unchanged" question / answer, a question whose answer did not
                            move -- cached_answer = fresh_answer = answer.
    The published external evaluation caps the mix at 10,000 edited and 33,456
    unchanged rows; the same caps and the same seed are used here so this arm
    stays comparable with the published EvolvingQA result. Sampling is seeded
    and reported."""
    import random
    from datasets import load_dataset, get_dataset_split_names
    ed, un = [], []
    for sp in get_dataset_split_names("kat-research/EvolvingQA", "edited"):
        for r in load_dataset("kat-research/EvolvingQA", "edited", split=sp):
            q = (r.get("question") or "").strip()
            a1 = (r.get("answer1") or "").strip()
            a2 = (r.get("answer2") or "").strip()
            if q and a1 and a2:
                ed.append({"question": q, "cached_answer": a1,
                           "fresh_answer": a2, "subset": f"edited:{sp}"})
    for sp in get_dataset_split_names("kat-research/EvolvingQA", "unchanged"):
        for r in load_dataset("kat-research/EvolvingQA", "unchanged", split=sp):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                un.append({"question": q, "cached_answer": a,
                           "fresh_answer": a, "subset": f"unchanged:{sp}"})
    rng = random.Random(seed)
    rng.shuffle(ed); rng.shuffle(un)
    rows = ed[:cap_edited] + un[:cap_unchanged]
    rng.shuffle(rows)
    print(f"  EvolvingQA pool: edited {len(ed):,}  unchanged {len(un):,}  "
          f"-> sampled {len(rows):,} (caps {cap_edited:,}/{cap_unchanged:,}, "
          f"seed {seed})", flush=True)
    return rows


def load_dailyqa():
    """Consecutive-day pairs: the same question asked on day d and day d+1.
    cached_answer = day d's answer, fresh_answer = day d+1's answer."""
    days = {}
    for p in sorted(DQ.glob("qa_*.jsonl")):
        by_q = {}
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = (d.get("question") or d.get("query") or "").strip()
            a = d.get("answer") or d.get("gold_answer") or d.get("answers")
            if isinstance(a, list):
                a = a[0] if a else ""
            a = (a or "").strip()
            if q and a:
                by_q[q] = a
        days[p.stem] = by_q
    keys = sorted(days)
    rows = []
    for d0, d1 in zip(keys, keys[1:]):
        for q, a0 in days[d0].items():
            a1 = days[d1].get(q)
            if a1 is None:
                continue
            rows.append({"question": q, "cached_answer": a0, "fresh_answer": a1,
                         "subset": f"{d0}->{d1}"})
    return rows


# ------------------------------------------------------------------ policies
def reuse_decision(policy, fc, age, theta_ok=True):
    if policy == "NoCache":
        return False
    if policy == "ExactNoTTL":
        return True
    if policy in ("ExactTTL", "SemanticTTL"):
        if policy == "SemanticTTL" and not theta_ok:
            return False
        v = exp.FIXED_TTL.get(fc, 0)
        return bool(v and age <= v)
    if policy in ("FreshCache", "FreshCache-L1Only"):
        return exp.p_stale(fc, age, "answer") <= exp.EPS_ANSWER
    raise ValueError(policy)


POLICIES = ["NoCache", "ExactNoTTL", "ExactTTL", "SemanticTTL",
            "FreshCache-L1Only", "FreshCache"]


def evaluate(rows, name, log, AGE=AGE, tag=""):
    def say(s=""):
        print(s, flush=True); log.append(s)

    for r in rows:
        r["fc"] = fclass(r["question"])
        r["changed"] = not answers_match(r["cached_answer"], r["fresh_answer"])
    cls = Counter(r["fc"] for r in rows)
    nch = sum(r["changed"] for r in rows)
    say(f"\n  ==== {name}{tag} ====")
    say(f"    replay age {AGE/86400:.2f} d")
    say(f"    pairs {len(rows):,}   unique questions "
        f"{len({r['question'] for r in rows}):,}   "
        f"changed answers {nch:,} ({100*nch/len(rows):.2f}%)")
    say(f"    inference-time predicted classes: "
        f"{ {k: cls[k] for k in sorted(cls)} }")
    say(f"    L2 / L3 activity, fetches and content drift: NOT DEFINED "
        f"(no URL lists or page bodies in this dataset)")

    out, per = {}, {}
    for pol in POLICIES:
        reuse = corr = wai = 0
        vec = []
        for r in rows:
            u = reuse_decision(pol, r["fc"], AGE)
            served = r["cached_answer"] if u else r["fresh_answer"]
            ok = answers_match(served, r["fresh_answer"])
            reuse += u
            corr += ok
            wai += int(u and not ok)     # fresh is correct by construction
            vec.append(ok)
        per[pol] = vec
        n = len(rows)
        out[pol] = {
            "n_pairs": n, "reuse": reuse,
            "reuse_rate_pct": round(100 * reuse / n, 4),
            "search_saved_pct": round(100 * reuse / n, 4),
            "accuracy_pct": round(100 * corr / n, 4),
            "accuracy_ci": wilson(corr, n),
            "cache_induced_error_n": wai,
            "cache_induced_error_per_query_pct": round(100 * wai / n, 4),
            "wai_conditional_on_hit_pct": (round(100 * wai / reuse, 4) if reuse else None),
            "cwai_ci": wilson(wai, reuse) if reuse else None,
            "coverage_pct": 100.0,       # every pair has a reference answer
            "l1_hits": reuse, "l2_hits": None, "l3_hits": None,
            "fetches": None, "content_drift_pct": None}
    say(f"\n    {'policy':<20}{'reuse%':>9}{'acc%':>9}{'acc 95% CI':>18}"
        f"{'CIE/query%':>12}{'C-WAI%':>9}{'n hits':>9}")
    for pol in POLICIES:
        r = out[pol]
        cw = f"{r['wai_conditional_on_hit_pct']:.2f}" if r['wai_conditional_on_hit_pct'] is not None else "-"
        say(f"    {pol:<20}{r['reuse_rate_pct']:>8.2f}%{r['accuracy_pct']:>8.2f}%"
            f"{str(r['accuracy_ci']):>18}{r['cache_induced_error_per_query_pct']:>11.2f}%"
            f"{cw:>9}{r['l1_hits']:>9,}")
    # exact McNemar, every policy against NoCache (the fresh reference)
    say(f"\n    exact McNemar vs NoCache (paired per-pair correctness)")
    base = per["NoCache"]
    for pol in POLICIES[1:]:
        v = per[pol]
        b = sum(1 for a, x in zip(base, v) if a and not x)
        c = sum(1 for a, x in zip(base, v) if x and not a)
        p = mcnemar_exact(b, c)
        out[pol]["mcnemar_vs_nocache"] = {"b_correct_to_wrong": b,
                                          "c_wrong_to_correct": c, "p": p}
        say(f"      {pol:<20} C->W {b:>6,}  W->C {c:>6,}  p = "
            f"{p:.4g}" + ("  (not significant; NOT an equivalence claim)"
                          if p > 0.05 else ""))
    # per-class breakdown, identical for both datasets
    byc = {}
    for pol in ("ExactTTL", "FreshCache"):
        byc[pol] = {}
        for fc in sorted(cls):
            sub = [r for r in rows if r["fc"] == fc]
            if not sub:
                continue
            ru = sum(reuse_decision(pol, r["fc"], AGE) for r in sub)
            er = sum(1 for r in sub if reuse_decision(pol, r["fc"], AGE)
                     and not answers_match(r["cached_answer"], r["fresh_answer"]))
            byc[pol][fc] = {"n": len(sub), "reuse": ru,
                            "reuse_pct": round(100 * ru / len(sub), 2),
                            "cache_induced_error": er}
    return {"n_pairs": len(rows), "changed": nch,
            "class_distribution": dict(cls), "policies": out,
            "per_class": byc}


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("V3 SECTION 9 -- symmetric external generalization")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    say(f"  frozen published parameters, identical on both datasets:")
    say(f"    HALF_LIFE  {dict(exp.HALF_LIFE)}")
    say(f"    FIXED_TTL  {dict(exp.FIXED_TTL)}")
    say(f"    EPS_ANSWER {exp.EPS_ANSWER}   TIER_MULT[answer] {exp.TIER_MULT['answer']}")
    say(f"    replay age {AGE/3600:.0f} h on both datasets")
    say(f"  no dataset-specific retuning of any kind")

    res = {}
    for name, loader in (("EvolvingQA", load_evolvingqa),
                         ("DailyQA", load_dailyqa)):
        try:
            rows = loader()
        except Exception as e:
            say(f"\n  ==== {name} ====\n    LOAD FAILED: {e}")
            res[name] = {"error": str(e)}
            continue
        if not rows:
            say(f"\n  ==== {name} ====\n    no usable pairs")
            res[name] = {"error": "no usable pairs"}
            continue
        res[name] = evaluate(rows, name, log, AGE, "  [common 24 h age]")
        nat = NATURAL_AGE[name]
        if abs(nat - AGE) > 1:
            res[name + "@natural"] = evaluate(
                rows, name, log, nat,
                f"  [dataset's own snapshot interval, {nat/86400:.0f} d]")

    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "age_seconds": AGE, "policies": POLICIES,
               "frozen_params": {"HALF_LIFE": dict(exp.HALF_LIFE),
                                 "FIXED_TTL": dict(exp.FIXED_TTL),
                                 "EPS_ANSWER": exp.EPS_ANSWER},
               "structural_limits": {
                   "L2_L3_undefined": "neither dataset carries retrieved URL "
                                      "lists or page bodies",
                   "ExactTTL_equals_SemanticTTL": "every pair re-asks the same "
                                                  "question, so the similarity "
                                                  "gate is always satisfied"},
               "results": res}, open(HERE / "external_results.json", "w"),
              indent=2)
    with open(HERE / "external_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "policy", "n_pairs", "reuse_rate_pct",
                    "accuracy_pct", "accuracy_ci_lo", "accuracy_ci_hi",
                    "cache_induced_error_per_query_pct",
                    "wai_conditional_on_hit_pct", "l1_hits",
                    "l2_hits", "l3_hits", "content_drift_pct",
                    "mcnemar_p_vs_nocache"])
        for ds, blk in res.items():
            if "policies" not in blk:
                continue
            for pol, r in blk["policies"].items():
                ci = r["accuracy_ci"]
                w.writerow([ds, pol, r["n_pairs"], r["reuse_rate_pct"],
                            r["accuracy_pct"], ci[0], ci[1],
                            r["cache_induced_error_per_query_pct"],
                            r["wai_conditional_on_hit_pct"], r["l1_hits"],
                            "undefined", "undefined", "undefined",
                            r.get("mcnemar_vs_nocache", {}).get("p", "")])
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("\n  wrote external_results.json/.csv, run.log")


if __name__ == "__main__":
    main()
