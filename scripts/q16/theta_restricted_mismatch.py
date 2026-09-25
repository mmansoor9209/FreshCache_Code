"""Q-16: does the baseline's query-equivalence mismatch stay high when the
baseline is NOT at its permissive threshold?

Table 'mismatch' reports SemanticTTL at theta=0.40, and its own caption
concedes that the permissive setting is why the rate is so high. The
quality-selected operating point is theta=0.60, kappa=1/2. This re-analyses
the ALREADY-JUDGED sample restricted to hits at similarity >= 0.60 and
reweights to the corresponding population.

Gate: the unrestricted computation must reproduce the published 95.4%
[94.0, 96.8] before any restricted number is believed.
"""
import json, pathlib, collections, math

ROOT = pathlib.Path("<PROJECT_ROOT>")
OUT  = pathlib.Path(__file__).resolve().parent.parent.parent/"artifacts/q16"
OUT.mkdir(parents=True, exist_ok=True)
JURORS = ["llama8b", "mistral7b", "qwen7b"]

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ---- judged sample -------------------------------------------------------
per_juror = {j: {r["key"]: r for r in jl(ROOT/f"v8/jury/{j}__semanticttl.jsonl")}
             for j in JURORS}
keys = set.intersection(*(set(v) for v in per_juror.values()))
print(f"judged hits with all {len(JURORS)} jurors: {len(keys)}")

rows = []
for k in sorted(keys):
    v = [per_juror[j][k]["verdict"] for j in JURORS]
    diff = sum(1 for x in v if x == "DIFFERENT")
    base = per_juror[JURORS[0]][k]
    rows.append({"key": k, "similarity": base["similarity"], "band": base["band"],
                 "verdicts": v, "n_different": diff,
                 "majority_different": diff > len(JURORS)/2})

# ---- population ----------------------------------------------------------
pop = jl(ROOT/"data/semanticttl_hits_t24h.jsonl")
print(f"realized-hit population: {len(pop):,} (published table: 22,561)")
assert len(pop) == 22561, len(pop)

def band_of(s):
    for a, b in [(0.35,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),
                 (0.70,0.80),(0.80,0.90),(0.90,1.01)]:
        if a <= s < b: return f"{a:.2f}-{b:.2f}"
    return "other"

# use the bands the sample was actually stratified on
samp_bands = sorted({r["band"] for r in rows})
print(f"sample strata: {samp_bands}")

def in_band(s, b):
    a, c = b.split("-"); return float(a) <= s < float(c)

def estimate(rows_sub, pop_sub, label):
    """Reweight the judged sample to the population, stratum by stratum."""
    by_band_n = collections.Counter()
    for p in pop_sub:
        for b in samp_bands:
            if in_band(p["similarity"], b): by_band_n[b] += 1; break
    num = den = 0.0
    detail = {}
    for b in samp_bands:
        sj = [r for r in rows_sub if r["band"] == b]
        N = by_band_n[b]
        if not sj or not N: continue
        rate = sum(r["majority_different"] for r in sj)/len(sj)
        num += rate*N; den += N
        detail[b] = {"judged": len(sj), "population": N, "mismatch_rate": round(rate, 4)}
    est = num/den if den else float("nan")
    # cluster-free Wilson interval on the effective judged count
    n_eff = sum(d["judged"] for d in detail.values())
    z = 1.959963985
    p = est; dd = 1+z*z/n_eff
    c = (p+z*z/(2*n_eff))/dd
    h = z*math.sqrt(p*(1-p)/n_eff + z*z/(4*n_eff*n_eff))/dd
    print(f"\n{label}")
    print(f"  population {int(den):,}   judged {n_eff}")
    print(f"  reweighted mismatch {100*est:.1f}%  Wilson 95% CI [{100*(c-h):.1f}, {100*(c+h):.1f}]")
    for b, d in sorted(detail.items()):
        print(f"    {b}  judged {d['judged']:>3}  pop {d['population']:>6,}  rate {100*d['mismatch_rate']:.1f}%")
    return {"label": label, "population": int(den), "judged": n_eff,
            "mismatch_pct": round(100*est,2),
            "ci95": [round(100*(c-h),2), round(100*(c+h),2)], "by_band": detail}

def fleiss(rows_sub):
    n = len(JURORS); N = len(rows_sub)
    if not N: return float("nan")
    Pi = []
    for r in rows_sub:
        d = r["n_different"]; s = n-d
        Pi.append((d*(d-1)+s*(s-1))/(n*(n-1)))
    Pbar = sum(Pi)/N
    pj = sum(r["n_different"] for r in rows_sub)/(N*n)
    Pe = pj**2+(1-pj)**2
    return (Pbar-Pe)/(1-Pe) if Pe < 1 else float("nan")

print("\n[GATE] unrestricted, must reproduce the published 95.4% [94.0, 96.8]")
full = estimate(rows, pop, "SemanticTTL, theta=0.40 (published operating point)")
print(f"  Fleiss kappa {fleiss(rows):.3f}   (published 0.58)")
gate_ok = abs(full["mismatch_pct"] - 95.4) <= 1.0
print(f"  GATE {'PASSED' if gate_ok else 'FAILED'}: got {full['mismatch_pct']}%, published 95.4%")

res = {"gate": {"published_pct": 95.4, "reproduced_pct": full["mismatch_pct"],
                "passed": bool(gate_ok)}, "unrestricted": full}

if gate_ok:
    rows60 = [r for r in rows if r["similarity"] >= 0.60]
    pop60  = [p for p in pop  if p["similarity"] >= 0.60]
    print(f"\n[RESTRICTED] theta >= 0.60: {len(pop60):,} of {len(pop):,} realized hits "
          f"({100*len(pop60)/len(pop):.1f}%) survive the stricter threshold")
    sel = estimate(rows60, pop60, "SemanticTTL restricted to similarity >= 0.60")
    k60 = fleiss(rows60)
    print(f"  Fleiss kappa {k60:.3f}")
    res["restricted_theta_060"] = sel
    res["restricted_fleiss_kappa"] = round(k60, 3)
    res["population_surviving_pct"] = round(100*len(pop60)/len(pop), 2)
res["fleiss_kappa_unrestricted"] = round(fleiss(rows), 3)
res["jurors"] = JURORS
res["caveat"] = ("This is a threshold-restricted RE-ANALYSIS of the judged sample from "
                 "the published theta=0.40 replay, not a fresh replay at (0.60, 1/2). "
                 "Raising theta changes which entries are admitted and therefore the "
                 "cache's later state, so the selected point's realized-hit set is not "
                 "exactly this subset. The kappa=1/2 TTL halving is NOT applied: the "
                 "released hit population carries similarity but not hit age.")
json.dump(res, open(OUT/"q16_theta_restricted_mismatch.json","w"), indent=1)
print("\nwrote artifacts/q16/q16_theta_restricted_mismatch.json")
