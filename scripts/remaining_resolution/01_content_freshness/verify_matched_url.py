#!/usr/bin/env python3
"""
Verification of the reported +0.783 pp same-URL (level 3) freshness
difference, from the SAVED mixed-age page-level rows only.

No replay is re-run, no parameter retuned, no page recollected, no existing
result modified. Inputs: pages_mixed_on.csv, pages_mixed_off.csv, plus the
query records (for cluster ids only).
"""
from __future__ import annotations
import csv, json, math, os, pathlib, random, statistics, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
CHANGED, UNCHANGED, UNOBS = "CHANGED", "UNCHANGED", "UNOBSERVABLE"
SEED, B = 42, 10_000


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 4),
            round(100 * min(1.0, (c + m) / d), 4))


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("VERIFICATION -- mixed-age same-URL (level 3) freshness difference")
    say("  source: saved page-level rows only; nothing re-run or modified")

    ON = list(csv.DictReader(open(HERE / "pages_mixed_on.csv", encoding="utf-8")))
    OFF = list(csv.DictReader(open(HERE / "pages_mixed_off.csv", encoding="utf-8")))
    say(f"  pages_mixed_on.csv  {len(ON):,} page rows")
    say(f"  pages_mixed_off.csv {len(OFF):,} page rows")

    # cluster ids (needed for the paired cluster bootstrap)
    import experiment as exp
    q = exp.load_jsonl(exp.QUERIES_FILE)
    mf = exp.load_jsonl(exp.MANIFEST_FILE)
    pa = (exp.load_jsonl(exp.PARAPHRASE_FILE)
          if exp.PARAPHRASE_FILE.exists() else [])
    recs = exp.build_query_records(q, mf, pa)
    cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in recs}

    def byreq(rows):
        d = defaultdict(list)
        for r in rows:
            d[r["request_id"]].append(r)
        return d
    RON, ROFF = byreq(ON), byreq(OFF)

    # ---- level 3: both expose content AND identical URL identity sets ----
    matched = [rq for rq in RON
               if rq in ROFF
               and sorted(x["url_hash"] for x in RON[rq])
               == sorted(x["url_hash"] for x in ROFF[rq])]
    mset = set(matched)
    pon = [p for p in ON if p["request_id"] in mset]
    poff = [p for p in OFF if p["request_id"] in mset]
    say(f"\n  == 1. matched population ==")
    say(f"    matched requests                 {len(matched):,}")
    say(f"    reused page observations  gate ON {len(pon):,}   "
        f"gate OFF {len(poff):,}")
    say(f"    unique URLs                      "
        f"{len({p['url_hash'] for p in pon}):,}")
    say(f"    independent request clusters     "
        f"{len({cid.get(rq, rq) for rq in matched}):,}")
    say(f"    (page rows are identical in COUNT and URL IDENTITY between arms "
        f"by the definition of level 3; only the artifact version can differ)")

    # ---- 2. change counts and denominators ----
    def stat(pgs):
        obs = [p for p in pgs if p["outcome"] in (CHANGED, UNCHANGED)]
        ch = sum(1 for p in obs if p["outcome"] == CHANGED)
        uo = sum(1 for p in pgs if p["outcome"] == UNOBS)
        return {"pages": len(pgs), "observable": len(obs), "changed": ch,
                "unobservable": uo,
                "rate_pct": round(100*ch/len(obs), 4) if obs else None,
                "wilson": wilson(ch, len(obs))}
    A, Bs = stat(pon), stat(poff)
    say(f"\n  == 2. change counts and denominators ==")
    say(f"    {'arm':<10}{'pages':>9}{'observable':>12}{'changed':>9}"
        f"{'unobs':>8}{'rate':>10}{'Wilson 95% CI':>22}")
    for nm, e in (("gate ON", A), ("gate OFF", Bs)):
        say(f"    {nm:<10}{e['pages']:>9,}{e['observable']:>12,}"
            f"{e['changed']:>9,}{e['unobservable']:>8,}"
            f"{e['rate_pct']:>9.4f}%{str(e['wilson']):>22}")
    naive = Bs["rate_pct"] - A["rate_pct"]
    say(f"    difference OFF - ON = {naive:+.4f} pp   "
        f"(reported +0.783; reproduces: {abs(naive-0.783) < 0.01})")

    # ---- the genuinely paired unit: (request, url) present in both arms ----
    kon = {(p["request_id"], p["url_hash"]): p for p in pon}
    koff = {(p["request_id"], p["url_hash"]): p for p in poff}
    keys = sorted(set(kon) & set(koff))
    both_obs = [k for k in keys
                if kon[k]["outcome"] in (CHANGED, UNCHANGED)
                and koff[k]["outcome"] in (CHANGED, UNCHANGED)]
    say(f"\n    paired at the (request, URL) level: {len(keys):,} shared units; "
        f"{len(both_obs):,} observable in BOTH arms")
    c_on = sum(1 for k in both_obs if kon[k]["outcome"] == CHANGED)
    c_off = sum(1 for k in both_obs if koff[k]["outcome"] == CHANGED)
    disc_off = sum(1 for k in both_obs if koff[k]["outcome"] == CHANGED
                   and kon[k]["outcome"] == UNCHANGED)
    disc_on = sum(1 for k in both_obs if kon[k]["outcome"] == CHANGED
                  and koff[k]["outcome"] == UNCHANGED)
    say(f"    on that jointly observable set: ON changed {c_on:,} "
        f"({100*c_on/len(both_obs):.4f}%)   OFF changed {c_off:,} "
        f"({100*c_off/len(both_obs):.4f}%)   "
        f"delta {100*(c_off-c_on)/len(both_obs):+.4f} pp")
    say(f"    discordant pairs: OFF-changed-only {disc_off:,}, "
        f"ON-changed-only {disc_on:,}")

    # ---- 3. paired cluster bootstrap ----
    say(f"\n  == 3. paired cluster bootstrap (95% CI for OFF - ON) ==")
    def boot(units):
        byc = defaultdict(list)
        for cl, a, b, d in units:
            byc[cl].append((a, b, d))
        ks = list(byc)
        rng = random.Random(SEED)
        out = []
        for _ in range(B):
            na = nb = dd = 0
            for _ in range(len(ks)):
                for a, b, d in byc[ks[rng.randrange(len(ks))]]:
                    na += a; nb += b; dd += d
            if dd:
                out.append(100*nb/dd - 100*na/dd)
        out.sort()
        return (round(out[int(.025*len(out))], 4),
                round(out[int(.975*len(out))], 4), len(ks))
    # (a) all observable pages, arm-specific denominators
    un_a = []
    for p in pon:
        if p["outcome"] in (CHANGED, UNCHANGED):
            un_a.append((cid.get(p["request_id"], p["request_id"]),
                         int(p["outcome"] == CHANGED), 0, 1))
    for p in poff:
        if p["outcome"] in (CHANGED, UNCHANGED):
            un_a.append((cid.get(p["request_id"], p["request_id"]),
                         0, int(p["outcome"] == CHANGED), 0))
    # denominators differ slightly between arms, so use the jointly observable
    # paired set for the identified comparison
    un_b = [(cid.get(k[0], k[0]),
             int(kon[k]["outcome"] == CHANGED),
             int(koff[k]["outcome"] == CHANGED), 1) for k in both_obs]
    lo, hi, nk = boot(un_b)
    d_paired = 100*(c_off-c_on)/len(both_obs)
    say(f"    PAIRED, jointly observable (request, URL) units")
    say(f"      n = {len(both_obs):,} units in {nk:,} clusters, "
        f"{B:,} resamples, seed {SEED}")
    say(f"      delta = {d_paired:+.4f} pp   95% CI [{lo:+.4f}, {hi:+.4f}] pp")
    say(f"      excludes zero: {lo > 0 or hi < 0}")
    say(f"    UNPAIRED marginal rates (arm-specific denominators, the figure "
        f"originally reported)")
    say(f"      delta = {naive:+.4f} pp   "
        f"ON {A['rate_pct']:.4f}% CI{A['wilson']}   "
        f"OFF {Bs['rate_pct']:.4f}% CI{Bs['wilson']}")

    # ---- 4. artifact-age distributions ----
    say(f"\n  == 4. artifact-age distribution on the matched URLs ==")
    def ages(pgs):
        v = [float(p["artifact_age_seconds"]) / 3600.0 for p in pgs]
        return v
    aon, aoff = ages(pon), ages(poff)
    say(f"    {'arm':<10}{'n':>8}{'mean h':>10}{'p25':>9}{'median':>10}"
        f"{'p75':>9}{'p95':>10}{'max':>10}")
    for nm, v in (("gate ON", aon), ("gate OFF", aoff)):
        say(f"    {nm:<10}{len(v):>8,}{statistics.mean(v):>10.3f}"
            f"{pct(v,.25):>9.3f}{statistics.median(v):>10.3f}"
            f"{pct(v,.75):>9.3f}{pct(v,.95):>10.3f}{max(v):>10.3f}")
    da = [(float(koff[k]["artifact_age_seconds"])
           - float(kon[k]["artifact_age_seconds"])) / 3600.0 for k in keys]
    older_off = sum(1 for x in da if x > 0)
    older_on = sum(1 for x in da if x < 0)
    same_age = sum(1 for x in da if x == 0)
    say(f"    paired age difference (OFF - ON) on the {len(keys):,} shared units:")
    say(f"      OFF older {older_off:,}   ON older {older_on:,}   "
        f"identical age {same_age:,}")
    say(f"      mean {statistics.mean(da):+.3f} h   median "
        f"{statistics.median(da):+.3f} h   p95 {pct(da,.95):+.3f} h")
    nver = sum(1 for k in keys
               if kon[k]["artifact_version"] != koff[k]["artifact_version"])
    say(f"      differing artifact VERSION: {nver:,} of {len(keys):,} units "
        f"({100*nver/len(keys):.2f}%)")
    say(f"    -> the level-3 difference can only arise from these "
        f"{nver:,} units; the rest are identical in both arms.")
    if nver:
        vk = [k for k in keys
              if kon[k]["artifact_version"] != koff[k]["artifact_version"]
              and kon[k]["outcome"] in (CHANGED, UNCHANGED)
              and koff[k]["outcome"] in (CHANGED, UNCHANGED)]
        vco = sum(1 for k in vk if kon[k]["outcome"] == CHANGED)
        vcf = sum(1 for k in vk if koff[k]["outcome"] == CHANGED)
        say(f"      among version-differing AND jointly observable "
            f"({len(vk):,}): ON changed {vco:,} "
            f"({100*vco/max(1,len(vk)):.2f}%), OFF changed {vcf:,} "
            f"({100*vcf/max(1,len(vk)):.2f}%)")

    # ---- 6. coverage and unobservables ----
    say(f"\n  == 6. coverage and unobservable outcomes in the matched set ==")
    for nm, e in (("gate ON", A), ("gate OFF", Bs)):
        say(f"    {nm:<10} observable {e['observable']:,}/{e['pages']:,} = "
            f"{100*e['observable']/e['pages']:.2f}%   unobservable "
            f"{e['unobservable']:,} ({100*e['unobservable']/e['pages']:.2f}%)")
    say(f"    jointly observable (both arms) {len(both_obs):,}/{len(keys):,} "
        f"= {100*len(both_obs)/len(keys):.2f}% of shared units")
    say(f"    unobservable is NEVER counted as unchanged.")

    out = {"matched_requests": len(matched),
           "page_rows": {"on": len(pon), "off": len(poff)},
           "unique_urls": len({p["url_hash"] for p in pon}),
           "clusters": len({cid.get(r, r) for r in matched}),
           "marginal": {"on": A, "off": Bs, "delta_pp": round(naive, 4)},
           "paired": {"shared_units": len(keys),
                      "jointly_observable": len(both_obs),
                      "changed_on": c_on, "changed_off": c_off,
                      "discordant_off_only": disc_off,
                      "discordant_on_only": disc_on,
                      "delta_pp": round(d_paired, 4),
                      "cluster_bootstrap_ci95": [lo, hi],
                      "clusters": nk, "resamples": B, "seed": SEED,
                      "excludes_zero": bool(lo > 0 or hi < 0)},
           "artifact_age_hours": {
               "on": {"mean": round(statistics.mean(aon), 4),
                      "median": round(statistics.median(aon), 4),
                      "p95": round(pct(aon, .95), 4)},
               "off": {"mean": round(statistics.mean(aoff), 4),
                       "median": round(statistics.median(aoff), 4),
                       "p95": round(pct(aoff, .95), 4)},
               "paired_diff_mean_h": round(statistics.mean(da), 4),
               "off_older": older_off, "on_older": older_on,
               "identical_age": same_age,
               "differing_version_units": nver},
           "coverage": {"on_observable_pct": round(100*A['observable']/A['pages'], 4),
                        "off_observable_pct": round(100*Bs['observable']/Bs['pages'], 4),
                        "jointly_observable_pct": round(100*len(both_obs)/len(keys), 4)}}
    json.dump(out, open(HERE / "matched_url_verification.json", "w"), indent=2)
    (HERE / "verify_matched_url.log").write_text("\n".join(log) + "\n",
                                                 encoding="utf-8")
    say(f"\n  wrote matched_url_verification.json")


if __name__ == "__main__":
    main()
