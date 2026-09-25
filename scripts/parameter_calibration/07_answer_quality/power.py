#!/usr/bin/env python3
"""How much change could this 400-request audit actually have detected?

A null result on 400 requests is only as strong as its power. This reports
(a) how many requests genuinely changed their served answer between
configurations, and (b) the smallest WAI increase the audit could reject.
"""
from __future__ import annotations
import json, math, pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
PC = HERE.parent
CFG = ["A_original", "B_mult_only", "C_budget_only", "D_full"]
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


S = json.load(open(HERE / "audit_sample.json", encoding="utf-8"))["requests"]
aq = json.load(open(HERE / "answer_quality.json", encoding="utf-8"))

say("POWER AND EXPOSURE of the 400-request audit")
say(f"\n  == how many requests actually changed? ==")
say(f"  {'contrast':<24}{'served key differs':>20}{'tier differs':>14}"
    f"{'answer text differs':>21}")
ans = {}
for f in ("cached_answers.jsonl", "answers_llama3b.jsonl"):
    p = HERE / f
    if p.exists():
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            ans[tuple(d["key"])] = d["answer"]
exposure = {}
for c in CFG[1:]:
    kd = sum(1 for d in S if d[f"{c}_key"] != d["A_original_key"])
    td = sum(1 for d in S if d[f"{c}_tier"] != d["A_original_tier"])
    ad = sum(1 for d in S
             if ans.get(tuple(d[f"{c}_key"])) != ans.get(tuple(d["A_original_key"])))
    exposure[c] = {"served_key_differs": kd, "tier_differs": td,
                   "answer_text_differs": ad}
    say(f"  {'A vs '+c:<24}{kd:>20}{td:>14}{ad:>21}")
say(f"  Only the requests whose served ANSWER TEXT differs can move any")
say(f"  quality metric at all; the rest are identical by construction.")

say(f"\n  == smallest detectable WAI difference ==")
n, k = 400, aq["per_config"]["A_original"]["wai"]
p0 = k / n
say(f"    baseline WAI {k}/{n} = {100*p0:.2f}%, Wilson 95% CI "
    f"{aq['per_config']['A_original']['wai_ci95']}")


def wilson_lo(k, n, z=1.96):
    if not n:
        return 0.0
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * max(0.0, (c - m) / d)


hi = aq["per_config"]["A_original"]["wai_ci95"][1]
need = None
for kk in range(k, n + 1):
    if wilson_lo(kk, n) > hi:
        need = kk
        break
say(f"    a configuration's WAI CI would first clear A's upper bound "
    f"({hi:.2f}%) at {need}/{n} = {100*need/n:.2f}%")
say(f"    i.e. this audit can only rule out WAI changes larger than roughly "
    f"{100*need/n - 100*p0:+.2f} pp ({need/k:.1f}x the baseline).")
say(f"    Smaller true differences are NOT excluded. 'No detectable "
    f"difference' here does NOT establish 'no difference'.")

say(f"\n  == exact paired bound instead of the marginal one ==")
for c in CFG[1:]:
    pr = aq["paired_vs_A"][f"A_vs_{c}"]
    say(f"    A vs {c:<16} WAI discordant pairs "
        f"{pr['wai_only_A']} vs {pr['wai_only_X']} "
        f"(shared {pr['wai_both']}), exact McNemar p = {pr['wai_mcnemar_p']:.4g}")
say(f"    With so few discordant pairs the paired test is likewise "
    f"low-powered; it constrains the difference to be small, not zero.")

json.dump({"exposure": exposure, "baseline_wai": k, "n": n,
           "min_detectable_wai_k": need,
           "min_detectable_wai_pct": round(100 * need / n, 4),
           "interpretation": "audit excludes only WAI changes larger than "
                             f"~{100*need/n - 100*p0:.2f} pp; smaller "
                             "differences remain possible"},
          open(HERE / "power.json", "w"), indent=2)
(PC / "logs" / "aq_power.log").write_text("\n".join(log) + "\n", encoding="utf-8")
say(f"\n  wrote power.json")
