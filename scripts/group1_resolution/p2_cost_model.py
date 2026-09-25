#!/usr/bin/env python3
"""N15 -- per-1,000-request cost model.

MEASURED inputs (from artifacts in this repository):
  search calls, fetches, generations per 1,000 requests; cache footprint;
  lookup latency.
ASSUMED inputs (list prices, cited, NOT measured here):
  search API call, LLM input/output tokens, egress, storage.

The two are never mixed in a single number without being labelled. A
sensitivity band is reported because the assumed prices dominate the result.
No API call is made by this script.
"""
from __future__ import annotations
import json, pathlib
from collections import OrderedDict

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
log = []


def say(s=""):
    print(s, flush=True); log.append(s)


# --------------------------------------------------------------- ASSUMPTIONS
PRICES = OrderedDict([
    ("search_call_usd", (0.0010, 0.0003, 0.0050,
                         "Serper $0.30/1k (low) .. Bing/Google CSE $5/1k (high); "
                         "$1.00/1k central")),
    ("llm_input_usd_per_1k_tok", (0.00015, 0.00005, 0.00060,
                                  "small hosted model, e.g. GPT-4o-mini class")),
    ("llm_output_usd_per_1k_tok", (0.00060, 0.00020, 0.00240, "same family")),
    ("fetch_egress_usd", (0.000002, 0.0000005, 0.00001,
                          "~2 KB extracted text at ~$0.09/GB egress")),
    ("storage_usd_per_gb_month", (0.023, 0.010, 0.100, "object/blob storage")),
])
# MEASURED token sizes (v8/deployment_cost_results.json, bytes -> ~4 B/token)
CTX_TOKENS_IN = 2800 / 4.0      # ctx_for MAXCHARS = 2800
ANS_TOKENS_OUT = 51.9 / 4.0     # mean generated answer 51.9 B


def cost_per_1k(sc, fe, ge, p):
    """sc/fe/ge are per-1,000-request counts (MEASURED)."""
    return (sc * p["search_call_usd"]
            + fe * p["fetch_egress_usd"]
            + ge * (CTX_TOKENS_IN / 1000 * p["llm_input_usd_per_1k_tok"]
                    + ANS_TOKENS_OUT / 1000 * p["llm_output_usd_per_1k_tok"]))


def main():
    say("N15 -- per-1,000-request cost model")
    say("  MEASURED: operation counts, footprint, latency (from artifacts).")
    say("  ASSUMED : unit prices (list prices, cited; NOT measured).")

    say(f"\n  == assumed unit prices ==")
    say(f"  {'parameter':<30}{'central':>12}{'low':>12}{'high':>12}  basis")
    for k, (c, lo, hi, note) in PRICES.items():
        say(f"  {k:<30}{c:>12.6f}{lo:>12.6f}{hi:>12.6f}  {note}")
    central = {k: v[0] for k, v in PRICES.items()}
    low = {k: v[1] for k, v in PRICES.items()}
    high = {k: v[2] for k, v in PRICES.items()}

    # ---- MEASURED operation counts, mixed-age primary workload ----
    bl = json.load(open(V3 / "remaining_critical_issues" / "02_l1only"
                        / "baseline_results.json", encoding="utf-8"))
    fb = json.load(open(V3 / "baseline_fidelity_resolution" / "02_faithful_baselines"
                        / "faithful_baselines_mixed.json", encoding="utf-8"))
    rows = OrderedDict()
    for name, r in bl["results"]["full_mixed_age"]["rows"].items():
        rows[name] = (r["search_per_1k"], r["fetch_per_1k"], r["gen_per_1k"])
    for name, r in fb["populations"]["full_mixed_age"].items():
        if name.startswith(("SCALM-HSC", "vCache-Reference")):
            rows[name] = (round(1000*r["searches"]/r["n_requests"], 2),
                          r["fetch_per_1k"], r["gen_per_1k"])

    say(f"\n  == MEASURED operation counts per 1,000 requests "
        f"(mixed-age, n=31,201) ==")
    say(f"  {'policy':<34}{'search':>9}{'fetch':>10}{'gen':>9}"
        f"{'USD/1k':>10}{'low':>9}{'high':>9}{'vs NoCache':>12}")
    base = None
    out = {}
    for name, (sc, fe, ge) in rows.items():
        c = cost_per_1k(sc, fe, ge, central)
        if name == "NoCache":
            base = c
    for name, (sc, fe, ge) in rows.items():
        c = cost_per_1k(sc, fe, ge, central)
        cl = cost_per_1k(sc, fe, ge, low)
        ch = cost_per_1k(sc, fe, ge, high)
        red = (100 * (1 - c / base)) if base else 0.0
        out[name] = {"measured_search_per_1k": sc, "measured_fetch_per_1k": fe,
                     "measured_gen_per_1k": ge,
                     "usd_per_1k_central": round(c, 4),
                     "usd_per_1k_low": round(cl, 4),
                     "usd_per_1k_high": round(ch, 4),
                     "pct_cheaper_than_nocache": round(red, 2)}
        say(f"  {name:<34}{sc:>9.2f}{fe:>10.2f}{ge:>9.2f}"
            f"{c:>10.4f}{cl:>9.4f}{ch:>9.4f}{red:>11.2f}%")

    # ---- what drives the cost ----
    sc, fe, ge = rows["FreshCache"]
    parts = {"search": sc * central["search_call_usd"],
             "fetch egress": fe * central["fetch_egress_usd"],
             "LLM input": ge * CTX_TOKENS_IN / 1000 * central["llm_input_usd_per_1k_tok"],
             "LLM output": ge * ANS_TOKENS_OUT / 1000 * central["llm_output_usd_per_1k_tok"]}
    tot = sum(parts.values())
    say(f"\n  == cost decomposition for FreshCache (central prices) ==")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        say(f"    {k:<16}{v:>10.4f} USD/1k   {100*v/tot:>6.2f}%")
    say(f"    NOTE the ranking of policies is driven almost entirely by the")
    say(f"    search-call price, which is ASSUMED. At the low search price the")
    say(f"    spread between policies narrows sharply; at the high price it widens.")

    # ---- storage, MEASURED ----
    dc = json.load(open(ROOT / "v8" / "deployment_cost_results.json",
                        encoding="utf-8"))
    be = dc["entry_sizes"]["bytes_per_entry"]
    say(f"\n  == MEASURED storage (v8/deployment_cost_results.json) ==")
    say(f"    bytes/entry  L1 {be['L1']}  L2 {be['L2']}  L3 {be['L3']}")
    gb = 141.6 / 1024
    say(f"    full-workload footprint at fp16: 141.6 MB = {gb:.4f} GB")
    say(f"    storage cost: {gb*central['storage_usd_per_gb_month']:.4f} USD/month "
        f"(central), range {gb*low['storage_usd_per_gb_month']:.4f}"
        f"-{gb*high['storage_usd_per_gb_month']:.4f}")
    say(f"    -> storage is negligible beside per-request search cost; the "
        f"index, not the cached content, is 83% of the footprint.")

    # ---- latency, MEASURED ----
    lc = json.load(open(ROOT / "data" / "lookup_cost_results.json",
                        encoding="utf-8"))
    say(f"\n  == MEASURED lookup cost (data/lookup_cost_results.json) ==")
    say(f"    encode p50 {lc['encode']['p50_ms']} ms (batched "
        f"{lc['encode']['batched_ms_per_query']} ms/query); "
        f"ANN@30k p50 {lc['ann']['30000']['p50_ms']} ms; "
        f"entity p50 {lc['entity']['p50_ms']} ms")
    say(f"    This is compute the cache ADDS; it is not priced above because it "
        f"is local GPU/CPU time, not a metered service.")

    json.dump({"measured_inputs": {"source_operation_counts":
               "remaining_critical_issues/02_l1only/baseline_results.json + "
               "baseline_fidelity_resolution/02_faithful_baselines",
               "ctx_tokens_in": CTX_TOKENS_IN,
               "answer_tokens_out": ANS_TOKENS_OUT,
               "footprint_mb_fp16": 141.6,
               "bytes_per_entry": be, "lookup_ms": lc},
               "assumed_prices": {k: {"central": v[0], "low": v[1],
                                      "high": v[2], "basis": v[3]}
                                  for k, v in PRICES.items()},
               "per_policy": out},
              open(HERE / "out" / "cost_model.json", "w"), indent=2)
    (HERE / "out" / "p2_cost.log").write_text("\n".join(log) + "\n",
                                              encoding="utf-8")
    say(f"\n  wrote out/cost_model.json")


if __name__ == "__main__":
    main()
