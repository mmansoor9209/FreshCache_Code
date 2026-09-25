#!/usr/bin/env python3
"""
Section 12 -- faithful prior-method baselines.

WHAT WAS LOOKED FOR
  vCache   The authors released code (github.com/vcache-project/vCache).
           A port of its VerifiedDecisionPolicy against that release already
           exists in the repository as vcache_implementation.py
           (sim_vcache_reference), with a written fidelity verification in
           validation/vcache_baseline_fidelity/. This section re-runs THAT port
           on the held-out TEST clusters and reports it next to the engine's
           simpler `vCache` rule that produced the manuscript numbers.
  SCALM    No official public implementation is present in this repository and
           none is vendored. The `SCALM` policy in engine_all.py is a scoped
           reimplementation: frequency-weighted, same-class-only nearest
           neighbour with no TTL. It does NOT implement SCALM's hierarchical
           clustering or its eviction policy, because FreshCache-Bench has no
           cache-capacity dimension -- every policy here holds an unbounded
           cache, so an eviction rule has nothing to act on and a clustering
           index changes only lookup cost, not the served entry. That is
           stated, not fabricated: the scoped name "SCALM-style" is retained.

This runs at the FIXED 24-hour horizon, the family in which the released
vCache port is defined (its correctness feedback is a single label map keyed by
URL, which has no meaning when entries have heterogeneous ages). The mixed-age
engine_all `vCache` result already exists in validation/vcache_mixed_age/ and
is cited rather than recomputed.

Held-out TEST clusters only. Read-only. CPU.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, pathlib, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    import setproctitle; setproctitle.setproctitle("anon-freshcache-finalvalidation")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
os.chdir(ROOT)

import numpy as np                       # noqa: E402
import experiment as exp                 # noqa: E402
import engine_all as ea                  # noqa: E402
import corrected_engine as ce            # noqa: E402
import vcache_implementation as vci      # noqa: E402

CHANGED, UNCHANGED, UNOBS = ce.CHANGED, ce.UNCHANGED, ce.UNOBS
POLICIES = ["NoCache", "SemanticTTL", "SCALM", "vCache", "L2Only",
            "FreshCache_Full"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * max(0.0, (c - m) / d), 3),
            round(100 * min(1.0, (c + m) / d), 3))


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("V3 SECTION 12 -- faithful prior-method baselines")
    say(f"  utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    # ---- provenance of the existing vCache port ----
    say("\n  == released-implementation inventory ==")
    fid = ROOT / "validation" / "vcache_baseline_fidelity"
    pv = fid / "port_verification.json"
    say(f"    vcache_implementation.py        present   sha {sha(ROOT/'vcache_implementation.py')[:16]}")
    say(f"      ported against github.com/vcache-project/vCache "
        f"(verified.py::_Algorithm) and arXiv:2502.03771v4 Alg 1-2, Eq 9-11")
    if pv.exists():
        say(f"    port_verification.json          present   "
            f"sha {sha(pv)[:16]}")
        try:
            d = json.load(open(pv, encoding="utf-8"))
            for k in list(d)[:12]:
                say(f"      {k} = {str(d[k])[:110]}")
        except Exception as e:
            say(f"      (unreadable: {e})")
    else:
        say("    port_verification.json          ABSENT")
    say(f"    official SCALM implementation   NOT PRESENT in this repository")
    say(f"      engine_all.CONFIG['SCALM'] = {ea.CONFIG['SCALM']}")
    say(f"      missing vs the published method: hierarchical clustering; "
        f"eviction policy")
    say(f"      reason recorded: FreshCache-Bench has no cache-capacity "
        f"dimension, so eviction has nothing to act on, and a clustering index "
        f"changes lookup cost, not the served entry")
    say(f"      -> the scoped name 'SCALM-style' is RETAINED, not upgraded")

    # ---- data, held-out clusters ----
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    split = json.load(open(ROOT / "validation" / "heldout_baseline_tuning"
                           / "split.json", encoding="utf-8"))
    test_c, val_c = set(split["test_clusters"]), set(split["validation_clusters"])
    test = [r for r in records
            if (r.get("cluster_id") or r["query_id"]) in test_c]
    assert not [r for r in test
                if (r.get("cluster_id") or r["query_id"]) in val_c]
    lab = ce.load_labels("rerun_24h")
    AGE = exp.SIM_AGES["24h"]
    say(f"\n  held-out TEST records {len(test):,} "
        f"({len(test_c):,} clusters); fixed horizon {AGE/3600:.0f} h; "
        f"validation leakage 0")

    rows = {}
    for pol in POLICIES:
        m, _ = ea.replay(test, lab, AGE, pol, rich=rich)
        rows[pol] = m

    # ---- the faithful released-port vCache ----
    stale = {u for u, v in lab.items() if v == CHANGED}
    say(f"\n  faithful vCache port: sim_vcache_reference, delta = "
        f"{exp.EPS_ANSWER} (the published default)")
    try:
        vres = vci.sim_vcache_reference(test, stale, AGE, exp.EPS_ANSWER)
        rows["vCache_released_port"] = vres
    except TypeError:
        vres = vci.sim_vcache_reference(test, stale, AGE)
        rows["vCache_released_port"] = vres
    except Exception as e:
        say(f"    PORT RUN FAILED: {e}")
        vres = None

    def get(m, *names, default=None):
        for n in names:
            if isinstance(m, dict) and n in m:
                return m[n]
        return default

    say(f"\n  {'policy':<24}{'saved%':>10}{'L1 hits':>10}{'stale%':>10}"
        f"{'searches':>11}{'fetches':>10}")
    out = {}
    for pol, m in rows.items():
        saved = get(m, "search_saved_pct", "saved_pct", "search_saved")
        l1 = get(m, "l1_hits", "hits", default=0)
        stl = get(m, "stale_pct", "drift_pct", "stale_rate")
        sc_ = get(m, "search_calls", "searches", default=0)
        fe = get(m, "fetches", default=0)
        out[pol] = {"search_saved_pct": saved, "l1_hits": l1,
                    "stale_pct": stl, "searches": sc_, "fetches": fe,
                    "raw": {k: v for k, v in m.items()
                            if isinstance(v, (int, float, str, type(None)))}}
        f = lambda x, w=9: (f"{x:>{w}.4f}" if isinstance(x, (int, float))
                            else f"{str(x):>{w}}")
        say(f"  {pol:<24}{f(saved)}%{l1:>10,}{f(stl)}%{sc_:>11,}{fe:>10,}")

    say("\n  the manuscript's mixed-age vCache number comes from "
        "validation/vcache_mixed_age/ and is NOT recomputed here; the released "
        "port is fixed-horizon by construction (its feedback is a single "
        "URL-keyed label map), so the two are reported in their own families.")
    say("  no signal absent from FreshCache-Bench was fabricated for either "
        "baseline.")

    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "split": "held-out TEST clusters", "horizon_seconds": AGE,
               "vcache_port_source": "vcache_implementation.py "
                                     "(ported against the released vCache)",
               "scalm_official_implementation_present": False,
               "scalm_missing_components": ["hierarchical clustering",
                                            "eviction policy"],
               "scalm_naming": "SCALM-style (scoped) -- retained",
               "results": out}, open(HERE / "faithful_results.json", "w"),
              indent=2, default=str)
    with open(HERE / "faithful_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "search_saved_pct", "l1_hits", "stale_pct",
                    "searches", "fetches"])
        for pol, r in out.items():
            w.writerow([pol, r["search_saved_pct"], r["l1_hits"],
                        r["stale_pct"], r["searches"], r["fetches"]])
    (HERE / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    say("  wrote faithful_results.json/.csv, run.log")


if __name__ == "__main__":
    main()
