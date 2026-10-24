"""Table 2 ExactTTL and vCache rows under one engine (routing is engine-independent; drift labels use mixed_age_replay.outcome and, via the copied
scorers, the engine's request aggregation). Keyset must reproduce the recorded rows."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, pathlib
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); me = E["me"]
sys.path.insert(1, str(REPO_ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/exact_ttl_and_l1_only")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache"))
import exact_ttl_l1_only_policies as L1POL, vcache_reference_replay as RF
assert "scripts/baselines/exact_ttl_and_l1_only/exact_ttl_l1_only_policies.py" in L1POL.__file__ and "scripts/baselines/vcache/vcache_reference_replay.py" in RF.__file__ and L1POL.me is me and RF.me is me
S = C.streams(E)["primary"]; SUP = E["support_of"](S, E["rounds"]); CH, UN = me.CHANGED, me.UNCHANGED
def score(m, per):
    ins = [o for q, o in per if o is not None and q in SUP]; ch = sum(o == CH for o in ins); un = sum(o == UN for o in ins); n = m["n_requests"]
    return {"search_saved_pct": round(m["search_saved_pct"], 4), "drift_pct": round(100 * ch / (ch + un), 4) if ch + un else None, "coverage_pct": round(100 * (ch + un) / len(SUP), 4),
            "determinable": ch + un, "changed": ch, "fetch_per_1k": round(1000 * m["fetches"] / n, 1), "gen_per_1k": round(1000 * m["generations"] / n, 1)}
out = {}
em, eper, _ = L1POL.replay_l1cache(S, E["rounds"], "exact", "fixed"); out["ExactTTL"] = score(em, eper)
m, o, _ = RF.reference_replay(S, rounds=E["rounds"], delta=0.02); sc = RF.score(o, SUP)
out["vCache_delta0.02"] = {"search_saved_pct": round(m["search_saved_pct"], 4), "drift_pct": None if sc["drift_pct"] is None else round(sc["drift_pct"], 4), "coverage_pct": round(sc["coverage_pct"], 4),
                           "determinable": sc["determinable"], "changed": sc["changed"], "fetch_per_1k": round(1000 * m["fetches"] / len(S), 1), "gen_per_1k": round(1000 * m["generations"] / len(S), 1)}
json.dump(out, open(C.W/"results"/f"baseline_measurement_{ENG}.json", "w"), indent=1); print(ENG, out, flush=True)
