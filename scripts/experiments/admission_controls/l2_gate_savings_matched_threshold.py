"""Learned L2 compatibility gate, stage 4 (VALIDATION ONLY): second pre-registered operating point of the same frozen gate model (gate_frozen.pkl, unchanged).
Rule (written to prereg2.json before any replay): tau2 = the threshold whose validation search savings are closest to FreshCache theta_L2 = 0.80's
validation savings (outputs/standalone_operating_points/selection_frozen.json); found by bisection on the validation replay (savings are
non-increasing in tau); ties -> the stricter (higher) tau. Also writes a validation-only frontier (savings at a fixed tau grid). Held-out is not read.
Page texts and embedding look-ups are memoised for speed only (no decision changes)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import datetime, functools, hashlib, json, os, pathlib, pickle, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
sys.dont_write_bytecode = True
O = ((OUTPUT_ROOT / "learned_l2_gate")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/admission_controls"))
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
SELJ = json.load(open(((OUTPUT_ROOT / "standalone_operating_points/selection_frozen.json")))); TARGET = SELJ["cells"]["FreshCache_thetaL2_0.80"]["search_saved_pct"]
GZ = pickle.load(open(O / "gate_frozen.pkl", "rb")); GJ = json.load(open(O / "gate_frozen.json"))
PRE2 = {"written_utc": now(), "first_prereg_sha256": json.load(open(O / "prereg.json"))["sha256"], "gate_model_sha256": GJ["model_sha256"], "first_tau_youden": GJ["tau_youden"],
        "rule": "tau2 = argmin_tau |validation search savings(tau) - FreshCache theta_L2 0.80 validation savings|; bisection on the validation replay; ties -> higher tau",
        "status": "second pre-registered operating point of the same frozen model; written after the Youden point was scored on held-out and BEFORE any replay for this point; chosen from validation only; both points will be reported, with the two held-out looks disclosed",
        "no_further_points": True}
PRE2["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in PRE2.items() if k != "written_utc"}, sort_keys=True).encode()).hexdigest()
if not (O / "prereg2.json").exists(): json.dump(PRE2, open(O / "prereg2.json", "w"), indent=1)
PRE2 = json.load(open(O / "prereg2.json")); print(f"[{now()}] prereg2 {PRE2['sha256'][:16]} target {TARGET}", flush=True)
import l2_gate_features as Lb   # noqa: E402
import numpy as np   # noqa: E402
Lb.page_text = functools.lru_cache(maxsize=None)(Lb.page_text)   # memoise (pure function of (uh, tv))
FEATS, MODEL = GZ["features"], GZ["model"]
V = Lb.C.streams(Lb.E)["validation"]; assert DS.check("validation_requests", len(V))
_fc = {}
class Gate:
    def __init__(self, tau): self.tau = tau
    def score(self, q, tent_ev, eq, sim, fc, age, efc, qid, urls):
        k = (q, tuple(tuple(x) for x in tent_ev), eq, round(age, 3))
        if k not in _fc:
            f = Lb.features(q, [tuple(x) for x in tent_ev], eq, sim, fc, age, efc); _fc[k] = float(MODEL.predict_proba(np.array([[f[x] for x in FEATS]], float))[:, 1][0])
        return _fc[k]
LOG = {}
def sav(tau):
    if tau not in LOG:
        t0 = time.time(); m, _, _ = Lb.S2.replay(V, gamma=tau, support=Gate(tau)); LOG[tau] = {"search_saved_pct": m["search_saved_pct"], "l2_hits": m["l2_hits"], "l2_rejected": m["l2_rejected"], "seconds": round(time.time() - t0, 1)}
        print(f"  tau {tau:.6f}: validation saved {m['search_saved_pct']} (L2 {m['l2_hits']}, rejected {m['l2_rejected']}, {LOG[tau]['seconds']}s)", flush=True)
    return LOG[tau]["search_saved_pct"]
lo, hi = 0.0, GJ["tau_youden"]   # sav(0) >= target >= sav(youden) expected
assert sav(lo) >= TARGET >= sav(hi), (sav(lo), sav(hi))
for _ in range(14):
    mid = round((lo + hi) / 2, 6)
    if sav(mid) >= TARGET: lo = mid
    else: hi = mid
cands = sorted(LOG, key=lambda t: (abs(LOG[t]["search_saved_pct"] - TARGET), -t)); tau2 = cands[0]
for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6): sav(t)   # validation-only frontier points for the paper
OUT = {"frozen_utc": now(), "prereg2_sha256": PRE2["sha256"], "target_validation_savings": TARGET, "tau2": tau2, "validation_savings_at_tau2": LOG[tau2]["search_saved_pct"], "replays": {f"{t:.6f}": v for t, v in sorted(LOG.items())}, "heldout_consulted": False}
json.dump(OUT, open(O / "tau2_frozen.json", "w"), indent=1); print(json.dumps({k: OUT[k] for k in ("tau2", "validation_savings_at_tau2", "target_validation_savings")}), flush=True)
