"""Gates the hybrid CategoryAware+FreshCache engine against frozen replays before any measurement with it.
The held-out stream is used only for exact equality with the frozen FreshCache / verifier / CategoryAware replays; no new held-out outcome
is computed here. Writes work/engine_gates.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, os, pickle, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import categoryaware_composition_helpers as CAH, categoryaware_freshcache_engine as HE
import pathlib
O = ((OUTPUT_ROOT / "categoryaware_freshcache")); LOG = open(O / "run.log", "a")
def log(*a): s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
E = CAH.engine(); p2 = E["p2"]; ma = sys.modules["page_version_replay"]; ST = CAH.C.streams(E); H, V = ST["heldout"], ST["validation"]
SUP = HE.Support(CAH.OV, ma); G = {}
TH, KA = CAH.ca_params(0.0, 1.0)
for nm, S_ in (("heldout", H), ("validation", V)):
    _, ref = p2.replay(S_, {}, "FreshCache"); m, rows = HE.replay(E, S_, l1="freshcache", floor=0.75)
    mp = {"REAL_TIME": "REAL_TIME"}
    G[f"{nm}: l1=freshcache reproduces audit_replay.replay FreshCache (tier + served evidence, every request)"] = all(rows[q]["tier"] == ref[q]["tier"] and [tuple(x) for x in rows[q]["ev"]] == [tuple(x) for x in (ref[q].get("stored_answer_ev") if ref[q]["tier"] == "L1" else ref[q]["ev"])] for q in ref)
    mc, rc = CAH.ca_replay(S_, TH, KA); mh, rh = HE.replay(E, S_, l1="ca", theta_by_class=TH, kappa_by_class=KA, l2l3=False)
    G[f"{nm}: l1=ca, L2/L3 off reproduces baseline-selection ca_replay (tier, evidence, searches, fetches)"] = all(rh[q]["tier"] == rc[q]["tier"] and [tuple(x) for x in rh[q]["ev"]] == [tuple(x) for x in rc[q]["ev"]] for q in rc) and (mh["search_calls"], mh["fetches"]) == (mc["search_calls"], mc["fetches"])
VR = pickle.load(open(CAH.CRR / "verifier/out/test_rows.pkl", "rb"))["L2Verify_g0.55"]; VO = json.load(open(CAH.CRR / "verifier/out/test_operational.json"))["results"]["L2Verify_g0.55"]
mv, rv = HE.replay(E, H, l1="freshcache", floor=0.75, gamma=0.55, support=SUP)
G["heldout: l1=freshcache + gamma 0.55 reproduces frozen verifier replay (tier, evidence, verdict; counts)"] = all(rv[q]["tier"] == VR[q]["tier"] and [tuple(x) for x in rv[q]["ev"]] == [tuple(x) for x in VR[q]["ev"]] and rv[q].get("verified") == VR[q]["verified"] for q in VR) and (mv["search_calls"], mv["fetches"], mv["l2_hits"]) == (VO["search_calls"], VO["fetches"], VO["l2_hits"])
o = E["exp"]._L2_EQ_SIM_FLOOR; E["exp"]._L2_EQ_SIM_FLOOR = 0.80
try: _, ref80 = p2.replay(H, {}, "FreshCache")
finally: E["exp"]._L2_EQ_SIM_FLOOR = o
m8, r8 = HE.replay(E, H, l1="freshcache", floor=0.80)
G["heldout: l1=freshcache floor 0.80 reproduces audit_replay.replay at theta_L2 0.80 (tiers)"] = all(r8[q]["tier"] == ref80[q]["tier"] for q in ref80)
for k, v in G.items(): log("ENGINE GATE", k, v)
json.dump(G, open(O / "work/engine_gates.json", "w"), indent=1)
if not all(G.values()): log("STOP: engine gate failed"); sys.exit(1)
