"""Builds the 24-test Holm family of Table 6 (12 WAI + 12 sufficiency exact McNemar tests against primary FreshCache).
The 23 tests collected by generation_cost_break_even.py (outputs/operating_point_summaries/operating_point_reaggregation.json, "holm") are
extended with NoCache sufficiency (nocache_sufficiency_scoring.py, vs_theta_0.75) and Holm-adjusted together; the result is written to
outputs/operating_point_summaries/holm_family_24.json, which holm_adjustment.py checks.
Also checks that the gate, verifier and composition p-values that generation_cost_break_even.py reads equal the outputs that
produce them: l2_gate_heldout_evaluation.py (gate_heldout_results.json, gate_heldout_results_tau2.json), verifier_gamma_paired_tests.py and
categoryaware_composition_paired_tests.py (4-decimal values as reported)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, os, pathlib, sys; OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); S = OUTPUT_ROOT / "operating_point_summaries"
d = json.load(open(S / "operating_point_reaggregation.json"))["holm"]
T = [(h["test"], h["p"]) for h in d]
T.append(("NoCache: Suff", json.load(open(S / "nocache_sufficiency.json"))["vs_theta_0.75"]["p"]))
m = len(T); order = sorted(range(m), key=lambda i: T[i][1]); adj = [None] * m; run = 0
for r, i in enumerate(order): run = max(run, min(1, (m - r) * T[i][1])); adj[i] = run
out = [{"test": t, "p": p, "p_holm": round(a, 6), "sig": a < 0.05} for (t, p), a in zip(T, adj)]
json.dump({"family": "all 24 paired tests of Table 4", "tests": out}, open(S / "holm_family_24.json", "w"), indent=1)
for o in out:
    if "CM (.60" in o["test"] or "SM" in o["test"] or "NoCache" in o["test"]: print(o)
print(m)
# provenance of the p-values read by generation_cost_break_even.py
P = dict(T); G1 = json.load(open(OUTPUT_ROOT / "learned_l2_gate/gate_heldout_results.json"))["paired_gate_vs"]["theta_0.75"]
G2 = json.load(open(OUTPUT_ROOT / "learned_l2_gate/gate_heldout_results_tau2.json"))["paired_gate_vs"]["theta_0.75"]
VG = json.load(open(S / "verifier_gamma_paired_tests.json"))["gamma_0.55"]
CF = json.load(open(S / "categoryaware_composition_paired_tests.json"))["paired_vs_H80"]["FreshCache primary (theta_L2=0.75)"]
CHK = {"gate tau: WAI": (P["gate tau: WAI"], G1["WAI_p"]), "gate tau: Suff": (P["gate tau: Suff"], G1["suff_p"]),
       "gate tau2: WAI": (P["gate tau2: WAI"], G2["WAI_p"]), "gate tau2: Suff": (P["gate tau2: Suff"], G2["suff_p"]),
       "verifier gamma: WAI": (P["verifier gamma: WAI"], VG["WAI_p"]), "verifier gamma: Suff": (P["verifier gamma: Suff"], round(VG["suff_p"], 4)),
       "CA+FC vs FC: WAI": (P["CA+FC vs FC: WAI"], CF["WAI_p"]), "CA+FC vs FC: Suff": (P["CA+FC vs FC: Suff"], round(CF["suff_p"], 4))}
ok = {k: abs(a - b) < 1e-12 for k, (a, b) in CHK.items()}
for k, (a, b) in CHK.items(): print("constant", k, a, "producer", b, "OK" if ok[k] else "MISMATCH")



if not all(ok.values()): sys.exit("constant p-values differ from their producing outputs")
