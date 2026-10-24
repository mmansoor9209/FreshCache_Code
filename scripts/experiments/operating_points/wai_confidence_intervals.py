"""95% intervals for every main-text paired WAI comparison on the held-out audit.
Discordant counts (FreshCache-only, other-only) are read from the outputs of the stages that produce Table 4 / Table 2 (frozen two-judge rule).
Difference = (other - FreshCache) WAI events per audited request, in percentage points, with a Wald interval for paired proportions (as unseen_query_scoring.pdiff) and Newcombe's hybrid score interval (method 10) as a check."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import csv, json, math, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "composition_summary"))
J = lambda p: json.load(open(OUTPUT_ROOT / p))
# discordant WAI counts (FreshCache-only, other-only) on the held-out audit, read from the stages that produce them
DT = {r["point"]: r for r in J("standalone_operating_points/heldout_decision_table.json")["decision_table"]}
DT.update({r["point"]: r for r in J("standalone_operating_points/heldout_decision_table_addendum.json")["decision_table"] if r["point"] == "sttl_t0.70_k2"})
dt = lambda pid: tuple(DT[pid]["vs_FreshCache_WAI_FConly_otheronly_p"][:2])
GPT = next(r for r in csv.DictReader(open(OUTPUT_ROOT / "baselines/gptcache/gptcache_answer_quality.csv")) if r["config"] == "GPTCache_native")
G1 = J("learned_l2_gate/gate_heldout_results.json")["paired_gate_vs"]["theta_0.75"]; G2 = J("learned_l2_gate/gate_heldout_results_tau2.json")["paired_gate_vs"]["theta_0.75"]
VG = J("operating_point_summaries/verifier_gamma_paired_tests.json")["gamma_0.55"]
CF = J("operating_point_summaries/categoryaware_composition_paired_tests.json")["paired_vs_H80"]["FreshCache primary (theta_L2=0.75)"]
CMP = {"GPTCache native": (int(GPT["WAI_FreshCache_only"]), int(GPT["WAI_GPTCache_only"])), "SemanticTTL (.80,1/4) quality-matched": dt("sttl_t0.80_k0.25"),
       "SemanticTTL (.80,2) relaxed": dt("sttl_t0.80_k2"), "SemanticTTL (.70,2) savings-matched": dt("sttl_t0.70_k2"), "SemanticTTL (.60,1/4) cost-matched": dt("sttl_t0.60_k0.25"),
       "CategoryAware-style quality-matched": dt("ca_ca_d+0.000_m1.00"), "CategoryAware-style (+.075) cost-matched": dt("ca_ca_d+0.075_m1.00"), "FreshCache theta 0.80": dt("FreshCache_thetaL2_0.80"),
       "Learned gate tau": (G1["WAI_other_only"], G1["WAI_gate_only"]), "Learned gate tau2": (G2["WAI_other_only"], G2["WAI_gate_only"]),
       "Verifier gamma": (VG["WAI_primary_only"], VG["WAI_point_only"]), "CA+FreshCache composition": (CF["WAI_other_only"], CF["WAI_H80_only"]), "NoCache": dt("NoCache")}
N = len(J("general_audit/sample.json")["requests"])
# Additional main-text comparisons with their own denominators: (FC-only or baseline-only, other-only, n)
MA = J("model_adjudication/results/adjudication_robustness.json"); DETP, ND = MA["deterministic"]["paired"], MA["deterministic"]["structured_requests"]
MJ = MA["results"]["majority"]["all_evaluable"]; pair = lambda P, b: (P[f"freshcache vs {b}"]["WAI_first_only"], P[f"freshcache vs {b}"]["WAI_second_only"])
Q14 = J("composition_checks/outputs/stronger_generator_answers.json")["audit"]["all"]["qwen14b"]; MAA = J("mixed_age_audit/recomputed_results.json")
EXTRA = {"Det: SemanticTTL (.60,1/4) cost-matched": (*pair(DETP, "op_sttl_t0.60_k0.25"), ND), "Det: SemanticTTL (.60,1/2)": (*pair(DETP, "semanticttl_quality_0.60_1-2"), ND),
         "Det: CategoryAware-style": (*pair(DETP, "categoryaware_style"), ND),
         "Majority: SemanticTTL (.60,1/4) cost-matched": (*pair(MJ["paired"], "op_sttl_t0.60_k0.25"), MJ["N"]), "Majority: SemanticTTL (.70,2) savings-matched": (*pair(MJ["paired"], "op_sttl_t0.70_k2"), MJ["N"]),
         "Majority: CategoryAware-style": (*pair(MJ["paired"], "categoryaware_style"), MJ["N"]),
         "14B: D vs B (B-only, D-only)": (Q14["B->D"]["WAI_discordant_new_old"][1], Q14["B->D"]["WAI_discordant_new_old"][0], Q14["n"]),   # [D-only, B-only]
         "Age gate: on-only / off-only": (MAA["mcnemar"]["discordant_on_only"], MAA["mcnemar"]["discordant_off_only"], MAA["N"])}   # mixed-age audit, gate on vs off
def wald(b, c, n=N):
    d = (c - b) / n; se = math.sqrt(max((b + c) - (c - b) ** 2 / n, 0)) / n; return [round(100 * d, 2), round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)]
R = {k: {"FC_only": b, "other_only": c, "n": N, "diff_pp_wald": wald(b, c)} for k, (b, c) in CMP.items()}
R.update({k: {"FC_only": b, "other_only": c, "n": n, "diff_pp_wald": wald(b, c, n)} for k, (b, c, n) in EXTRA.items()})
json.dump({"n": N, "note": "difference = other - FreshCache, WAI events per audited request (pp)", "comparisons": R}, open(O / "outputs/wai_confidence_intervals.json", "w"), indent=1)
for k, v in R.items(): print(f"{k:42s} {v['FC_only']}/{v['other_only']}  {v['diff_pp_wald']}")
